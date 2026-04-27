"""
SceneComposer: assembles synthetic audio scenes from a segment manifest and
scene-configuration dict.

Implements the core composition logic for:
  - positive scenes (TARGET_CHILD + ADULT alternating turns)
  - adult_only_negative scenes (ADULT turns only, no child)
  - background_speech_negative / noise_only_negative (stub; extended in T021)

All speaker timelines follow ``contracts/scene-metadata.md``.
Audio is padded / truncated to exactly ``scene_duration_sec`` seconds.
"""

from __future__ import annotations

import hashlib
import json
import sys
import numpy as np
import pandas as pd
import soundfile as sf
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from synth.audio_utils import (
    apply_crossfade,
    convolve_rir,
    mix_at_snr,
    peak_normalize,
    resample_to_16k,
)
from synth.labels import write_clip_labels_row, write_rttm, write_scene_metadata
from synth.turn_taking import TurnTakingSimulator


_SR = 16000


class SceneComposer:
    """Compose synthetic audio scenes from a manifest and config dict.

    Parameters
    ----------
    config : dict
        Scene configuration loaded from a YAML file matching
        ``contracts/scene-config.md``.
    manifest_df : pd.DataFrame
        Segment manifest (as returned by
        :func:`synth.manifest.load_manifest`).  Only rows with
        ``usable_for_training = True`` and a valid audio_path are used.
    """

    def __init__(self, config: dict, manifest_df: pd.DataFrame) -> None:
        self.config = config
        self.manifest = manifest_df

        proj = config.get("project", {})
        self._config_name: str = str(proj.get("name", "unknown"))
        self._sr: int = int(proj.get("sample_rate", _SR))

        scene = config.get("scene", {})
        self._duration_sec: float = float(scene.get("duration_sec", 30.0))
        self._age_band: str = str(scene.get("target_age_band", "14_18_months"))

        self._sampling = config.get("sampling", {})
        self._turn_cfg = config.get("turn_taking", {})
        self._mixing = config.get("mixing", {})
        self._sources = config.get("sources", {})
        self._config_hash: Optional[str] = None

        self._rir_pool: List[Path] = self._load_file_pool(
            str(self._mixing.get("rir_dir", "")), ("*.wav", "*.flac"), "RIR"
        )
        self._noise_pool: List[Path] = self._load_file_pool(
            str(self._mixing.get("noise_dir", "")), ("*.wav",), "noise"
        )

        self._child_df, self._adult_df = self._build_subsets(manifest_df)

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    def _load_file_pool(
        self, dir_path: str, patterns: tuple, label: str
    ) -> List[Path]:
        """Scan *dir_path* for audio files matching *patterns* and return a
        sorted list of Paths.  Returns [] with a warning when the directory is
        absent, empty, or not configured (empty string).
        """
        if not dir_path:
            return []
        root = Path(dir_path)
        if not root.exists():
            print(
                f"  [WARN] {label} dir not found: {root} — clean-mix fallback active.",
                file=sys.stderr,
            )
            return []
        pool: List[Path] = []
        for pat in patterns:
            pool.extend(root.rglob(pat))
        pool = sorted(set(pool))
        if not pool:
            print(
                f"  [WARN] {label} dir exists but contains no matching files: {root}",
                file=sys.stderr,
            )
        else:
            print(f"  {label} pool: {len(pool)} files from {root}")
        return pool

    def _build_subsets(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Partition the manifest into child and adult usable subsets."""
        child_datasets = self._sources.get("child_segments") or None
        adult_datasets = self._sources.get("adult_segments") or None

        usable = df[df["usable_for_training"].astype(bool)].copy()

        # ---- child subset ----
        child_roles = {"target_child", "non_target_child", "unknown_child"}
        child_mask = usable["speaker_role"].isin(child_roles)
        child_pool = usable[child_mask]
        if child_datasets:
            ds_mask = child_pool["source_dataset"].isin(child_datasets)
            if ds_mask.any():
                child_pool = child_pool[ds_mask]
        # prefer matching age band; fall back to all child rows
        age_mask = child_pool["age_band"] == self._age_band
        child_df = child_pool[age_mask] if age_mask.any() else child_pool

        # ---- adult subset ----
        adult_pool = usable[usable["speaker_role"] == "adult"]
        if adult_datasets:
            ds_mask = adult_pool["source_dataset"].isin(adult_datasets)
            if ds_mask.any():
                adult_pool = adult_pool[ds_mask]
        adult_df = adult_pool

        return child_df.reset_index(drop=True), adult_df.reset_index(drop=True)

    # ------------------------------------------------------------------
    # Config hash
    # ------------------------------------------------------------------

    def get_config_hash(self) -> str:
        """Return a 12-char MD5 hex digest of the config dict."""
        if self._config_hash is None:
            config_str = json.dumps(self.config, sort_keys=True, default=str)
            self._config_hash = hashlib.md5(config_str.encode()).hexdigest()[:12]
        return self._config_hash

    # ------------------------------------------------------------------
    # Scene-type selection
    # ------------------------------------------------------------------

    def _sample_scene_type(self, rng: np.random.Generator) -> str:
        """Sample a scene type from the probability distribution in config.

        Supports the 4 base types plus 4 extended types (T021):
          hard_overlap_positive, hard_overlap_negative,
          short_vocalization_positive, low_snr_positive.
        Extended types default to 0 probability when absent from config.
        All probabilities are normalised so they need not sum to exactly 1.
        """
        probs = {
            "positive":                     float(self._sampling.get("positive_scene_probability", 0.5)),
            "adult_only_negative":          float(self._sampling.get("adult_only_negative_probability", 0.25)),
            "background_speech_negative":   float(self._sampling.get("background_speech_negative_probability", 0.15)),
            "noise_only_negative":          float(self._sampling.get("noise_only_negative_probability", 0.10)),
            "hard_overlap_positive":        float(self._sampling.get("hard_overlap_positive_probability", 0.0)),
            "hard_overlap_negative":        float(self._sampling.get("hard_overlap_negative_probability", 0.0)),
            "short_vocalization_positive":  float(self._sampling.get("short_vocalization_positive_probability", 0.0)),
            "low_snr_positive":             float(self._sampling.get("low_snr_positive_probability", 0.0)),
        }
        types = list(probs.keys())
        weights = np.array([probs[t] for t in types], dtype=np.float64)
        total = weights.sum()
        if total < 1e-9:
            return "positive"

        weights /= total
        idx = int(rng.choice(len(types), p=weights))
        return types[idx]

    # ------------------------------------------------------------------
    # Segment audio loading
    # ------------------------------------------------------------------

    def _load_seg_audio(
        self, seg: dict, target_dur_sec: float
    ) -> np.ndarray:
        """Load a segment WAV and tile / trim to ``target_dur_sec`` seconds."""
        n_target = max(1, int(target_dur_sec * self._sr))
        path = str(seg.get("audio_path", ""))

        if path and Path(path).exists():
            wav, sr = sf.read(path, dtype="float32", always_2d=False)
            if wav.ndim > 1:
                wav = wav.mean(axis=1)
            wav = resample_to_16k(wav, sr)
        else:
            # Fallback to zeros when file is missing (test stubs)
            seg_n = max(1, int(float(seg.get("duration_sec", target_dur_sec)) * self._sr))
            wav = np.zeros(seg_n, dtype=np.float32)

        if len(wav) == 0:
            return np.zeros(n_target, dtype=np.float32)
        # Tile to ensure we have enough samples, then trim
        if len(wav) < n_target:
            repeats = int(np.ceil(n_target / len(wav)))
            wav = np.tile(wav, repeats)
        return wav[:n_target].astype(np.float32)

    def _sample_one(self, df: pd.DataFrame, rng: np.random.Generator) -> dict:
        """Uniform random sample of one row from ``df``."""
        idx = int(rng.integers(0, len(df)))
        return df.iloc[idx].to_dict()

    # ------------------------------------------------------------------
    # Timeline builders
    # ------------------------------------------------------------------

    def _build_positive_timeline(
        self, rng: np.random.Generator
    ) -> List[Dict[str, Any]]:
        """Build an interleaved TARGET_CHILD / ADULT turn timeline."""
        turn_cfg = self._turn_cfg
        simulator = TurnTakingSimulator(
            age_band=self._age_band,
            overlap_prob=float(self._sampling.get("overlap_probability", 0.15)),
            n_turns_min=int(turn_cfg.get("n_turns_min", 2)),
            n_turns_max=int(turn_cfg.get("n_turns_max", 20)),
            child_dur_mean=turn_cfg.get("child_turn_duration_mean_sec"),
            child_dur_std=turn_cfg.get("child_turn_duration_std_sec"),
            adult_dur_mean=float(turn_cfg.get("adult_turn_duration_mean_sec", 3.5)),
            adult_dur_std=float(turn_cfg.get("adult_turn_duration_std_sec", 1.5)),
            pause_mean=float(turn_cfg.get("pause_mean_sec", 0.8)),
            pause_std=float(turn_cfg.get("pause_std_sec", 0.3)),
        )
        turns = simulator.sample_turns(rng)

        timeline: List[Dict[str, Any]] = []
        current_time = 0.0

        for turn in turns:
            pause = float(turn["pause_before_sec"])
            current_time = max(0.0, current_time + pause)
            if current_time >= self._duration_sec:
                break

            role = turn["speaker_role"]
            dur = float(turn["duration_sec"])
            end_time = min(current_time + dur, self._duration_sec)

            if role == "TARGET_CHILD" and not self._child_df.empty:
                seg = self._sample_one(self._child_df, rng)
                speaker_label = "TARGET_CHILD"
            elif role == "ADULT" and not self._adult_df.empty:
                seg = self._sample_one(self._adult_df, rng)
                speaker_label = "ADULT_0"
            else:
                current_time = current_time + dur
                continue

            timeline.append(
                {
                    "speaker_label": speaker_label,
                    "start_sec": current_time,
                    "end_sec": end_time,
                    "seg_row": seg,
                    "gain_db": 0.0,
                    "rir_id": None,
                }
            )
            current_time = current_time + dur

        return timeline

    def _build_short_vocalization_positive_timeline(
        self, rng: np.random.Generator
    ) -> List[Dict[str, Any]]:
        """Positive timeline with TARGET_CHILD turns capped at short_threshold_sec."""
        threshold = float(self._sampling.get("short_threshold_sec", 0.5))
        timeline = self._build_positive_timeline(rng)
        for t in timeline:
            if t["speaker_label"] == "TARGET_CHILD":
                max_end = t["start_sec"] + threshold
                t["end_sec"] = min(t["end_sec"], max_end)
        return [t for t in timeline if t["end_sec"] > t["start_sec"]]

    def _build_hard_overlap_positive_timeline(
        self, rng: np.random.Generator
    ) -> List[Dict[str, Any]]:
        """Positive timeline forcing >= 1 overlap event with TARGET_CHILD."""
        if self._child_df.empty or self._adult_df.empty:
            return self._build_positive_timeline(rng)

        turn_cfg = self._turn_cfg
        child_dur_mean = turn_cfg.get("child_turn_duration_mean_sec")
        child_dur_std = turn_cfg.get("child_turn_duration_std_sec")

        # Use high overlap probability to guarantee overlaps
        simulator = TurnTakingSimulator(
            age_band=self._age_band,
            overlap_prob=1.0,   # force every transition to overlap
            n_turns_min=max(2, int(turn_cfg.get("n_turns_min", 2))),
            n_turns_max=int(turn_cfg.get("n_turns_max", 20)),
            child_dur_mean=child_dur_mean,
            child_dur_std=child_dur_std,
            adult_dur_mean=float(turn_cfg.get("adult_turn_duration_mean_sec", 3.5)),
            adult_dur_std=float(turn_cfg.get("adult_turn_duration_std_sec", 1.5)),
            pause_mean=float(turn_cfg.get("pause_mean_sec", 0.8)),
            pause_std=float(turn_cfg.get("pause_std_sec", 0.3)),
        )
        turns = simulator.sample_turns(rng)
        timeline: List[Dict[str, Any]] = []
        current_time = 0.0
        for turn in turns:
            current_time = max(0.0, current_time + float(turn["pause_before_sec"]))
            if current_time >= self._duration_sec:
                break
            dur = float(turn["duration_sec"])
            end_time = min(current_time + dur, self._duration_sec)
            role = turn["speaker_role"]
            if role == "TARGET_CHILD" and not self._child_df.empty:
                seg = self._sample_one(self._child_df, rng)
                label = "TARGET_CHILD"
            elif not self._adult_df.empty:
                seg = self._sample_one(self._adult_df, rng)
                label = "ADULT_0"
            else:
                current_time = current_time + dur
                continue
            timeline.append({"speaker_label": label, "start_sec": current_time,
                              "end_sec": end_time, "seg_row": seg,
                              "gain_db": 0.0, "rir_id": None})
            current_time = current_time + dur
        return timeline

    def _build_hard_overlap_negative_timeline(
        self, rng: np.random.Generator
    ) -> List[Dict[str, Any]]:
        """Adult-only timeline where adults overlap each other (no child)."""
        if self._adult_df.empty:
            return []

        turn_cfg = self._turn_cfg
        adult_dur_mean = float(turn_cfg.get("adult_turn_duration_mean_sec", 3.5))
        adult_dur_std = float(turn_cfg.get("adult_turn_duration_std_sec", 1.5))
        n_turns_max = int(turn_cfg.get("n_turns_max", 20))

        timeline: List[Dict[str, Any]] = []
        current_time = 0.0
        adult_idx = 0

        for _ in range(n_turns_max):
            if current_time >= self._duration_sec:
                break
            dur = max(0.1, float(rng.normal(adult_dur_mean, adult_dur_std)))
            end_time = min(current_time + dur, self._duration_sec)
            seg = self._sample_one(self._adult_df, rng)
            speaker_label = f"ADULT_{adult_idx % 2}"
            adult_idx += 1
            timeline.append({"speaker_label": speaker_label, "start_sec": current_time,
                              "end_sec": end_time, "seg_row": seg,
                              "gain_db": 0.0, "rir_id": None})
            # Force overlap: next speaker starts during current turn
            overlap_dur = max(0.1, float(rng.normal(0.4, 0.2)))
            current_time = max(0.0, end_time - overlap_dur)

        return timeline

    def _build_adult_only_timeline(
        self, rng: np.random.Generator
    ) -> List[Dict[str, Any]]:
        """Build an adult-only (negative) scene timeline."""
        if self._adult_df.empty:
            return []

        turn_cfg = self._turn_cfg
        adult_dur_mean = float(turn_cfg.get("adult_turn_duration_mean_sec", 3.5))
        adult_dur_std = float(turn_cfg.get("adult_turn_duration_std_sec", 1.5))
        pause_mean = float(turn_cfg.get("pause_mean_sec", 0.8))
        pause_std = float(turn_cfg.get("pause_std_sec", 0.3))
        n_turns_max = int(turn_cfg.get("n_turns_max", 20))

        timeline: List[Dict[str, Any]] = []
        current_time = 0.0

        for _ in range(n_turns_max):
            if current_time >= self._duration_sec:
                break
            dur = max(0.1, float(rng.normal(adult_dur_mean, adult_dur_std)))
            end_time = min(current_time + dur, self._duration_sec)
            seg = self._sample_one(self._adult_df, rng)
            timeline.append(
                {
                    "speaker_label": "ADULT_0",
                    "start_sec": current_time,
                    "end_sec": end_time,
                    "seg_row": seg,
                    "gain_db": 0.0,
                    "rir_id": None,
                }
            )
            pause = max(0.0, float(rng.normal(pause_mean, pause_std)))
            current_time = end_time + pause

        return timeline

    # ------------------------------------------------------------------
    # Audio mixing
    # ------------------------------------------------------------------

    def _mix_scene_audio(
        self, timeline: List[Dict[str, Any]], rng: np.random.Generator
    ) -> Tuple[np.ndarray, Optional[float], Optional[str], Optional[str]]:
        """Mix speaker tracks into a scene WAV array.

        Returns
        -------
        tuple of (mixed_wav, mean_snr_db, rir_id, noise_id)
        """
        n_samples = int(self._duration_sec * self._sr)
        mix = np.zeros(n_samples, dtype=np.float64)

        crossfade_ms = float(self._mixing.get("crossfade_ms", 20.0))
        crossfade_samples = int(crossfade_ms * self._sr / 1000.0)
        do_peak_norm = bool(self._mixing.get("peak_normalize", True))

        for track in timeline:
            start_sec = float(track["start_sec"])
            end_sec = float(track["end_sec"])
            seg_dur = end_sec - start_sec
            if seg_dur <= 0:
                continue

            seg_audio = self._load_seg_audio(track["seg_row"], seg_dur)
            seg_audio = apply_crossfade(seg_audio, crossfade_samples)

            start_sample = int(start_sec * self._sr)
            end_sample = start_sample + len(seg_audio)
            end_sample_clipped = min(end_sample, n_samples)
            actual_len = end_sample_clipped - start_sample
            if actual_len > 0:
                mix[start_sample:end_sample_clipped] += seg_audio[:actual_len]

        mix = mix.astype(np.float32)

        mean_snr_db: Optional[float] = None
        noise_id: Optional[str] = None
        rir_id: Optional[str] = None

        apply_rir_prob = float(self._mixing.get("apply_rir_probability", 0.0))
        apply_noise_prob = float(self._mixing.get("apply_noise_probability", 0.0))
        snr_db_min = float(self._mixing.get("snr_db_min", 0.0))
        snr_db_max = float(self._mixing.get("snr_db_max", 25.0))

        # --- RIR convolution (FR-001, FR-002) ---
        if self._rir_pool and rng.random() < apply_rir_prob:
            rir_path = self._rir_pool[int(rng.integers(len(self._rir_pool)))]
            try:
                rir_wav, rir_sr = sf.read(str(rir_path), dtype="float32", always_2d=False)
                if rir_wav.ndim > 1:
                    rir_wav = rir_wav.mean(axis=1)
                if rir_sr != self._sr:
                    rir_wav = resample_to_16k(rir_wav, rir_sr)
                mix = convolve_rir(mix, rir_wav)
                rir_id = rir_path.stem
            except Exception as exc:
                print(f"  [WARN] Skipping RIR {rir_path.name}: {exc}", file=sys.stderr)

        # --- Noise mixing (FR-003, FR-004) ---
        if self._noise_pool and rng.random() < apply_noise_prob:
            noise_path = self._noise_pool[int(rng.integers(len(self._noise_pool)))]
            try:
                noise_wav, noise_sr = sf.read(str(noise_path), dtype="float32", always_2d=False)
                if noise_wav.ndim > 1:
                    noise_wav = noise_wav.mean(axis=1)
                if noise_sr != self._sr:
                    noise_wav = resample_to_16k(noise_wav, noise_sr)
                snr_db = float(np.clip(rng.uniform(snr_db_min, snr_db_max), snr_db_min, snr_db_max))
                mix = mix_at_snr(mix, noise_wav, snr_db)
                noise_id = noise_path.stem
                mean_snr_db = round(snr_db, 2)
            except Exception as exc:
                print(f"  [WARN] Skipping noise {noise_path.name}: {exc}", file=sys.stderr)

        if do_peak_norm and np.max(np.abs(mix)) > 1e-8:
            mix = peak_normalize(mix)

        return mix, mean_snr_db, rir_id, noise_id

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compose(self, scene_id: str, rng: np.random.Generator) -> dict:
        """Assemble one synthetic scene.

        Parameters
        ----------
        scene_id : str
            Globally unique scene identifier (used in RTTM and filenames).
        rng : np.random.Generator
            Per-scene random generator (seeded by caller for reproducibility).

        Returns
        -------
        dict
            Scene metadata matching ``contracts/scene-metadata.md`` plus
            internal keys ``_mixed_wav`` (the float32 WAV array),
            ``tracks`` (RTTM track list), ``audio_path``, and ``rttm_path``
            (both initially empty; filled in by :meth:`write`).
        """
        scene_type = self._sample_scene_type(rng)

        if scene_type == "positive":
            timeline = self._build_positive_timeline(rng)
        elif scene_type == "adult_only_negative":
            timeline = self._build_adult_only_timeline(rng)
        elif scene_type == "background_speech_negative":
            # Use adult pool as background speech; MUSAN subset can be added
            # by populating the manifest with source_dataset="musan_speech"
            # segments. Falls back to adult pool if unavailable.
            timeline = self._build_adult_only_timeline(rng)
        elif scene_type in ("noise_only_negative", "silence_noise_negative"):
            timeline = []
            scene_type = "noise_only_negative"
        elif scene_type == "hard_overlap_positive":
            timeline = self._build_hard_overlap_positive_timeline(rng)
        elif scene_type == "hard_overlap_negative":
            timeline = self._build_hard_overlap_negative_timeline(rng)
        elif scene_type == "short_vocalization_positive":
            timeline = self._build_short_vocalization_positive_timeline(rng)
        elif scene_type == "low_snr_positive":
            # Same timeline as positive; low SNR is enforced during mixing below
            timeline = self._build_positive_timeline(rng)
        else:
            timeline = self._build_positive_timeline(rng)

        mixed_wav, mean_snr_db, rir_id, noise_id = self._mix_scene_audio(
            timeline, rng
        )

        # RTTM-ready track list
        rttm_tracks = [
            {
                "speaker_label": t["speaker_label"],
                "start_sec": t["start_sec"],
                "end_sec": t["end_sec"],
            }
            for t in timeline
        ]

        # Provenance
        source_segments = [
            {
                "speaker_label": t["speaker_label"],
                "segment_id": str(t["seg_row"].get("segment_id", "")),
                "source_dataset": str(t["seg_row"].get("source_dataset", "")),
                "start_sec": float(t["start_sec"]),
                "end_sec": float(t["end_sec"]),
                "gain_db": float(t.get("gain_db", 0.0)),
                "rir_id": t.get("rir_id"),
            }
            for t in timeline
        ]

        target_child_dur = sum(
            t["end_sec"] - t["start_sec"]
            for t in timeline
            if t["speaker_label"] == "TARGET_CHILD"
        )
        adult_dur = sum(
            t["end_sec"] - t["start_sec"]
            for t in timeline
            if t["speaker_label"].startswith("ADULT_")
        )
        other_child_dur = sum(
            t["end_sec"] - t["start_sec"]
            for t in timeline
            if t["speaker_label"] == "OTHER_CHILD_0"
        )
        unique_speakers = sorted({t["speaker_label"] for t in timeline})

        scene_meta: dict = {
            # --- contract fields ---
            "synthetic_scene_id": scene_id,
            "duration_sec": self._duration_sec,
            "sample_rate": self._sr,
            "target_age_band": self._age_band,
            "scene_type": scene_type,
            "target_child_present": target_child_dur > 0,
            "target_child_vocalized": target_child_dur > 0,
            "target_child_duration_sec": float(target_child_dur),
            "adult_present": adult_dur > 0,
            "adult_duration_sec": float(adult_dur),
            "non_target_child_present": other_child_dur > 0,
            "other_child_duration_sec": float(other_child_dur),
            "overlap_present": False,   # recomputed by labels.write_clip_labels_row
            "max_overlap_speakers": 1,
            "mean_snr_db": mean_snr_db,
            "rir_id": rir_id,
            "noise_id": noise_id,
            "generation_config_name": self._config_name,
            "generation_config_hash": self.get_config_hash(),
            "random_seed": -1,          # filled in by generate_scenes.py
            "source_segments": source_segments,
            "speakers": unique_speakers,
            # --- internal keys consumed by write() ---
            "_mixed_wav": mixed_wav,
            "tracks": rttm_tracks,
            "audio_path": "",
            "rttm_path": "",
            # --- extra fields needed by write_clip_labels_row ---
            "snr_db": mean_snr_db,
            "noise_type": noise_id or "",
            "rir_type": rir_id or "",
            "age_band": self._age_band,
        }

        return scene_meta

    def write(self, scene_meta: dict, output_dir: str) -> dict:
        """Write WAV, RTTM, and JSON files for a composed scene.

        Parameters
        ----------
        scene_meta : dict
            Scene metadata as returned by :meth:`compose`.  The ``_mixed_wav``
            key is consumed (removed) and the scene WAV is written to disk.
        output_dir : str
            Base output directory.  Sub-directories ``wav/``, ``rttm/``, and
            ``json/`` are created automatically.

        Returns
        -------
        dict
            The same ``scene_meta`` dict with ``audio_path`` and ``rttm_path``
            updated to the absolute paths of the written files.
        """
        base = Path(output_dir)
        scene_id = scene_meta["synthetic_scene_id"]

        wav_path = base / "wav" / f"{scene_id}.wav"
        rttm_path = base / "rttm" / f"{scene_id}.rttm"
        json_path = base / "json" / f"{scene_id}.json"

        for p in (wav_path, rttm_path, json_path):
            p.parent.mkdir(parents=True, exist_ok=True)

        # Write WAV (pops private key)
        mixed_wav: np.ndarray = scene_meta.pop("_mixed_wav")
        sf.write(str(wav_path), mixed_wav, self._sr, subtype="PCM_16")

        # Update paths
        scene_meta["audio_path"] = str(wav_path.resolve())
        scene_meta["rttm_path"] = str(rttm_path.resolve())

        # Write RTTM
        write_rttm(scene_meta["tracks"], scene_id, str(rttm_path))

        # Write JSON (exclude internal / non-contract keys)
        _internal = {"_mixed_wav", "tracks", "snr_db", "noise_type",
                     "rir_type", "age_band"}
        json_meta = {k: v for k, v in scene_meta.items() if k not in _internal}
        write_scene_metadata(json_meta, str(json_path))

        return scene_meta
