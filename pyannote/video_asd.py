"""
video_asd.py — TalkNetASD and TSTalkNet DiarizationFrontend implementations.

Calls video/run_asd.py via subprocess in the isolated Python 3.10 video/ env.
Audio-only datasets (Providence, Playlogue) return [] gracefully when the
subprocess reports FileNotFoundError for the .mp4 file.
"""

import hashlib
import json
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# Standalone config (mirrors the fields added to BaseConfig in unified.py)
# ---------------------------------------------------------------------------

@dataclass
class VideoASDConfig:
    model_name: str = ""
    rttm_cache_dir: str = "pyannote/video_asd_rttm_cache"
    face_cache_dir: str = "pyannote/video_face_cache"
    video_env_python: str = "video/.venv/bin/python"
    run_asd_script: str = "video/run_asd.py"
    pretrain_dir: str = "video/pretrain"


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _rttm_cache_path(audio_path: str, model_name: str, rttm_cache_dir: str) -> str:
    key = hashlib.md5(audio_path.encode("utf-8")).hexdigest()
    stem = Path(audio_path).stem
    return os.path.join(rttm_cache_dir, model_name, f"{stem}__{key}.rttm")


def _parse_chi_rttm(rttm_path: str, min_dur: float) -> List[Dict[str, float]]:
    segs = []
    if not os.path.exists(rttm_path):
        return segs
    with open(rttm_path) as f:
        for line in f:
            line = line.strip()
            if not line or not line.startswith("SPEAKER"):
                continue
            parts = line.split()
            if len(parts) < 8:
                continue
            start, dur, label = float(parts[3]), float(parts[4]), parts[7]
            if label == "CHI" and dur >= min_dur:
                segs.append({"start": start, "end": start + dur, "dur": dur})
    return segs


def _derive_video_path(audio_path: str) -> str:
    """Return the expected video path for a BIDS audio file, or '' if not applicable."""
    if not audio_path.endswith("_audio.wav"):
        return ""
    return audio_path.replace("_audio.wav", "_desc-processed_beh.mp4")


def _run_asd_subprocess(cmd: list, audio_path: str) -> bool:
    """Run run_asd.py; return True on success, False if audio-only dataset."""
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode == 0:
        return True
    combined = (result.stdout + result.stderr).lower()
    if "video file not found" in combined:
        return False
    raise RuntimeError(
        f"run_asd.py failed (exit {result.returncode}) for {audio_path}:\n"
        f"--- stderr ---\n{result.stderr}\n--- stdout ---\n{result.stdout}"
    )


# ---------------------------------------------------------------------------
# TalkNet-ASD frontend
# ---------------------------------------------------------------------------

class TalkNetASDFrontend:
    """TalkNet-ASD active speaker detection frontend.

    Derives the SAILS video path from the audio path (BIDS _audio.wav →
    _desc-processed_beh.mp4), runs S3FD face detection + TalkNet-ASD
    inference, and returns child vocalization segments as
    List[{"start", "end", "dur"}].

    Returns [] for audio-only datasets (Providence, Playlogue) where the
    corresponding .mp4 does not exist.
    """

    def __init__(self, cfg):
        os.makedirs(os.path.join(cfg.video_asd_rttm_cache_dir, "talknet_asd"), exist_ok=True)
        os.makedirs(cfg.video_face_cache_dir, exist_ok=True)
        self._n_processed = 0

    def get_segments(self, audio_path: str, cfg) -> List[Dict[str, float]]:
        # Skip subprocess entirely for audio-only datasets
        video_path = _derive_video_path(audio_path)
        if not video_path or not os.path.exists(video_path):
            return []

        cache = _rttm_cache_path(audio_path, "talknet_asd", cfg.video_asd_rttm_cache_dir)
        if not os.path.exists(cache):
            os.makedirs(os.path.dirname(cache), exist_ok=True)
            ok = _run_asd_subprocess([
                cfg.video_env_python,
                cfg.video_run_asd_script,
                "--audio_path", audio_path,
                "--model", "talknet_asd",
                "--out_rttm", cache,
                "--face_cache_dir", cfg.video_face_cache_dir,
                "--pretrain_dir", cfg.video_pretrain_dir,
            ], audio_path)
            if not ok:
                return []

        self._n_processed += 1
        if self._n_processed % 50 == 0:
            print(f"  [talknet_asd] {self._n_processed} clips processed", flush=True)

        return _parse_chi_rttm(cache, cfg.min_seg_dur_sec)


# ---------------------------------------------------------------------------
# TS-TalkNet frontend
# ---------------------------------------------------------------------------

def _find_ref_audio(audio_path: str, split_dir: str) -> str:
    """Find a training-split reference clip for the same child.

    Parses sub-{ID} from the BIDS audio_path; looks up child_id in train.csv;
    returns the first audio_path where audio_exists=True and the file exists on
    disk (excluding the query clip itself). Returns "" if none found.
    """
    import pandas as pd

    m = re.search(r"sub-([A-Za-z0-9]+)", audio_path)
    if not m:
        return ""
    child_id = m.group(1)

    train_csv = os.path.join(split_dir, "train.csv")
    if not os.path.exists(train_csv):
        return ""

    df = pd.read_csv(train_csv)
    candidates = df[df["child_id"] == child_id]
    candidates = candidates[candidates["audio_path"] != audio_path]
    if "audio_exists" in candidates.columns:
        candidates = candidates[candidates["audio_exists"] == True]

    for ref in candidates["audio_path"].tolist():
        if os.path.exists(ref):
            return ref
    return ""


class TSTalkNetFrontend:
    """TS-TalkNet speaker-conditioned ASD frontend.

    Uses a reference audio clip from the target child's training split to
    condition the ASD model for speaker-specific active speaker detection.
    Falls back to [] if no reference clip is found, if the dataset is
    audio-only (no .mp4), or if the required checkpoints are not present.

    Required checkpoints (not publicly released — obtain from TS-TalkNet authors):
      {pretrain_dir}/ts_talknet.model
      {video_dir}/TS-TalkNet/exps/pretrain.model
    """

    def __init__(self, cfg):
        os.makedirs(os.path.join(cfg.video_asd_rttm_cache_dir, "ts_talknet"), exist_ok=True)
        os.makedirs(cfg.video_face_cache_dir, exist_ok=True)

        ckpt = os.path.join(cfg.video_pretrain_dir, "ts_talknet.model")
        video_dir = str(Path(cfg.video_run_asd_script).parent)
        ecapa_ckpt = os.path.join(video_dir, "TS-TalkNet", "exps", "pretrain.model")
        self._checkpoints_available = os.path.exists(ckpt) and os.path.exists(ecapa_ckpt)
        if not self._checkpoints_available:
            print(
                f"[TSTalkNetFrontend] Checkpoints not found — skipping this frontend.\n"
                f"  Required: {ckpt}\n"
                f"  Required: {ecapa_ckpt}\n"
                f"  Obtain from the TS-TalkNet authors (not publicly released).",
                flush=True,
            )

    def get_segments(self, audio_path: str, cfg) -> List[Dict[str, float]]:
        if not self._checkpoints_available:
            return []

        # Skip subprocess entirely for audio-only datasets
        video_path = _derive_video_path(audio_path)
        if not video_path or not os.path.exists(video_path):
            return []

        cache = _rttm_cache_path(audio_path, "ts_talknet", cfg.video_asd_rttm_cache_dir)
        if not os.path.exists(cache):
            ref_audio = _find_ref_audio(audio_path, cfg.split_dir)
            if not ref_audio:
                return []
            os.makedirs(os.path.dirname(cache), exist_ok=True)
            ok = _run_asd_subprocess([
                cfg.video_env_python,
                cfg.video_run_asd_script,
                "--audio_path", audio_path,
                "--model", "ts_talknet",
                "--ref_audio", ref_audio,
                "--out_rttm", cache,
                "--face_cache_dir", cfg.video_face_cache_dir,
                "--pretrain_dir", cfg.video_pretrain_dir,
            ], audio_path)
            if not ok:
                return []
        return _parse_chi_rttm(cache, cfg.min_seg_dur_sec)


# ---------------------------------------------------------------------------
# LocoNet + ECAPA speaker-identity frontend
# ---------------------------------------------------------------------------

class LocoNetECAPAFrontend:
    """LocoNet ASD with ECAPA speaker-identity matching.

    Runs LocoNet independently on every face track in the clip, then uses
    ECAPA cosine similarity against a train-split reference audio for the
    same child to identify which track is the target child.

    Advantages over TalkNet-ASD:
      - Not limited to the smallest-face heuristic.
      - Uses speaker identity rather than face size.
      - Falls back to smallest-face when no reference is available.

    Requires: video/LoCoNet_ASD/ downloaded, pytorch_model.bin present.
    """

    def __init__(self, cfg):
        cache_root = os.path.join(cfg.video_asd_rttm_cache_dir, "loconet_ecapa")
        self._rttm_dir = cache_root
        self._tracks_dir = cache_root + "_tracks"
        os.makedirs(self._rttm_dir, exist_ok=True)
        os.makedirs(self._tracks_dir, exist_ok=True)
        os.makedirs(cfg.video_face_cache_dir, exist_ok=True)

        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        from speechbrain.inference.speaker import EncoderClassifier
        self._ecapa = EncoderClassifier.from_hparams(
            source=cfg.ecapa_source,
            run_opts={"device": device},
        )
        self._device = device
        self._n_processed = 0

    def _embed_path(self, audio_path: str,
                    segments: Optional[List[Dict]] = None) -> "np.ndarray":
        """Return an ECAPA embedding for audio_path averaged over segments.

        If segments is None, embeds the whole file in one shot.
        """
        import numpy as np
        import soundfile as sf
        import torch

        audio, sr = sf.read(audio_path, dtype="float32", always_2d=True)
        audio = audio.mean(axis=1)
        if sr != 16000:
            import torchaudio
            audio_t = torch.tensor(audio).unsqueeze(0)
            audio_t = torchaudio.functional.resample(audio_t, sr, 16000)
            audio = audio_t.squeeze(0).numpy()
            sr = 16000

        if segments is None:
            clip_t = torch.tensor(audio).unsqueeze(0).to(self._device)
            emb = self._ecapa.encode_batch(clip_t).squeeze().detach().cpu().numpy()
            return emb

        embs = []
        for seg in segments:
            s = int(seg["start"] * sr)
            e = int(seg["end"] * sr)
            clip = audio[s:e]
            if len(clip) < int(0.25 * sr):
                continue
            clip_t = torch.tensor(clip).unsqueeze(0).to(self._device)
            emb = self._ecapa.encode_batch(clip_t).squeeze().detach().cpu().numpy()
            embs.append(emb)

        if not embs:
            return np.zeros(192, dtype=np.float32)
        return np.mean(embs, axis=0)

    @staticmethod
    def _cosine(a: "np.ndarray", b: "np.ndarray") -> float:
        import numpy as np
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na < 1e-8 or nb < 1e-8:
            return 0.0
        return float(np.dot(a, b) / (na * nb))

    def get_segments(self, audio_path: str, cfg) -> List[Dict[str, float]]:
        video_path = _derive_video_path(audio_path)
        if not video_path or not os.path.exists(video_path):
            return []

        key = hashlib.md5(audio_path.encode("utf-8")).hexdigest()
        stem = Path(audio_path).stem
        tracks_json = os.path.join(self._tracks_dir, f"{stem}__{key}.json")
        rttm_cache = os.path.join(self._rttm_dir, f"{stem}__{key}.rttm")

        if not os.path.exists(tracks_json):
            checkpoint = getattr(cfg, "video_loconet_checkpoint", "")
            cmd = [
                cfg.video_env_python,
                cfg.video_run_asd_script,
                "--audio_path", audio_path,
                "--model", "loconet",
                "--out_rttm", rttm_cache,
                "--face_cache_dir", cfg.video_face_cache_dir,
                "--pretrain_dir", cfg.video_pretrain_dir,
                "--output_tracks_json", tracks_json,
            ]
            if checkpoint:
                cmd.extend(["--checkpoint", checkpoint])
            ok = _run_asd_subprocess(cmd, audio_path)
            if not ok:
                return []

        if not os.path.exists(tracks_json):
            return []

        with open(tracks_json) as f:
            track_data = json.load(f)

        if not track_data:
            return []

        # Try to find a reference audio from the same child's training split
        ref_audio = _find_ref_audio(audio_path, cfg.split_dir)
        if not ref_audio:
            # No reference: fall back to smallest-face track
            best = min(track_data, key=lambda t: t.get("mean_area", float("inf")))
            segs = best.get("segments", [])
        else:
            ref_emb = self._embed_path(ref_audio)
            best_track, best_sim = None, -2.0
            for track in track_data:
                segs = track.get("segments", [])
                if not segs:
                    continue
                track_emb = self._embed_path(audio_path, segs)
                sim = self._cosine(track_emb, ref_emb)
                if sim > best_sim:
                    best_sim = sim
                    best_track = track
            segs = best_track.get("segments", []) if best_track else []

        self._n_processed += 1
        if self._n_processed % 50 == 0:
            print(f"  [loconet_ecapa] {self._n_processed} clips processed", flush=True)

        return [
            {"start": s["start"], "end": s["end"], "dur": s["end"] - s["start"]}
            for s in segs
            if s["end"] - s["start"] >= cfg.min_seg_dur_sec
        ]
