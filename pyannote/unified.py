"""
Unified diarization + speaker embedding enrollment pipeline.

Supports three diarizer front-ends and is extensible to new embedders.

Usage:
    python unified_enrollment.py --diarizer usc_sail
    python unified_enrollment.py --diarizer pyannote
    python unified_enrollment.py --diarizer babar --babar-dir /path/to/BabAR
    python unified_enrollment.py --diarizer babar --babar-dir /path/to/BabAR --skip-role-only
"""

import abc
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torchaudio
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

import video_asd
import nemo_diar


# =============================================================
# Config
# =============================================================

@dataclass
class BaseConfig:
    """Settings shared across all diarizer / embedder backends."""

    split_dir: str = "/home/manaal/orcd/scratch/child-adult-diarization/whisper-modeling/seen_child_splits"
    results_dir: str = ""  # set at runtime based on diarizer + embedder

    sample_rate: int = 16000
    min_seg_dur_sec: float = 0.4
    max_enrollment_segments_per_child: int = 200

    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # ECAPA
    ecapa_source: str = "speechbrain/spkrec-ecapa-voxceleb"

    # role-only threshold tuning
    duration_threshold_grid: Tuple[float, ...] = (
        0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0,
    )

    # enrollment threshold tuning
    similarity_threshold_min: float = 0.1
    similarity_threshold_max: float = 0.95
    similarity_threshold_steps: int = 171

    # --- USC-SAIL settings ---
    usc_sail_repo_dir: str = "/home/manaal/orcd/scratch/child-adult-diarization/whisper-modeling"
    usc_sail_script: str = "/home/manaal/orcd/scratch/child-adult-diarization/whisper-modeling/scripts/infer_long_wav_files.py"
    usc_sail_model_path: str = "/home/manaal/orcd/scratch/child-adult-diarization/whisper-modeling/whisper-base_rank8_pretrained_50k.pt"
    usc_sail_python: str = "python"
    usc_window_size: float = 10.0
    usc_stride: float = 5.0
    segment_cache_dir: str = "/home/manaal/orcd/scratch/child-adult-diarization/whisper-modeling/usc_sail_segment_cache"
    rttm_cache_dir: str = "/home/manaal/orcd/scratch/child-adult-diarization/whisper-modeling/usc_sail_rttm_cache"

    # --- Pyannote settings ---
    pyannote_model: str = "pyannote/speaker-diarization-community-1"
    hf_token: str = os.environ.get("HF_TOKEN", "")
    pyannote_cache_dir: str = "/home/manaal/orcd/scratch/child-adult-diarization/pyannote/pyannote_rttm_cache"

    # --- BabAR / VTC 2.0 settings ---
    babar_dir: str = "/home/manaal/orcd/scratch/child-adult-diarization/BabAR/"  # path to cloned BabAR repo
    babar_output_dir: str = "/home/manaal/orcd/scratch/child-adult-diarization/babar/babar_output"
    babar_batch_size: int = 32

    # --- VTC standalone (no BabAR phoneme step) ---
    # Setup: cd BabAR/VTC && uv sync
    vtc_dir: str = "/home/manaal/orcd/scratch/child-adult-diarization/BabAR/VTC"
    vtc_rttm_cache_dir: str = "/home/manaal/orcd/scratch/child-adult-diarization/pyannote/vtc_rttm_cache"
    vtc_input_staging_dir: str = "/home/manaal/orcd/scratch/child-adult-diarization/pyannote/vtc_input_staging"
    vtc_batch_size: int = 64

    # --- VBx diarization ---
    # Setup: cd VBx && uv sync
    vbx_dir: str = "/home/manaal/orcd/scratch/child-adult-diarization/VBx"
    vbx_rttm_cache_dir: str = "/home/manaal/orcd/scratch/child-adult-diarization/pyannote/vbx_rttm_cache"
    vbx_max_speakers: int = 8
    vbx_niters: int = 10
    vbx_Fa: float = 0.1
    vbx_Fb: float = 17.0
    vbx_loopP: float = 0.99
    vbx_win_duration: float = 1.5
    vbx_win_step: float = 0.25

    # --- Video ASD (TalkNet-ASD, TS-TalkNet, LocoNetECAPAFrontend) ---
    # Setup: cd video && uv sync; clone repos; download checkpoints (see video/SETUP.md)
    video_asd_rttm_cache_dir: str = "/home/manaal/orcd/scratch/child-adult-diarization/pyannote/video_asd_rttm_cache"
    video_face_cache_dir: str = "/home/manaal/orcd/scratch/child-adult-diarization/pyannote/video_face_cache"
    video_env_python: str = "/home/manaal/orcd/scratch/child-adult-diarization/video/.venv/bin/python"
    video_run_asd_script: str = "/home/manaal/orcd/scratch/child-adult-diarization/video/run_asd.py"
    video_pretrain_dir: str = "/home/manaal/orcd/scratch/child-adult-diarization/video/pretrain"
    # LocoNet checkpoint; defaults to video/LoCoNet_ASD/pytorch_model.bin if empty
    video_loconet_checkpoint: str = "/home/manaal/orcd/scratch/child-adult-diarization/video/LoCoNet_ASD/pytorch_model.bin"

    # --- EEND-EDA (ESPnet2) ---
    # Setup: pip install espnet espnet_model_zoo soundfile
    # Find models: python -c "from espnet_model_zoo.downloader import ModelDownloader; \
    #   d=ModelDownloader(); [print(r['name']) for r in d.query('diar')]"
    eend_eda_rttm_cache_dir: str = "/home/manaal/orcd/scratch/child-adult-diarization/pyannote/eend_eda_rttm_cache"
    eend_eda_env_python: str = "python"   # override if ESPnet lives in its own venv
    eend_eda_model_tag: str = "espnet/diar_ami_eend_eda"
    eend_eda_num_spks: int = 0            # 0 = let EDA determine automatically

    # --- Sortformer (NeMo) ---
    # Setup: pip install nemo_toolkit[asr]
    # Model downloads from NGC automatically on first run.
    sortformer_rttm_cache_dir: str = "/home/manaal/orcd/scratch/child-adult-diarization/pyannote/sortformer_rttm_cache"
    sortformer_env_python: str = "python"  # override if NeMo lives in its own venv
    sortformer_model: str = "nvidia/diar_sortformer_4spk-v1"
    sortformer_max_speakers: int = 4


# =============================================================
# General utilities
# =============================================================

def save_json(obj, path: str):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def load_split(split_dir: str):
    train_df = pd.read_csv(os.path.join(split_dir, "train.csv"))
    val_df = pd.read_csv(os.path.join(split_dir, "val.csv"))
    test_df = pd.read_csv(os.path.join(split_dir, "test.csv"))
    return train_df, val_df, test_df


def l2_normalize(x: np.ndarray, eps: float = 1e-8):
    return x / max(np.linalg.norm(x), eps)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(l2_normalize(a), l2_normalize(b)))


def audio_to_cache_id(audio_path: str) -> str:
    return hashlib.md5(audio_path.encode("utf-8")).hexdigest()


def compute_metrics(y_true, y_prob, threshold: float = 0.5) -> dict:
    y_pred = (y_prob >= threshold).astype(int)
    metrics = {
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
    }
    try:
        metrics["auroc"] = float(roc_auc_score(y_true, y_prob))
    except Exception:
        metrics["auroc"] = float("nan")
    try:
        metrics["auprc"] = float(average_precision_score(y_true, y_prob))
    except Exception:
        metrics["auprc"] = float("nan")
    return metrics


def add_pred_labels(pred_df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    out = pred_df.copy()
    out["pred_label"] = (out["prob"] >= threshold).astype(int)
    return out


def per_timepoint_metrics(pred_df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    rows = []
    for tp, sub in pred_df.groupby("timepoint_norm"):
        m = compute_metrics(
            sub["label"].to_numpy(),
            sub["prob"].to_numpy(),
            threshold=threshold,
        )
        m["timepoint"] = tp
        m["n"] = int(len(sub))
        rows.append(m)
    return pd.DataFrame(rows)


# =============================================================
# Audio helpers
# =============================================================

def load_audio_mono(audio_path: str, target_sr: int = 16000) -> torch.Tensor:
    wav, sr = torchaudio.load(audio_path)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
    return wav.squeeze(0)


def crop_segment(wav: torch.Tensor, sr: int, start: float, end: float) -> torch.Tensor:
    s = max(0, int(round(start * sr)))
    e = min(wav.numel(), int(round(end * sr)))
    if e <= s:
        return torch.zeros(1, dtype=wav.dtype)
    return wav[s:e]


# =============================================================
# Abstract embedder + ECAPA implementation
# =============================================================

class SpeakerEmbedder(abc.ABC):
    @abc.abstractmethod
    def embed_waveform(self, wav_1d: torch.Tensor) -> np.ndarray:
        ...


class ECAPAEmbedder(SpeakerEmbedder):
    def __init__(self, source: str, device: str):
        from speechbrain.inference.speaker import EncoderClassifier

        self.model = EncoderClassifier.from_hparams(
            source=source,
            run_opts={"device": device},
        )
        self.device = device

    @torch.no_grad()
    def embed_waveform(self, wav_1d: torch.Tensor) -> np.ndarray:
        wav = wav_1d.unsqueeze(0).to(self.device)
        emb = self.model.encode_batch(wav)
        return emb.squeeze().detach().cpu().numpy()


# =============================================================
# Abstract diarizer front-end + implementations
# =============================================================

class DiarizationFrontend(abc.ABC):
    @abc.abstractmethod
    def get_segments(
        self, audio_path: str, cfg: BaseConfig,
    ) -> List[Dict[str, float]]:
        ...


# ----- USC-SAIL ------------------------------------------------

class USCSailFrontend(DiarizationFrontend):
    def __init__(self, cfg: BaseConfig):
        self.cfg = cfg
        os.makedirs(cfg.rttm_cache_dir, exist_ok=True)
        os.makedirs(cfg.segment_cache_dir, exist_ok=True)

    def _rttm_cache_path(self, audio_path: str) -> str:
        cid = audio_to_cache_id(audio_path)
        stem = Path(audio_path).stem
        return os.path.join(self.cfg.rttm_cache_dir, f"{stem}__{cid}.rttm")

    def _segment_cache_path(self, audio_path: str) -> str:
        cid = audio_to_cache_id(audio_path)
        return os.path.join(self.cfg.segment_cache_dir, f"{cid}.json")

    @staticmethod
    def _parse_rttm_for_child_segments(rttm_path: str) -> List[Dict[str, float]]:
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
                if label == "CHI":
                    segs.append({"start": start, "end": start + dur})
        return segs

    def _run_inference(self, audio_path: str) -> str:
        target = self._rttm_cache_path(audio_path)
        if os.path.exists(target):
            return target

        cmd = [
            self.cfg.usc_sail_python,
            self.cfg.usc_sail_script,
            "--wav_file", audio_path,
            "--out_dir", self.cfg.rttm_cache_dir,
            "--model_path", self.cfg.usc_sail_model_path,
            "--device", "cuda" if "cuda" in self.cfg.device else "cpu",
            "--window_size", str(self.cfg.usc_window_size),
            "--stride", str(self.cfg.usc_stride),
        ]
        env = os.environ.copy()
        env["PYTHONPATH"] = self.cfg.usc_sail_repo_dir
        try:
            subprocess.run(cmd, cwd=self.cfg.usc_sail_repo_dir, env=env, check=True)
        except subprocess.CalledProcessError as e:
            # Short audio files (<10s) crash Whisper with mel-length mismatch;
            # write an empty RTTM so the cache is filled and callers get no segments.
            import warnings
            warnings.warn(f"USC-SAIL inference failed for {audio_path}: {e}. Writing empty RTTM.")
            os.makedirs(os.path.dirname(target), exist_ok=True)
            open(target, "w").close()
            return target

        if not os.path.exists(target):
            raise FileNotFoundError(
                f"USC-SAIL finished but expected RTTM not found: {target}"
            )
        return target

    def get_segments(self, audio_path: str, cfg: BaseConfig) -> List[Dict[str, float]]:
        seg_cache = self._segment_cache_path(audio_path)
        if os.path.exists(seg_cache):
            with open(seg_cache) as f:
                raw = json.load(f)
        else:
            rttm = self._run_inference(audio_path)
            raw = self._parse_rttm_for_child_segments(rttm)
            with open(seg_cache, "w") as f:
                json.dump(raw, f)

        out = []
        for s in raw:
            start, end = float(s["start"]), float(s["end"])
            dur = end - start
            if dur >= cfg.min_seg_dur_sec:
                out.append({"start": start, "end": end, "dur": dur})
        return out


# ----- Pyannote ------------------------------------------------

class PyannoteFrontend(DiarizationFrontend):
    def __init__(self, cfg: BaseConfig):
        self.cfg = cfg
        os.makedirs(cfg.pyannote_cache_dir, exist_ok=True)

        from pyannote.audio import Pipeline as PyannotePipeline

        if not cfg.hf_token:
            raise ValueError("Set HF_TOKEN in environment before running.")
        self.pipeline = PyannotePipeline.from_pretrained(
            cfg.pyannote_model, token=cfg.hf_token,
        )
        if "cuda" in cfg.device and torch.cuda.is_available():
            self.pipeline.to(torch.device("cuda"))

    def _rttm_cache_path(self, audio_path: str) -> str:
        cid = audio_to_cache_id(audio_path)
        stem = Path(audio_path).stem
        return os.path.join(self.cfg.pyannote_cache_dir, f"{stem}__{cid}.rttm")

    @staticmethod
    def _parse_rttm(rttm_path: str) -> List[Dict]:
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
                start, dur = float(parts[3]), float(parts[4])
                if dur <= 0:
                    continue
                segs.append({
                    "start": start, "end": start + dur,
                    "dur": dur, "speaker": parts[7],
                })
        return segs

    def _run_inference(self, audio_path: str) -> str:
        target = self._rttm_cache_path(audio_path)
        if os.path.exists(target):
            return target

        wav, sr = torchaudio.load(audio_path)
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        out = self.pipeline({"waveform": wav, "sample_rate": sr})
        ann = getattr(out, "speaker_diarization", out)
        with open(target, "w") as f:
            ann.write_rttm(f)
        return target

    def get_segments(self, audio_path: str, cfg: BaseConfig) -> List[Dict[str, float]]:
        rttm = self._run_inference(audio_path)
        segs = self._parse_rttm(rttm)
        return [s for s in segs if s["dur"] >= cfg.min_seg_dur_sec]


# ----- BabAR / VTC 2.0 ----------------------------------------

class BabARFrontend(DiarizationFrontend):
    """
    Runs the BabAR pipeline (which includes VTC 2.0) as a subprocess.
    VTC 2.0 produces RTTM with labels: KCHI, OCH, FEM, MAL.
    We extract KCHI segments for enrollment.

    BabAR expects a *folder* of wav files.  To avoid re-running the
    whole pipeline for every single clip, we batch-process all unique
    audio files up front via prepare(), then get_segments() just reads
    the cached RTTM.
    """

    def __init__(self, cfg: BaseConfig):
        self.cfg = cfg
        if not os.path.isdir(cfg.babar_dir):
            raise FileNotFoundError(f"BabAR directory not found: {cfg.babar_dir}")

        self.rttm_dir = os.path.join(cfg.babar_output_dir, "rttm")
        os.makedirs(cfg.babar_output_dir, exist_ok=True)

    def _rttm_path_for(self, audio_path: str) -> str:
        """BabAR names its RTTM files after the wav stem."""
        stem = Path(audio_path).stem
        return os.path.join(self.rttm_dir, f"{stem}.rttm")

    def prepare(self, audio_paths: List[str]):
        """
        Batch-run BabAR on all audio files that don't already have
        cached RTTM output.  BabAR expects a folder of wavs, so we
        symlink the needed files into a temp input directory.
        """
        missing = [p for p in audio_paths if not os.path.exists(self._rttm_path_for(p))]
        if not missing:
            print("BabAR: all RTTM files already cached.")
            return

        print(f"BabAR: running VTC 2.0 on {len(missing)} audio files...")

        # Create a temp input folder with symlinks
        input_dir = os.path.join(self.cfg.babar_output_dir, "_input_staging")
        os.makedirs(input_dir, exist_ok=True)

        # Symlink each file; handle name collisions by adding cache_id
        for ap in missing:
            stem = Path(ap).stem
            cid = audio_to_cache_id(ap)
            link_name = f"{stem}__{cid}.wav"
            link_path = os.path.join(input_dir, link_name)
            if not os.path.exists(link_path):
                os.symlink(os.path.abspath(ap), link_path)

        # Run BabAR pipeline
        device = "gpu" if "cuda" in self.cfg.device else "cpu"
        cmd = [
            "uv", "run", "src/pipeline.py",
            "--wavs", input_dir,
            "--output", self.cfg.babar_output_dir,
            "--device", device,
            "--batch_size", str(self.cfg.babar_batch_size),
        ]

        subprocess.run(
            cmd,
            cwd=self.cfg.babar_dir,
            check=True,
        )

        # BabAR names output by the symlink stem, so rename to match
        # original file stems for easier lookup.  Also keep the
        # collision-safe version.
        for ap in missing:
            stem = Path(ap).stem
            cid = audio_to_cache_id(ap)
            babar_name = f"{stem}__{cid}.rttm"
            babar_rttm = os.path.join(self.rttm_dir, babar_name)
            canonical = self._rttm_path_for(ap)

            # If original stems are unique (likely), copy to stem.rttm
            if babar_rttm != canonical and os.path.exists(babar_rttm):
                if not os.path.exists(canonical):
                    os.rename(babar_rttm, canonical)

    @staticmethod
    def _parse_rttm_for_kchi(rttm_path: str) -> List[Dict[str, float]]:
        """Parse RTTM and return only KCHI (key child) segments."""
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
                if label == "KCHI":
                    segs.append({"start": start, "end": start + dur, "dur": dur})
        return segs

    def get_segments(self, audio_path: str, cfg: BaseConfig) -> List[Dict[str, float]]:
        rttm = self._rttm_path_for(audio_path)
        if not os.path.exists(rttm):
            # If prepare() wasn't called or this file was missed,
            # run on just this file
            self.prepare([audio_path])
        segs = self._parse_rttm_for_kchi(rttm)
        return [s for s in segs if s["dur"] >= cfg.min_seg_dur_sec]


# =============================================================
# Shared pipeline
# =============================================================

def extract_segment_embeddings(
    audio_path: str,
    segments: List[Dict[str, float]],
    embedder: SpeakerEmbedder,
    cfg: BaseConfig,
    wav: Optional[torch.Tensor] = None,
) -> List[Tuple[np.ndarray, float]]:
    if wav is None:
        wav = load_audio_mono(audio_path, cfg.sample_rate)

    pairs: List[Tuple[np.ndarray, float]] = []
    for seg in segments:
        clip = crop_segment(wav, cfg.sample_rate, seg["start"], seg["end"])
        if clip.numel() < int(cfg.min_seg_dur_sec * cfg.sample_rate):
            continue
        try:
            emb = embedder.embed_waveform(clip)
            pairs.append((emb, seg["dur"]))
        except Exception:
            continue
    return pairs


# ---------- prototype building ---------------------------------

def build_child_prototypes(
    train_df: pd.DataFrame,
    frontend: DiarizationFrontend,
    embedder: SpeakerEmbedder,
    cfg: BaseConfig,
):
    prototypes: Dict[str, np.ndarray] = {}
    stats = []

    pos_train = train_df[train_df["label"] == 1].copy()

    for child_id, sub in pos_train.groupby("child_id"):
        all_pairs: List[Tuple[np.ndarray, float]] = []

        for _, row in sub.iterrows():
            ap = row["audio_path"]
            segs = frontend.get_segments(ap, cfg)
            pairs = extract_segment_embeddings(ap, segs, embedder, cfg)
            all_pairs.extend(pairs)
            if len(all_pairs) >= cfg.max_enrollment_segments_per_child:
                all_pairs = all_pairs[: cfg.max_enrollment_segments_per_child]
                break

        if not all_pairs:
            stats.append({"child_id": child_id, "n_segments": 0, "status": "no_valid_segments"})
            continue

        embs = np.stack([e for e, _ in all_pairs])
        weights = np.array([d for _, d in all_pairs])
        proto = np.average(embs, axis=0, weights=weights)
        prototypes[child_id] = l2_normalize(proto)
        stats.append({"child_id": child_id, "n_segments": len(all_pairs), "status": "ok"})

    return prototypes, pd.DataFrame(stats)


# ---------- scoring --------------------------------------------

def score_clip(
    audio_path: str,
    target_child_id: str,
    prototypes: Dict[str, np.ndarray],
    frontend: DiarizationFrontend,
    embedder: SpeakerEmbedder,
    cfg: BaseConfig,
) -> float:
    if target_child_id not in prototypes:
        return 0.0

    segs = frontend.get_segments(audio_path, cfg)
    if not segs:
        return 0.0

    wav = load_audio_mono(audio_path, cfg.sample_rate)
    proto = prototypes[target_child_id]

    scored: List[Tuple[float, float]] = []
    for seg in segs:
        clip = crop_segment(wav, cfg.sample_rate, seg["start"], seg["end"])
        if clip.numel() < int(cfg.min_seg_dur_sec * cfg.sample_rate):
            continue
        try:
            emb = embedder.embed_waveform(clip)
            scored.append((cosine_similarity(emb, proto), seg["dur"]))
        except Exception:
            continue

    if not scored:
        return 0.0

    total_dur = sum(d for _, d in scored)
    return float(sum(s * d for s, d in scored) / total_dur)


def run_enrollment(
    df: pd.DataFrame,
    prototypes: Dict[str, np.ndarray],
    frontend: DiarizationFrontend,
    embedder: SpeakerEmbedder,
    cfg: BaseConfig,
) -> pd.DataFrame:
    rows = []
    for _, row in df.iterrows():
        s = score_clip(row["audio_path"], row["child_id"], prototypes, frontend, embedder, cfg)
        rows.append({
            "audio_path": row["audio_path"],
            "child_id": row["child_id"],
            "timepoint_norm": row["timepoint_norm"],
            "label": int(row["label"]),
            "prob": float(s),
        })
    return pd.DataFrame(rows)


# ---------- role-only baseline ---------------------------------

def total_child_duration(
    audio_path: str, frontend: DiarizationFrontend, cfg: BaseConfig,
) -> float:
    return float(sum(s["dur"] for s in frontend.get_segments(audio_path, cfg)))


def run_role_only(
    df: pd.DataFrame, frontend: DiarizationFrontend, cfg: BaseConfig,
) -> pd.DataFrame:
    rows = []
    for _, row in df.iterrows():
        dur = total_child_duration(row["audio_path"], frontend, cfg)
        rows.append({
            "audio_path": row["audio_path"],
            "child_id": row["child_id"],
            "timepoint_norm": row["timepoint_norm"],
            "label": int(row["label"]),
            "score_duration_sec": float(dur),
        })
    return pd.DataFrame(rows)


def tune_role_only_threshold(val_role_df: pd.DataFrame, cfg: BaseConfig):
    y_true = val_role_df["label"].to_numpy()
    y_cont = val_role_df["score_duration_sec"].to_numpy().astype(float)

    best_t, best_f1 = cfg.duration_threshold_grid[0], -1.0
    for t in cfg.duration_threshold_grid:
        f = float(f1_score(y_true, (y_cont >= t).astype(int), zero_division=0))
        if f > best_f1:
            best_f1, best_t = f, t

    best_metrics = compute_metrics(y_true, y_cont, threshold=best_t)
    return float(best_t), best_metrics


def role_df_to_pred_df(role_df: pd.DataFrame, threshold_sec: float) -> pd.DataFrame:
    out = role_df.copy()
    out["prob"] = out["score_duration_sec"]
    out["pred_label"] = (out["score_duration_sec"] >= threshold_sec).astype(int)
    return out


# ---------- threshold tuning -----------------------------------

def tune_similarity_threshold(val_pred_df: pd.DataFrame, cfg: BaseConfig):
    y_true = val_pred_df["label"].to_numpy()
    y_prob = val_pred_df["prob"].to_numpy()

    thresholds = np.linspace(
        cfg.similarity_threshold_min,
        cfg.similarity_threshold_max,
        cfg.similarity_threshold_steps,
    )

    best_t = 0.5
    best_metrics = compute_metrics(y_true, y_prob, threshold=best_t)
    best_f1 = best_metrics["f1"]

    for t in thresholds:
        m = compute_metrics(y_true, y_prob, threshold=float(t))
        if m["f1"] > best_f1:
            best_f1 = m["f1"]
            best_t = float(t)
            best_metrics = m

    return best_t, best_metrics


# ----- VTC standalone ------------------------------------------

class VTCFrontend(DiarizationFrontend):
    """
    Standalone VTC 2.0 frontend (no BabAR phoneme step).

    VTC 2.0 label vocabulary: KCHI (key child), OCH (other child),
    MAL (male adult), FEM (female adult).

    diarizer_name selects which labels count as child:
      'vtc'      → child = [KCHI, OCH]
      'vtc_kchi' → child = [KCHI]

    Setup (once): cd BabAR/VTC && uv sync
    """

    def __init__(self, cfg: BaseConfig, diarizer_name: str):
        self.cfg = cfg
        self.child_labels = {"KCHI", "OCH"} if diarizer_name == "vtc" else {"KCHI"}
        os.makedirs(cfg.vtc_rttm_cache_dir, exist_ok=True)
        os.makedirs(cfg.vtc_input_staging_dir, exist_ok=True)
        ckpt = os.path.join(cfg.vtc_dir, "VTC-2.0", "model", "best.ckpt")
        if not os.path.exists(ckpt):
            raise FileNotFoundError(
                f"VTC checkpoint not found: {ckpt}\n"
                f"Run: cd {cfg.vtc_dir} && uv sync"
            )

    def _rttm_cache_path(self, audio_path: str) -> str:
        cid = audio_to_cache_id(audio_path)
        stem = Path(audio_path).stem
        return os.path.join(self.cfg.vtc_rttm_cache_dir, f"{stem}__{cid}.rttm")

    def _stage_audio(self, audio_path: str) -> str:
        cid = audio_to_cache_id(audio_path)
        stem = Path(audio_path).stem
        staged = os.path.join(self.cfg.vtc_input_staging_dir, f"{stem}__{cid}.wav")
        if not os.path.exists(staged):
            wav, sr = torchaudio.load(audio_path)
            if wav.shape[0] > 1:
                wav = wav.mean(dim=0, keepdim=True)
            if sr != 16000:
                wav = torchaudio.functional.resample(wav, sr, 16000)
            torchaudio.save(staged, wav, 16000)
        return staged

    def prepare(self, audio_paths: List[str]):
        """Batch-process all files that don't yet have a cached RTTM."""
        missing = [p for p in audio_paths
                   if not os.path.exists(self._rttm_cache_path(p))]
        if not missing:
            print("VTC: all RTTM files already cached.")
            return

        print(f"VTC: staging {len(missing)} audio file(s)...")
        staged_paths = [self._stage_audio(ap) for ap in missing]

        with tempfile.TemporaryDirectory() as tmp_out:
            input_dir = os.path.join(tmp_out, "wavs")
            os.makedirs(input_dir)
            for sp in staged_paths:
                dst = os.path.join(input_dir, Path(sp).name)
                if not os.path.exists(dst):
                    os.symlink(sp, dst)

            device = "cuda" if "cuda" in self.cfg.device else "cpu"
            cmd = [
                "uv", "run", "python", "scripts/infer.py",
                "--wavs", input_dir,
                "--output", tmp_out,
                "--config",
                os.path.join(self.cfg.vtc_dir, "VTC-2.0", "model", "config.yml"),
                "--checkpoint",
                os.path.join(self.cfg.vtc_dir, "VTC-2.0", "model", "best.ckpt"),
                "--device", device,
                "--batch_size", str(self.cfg.vtc_batch_size),
                "--min_duration_on_s", "0.1",
                "--min_duration_off_s", "0.1",
            ]
            subprocess.run(cmd, cwd=self.cfg.vtc_dir, check=True)

            for ap, sp in zip(missing, staged_paths):
                staged_stem = Path(sp).stem
                src = os.path.join(tmp_out, "rttm", f"{staged_stem}.rttm")
                dst = self._rttm_cache_path(ap)
                if os.path.exists(src):
                    shutil.copy2(src, dst)
                else:
                    open(dst, "w").close()

    def _parse_rttm_for_child(self, rttm_path: str) -> List[Dict[str, float]]:
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
                if label in self.child_labels and dur > 0:
                    segs.append({"start": start, "end": start + dur, "dur": dur})
        return segs

    def get_segments(self, audio_path: str, cfg: BaseConfig) -> List[Dict[str, float]]:
        rttm = self._rttm_cache_path(audio_path)
        if not os.path.exists(rttm):
            self.prepare([audio_path])
        segs = self._parse_rttm_for_child(rttm)
        return [s for s in segs if s["dur"] >= cfg.min_seg_dur_sec]


# ----- VBx -----------------------------------------------------

class VBxFrontend(DiarizationFrontend):
    """
    VBx speaker diarization frontend for the enrollment pipeline.

    Produces anonymous SPEAKER_XX segments.  The enrollment step
    (ECAPA cosine similarity) identifies which cluster matches the
    target child — no explicit role assignment needed.

    Requires HF_TOKEN (same as PyannoteFrontend).
    Setup (once): cd VBx && uv sync
    """

    def __init__(self, cfg: BaseConfig):
        self.cfg = cfg
        os.makedirs(cfg.vbx_rttm_cache_dir, exist_ok=True)
        if not os.path.isdir(cfg.vbx_dir):
            raise FileNotFoundError(
                f"VBx directory not found: {cfg.vbx_dir}\n"
                f"Run: cd {cfg.vbx_dir} && uv sync"
            )
        if not cfg.hf_token:
            raise ValueError(
                "Set HF_TOKEN env var before running VBx "
                "(needed for pyannote/segmentation-3.0 and pyannote/embedding)."
            )

    def _rttm_cache_path(self, audio_path: str) -> str:
        cid = audio_to_cache_id(audio_path)
        stem = Path(audio_path).stem
        return os.path.join(self.cfg.vbx_rttm_cache_dir, f"{stem}__{cid}.rttm")

    def _vbx_cmd_base(self, output_dir: str) -> List[str]:
        script = os.path.join(self.cfg.vbx_dir, "run_vbx.py")
        return [
            "uv", "run", "python", script,
            "--output", output_dir,
            "--hf-token", self.cfg.hf_token,
            "--max-speakers", str(self.cfg.vbx_max_speakers),
            "--niters", str(self.cfg.vbx_niters),
            "--Fa", str(self.cfg.vbx_Fa),
            "--Fb", str(self.cfg.vbx_Fb),
            "--loopP", str(self.cfg.vbx_loopP),
            "--win-duration", str(self.cfg.vbx_win_duration),
            "--win-step", str(self.cfg.vbx_win_step),
        ]

    def prepare(self, audio_paths: List[str]):
        """Batch-process all files, loading VBx models only once."""
        missing = [p for p in audio_paths
                   if not os.path.exists(self._rttm_cache_path(p))]
        if not missing:
            print("VBx: all RTTM files already cached.")
            return

        print(f"VBx: running on {len(missing)} audio file(s) (batch mode)...")
        with tempfile.TemporaryDirectory() as staging:
            input_dir = os.path.join(staging, "wavs")
            output_dir = os.path.join(staging, "output")
            os.makedirs(input_dir)

            for ap in missing:
                cid = audio_to_cache_id(ap)
                stem = Path(ap).stem
                dest = os.path.join(input_dir, f"{stem}__{cid}.wav")
                if not os.path.exists(dest):
                    wav, sr = torchaudio.load(ap)
                    if wav.shape[0] > 1:
                        wav = wav.mean(dim=0, keepdim=True)
                    if sr != 16000:
                        wav = torchaudio.functional.resample(wav, sr, 16000)
                    torchaudio.save(dest, wav, 16000)

            cmd = self._vbx_cmd_base(output_dir) + ["--audio-dir", input_dir]
            subprocess.run(cmd, cwd=self.cfg.vbx_dir, check=True)

            for ap in missing:
                cid = audio_to_cache_id(ap)
                stem = Path(ap).stem
                src = os.path.join(output_dir, "rttm", f"{stem}__{cid}.rttm")
                dst = self._rttm_cache_path(ap)
                if os.path.exists(src):
                    shutil.copy2(src, dst)
                else:
                    open(dst, "w").close()

    @staticmethod
    def _parse_rttm_all(rttm_path: str) -> List[Dict[str, float]]:
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
                start, dur = float(parts[3]), float(parts[4])
                if dur > 0:
                    segs.append({"start": start, "end": start + dur, "dur": dur})
        return segs

    def get_segments(self, audio_path: str, cfg: BaseConfig) -> List[Dict[str, float]]:
        rttm = self._rttm_cache_path(audio_path)
        if not os.path.exists(rttm):
            self.prepare([audio_path])
        segs = self._parse_rttm_all(rttm)
        return [s for s in segs if s["dur"] >= cfg.min_seg_dur_sec]


# =============================================================
# Factories
# =============================================================

def build_frontend(name: str, cfg: BaseConfig) -> DiarizationFrontend:
    if name == "usc_sail":
        return USCSailFrontend(cfg)
    elif name == "pyannote":
        return PyannoteFrontend(cfg)
    elif name == "babar":
        return BabARFrontend(cfg)
    elif name in ("vtc", "vtc_kchi"):
        return VTCFrontend(cfg, diarizer_name=name)
    elif name == "vbx":
        return VBxFrontend(cfg)
    elif name == "talknet_asd":
        return video_asd.TalkNetASDFrontend(cfg)
    elif name == "ts_talknet":
        return video_asd.TSTalkNetFrontend(cfg)
    elif name == "loconet_ecapa":
        return video_asd.LocoNetECAPAFrontend(cfg)
    elif name == "eend_eda":
        return nemo_diar.EENDEDAFrontend(cfg)
    elif name == "sortformer":
        return nemo_diar.SortformerFrontend(cfg)
    else:
        raise ValueError(f"Unknown diarizer: {name!r}")


def build_results_dir(diarizer: str) -> str:
    base = "/home/manaal/orcd/scratch/child-adult-diarization"
    if diarizer in ("talknet_asd", "ts_talknet", "loconet_ecapa"):
        return os.path.join(base, "video_asd_ecapa_enrollment_runs", diarizer)
    return os.path.join(base, f"{diarizer}_ecapa_enrollment_runs")





# =============================================================
# Main
# =============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--diarizer", required=True,
        choices=["usc_sail", "pyannote", "babar", "vtc", "vtc_kchi", "vbx",
                 "talknet_asd", "ts_talknet", "loconet_ecapa", "eend_eda", "sortformer"],
    )
    parser.add_argument(
        "--babar-dir", default="",
        help="Path to cloned BabAR repo (required if --diarizer babar).",
    )
    parser.add_argument(
        "--babar-batch-size", type=int, default=32,
        help="Batch size for BabAR GPU inference.",
    )
    parser.add_argument(
        "--vtc-dir", default="",
        help="Path to BabAR/VTC directory (overrides default).",
    )
    parser.add_argument(
        "--vtc-batch-size", type=int, default=64,
        help="Batch size for VTC GPU inference.",
    )
    parser.add_argument(
        "--vbx-dir", default="",
        help="Path to VBx directory (overrides default).",
    )
    parser.add_argument(
        "--vbx-max-speakers", type=int, default=8,
    )
    parser.add_argument(
        "--vbx-niters", type=int, default=10,
    )
    parser.add_argument("--vbx-Fa", type=float, default=0.1)
    parser.add_argument("--vbx-Fb", type=float, default=17.0)
    parser.add_argument("--vbx-loopP", type=float, default=0.99)
    parser.add_argument("--vbx-win-duration", type=float, default=1.5)
    parser.add_argument("--vbx-win-step", type=float, default=0.25)
    parser.add_argument(
        "--eend-eda-model-tag", default="",
        help="ESPnet Model Zoo tag or local dir for EEND-EDA (overrides default).",
    )
    parser.add_argument(
        "--eend-eda-num-spks", type=int, default=0,
        help="Fixed speaker count for EEND-EDA (0 = auto via EDA attractors).",
    )
    parser.add_argument(
        "--eend-eda-env-python", default="",
        help="Python interpreter for EEND-EDA subprocess (default: 'python').",
    )
    parser.add_argument(
        "--sortformer-model", default="",
        help="NeMo model name or .nemo path for Sortformer (overrides default).",
    )
    parser.add_argument(
        "--sortformer-max-speakers", type=int, default=0,
        help="Max speakers for Sortformer (0 = use default from config).",
    )
    parser.add_argument(
        "--sortformer-env-python", default="",
        help="Python interpreter for Sortformer subprocess (default: 'python').",
    )
    parser.add_argument(
        "--skip-role-only", action="store_true",
        help="Skip the role-only baseline.",
    )
    parser.add_argument(
        "--train-csv", default="",
        help="Override default train.csv (e.g. synthetic-augmented manifest).",
    )
    parser.add_argument(
        "--val-csv", default="",
        help="Override default val.csv.",
    )
    parser.add_argument(
        "--test-csv", default="",
        help="Override default test.csv.",
    )
    parser.add_argument(
        "--output-dir", default="",
        help="Override default results directory.",
    )
    args = parser.parse_args()

    cfg = BaseConfig()
    cfg.results_dir = args.output_dir if args.output_dir else build_results_dir(args.diarizer)
    if args.babar_dir:
        cfg.babar_dir = args.babar_dir
    cfg.babar_batch_size = args.babar_batch_size
    if args.vtc_dir:
        cfg.vtc_dir = args.vtc_dir
    cfg.vtc_batch_size = args.vtc_batch_size
    if args.vbx_dir:
        cfg.vbx_dir = args.vbx_dir
    cfg.vbx_max_speakers = args.vbx_max_speakers
    cfg.vbx_niters = args.vbx_niters
    cfg.vbx_Fa = args.vbx_Fa
    cfg.vbx_Fb = args.vbx_Fb
    cfg.vbx_loopP = args.vbx_loopP
    cfg.vbx_win_duration = args.vbx_win_duration
    cfg.vbx_win_step = args.vbx_win_step
    if args.eend_eda_model_tag:
        cfg.eend_eda_model_tag = args.eend_eda_model_tag
    if args.eend_eda_num_spks:
        cfg.eend_eda_num_spks = args.eend_eda_num_spks
    if args.eend_eda_env_python:
        cfg.eend_eda_env_python = args.eend_eda_env_python
    if args.sortformer_model:
        cfg.sortformer_model = args.sortformer_model
    if args.sortformer_max_speakers:
        cfg.sortformer_max_speakers = args.sortformer_max_speakers
    if args.sortformer_env_python:
        cfg.sortformer_env_python = args.sortformer_env_python

    os.makedirs(cfg.results_dir, exist_ok=True)
    save_json(asdict(cfg), os.path.join(cfg.results_dir, "config.json"))

    if args.train_csv or args.val_csv or args.test_csv:
        train_df = pd.read_csv(args.train_csv) if args.train_csv else pd.read_csv(os.path.join(cfg.split_dir, "train.csv"))
        val_df   = pd.read_csv(args.val_csv)   if args.val_csv   else pd.read_csv(os.path.join(cfg.split_dir, "val.csv"))
        test_df  = pd.read_csv(args.test_csv)  if args.test_csv  else pd.read_csv(os.path.join(cfg.split_dir, "test.csv"))
    else:
        train_df, val_df, test_df = load_split(cfg.split_dir)
    print(f"Train: {len(train_df)}  Val: {len(val_df)}  Test: {len(test_df)}")
    print(f"Diarizer: {args.diarizer}")

    frontend = build_frontend(args.diarizer, cfg)

    # Batch-process all audio files up front for diarizers that support it
    if isinstance(frontend, (BabARFrontend, VTCFrontend, VBxFrontend,
                              nemo_diar.EENDEDAFrontend, nemo_diar.SortformerFrontend)):
        all_audio = list(set(
            train_df["audio_path"].tolist()
            + val_df["audio_path"].tolist()
            + test_df["audio_path"].tolist()
        ))
        frontend.prepare(all_audio)

    # ---------------------------------------------------------
    # Role-only baseline
    # ---------------------------------------------------------
    if not args.skip_role_only:
        print("\nRunning role-only baseline...")
        val_role_df = run_role_only(val_df, frontend, cfg)
        test_role_df = run_role_only(test_df, frontend, cfg)

        role_t, role_val_metrics = tune_role_only_threshold(val_role_df, cfg)

        val_role_pred = role_df_to_pred_df(val_role_df, role_t)
        test_role_pred = role_df_to_pred_df(test_role_df, role_t)

        val_role_pred.to_csv(os.path.join(cfg.results_dir, "role_only_val_predictions.csv"), index=False)
        test_role_pred.to_csv(os.path.join(cfg.results_dir, "role_only_test_predictions.csv"), index=False)

        save_json({"threshold_sec": role_t, **role_val_metrics},
                  os.path.join(cfg.results_dir, "role_only_val_metrics.json"))

        role_test_metrics = compute_metrics(
            test_role_pred["label"].to_numpy(),
            test_role_pred["prob"].to_numpy(),
            threshold=role_t,
        )
        save_json({"threshold_sec": role_t, **role_test_metrics},
                  os.path.join(cfg.results_dir, "role_only_test_metrics.json"))

        per_timepoint_metrics(val_role_pred, role_t).to_csv(
            os.path.join(cfg.results_dir, "role_only_val_metrics_by_timepoint.csv"), index=False)
        per_timepoint_metrics(test_role_pred, role_t).to_csv(
            os.path.join(cfg.results_dir, "role_only_test_metrics_by_timepoint.csv"), index=False)

        print(f"Role-only threshold (sec): {role_t}")

    # ---------------------------------------------------------
    # Enrollment
    # ---------------------------------------------------------
    print("\nLoading ECAPA...")
    embedder = ECAPAEmbedder(cfg.ecapa_source, cfg.device)

    print("Building child prototypes from positive train clips...")
    prototypes, child_stats_df = build_child_prototypes(train_df, frontend, embedder, cfg)
    child_stats_df.to_csv(os.path.join(cfg.results_dir, "child_prototype_stats.csv"), index=False)
    print(f"Built prototypes for {len(prototypes)} children.")

    seen_children = set(train_df["child_id"].unique())
    missing = seen_children - set(prototypes.keys())
    if missing:
        print(f"WARNING: {len(missing)} seen children have no prototype "
              f"(diarizer found no segments): {missing}")

    print("Running enrollment on val / test...")
    val_enroll_df = run_enrollment(val_df, prototypes, frontend, embedder, cfg)
    test_enroll_df = run_enrollment(test_df, prototypes, frontend, embedder, cfg)

    sim_t, val_sim_metrics = tune_similarity_threshold(val_enroll_df, cfg)

    val_enroll_df = add_pred_labels(val_enroll_df, sim_t)
    test_enroll_df = add_pred_labels(test_enroll_df, sim_t)

    val_enroll_df.to_csv(os.path.join(cfg.results_dir, "enroll_val_predictions.csv"), index=False)
    test_enroll_df.to_csv(os.path.join(cfg.results_dir, "enroll_test_predictions.csv"), index=False)

    save_json({"threshold": sim_t, **val_sim_metrics},
              os.path.join(cfg.results_dir, "enroll_val_metrics.json"))

    test_sim_metrics = compute_metrics(
        test_enroll_df["label"].to_numpy(),
        test_enroll_df["prob"].to_numpy(),
        threshold=sim_t,
    )
    save_json({"threshold": sim_t, **test_sim_metrics},
              os.path.join(cfg.results_dir, "enroll_test_metrics.json"))

    per_timepoint_metrics(val_enroll_df, sim_t).to_csv(
        os.path.join(cfg.results_dir, "enroll_val_metrics_by_timepoint.csv"), index=False)
    per_timepoint_metrics(test_enroll_df, sim_t).to_csv(
        os.path.join(cfg.results_dir, "enroll_test_metrics_by_timepoint.csv"), index=False)

    print(f"\nDone.  Enrollment threshold: {sim_t:.3f}")
    print(f"Test metrics: {test_sim_metrics}")


if __name__ == "__main__":
    main()
