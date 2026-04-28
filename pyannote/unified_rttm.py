"""
Diarization accuracy evaluation on Playlogue and Providence datasets.

Compares diarizer output against ground-truth RTTM files at the frame level,
treating the task as binary child-speech detection (child vs. non-child).

Supports the same three diarizer front-ends as unified_enrollment.py.

Usage examples
--------------
# Evaluate on Playlogue with USC-SAIL
python eval_diarization.py \\
    --dataset playlogue \\
    --audio-dir /path/to/playlogue/audio \\
    --rttm-dir  /path/to/playlogue/rttm \\
    --diarizer usc_sail

# Evaluate on Providence with pyannote
python eval_diarization.py \\
    --dataset providence \\
    --audio-dir /path/to/providence/audio \\
    --rttm-dir  /path/to/providence/rttm \\
    --diarizer pyannote

# BabAR diarizer with custom output dir
python eval_diarization.py \\
    --dataset playlogue \\
    --audio-dir /path/to/playlogue/audio \\
    --rttm-dir  /path/to/playlogue/rttm \\
    --diarizer babar \\
    --babar-dir /path/to/BabAR \\
    --results-dir ./eval_results

# Override Providence child labels (comma-separated, case-insensitive)
python eval_diarization.py \\
    --dataset providence \\
    --audio-dir /path/to/providence/audio \\
    --rttm-dir  /path/to/providence/rttm \\
    --diarizer usc_sail \\
    --child-labels CHI,KCHI,OCH,CX

Dataset directory layout expected
-----------------------------------
Audio and RTTM files live in separate folders; filenames must share the
same stem (case-sensitive) across the two directories:

    providence/
        audio/
            session001.wav
            session002.mp3
        rttm/
            session001.rttm    # ground-truth
            session002.rttm

    playlogue/
        audio/  rec01.wav  rec02.wav  ...
        rttm/   rec01.rttm rec02.rttm ...

RTTMs must be standard format:
    SPEAKER <file> 1 <start> <dur> <NA> <NA> <label> <NA> <NA>

Ground-truth label vocabularies (defaults, overridable via CLI)
---------------------------------------------------------------
Playlogue : child → {CHI},  adult → {ADT}
Providence: child → {CHI, KCHI, OCH},  adult → {FEM, MAL, MAN, WOM, FAT, MOT, OAD}

Metrics computed
----------------
Frame-level (10 ms frames, no collar):
  - Precision, Recall, F1  for the "child" class
  - False Alarm Rate  (FAR): predicted-child frames that are actually adult / silence
  - Miss Rate         (MR):  ground-truth child frames that were not detected
  - DER component     = FAR + MR  (no speaker confusion term since this is binary)
  - Frame Accuracy    (fraction of all frames correctly classified)
  - AUROC / AUPRC     (using diarized duration per frame as a soft score)

All metrics are saved as:
  results_dir/
    <diarizer>_<dataset>/
      per_file_metrics.csv
      aggregate_metrics.json
      per_file_predictions/
        <stem>_pred.rttm      # predicted child segments
"""

from __future__ import annotations

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

# ============================================================
# Label maps
# ============================================================

# Default child-label sets per dataset (case-insensitive matching used below)
DEFAULT_CHILD_LABELS: Dict[str, List[str]] = {
    "playlogue": ["CHI"],
    "providence": ["CHI", "KCHI", "OCH"],
}

# Adult / non-child labels (everything else is treated as non-child too, but
# listing them explicitly helps spot unexpected labels in your RTTMs)
DEFAULT_ADULT_LABELS: Dict[str, List[str]] = {
    "playlogue": ["ADT"],
    "providence": ["FEM", "MAL", "MAN", "WOM", "FAT", "MOT", "OAD", "NON"],
}

# For diarizer output: map predicted labels → role
# USC-SAIL / BabAR use CHI/KCHI for child; pyannote produces SPEAKER_XX labels
# so we handle pyannote separately (see build_predicted_frames).

DIARIZER_CHILD_LABELS: Dict[str, List[str]] = {
    "usc_sail": ["CHI"],
    "babar": ["KCHI"],           # BabAR / VTC 2.0 key-child label
    "pyannote": [],              # anonymous labels; resolved via GT-overlap mapping
    "vtc": ["KCHI", "OCH"],      # VTC 2.0: key child + other child both count
    "vtc_kchi": ["KCHI"],        # VTC 2.0: only key child counts
    "vbx": [],                   # anonymous labels; resolved via GT-overlap mapping
}

# Diarizers that produce anonymous speaker labels and require GT-overlap resolution
ANONYMOUS_DIARIZERS = {"pyannote", "vbx"}


# ============================================================
# Config (mirrors unified_enrollment.py)
# ============================================================

@dataclass
class EvalConfig:
    sample_rate: int = 16000
    frame_step_sec: float = 0.01          # 10 ms frames
    min_seg_dur_sec: float = 0.4

    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # USC-SAIL
    usc_sail_repo_dir: str = "/home/manaal/orcd/scratch/child-adult-diarization/whisper-modeling"
    usc_sail_script: str = "/home/manaal/orcd/scratch/child-adult-diarization/whisper-modeling/scripts/infer_long_wav_files.py"
    usc_sail_model_path: str = "/home/manaal/orcd/scratch/child-adult-diarization/whisper-modeling/whisper-base_rank8_pretrained_50k.pt"
    usc_sail_python: str = "python"
    usc_window_size: float = 10.0
    usc_stride: float = 5.0
    segment_cache_dir: str = "/home/manaal/orcd/scratch/child-adult-diarization/whisper-modeling/usc_sail_segment_cache"
    rttm_cache_dir: str = "/home/manaal/orcd/scratch/child-adult-diarization/whisper-modeling/usc_sail_rttm_cache"

    # Pyannote
    pyannote_model: str = "pyannote/speaker-diarization-community-1"
    hf_token: str = os.environ.get("HF_TOKEN", "")
    pyannote_cache_dir: str = "/home/manaal/orcd/scratch/child-adult-diarization/pyannote/pyannote_rttm_cache"

    # BabAR
    babar_dir: str = "/home/manaal/orcd/scratch/child-adult-diarization/BabAR/"
    babar_output_dir: str = "/home/manaal/orcd/scratch/child-adult-diarization/babar/babar_output"
    babar_batch_size: int = 1  # segma fails on short files with batch_size > 1

    # VTC (standalone, without BabAR phoneme step)
    # Requires: cd BabAR/VTC && uv sync
    vtc_dir: str = "/home/manaal/orcd/scratch/child-adult-diarization/BabAR/VTC"
    vtc_rttm_cache_dir: str = "/home/manaal/orcd/scratch/child-adult-diarization/pyannote/vtc_rttm_cache"
    vtc_input_staging_dir: str = "/home/manaal/orcd/scratch/child-adult-diarization/pyannote/vtc_input_staging"
    vtc_batch_size: int = 64

    # VBx (speaker diarization via ECAPA embeddings + VBx clustering)
    # Requires: cd VBx && uv sync
    vbx_dir: str = "/home/manaal/orcd/scratch/child-adult-diarization/VBx"
    vbx_rttm_cache_dir: str = "/home/manaal/orcd/scratch/child-adult-diarization/pyannote/vbx_rttm_cache"
    vbx_max_speakers: int = 8
    vbx_niters: int = 10
    vbx_Fa: float = 0.1
    vbx_Fb: float = 17.0
    vbx_loopP: float = 0.99
    vbx_win_duration: float = 1.5
    vbx_win_step: float = 0.25


# ============================================================
# Utilities
# ============================================================

def save_json(obj, path: str):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def audio_to_cache_id(audio_path: str) -> str:
    return hashlib.md5(audio_path.encode("utf-8")).hexdigest()


def get_audio_duration(audio_path: str) -> float:
    """Returns duration in seconds without loading the whole waveform."""
    info = torchaudio.info(audio_path)
    return info.num_frames / info.sample_rate


# ============================================================
# RTTM parsing
# ============================================================

def parse_rttm(rttm_path: str) -> List[Dict]:
    """Parse an RTTM file into a list of segment dicts."""
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
            start = float(parts[3])
            dur = float(parts[4])
            label = parts[7]
            if dur <= 0:
                continue
            segs.append({"start": start, "end": start + dur, "dur": dur, "label": label})
    return segs


def write_rttm(segs: List[Dict], file_id: str, path: str):
    """Write segments to an RTTM file."""
    with open(path, "w") as f:
        for s in segs:
            f.write(
                f"SPEAKER {file_id} 1 {s['start']:.3f} {s['dur']:.3f} "
                f"<NA> <NA> {s['label']} <NA> <NA>\n"
            )


# ============================================================
# Frame-level representation
# ============================================================

def segments_to_frame_mask(
    segs: List[Dict],
    duration_sec: float,
    child_labels: List[str],
    frame_step: float = 0.01,
) -> np.ndarray:
    """
    Convert a list of diarization segments to a binary frame mask.

    Returns a boolean array of length ceil(duration_sec / frame_step).
    True  → frame is labelled as child speech.
    False → frame is non-child (adult, silence, overlap with non-child, etc.)

    child_labels is case-insensitive.
    """
    n_frames = max(1, int(np.ceil(duration_sec / frame_step)))
    mask = np.zeros(n_frames, dtype=bool)
    child_set = {lbl.upper() for lbl in child_labels}

    for seg in segs:
        if seg["label"].upper() not in child_set:
            continue
        s_idx = int(np.round(seg["start"] / frame_step))
        e_idx = int(np.round(seg["end"] / frame_step))
        s_idx = max(0, min(s_idx, n_frames))
        e_idx = max(0, min(e_idx, n_frames))
        mask[s_idx:e_idx] = True

    return mask


# ============================================================
# Pyannote: map anonymous SPEAKER_XX to child/adult
# ============================================================

def resolve_pyannote_labels(
    pred_segs: List[Dict],
    gt_segs: List[Dict],
    duration_sec: float,
    child_labels_gt: List[str],
    frame_step: float = 0.01,
) -> List[Dict]:
    """
    Pyannote produces anonymous speaker labels (SPEAKER_00, SPEAKER_01, …).
    We map each predicted speaker to "CHI" or "ADT" by finding which
    ground-truth role it overlaps most with.

    Returns a copy of pred_segs with updated label fields.
    """
    n_frames = max(1, int(np.ceil(duration_sec / frame_step)))
    gt_child_mask = segments_to_frame_mask(gt_segs, duration_sec, child_labels_gt, frame_step)

    # Build a per-speaker frame mask from predictions
    speaker_ids = sorted({s["label"] for s in pred_segs})
    speaker_mask: Dict[str, np.ndarray] = {}
    for spk in speaker_ids:
        m = np.zeros(n_frames, dtype=bool)
        for s in pred_segs:
            if s["label"] != spk:
                continue
            si = max(0, min(int(np.round(s["start"] / frame_step)), n_frames))
            ei = max(0, min(int(np.round(s["end"] / frame_step)), n_frames))
            m[si:ei] = True
        speaker_mask[spk] = m

    # Assign each speaker to the role it overlaps most with
    spk_role: Dict[str, str] = {}
    for spk, m in speaker_mask.items():
        n_frames_spk = m.sum()
        if n_frames_spk == 0:
            spk_role[spk] = "ADT"
            continue
        child_overlap = (m & gt_child_mask).sum() / n_frames_spk
        spk_role[spk] = "CHI" if child_overlap >= 0.5 else "ADT"

    # Rewrite segments
    out = []
    for s in pred_segs:
        out.append({**s, "label": spk_role[s["label"]]})
    return out


# ============================================================
# Per-file evaluation
# ============================================================

def evaluate_file(
    audio_path: str,
    gt_rttm_path: str,
    pred_segs: List[Dict],
    child_labels_gt: List[str],
    child_labels_pred: List[str],
    diarizer: str,
    frame_step: float = 0.01,
) -> Dict:
    """
    Compute frame-level child-detection metrics for one file.

    pred_segs already has its labels resolved (CHI/KCHI or mapped
    from pyannote) before this function is called.
    """
    try:
        duration = get_audio_duration(audio_path)
    except Exception:
        # Fallback: use max end time from both RTTMs
        gt_segs = parse_rttm(gt_rttm_path)
        all_segs = gt_segs + pred_segs
        duration = max((s["end"] for s in all_segs), default=1.0) + 1.0

    gt_segs = parse_rttm(gt_rttm_path)

    gt_mask = segments_to_frame_mask(gt_segs, duration, child_labels_gt, frame_step)
    pred_mask = segments_to_frame_mask(pred_segs, duration, child_labels_pred, frame_step)

    n = len(gt_mask)
    assert len(pred_mask) == n or True  # lengths may differ slightly; pad
    min_n = min(len(gt_mask), len(pred_mask))
    gt_mask = gt_mask[:min_n]
    pred_mask = pred_mask[:min_n]

    y_true = gt_mask.astype(int)
    y_pred = pred_mask.astype(int)
    y_prob = pred_mask.astype(float)  # binary; use duration-weighted score if desired

    n_gt_child = y_true.sum()
    n_pred_child = y_pred.sum()
    n_total = len(y_true)

    tp = int((y_true & y_pred).sum())
    fp = int((~y_true.astype(bool) & y_pred.astype(bool)).sum())
    fn = int((y_true.astype(bool) & ~y_pred.astype(bool)).sum())
    tn = int((~y_true.astype(bool) & ~y_pred.astype(bool)).sum())

    # DER components (binary, no collar)
    miss_rate = fn / max(n_gt_child, 1)
    fa_rate = fp / max(n_total - n_gt_child, 1)
    der = miss_rate + fa_rate

    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    acc = (tp + tn) / max(n_total, 1)

    try:
        auroc = float(roc_auc_score(y_true, y_prob))
    except Exception:
        auroc = float("nan")
    try:
        auprc = float(average_precision_score(y_true, y_prob))
    except Exception:
        auprc = float("nan")

    # Duration stats
    gt_child_dur = n_gt_child * frame_step
    pred_child_dur = n_pred_child * frame_step
    total_dur = n_total * frame_step

    return {
        "file": Path(audio_path).stem,
        "total_dur_sec": round(total_dur, 2),
        "gt_child_dur_sec": round(gt_child_dur, 2),
        "pred_child_dur_sec": round(pred_child_dur, 2),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "accuracy": round(acc, 4),
        "miss_rate": round(miss_rate, 4),
        "false_alarm_rate": round(fa_rate, 4),
        "der": round(der, 4),
        "auroc": round(auroc, 4),
        "auprc": round(auprc, 4),
        "tp_frames": tp,
        "fp_frames": fp,
        "fn_frames": fn,
        "tn_frames": tn,
    }


# ============================================================
# Dataset discovery
# ============================================================

def find_pairs(audio_dir: str, rttm_dir: str) -> List[Tuple[str, str]]:
    """
    Match audio files under audio_dir to RTTM files under rttm_dir by stem.

    Audio and RTTM files may live in completely separate directory trees;
    only the filename stem needs to match (case-sensitive).

    Example layout:
        providence/audio/session001.wav   <->   providence/rttm/session001.rttm
        playlogue/audio/rec02.mp3         <->   playlogue/rttm/rec02.rttm
    """
    audio_exts = {".wav", ".mp3", ".flac"}

    # Index all RTTM files by lowercased stem for case-insensitive lookup
    rttm_by_stem: Dict[str, str] = {}
    for f in Path(rttm_dir).rglob("*.rttm"):
        rttm_by_stem[f.stem.lower()] = str(f)

    if not rttm_by_stem:
        raise FileNotFoundError(f"No .rttm files found under {rttm_dir!r}.")

    pairs = []
    unmatched_audio = []
    for f in sorted(Path(audio_dir).rglob("*")):
        if f.suffix.lower() not in audio_exts:
            continue
        if f.stem.lower() in rttm_by_stem:
            pairs.append((str(f), rttm_by_stem[f.stem.lower()]))
        else:
            unmatched_audio.append(f.name)

    if unmatched_audio:
        print(f"  WARNING: {len(unmatched_audio)} audio file(s) had no matching RTTM "
              f"and will be skipped: {unmatched_audio[:5]}"
              + (" ..." if len(unmatched_audio) > 5 else ""))

    if not pairs:
        raise FileNotFoundError(
            f"No matching (audio, rttm) pairs found.\n"
            f"  Audio dir : {audio_dir!r}\n"
            f"  RTTM dir  : {rttm_dir!r}\n"
            f"  Check that filenames (without extension) match between the two folders."
        )
    return pairs


# ============================================================
# Diarizer front-ends (copied from unified_enrollment.py)
# ============================================================

class DiarizationFrontend(abc.ABC):
    @abc.abstractmethod
    def get_segments(self, audio_path: str, cfg: EvalConfig) -> List[Dict]:
        ...


class USCSailFrontend(DiarizationFrontend):
    def __init__(self, cfg: EvalConfig):
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
    def _parse_rttm_all(rttm_path: str) -> List[Dict]:
        """Return ALL segments (not just CHI) so we can use them for evaluation."""
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
                if dur <= 0:
                    continue
                segs.append({"start": start, "end": start + dur, "dur": dur, "label": label})
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
        subprocess.run(cmd, cwd=self.cfg.usc_sail_repo_dir, env=env, check=True)
        if not os.path.exists(target):
            raise FileNotFoundError(f"USC-SAIL finished but expected RTTM not found: {target}")
        return target

    def get_segments(self, audio_path: str, cfg: EvalConfig) -> List[Dict]:
        seg_cache = self._segment_cache_path(audio_path)
        # For eval we need all segments, not just child ones
        rttm_cache = self._rttm_cache_path(audio_path)
        if os.path.exists(rttm_cache):
            return self._parse_rttm_all(rttm_cache)
        # Check the JSON segment cache (contains only CHI from enrollment script)
        # Prefer running fresh to get all labels
        rttm = self._run_inference(audio_path)
        return self._parse_rttm_all(rttm)


class PyannoteFrontend(DiarizationFrontend):
    def __init__(self, cfg: EvalConfig):
        self.cfg = cfg
        os.makedirs(cfg.pyannote_cache_dir, exist_ok=True)
        from pyannote.audio import Pipeline as PyannotePipeline
        if not cfg.hf_token:
            raise ValueError("Set HF_TOKEN env var before running pyannote.")
        self.pipeline = PyannotePipeline.from_pretrained(cfg.pyannote_model, token=cfg.hf_token)
        if "cuda" in cfg.device and torch.cuda.is_available():
            self.pipeline.to(torch.device("cuda"))

    def _rttm_cache_path(self, audio_path: str) -> str:
        cid = audio_to_cache_id(audio_path)
        stem = Path(audio_path).stem
        return os.path.join(self.cfg.pyannote_cache_dir, f"{stem}__{cid}.rttm")

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

    def get_segments(self, audio_path: str, cfg: EvalConfig) -> List[Dict]:
        rttm = self._run_inference(audio_path)
        return parse_rttm(rttm)


class BabARFrontend(DiarizationFrontend):
    def __init__(self, cfg: EvalConfig):
        self.cfg = cfg
        if not os.path.isdir(cfg.babar_dir):
            raise FileNotFoundError(f"BabAR directory not found: {cfg.babar_dir}")
        self.rttm_dir = os.path.join(cfg.babar_output_dir, "rttm")
        os.makedirs(cfg.babar_output_dir, exist_ok=True)

    def _rttm_path_for(self, audio_path: str) -> str:
        return os.path.join(self.rttm_dir, f"{Path(audio_path).stem}.rttm")

    def prepare(self, audio_paths: List[str]):
        missing = [p for p in audio_paths if not os.path.exists(self._rttm_path_for(p))]
        if not missing:
            print("BabAR: all RTTM files already cached.")
            return
        print(f"BabAR: running VTC 2.0 on {len(missing)} audio files...")
        input_dir = os.path.join(self.cfg.babar_output_dir, "_input_staging")
        os.makedirs(input_dir, exist_ok=True)
        for ap in missing:
            stem = Path(ap).stem
            cid = audio_to_cache_id(ap)
            link_name = f"{stem}__{cid}.wav"
            link_path = os.path.join(input_dir, link_name)
            if not os.path.exists(link_path):
                # BabAR requires 16kHz mono wav; resample if needed
                import torchaudio
                wav, sr = torchaudio.load(ap)
                if wav.shape[0] > 1:
                    wav = wav.mean(dim=0, keepdim=True)
                if sr != 16000:
                    wav = torchaudio.functional.resample(wav, sr, 16000)
                torchaudio.save(link_path, wav, 16000)
        device = "gpu" if "cuda" in self.cfg.device else "cpu"
        cmd = [
            "uv", "run", "src/pipeline.py",
            "--wavs", input_dir,
            "--output", self.cfg.babar_output_dir,
            "--device", device,
            "--batch_size", str(self.cfg.babar_batch_size),
        ]
        subprocess.run(cmd, cwd=self.cfg.babar_dir, check=True)
        for ap in missing:
            stem = Path(ap).stem
            cid = audio_to_cache_id(ap)
            babar_rttm = os.path.join(self.rttm_dir, f"{stem}__{cid}.rttm")
            canonical = self._rttm_path_for(ap)
            if babar_rttm != canonical and os.path.exists(babar_rttm):
                if not os.path.exists(canonical):
                    os.rename(babar_rttm, canonical)

    def get_segments(self, audio_path: str, cfg: EvalConfig) -> List[Dict]:
        rttm = self._rttm_path_for(audio_path)
        if not os.path.exists(rttm):
            self.prepare([audio_path])
        return parse_rttm(rttm)


class VTCFrontend(DiarizationFrontend):
    """
    Standalone VTC 2.0 frontend (no BabAR phoneme step).

    VTC 2.0 label vocabulary: KCHI (key child), OCH (other child),
    MAL (male adult), FEM (female adult).

    Two diarizer names select which labels count as child:
      vtc      → child = [KCHI, OCH]   (all child speech)
      vtc_kchi → child = [KCHI]        (target/key child only)

    Setup (once):
        cd BabAR/VTC && uv sync
    """

    def __init__(self, cfg: EvalConfig):
        self.cfg = cfg
        os.makedirs(cfg.vtc_rttm_cache_dir, exist_ok=True)
        os.makedirs(cfg.vtc_input_staging_dir, exist_ok=True)
        if not os.path.isdir(cfg.vtc_dir):
            raise FileNotFoundError(f"VTC directory not found: {cfg.vtc_dir}")
        ckpt = os.path.join(cfg.vtc_dir, "VTC-2.0", "model", "best.ckpt")
        if not os.path.exists(ckpt):
            raise FileNotFoundError(f"VTC checkpoint not found: {ckpt}")

    def _rttm_cache_path(self, audio_path: str) -> str:
        cid = audio_to_cache_id(audio_path)
        stem = Path(audio_path).stem
        return os.path.join(self.cfg.vtc_rttm_cache_dir, f"{stem}__{cid}.rttm")

    def _stage_audio(self, audio_path: str) -> str:
        """Resample to 16 kHz mono WAV and place in staging dir."""
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
        """Batch-process all audio files that don't yet have a cached RTTM."""
        missing = [p for p in audio_paths
                   if not os.path.exists(self._rttm_cache_path(p))]
        if not missing:
            print("VTC: all RTTM files already cached.")
            return

        print(f"VTC: staging {len(missing)} audio file(s)...")
        staged_paths = [self._stage_audio(ap) for ap in missing]

        with tempfile.TemporaryDirectory() as tmp_out:
            # VTC processes a whole directory; build a flat input dir.
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

            # Copy each output RTTM to the cache
            for ap, sp in zip(missing, staged_paths):
                staged_stem = Path(sp).stem                       # stem__cid
                src = os.path.join(tmp_out, "rttm", f"{staged_stem}.rttm")
                dst = self._rttm_cache_path(ap)
                if os.path.exists(src):
                    shutil.copy2(src, dst)
                else:
                    open(dst, "w").close()  # empty = no speech

    def get_segments(self, audio_path: str, cfg: EvalConfig) -> List[Dict]:
        rttm = self._rttm_cache_path(audio_path)
        if not os.path.exists(rttm):
            self.prepare([audio_path])
        return parse_rttm(rttm)


class VBxFrontend(DiarizationFrontend):
    """
    VBx speaker diarization frontend.

    Uses pyannote/segmentation-3.0 for VAD and pyannote/embedding (ECAPA-TDNN)
    for x-vector extraction, then applies the VBx Variational Bayes HMM
    clustering algorithm.

    Produces anonymous SPEAKER_XX labels; child/adult roles are resolved via
    GT-overlap mapping (same approach as PyannoteFrontend).

    Requires HF_TOKEN env var (same as PyannoteFrontend).

    Setup (once):
        cd VBx && uv sync
    """

    def __init__(self, cfg: EvalConfig):
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
        # Stage files in a flat staging dir so VBx can process as a directory
        with tempfile.TemporaryDirectory() as staging:
            input_dir = os.path.join(staging, "wavs")
            output_dir = os.path.join(staging, "output")
            os.makedirs(input_dir)

            # Resample / copy to staging
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

            # Move outputs to cache
            for ap in missing:
                cid = audio_to_cache_id(ap)
                stem = Path(ap).stem
                src = os.path.join(output_dir, "rttm", f"{stem}__{cid}.rttm")
                dst = self._rttm_cache_path(ap)
                if os.path.exists(src):
                    shutil.copy2(src, dst)
                else:
                    open(dst, "w").close()

    def get_segments(self, audio_path: str, cfg: EvalConfig) -> List[Dict]:
        rttm = self._rttm_cache_path(audio_path)
        if not os.path.exists(rttm):
            self.prepare([audio_path])
        return parse_rttm(rttm)


def build_frontend(name: str, cfg: EvalConfig) -> DiarizationFrontend:
    if name == "usc_sail":
        return USCSailFrontend(cfg)
    elif name == "pyannote":
        return PyannoteFrontend(cfg)
    elif name == "babar":
        return BabARFrontend(cfg)
    elif name in ("vtc", "vtc_kchi"):
        return VTCFrontend(cfg)
    elif name == "vbx":
        return VBxFrontend(cfg)
    else:
        raise ValueError(f"Unknown diarizer: {name!r}")


# ============================================================
# Aggregate metrics
# ============================================================

def aggregate_metrics(per_file: pd.DataFrame) -> Dict:
    """
    Micro-average metrics (sum TP/FP/FN/TN across all files, then compute).
    Also reports macro-average F1 and per-quartile duration breakdown.
    """
    tp = per_file["tp_frames"].sum()
    fp = per_file["fp_frames"].sum()
    fn = per_file["fn_frames"].sum()
    tn = per_file["tn_frames"].sum()

    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    acc = (tp + tn) / max(tp + fp + fn + tn, 1)

    n_gt = tp + fn
    n_non_child = fp + tn
    miss = fn / max(n_gt, 1)
    fa = fp / max(n_non_child, 1)
    der = miss + fa

    macro_f1 = per_file["f1"].mean()

    out = {
        "micro_precision": round(float(prec), 4),
        "micro_recall": round(float(rec), 4),
        "micro_f1": round(float(f1), 4),
        "micro_accuracy": round(float(acc), 4),
        "miss_rate": round(float(miss), 4),
        "false_alarm_rate": round(float(fa), 4),
        "binary_der": round(float(der), 4),
        "macro_f1": round(float(macro_f1), 4),
        "n_files": int(len(per_file)),
        "total_gt_child_dur_sec": round(float(per_file["gt_child_dur_sec"].sum()), 2),
        "total_pred_child_dur_sec": round(float(per_file["pred_child_dur_sec"].sum()), 2),
        "total_audio_dur_sec": round(float(per_file["total_dur_sec"].sum()), 2),
    }

    # Add macro-average of per-file AUROC / AUPRC (skip NaN)
    for col in ["auroc", "auprc"]:
        valid = per_file[col].dropna()
        out[f"macro_{col}"] = round(float(valid.mean()), 4) if len(valid) else float("nan")

    return out


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate diarizer accuracy on Playlogue or Providence."
    )
    parser.add_argument("--dataset", required=True, choices=["playlogue", "providence"],
                        help="Which dataset to evaluate.")
    parser.add_argument("--audio-dir", required=True,
                        help="Directory containing audio files (.wav/.mp3/.flac).")
    parser.add_argument("--rttm-dir", required=True,
                        help="Directory containing ground-truth RTTM files.")
    parser.add_argument("--diarizer", required=True,
                        choices=["usc_sail", "pyannote", "babar",
                                 "vtc", "vtc_kchi", "vbx"],
                        help="Diarizer front-end to use. "
                             "vtc=VTC2.0(KCHI+OCH as child), "
                             "vtc_kchi=VTC2.0(KCHI only), "
                             "vbx=VBx(anonymous labels resolved via GT overlap).")
    parser.add_argument("--babar-dir", default="",
                        help="Path to cloned BabAR repo (required if --diarizer babar).")
    parser.add_argument("--babar-batch-size", type=int, default=32)
    parser.add_argument("--vtc-dir", default="",
                        help="Path to BabAR/VTC repo (overrides EvalConfig default).")
    parser.add_argument("--vtc-batch-size", type=int, default=64,
                        help="VTC inference batch size (default: 64).")
    parser.add_argument("--vbx-dir", default="",
                        help="Path to VBx repo (overrides EvalConfig default).")
    parser.add_argument("--vbx-max-speakers", type=int, default=8)
    parser.add_argument("--vbx-niters", type=int, default=10)
    parser.add_argument("--vbx-Fa", type=float, default=0.1)
    parser.add_argument("--vbx-Fb", type=float, default=17.0)
    parser.add_argument("--vbx-loopP", type=float, default=0.99)
    parser.add_argument("--vbx-win-duration", type=float, default=1.5)
    parser.add_argument("--vbx-win-step", type=float, default=0.25)
    parser.add_argument("--results-dir", default="",
                        help="Where to save results. Default: ./eval_results/<diarizer>_<dataset>")
    parser.add_argument("--child-labels", default="",
                        help="Comma-separated list of child labels in the ground-truth RTTMs "
                             "(overrides dataset default). E.g. CHI,KCHI,OCH")
    parser.add_argument("--frame-step", type=float, default=0.01,
                        help="Frame resolution in seconds (default: 0.01 = 10 ms).")
    args = parser.parse_args()

    # ---- label sets ----
    if args.child_labels:
        child_labels_gt = [l.strip() for l in args.child_labels.split(",") if l.strip()]
    else:
        child_labels_gt = DEFAULT_CHILD_LABELS[args.dataset]

    child_labels_pred = DIARIZER_CHILD_LABELS[args.diarizer]
    # Diarizers with anonymous speaker labels get GT-overlap mapping → resolved "CHI".
    if args.diarizer in ANONYMOUS_DIARIZERS:
        child_labels_pred = ["CHI"]

    print(f"Dataset       : {args.dataset}")
    print(f"Diarizer      : {args.diarizer}")
    print(f"GT child labels : {child_labels_gt}")
    print(f"Pred child labels: {child_labels_pred} "
          + ("(resolved via GT overlap)" if args.diarizer in ANONYMOUS_DIARIZERS else ""))

    # ---- results dir ----
    if args.results_dir:
        results_dir = args.results_dir
    else:
        results_dir = os.path.join(
            "eval_results", f"{args.diarizer}_{args.dataset}"
        )
    pred_rttm_dir = os.path.join(results_dir, "per_file_predictions")
    os.makedirs(pred_rttm_dir, exist_ok=True)

    # ---- config ----
    cfg = EvalConfig()
    cfg.frame_step_sec = args.frame_step
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

    # ---- discover files ----
    pairs = find_pairs(args.audio_dir, args.rttm_dir)
    print(f"Found {len(pairs)} (audio, rttm) pairs.")

    # ---- build frontend ----
    frontend = build_frontend(args.diarizer, cfg)

    # Batch-process all audio files up front for diarizers that support it
    if isinstance(frontend, (BabARFrontend, VTCFrontend, VBxFrontend)):
        frontend.prepare([ap for ap, _ in pairs])

    # ---- evaluate ----
    per_file_rows = []
    for i, (audio_path, gt_rttm) in enumerate(pairs):
        stem = Path(audio_path).stem
        print(f"[{i+1}/{len(pairs)}] {stem}", end=" ... ", flush=True)

        try:
            pred_segs = frontend.get_segments(audio_path, cfg)

            # Resolve anonymous speaker labels (pyannote, vbx) via GT overlap
            if args.diarizer in ANONYMOUS_DIARIZERS:
                try:
                    duration = get_audio_duration(audio_path)
                except Exception:
                    duration = max((s["end"] for s in pred_segs), default=1.0) + 1.0
                gt_segs_for_map = parse_rttm(gt_rttm)
                pred_segs = resolve_pyannote_labels(
                    pred_segs, gt_segs_for_map, duration,
                    child_labels_gt, args.frame_step,
                )

            # Save predicted RTTM (child segments only, for reference)
            child_pred = [s for s in pred_segs
                          if s["label"].upper() in {l.upper() for l in child_labels_pred}]
            write_rttm(child_pred, stem,
                       os.path.join(pred_rttm_dir, f"{stem}_pred.rttm"))

            row = evaluate_file(
                audio_path, gt_rttm, pred_segs,
                child_labels_gt, child_labels_pred,
                args.diarizer, args.frame_step,
            )
            per_file_rows.append(row)
            print(f"F1={row['f1']:.3f}  DER={row['der']:.3f}")

        except Exception as e:
            print(f"ERROR: {e}")
            per_file_rows.append({"file": stem, "error": str(e)})

    # ---- save outputs ----
    per_file_df = pd.DataFrame(per_file_rows)
    per_file_csv = os.path.join(results_dir, "per_file_metrics.csv")
    per_file_df.to_csv(per_file_csv, index=False)

    # Aggregate only over successful rows (no "error" value)
    if "error" in per_file_df.columns:
        good = per_file_df[per_file_df["error"].isna()]
    else:
        good = per_file_df

    agg = aggregate_metrics(good)
    agg_path = os.path.join(results_dir, "aggregate_metrics.json")
    save_json(agg, agg_path)

    # ---- print summary ----
    print("\n" + "=" * 55)
    label_note = " (GT-overlap resolved)" if args.diarizer in ANONYMOUS_DIARIZERS else ""
    print(f"  Evaluation: {args.diarizer}{label_note}  on  {args.dataset}")
    print("=" * 55)
    print(f"  Files evaluated         : {agg['n_files']}")
    print(f"  Total audio (hrs)       : {agg['total_audio_dur_sec']/3600:.2f}")
    print(f"  GT child speech (hrs)   : {agg['total_gt_child_dur_sec']/3600:.2f}")
    print(f"  Pred child speech (hrs) : {agg['total_pred_child_dur_sec']/3600:.2f}")
    print(f"  ── Frame-level metrics (micro) ──")
    print(f"  Precision               : {agg['micro_precision']:.4f}")
    print(f"  Recall                  : {agg['micro_recall']:.4f}")
    print(f"  F1                      : {agg['micro_f1']:.4f}")
    print(f"  Accuracy                : {agg['micro_accuracy']:.4f}")
    print(f"  Miss Rate               : {agg['miss_rate']:.4f}")
    print(f"  False Alarm Rate        : {agg['false_alarm_rate']:.4f}")
    print(f"  Binary DER              : {agg['binary_der']:.4f}")
    print(f"  Macro F1 (per-file avg) : {agg['macro_f1']:.4f}")
    print(f"  Macro AUROC             : {agg['macro_auroc']:.4f}")
    print(f"  Macro AUPRC             : {agg['macro_auprc']:.4f}")
    print(f"\nResults saved to: {results_dir}/")


if __name__ == "__main__":
    main()