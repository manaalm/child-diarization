"""
video_asd.py — TalkNetASD and TSTalkNet DiarizationFrontend implementations.

Calls video/run_asd.py via subprocess in the isolated Python 3.10 video/ env.
Audio-only datasets (Providence, Playlogue) return [] gracefully when the
subprocess reports FileNotFoundError for the .mp4 file.
"""

import hashlib
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List


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


def _run_asd_subprocess(cmd: list, audio_path: str) -> bool:
    """Run run_asd.py; return True on success, False if audio-only dataset."""
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode == 0:
        return True
    combined = (result.stdout + result.stderr).lower()
    if "video file not found" in combined or (
        "filenotfounderror" in combined and "mp4" in combined
    ):
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

    def get_segments(self, audio_path: str, cfg) -> List[Dict[str, float]]:
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
    Falls back to [] if no reference clip is found or if the dataset is
    audio-only (no .mp4).
    """

    def __init__(self, cfg):
        os.makedirs(os.path.join(cfg.video_asd_rttm_cache_dir, "ts_talknet"), exist_ok=True)
        os.makedirs(cfg.video_face_cache_dir, exist_ok=True)

    def get_segments(self, audio_path: str, cfg) -> List[Dict[str, float]]:
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
