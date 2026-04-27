"""Extract TalkNet-ASD active speaker features for av_fusion experiments.

Calls video/run_asd.py via subprocess in the isolated Python 3.10 video/ env,
aggregates per-clip RTTM output into ASDFeatures fields, and writes
asd_features.csv with one row per clip.

Clips with no video file or failed detection produce all-zero ASD scores
rather than being dropped.  RTTM is cached per clip.

Feature notes:
  - Score fields (max_asd_score_*, mean_asd_score_*) are derived as
    duration-fraction proxies from the RTTM (fraction of clip time classified
    as active speaker), since raw per-frame TalkNet logits are not exposed
    through the RTTM interface.
  - fraction_frames_child_active uses the "CHI" track (smallest face track
    as identified by run_asd.py).
  - fraction_frames_any_active uses all active-speaker segments.

Usage:
    python av_fusion/scripts/extract_asd_features.py \\
        --metadata-csv  whisper-modeling/seen_child_splits/master_with_split.csv \\
        --output        av_fusion/av_results/run1/asd_features.csv \\
        [--rttm-cache-dir  av_fusion/av_results/run1/asd_rttm_cache] \\
        [--face-cache-dir  av_fusion/face_track_cache] \\
        [--workers         1]

Exit codes:
    0 = success
    1 = required checkpoint not found (video/pretrain/talknet_asd.model)
    2 = metadata CSV not found
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import get_repo_root, save_json

_REPO = get_repo_root()
_VIDEO_PYTHON = os.path.join(_REPO, "video", ".venv", "bin", "python")
_RUN_ASD_SCRIPT = os.path.join(_REPO, "video", "run_asd.py")
_TALKNET_CHECKPOINT = os.path.join(_REPO, "video", "pretrain", "talknet_asd.model")
_LOCONET_DEFAULT_DIR = os.path.join(_REPO, "video", "LoCoNet_ASD")
_LIGHT_ASD_DEFAULT_DIR = os.path.join(_REPO, "video", "Light-ASD")


def _clip_id(row: pd.Series) -> str:
    if "clip_id" in row.index:
        return str(row["clip_id"])
    if "Unnamed: 0" in row.index:
        return str(int(row["Unnamed: 0"]))
    return str(row.name)


def _resolve_video_path(row: pd.Series) -> Optional[str]:
    for col in ("BidsProcessed", "BidsRaw", "video_path"):
        if col in row.index:
            val = row[col]
            if pd.notna(val) and str(val).strip() and os.path.exists(str(val)):
                return str(val)
    return None


def _resolve_audio_path(row: pd.Series) -> Optional[str]:
    for col in ("audio_path", "AudioRaw", "BidsProcessed"):
        if col in row.index:
            val = row[col]
            if pd.notna(val) and str(val).strip():
                path = str(val)
                if col == "BidsProcessed":
                    path = path.replace("_desc-processed_beh.mp4", "_audio.wav")
                if os.path.exists(path):
                    return path
    return None


def _rttm_cache_path(cache_dir: str, clip_id: str) -> str:
    h = hashlib.md5(clip_id.encode()).hexdigest()[:8]
    return os.path.join(cache_dir, f"{clip_id}__{h}.rttm")


def _clip_duration_sec(audio_path: Optional[str]) -> float:
    if audio_path and os.path.exists(audio_path):
        try:
            info = sf.info(audio_path)
            return info.duration
        except Exception:
            pass
    return 0.0


def _parse_rttm(rttm_path: str) -> Tuple[float, float, int]:
    """Parse RTTM → (chi_duration_sec, any_speaker_duration_sec, n_tracks).

    Returns (0.0, 0.0, 0) if RTTM does not exist or is empty.
    """
    if not os.path.exists(rttm_path):
        return 0.0, 0.0, 0

    chi_dur = 0.0
    any_dur = 0.0
    track_labels: set = set()

    with open(rttm_path) as f:
        for line in f:
            line = line.strip()
            if not line or not line.startswith("SPEAKER"):
                continue
            parts = line.split()
            if len(parts) < 8:
                continue
            try:
                dur = float(parts[4])
                label = parts[7]
            except (ValueError, IndexError):
                continue
            any_dur += dur
            track_labels.add(label)
            if label == "CHI":
                chi_dur += dur

    return chi_dur, any_dur, len(track_labels)


def _zero_features(clip_id: str, asd_model: str = "talknet") -> Dict[str, Any]:
    return {
        "clip_id": clip_id,
        "asd_model": asd_model,
        "max_asd_score_any_face": 0.0,
        "mean_asd_score_any_face": 0.0,
        "max_asd_score_target_candidate": 0.0,
        "mean_asd_score_target_candidate": 0.0,
        "fraction_frames_active_speaker": 0.0,
        "n_active_speaker_tracks": 0,
        "asd_confidence_summary": 0.0,
        # Legacy aliases
        "max_asd_score_smallest_face": 0.0,
        "mean_asd_score_smallest_face": 0.0,
        "fraction_frames_any_active": 0.0,
        "fraction_frames_child_active": 0.0,
    }


def _asd_feature_row(
    clip_id: str,
    asd_model: str,
    chi_dur: float,
    any_dur: float,
    clip_dur: float,
    n_tracks: int,
) -> Dict[str, Any]:
    """Build ASDFeatureRow from parsed RTTM durations."""
    frac_any = min(1.0, any_dur / clip_dur) if clip_dur > 0 else 0.0
    frac_chi = min(1.0, chi_dur / clip_dur) if clip_dur > 0 else 0.0
    return {
        "clip_id": clip_id,
        "asd_model": asd_model,
        "max_asd_score_any_face": frac_any,
        "mean_asd_score_any_face": frac_any,
        "max_asd_score_target_candidate": frac_chi,
        "mean_asd_score_target_candidate": frac_chi,
        "fraction_frames_active_speaker": frac_any,
        "n_active_speaker_tracks": n_tracks,
        "asd_confidence_summary": frac_any,
        # Legacy column aliases for backward compat with 006 consumers
        "max_asd_score_smallest_face": frac_chi,
        "mean_asd_score_smallest_face": frac_chi,
        "fraction_frames_any_active": frac_any,
        "fraction_frames_child_active": frac_chi,
    }


def process_clip(
    clip_id: str,
    audio_path: Optional[str],
    video_path: Optional[str],
    rttm_cache_dir: str,
    face_cache_dir: str,
    asd_model: str = "talknet",
    loconet_checkpoint: Optional[str] = None,
    light_asd_checkpoint: Optional[str] = None,
) -> Dict[str, Any]:
    """Run ASD inference and aggregate per-clip features.

    Falls back to zero features when video is missing or inference fails.
    Supports asd_model ∈ {talknet, loconet, light_asd}.
    """
    if audio_path is None or video_path is None:
        return _zero_features(clip_id)

    rttm_path = _rttm_cache_path(rttm_cache_dir, clip_id)

    if not os.path.exists(rttm_path):
        if asd_model == "talknet":
            cmd = [
                _VIDEO_PYTHON, _RUN_ASD_SCRIPT,
                "--audio_path", audio_path,
                "--model", "talknet_asd",
                "--out_rttm", rttm_path,
                "--face_cache_dir", face_cache_dir,
                "--pretrain_dir", os.path.join(_REPO, "video", "pretrain"),
            ]
        elif asd_model == "loconet":
            cmd = [
                _VIDEO_PYTHON, _RUN_ASD_SCRIPT,
                "--audio_path", audio_path,
                "--model", "loconet",
                "--checkpoint", loconet_checkpoint or "",
                "--out_rttm", rttm_path,
                "--face_cache_dir", face_cache_dir,
                "--pretrain_dir", os.path.join(_REPO, "video", "pretrain"),
            ]
        elif asd_model == "light_asd":
            cmd = [
                _VIDEO_PYTHON, _RUN_ASD_SCRIPT,
                "--audio_path", audio_path,
                "--model", "light_asd",
                "--checkpoint", light_asd_checkpoint or "",
                "--out_rttm", rttm_path,
                "--face_cache_dir", face_cache_dir,
                "--pretrain_dir", os.path.join(_REPO, "video", "pretrain"),
            ]
        else:
            print(f"  WARNING: unknown asd_model '{asd_model}'; using zero features", file=sys.stderr)
            return _zero_features(clip_id)

        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            combined = (result.stdout + result.stderr).lower()
            if "video file not found" in combined or "no such file" in combined:
                return _zero_features(clip_id)
            print(
                f"  WARNING: run_asd.py failed for clip {clip_id} "
                f"(exit {result.returncode}); using zero features.",
                file=sys.stderr,
            )
            return _zero_features(clip_id)

    clip_dur = _clip_duration_sec(audio_path)
    chi_dur, any_dur, n_tracks = _parse_rttm(rttm_path)

    if clip_dur <= 0.0:
        return _zero_features(clip_id)

    return _asd_feature_row(clip_id, asd_model, chi_dur, any_dur, clip_dur, n_tracks)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract TalkNet-ASD features per clip into asd_features.csv."
    )
    parser.add_argument("--metadata-csv", required=True,
                        help="CSV with clip metadata (BidsProcessed, audio_path, etc.)")
    parser.add_argument("--output", required=True,
                        help="Output path for asd_features_{model}.csv")
    parser.add_argument("--model", default="talknet",
                        choices=["talknet", "loconet", "light_asd"],
                        help="ASD model to use (default: talknet)")
    parser.add_argument("--loconet-checkpoint", default=None,
                        help="Path to LocoNet checkpoint (.ckpt); required if --model loconet")
    parser.add_argument("--light-asd-checkpoint", default=None,
                        help="Path to Light-ASD checkpoint (.pt); required if --model light_asd")
    parser.add_argument("--rttm-cache-dir", default=None,
                        help="Directory to cache per-clip RTTM output")
    parser.add_argument("--face-cache-dir", default=None,
                        help="Directory to cache S3FD face detections")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Batch size for ASD inference (default: 16)")
    parser.add_argument("--device", default="cuda",
                        help="Device for inference (default: cuda)")
    args = parser.parse_args()

    # Validate checkpoints
    if args.model == "talknet":
        if not os.path.exists(_TALKNET_CHECKPOINT):
            print(
                f"ERROR: TalkNet checkpoint not found: {_TALKNET_CHECKPOINT}\n"
                "Download per video/SETUP.md before running ASD extraction.",
                file=sys.stderr,
            )
            sys.exit(1)
    elif args.model == "loconet":
        ckpt = args.loconet_checkpoint
        # Auto-locate pytorch_model.bin if no explicit checkpoint given
        if not ckpt or not os.path.exists(ckpt):
            repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            default_bin = os.path.join(repo_root, "video", "LoCoNet_ASD", "pytorch_model.bin")
            if os.path.exists(default_bin):
                args.loconet_checkpoint = default_bin
                ckpt = default_bin
                print(f"  Auto-located LocoNet checkpoint: {ckpt}")
            else:
                print(
                    f"ERROR: LocoNet checkpoint not found: {ckpt}\n"
                    "Download with:\n"
                    "  huggingface-cli download Superxixixi/LoCoNet_ASD --local-dir video/LoCoNet_ASD/\n"
                    "Then pass --loconet-checkpoint video/LoCoNet_ASD/pytorch_model.bin",
                    file=sys.stderr,
                )
                sys.exit(1)
    elif args.model == "light_asd":
        ckpt = args.light_asd_checkpoint
        if not ckpt or not os.path.exists(ckpt):
            print(
                f"ERROR: Light-ASD checkpoint not found: {ckpt}\n"
                "Download with:\n"
                "  git clone https://github.com/Junhua-Liao/Light-ASD video/Light-ASD\n"
                "Then pass --light-asd-checkpoint video/Light-ASD/weight/pretrain_AVA_CVPR22.pt",
                file=sys.stderr,
            )
            sys.exit(1)

    metadata_csv = args.metadata_csv if os.path.isabs(args.metadata_csv) else os.path.join(_REPO, args.metadata_csv)
    output = args.output if os.path.isabs(args.output) else os.path.join(_REPO, args.output)

    if not os.path.exists(metadata_csv):
        print(f"ERROR: metadata CSV not found: {metadata_csv}", file=sys.stderr)
        sys.exit(2)

    rttm_cache_dir = args.rttm_cache_dir if args.rttm_cache_dir else os.path.join(
        _REPO, "av_fusion", "av_results", "asd_rttm_cache"
    )
    if not os.path.isabs(rttm_cache_dir):
        rttm_cache_dir = os.path.join(_REPO, rttm_cache_dir)
    os.makedirs(rttm_cache_dir, exist_ok=True)

    face_cache_dir = args.face_cache_dir if args.face_cache_dir else os.path.join(
        _REPO, "av_fusion", "face_track_cache"
    )
    if not os.path.isabs(face_cache_dir):
        face_cache_dir = os.path.join(_REPO, face_cache_dir)
    os.makedirs(face_cache_dir, exist_ok=True)

    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)

    df = pd.read_csv(metadata_csv, low_memory=False)
    print(f"Processing {len(df)} clips from {metadata_csv}", flush=True)

    rows = []
    n_no_video = 0
    n_success = 0

    for i, (_, row) in enumerate(df.iterrows()):
        cid = _clip_id(row)
        audio_path = _resolve_audio_path(row)
        video_path = _resolve_video_path(row)

        if video_path is None:
            n_no_video += 1

        feats = process_clip(
            cid, audio_path, video_path, rttm_cache_dir, face_cache_dir,
            asd_model=args.model,
            loconet_checkpoint=getattr(args, "loconet_checkpoint", None),
            light_asd_checkpoint=getattr(args, "light_asd_checkpoint", None),
        )
        rows.append(feats)

        if feats["n_active_speaker_tracks"] > 0 or feats["fraction_frames_child_active"] > 0:
            n_success += 1

        if (i + 1) % 50 == 0:
            print(f"  Processed {i + 1}/{len(df)} clips", flush=True)

    out_df = pd.DataFrame(rows)
    out_df.to_csv(output, index=False)

    print(f"\nASD features written to: {output}")
    print(f"  Model: {args.model}")
    print(f"  Total clips: {len(out_df)}")
    print(f"  Clips with no video: {n_no_video}")
    print(f"  Clips with ASD detections: {n_success}")
    chi_col = "fraction_frames_child_active" if "fraction_frames_child_active" in out_df.columns else "fraction_frames_active_speaker"
    mean_chi = out_df[chi_col].mean()
    print(f"  Mean child active fraction: {mean_chi:.3f}")


if __name__ == "__main__":
    main()
