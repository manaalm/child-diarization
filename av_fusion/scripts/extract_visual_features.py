"""Extract automatic visual features from BIDS video clips.

Reads a metadata CSV, samples frames from each video at --sample-fps, runs
face detection and tracking via face_utils, and writes visual_features.csv
with one row per clip.

Clips with missing or unreadable video files produce NaN feature values and
off_camera_likely_score = 1.0, rather than being dropped.

Results are cached as per-clip JSON files under --face-cache-dir so the script
is idempotent: re-running with an existing cache skips already-processed clips.

Usage:
    python av_fusion/scripts/extract_visual_features.py \\
        --metadata-csv  whisper-modeling/seen_child_splits/master_with_split.csv \\
        --output        av_fusion/av_results/run1/visual_features.csv \\
        --sample-fps    2 \\
        --detector      yunet \\
        --face-cache-dir av_fusion/face_track_cache \\
        --workers       4

Exit codes:
    0 = success
    1 = metadata CSV not found or unreadable
    2 = output directory not writable
"""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from face_utils import (
    IouCentroidTracker,
    child_candidate_score,
    compute_visual_eligibility,
    make_detector,
    visual_quality_score,
)
from utils import get_repo_root, save_json

_REPO = get_repo_root()


def _clip_id(row: pd.Series) -> str:
    """Derive a stable clip_id from the row index (uses unnamed index column or row index)."""
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


def _cache_path(cache_dir: str, clip_id: str) -> str:
    h = hashlib.md5(clip_id.encode()).hexdigest()[:8]
    return os.path.join(cache_dir, f"{clip_id}__{h}.json")


def _nan_features(clip_id: str) -> Dict[str, Any]:
    return {
        "clip_id": clip_id,
        "n_faces_detected_mean": float("nan"),
        "n_faces_detected_max": float("nan"),
        "n_face_tracks": float("nan"),
        "max_face_track_duration_sec": float("nan"),
        "max_face_track_fraction_clip": float("nan"),
        "mean_face_detection_confidence": float("nan"),
        "max_face_detection_confidence": float("nan"),
        "mean_face_box_area_fraction": float("nan"),
        "max_face_box_area_fraction": float("nan"),
        "min_face_box_area_fraction": float("nan"),
        "face_center_motion_std": float("nan"),
        "visual_quality_score": float("nan"),
        "child_visible_score": 0.0,
        "off_camera_likely_score": 1.0,
        "visual_eligibility_score": 0.0,
    }


def process_clip(
    clip_id: str,
    video_path: Optional[str],
    sample_fps: float,
    detector_name: str,
    eligibility_weights: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """Extract visual features for one clip. Returns feature dict."""
    if video_path is None:
        return _nan_features(clip_id)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        cap.release()
        return _nan_features(clip_id)

    video_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 1
    frame_area = float(frame_w * frame_h)
    clip_dur_sec = total_frames / video_fps if video_fps > 0 else 0.0
    step = max(1, int(round(video_fps / sample_fps)))

    detector = make_detector(detector_name)
    tracker = IouCentroidTracker(iou_threshold=0.3)
    sampled_frames: List[np.ndarray] = []
    det_counts: List[int] = []
    all_confidences: List[float] = []
    all_areas: List[float] = []
    frame_idx = 0
    sampled_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % step == 0:
            dets = detector.detect(frame)
            tracker.update(sampled_idx, dets)
            det_counts.append(len(dets))
            for (_, _, fw, fh, conf) in dets:
                all_confidences.append(conf)
                all_areas.append((fw * fh) / frame_area)
            sampled_frames.append(frame)
            sampled_idx += 1
        frame_idx += 1

    cap.release()

    tracks = tracker.get_tracks()
    n_sampled = max(1, sampled_idx)

    # Quality
    quality = visual_quality_score(sampled_frames[::max(1, len(sampled_frames) // 10)]) if sampled_frames else 0.0

    # Track stats
    n_face_tracks = len(tracks)
    track_lengths = [len(v) for v in tracks.values()]
    max_track_len = max(track_lengths) if track_lengths else 0
    max_track_fraction = max_track_len / n_sampled
    max_track_duration_sec = (max_track_len / sample_fps) if track_lengths else 0.0

    # Face motion
    face_center_motion_std = 0.0
    if track_lengths:
        max_tid = max(tracks, key=lambda t: len(tracks[t]))
        centers = [(x + w / 2, y + h / 2) for (_, (x, y, w, h, _)) in tracks[max_tid]]
        if len(centers) > 1:
            diffs = np.diff(np.array(centers), axis=0)
            face_center_motion_std = float(np.std(np.linalg.norm(diffs, axis=1)))

    # Child candidate
    child_visible, off_camera, _ = child_candidate_score(tracks, n_sampled, frame_area)

    # Detection confidence
    mean_conf = float(np.mean(all_confidences)) if all_confidences else 0.0
    max_conf = float(np.max(all_confidences)) if all_confidences else 0.0

    # Area stats
    mean_area = float(np.mean(all_areas)) if all_areas else 0.0
    max_area = float(np.max(all_areas)) if all_areas else 0.0
    min_area = float(np.min(all_areas)) if all_areas else 0.0

    eligibility = compute_visual_eligibility(
        child_visible=child_visible,
        track_fraction=max_track_fraction,
        quality=quality,
        detection_confidence=mean_conf,
        weights=eligibility_weights,
    )

    return {
        "clip_id": clip_id,
        "n_faces_detected_mean": float(np.mean(det_counts)) if det_counts else 0.0,
        "n_faces_detected_max": int(np.max(det_counts)) if det_counts else 0,
        "n_face_tracks": n_face_tracks,
        "max_face_track_duration_sec": max_track_duration_sec,
        "max_face_track_fraction_clip": max_track_fraction,
        "mean_face_detection_confidence": mean_conf,
        "max_face_detection_confidence": max_conf,
        "mean_face_box_area_fraction": mean_area,
        "max_face_box_area_fraction": max_area,
        "min_face_box_area_fraction": min_area,
        "face_center_motion_std": face_center_motion_std,
        "visual_quality_score": quality,
        "child_visible_score": child_visible,
        "off_camera_likely_score": off_camera,
        "visual_eligibility_score": eligibility,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract automatic visual features from BIDS video clips."
    )
    parser.add_argument("--metadata-csv", required=True,
                        help="CSV with clip metadata including BidsProcessed column")
    parser.add_argument("--output", required=True,
                        help="Output path for visual_features.csv")
    parser.add_argument("--sample-fps", type=float, default=2.0,
                        help="Frame sampling rate for face detection")
    parser.add_argument("--detector", default="yunet", choices=["yunet", "mediapipe"],
                        help="Face detector backend")
    parser.add_argument("--face-cache-dir", default=None,
                        help="Directory to cache per-clip detection results")
    parser.add_argument("--workers", type=int, default=1,
                        help="Number of parallel workers (>1 uses multiprocessing)")
    args = parser.parse_args()

    metadata_csv = args.metadata_csv if os.path.isabs(args.metadata_csv) else os.path.join(_REPO, args.metadata_csv)
    output = args.output if os.path.isabs(args.output) else os.path.join(_REPO, args.output)
    cache_dir = None
    if args.face_cache_dir:
        cache_dir = args.face_cache_dir if os.path.isabs(args.face_cache_dir) else os.path.join(_REPO, args.face_cache_dir)
        os.makedirs(cache_dir, exist_ok=True)

    if not os.path.exists(metadata_csv):
        print(f"ERROR: metadata CSV not found: {metadata_csv}", file=sys.stderr)
        sys.exit(1)

    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)

    df = pd.read_csv(metadata_csv, low_memory=False)
    print(f"Processing {len(df)} clips from {metadata_csv}", flush=True)

    rows = []
    for i, (_, row) in enumerate(df.iterrows()):
        cid = _clip_id(row)
        cached = False

        if cache_dir:
            cp = _cache_path(cache_dir, cid)
            if os.path.exists(cp):
                with open(cp) as f:
                    rows.append(json.load(f))
                cached = True

        if not cached:
            vpath = _resolve_video_path(row)
            feats = process_clip(cid, vpath, args.sample_fps, args.detector)
            rows.append(feats)
            if cache_dir:
                cp = _cache_path(cache_dir, cid)
                save_json({k: (None if (isinstance(v, float) and np.isnan(v)) else v) for k, v in feats.items()}, cp)

        if (i + 1) % 50 == 0:
            print(f"  Processed {i + 1}/{len(df)} clips", flush=True)

    out_df = pd.DataFrame(rows)
    out_df.to_csv(output, index=False)
    print(f"\nVisual features written to: {output}")
    print(f"  Total clips: {len(out_df)}")
    missing_video = int(out_df["n_face_tracks"].isna().sum())
    print(f"  Clips with no video: {missing_video}")
    print(f"  Mean face tracks per clip: {out_df['n_face_tracks'].mean():.2f}")
    print(f"  Mean eligibility score: {out_df['visual_eligibility_score'].mean():.3f}")


if __name__ == "__main__":
    main()
