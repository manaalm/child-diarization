"""Compute per-clip visual-eligibility features from cached face tracks.

Used by spec-015 US1 to augment the spec-012 metadata stacker with automatic
visual-quality features (in addition to the manual BIDS metadata).

Source cache: av_fusion/face_track_cache/<md5(BidsProcessed)>.json
  - One JSON per clip; list of {track_id, frames: [{frame_idx, timestamp, bbox, score}]}.
  - 2183/2183 clips have a cache file; 1921 have ≥1 detected face.

Output features (per clip, written to pseudo_frame/visual_features/visual_eligibility.csv):
  - face_count_max:           max number of simultaneously visible faces in any frame
  - face_count_mean:          mean across frames where any face was seen
  - face_area_max_norm:       max bbox area / video frame area (uses 1280x720 fallback if unknown)
  - face_area_mean_norm:      mean of per-frame max bbox area, normalized
  - face_confidence_mean:     mean detection confidence over all detected faces
  - face_track_coverage_ratio: max(track_duration) / clip_duration  (∈ [0, 1])
  - n_distinct_tracks:        number of distinct track ids
  - has_any_face:             1 if any face was detected, else 0
  - eligibility_score:        composite — proxy for "visually eligible for AV fusion"

Usage:
  python pseudo_frame/visual_eligibility.py
  python pseudo_frame/visual_eligibility.py --limit 10  # smoke
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd
import torchaudio

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)

CACHE_DIR = os.path.join(_REPO, "av_fusion/face_track_cache")
SPLIT_CSV = os.path.join(_REPO, "whisper-modeling/seen_child_splits/master_with_split.csv")
OUT_DIR = os.path.join(_REPO, "pseudo_frame/visual_features")

# Default video frame area (BIDS preprocessed videos appear to be 1280x720)
DEFAULT_FRAME_AREA = 1280.0 * 720.0


def cache_key(bids_processed_path: str) -> str:
    return hashlib.md5(str(bids_processed_path).encode()).hexdigest()


def cache_path(bids_processed_path: str) -> str:
    return os.path.join(CACHE_DIR, f"{cache_key(bids_processed_path)}.json")


def get_audio_duration(audio_path: str) -> float:
    info = torchaudio.info(audio_path)
    return info.num_frames / float(info.sample_rate)


def per_frame_face_counts(tracks):
    """Return dict {frame_idx: n_faces} aggregated over all tracks."""
    counts = {}
    for tr in tracks:
        for f in tr.get("frames", []):
            fi = int(f["frame_idx"])
            counts[fi] = counts.get(fi, 0) + 1
    return counts


def extract_features(tracks, clip_duration: float):
    """Compute per-clip features from face tracks. Empty tracks → all zeros."""
    if not tracks:
        return {
            "face_count_max": 0,
            "face_count_mean": 0.0,
            "face_area_max_norm": 0.0,
            "face_area_mean_norm": 0.0,
            "face_confidence_mean": 0.0,
            "face_track_coverage_ratio": 0.0,
            "n_distinct_tracks": 0,
            "has_any_face": 0,
        }

    frame_counts = per_frame_face_counts(tracks)
    face_count_max = max(frame_counts.values()) if frame_counts else 0
    face_count_mean = float(np.mean(list(frame_counts.values()))) if frame_counts else 0.0

    all_areas = []
    all_confs = []
    per_frame_max_area = {}
    track_durations = []
    for tr in tracks:
        frames = tr.get("frames", [])
        if not frames:
            continue
        ts = [float(f["timestamp"]) for f in frames]
        track_durations.append(max(ts) - min(ts))
        for f in frames:
            x1, y1, x2, y2 = f["bbox"]
            area = max(0.0, (x2 - x1)) * max(0.0, (y2 - y1))
            all_areas.append(area)
            all_confs.append(float(f.get("score", 0.0)))
            fi = int(f["frame_idx"])
            per_frame_max_area[fi] = max(per_frame_max_area.get(fi, 0.0), area)

    face_area_max_norm = (max(all_areas) / DEFAULT_FRAME_AREA) if all_areas else 0.0
    face_area_mean_norm = (
        float(np.mean(list(per_frame_max_area.values()))) / DEFAULT_FRAME_AREA
        if per_frame_max_area else 0.0
    )
    face_confidence_mean = float(np.mean(all_confs)) if all_confs else 0.0
    face_track_coverage_ratio = (
        float(max(track_durations)) / max(clip_duration, 1e-6) if track_durations else 0.0
    )
    face_track_coverage_ratio = float(np.clip(face_track_coverage_ratio, 0.0, 1.0))

    return {
        "face_count_max": int(face_count_max),
        "face_count_mean": float(face_count_mean),
        "face_area_max_norm": float(face_area_max_norm),
        "face_area_mean_norm": float(face_area_mean_norm),
        "face_confidence_mean": float(face_confidence_mean),
        "face_track_coverage_ratio": float(face_track_coverage_ratio),
        "n_distinct_tracks": int(len(tracks)),
        "has_any_face": 1 if all_areas else 0,
    }


def eligibility_score(feats: dict) -> float:
    """Composite proxy for whether the clip is visually eligible for AV fusion.

    Weights chosen heuristically; the metadata stacker re-learns them via LR.
    Range [0, 1].
    """
    s = (
        0.35 * min(1.0, feats["face_confidence_mean"])
        + 0.25 * min(1.0, feats["face_track_coverage_ratio"])
        + 0.20 * min(1.0, feats["face_area_mean_norm"] * 20.0)  # face area ~5% → 1.0
        + 0.10 * min(1.0, feats["n_distinct_tracks"] / 3.0)
        + 0.10 * float(feats["has_any_face"])
    )
    return float(np.clip(s, 0.0, 1.0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)

    df = pd.read_csv(SPLIT_CSV)
    df = df[df["audio_exists"] == True].reset_index(drop=True)
    if args.limit:
        df = df.head(args.limit)

    print(f"Computing visual eligibility for {len(df)} clips → {OUT_DIR}", flush=True)

    rows = []
    n_no_cache = 0
    n_no_face = 0
    for i, r in enumerate(df.itertuples(index=False)):
        ap_audio = r.audio_path
        bp = r.BidsProcessed if hasattr(r, "BidsProcessed") else getattr(r, "bidsprocessed", "")
        cp = cache_path(bp) if bp and isinstance(bp, str) else None
        try:
            duration = get_audio_duration(ap_audio)
        except Exception:
            duration = 30.0  # fallback

        if not cp or not os.path.exists(cp):
            n_no_cache += 1
            feats = extract_features([], duration)
        else:
            try:
                tracks = json.load(open(cp))
            except Exception:
                tracks = []
            if not tracks:
                n_no_face += 1
            feats = extract_features(tracks, duration)

        feats["audio_path"] = ap_audio
        feats["clip_duration_sec"] = duration
        feats["eligibility_score"] = eligibility_score(feats)
        rows.append(feats)

        if (i + 1) % 200 == 0 or (i + 1) == len(df):
            print(f"  {i+1}/{len(df)}", flush=True)

    out = pd.DataFrame(rows)
    cols = [
        "audio_path",
        "clip_duration_sec",
        "face_count_max",
        "face_count_mean",
        "face_area_max_norm",
        "face_area_mean_norm",
        "face_confidence_mean",
        "face_track_coverage_ratio",
        "n_distinct_tracks",
        "has_any_face",
        "eligibility_score",
    ]
    out = out[cols]
    out_path = os.path.join(OUT_DIR, "visual_eligibility.csv")
    out.to_csv(out_path, index=False)

    print("\n=== STATS ===")
    print(f"Total clips: {len(out)}")
    print(f"Cache missing: {n_no_cache}")
    print(f"Cache present but no face detected: {n_no_face}")
    print(f"Has any face: {out['has_any_face'].sum()} ({100*out['has_any_face'].mean():.1f}%)")
    print(f"Mean eligibility_score: {out['eligibility_score'].mean():.3f}")
    print(f"Mean face_count_max: {out['face_count_max'].mean():.2f}")
    print(f"Mean face_track_coverage: {out['face_track_coverage_ratio'].mean():.3f}")
    print(f"\nWrote: {out_path}")


if __name__ == "__main__":
    main()
