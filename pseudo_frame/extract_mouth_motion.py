"""Extract per-clip face-motion features for spec-015 US2.

This is the practical implementation of US2's "frozen visual encoder" idea
(audio_visual.txt §63, §151, §157). The literature recommends AV-HuBERT, but
its install requires fairseq which is non-trivial on this cluster. We
substitute hand-engineered face-motion features computed directly on the
face-bbox crop using OpenCV — capturing the same temporal motion signal that
AV-HuBERT learns end-to-end (lip articulation manifests as mouth-region motion
energy).

Per face track (longest track per clip, audio_visual.txt §40 caveat applies):
For each sampled frame, extract three features from the face bbox crop:
  - face_intensity_std: spatial std of the grayscale face crop (texture/content)
  - face_motion_energy: frame-to-frame mean absolute difference (talking/turning)
  - mouth_region_motion_energy: same but on the lower-third of the face crop
                                (a hand-engineered approximation of lip ROI)

Per clip aggregation:
  - mean, std, max, p95 of each frame-level feature
  - Speaking-energy proxy: variance of face_motion_energy over time (talkers move rhythmically)
  - n_mouth_frames, track_n_frames, mouth_extraction_rate

Output: pseudo_frame/visual_features/mouth_motion.csv

Usage:
  python pseudo_frame/extract_mouth_motion.py --limit 5
  python pseudo_frame/extract_mouth_motion.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import warnings
from typing import Dict, List, Optional

warnings.filterwarnings("ignore")

import cv2
import numpy as np
import pandas as pd

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)

CACHE_DIR = os.path.join(_REPO, "av_fusion/face_track_cache")
SPLIT_CSV = os.path.join(_REPO, "whisper-modeling/seen_child_splits/master_with_split.csv")
OUT_DIR = os.path.join(_REPO, "pseudo_frame/visual_features")


def cache_key(bp: str) -> str:
    return hashlib.md5(str(bp).encode()).hexdigest()


def face_cache_path(bp: str) -> str:
    return os.path.join(CACHE_DIR, f"{cache_key(bp)}.json")


def select_longest_track(tracks: List[dict]) -> Optional[dict]:
    if not tracks:
        return None
    return max(tracks, key=lambda t: len(t.get("frames", [])))


def extract_motion_features_for_track(
    video_path: str,
    track: dict,
    sample_every_n: int = 3,
    crop_size: int = 96,
) -> Optional[Dict[str, float]]:
    """Sample frames from the track, extract face crop per frame, compute motion features."""
    frames = track.get("frames", [])
    if len(frames) < 5:
        return None

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    sampled = frames[::sample_every_n]
    feature_seq = []
    prev_face_gray = None
    prev_mouth_gray = None

    try:
        for f in sampled:
            frame_idx = int(f["frame_idx"])
            x1, y1, x2, y2 = [int(round(c)) for c in f["bbox"]]
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, img = cap.read()
            if not ok or img is None:
                continue
            H, W = img.shape[:2]
            x1 = max(0, x1); y1 = max(0, y1)
            x2 = min(W, x2); y2 = min(H, y2)
            if x2 - x1 < 16 or y2 - y1 < 16:
                continue
            crop = img[y1:y2, x1:x2]
            face_gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            face_resized = cv2.resize(face_gray, (crop_size, crop_size))

            # Lower third = approximate mouth region
            mouth_h_start = int(crop_size * 0.6)
            mouth_resized = face_resized[mouth_h_start:, :]

            face_intensity_std = float(np.std(face_resized))
            mouth_intensity_std = float(np.std(mouth_resized))
            if prev_face_gray is not None:
                face_motion = float(np.mean(np.abs(face_resized.astype(np.float32)
                                                   - prev_face_gray.astype(np.float32))))
                mouth_motion = float(np.mean(np.abs(mouth_resized.astype(np.float32)
                                                    - prev_mouth_gray.astype(np.float32))))
            else:
                face_motion = 0.0
                mouth_motion = 0.0
            prev_face_gray = face_resized
            prev_mouth_gray = mouth_resized

            feature_seq.append({
                "frame_idx": frame_idx,
                "face_intensity_std": face_intensity_std,
                "mouth_intensity_std": mouth_intensity_std,
                "face_motion_energy": face_motion,
                "mouth_region_motion_energy": mouth_motion,
            })
    finally:
        cap.release()

    if len(feature_seq) < 3:
        return None

    df = pd.DataFrame(feature_seq)
    feats = {}
    for col in ["face_intensity_std", "mouth_intensity_std",
                "face_motion_energy", "mouth_region_motion_energy"]:
        feats[f"{col}_mean"] = float(df[col].mean())
        feats[f"{col}_std"]  = float(df[col].std())
        feats[f"{col}_max"]  = float(df[col].max())
        feats[f"{col}_p95"]  = float(df[col].quantile(0.95))
    feats["n_mouth_frames"] = len(feature_seq)
    feats["track_n_frames"] = len(frames)
    feats["mouth_extraction_rate"] = len(feature_seq) / max(len(sampled), 1)
    feats["mouth_motion_variance"] = float(df["mouth_region_motion_energy"].var())
    feats["face_motion_log_max"] = float(np.log1p(df["face_motion_energy"].max()))
    return feats


def _all_zero_features() -> Dict[str, float]:
    feats = {}
    for col in ["face_intensity_std", "mouth_intensity_std",
                "face_motion_energy", "mouth_region_motion_energy"]:
        for stat in ["mean", "std", "max", "p95"]:
            feats[f"{col}_{stat}"] = 0.0
    feats.update({
        "n_mouth_frames": 0,
        "track_n_frames": 0,
        "mouth_extraction_rate": 0.0,
        "mouth_motion_variance": 0.0,
        "face_motion_log_max": 0.0,
        "mouth_extraction_failed": 1,
    })
    return feats


def process_clip(audio_path: str, bp: str) -> Dict:
    feats = {"audio_path": audio_path}
    if not bp or pd.isna(bp) or not os.path.exists(str(bp)):
        feats.update(_all_zero_features())
        return feats
    cp = face_cache_path(bp)
    if not os.path.exists(cp):
        feats.update(_all_zero_features())
        return feats
    try:
        tracks = json.load(open(cp))
    except Exception:
        feats.update(_all_zero_features())
        return feats
    track = select_longest_track(tracks)
    if track is None:
        feats.update(_all_zero_features())
        return feats
    try:
        track_feats = extract_motion_features_for_track(str(bp), track)
    except Exception as e:
        feats["error"] = str(e)[:200]
        track_feats = None
    if track_feats is None:
        feats.update(_all_zero_features())
        return feats
    feats.update(track_feats)
    feats["mouth_extraction_failed"] = 0
    return feats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--start-row", type=int, default=0,
                    help="Start row index in the seen-child master CSV (after audio_exists filter).")
    ap.add_argument("--end-row", type=int, default=None,
                    help="End row index (exclusive). Default processes all remaining rows.")
    ap.add_argument("--out", default=os.path.join(OUT_DIR, "mouth_motion.csv"))
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)

    df = pd.read_csv(SPLIT_CSV)
    df = df[df["audio_exists"] == True].reset_index(drop=True)
    if args.start_row or args.end_row is not None:
        end = args.end_row if args.end_row is not None else len(df)
        df = df.iloc[args.start_row:end].reset_index(drop=True)
        print(f"Sharding: rows [{args.start_row}, {end}) → {len(df)} clips", flush=True)
    if args.limit:
        df = df.head(args.limit)

    print(f"Extracting face/mouth-motion features for {len(df)} clips → {args.out}", flush=True)

    if os.path.exists(args.out):
        already = pd.read_csv(args.out)
        seen = set(already["audio_path"].tolist())
        df_remaining = df[~df["audio_path"].isin(seen)].reset_index(drop=True)
        print(f"  Resuming: already have {len(seen)}; processing {len(df_remaining)} new",
              flush=True)
        rows = already.to_dict(orient="records")
    else:
        df_remaining = df
        rows = []

    n_failed = 0
    for i, row in enumerate(df_remaining.itertuples(index=False)):
        feats = process_clip(row.audio_path, row.BidsProcessed)
        rows.append(feats)
        if feats.get("mouth_extraction_failed", 0):
            n_failed += 1
        if (i + 1) % 50 == 0 or (i + 1) == len(df_remaining):
            pd.DataFrame(rows).to_csv(args.out, index=False)
            print(f"  {i+1}/{len(df_remaining)}  ({n_failed} failed)", flush=True)

    final = pd.DataFrame(rows)
    final.to_csv(args.out, index=False)

    print("\n=== STATS ===")
    print(f"Total clips: {len(final)}")
    print(f"Mouth-extraction failed (or no track): {final.get('mouth_extraction_failed', pd.Series([0]*len(final))).sum()}")
    succ = final[final.get("mouth_extraction_failed", 0) == 0]
    if len(succ):
        print(f"Mean n_mouth_frames (successful): {succ['n_mouth_frames'].mean():.1f}")
        print(f"Mean mouth_motion_energy_mean: {succ['mouth_region_motion_energy_mean'].mean():.3f}")
    print(f"Wrote: {args.out}")


if __name__ == "__main__":
    main()
