"""
Per-face-track ECAPA speaker scoring (Clarke 2025 simplification).

For each clip:
  1. Load face tracks from av_fusion/face_track_cache/<md5(BidsProcessed)>.json
  2. For each track, derive the contiguous time interval [t_start, t_end]
  3. Embed audio in that interval with ECAPA, score vs the (child, timepoint)
     prototype from mil/prototypes/babar_vtc.npz
  4. Aggregate to clip-level via max-track (also reports mean, top-2)
  5. Fallback: clips with no face track use the BabAR clip-level enrollment
     score (this is the same as the existing US3 audio_speaker_prob)

Output: pseudo_frame/visual_features/per_track_speaker_score.csv with columns:
  audio_path, n_face_tracks_used, max_track_cosine, mean_track_cosine,
  top2_mean_track_cosine, has_any_track, fallback_clip_score
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import sys
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Import unified.py from pyannote/ but make video_asd resolvable on sys.path
sys.path.insert(0, os.path.join(_REPO, "pyannote"))
sys.path.insert(0, _REPO)

from pyannote.unified import (
    ECAPAEmbedder,
    cosine_similarity,
    crop_segment,
    l2_normalize,
    load_audio_mono,
)

CACHE_DIR = os.path.join(_REPO, "av_fusion/face_track_cache")
SPLIT_CSV = os.path.join(_REPO, "whisper-modeling/seen_child_splits/master_with_split.csv")
PROTO_PATH = os.path.join(_REPO, "mil/prototypes/babar_vtc.npz")
BABAR_VAL = os.path.join(_REPO, "babar_ecapa_enrollment_runs/enroll_val_predictions.csv")
BABAR_TEST = os.path.join(_REPO, "babar_ecapa_enrollment_runs/enroll_test_predictions.csv")
OUT_DIR = os.path.join(_REPO, "pseudo_frame/visual_features")
OUT_PATH = os.path.join(OUT_DIR, "per_track_speaker_score.csv")

ECAPA_SOURCE = "speechbrain/spkrec-ecapa-voxceleb"
SAMPLE_RATE = 16000
MIN_TRACK_DUR_SEC = 0.5
TRACK_BUFFER_SEC = 0.25


def cache_key(bids_processed_path: str) -> str:
    return hashlib.md5(str(bids_processed_path).encode()).hexdigest()


def load_face_tracks(bp_path: str):
    if not bp_path or not isinstance(bp_path, str):
        return []
    cp = os.path.join(CACHE_DIR, f"{cache_key(bp_path)}.json")
    if not os.path.exists(cp):
        return []
    try:
        return json.load(open(cp))
    except Exception:
        return []


def track_intervals(tracks) -> List[Tuple[float, float]]:
    """Return [(t_start, t_end), ...] per face track."""
    out = []
    for tr in tracks:
        frames = tr.get("frames", [])
        if not frames:
            continue
        ts = sorted(float(f["timestamp"]) for f in frames)
        s, e = ts[0], ts[-1]
        # Pad each track by a small buffer; clamps applied at audio crop time
        s = max(0.0, s - TRACK_BUFFER_SEC)
        e = e + TRACK_BUFFER_SEC
        if e - s >= MIN_TRACK_DUR_SEC:
            out.append((s, e))
    return out


def load_prototypes(npz_path: str):
    arrs = np.load(npz_path)
    return {k: arrs[k].astype(np.float64) for k in arrs.files}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--splits", nargs="+", default=["val", "test"])
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)

    df = pd.read_csv(SPLIT_CSV)
    df = df[df["audio_exists"] == True].reset_index(drop=True)
    df = df[df["split"].isin(args.splits)].reset_index(drop=True)
    if args.limit:
        df = df.head(args.limit)
    print(f"Per-track scoring on {len(df)} clips (splits={args.splits}); device={args.device}", flush=True)

    print("Loading prototypes ...", flush=True)
    protos = load_prototypes(PROTO_PATH)
    # Normalize (file already stores L2-normed but be safe)
    protos = {k: l2_normalize(v) for k, v in protos.items()}
    print(f"  {len(protos)} (child, timepoint) prototypes loaded", flush=True)

    print("Loading ECAPA embedder ...", flush=True)
    embedder = ECAPAEmbedder(ECAPA_SOURCE, args.device)

    # Fallback clip-level scores (BabAR enrollment) — used when no face track is available
    fallback = {}
    for split, path in [("val", BABAR_VAL), ("test", BABAR_TEST)]:
        if os.path.exists(path):
            f = pd.read_csv(path)[["audio_path", "prob"]].rename(columns={"prob": "fallback_score"})
            for _, r in f.iterrows():
                fallback[r["audio_path"]] = float(r["fallback_score"])

    rows = []
    n_no_proto = 0
    n_no_track = 0
    n_no_audio = 0
    for i, r in enumerate(df.itertuples(index=False)):
        ap_audio = r.audio_path
        bp = r.BidsProcessed if hasattr(r, "BidsProcessed") else ""
        proto_key = f"{r.child_id}__{r.timepoint_norm}"
        proto = protos.get(proto_key)

        if proto is None:
            n_no_proto += 1
            rows.append({
                "audio_path": ap_audio,
                "n_face_tracks_used": 0,
                "max_track_cosine": 0.0,
                "mean_track_cosine": 0.0,
                "top2_mean_track_cosine": 0.0,
                "has_any_track": 0,
                "fallback_clip_score": float(fallback.get(ap_audio, 0.0)),
                "no_prototype": 1,
            })
            continue

        tracks = load_face_tracks(bp)
        intervals = track_intervals(tracks)

        if not intervals:
            n_no_track += 1
            rows.append({
                "audio_path": ap_audio,
                "n_face_tracks_used": 0,
                "max_track_cosine": 0.0,
                "mean_track_cosine": 0.0,
                "top2_mean_track_cosine": 0.0,
                "has_any_track": 0,
                "fallback_clip_score": float(fallback.get(ap_audio, 0.0)),
                "no_prototype": 0,
            })
            continue

        try:
            wav = load_audio_mono(ap_audio, SAMPLE_RATE)
        except Exception:
            n_no_audio += 1
            rows.append({
                "audio_path": ap_audio,
                "n_face_tracks_used": 0,
                "max_track_cosine": 0.0,
                "mean_track_cosine": 0.0,
                "top2_mean_track_cosine": 0.0,
                "has_any_track": 0,
                "fallback_clip_score": float(fallback.get(ap_audio, 0.0)),
                "no_prototype": 0,
            })
            continue

        cosines = []
        for (s, e) in intervals:
            seg = crop_segment(wav, SAMPLE_RATE, s, e)
            if seg.numel() < int(MIN_TRACK_DUR_SEC * SAMPLE_RATE):
                continue
            try:
                emb = embedder.embed_waveform(seg)
                cosines.append(float(cosine_similarity(emb, proto)))
            except Exception:
                continue

        if not cosines:
            rows.append({
                "audio_path": ap_audio,
                "n_face_tracks_used": 0,
                "max_track_cosine": 0.0,
                "mean_track_cosine": 0.0,
                "top2_mean_track_cosine": 0.0,
                "has_any_track": 0,
                "fallback_clip_score": float(fallback.get(ap_audio, 0.0)),
                "no_prototype": 0,
            })
            continue

        cosines.sort(reverse=True)
        top2 = cosines[:2]
        rows.append({
            "audio_path": ap_audio,
            "n_face_tracks_used": len(cosines),
            "max_track_cosine": float(cosines[0]),
            "mean_track_cosine": float(np.mean(cosines)),
            "top2_mean_track_cosine": float(np.mean(top2)),
            "has_any_track": 1,
            "fallback_clip_score": float(fallback.get(ap_audio, 0.0)),
            "no_prototype": 0,
        })

        if (i + 1) % 50 == 0 or (i + 1) == len(df):
            print(f"  {i+1}/{len(df)}  (no_proto={n_no_proto}  no_track={n_no_track}  no_audio={n_no_audio})", flush=True)

    out = pd.DataFrame(rows)
    out.to_csv(OUT_PATH, index=False)
    print(f"\nWrote: {OUT_PATH}  ({len(out)} rows)")
    print(f"  has_any_track: {out['has_any_track'].sum()} ({100*out['has_any_track'].mean():.1f}%)")
    print(f"  mean n_face_tracks_used (when present): "
          f"{out.loc[out['has_any_track']==1, 'n_face_tracks_used'].mean():.2f}")
    print(f"  mean max_track_cosine (when present): "
          f"{out.loc[out['has_any_track']==1, 'max_track_cosine'].mean():.3f}")


if __name__ == "__main__":
    main()
