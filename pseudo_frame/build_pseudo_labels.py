"""Build per-clip frame-level pseudo-labels from existing diarizer RTTM caches.

Sources:
  - VTC 2.0 (pyannote/vtc_rttm_cache/)        → KCHI segments (target child)
  - USC-SAIL Whisper (whisper-modeling/usc_sail_rttm_cache/) → CHI segments (any child)

Pseudo-label per frame (20 ms / 50 Hz, matching WavLM-Base+ output rate):
  - For positive clips (label=1):  mean of available source masks  ∈ {0, 0.5, 1}
  - For negative clips (label=0):  all zeros (clip-level supervision overrides)

Cache layout:
  pseudo_frame/pseudo_labels/{md5(audio_path)}.npy   (float32, shape=(T,))
  pseudo_frame/pseudo_labels/index.csv               audio_path,split,label,n_frames,
                                                     n_sources,sources,clip_pos_rate

Usage:
  python pseudo_frame/build_pseudo_labels.py
  python pseudo_frame/build_pseudo_labels.py --frame-step 0.02
"""
import argparse
import hashlib
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torchaudio

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)

from pyannote.unified_rttm import parse_rttm, segments_to_frame_mask  # noqa: E402

OUT_DIR = os.path.join(_REPO, "pseudo_frame/pseudo_labels")
SPLIT_CSV = os.path.join(_REPO, "whisper-modeling/seen_child_splits/master_with_split.csv")

# ── RTTM source registry ─────────────────────────────────────────────────────
SOURCES = [
    {
        "name": "vtc_kchi",
        "cache_dir": os.path.join(_REPO, "pyannote/vtc_rttm_cache"),
        "child_labels": ["KCHI"],
    },
    {
        "name": "usc_sail_chi",
        "cache_dir": os.path.join(_REPO, "whisper-modeling/usc_sail_rttm_cache"),
        "child_labels": ["CHI"],
    },
]


def cache_path(audio_path: str, cache_dir: str) -> str:
    stem = Path(audio_path).stem
    h = hashlib.md5(audio_path.encode("utf-8")).hexdigest()
    return os.path.join(cache_dir, f"{stem}__{h}.rttm")


def audio_id(audio_path: str) -> str:
    return hashlib.md5(audio_path.encode("utf-8")).hexdigest()


def get_duration(audio_path: str) -> float:
    info = torchaudio.info(audio_path)
    return info.num_frames / float(info.sample_rate)


def build_one(audio_path: str, label: int, frame_step: float):
    """Return (mask, n_sources, sources_used, clip_pos_rate, n_frames)."""
    duration = get_duration(audio_path)
    n_frames = max(1, int(np.ceil(duration / frame_step)))

    if label == 0:
        return np.zeros(n_frames, dtype=np.float32), 0, [], 0.0, n_frames

    masks = []
    used = []
    for src in SOURCES:
        rttm_path = cache_path(audio_path, src["cache_dir"])
        if not os.path.exists(rttm_path):
            continue
        segs = parse_rttm(rttm_path)
        if not segs:
            continue
        m = segments_to_frame_mask(segs, duration, src["child_labels"], frame_step)
        masks.append(m.astype(np.float32))
        used.append(src["name"])

    if not masks:
        # No sources for a positive clip — fall back to a flat 0.5 prior
        # (model learns: clip is positive somewhere; weak signal but better than dropping)
        mean = np.full(n_frames, 0.5, dtype=np.float32)
    else:
        # Align lengths (RTTM rounding may give ±1 frame differences)
        L = max(len(m) for m in masks)
        masks = [np.pad(m, (0, L - len(m)), constant_values=0) for m in masks]
        mean = np.mean(np.stack(masks, axis=0), axis=0).astype(np.float32)
        # Trim/pad to n_frames
        if len(mean) > n_frames:
            mean = mean[:n_frames]
        elif len(mean) < n_frames:
            mean = np.pad(mean, (0, n_frames - len(mean)), constant_values=0)

    return mean, len(used), used, float(mean.mean()), n_frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame-step", type=float, default=0.02,
                    help="Frame step in seconds (default 0.02 = 50 Hz, matches WavLM-Base+)")
    ap.add_argument("--limit", type=int, default=None,
                    help="If set, only process first N clips (smoke test)")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)

    df = pd.read_csv(SPLIT_CSV)
    df = df[df["audio_exists"] == True].reset_index(drop=True)
    if args.limit:
        df = df.head(args.limit)

    print(f"Building pseudo-labels for {len(df)} clips → {OUT_DIR}", flush=True)
    print(f"Sources: {[s['name'] for s in SOURCES]}", flush=True)
    print(f"Frame step: {args.frame_step:.4f}s ({1/args.frame_step:.1f} Hz)", flush=True)

    rows = []
    n_skipped_no_src_pos = 0
    for i, row in enumerate(df.itertuples(index=False)):
        audio_path = row.audio_path
        label = int(row.label)
        try:
            mask, n_src, used, pos_rate, n_frames = build_one(audio_path, label, args.frame_step)
        except Exception as e:
            print(f"  ERROR on {audio_path}: {e}", flush=True)
            continue

        if label == 1 and n_src == 0:
            n_skipped_no_src_pos += 1

        npy_path = os.path.join(OUT_DIR, f"{audio_id(audio_path)}.npy")
        np.save(npy_path, mask)

        rows.append({
            "audio_path": audio_path,
            "split": row.split,
            "label": label,
            "n_frames": n_frames,
            "n_sources": n_src,
            "sources": ",".join(used) if used else "",
            "clip_pos_rate": round(pos_rate, 4),
            "npy_path": npy_path,
        })

        if (i + 1) % 200 == 0 or (i + 1) == len(df):
            print(f"  {i+1}/{len(df)}", flush=True)

    idx = pd.DataFrame(rows)
    idx_path = os.path.join(OUT_DIR, "index.csv")
    idx.to_csv(idx_path, index=False)

    # Stats
    print("\n=== STATS ===")
    print(f"Total clips: {len(idx)}")
    print(f"By split:    {idx['split'].value_counts().to_dict()}")
    print(f"By label:    {idx['label'].value_counts().to_dict()}")
    print(f"Sources used distribution (positives):")
    pos = idx[idx["label"] == 1]
    print(f"  both VTC+USC-SAIL: {(pos['n_sources']==2).sum()}")
    print(f"  one source only:   {(pos['n_sources']==1).sum()}")
    print(f"  no sources (fallback 0.5): {(pos['n_sources']==0).sum()}")
    print(f"Mean clip-level pos rate (positives): {pos['clip_pos_rate'].mean():.3f}")
    print(f"Wrote index → {idx_path}")


if __name__ == "__main__":
    main()
