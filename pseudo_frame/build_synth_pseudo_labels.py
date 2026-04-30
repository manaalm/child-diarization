"""Build pseudo-frame labels for synthetic scenes using ground-truth RTTMs.

Unlike the real-clip pipeline (which averages noisy VTC + USC-SAIL masks),
synth scenes have exact TARGET_CHILD frame labels. We treat those as 1.0 and
everything else as 0.0 — so synth-derived labels are the cleanest training
signal in the project.

Output:
  - synth pseudo-label .npy files written into pseudo_frame/pseudo_labels/
    keyed on md5(audio_path), same convention as real clips
  - rows appended to pseudo_frame/pseudo_labels/index.csv (sources="synth_gt",
    n_sources=2 sentinel so the train loop never down-weights them)

Usage:
  python pseudo_frame/build_synth_pseudo_labels.py
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
INDEX_PATH = os.path.join(OUT_DIR, "index.csv")
SYNTH_MANIFEST = os.path.join(_REPO, "synth_results/manifests/synthetic_manifest.csv")


def audio_id(audio_path: str) -> str:
    return hashlib.md5(audio_path.encode("utf-8")).hexdigest()


def get_duration(audio_path: str) -> float:
    info = torchaudio.info(audio_path)
    return info.num_frames / float(info.sample_rate)


def build_synth_one(audio_path: str, rttm_path: str, label: int, frame_step: float):
    duration = get_duration(audio_path)
    n_frames = max(1, int(np.ceil(duration / frame_step)))
    if label == 0:
        return np.zeros(n_frames, dtype=np.float32), n_frames
    if not os.path.exists(rttm_path):
        # Positive synth scene with no RTTM — should not happen, but fall back to 0.5.
        return np.full(n_frames, 0.5, dtype=np.float32), n_frames
    segs = parse_rttm(rttm_path)
    mask = segments_to_frame_mask(segs, duration, ["TARGET_CHILD"], frame_step).astype(np.float32)
    if len(mask) > n_frames:
        mask = mask[:n_frames]
    elif len(mask) < n_frames:
        mask = np.pad(mask, (0, n_frames - len(mask)), constant_values=0)
    return mask, n_frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame-step", type=float, default=0.02)
    ap.add_argument("--manifest", default=SYNTH_MANIFEST)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    df = pd.read_csv(args.manifest)
    if args.limit:
        df = df.head(args.limit)
    print(f"Building synth pseudo-labels for {len(df)} scenes", flush=True)

    rows = []
    for i, row in enumerate(df.itertuples(index=False)):
        audio_path = row.audio_path
        rttm_path = row.rttm_path
        label = int(row.target_child_vocalized)
        try:
            mask, n_frames = build_synth_one(audio_path, rttm_path, label, args.frame_step)
        except Exception as e:
            print(f"  ERROR on {audio_path}: {e}", flush=True)
            continue
        npy_path = os.path.join(OUT_DIR, f"{audio_id(audio_path)}.npy")
        np.save(npy_path, mask)
        rows.append({
            "audio_path": audio_path,
            "split": "train",
            "label": label,
            "n_frames": n_frames,
            "n_sources": 2,  # sentinel: ground-truth equivalent of "both sources agreed"
            "sources": "synth_gt",
            "clip_pos_rate": float(mask.mean()),
            "npy_path": npy_path,
        })
        if (i + 1) % 500 == 0 or (i + 1) == len(df):
            print(f"  {i+1}/{len(df)}", flush=True)

    new = pd.DataFrame(rows)
    if os.path.exists(INDEX_PATH):
        existing = pd.read_csv(INDEX_PATH)
        # Drop any synth rows from prior runs to avoid duplicates
        existing = existing[~existing["audio_path"].isin(new["audio_path"])]
        merged = pd.concat([existing, new], ignore_index=True)
    else:
        merged = new
    merged.to_csv(INDEX_PATH, index=False)
    print(f"Wrote {len(new)} new rows; index now has {len(merged)} total → {INDEX_PATH}")


if __name__ == "__main__":
    main()
