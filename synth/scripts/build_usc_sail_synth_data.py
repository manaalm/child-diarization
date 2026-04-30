"""Build a USC-SAIL training data layout from synth scenes.

Output structure (matches whisper-modeling/configs/config.yaml schema):
  synth_results/usc_sail_data/
    audios/{train,val}/<scene_id>.wav      (symlinks)
    labels/{train,val}/<scene_id>.csv      (frame label CSVs)

CSV format (matches preprocess_one_file 3-col branch): label,start,end
Speaker label mapping (with overlap detection):
  TARGET_CHILD only       → c (child)
  ADULT_0 only            → a (adult)
  TARGET_CHILD + ADULT_0  → o (overlap)
  none                    → si (silence)

Train/val split: 90/10, seeded.
"""
import argparse
import csv
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SYNTH_MANIFEST = os.path.join(REPO, "synth_results/manifests/synthetic_manifest.csv")
OUT_BASE = os.path.join(REPO, "synth_results/usc_sail_data")

# Fine-grained time grid for overlap detection — 5 ms is well below the 20 ms
# frame stride used by USC-SAIL preprocess.py, so no information loss.
GRID_STEP = 0.005
SCENE_DURATION = 30.0


def parse_rttm(rttm_path):
    segs = []
    with open(rttm_path) as fh:
        for line in fh:
            parts = line.strip().split()
            if len(parts) < 8 or parts[0] != "SPEAKER":
                continue
            start = float(parts[3])
            duration = float(parts[4])
            label = parts[7]
            segs.append((start, start + duration, label))
    return segs


def rttm_to_segments(rttm_path):
    """Return list of {start, end, label} dicts in CSV order, with overlap detection."""
    segs = parse_rttm(rttm_path)
    n_frames = int(round(SCENE_DURATION / GRID_STEP))
    child = np.zeros(n_frames, dtype=bool)
    adult = np.zeros(n_frames, dtype=bool)
    for start, end, label in segs:
        s = max(0, int(round(start / GRID_STEP)))
        e = min(n_frames, int(round(end / GRID_STEP)))
        if "CHILD" in label:
            child[s:e] = True
        elif "ADULT" in label:
            adult[s:e] = True

    # Per-frame combined label: 0=si, 1=c, 2=a, 3=o
    code = np.zeros(n_frames, dtype=np.int8)
    code[child & ~adult] = 1
    code[~child & adult] = 2
    code[child & adult] = 3
    label_names = {0: "si", 1: "c", 2: "a", 3: "o"}

    # Compress runs into segments
    segments = []
    i = 0
    while i < n_frames:
        j = i
        while j < n_frames and code[j] == code[i]:
            j += 1
        if code[i] != 0:  # silence is implicit (preprocess.py auto-fills gaps)
            segments.append({
                "label": label_names[int(code[i])],
                "start": round(i * GRID_STEP, 3),
                "end": round(j * GRID_STEP, 3),
            })
        i = j
    return segments


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=SYNTH_MANIFEST)
    ap.add_argument("--output", default=OUT_BASE)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val-fraction", type=float, default=0.1)
    args = ap.parse_args()

    df = pd.read_csv(args.manifest)
    rng = random.Random(args.seed)
    indices = list(range(len(df)))
    rng.shuffle(indices)
    n_val = int(round(len(df) * args.val_fraction))
    val_idx = set(indices[:n_val])

    audio_dir = os.path.join(args.output, "audios")
    label_dir = os.path.join(args.output, "labels")
    for split in ["train", "val"]:
        os.makedirs(os.path.join(audio_dir, split), exist_ok=True)
        os.makedirs(os.path.join(label_dir, split), exist_ok=True)

    n_train = n_val_real = 0
    for i, row in enumerate(df.itertuples(index=False)):
        split = "val" if i in val_idx else "train"
        scene_id = row.synthetic_scene_id
        audio_link = os.path.join(audio_dir, split, f"{scene_id}.wav")
        if not os.path.exists(audio_link):
            os.symlink(os.path.abspath(row.audio_path), audio_link)

        csv_path = os.path.join(label_dir, split, f"{scene_id}.csv")
        if int(row.target_child_vocalized) == 0 or not os.path.exists(row.rttm_path):
            # Negative scenes: write only ADULT_0 spans as 'a' for richer supervision
            segs = rttm_to_segments(row.rttm_path) if os.path.exists(row.rttm_path) else []
        else:
            segs = rttm_to_segments(row.rttm_path)

        with open(csv_path, "w", newline="") as fh:
            w = csv.writer(fh)
            for s in segs:
                w.writerow([s["label"], s["start"], s["end"]])

        if split == "train":
            n_train += 1
        else:
            n_val_real += 1

    print(f"Train: {n_train}  Val: {n_val_real}  Total: {len(df)}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
