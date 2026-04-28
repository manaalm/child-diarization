"""Build augmented seen-child splits that include TinyVox Providence clips as extra positives.

Creates whisper-modeling/seen_child_splits_tinyvox/ with:
  train.csv = original train.csv + phon_Eng-NA_Providence_* TinyVox clips (label=1)
  val.csv   = original val.csv (unchanged — TinyVox is training-only)
  test.csv  = original test.csv (unchanged)

TinyVox clips are short (1-8s) and will be padded to 10s by MILBagDataset when
pad_to_sec=10.0 is set in the config, so each clip produces the same number of
windows as a real 10s training clip.

Usage:
    python mil/build_tinyvox_splits.py
    # or with explicit paths:
    python mil/build_tinyvox_splits.py \
        --tinyvox-dir data/tinyvox/audio \
        --split-dir   whisper-modeling/seen_child_splits \
        --output-dir  whisper-modeling/seen_child_splits_tinyvox
"""

import argparse
import glob
import os
import re
import shutil
import sys

import pandas as pd

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def build(tinyvox_dir: str, split_dir: str, output_dir: str) -> None:
    tinyvox_dir = os.path.join(_REPO, tinyvox_dir)
    split_dir   = os.path.join(_REPO, split_dir)
    output_dir  = os.path.join(_REPO, output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # ── Copy val and test unchanged ────────────────────────────────────────
    for split in ("val", "test"):
        src = os.path.join(split_dir, f"{split}.csv")
        dst = os.path.join(output_dir, f"{split}.csv")
        shutil.copy2(src, dst)
        print(f"Copied {split}.csv ({len(pd.read_csv(dst))} rows)")

    # ── Load original train ────────────────────────────────────────────────
    train_csv = os.path.join(split_dir, "train.csv")
    train_df  = pd.read_csv(train_csv)
    print(f"Original train: {len(train_df)} rows  "
          f"({train_df['label'].sum():.0f} positive, "
          f"{(1 - train_df['label']).sum():.0f} negative)")

    # ── Discover TinyVox Providence clips ─────────────────────────────────
    pattern = os.path.join(tinyvox_dir, "phon_Eng-NA_Providence_*.wav")
    wav_files = sorted(glob.glob(pattern))
    if not wav_files:
        sys.exit(f"ERROR: no files matching {pattern}")
    print(f"TinyVox Providence clips found: {len(wav_files)}")

    # Extract child name from filename: phon_Eng-NA_Providence_<Name>_<date>_...
    name_re = re.compile(r"phon_Eng-NA_Providence_([A-Za-z]+)_")
    rows = []
    for wav in wav_files:
        fname = os.path.basename(wav)
        m = name_re.match(fname)
        child_name = m.group(1) if m else "Unknown"
        rows.append({
            "audio_path":     wav,
            "child_id":       f"tinyvox_{child_name}",
            "timepoint_norm": "tinyvox",
            "label":          1,
            "audio_exists":   True,
            "split":          "train",
        })

    tinyvox_df = pd.DataFrame(rows)
    per_child  = tinyvox_df["child_id"].value_counts().to_dict()
    print("  TinyVox clips per child:", {k: v for k, v in sorted(per_child.items())})

    # ── Concatenate and save ───────────────────────────────────────────────
    aug_train = pd.concat([train_df, tinyvox_df], ignore_index=True)
    pos = int(aug_train["label"].sum())
    neg = len(aug_train) - pos
    print(f"Augmented train: {len(aug_train)} rows  ({pos} positive, {neg} negative, "
          f"pos_ratio={pos/len(aug_train):.2f})")

    out_train = os.path.join(output_dir, "train.csv")
    aug_train.to_csv(out_train, index=False)
    print(f"Saved → {out_train}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tinyvox-dir", default="data/tinyvox/audio")
    parser.add_argument("--split-dir",   default="whisper-modeling/seen_child_splits")
    parser.add_argument("--output-dir",  default="whisper-modeling/seen_child_splits_tinyvox")
    args = parser.parse_args()
    build(args.tinyvox_dir, args.split_dir, args.output_dir)


if __name__ == "__main__":
    main()
