"""make_kfold_seen_child_split.py — k-fold within-child splits.

Generates K disjoint folds for cross-validating any system that uses the
seen-child within-child paradigm. Within each (child, timepoint) cell, the
clips are partitioned into K non-overlapping groups; for fold k, group k
becomes the test set and the remaining K−1 groups are split 75/25 into
train/val (stratified by cell).

This preserves the within-child paradigm (the same 109 children appear in
train/val/test of every fold), so training scripts that accept `--split-dir`
can simply point at the appropriate fold directory without code changes.

Output layout:
  whisper-modeling/seen_child_splits_kfold/
    fold_0/{train,val,test}.csv + master_with_split.csv + split_summary.json
    fold_1/...
    fold_2/...
    ...
    kfold_summary.json   ← top-level summary

Usage:
  python whisper-modeling/make_kfold_seen_child_split.py
  python whisper-modeling/make_kfold_seen_child_split.py --k 3 --seed 42
  python whisper-modeling/make_kfold_seen_child_split.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _THIS_DIR)
# Reuse the existing master-dataframe builder so label/timepoint/path
# normalization stays in one place.
from make_seen_child_split import Config as SCConfig, build_master_dataframe


def _split_cell_kfold(n: int, k: int, rng: np.random.RandomState) -> np.ndarray:
    """Return a length-n array of fold ids (0..k-1) covering all rows.
    Cells with fewer rows than k get fold ids assigned modulo k.
    """
    idx = np.arange(n)
    rng.shuffle(idx)
    fold_ids = np.empty(n, dtype=int)
    # Even partition: row i (in shuffled order) → fold i % k
    for pos, i in enumerate(idx):
        fold_ids[i] = pos % k
    return fold_ids


def _within_fold_train_val_split(
    df: pd.DataFrame,
    val_frac: float,
    rng: np.random.RandomState,
    group_cols: Tuple[str, str] = ("child_id", "timepoint_norm"),
) -> pd.DataFrame:
    """Split a (train+val pool) DataFrame into train/val, stratified by cell.

    Returns the same DataFrame with a 'split' column added (values: 'train'/'val').
    Cells with only one row go to train.
    """
    out_split = np.empty(len(df), dtype=object)
    out_split[:] = "train"
    for _, sub in df.groupby(list(group_cols), dropna=False):
        idxs = np.array(sub.index)
        n = len(idxs)
        if n <= 1:
            continue
        rng.shuffle(idxs)
        n_val = max(1, int(round(n * val_frac)))
        val_idxs = idxs[:n_val]
        out_split[df.index.get_indexer(val_idxs)] = "val"
    df = df.copy()
    df["split"] = out_split
    return df


def make_kfold_split(
    k: int,
    seed: int,
    out_dir: str,
    val_frac_within_train_pool: float = 0.25,
    annotations_csv: str | None = None,
) -> None:
    """Build K folds and write train/val/test CSVs per fold."""
    cfg = SCConfig()
    if annotations_csv:
        cfg.annotations_csv = annotations_csv
    df = build_master_dataframe(cfg)

    rng = np.random.RandomState(seed)
    group_cols = ["child_id", "timepoint_norm"]

    # Assign a fold id (0..k-1) to every clip, stratified within (child, timepoint)
    fold_id_col = np.empty(len(df), dtype=int)
    fold_id_col[:] = -1
    dropped_groups: List[dict] = []

    for group_key, sub in df.groupby(group_cols, dropna=False):
        if len(sub) < cfg.min_clips_per_child:
            dropped_groups.append({"group": str(group_key), "n_rows": int(len(sub))})
            continue
        sub_idx = np.array(sub.index)
        fold_ids = _split_cell_kfold(len(sub_idx), k, rng)
        fold_id_col[df.index.get_indexer(sub_idx)] = fold_ids

    df = df.assign(fold_id=fold_id_col)
    df = df[df["fold_id"] >= 0].reset_index(drop=True)

    os.makedirs(out_dir, exist_ok=True)

    fold_summaries: List[dict] = []
    for fold in range(k):
        fold_dir = os.path.join(out_dir, f"fold_{fold}")
        os.makedirs(fold_dir, exist_ok=True)

        test_df = df[df["fold_id"] == fold].copy()
        pool_df = df[df["fold_id"] != fold].copy()

        # train/val split inside the pool (use a different rng per fold so val
        # selection is independent across folds even given the same seed).
        pool_rng = np.random.RandomState(seed + 10_000 * (fold + 1))
        pool_split = _within_fold_train_val_split(
            pool_df, val_frac=val_frac_within_train_pool, rng=pool_rng,
        )
        train_df = pool_split[pool_split["split"] == "train"].copy()
        val_df = pool_split[pool_split["split"] == "val"].copy()

        # Tag the master CSV with split labels for traceability
        master = df.copy()
        master["split"] = "test"  # default
        master.loc[master["fold_id"] != fold, "split"] = pool_split["split"].values

        master.to_csv(os.path.join(fold_dir, "master_with_split.csv"), index=False)
        train_df.to_csv(os.path.join(fold_dir, "train.csv"), index=False)
        val_df.to_csv(os.path.join(fold_dir, "val.csv"), index=False)
        test_df.to_csv(os.path.join(fold_dir, "test.csv"), index=False)

        summary = {
            "fold": fold,
            "k": k,
            "seed": seed,
            "n_total": int(len(df)),
            "n_train": int(len(train_df)),
            "n_val": int(len(val_df)),
            "n_test": int(len(test_df)),
            "n_children_total": int(df["child_id"].nunique()),
            "n_children_train": int(train_df["child_id"].nunique()),
            "n_children_val": int(val_df["child_id"].nunique()),
            "n_children_test": int(test_df["child_id"].nunique()),
            "timepoints_test": test_df["timepoint_norm"].value_counts().to_dict(),
            "labels_train": train_df["label"].value_counts().to_dict(),
            "labels_val": val_df["label"].value_counts().to_dict(),
            "labels_test": test_df["label"].value_counts().to_dict(),
            "test_prevalence": float(test_df["label"].mean()),
        }
        with open(os.path.join(fold_dir, "split_summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
        fold_summaries.append(summary)

    top = {
        "k": k,
        "seed": seed,
        "n_total": int(len(df)),
        "n_children": int(df["child_id"].nunique()),
        "n_dropped_groups": len(dropped_groups),
        "dropped_groups": dropped_groups,
        "val_frac_within_train_pool": val_frac_within_train_pool,
        "fold_sizes": [
            {"fold": s["fold"],
             "n_train": s["n_train"], "n_val": s["n_val"], "n_test": s["n_test"],
             "test_prevalence": round(s["test_prevalence"], 3)}
            for s in fold_summaries
        ],
        "annotations_csv": cfg.annotations_csv,
    }
    with open(os.path.join(out_dir, "kfold_summary.json"), "w") as f:
        json.dump(top, f, indent=2)
    print(json.dumps(top, indent=2))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--k", type=int, default=3,
                    help="Number of folds (default: 3 — start small per "
                         "the spec-budget plan, escalate to 5 only if 3-fold "
                         "deltas are stable).")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--val-frac", type=float, default=0.25,
                    help="Fraction of the (train+val) pool used for val each fold "
                         "(default 0.25 → ~60/20/20 train/val/test overall, matching "
                         "the original within-child split).")
    ap.add_argument("--annotations-csv", default=None)
    ap.add_argument("--dry-run", action="store_true",
                    help="Print summary only; don't write files.")
    args = ap.parse_args()

    out_dir = args.out_dir or os.path.join(
        _THIS_DIR, f"seen_child_splits_kfold_{args.k}fold"
    )

    if args.dry_run:
        print(f"[dry-run] would write to: {out_dir}")
        print(f"[dry-run] k={args.k}, seed={args.seed}, val_frac={args.val_frac}")
        return 0

    make_kfold_split(
        k=args.k,
        seed=args.seed,
        out_dir=out_dir,
        val_frac_within_train_pool=args.val_frac,
        annotations_csv=args.annotations_csv,
    )
    print(f"\nWrote {args.k} folds to: {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
