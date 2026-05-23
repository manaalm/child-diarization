"""Leave-One-Out (Child) Cross-Validation splitter (spec-022 follow-up).

Holds out one child at a time as the test set; remaining N-1 children are
split into train + val via StratifiedGroupKFold (group-disjoint within the
training pool). With N=130 children in the BIDS-corrected seen-child master,
this produces 130 folds.

Single-child test sets are too small for stable per-fold metrics — the
intended consumption is the *pooled* estimate via aggregate_loocv.py
(concatenate all per-fold test_predictions.csv into a single test set,
compute one metrics dict at n=3145).

Output layout (mirrors group_stratified_kfold.py so existing training
harnesses reuse the --split-dir flag):

  whisper-modeling/seen_child_splits_loocv/
    fold_0/{train,val,test,master_with_split}.csv + split_summary.json
    fold_1/...
    ...
    fold_129/...
    loocv_summary.json

Usage:
  python evaluation/loocv_split.py --seed 42
"""
import argparse
import json
import os
import sys
from typing import List

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, GroupKFold

REPO_ROOT = "/orcd/scratch/orcd/008/manaal/child-adult-diarization"
DEFAULT_MASTER = os.path.join(REPO_ROOT, "whisper-modeling", "seen_child_splits", "master_with_split.csv")


def _carve_val(pool_df: pd.DataFrame, seed: int, n_subfolds: int = 4) -> np.ndarray:
    """Group-disjoint val carve inside the training pool. Returns 0/non-0 ids;
    callers treat id==0 as val (~25% of pool when n_subfolds=4)."""
    try:
        sgkf = StratifiedGroupKFold(n_splits=n_subfolds, shuffle=True, random_state=seed)
        ids = np.full(len(pool_df), -1, dtype=int)
        for i, (_, test_idx) in enumerate(sgkf.split(np.zeros(len(pool_df)),
                                                     pool_df["label"].astype(int),
                                                     groups=pool_df["child_id"])):
            ids[test_idx] = i
        return ids
    except ValueError:
        gkf = GroupKFold(n_splits=n_subfolds)
        ids = np.full(len(pool_df), -1, dtype=int)
        for i, (_, test_idx) in enumerate(gkf.split(np.zeros(len(pool_df)),
                                                    pool_df["label"].astype(int),
                                                    groups=pool_df["child_id"])):
            ids[test_idx] = i
        return ids


def make_loocv_split(master_csv: str, seed: int, out_dir: str, val_subfolds: int = 4) -> dict:
    df = pd.read_csv(master_csv)
    if "split" in df.columns:
        df = df.drop(columns=["split"])

    children = sorted(df["child_id"].astype(str).unique())
    n_children = len(children)
    os.makedirs(out_dir, exist_ok=True)

    fold_summaries: List[dict] = []
    overall_pos = float(df["label"].mean())

    for fold, held_child in enumerate(children):
        fold_dir = os.path.join(out_dir, f"fold_{fold}")
        os.makedirs(fold_dir, exist_ok=True)

        test_df = df[df["child_id"].astype(str) == held_child].copy()
        pool_df = df[df["child_id"].astype(str) != held_child].copy()

        sub_seed = seed + (fold + 1) * 7919  # large prime; deterministic per fold
        val_ids = _carve_val(pool_df, sub_seed, n_subfolds=val_subfolds)
        pool_df = pool_df.assign(_is_val=val_ids == 0)
        train_df = pool_df[~pool_df["_is_val"]].drop(columns=["_is_val"]).copy()
        val_df = pool_df[pool_df["_is_val"]].drop(columns=["_is_val"]).copy()

        master = df.copy()
        master["split"] = "test"
        master.loc[pool_df.index, "split"] = np.where(pool_df["_is_val"].values, "val", "train")

        master.to_csv(os.path.join(fold_dir, "master_with_split.csv"), index=False)
        train_df.to_csv(os.path.join(fold_dir, "train.csv"), index=False)
        val_df.to_csv(os.path.join(fold_dir, "val.csv"), index=False)
        test_df.to_csv(os.path.join(fold_dir, "test.csv"), index=False)

        train_kids = set(train_df["child_id"].astype(str))
        val_kids = set(val_df["child_id"].astype(str))
        assert held_child not in train_kids, f"fold {fold}: held child leaked into train"
        assert held_child not in val_kids, f"fold {fold}: held child leaked into val"
        assert not (train_kids & val_kids), f"fold {fold}: train and val share children"

        summary = {
            "fold": fold,
            "seed": seed,
            "held_child": held_child,
            "n_test": int(len(test_df)),
            "n_train": int(len(train_df)),
            "n_val": int(len(val_df)),
            "n_children_train": len(train_kids),
            "n_children_val": len(val_kids),
            "test_label_distribution": test_df["label"].astype(int).value_counts().to_dict(),
            "test_prevalence": float(test_df["label"].mean()) if len(test_df) else 0.0,
        }
        with open(os.path.join(fold_dir, "split_summary.json"), "w") as fh:
            json.dump(summary, fh, indent=2)
        fold_summaries.append(summary)

    test_sizes = [s["n_test"] for s in fold_summaries]
    audit = {
        "n_folds": n_children,
        "seed": seed,
        "master": master_csv,
        "n_total": int(len(df)),
        "n_children_total": n_children,
        "overall_positive_rate": overall_pos,
        "test_size_min": int(np.min(test_sizes)),
        "test_size_max": int(np.max(test_sizes)),
        "test_size_median": float(np.median(test_sizes)),
        "test_size_mean": float(np.mean(test_sizes)),
        "n_children_with_lt_10_clips": sum(1 for s in test_sizes if s < 10),
    }
    with open(os.path.join(out_dir, "loocv_summary.json"), "w") as fh:
        json.dump(audit, fh, indent=2)
    return audit


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--master", default=DEFAULT_MASTER)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--val-subfolds", type=int, default=4)
    args = ap.parse_args()

    out_dir = args.out_dir or os.path.join(REPO_ROOT, "whisper-modeling", "seen_child_splits_loocv")
    audit = make_loocv_split(args.master, args.seed, out_dir, val_subfolds=args.val_subfolds)
    print(json.dumps(audit, indent=2))
    print(f"\n{audit['n_folds']} LOOCV folds written to {out_dir}", file=sys.stderr)


if __name__ == "__main__":
    main()
