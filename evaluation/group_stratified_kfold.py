"""Group-stratified k-fold splitter (spec 022 US2 / FR-009).

Splits the BIDS-corrected seen-child master into k folds where:
  - children are disjoint across folds (groups=child_id)
  - per-fold positive rate is balanced (stratification target = label)
  - random seed = 42

Output layout mirrors make_kfold_seen_child_split.py so existing per-system
training harnesses can reuse the same --split-dir flag:

  whisper-modeling/seen_child_splits_groupstrat_<k>fold/
    fold_0/{train,val,test,master_with_split}.csv + split_summary.json
    fold_1/...
    ...
    kfold_summary.json

Within each fold's (train+val pool), the val partition is carved from the
training children using StratifiedGroupKFold again with k=4 (so val is ~25%
of pool size; matches the legacy within-child convention).

Usage:
  python evaluation/group_stratified_kfold.py --k 5 --seed 42
  python evaluation/group_stratified_kfold.py --split-only        # default
  python evaluation/group_stratified_kfold.py --aggregate-summary <result-dirs-glob>
"""

import argparse
import json
import os
import sys
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

REPO_ROOT = "/orcd/scratch/orcd/008/manaal/child-adult-diarization"
DEFAULT_MASTER = os.path.join(REPO_ROOT, "whisper-modeling", "seen_child_splits", "master_with_split.csv")


def _build_folds(df: pd.DataFrame, k: int, seed: int) -> np.ndarray:
    """Assign fold_id (0..k-1) to each row using StratifiedGroupKFold."""
    sgkf = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed)
    fold_ids = np.full(len(df), -1, dtype=int)
    for i, (_, test_idx) in enumerate(sgkf.split(np.zeros(len(df)), df["label"].astype(int), groups=df["child_id"])):
        fold_ids[test_idx] = i
    return fold_ids


def make_split(master_csv: str, k: int, seed: int, out_dir: str, val_frac_groups: int = 4) -> dict:
    df = pd.read_csv(master_csv)
    # Drop the legacy `split` column if present — we will rewrite it per fold
    if "split" in df.columns:
        df = df.drop(columns=["split"])
    if "fold_id" in df.columns:
        df = df.drop(columns=["fold_id"])

    fold_ids = _build_folds(df, k, seed)
    df = df.assign(fold_id=fold_ids).reset_index(drop=True)

    os.makedirs(out_dir, exist_ok=True)
    fold_summaries = []

    # Audit: positive-rate per fold should be within bootstrap noise of overall
    pos_rates = []
    for f in range(k):
        sub = df[df["fold_id"] == f]
        pos_rates.append(float(sub["label"].mean()))

    for fold in range(k):
        fold_dir = os.path.join(out_dir, f"fold_{fold}")
        os.makedirs(fold_dir, exist_ok=True)

        test_df = df[df["fold_id"] == fold].copy()
        pool_df = df[df["fold_id"] != fold].copy()

        # Sub-fold the pool to carve val (group-disjoint within pool)
        sub_seed = seed + 10_000 * (fold + 1)
        val_ids = _sub_fold_val(pool_df, n_subfolds=val_frac_groups, seed=sub_seed)
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

        # Disjointness guard
        train_kids = set(train_df["child_id"].astype(str))
        val_kids = set(val_df["child_id"].astype(str))
        test_kids = set(test_df["child_id"].astype(str))
        assert not (train_kids & test_kids), f"fold {fold}: train and test share children — bug"
        assert not (train_kids & val_kids), f"fold {fold}: train and val share children — bug"
        assert not (val_kids & test_kids), f"fold {fold}: val and test share children — bug"

        summary = {
            "fold": fold, "k": k, "seed": seed,
            "n_total": int(len(df)),
            "n_train": int(len(train_df)), "n_val": int(len(val_df)), "n_test": int(len(test_df)),
            "n_children_train": len(train_kids),
            "n_children_val": len(val_kids),
            "n_children_test": len(test_kids),
            "test_children": sorted(test_kids),
            "val_children": sorted(val_kids),
            "labels_train": train_df["label"].astype(int).value_counts().to_dict(),
            "labels_val": val_df["label"].astype(int).value_counts().to_dict(),
            "labels_test": test_df["label"].astype(int).value_counts().to_dict(),
            "test_prevalence": float(test_df["label"].mean()),
            "train_prevalence": float(train_df["label"].mean()),
        }
        with open(os.path.join(fold_dir, "split_summary.json"), "w") as fh:
            json.dump(summary, fh, indent=2)
        fold_summaries.append(summary)

    # Stratification audit
    overall_pos = float(df["label"].mean())
    pos_rate_gap = max(pos_rates) - min(pos_rates)
    audit = {
        "k": k, "seed": seed, "master": master_csv,
        "n_total": int(len(df)),
        "n_children_total": int(df["child_id"].nunique()),
        "overall_positive_rate": overall_pos,
        "pos_rate_per_fold": pos_rates,
        "pos_rate_gap_max_minus_min": pos_rate_gap,
        "fold_sizes": [{"fold": s["fold"], "n_test": s["n_test"], "n_children_test": s["n_children_test"],
                        "test_prevalence": round(s["test_prevalence"], 3)} for s in fold_summaries],
        "stratification_guard_pass": pos_rate_gap <= 0.10,
    }
    with open(os.path.join(out_dir, "kfold_summary.json"), "w") as fh:
        json.dump(audit, fh, indent=2)
    return audit


def _sub_fold_val(pool_df: pd.DataFrame, n_subfolds: int, seed: int) -> np.ndarray:
    """Group-disjoint sub-fold inside the training pool. Returns 0/non-0 ids;
    callers treat id==0 as val. Falls back to GroupKFold if stratification is
    infeasible at the requested k (very small groups)."""
    try:
        sgkf = StratifiedGroupKFold(n_splits=n_subfolds, shuffle=True, random_state=seed)
        ids = np.full(len(pool_df), -1, dtype=int)
        for i, (_, test_idx) in enumerate(sgkf.split(np.zeros(len(pool_df)), pool_df["label"].astype(int),
                                                     groups=pool_df["child_id"])):
            ids[test_idx] = i
        return ids
    except ValueError:
        # Fallback: GroupKFold (unstratified)
        from sklearn.model_selection import GroupKFold
        gkf = GroupKFold(n_splits=n_subfolds)
        ids = np.full(len(pool_df), -1, dtype=int)
        for i, (_, test_idx) in enumerate(gkf.split(np.zeros(len(pool_df)), pool_df["label"].astype(int),
                                                    groups=pool_df["child_id"])):
            ids[test_idx] = i
        return ids


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--master", default=DEFAULT_MASTER)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--val-subfolds", type=int, default=4,
                    help="Sub-fold k inside training pool for carving val (default 4 -> ~25% val).")
    args = ap.parse_args()

    out_dir = args.out_dir or os.path.join(
        REPO_ROOT, "whisper-modeling", f"seen_child_splits_groupstrat_{args.k}fold"
    )

    audit = make_split(args.master, args.k, args.seed, out_dir, val_frac_groups=args.val_subfolds)
    print(json.dumps(audit, indent=2))
    if not audit["stratification_guard_pass"]:
        print(f"\nWARNING: stratification gap {audit['pos_rate_gap_max_minus_min']:.3f} > 0.10; "
              f"consider k=3", file=sys.stderr)


if __name__ == "__main__":
    main()
