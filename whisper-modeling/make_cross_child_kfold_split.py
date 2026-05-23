"""Generate cross-child k-fold splits.

Partitions all 139 children into K disjoint cohorts; for fold k, cohort k is
the test set. Within the remaining K-1 cohorts, 80% of children → train,
20% → val (further partitioned by child).

Output:
  baselines/splits_kfold/fold_{0..K-1}/{train,val,test}.csv
"""

from __future__ import annotations

import argparse
import os
import json

import numpy as np
import pandas as pd

REPO = "/orcd/scratch/orcd/008/manaal/child-adult-diarization"
MASTER = os.path.join(REPO, "baselines/splits/master_with_split.csv")
OUT_BASE = os.path.join(REPO, "baselines/splits_kfold")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = pd.read_csv(MASTER, low_memory=False)
    children = sorted(df["child_id"].unique())
    print(f"Total children: {len(children)}")

    rng = np.random.default_rng(args.seed)
    perm = list(children)
    rng.shuffle(perm)

    # Partition children into K folds
    folds = np.array_split(perm, args.k)
    folds = [list(f) for f in folds]

    summary = {"k": args.k, "seed": args.seed, "n_children": len(children),
               "fold_sizes": [len(f) for f in folds]}

    for k in range(args.k):
        test_children = set(folds[k])
        non_test = [c for j in range(args.k) if j != k for c in folds[j]]
        rng2 = np.random.default_rng(args.seed + k * 7919)
        rng2.shuffle(non_test)
        n_val = max(1, int(0.2 * len(non_test)))
        val_children = set(non_test[:n_val])
        train_children = set(non_test[n_val:])

        out_dir = os.path.join(OUT_BASE, f"fold_{k}")
        os.makedirs(out_dir, exist_ok=True)

        train_df = df[df["child_id"].isin(train_children)].copy()
        val_df = df[df["child_id"].isin(val_children)].copy()
        test_df = df[df["child_id"].isin(test_children)].copy()

        # set split column to be consistent
        train_df["split"] = "train"
        val_df["split"] = "val"
        test_df["split"] = "test"

        train_df.to_csv(os.path.join(out_dir, "train.csv"), index=False)
        val_df.to_csv(os.path.join(out_dir, "val.csv"), index=False)
        test_df.to_csv(os.path.join(out_dir, "test.csv"), index=False)

        master_out = pd.concat([train_df, val_df, test_df]).reset_index(drop=True)
        master_out.to_csv(os.path.join(out_dir, "master_with_split.csv"), index=False)

        fold_summary = {
            "fold": k,
            "n_train_children": len(train_children),
            "n_val_children": len(val_children),
            "n_test_children": len(test_children),
            "n_train_clips": len(train_df),
            "n_val_clips": len(val_df),
            "n_test_clips": len(test_df),
        }
        with open(os.path.join(out_dir, "split_summary.json"), "w") as f:
            json.dump(fold_summary, f, indent=2)
        summary[f"fold_{k}"] = fold_summary
        print(f"  fold {k}: {len(train_children)} train + {len(val_children)} val + {len(test_children)} test children "
              f"({len(train_df)} + {len(val_df)} + {len(test_df)} clips)")

    with open(os.path.join(OUT_BASE, "kfold_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote splits to {OUT_BASE}")


if __name__ == "__main__":
    main()
