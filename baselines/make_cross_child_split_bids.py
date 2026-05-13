"""Rebuild the cross-child split (baselines/splits/) from the BIDS-corrected
master, so cross-child training CSVs carry BIDS-derived timepoint_norm
instead of the spreadsheet's. spec 022 polish for the cross-child gap
identified after US1 (2026-05-12).

Source: builds its own master via the BIDS-aware build_master_dataframe with
RELAXED filters (require_timepoint=True, min_clips_per_child=1) to match the
legacy cross-child population (139 children — no ≥5-clip-per-cell guard).
This is broader than the seen-child master (130 children, ≥5/cell).

Writes:
  baselines/splits/master_with_split.csv  (legacy backed up to *.legacy_pre_bids_022)
  baselines/splits/train.csv / val.csv / test.csv
  baselines/splits/split_summary.json (seed=42, GroupShuffleSplit 70/15/15)

Children-disjoint split: 70 / 15 / 15 by child, seeded at 42. Same convention
as the prior cross-child split (make_reusable_group_split in encoders/
baseline_encoders.py) — just driven from the BIDS-corrected source instead.
"""

import argparse
import json
import os
import shutil
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

REPO_ROOT = "/orcd/scratch/orcd/008/manaal/child-adult-diarization"
OUT_DIR = os.path.join(REPO_ROOT, "baselines", "splits")

sys.path.insert(0, os.path.join(REPO_ROOT, "whisper-modeling"))
from make_seen_child_split import Config as SCConfig, build_master_dataframe  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train-frac", type=float, default=0.70)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--test-frac", type=float, default=0.15)
    args = ap.parse_args()

    if abs(args.train_frac + args.val_frac + args.test_frac - 1.0) > 1e-8:
        sys.exit(f"fractions must sum to 1.0; got {args.train_frac+args.val_frac+args.test_frac}")

    # Build a relaxed BIDS-corrected master (min_clips_per_child=1, require_timepoint=True)
    cfg = SCConfig()
    cfg.use_bids_timepoint = True
    cfg.require_timepoint = True
    cfg.min_clips_per_child = 1
    df = build_master_dataframe(cfg)
    print(f"built relaxed BIDS master: {len(df)} rows / {df['child_id'].nunique()} children")
    assert "bids_timepoint" in df.columns, "build_master_dataframe should expose bids_timepoint"

    # 1. Backup any existing cross-child split files
    os.makedirs(OUT_DIR, exist_ok=True)
    for fname in ("master_with_split.csv", "train.csv", "val.csv", "test.csv", "split_summary.json"):
        src = os.path.join(OUT_DIR, fname)
        if os.path.exists(src):
            backup = src + ".legacy_pre_bids_022"
            if not os.path.exists(backup):
                shutil.copyfile(src, backup)
                print(f"  backed up {fname} -> *.legacy_pre_bids_022")

    # 2. Drop the old `split` column if present — we'll rewrite per child-group
    df_work = df.copy()
    if "split" in df_work.columns:
        df_work = df_work.drop(columns=["split"])

    # 3. GroupShuffleSplit: 70% train, 30% temp; then 50/50 split of temp -> val/test
    groups = df_work["child_id"].values
    idx = np.arange(len(df_work))

    gss1 = GroupShuffleSplit(n_splits=1, train_size=args.train_frac, random_state=args.seed)
    train_idx, temp_idx = next(gss1.split(idx, groups=groups))
    train_df = df_work.iloc[train_idx].copy()
    temp_df = df_work.iloc[temp_idx].copy()

    rel_val = args.val_frac / (args.val_frac + args.test_frac)
    gss2 = GroupShuffleSplit(n_splits=1, train_size=rel_val, random_state=args.seed + 1)
    temp_groups = temp_df["child_id"].values
    val_idx2, test_idx2 = next(gss2.split(np.arange(len(temp_df)), groups=temp_groups))

    val_df = temp_df.iloc[val_idx2].copy()
    test_df = temp_df.iloc[test_idx2].copy()

    # 4. Disjointness guard
    train_kids = set(train_df["child_id"])
    val_kids = set(val_df["child_id"])
    test_kids = set(test_df["child_id"])
    assert not (train_kids & val_kids), "train/val share children"
    assert not (train_kids & test_kids), "train/test share children"
    assert not (val_kids & test_kids), "val/test share children"

    train_df["split"] = "train"
    val_df["split"] = "val"
    test_df["split"] = "test"

    full = pd.concat([train_df, val_df, test_df], axis=0).sort_index().reset_index(drop=True)

    # 5. Write
    full.to_csv(os.path.join(OUT_DIR, "master_with_split.csv"), index=False)
    train_df.to_csv(os.path.join(OUT_DIR, "train.csv"), index=False)
    val_df.to_csv(os.path.join(OUT_DIR, "val.csv"), index=False)
    test_df.to_csv(os.path.join(OUT_DIR, "test.csv"), index=False)

    summary = {
        "seed": args.seed,
        "source": "make_seen_child_split.build_master_dataframe(cfg) with use_bids_timepoint=True, min_clips_per_child=1, require_timepoint=True",
        "bids_corrected": True,
        "n_total_rows": int(len(full)),
        "n_train_rows": int(len(train_df)),
        "n_val_rows": int(len(val_df)),
        "n_test_rows": int(len(test_df)),
        "n_children_total": len(train_kids | val_kids | test_kids),
        "n_train_children": len(train_kids),
        "n_val_children": len(val_kids),
        "n_test_children": len(test_kids),
        "train_prevalence": float(train_df["label"].mean()),
        "val_prevalence": float(val_df["label"].mean()),
        "test_prevalence": float(test_df["label"].mean()),
        "timepoints_train": train_df["timepoint_norm"].value_counts().to_dict(),
        "timepoints_val": val_df["timepoint_norm"].value_counts().to_dict(),
        "timepoints_test": test_df["timepoint_norm"].value_counts().to_dict(),
        "regenerated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    with open(os.path.join(OUT_DIR, "split_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
