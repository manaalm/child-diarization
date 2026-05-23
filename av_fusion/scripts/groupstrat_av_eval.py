"""Group-stratified 3-fold AV fusion evaluation.

For each fold (0/1/2):
  1. Slice av_master_features.csv into train/val/test by the fold's child sets.
  2. Refit AV fusion (audio_only, video_only, always_fuse, gated_av) on the fold's
     val features (val is what train_av_fusion.py uses for the visual model).
  3. Evaluate on the fold's test features.
  4. Write fold-specific outputs to
     av_fusion/av_results/manual_only_groupstrat3_f<fold>/.

The audio score column is already BIDS-corrected (from BabAR's
enroll_*_predictions.csv joined into av_master_features.csv). No retraining
needed for the audio stream — the score is constant across folds.

Usage:
    python av_fusion/scripts/groupstrat_av_eval.py
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

_REPO = Path("/orcd/scratch/orcd/008/manaal/child-adult-diarization")
_MASTER = _REPO / "av_fusion" / "av_results" / "manual_only" / "av_master_features.csv"
_SPLIT_ROOT = _REPO / "whisper-modeling" / "seen_child_splits_groupstrat_3fold"


def _build_fold_feature_dir(fold: int, out_dir: Path) -> None:
    """Slice av_master_features.csv by the fold's child sets."""
    out_dir.mkdir(parents=True, exist_ok=True)
    master = pd.read_csv(_MASTER, low_memory=False)

    fold_dir = _SPLIT_ROOT / f"fold_{fold}"
    train_kids = set(pd.read_csv(fold_dir / "train.csv")["child_id"].astype(str))
    val_kids = set(pd.read_csv(fold_dir / "val.csv")["child_id"].astype(str))
    test_kids = set(pd.read_csv(fold_dir / "test.csv")["child_id"].astype(str))

    train_df = master[master["child_id"].astype(str).isin(train_kids)].copy()
    val_df = master[master["child_id"].astype(str).isin(val_kids)].copy()
    test_df = master[master["child_id"].astype(str).isin(test_kids)].copy()

    # Set the split column for downstream consumers (some av_fusion scripts read it).
    for df, split_label in [(train_df, "train"), (val_df, "val"), (test_df, "test")]:
        df["split"] = split_label

    train_df.to_csv(out_dir / "av_train.csv", index=False)
    val_df.to_csv(out_dir / "av_val.csv", index=False)
    test_df.to_csv(out_dir / "av_test.csv", index=False)
    # Master also wanted by some downstream tools.
    pd.concat([train_df, val_df, test_df], ignore_index=True).to_csv(
        out_dir / "av_master_features.csv", index=False
    )

    summary = {
        "fold": fold,
        "n_train": int(len(train_df)),
        "n_val": int(len(val_df)),
        "n_test": int(len(test_df)),
        "train_children": len(train_kids),
        "val_children": len(val_kids),
        "test_children": len(test_kids),
    }
    with open(out_dir / "fold_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  fold {fold}: train n={summary['n_train']}, val n={summary['n_val']}, "
          f"test n={summary['n_test']}", flush=True)


def _train_and_eval(fold_dir: Path) -> dict:
    """Run train_av_fusion + evaluate_av_fusion on a fold-specific feature dir."""
    models_dir = fold_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n--- Training AV fusion on {fold_dir.name} ---", flush=True)
    subprocess.run([
        sys.executable, "av_fusion/scripts/train_av_fusion.py",
        "--feature-dir", str(fold_dir),
        "--output-dir", str(models_dir),
        "--config", "av_fusion/configs/av_fusion.yaml",
        "--seed", "42",
    ], cwd=_REPO, check=True)

    print(f"\n--- Evaluating AV fusion on {fold_dir.name} ---", flush=True)
    subprocess.run([
        sys.executable, "av_fusion/scripts/evaluate_av_fusion.py",
        "--feature-dir", str(fold_dir),
        "--model-dir", str(models_dir),
        "--output-dir", str(fold_dir),
    ], cwd=_REPO, check=True)

    # Read back test metrics
    metrics_path = fold_dir / "metrics_overall.json"
    if metrics_path.exists():
        return json.load(open(metrics_path))
    return {}


def main():
    if not _MASTER.exists():
        print(f"ERROR: master features missing at {_MASTER}", file=sys.stderr)
        return 1

    all_metrics = []
    for fold in range(3):
        out_dir = _REPO / "av_fusion" / "av_results" / f"manual_only_groupstrat3_f{fold}"
        print(f"\n==== Fold {fold} ====", flush=True)
        _build_fold_feature_dir(fold, out_dir)
        metrics = _train_and_eval(out_dir)
        if metrics:
            metrics["fold"] = fold
            all_metrics.append(metrics)

    summary_path = _REPO / "evaluation" / "groupstrat3_av_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\nWrote {summary_path}")


if __name__ == "__main__":
    sys.exit(main() or 0)
