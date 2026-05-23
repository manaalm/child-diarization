"""Reproduce the post-training eval tail of ``encoders/baseline_encoders.py``
from a saved ``best_model.pt`` checkpoint.

Use when a training run has overfit past its best epoch and the best checkpoint
is already saved on disk, but ``test_metrics_tuned.json`` has not yet landed
(the trainer is a single ``train -> eval`` monolith with no ``--eval-only``
flag, so cancelling mid-training would lose the eval). This script loads the
best checkpoint, rebuilds the val/test loaders from the checkpoint's saved
config, and writes the same artefacts the trainer's tail would have written:

  - val_predictions.csv, val_metrics_tuned.json, val_metrics_by_timepoint.csv
  - test_predictions.csv, test_metrics_tuned.json, test_metrics_by_timepoint.csv
  - per_timepoint_thresholds.json (when cfg.per_timepoint_threshold is True)
  - layer_weights.json (when cfg.use_layer_weights is True)

Usage:
    python encoders/eval_from_checkpoint.py \\
        --ckpt baseline_results_seen_child/fused_attn_unfreeze2/best_model.pt
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import asdict, fields
from pathlib import Path

import pandas as pd
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from encoders.baseline_encoders import (  # noqa: E402
    Config,
    add_pred_labels,
    apply_per_timepoint_thresholds,
    build_model_and_loaders,
    collect_predictions,
    compute_metrics,
    load_or_create_split,
    per_timepoint_metrics,
    save_json,
    tune_per_timepoint_thresholds,
    tune_threshold_for_f1,
    _save_layer_weights,
)
import torch.nn as nn


def _config_from_ckpt(ckpt_cfg: dict) -> Config:
    """Reconstruct a Config dataclass from the dict saved in the checkpoint,
    silently dropping any keys that no longer exist on the dataclass."""
    valid = {f.name for f in fields(Config)}
    return Config(**{k: v for k, v in ckpt_cfg.items() if k in valid})


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True, help="path to best_model.pt")
    ap.add_argument("--device", default=None,
                    help="override cfg.device (e.g. 'cuda', 'cuda:0', 'cpu')")
    args = ap.parse_args()

    ckpt_path = Path(args.ckpt).resolve()
    if not ckpt_path.exists():
        print(f"checkpoint not found: {ckpt_path}", file=sys.stderr)
        return 2

    print(f"loading {ckpt_path}")
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    cfg = _config_from_ckpt(ckpt["config"])
    if args.device is not None:
        cfg.device = args.device
    exp_dir = ckpt_path.parent
    print(f"experiment: {cfg.experiment_name}  (saved epoch={ckpt.get('epoch')})")
    print(f"best val@0.5 in ckpt: {ckpt.get('best_val_metrics_at_05')}")

    # Reload split (uses cfg.seen_child_splits + cfg.seen_child_split_dir or
    # cfg.split_dir, matching whatever the original run used).
    train_df, val_df, test_df = load_or_create_split(cfg)
    print(f"split sizes: train={len(train_df)} val={len(val_df)} test={len(test_df)}")

    # Rebuild model + loaders, load state, move to device.
    model, train_loader, val_loader, test_loader = build_model_and_loaders(
        cfg, train_df, val_df, test_df
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(cfg.device).eval()

    if cfg.use_layer_weights:
        _save_layer_weights(model, cfg, str(exp_dir))

    criterion = nn.BCEWithLogitsLoss()

    # ---- Validation pass (for threshold tuning) ----
    val_pred_df, val_loss = collect_predictions(
        model, val_loader, criterion, cfg.device, cfg.model_type
    )

    tuned_threshold, _ = tune_threshold_for_f1(val_pred_df)
    print(f"val-tuned threshold (global F1-max): {tuned_threshold:.4f}")

    tp_thresholds = None
    if cfg.per_timepoint_threshold:
        tp_thresholds = tune_per_timepoint_thresholds(val_pred_df)
        save_json(tp_thresholds, str(exp_dir / "per_timepoint_thresholds.json"))
        val_pred_df = apply_per_timepoint_thresholds(val_pred_df, tp_thresholds)
        print(f"per-timepoint thresholds: {tp_thresholds}")
    else:
        val_pred_df = add_pred_labels(val_pred_df, tuned_threshold)

    val_pred_df.to_csv(exp_dir / "val_predictions.csv", index=False)

    val_overall_metrics = compute_metrics(
        val_pred_df["label"].to_numpy(),
        val_pred_df["prob"].to_numpy(),
        threshold=tuned_threshold,
    )
    val_overall_metrics["loss"] = float(val_loss)
    val_overall_metrics["threshold"] = float(tuned_threshold)
    save_json(val_overall_metrics, str(exp_dir / "val_metrics_tuned.json"))
    per_timepoint_metrics(val_pred_df, tuned_threshold).to_csv(
        exp_dir / "val_metrics_by_timepoint.csv", index=False
    )

    # ---- Test pass ----
    test_pred_df, test_loss = collect_predictions(
        model, test_loader, criterion, cfg.device, cfg.model_type
    )

    if cfg.per_timepoint_threshold and tp_thresholds is not None:
        test_pred_df = apply_per_timepoint_thresholds(test_pred_df, tp_thresholds)
    else:
        test_pred_df = add_pred_labels(test_pred_df, tuned_threshold)
    test_pred_df.to_csv(exp_dir / "test_predictions.csv", index=False)

    test_overall_metrics = compute_metrics(
        test_pred_df["label"].to_numpy(),
        test_pred_df["prob"].to_numpy(),
        threshold=tuned_threshold,
    )
    test_overall_metrics["loss"] = float(test_loss)
    test_overall_metrics["threshold"] = float(tuned_threshold)
    save_json(test_overall_metrics, str(exp_dir / "test_metrics_tuned.json"))
    per_timepoint_metrics(test_pred_df, tuned_threshold).to_csv(
        exp_dir / "test_metrics_by_timepoint.csv", index=False
    )

    print(f"wrote artefacts under {exp_dir}")
    print(f"  test F1={test_overall_metrics['f1']:.4f}  "
          f"AUROC={test_overall_metrics['auroc']:.4f}  "
          f"AUPRC={test_overall_metrics['auprc']:.4f}  "
          f"threshold={tuned_threshold:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
