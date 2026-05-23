"""Eval-only wrapper to harvest wavlm_mean's mid-training best_model.pt
(epoch 12, val F1=0.8271) so we don't lose progress when killing the
sequential encoder job for parallelization."""
import os
import sys
from pathlib import Path

import torch
from dataclasses import replace

REPO = Path("/orcd/scratch/orcd/008/manaal/child-adult-diarization")
sys.path.insert(0, str(REPO))
os.chdir(REPO)

# Match the same imports / Config the main script uses for wavlm_mean
from encoders.baseline_encoders import (  # noqa: E402
    Config,
    build_model_and_loaders,
    collect_predictions,
    tune_threshold_for_f1,
    compute_metrics,
    per_timepoint_metrics,
    load_or_create_split,
    save_json,
)
import torch.nn as nn  # noqa: E402

EXP = "wavlm_mean"
RESULTS_ROOT = REPO / "baselines/baseline_results_cross_child_bids"
CKPT = RESULTS_ROOT / EXP / "best_model.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Mirror the wavlm_mean line from baseline_encoders.py main() (line ~1223-1224)
CFG = Config()
CFG.results_root = str(RESULTS_ROOT)
CFG.seen_child_splits = False
cfg = replace(
    CFG,
    experiment_name=EXP,
    model_type="wavlm",
    pooling="mean",
    batch_size=2,
    num_workers=2,
    save_path=str(CKPT),
)

print(f"Loading splits...")
train_df, val_df, test_df = load_or_create_split(cfg)
print(f"  train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")

print(f"Building model + loaders...")
model, train_loader, val_loader, test_loader = build_model_and_loaders(cfg, train_df, val_df, test_df)
print(f"Loading checkpoint {CKPT}...")
state = torch.load(str(CKPT), map_location=DEVICE)
# Saved-best wrapper has {model_state_dict, config, best_val_metrics_at_05, epoch}
if isinstance(state, dict) and "model_state_dict" in state:
    print(f"  checkpoint is wrapped; epoch={state.get('epoch','?')} best_val_metrics={state.get('best_val_metrics_at_05','?')}")
    state = state["model_state_dict"]
model.load_state_dict(state)
model.to(DEVICE).eval()

criterion = nn.BCEWithLogitsLoss()

print(f"Collecting val predictions...")
val_pred_df, val_loss = collect_predictions(model, val_loader, criterion, DEVICE, cfg.model_type)
print(f"  val n={len(val_pred_df)}, loss={val_loss:.4f}")
print(f"BA-tuning threshold on val...")
best_t, val_metrics_at_t = tune_threshold_for_f1(val_pred_df)
print(f"  best_threshold={best_t:.4f}  val_BA={val_metrics_at_t['balanced_accuracy']:.4f}  val_F1={val_metrics_at_t['f1']:.4f}")

print(f"Collecting test predictions...")
test_pred_df, test_loss = collect_predictions(model, test_loader, criterion, DEVICE, cfg.model_type)
print(f"  test n={len(test_pred_df)}, loss={test_loss:.4f}")

import numpy as np
test_metrics = compute_metrics(test_pred_df["label"].to_numpy(),
                               test_pred_df["prob"].to_numpy(),
                               threshold=best_t)
test_metrics["threshold"] = best_t
test_metrics["loss"] = test_loss
test_metrics["n"] = int(len(test_pred_df))
test_metrics["note"] = "BIDS cross-child eval-only on epoch-12 best ckpt (training cut short before patience-stop; epoch 12 was the saved best by val F1)"
print(f"Test metrics: AUROC={test_metrics['auroc']:.4f}  BA={test_metrics['balanced_accuracy']:.4f}  F1={test_metrics['f1']:.4f}  thr={best_t:.4f}")

exp_dir = RESULTS_ROOT / EXP
exp_dir.mkdir(parents=True, exist_ok=True)
save_json(test_metrics, str(exp_dir / "test_metrics_tuned.json"))

val_metrics_at_t["threshold"] = best_t
val_metrics_at_t["loss"] = val_loss
val_metrics_at_t["n"] = int(len(val_pred_df))
save_json(val_metrics_at_t, str(exp_dir / "val_metrics_tuned.json"))

# Add pred_label column and save predictions
val_pred_df["pred_label"] = (val_pred_df["prob"] >= best_t).astype(int)
test_pred_df["pred_label"] = (test_pred_df["prob"] >= best_t).astype(int)
val_pred_df.to_csv(exp_dir / "val_predictions.csv", index=False)
test_pred_df.to_csv(exp_dir / "test_predictions.csv", index=False)

# Per-timepoint metrics
tp_test = per_timepoint_metrics(test_pred_df, best_t)
tp_test.to_csv(exp_dir / "test_metrics_by_timepoint.csv", index=False)
print(f"Saved everything to {exp_dir}")
