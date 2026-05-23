"""Eval-only wrapper for fused_attn_unfreeze2_whisper_medium cross-child BIDS.
Loads best_model.pt (epoch 1 best), runs BIDS val/test, writes test_metrics_tuned.json."""
import os
import sys
from pathlib import Path
from dataclasses import replace

import torch
import torch.nn as nn

REPO = Path("/orcd/scratch/orcd/008/manaal/child-adult-diarization")
sys.path.insert(0, str(REPO))
os.chdir(REPO)

from encoders.baseline_encoders import (  # noqa: E402
    CFG,
    build_model_and_loaders,
    collect_predictions,
    tune_threshold_for_f1,
    compute_metrics,
    per_timepoint_metrics,
    load_or_create_split,
    save_json,
)

EXP = "fused_attn_unfreeze2_whisper_medium"
RESULTS_ROOT = REPO / "baselines/baseline_results_cross_child_bids"
CKPT = RESULTS_ROOT / EXP / "best_model.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

base = replace(
    CFG,
    seen_child_splits=False,
    results_root=str(RESULTS_ROOT),
    whisper_name="openai/whisper-medium",
)
cfg = replace(
    base,
    experiment_name=EXP,
    model_type="fused",
    pooling="attn",
    use_layer_weights=False,
    unfreeze_last_n_layers=2,
    batch_size=1,
    num_workers=2,
    per_timepoint_threshold=False,
    save_path=str(CKPT),
)

print(f"Loading splits (BIDS cross-child) ...")
train_df, val_df, test_df = load_or_create_split(cfg)
print(f"  train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")

print(f"Building model + loaders ({cfg.whisper_name} fused) ...")
model, train_loader, val_loader, test_loader = build_model_and_loaders(cfg, train_df, val_df, test_df)
print(f"Loading checkpoint {CKPT} ...")
state = torch.load(str(CKPT), map_location=DEVICE)
if isinstance(state, dict) and "model_state_dict" in state:
    print(f"  wrapper found; epoch={state.get('epoch','?')} best_val_metrics={state.get('best_val_metrics_at_05','?')}")
    state = state["model_state_dict"]
model.load_state_dict(state)
model.to(DEVICE).eval()
criterion = nn.BCEWithLogitsLoss()

print("Collecting val predictions ...")
val_pred_df, val_loss = collect_predictions(model, val_loader, criterion, DEVICE, cfg.model_type)
best_t, val_metrics = tune_threshold_for_f1(val_pred_df)
print(f"  thr={best_t:.4f}  val BA={val_metrics['balanced_accuracy']:.4f}  val F1={val_metrics['f1']:.4f}")

print("Collecting test predictions ...")
test_pred_df, test_loss = collect_predictions(model, test_loader, criterion, DEVICE, cfg.model_type)
test_metrics = compute_metrics(test_pred_df["label"].to_numpy(),
                               test_pred_df["prob"].to_numpy(), threshold=best_t)
test_metrics.update({"threshold": best_t, "loss": test_loss, "n": int(len(test_pred_df)),
                     "note": "BIDS cross-child eval-only on best ckpt (training cancelled past effective patience)"})
print(f"Test: AUROC={test_metrics['auroc']:.4f}  BA={test_metrics['balanced_accuracy']:.4f}  F1={test_metrics['f1']:.4f}")

exp_dir = RESULTS_ROOT / EXP
exp_dir.mkdir(parents=True, exist_ok=True)
save_json(test_metrics, str(exp_dir / "test_metrics_tuned.json"))
val_metrics.update({"threshold": best_t, "loss": val_loss, "n": int(len(val_pred_df))})
save_json(val_metrics, str(exp_dir / "val_metrics_tuned.json"))
val_pred_df["pred_label"] = (val_pred_df["prob"] >= best_t).astype(int)
test_pred_df["pred_label"] = (test_pred_df["prob"] >= best_t).astype(int)
val_pred_df.to_csv(exp_dir / "val_predictions.csv", index=False)
test_pred_df.to_csv(exp_dir / "test_predictions.csv", index=False)
tp_test = per_timepoint_metrics(test_pred_df, best_t)
tp_test.to_csv(exp_dir / "test_metrics_by_timepoint.csv", index=False)
print(f"Saved {exp_dir}")
