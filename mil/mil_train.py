"""MIL training entry point.

Usage:
    python mil/mil_train.py --config mil/configs/wavlm_mil.yaml
"""

import argparse
import os
import random
import sys

import numpy as np
import pandas as pd
import torch
import yaml

# Resolve repo root so imports work regardless of CWD
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)

from mil.mil_dataset import MILBagDataset
from mil.mil_model import build_mil_model
from mil.mil_utils import (
    compute_metrics,
    per_timepoint_metrics,
    save_csv,
    save_json,
    tune_threshold,
)


def parse_config(path: str) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    required = ["variant_name", "backbone", "split_dir", "seed"]
    for k in required:
        if k not in cfg:
            raise ValueError(f"Config missing required key: {k}")
    return cfg


def setup_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_split(split_dir: str, split: str) -> pd.DataFrame:
    path = os.path.join(_REPO, split_dir, f"{split}.csv")
    df = pd.read_csv(path)
    if "audio_exists" in df.columns:
        df = df[df["audio_exists"] == True]
    if "timepoint_norm" not in df.columns and "timepoint" in df.columns:
        df = df.rename(columns={"timepoint": "timepoint_norm"})
    return df.reset_index(drop=True)


def _precompute_embeddings(model, dataset, device):
    """Run all clips through the frozen backbone once; return dict[audio_path → (N_windows, D)]."""
    cache = {}
    model.backbone.eval()
    n = len(dataset)
    with torch.no_grad():
        for idx in range(n):
            item = dataset[idx]
            path = item["audio_path"]
            if path in cache:
                continue
            embs = []
            for w in item["windows"]:
                w_t = w.unsqueeze(0).to(device)       # (1, 1, T)
                frames = model.backbone(w_t)            # (1, T_frames, D)
                emb = frames.mean(dim=1).squeeze(0)    # (D,)
                embs.append(emb.cpu())
            cache[path] = torch.stack(embs, dim=0)    # (N_windows, D) on CPU
            if (idx + 1) % 100 == 0 or (idx + 1) == n:
                print(f"  {idx + 1}/{n} clips embedded", flush=True)
    return cache


def train(cfg: dict) -> None:
    setup_seed(cfg["seed"])
    device = torch.device(cfg.get("device", "cuda") if torch.cuda.is_available() else "cpu")
    variant = cfg["variant_name"]
    result_dir = os.path.join(_REPO, "mil", "mil_results", variant)
    os.makedirs(result_dir, exist_ok=True)

    # ── Data ───────────────────────────────────────────────────────────────
    train_df = load_split(cfg["split_dir"], "train")
    val_df = load_split(cfg["split_dir"], "val")

    extra_neg_csv = cfg.get("extra_negatives_csv")
    if extra_neg_csv:
        extra_path = os.path.join(_REPO, extra_neg_csv)
        extra_df = pd.read_csv(extra_path)
        # Sub-sample to at most `extra_negatives_cap` rows (default: match original neg count)
        cap = cfg.get("extra_negatives_cap", len(train_df[train_df["label"] == 0]))
        if len(extra_df) > cap:
            extra_df = extra_df.sample(n=cap, random_state=cfg["seed"])
        train_df = pd.concat([train_df, extra_df], ignore_index=True)
        n_orig_neg = (train_df["label"] == 0).sum() - len(extra_df)
        print(f"Extra negatives: {len(extra_df)} rows from {extra_neg_csv}", flush=True)
        print(f"  New train label dist: pos={( train_df['label']==1).sum()} "
              f"neg={(train_df['label']==0).sum()}", flush=True)

    print(f"Train clips: {len(train_df)}  |  Val clips: {len(val_df)}", flush=True)

    w_sec = cfg.get("window_sec", 2.0)
    s_sec = cfg.get("stride_sec", 1.0)
    pad_to_sec = cfg.get("pad_to_sec", None)
    train_ds = MILBagDataset(train_df, window_sec=w_sec, stride_sec=s_sec, pad_to_sec=pad_to_sec)
    val_ds = MILBagDataset(val_df, window_sec=w_sec, stride_sec=s_sec, pad_to_sec=pad_to_sec)

    # ── Model ──────────────────────────────────────────────────────────────
    model = build_mil_model(cfg).to(device)
    print(f"Backbone: {cfg['backbone']} | MIL head params: "
          f"{sum(p.numel() for p in model.mil_head.parameters()):,}", flush=True)

    # ── Pre-compute backbone embeddings (frozen backbone — run once only) ──
    print("Pre-computing train embeddings ...", flush=True)
    train_emb_cache = _precompute_embeddings(model, train_ds, device)
    print("Pre-computing val embeddings ...", flush=True)
    val_emb_cache = _precompute_embeddings(model, val_ds, device)

    # Build flat record lists (embeddings stay on CPU; moved to device per batch)
    train_records = [
        {
            "emb": train_emb_cache[str(row["audio_path"])],
            "label": float(row["label"]),
            "audio_path": str(row["audio_path"]),
            "child_id": str(row["child_id"]),
            "timepoint_norm": str(row["timepoint_norm"]),
        }
        for _, row in train_df.iterrows()
    ]
    val_records = [
        {
            "emb": val_emb_cache[str(row["audio_path"])],
            "label": float(row["label"]),
            "audio_path": str(row["audio_path"]),
            "child_id": str(row["child_id"]),
            "timepoint_norm": str(row["timepoint_norm"]),
        }
        for _, row in val_df.iterrows()
    ]

    # Only optimize MIL head (backbone is frozen)
    optimizer = torch.optim.Adam(model.mil_head.parameters(), lr=cfg.get("lr", 1e-3))

    pos_weight = cfg.get("pos_weight")
    if pos_weight is not None:
        criterion = torch.nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([pos_weight], device=device)
        )
    else:
        criterion = torch.nn.BCEWithLogitsLoss()

    # ── Training loop ──────────────────────────────────────────────────────
    best_val_f1 = -1.0
    best_ckpt_path = os.path.join(result_dir, "best_checkpoint.pt")
    patience = cfg.get("patience", 5)
    no_improve = 0
    history = []
    batch_size = cfg.get("batch_size", 8)

    for epoch in range(1, cfg.get("epochs", 20) + 1):
        model.mil_head.train()
        indices = torch.randperm(len(train_records)).tolist()
        train_losses = []

        for start in range(0, len(indices), batch_size):
            batch_recs = [train_records[i] for i in indices[start:start + batch_size]]
            batch_labels = torch.tensor(
                [r["label"] for r in batch_recs], dtype=torch.float32, device=device
            )
            batch_logits = []
            for rec in batch_recs:
                emb = rec["emb"].to(device)   # (N_windows, D)
                logit, _ = model.mil_head(emb)
                batch_logits.append(logit)

            logits = torch.stack(batch_logits)
            loss = criterion(logits, batch_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        # Val pass
        model.mil_head.eval()
        val_scores, val_labels_list, val_losses = [], [], []
        with torch.no_grad():
            for rec in val_records:
                emb = rec["emb"].to(device)
                logit, _ = model.mil_head(emb)
                score = float(torch.sigmoid(logit).item())
                val_scores.append(score)
                val_labels_list.append(int(rec["label"]))
                val_losses.append(
                    criterion(logit, torch.tensor(rec["label"], device=device)).item()
                )

        val_metrics = compute_metrics(val_labels_list, val_scores, threshold=0.5)
        mean_train_loss = float(np.mean(train_losses))
        mean_val_loss = float(np.mean(val_losses))

        print(f"Epoch {epoch:3d} | train_loss={mean_train_loss:.4f} "
              f"val_loss={mean_val_loss:.4f} val_f1={val_metrics['f1']:.4f} "
              f"val_auroc={val_metrics['auroc']:.4f}", flush=True)

        history.append({
            "epoch": epoch,
            "train_loss": mean_train_loss,
            "val_loss": mean_val_loss,
            "val_f1": val_metrics["f1"],
            "val_auroc": val_metrics["auroc"],
        })

        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            torch.save({"model_state": model.state_dict(), "cfg": cfg}, best_ckpt_path)
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping at epoch {epoch} (patience={patience})", flush=True)
                break

    # ── Post-training: threshold tuning + save outputs ─────────────────────
    print("Loading best checkpoint for val predictions …", flush=True)
    ckpt = torch.load(best_ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.mil_head.eval()

    val_scores, val_labels_list, val_meta = [], [], []
    with torch.no_grad():
        for rec in val_records:
            emb = rec["emb"].to(device)
            logit, _ = model.mil_head(emb)
            val_scores.append(float(torch.sigmoid(logit).item()))
            val_labels_list.append(int(rec["label"]))
            val_meta.append({
                "audio_path": rec["audio_path"],
                "child_id": rec["child_id"],
                "timepoint_norm": rec["timepoint_norm"],
            })

    threshold = tune_threshold(val_labels_list, val_scores)
    val_metrics_tuned = compute_metrics(val_labels_list, val_scores, threshold=threshold)
    val_metrics_tuned["threshold"] = threshold

    val_preds_df = pd.DataFrame([
        {**meta, "label": lbl, "score": sc, "prediction": int(sc >= threshold)}
        for meta, lbl, sc in zip(val_meta, val_labels_list, val_scores)
    ])

    save_json(cfg, os.path.join(result_dir, "config.json"))
    save_json(val_metrics_tuned, os.path.join(result_dir, "val_metrics_tuned.json"))
    save_csv(pd.DataFrame(history), os.path.join(result_dir, "training_history.csv"))
    save_csv(val_preds_df, os.path.join(result_dir, "val_predictions.csv"))
    tp_df = per_timepoint_metrics(val_preds_df)
    save_csv(tp_df, os.path.join(result_dir, "val_metrics_by_timepoint.csv"))

    print(f"\n=== Training complete ===", flush=True)
    print(f"  Val F1 (tuned, threshold={threshold:.2f}): {val_metrics_tuned['f1']:.4f}", flush=True)
    print(f"  Val AUROC: {val_metrics_tuned['auroc']:.4f}", flush=True)
    print(f"  Results in: {result_dir}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train MIL child presence model")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    args = parser.parse_args()
    cfg = parse_config(args.config)
    train(cfg)


if __name__ == "__main__":
    main()
