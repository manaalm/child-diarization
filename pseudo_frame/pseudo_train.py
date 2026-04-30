"""Train WavLM-Base+ frame classifier on diarizer-derived pseudo-labels.

Loss:  per-frame BCEWithLogitsLoss against soft pseudo-labels in [0, 1].
       Optionally weighted by confidence (|target - 0.5| * 2 ∈ {0, 1}) so frames
       where VTC and USC-SAIL disagree contribute less.

Val:   clip-level F1/AUROC tuned by max-pooling frame probs and sweeping threshold.
       Best checkpoint by val_f1.

Usage:
  python pseudo_frame/pseudo_train.py --config pseudo_frame/configs/wavlm_pseudo.yaml
"""
import argparse
import os
import random
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)

from mil.mil_utils import compute_metrics, save_csv, save_json, tune_threshold  # noqa: E402
from pseudo_frame.pseudo_dataset import PseudoFrameDataset, collate  # noqa: E402
from pseudo_frame.pseudo_model import PseudoFrameModel  # noqa: E402


def parse_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def setup_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_split(split_dir: str, split: str) -> pd.DataFrame:
    df = pd.read_csv(os.path.join(_REPO, split_dir, f"{split}.csv"))
    if "audio_exists" in df.columns:
        df = df[df["audio_exists"] == True]
    if "timepoint_norm" not in df.columns and "timepoint" in df.columns:
        df = df.rename(columns={"timepoint": "timepoint_norm"})
    return df.reset_index(drop=True)


def align_frames(frame_logits: torch.Tensor, target_mask: torch.Tensor,
                 valid: torch.Tensor):
    """Truncate to matching frame count between WavLM output and pseudo-mask.

    WavLM-Base+ produces ⌊T_audio/320⌋ frames; pseudo-mask is built at the same
    rate but RTTM rounding can give ±1 frame discrepancy.
    """
    T_pred = frame_logits.shape[1]
    T_targ = target_mask.shape[1]
    T = min(T_pred, T_targ)
    return frame_logits[:, :T], target_mask[:, :T], valid[:, :T]


def frame_bce_loss(logits, target, valid, confidence_weight: bool = True,
                   pos_weight: float = 1.0):
    """Soft BCE per frame. `valid` masks padding. `confidence_weight` downweights
    frames where the two source diarizers disagreed (target = 0.5)."""
    bce = F.binary_cross_entropy_with_logits(
        logits, target,
        pos_weight=torch.tensor(pos_weight, device=logits.device),
        reduction="none",
    )
    w = valid.clone()
    if confidence_weight:
        # 1 at target ∈ {0, 1}; 0 at target = 0.5
        conf = (2.0 * (target - 0.5).abs()).clamp(0.0, 1.0)
        w = w * conf
    denom = w.sum().clamp(min=1.0)
    return (bce * w).sum() / denom


def epoch_pass(model, loader, device, optimizer=None, **loss_kw):
    is_train = optimizer is not None
    model.head.train(is_train)
    losses, scores, labels = [], [], []
    for batch in loader:
        wav = batch["waveform"].to(device)
        mask = batch["mask"].to(device)
        valid = batch["valid"].to(device)
        with torch.set_grad_enabled(is_train):
            logits = model(wav)
            logits, mask_a, valid_a = align_frames(logits, mask, valid)
            loss = frame_bce_loss(logits, mask_a, valid_a, **loss_kw)
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        losses.append(loss.item())
        with torch.no_grad():
            probs = torch.sigmoid(logits)
            clip = model.clip_score(probs, valid_a)
        scores.extend(clip.detach().cpu().tolist())
        labels.extend(batch["label"].tolist())
    return float(np.mean(losses)), scores, labels


def train(cfg: dict) -> None:
    setup_seed(cfg["seed"])
    device = torch.device(cfg.get("device", "cuda") if torch.cuda.is_available() else "cpu")
    variant = cfg["variant_name"]
    out_dir = os.path.join(_REPO, "pseudo_frame/results", variant)
    os.makedirs(out_dir, exist_ok=True)

    # Data
    pseudo_labels_dir = cfg.get("pseudo_labels_dir", "pseudo_frame/pseudo_labels")
    idx_path = os.path.join(_REPO, pseudo_labels_dir, "index.csv")
    if not os.path.exists(idx_path):
        raise FileNotFoundError(
            f"Missing pseudo-label index: {idx_path}. "
            f"Run the appropriate pseudo_frame/build_*.py first "
            f"(or distill_c1_pseudo_labels.py for C1 self-distillation)."
        )
    pl_index = pd.read_csv(idx_path)

    train_df = load_split(cfg["split_dir"], "train")
    val_df = load_split(cfg["split_dir"], "val")
    print(f"Train: {len(train_df)}  Val: {len(val_df)}", flush=True)

    crop_sec = cfg.get("crop_sec", 10.0)
    train_ds = PseudoFrameDataset(train_df, pl_index, crop_sec=crop_sec, deterministic=False)
    val_ds = PseudoFrameDataset(val_df, pl_index, crop_sec=crop_sec, deterministic=True)

    bsz = cfg.get("batch_size", 8)
    train_loader = DataLoader(train_ds, batch_size=bsz, shuffle=True,
                              num_workers=cfg.get("num_workers", 2),
                              collate_fn=collate, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=bsz, shuffle=False,
                            num_workers=cfg.get("num_workers", 2),
                            collate_fn=collate, drop_last=False)

    # Model
    model = PseudoFrameModel(
        backbone_name=cfg.get("backbone", "microsoft/wavlm-base-plus"),
        backbone_layer=cfg.get("backbone_layer", -1),
        hidden_dim=cfg.get("hidden_dim", 256),
        dropout=cfg.get("dropout", 0.2),
    ).to(device)
    n_params = sum(p.numel() for p in model.head.parameters())
    print(f"Frame head params: {n_params:,}", flush=True)

    optimizer = torch.optim.AdamW(model.head.parameters(),
                                  lr=cfg.get("lr", 1e-3),
                                  weight_decay=cfg.get("weight_decay", 1e-4))

    loss_kw = dict(
        confidence_weight=cfg.get("confidence_weight", True),
        pos_weight=cfg.get("pos_weight", 3.0),
    )

    best_f1 = -1.0
    no_improve = 0
    patience = cfg.get("patience", 5)
    history = []

    for epoch in range(1, cfg.get("epochs", 20) + 1):
        tr_loss, _, _ = epoch_pass(model, train_loader, device, optimizer, **loss_kw)
        va_loss, va_scores, va_labels = epoch_pass(model, val_loader, device, None, **loss_kw)

        # Convert max-pool frame probs to clip score (already produced by clip_score)
        # Replace -inf or NaN that may come from all-padding clips
        va_scores = [0.0 if (not np.isfinite(s)) else float(s) for s in va_scores]
        m05 = compute_metrics(va_labels, va_scores, threshold=0.5)
        thr = tune_threshold(va_labels, va_scores)
        m_tuned = compute_metrics(va_labels, va_scores, threshold=thr)

        print(
            f"Epoch {epoch:3d} | tr_loss={tr_loss:.4f} va_loss={va_loss:.4f} "
            f"va_f1@0.5={m05['f1']:.4f} va_f1*={m_tuned['f1']:.4f}@thr={thr:.2f} "
            f"va_auroc={m_tuned['auroc']:.4f} va_auprc={m_tuned['auprc']:.4f}",
            flush=True,
        )
        history.append({
            "epoch": epoch, "train_loss": tr_loss, "val_loss": va_loss,
            "val_f1": m_tuned["f1"], "val_auroc": m_tuned["auroc"],
            "val_threshold": thr,
        })

        if m_tuned["f1"] > best_f1:
            best_f1 = m_tuned["f1"]
            torch.save({
                "head_state": model.head.state_dict(),
                "cfg": cfg,
                "val_threshold": thr,
                "val_metrics": m_tuned,
                "epoch": epoch,
            }, os.path.join(out_dir, "best_checkpoint.pt"))
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping at epoch {epoch} (patience={patience})", flush=True)
                break

    # Save artifacts
    save_json(cfg, os.path.join(out_dir, "config.json"))
    save_csv(pd.DataFrame(history), os.path.join(out_dir, "training_history.csv"))

    # Reload best & dump val predictions
    ckpt = torch.load(os.path.join(out_dir, "best_checkpoint.pt"), map_location=device)
    model.head.load_state_dict(ckpt["head_state"])
    thr = ckpt["val_threshold"]
    _, va_scores, va_labels = epoch_pass(model, val_loader, device, None, **loss_kw)
    va_scores = [0.0 if not np.isfinite(s) else float(s) for s in va_scores]
    val_metrics = compute_metrics(va_labels, va_scores, threshold=thr)
    val_metrics["threshold"] = thr
    save_json(val_metrics, os.path.join(out_dir, "val_metrics_tuned.json"))

    val_records = []
    for batch_meta_idx, item in enumerate(val_ds):
        # Re-iterate to record audio_path order matching scores
        pass
    # Simpler: rebuild from val_ds + scores in same order
    for i, sc in enumerate(va_scores):
        rec = val_ds.records.iloc[i]
        val_records.append({
            "audio_path": str(rec["audio_path"]),
            "child_id":   str(rec["child_id"]),
            "timepoint_norm": str(rec["timepoint_norm"]),
            "label": int(rec["label"]),
            "score": sc,
            "prediction": int(sc >= thr),
        })
    save_csv(pd.DataFrame(val_records), os.path.join(out_dir, "val_predictions.csv"))

    print(f"\n=== TRAINING COMPLETE ===", flush=True)
    print(f"  Best val F1: {best_f1:.4f}", flush=True)
    print(f"  Val tuned: F1={val_metrics['f1']:.4f} AUROC={val_metrics['auroc']:.4f} "
          f"AUPRC={val_metrics['auprc']:.4f} threshold={thr:.2f}", flush=True)
    print(f"  Output → {out_dir}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = parse_config(args.config)
    train(cfg)


if __name__ == "__main__":
    main()
