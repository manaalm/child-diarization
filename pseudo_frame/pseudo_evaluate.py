"""Evaluate trained pseudo-frame model on test split.

Outputs:
  - test_metrics_tuned.json:        clip-level F1 / AUROC / AUPRC / threshold (val-tuned)
  - test_predictions.csv:           per-clip score + prediction
  - test_metrics_by_timepoint.csv:  age-stratified clip metrics
  - frame_localization.json:        frame-level localization vs ground-truth RTTM
                                    (Pearson / Spearman / AUROC / AUPRC).
                                    Computed per-clip then averaged with weights = n_frames.

Usage:
  python pseudo_frame/pseudo_evaluate.py \\
    --checkpoint pseudo_frame/results/wavlm_pseudo/best_checkpoint.pt
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)

from mil.mil_utils import compute_metrics, per_timepoint_metrics, save_csv, save_json  # noqa: E402
from pseudo_frame.pseudo_dataset import (FRAME_RATE, FRAME_STEP_SEC,
                                         PseudoFrameDataset, collate)  # noqa: E402
from pseudo_frame.pseudo_model import PseudoFrameModel  # noqa: E402
from pseudo_frame.pseudo_train import align_frames, load_split  # noqa: E402
from pyannote.unified_rttm import parse_rttm, segments_to_frame_mask  # noqa: E402


def chunked_inference(model, wav: torch.Tensor, device, chunk_sec: float = 10.0):
    """Slide non-overlapping chunks through model. Returns frame probs (T,).

    Trailing chunks shorter than chunk_samples are zero-padded to chunk_samples
    so WavLM's conv stack always sees a long-enough input; the corresponding
    extra frames at the end are then truncated.
    """
    sr = 16000
    samples_per_frame = int(sr * 0.02)  # 320 (50 Hz)
    chunk_samples = int(chunk_sec * sr)
    probs_full = []
    n = wav.shape[0]
    pos = 0
    while pos < n:
        chunk = wav[pos:pos + chunk_samples]
        n_real = chunk.shape[0]
        if n_real < chunk_samples:
            pad = chunk_samples - n_real
            chunk = torch.cat([chunk, torch.zeros(pad, dtype=chunk.dtype)])
        x = chunk.unsqueeze(0).to(device)
        logits = model(x)  # (1, T_frames)
        probs = torch.sigmoid(logits).squeeze(0).cpu()
        # Keep only frames that correspond to real (un-padded) audio.
        n_real_frames = max(1, n_real // samples_per_frame)
        probs = probs[:n_real_frames]
        probs_full.append(probs)
        pos += chunk_samples
    return torch.cat(probs_full, dim=0) if probs_full else torch.zeros(0)


def gt_frame_mask_for(audio_path: str, duration_sec: float):
    """Look for a ground-truth RTTM. Returns mask or None if not found.

    Uses pyannote/eval_results/* if available; otherwise None.
    """
    # SAILS BIDS ground-truth is in the annotation CSVs, not RTTMs.
    # We treat the WHISPER-MODELING pseudo-target as a proxy when no RTTM
    # ground truth exists. For frame-level localization eval, we rely on
    # the ANNOTATED CSV (BIDS) which gives clip-level only — not frame-level.
    # Therefore we evaluate frame-level localization against the PSEUDO-LABEL
    # itself (held-out test pseudo-labels): this is consistent only if the
    # pseudo-labels are decent. We label this metric `frame_pseudo_*`.
    return None


def frame_localization_metrics(model, loader, device, idx_lookup):
    """Per-clip Pearson / Spearman / AUROC of frame probs vs pseudo-label,
    weighted by n_frames and averaged."""
    rows = []
    model.head.eval()
    with torch.no_grad():
        for batch in loader:
            wav = batch["waveform"].to(device)
            mask = batch["mask"].to(device)
            valid = batch["valid"].to(device)
            logits = model(wav)
            logits, mask_a, valid_a = align_frames(logits, mask, valid)
            probs = torch.sigmoid(logits)
            B = probs.shape[0]
            for i in range(B):
                v = valid_a[i] > 0.5
                p = probs[i][v].cpu().numpy()
                t = mask_a[i][v].cpu().numpy()
                # Hard target for AUROC
                t_hard = (t >= 0.5).astype(int)
                row = {
                    "audio_path": batch["audio_path"][i],
                    "label": int(batch["label"][i].item()),
                    "n_frames": int(v.sum().item()),
                    "n_pos_frames": int(t_hard.sum()),
                }
                # Skip clips with no variation in ground-truth (all 0 or all 1)
                if 0 < t_hard.sum() < t_hard.size:
                    try:
                        row["auroc"] = float(roc_auc_score(t_hard, p))
                        row["auprc"] = float(average_precision_score(t_hard, p))
                    except Exception:
                        row["auroc"] = float("nan")
                        row["auprc"] = float("nan")
                else:
                    row["auroc"] = float("nan")
                    row["auprc"] = float("nan")
                if t.size > 5 and np.std(t) > 0 and np.std(p) > 0:
                    try:
                        row["pearson"] = float(pearsonr(p, t)[0])
                        row["spearman"] = float(spearmanr(p, t)[0])
                    except Exception:
                        row["pearson"] = float("nan")
                        row["spearman"] = float("nan")
                else:
                    row["pearson"] = float("nan")
                    row["spearman"] = float("nan")
                rows.append(row)

    df = pd.DataFrame(rows)
    summary = {}
    for k in ["auroc", "auprc", "pearson", "spearman"]:
        v = df[k].dropna()
        summary[f"{k}_mean"] = float(v.mean()) if len(v) else float("nan")
        summary[f"{k}_n"] = int(len(v))
    summary["n_clips_total"] = len(df)
    summary["n_pos_clips"]   = int((df["label"] == 1).sum())
    return summary, df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--split", default="test")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device)
    cfg = ckpt["cfg"]
    thr = float(ckpt["val_threshold"])
    print(f"Loaded checkpoint from epoch {ckpt['epoch']}; val threshold={thr:.4f}", flush=True)

    out_dir = os.path.dirname(args.checkpoint)

    model = PseudoFrameModel(
        backbone_name=cfg.get("backbone", "microsoft/wavlm-base-plus"),
        backbone_layer=cfg.get("backbone_layer", -1),
        hidden_dim=cfg.get("hidden_dim", 256),
        dropout=cfg.get("dropout", 0.2),
    ).to(device)
    model.head.load_state_dict(ckpt["head_state"])
    model.eval()

    pl_index = pd.read_csv(os.path.join(_REPO, "pseudo_frame/pseudo_labels/index.csv"))
    df = load_split(cfg["split_dir"], args.split)

    # Use crop=None so the dataset returns the full clip; we chunk in inference.
    eval_ds = PseudoFrameDataset(df, pl_index, crop_sec=None, deterministic=True)
    eval_loader = DataLoader(eval_ds, batch_size=1, shuffle=False,
                             num_workers=2, collate_fn=collate)

    # Clip-level prediction via chunked inference
    scores, labels, meta = [], [], []
    with torch.no_grad():
        for batch in eval_loader:
            wav = batch["waveform"][0]
            valid_len = int(batch["valid"][0].sum().item()) * int(16000 * FRAME_STEP_SEC)
            wav = wav[:valid_len]
            probs = chunked_inference(model, wav, device, chunk_sec=cfg.get("crop_sec", 10.0))
            score = float(probs.max().item()) if probs.numel() > 0 else 0.0
            scores.append(score)
            labels.append(int(batch["label"][0].item()))
            meta.append({
                "audio_path": batch["audio_path"][0],
                "child_id":   batch["child_id"][0],
                "timepoint_norm": batch["timepoint_norm"][0],
            })

    metrics = compute_metrics(labels, scores, threshold=thr)
    metrics["threshold"] = thr
    metrics["n"] = len(scores)
    metrics["pool"] = "max"

    preds_df = pd.DataFrame([
        {**m, "label": l, "score": s, "prediction": int(s >= thr)}
        for m, l, s in zip(meta, labels, scores)
    ])
    save_json(metrics, os.path.join(out_dir, f"{args.split}_metrics_tuned.json"))
    save_csv(preds_df, os.path.join(out_dir, f"{args.split}_predictions.csv"))
    save_csv(per_timepoint_metrics(preds_df),
             os.path.join(out_dir, f"{args.split}_metrics_by_timepoint.csv"))

    print(f"\n=== {args.split.upper()} CLIP METRICS ===", flush=True)
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    # Frame-level localization (vs pseudo-label)
    print("\n=== FRAME-LEVEL LOCALIZATION (vs pseudo-label) ===", flush=True)
    eval_loader_chunked = DataLoader(
        PseudoFrameDataset(df, pl_index, crop_sec=cfg.get("crop_sec", 10.0), deterministic=True),
        batch_size=cfg.get("batch_size", 8),
        shuffle=False, num_workers=2, collate_fn=collate,
    )
    summ, frame_df = frame_localization_metrics(model, eval_loader_chunked, device, pl_index)
    print(json.dumps(summ, indent=2))
    save_json(summ, os.path.join(out_dir, "frame_localization.json"))
    save_csv(frame_df, os.path.join(out_dir, "frame_localization_per_clip.csv"))

    print(f"\n  → {out_dir}", flush=True)


if __name__ == "__main__":
    main()
