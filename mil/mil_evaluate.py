"""Evaluate a trained MIL checkpoint on the test split.

Usage:
    python mil/mil_evaluate.py \\
        --checkpoint mil/mil_results/wavlm_mil/best_checkpoint.pt \\
        --config     mil/mil_results/wavlm_mil/config.json
"""

import argparse
import json
import os
import sys

import pandas as pd
import torch

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)

from mil.mil_dataset import MILBagDataset
from mil.mil_model import build_mil_model
from mil.mil_utils import compute_metrics, per_timepoint_metrics, save_csv, save_json
from mil.mil_train import load_split, _precompute_embeddings


def evaluate(checkpoint_path: str, config_path: str) -> None:
    with open(config_path) as f:
        cfg = json.load(f)

    result_dir = os.path.dirname(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    ckpt = torch.load(checkpoint_path, map_location=device)
    model = build_mil_model(cfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # Load threshold from val metrics
    val_metrics_path = os.path.join(result_dir, "val_metrics_tuned.json")
    with open(val_metrics_path) as f:
        val_metrics = json.load(f)
    threshold = float(val_metrics["threshold"])
    print(f"Using val-tuned threshold: {threshold:.4f}", flush=True)

    def _run_split(split_name: str, out_prefix: str) -> None:
        df = load_split(cfg["split_dir"], split_name)
        w_sec = cfg.get("window_sec", 2.0)
        s_sec = cfg.get("stride_sec", 1.0)
        pad_to_sec = cfg.get("pad_to_sec", None)
        ds = MILBagDataset(df, window_sec=w_sec, stride_sec=s_sec, pad_to_sec=pad_to_sec)

        print(f"Pre-computing {split_name} embeddings ...", flush=True)
        emb_cache = _precompute_embeddings(model, ds, device)

        scores, labels, meta = [], [], []
        with torch.no_grad():
            for _, row in df.iterrows():
                path = str(row["audio_path"])
                emb = emb_cache[path].to(device)
                logit, _ = model.mil_head(emb)
                scores.append(float(torch.sigmoid(logit).item()))
                labels.append(int(row["label"]))
                meta.append({
                    "audio_path": path,
                    "child_id": str(row["child_id"]),
                    "timepoint_norm": str(row["timepoint_norm"]),
                })

        metrics = compute_metrics(labels, scores, threshold=threshold)
        metrics["threshold"] = threshold

        preds_df = pd.DataFrame([
            {**m, "label": lbl, "score": sc, "prediction": int(sc >= threshold)}
            for m, lbl, sc in zip(meta, labels, scores)
        ])
        tp_df = per_timepoint_metrics(preds_df)

        save_json(metrics, os.path.join(result_dir, f"{out_prefix}_metrics_tuned.json"))
        save_csv(preds_df, os.path.join(result_dir, f"{out_prefix}_predictions.csv"))
        save_csv(tp_df, os.path.join(result_dir, f"{out_prefix}_metrics_by_timepoint.csv"))
        print(f"  {split_name}: F1={metrics['f1']:.4f}  AUROC={metrics['auroc']:.4f}  "
              f"AUPRC={metrics['auprc']:.4f}", flush=True)

    print("=== Evaluating test split ===", flush=True)
    _run_split("test", "test")

    val_tp_path = os.path.join(result_dir, "val_metrics_by_timepoint.csv")
    if not os.path.exists(val_tp_path):
        print("=== Producing val by-timepoint metrics ===", flush=True)
        _run_split("val", "val")

    print(f"Results written to: {result_dir}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate MIL checkpoint on test split")
    parser.add_argument("--checkpoint", required=True, help="Path to best_checkpoint.pt")
    parser.add_argument("--config", required=True, help="Path to config.json")
    args = parser.parse_args()
    evaluate(args.checkpoint, args.config)


if __name__ == "__main__":
    main()
