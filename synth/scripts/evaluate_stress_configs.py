"""Evaluate a trained MIL checkpoint on synth stress-test scenes.

Reads scene JSONs produced by generate_scenes.py for one or more stress configs,
builds a DataFrame compatible with MILBagDataset, runs the model, and writes
clip-level metrics (F1/AUROC/AUPRC + scene_type breakdown).

Usage:
    python synth/scripts/evaluate_stress_configs.py \
        --checkpoint mil/mil_results/whisper_mil/best_checkpoint.pt \
        --config     mil/mil_results/whisper_mil/config.json \
        --configs    hard_negatives overlap_stress low_snr_stress \
        --output-dir synth_results/stress_test_results/
"""

import argparse
import json
import os
import sys
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd
import torch

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _REPO)

from mil.mil_dataset import MILBagDataset
from mil.mil_model import ACMILHead, build_mil_model
from mil.mil_train import (
    _head_forward,
    _make_instance_embeddings,
    _precompute_embeddings_last_layer,
    _precompute_embeddings_per_layer,
)
from mil.mil_utils import compute_metrics, save_csv, save_json


def build_stress_dataframe(scenes_dir: Path, config_name: str) -> pd.DataFrame:
    json_paths = sorted(glob(str(scenes_dir / "json" / f"{config_name}_*.json")))
    rows = []
    for jp in json_paths:
        with open(jp) as f:
            meta = json.load(f)
        rows.append({
            "audio_path": meta["audio_path"],
            "label": int(bool(meta.get("target_child_vocalized", False))),
            "child_id": "synth",
            "timepoint_norm": "14_18mo",
            "scene_type": meta.get("scene_type", ""),
            "scene_id": meta.get("synthetic_scene_id", Path(jp).stem),
            "mean_snr_db": meta.get("mean_snr_db"),
            "rir_id": meta.get("rir_id"),
            "noise_id": meta.get("noise_id"),
            "max_overlap_speakers": meta.get("max_overlap_speakers", 1),
            "non_target_child_present": bool(meta.get("non_target_child_present", False)),
        })
    return pd.DataFrame(rows)


def evaluate_config(
    model,
    cfg: dict,
    threshold: float,
    df: pd.DataFrame,
    device,
    use_weighted_sum: bool,
) -> tuple[pd.DataFrame, dict]:
    w_sec = cfg.get("window_sec", 2.0)
    s_sec = cfg.get("stride_sec", 1.0)
    pad_to_sec = cfg.get("pad_to_sec", None)
    ds = MILBagDataset(df, window_sec=w_sec, stride_sec=s_sec, pad_to_sec=pad_to_sec)

    print(f"  Pre-computing embeddings for {len(df)} scenes …", flush=True)
    if use_weighted_sum:
        emb_cache = _precompute_embeddings_per_layer(model, ds, device)
    else:
        emb_cache = _precompute_embeddings_last_layer(model, ds, device)

    records = []
    with torch.no_grad():
        for i, row in df.reset_index(drop=True).iterrows():
            emb = _make_instance_embeddings(model, emb_cache, i, device, use_weighted_sum)
            logit, _, _ = _head_forward(model, emb, prototype=None)
            prob = float(torch.sigmoid(logit).cpu().item())
            records.append({
                "scene_id": row["scene_id"],
                "audio_path": row["audio_path"],
                "label": int(row["label"]),
                "prob": prob,
                "pred": int(prob >= threshold),
                "scene_type": row["scene_type"],
                "mean_snr_db": row["mean_snr_db"],
                "max_overlap_speakers": row["max_overlap_speakers"],
                "non_target_child_present": row["non_target_child_present"],
            })

    pred_df = pd.DataFrame(records)
    metrics = compute_metrics(
        pred_df["label"].to_numpy(),
        pred_df["prob"].to_numpy(),
        threshold=threshold,
    )
    metrics["n"] = int(len(pred_df))
    metrics["n_positive"] = int(pred_df["label"].sum())
    metrics["threshold"] = float(threshold)
    return pred_df, metrics


def metrics_by_scene_type(pred_df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    rows = []
    for scene_type, grp in pred_df.groupby("scene_type"):
        labels = grp["label"].to_numpy()
        probs = grp["prob"].to_numpy()
        if labels.sum() == 0 or labels.sum() == len(labels):
            # Single-class: only F1 / accuracy meaningful
            preds = (probs >= threshold).astype(int)
            tp = int(((preds == 1) & (labels == 1)).sum())
            fp = int(((preds == 1) & (labels == 0)).sum())
            fn = int(((preds == 0) & (labels == 1)).sum())
            tn = int(((preds == 0) & (labels == 0)).sum())
            rows.append({
                "scene_type": scene_type,
                "n": len(labels),
                "n_positive": int(labels.sum()),
                "false_positive_rate": fp / max(fp + tn, 1),
                "true_positive_rate": tp / max(tp + fn, 1),
                "auroc": np.nan,
                "auprc": np.nan,
                "f1": np.nan,
            })
        else:
            m = compute_metrics(labels, probs, threshold=threshold)
            rows.append({
                "scene_type": scene_type,
                "n": len(labels),
                "n_positive": int(labels.sum()),
                "f1": m["f1"],
                "auroc": m["auroc"],
                "auprc": m["auprc"],
                "false_positive_rate": np.nan,
                "true_positive_rate": np.nan,
            })
    return pd.DataFrame(rows).sort_values("scene_type")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--scenes-dir", default="synth_results/synthetic_scenes")
    ap.add_argument("--configs", nargs="+",
                    default=["hard_negatives", "overlap_stress", "low_snr_stress"])
    ap.add_argument("--output-dir", default="synth_results/stress_test_results")
    args = ap.parse_args()

    repo = Path(_REPO)
    scenes_dir = (repo / args.scenes_dir).resolve()
    out_root = (repo / args.output_dir).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    with open(args.config) as f:
        cfg = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device)
    model = build_mil_model(cfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    use_weighted_sum = cfg.get("layer_aggregation", "last") == "weighted_sum"

    val_metrics_path = os.path.join(os.path.dirname(args.config), "val_metrics_tuned.json")
    with open(val_metrics_path) as f:
        threshold = float(json.load(f)["threshold"])
    print(f"Using val-tuned threshold: {threshold:.4f}", flush=True)

    summary = {}
    for config_name in args.configs:
        print(f"\n=== {config_name} ===", flush=True)
        df = build_stress_dataframe(scenes_dir, config_name)
        if df.empty:
            print(f"  No scenes found for {config_name} — skipping.", flush=True)
            summary[config_name] = {"status": "no_scenes"}
            continue
        print(f"  {len(df)} scenes; positive_rate={df['label'].mean():.3f}", flush=True)

        pred_df, metrics = evaluate_config(
            model, cfg, threshold, df, device, use_weighted_sum
        )

        cfg_out = out_root / config_name
        cfg_out.mkdir(parents=True, exist_ok=True)
        save_json(metrics, cfg_out / "test_metrics.json")
        pred_df.to_csv(cfg_out / "predictions.csv", index=False)

        by_type = metrics_by_scene_type(pred_df, threshold)
        by_type.to_csv(cfg_out / "metrics_by_scene_type.csv", index=False)

        run_meta = {
            "checkpoint": args.checkpoint,
            "config": args.config,
            "threshold": threshold,
            "n_scenes": int(len(df)),
            "n_positive": int(df["label"].sum()),
        }
        save_json(run_meta, cfg_out / "config.json")

        print(f"  F1={metrics['f1']:.4f}  AUROC={metrics['auroc']:.4f}  "
              f"AUPRC={metrics['auprc']:.4f}", flush=True)
        summary[config_name] = {
            "f1": metrics["f1"], "auroc": metrics["auroc"],
            "auprc": metrics["auprc"], "n": metrics["n"],
            "n_positive": metrics["n_positive"],
        }

    save_json(summary, out_root / "summary.json")
    print(f"\nWrote summary → {out_root}/summary.json", flush=True)


if __name__ == "__main__":
    main()
