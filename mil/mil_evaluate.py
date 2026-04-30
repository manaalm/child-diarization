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

import numpy as np
import pandas as pd
import torch

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)

from mil.mil_dataset import MILBagDataset
from mil.mil_model import ACMILHead, build_mil_model
from mil.mil_train import (
    _attach_prototypes,
    _head_forward,
    _load_prototype_cache,
    _make_instance_embeddings,
    _precompute_embeddings_last_layer,
    _precompute_embeddings_per_layer,
    load_split,
)
from mil.mil_utils import compute_metrics, per_timepoint_metrics, save_csv, save_json


def evaluate(checkpoint_path: str, config_path: str) -> None:
    with open(config_path) as f:
        cfg = json.load(f)

    result_dir = os.path.dirname(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(checkpoint_path, map_location=device)
    model = build_mil_model(cfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    use_weighted_sum = cfg.get("layer_aggregation", "last") == "weighted_sum"
    is_acmil = isinstance(model.mil_head, ACMILHead)

    val_metrics_path = os.path.join(result_dir, "val_metrics_tuned.json")
    with open(val_metrics_path) as f:
        val_metrics = json.load(f)
    threshold = float(val_metrics["threshold"])
    print(f"Using val-tuned threshold: {threshold:.4f}", flush=True)

    # Optional prototype cache (US4)
    proto_cache = None
    proto_cache_path = cfg.get("prototype_cache")
    if proto_cache_path:
        proto_cache = _load_prototype_cache(proto_cache_path)

    def _run_split(split_name: str, out_prefix: str) -> None:
        df = load_split(cfg["split_dir"], split_name)
        w_sec = cfg.get("window_sec", 2.0)
        s_sec = cfg.get("stride_sec", 1.0)
        pad_to_sec = cfg.get("pad_to_sec", None)
        ds = MILBagDataset(df, window_sec=w_sec, stride_sec=s_sec, pad_to_sec=pad_to_sec)

        print(f"Pre-computing {split_name} embeddings ...", flush=True)
        if use_weighted_sum:
            emb_cache = _precompute_embeddings_per_layer(model, ds, device)
        else:
            emb_cache = _precompute_embeddings_last_layer(model, ds, device)

        records = []
        for _, row in df.iterrows():
            rec = {
                "label": float(row["label"]),
                "audio_path": str(row["audio_path"]),
                "child_id": str(row["child_id"]),
                "timepoint_norm": str(row["timepoint_norm"]),
                "emb": emb_cache[str(row["audio_path"])],
            }
            records.append(rec)

        if proto_cache is not None:
            records, missing = _attach_prototypes(records, proto_cache, split_name)
            save_json(missing, os.path.join(result_dir, f"missing_prototypes_{split_name}.json"))

        scores, labels, meta = [], [], []
        branch_attns_per_clip = []  # (n_clips, n_branches) list of mean-attention-per-branch
        with torch.no_grad():
            for rec in records:
                emb = _make_instance_embeddings(
                    rec, use_weighted_sum, model.backbone.layer_weights, device
                )
                proto = rec.get("prototype")
                if proto is not None:
                    proto = proto.to(device)
                if is_acmil:
                    logit, attn, branch_attn, _div = model.mil_head(emb)
                    branch_attns_per_clip.append(branch_attn.cpu().numpy())
                else:
                    logit, _attn, _div = _head_forward(model, emb, proto)
                scores.append(float(torch.sigmoid(logit).item()))
                labels.append(int(rec["label"]))
                meta.append({
                    "audio_path": rec["audio_path"],
                    "child_id": rec["child_id"],
                    "timepoint_norm": rec["timepoint_norm"],
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
        print(
            f"  {split_name}: F1={metrics['f1']:.4f}  AUROC={metrics['auroc']:.4f}  "
            f"AUPRC={metrics['auprc']:.4f}",
            flush=True,
        )

        # ACMIL-only outputs (US3 FR-013, FR-017): branch_weights summary + per-clip CSV
        if is_acmil and branch_attns_per_clip:
            branch_summary = {}
            n_branches = branch_attns_per_clip[0].shape[0]
            # Per-branch attention statistics, averaged across clips. Variable bag length
            # → use mean and std of mean-attention-per-clip for each branch.
            for b in range(n_branches):
                per_clip_means = [arr[b].mean() for arr in branch_attns_per_clip]
                per_clip_stds = [arr[b].std() for arr in branch_attns_per_clip]
                branch_summary[f"branch_{b}"] = {
                    "mean_of_clip_means": float(np.mean(per_clip_means)),
                    "std_of_clip_means": float(np.std(per_clip_means)),
                    "mean_of_clip_stds": float(np.mean(per_clip_stds)),
                }
            save_json(
                branch_summary, os.path.join(result_dir, f"branch_weights_{out_prefix}.json")
            )
            # Wide per-clip-per-instance CSV (one row per (clip, instance), col per branch).
            wide_rows = []
            for arr, m in zip(branch_attns_per_clip, meta):
                # arr: (n_branches, n_instances)
                n_instances = arr.shape[1]
                for inst_idx in range(n_instances):
                    row = {
                        "audio_path": m["audio_path"],
                        "instance_idx": inst_idx,
                    }
                    branch_means = []
                    for b in range(n_branches):
                        v = float(arr[b, inst_idx])
                        row[f"branch_{b}_weight"] = v
                        branch_means.append(v)
                    row["mean_weight"] = float(np.mean(branch_means))
                    wide_rows.append(row)
            save_csv(
                pd.DataFrame(wide_rows),
                os.path.join(result_dir, f"branch_attention_{out_prefix}.csv"),
            )

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
