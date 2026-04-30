"""Re-evaluate an ACMIL checkpoint using per-branch selection (no retrain).

The trained ACMIL model has n_branches independent attention+bag-head pairs,
combined at inference time by mean(branch_logits). This script evaluates each
branch's logit *individually*, computes val and test metrics per branch,
picks the best branch by val AUROC, and reports its test metrics.

This is the "single_best" branch-aggregation strategy applied at inference time
to an already-trained "mean" checkpoint — no GPU training required.

Usage:
    python mil/eval_acmil_branch_selection.py \\
        --results-dir mil/mil_results/wavlm_mil_acmil

Output:
    {results_dir}/branch_selection.json   — per-branch val/test metrics + best-branch summary
    {results_dir}/branch_selection.csv    — per-branch metrics in tabular form
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
    _load_prototype_cache,
    _make_instance_embeddings,
    _precompute_embeddings_last_layer,
    _precompute_embeddings_per_layer,
    load_split,
)
from mil.mil_utils import compute_metrics, save_csv, save_json, tune_threshold


def _gather_branch_scores(model, df, cfg, device, proto_cache):
    use_weighted_sum = cfg.get("layer_aggregation", "last") == "weighted_sum"
    w_sec = cfg.get("window_sec", 2.0)
    s_sec = cfg.get("stride_sec", 1.0)
    pad_to_sec = cfg.get("pad_to_sec", None)
    ds = MILBagDataset(df, window_sec=w_sec, stride_sec=s_sec, pad_to_sec=pad_to_sec)
    if use_weighted_sum:
        emb_cache = _precompute_embeddings_per_layer(model, ds, device)
    else:
        emb_cache = _precompute_embeddings_last_layer(model, ds, device)

    records = []
    for _, row in df.iterrows():
        records.append({
            "label": float(row["label"]),
            "audio_path": str(row["audio_path"]),
            "child_id": str(row["child_id"]),
            "timepoint_norm": str(row["timepoint_norm"]),
            "emb": emb_cache[str(row["audio_path"])],
        })

    if proto_cache is not None:
        records, _missing = _attach_prototypes(records, proto_cache, "split")

    head: ACMILHead = model.mil_head
    n_branches = head.n_branches

    branch_scores = np.zeros((len(records), n_branches), dtype=np.float64)
    labels = np.zeros(len(records), dtype=np.int64)
    meta = []
    head.eval()
    with torch.no_grad():
        for i, rec in enumerate(records):
            emb = _make_instance_embeddings(
                rec, use_weighted_sum, model.backbone.layer_weights, device
            )
            logits, _attn = head.forward_branches(emb)  # (B,)
            branch_scores[i] = torch.sigmoid(logits).cpu().numpy()
            labels[i] = int(rec["label"])
            meta.append({
                "audio_path": rec["audio_path"],
                "child_id": rec["child_id"],
                "timepoint_norm": rec["timepoint_norm"],
            })
    return branch_scores, labels, meta, n_branches


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", required=True)
    ap.add_argument("--checkpoint-name", default="best_checkpoint.pt")
    args = ap.parse_args()

    cfg_path = os.path.join(args.results_dir, "config.json")
    ckpt_path = os.path.join(args.results_dir, args.checkpoint_name)
    if not os.path.isfile(cfg_path) or not os.path.isfile(ckpt_path):
        print(f"ERROR: missing config.json or checkpoint in {args.results_dir}", file=sys.stderr)
        sys.exit(2)
    with open(cfg_path) as f:
        cfg = json.load(f)
    if cfg.get("head") != "acmil":
        print(f"ERROR: head != 'acmil' (got {cfg.get('head')!r})", file=sys.stderr)
        sys.exit(2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(ckpt_path, map_location=device)
    model = build_mil_model(cfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    proto_cache = None
    if cfg.get("prototype_cache"):
        proto_cache = _load_prototype_cache(cfg["prototype_cache"])

    val_df = load_split(cfg["split_dir"], "val")
    test_df = load_split(cfg["split_dir"], "test")

    print(f"Computing per-branch val scores ({len(val_df)} clips) ...", flush=True)
    val_scores, val_labels, _val_meta, n_branches = _gather_branch_scores(
        model, val_df, cfg, device, proto_cache
    )
    print(f"Computing per-branch test scores ({len(test_df)} clips) ...", flush=True)
    test_scores, test_labels, test_meta, _ = _gather_branch_scores(
        model, test_df, cfg, device, proto_cache
    )

    rows = []
    per_branch_metrics = {}
    for b in range(n_branches):
        v_thr = float(tune_threshold(val_labels.tolist(), val_scores[:, b].tolist()))
        v_metrics = compute_metrics(val_labels, val_scores[:, b], threshold=v_thr)
        t_metrics = compute_metrics(test_labels, test_scores[:, b], threshold=v_thr)
        per_branch_metrics[f"branch_{b}"] = {
            "val": v_metrics, "test": t_metrics, "val_threshold": v_thr,
        }
        rows.append({
            "branch": f"branch_{b}",
            "val_f1": v_metrics["f1"], "val_auroc": v_metrics["auroc"], "val_auprc": v_metrics["auprc"],
            "test_f1": t_metrics["f1"], "test_auroc": t_metrics["auroc"], "test_auprc": t_metrics["auprc"],
            "val_threshold": v_thr,
        })

    # mean (current default) for reference
    val_mean = val_scores.mean(axis=1)
    test_mean = test_scores.mean(axis=1)
    v_thr = float(tune_threshold(val_labels.tolist(), val_mean.tolist()))
    rows.append({
        "branch": "mean",
        "val_f1": compute_metrics(val_labels, val_mean, threshold=v_thr)["f1"],
        "val_auroc": compute_metrics(val_labels, val_mean, threshold=v_thr)["auroc"],
        "val_auprc": compute_metrics(val_labels, val_mean, threshold=v_thr)["auprc"],
        "test_f1": compute_metrics(test_labels, test_mean, threshold=v_thr)["f1"],
        "test_auroc": compute_metrics(test_labels, test_mean, threshold=v_thr)["auroc"],
        "test_auprc": compute_metrics(test_labels, test_mean, threshold=v_thr)["auprc"],
        "val_threshold": v_thr,
    })

    # max (DSMIL-style) for reference
    val_max = val_scores.max(axis=1)
    test_max = test_scores.max(axis=1)
    v_thr = float(tune_threshold(val_labels.tolist(), val_max.tolist()))
    rows.append({
        "branch": "max_over_branches",
        "val_f1": compute_metrics(val_labels, val_max, threshold=v_thr)["f1"],
        "val_auroc": compute_metrics(val_labels, val_max, threshold=v_thr)["auroc"],
        "val_auprc": compute_metrics(val_labels, val_max, threshold=v_thr)["auprc"],
        "test_f1": compute_metrics(test_labels, test_max, threshold=v_thr)["f1"],
        "test_auroc": compute_metrics(test_labels, test_max, threshold=v_thr)["auroc"],
        "test_auprc": compute_metrics(test_labels, test_max, threshold=v_thr)["auprc"],
        "val_threshold": v_thr,
    })

    # Best branch (selected by val AUROC)
    best_idx = int(np.argmax([per_branch_metrics[f"branch_{b}"]["val"]["auroc"] for b in range(n_branches)]))
    best_v = per_branch_metrics[f"branch_{best_idx}"]["val"]
    best_t = per_branch_metrics[f"branch_{best_idx}"]["test"]
    print(f"Best single branch by val AUROC: branch_{best_idx} "
          f"(val AUROC={best_v['auroc']:.4f}, test AUROC={best_t['auroc']:.4f})")

    # Top-K mean (K = ceil(n_branches / 2)) — uses branches selected by val AUROC
    k = max(1, n_branches // 2 + n_branches % 2)
    branch_val_aurocs = [per_branch_metrics[f"branch_{b}"]["val"]["auroc"] for b in range(n_branches)]
    topk_idx = sorted(range(n_branches), key=lambda b: -branch_val_aurocs[b])[:k]
    val_topk_mean = val_scores[:, topk_idx].mean(axis=1)
    test_topk_mean = test_scores[:, topk_idx].mean(axis=1)
    v_thr = float(tune_threshold(val_labels.tolist(), val_topk_mean.tolist()))
    rows.append({
        "branch": f"topk_mean_(k={k},idx={topk_idx})",
        "val_f1": compute_metrics(val_labels, val_topk_mean, threshold=v_thr)["f1"],
        "val_auroc": compute_metrics(val_labels, val_topk_mean, threshold=v_thr)["auroc"],
        "val_auprc": compute_metrics(val_labels, val_topk_mean, threshold=v_thr)["auprc"],
        "test_f1": compute_metrics(test_labels, test_topk_mean, threshold=v_thr)["f1"],
        "test_auroc": compute_metrics(test_labels, test_topk_mean, threshold=v_thr)["auroc"],
        "test_auprc": compute_metrics(test_labels, test_topk_mean, threshold=v_thr)["auprc"],
        "val_threshold": v_thr,
    })

    out_df = pd.DataFrame(rows)
    save_csv(out_df, os.path.join(args.results_dir, "branch_selection.csv"))
    save_json({
        "n_branches": n_branches,
        "best_branch_by_val_auroc": f"branch_{best_idx}",
        "best_branch_val": per_branch_metrics[f"branch_{best_idx}"]["val"],
        "best_branch_test": per_branch_metrics[f"branch_{best_idx}"]["test"],
        "topk_branches": [f"branch_{i}" for i in topk_idx],
        "per_branch": per_branch_metrics,
    }, os.path.join(args.results_dir, "branch_selection.json"))

    print("\n=== Branch selection summary ===")
    print(out_df.to_string(index=False))


if __name__ == "__main__":
    main()
