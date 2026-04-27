#!/usr/bin/env python3
"""Correlate weak diarization attention scores with segment-MIL classification AUROC."""
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

REPO = Path("/orcd/scratch/orcd/008/manaal/child-adult-diarization")
OUT_DIR = REPO / "evaluation"
OUT_DIR.mkdir(parents=True, exist_ok=True)

weak_path = REPO / "mil/mil_results/seg_mil/weak_diarization_eval.csv"
configs_path = REPO / "mil/mil_results/seg_mil/all_configs.json"

weak_df = pd.read_csv(weak_path)
with open(configs_path) as f:
    all_cfgs = json.load(f)

print(f"Weak diar eval: {weak_df.shape}")
print(f"Configs: {len(all_cfgs)}")
print(f"Weak cols: {list(weak_df.columns)}")

# Build lookup: (frontend, aggregator) → test_auroc, test_f1, test_auprc
cfg_lookup = {}
for c in all_cfgs:
    key = (c["frontend"], c["aggregator"])
    cfg_lookup[key] = {"test_auroc": c.get("test_auroc"), "test_f1": c.get("test_f1"),
                       "test_auprc": c.get("test_auprc"),
                       "test_auroc_14month": c.get("test_auroc_14month"),
                       "test_auroc_36month": c.get("test_auroc_36month")}

# Merge with classification metrics
rows = []
for _, row in weak_df.iterrows():
    fe = row.get("frontend")
    agg = row.get("aggregator")
    tp = row.get("timepoint")
    key = (fe, agg)
    if key in cfg_lookup:
        r = dict(row)
        r.update(cfg_lookup[key])
        rows.append(r)

merged = pd.DataFrame(rows)
print(f"Merged: {merged.shape}")
print(merged.head(3).to_string())

# Correlation: attention AUROC ranking vs classification AUROC (all timepoints)
all_tp = merged[merged["timepoint"] == "all"].copy()
print(f"\n--- Correlations (all timepoints, n={len(all_tp)}) ---")

for weak_metric in ["pearson_r", "spearman_rho", "auroc_ranking"]:
    for clf_metric in ["test_auroc", "test_f1", "test_auprc"]:
        valid = all_tp[[weak_metric, clf_metric]].dropna()
        if len(valid) < 5:
            continue
        r, p = stats.spearmanr(valid[weak_metric], valid[clf_metric])
        print(f"  {weak_metric:20s} vs {clf_metric:15s}: Spearman r={r:.3f}, p={p:.3f} (n={len(valid)})")

# Table: per frontend/aggregator
summary = all_tp[["frontend", "aggregator", "pearson_r", "spearman_rho", "auroc_ranking",
                   "test_auroc", "test_f1", "test_auprc", "n_segments", "n_clips"]].copy()
summary = summary.sort_values("test_auroc", ascending=False)

out_path = OUT_DIR / "weak_diarization_correlation_analysis.csv"
summary.to_csv(out_path, index=False, float_format="%.4f")
print(f"\nWrote {out_path}")
print(summary.to_string(index=False))
