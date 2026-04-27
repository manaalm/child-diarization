#!/usr/bin/env python3
"""Build cross-diarizer master metrics table with bootstrap confidence intervals."""
import glob
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score, precision_score, recall_score

REPO = Path("/orcd/scratch/orcd/008/manaal/child-adult-diarization")
OUT_DIR = REPO / "evaluation"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DIARIZERS = [
    ("BabAR",       REPO / "babar_ecapa_enrollment_runs",                          "enroll_test_metrics.json",  "enroll_test_predictions.csv"),
    ("Pyannote",    REPO / "pyannote/pyannote_enrollment_runs",                    "test_metrics.json",          "test_predictions.csv"),
    ("USC-SAIL",    REPO / "whisper-modeling/usc_sail_enrollment_runs",            "enroll_test_metrics.json",  "enroll_test_predictions.csv"),
    ("VTC",         REPO / "vtc_ecapa_enrollment_runs",                            "enroll_test_metrics.json",  "enroll_test_predictions.csv"),
    ("VTC-KCHI",    REPO / "vtc_kchi_ecapa_enrollment_runs",                       "enroll_test_metrics.json",  "enroll_test_predictions.csv"),
    ("VBx",         REPO / "vbx_ecapa_enrollment_runs",                            "enroll_test_metrics.json",  "enroll_test_predictions.csv"),
    ("TalkNet-ASD", REPO / "video_asd_ecapa_enrollment_runs/talknet_asd",          "enroll_test_metrics.json",  "enroll_test_predictions.csv"),
    ("EEND-EDA",    REPO / "eend_eda_ecapa_enrollment_runs",                       "enroll_test_metrics.json",  "enroll_test_predictions.csv"),
    ("Sortformer",  REPO / "sortformer_ecapa_enrollment_runs",                     "enroll_test_metrics.json",  "enroll_test_predictions.csv"),
    ("WavLM-MIL",   REPO / "mil/mil_results/wavlm_mil",                            "test_metrics_tuned.json",   "test_predictions.csv"),
    ("Whisper-MIL", REPO / "mil/mil_results/whisper_mil",                          "test_metrics_tuned.json",   "test_predictions.csv"),
]

SEG_MIL_DIR = REPO / "mil/mil_results/seg_mil"
ALL_CONFIGS_PATH = SEG_MIL_DIR / "all_configs.json"

def bootstrap_metric(y_true, y_score, threshold, n_boot=1000, seed=42):
    rng = np.random.RandomState(seed)
    n = len(y_true)
    metrics = {"f1": [], "auroc": [], "auprc": [], "precision": [], "recall": []}
    for _ in range(n_boot):
        idx = rng.randint(0, n, size=n)
        yt, ys = y_true[idx], y_score[idx]
        if yt.sum() == 0 or yt.sum() == len(yt):
            continue
        yp = (ys >= threshold).astype(int)
        metrics["f1"].append(f1_score(yt, yp, zero_division=0))
        metrics["auroc"].append(roc_auc_score(yt, ys))
        metrics["auprc"].append(average_precision_score(yt, ys))
        metrics["precision"].append(precision_score(yt, yp, zero_division=0))
        metrics["recall"].append(recall_score(yt, yp, zero_division=0))
    cis = {}
    for k, vals in metrics.items():
        if vals:
            cis[f"{k}_ci_lo"] = np.percentile(vals, 2.5)
            cis[f"{k}_ci_hi"] = np.percentile(vals, 97.5)
    return cis

rows = []
for name, d, metrics_file, preds_file in DIARIZERS:
    mpath = d / metrics_file
    ppath = d / preds_file
    if not mpath.exists():
        print(f"SKIP {name}: {mpath} not found")
        continue
    with open(mpath) as f:
        m = json.load(f)
    row = {"diarizer": name,
           "f1": m.get("f1"), "precision": m.get("precision"), "recall": m.get("recall"),
           "auroc": m.get("auroc"), "auprc": m.get("auprc"),
           "threshold": m.get("threshold")}
    
    # Bootstrap CIs
    if ppath.exists():
        preds = pd.read_csv(ppath)
        # Resolve column names
        label_col = next((c for c in preds.columns if c.lower() in ("label", "y_true", "gt")), None)
        prob_col  = next((c for c in preds.columns if c.lower() in ("prob", "score", "probability")), None)
        if label_col and prob_col:
            y_true  = preds[label_col].astype(int).values
            y_score = preds[prob_col].astype(float).values
            thr = float(row["threshold"]) if row["threshold"] is not None else 0.5
            cis = bootstrap_metric(y_true, y_score, thr)
            row.update(cis)
            print(f"  {name}: F1={row['f1']:.3f} [{row.get('f1_ci_lo',float('nan')):.3f},{row.get('f1_ci_hi',float('nan')):.3f}] AUROC={row['auroc']:.3f} [{row.get('auroc_ci_lo',float('nan')):.3f},{row.get('auroc_ci_hi',float('nan')):.3f}]")
        else:
            print(f"  {name}: columns {list(preds.columns)[:5]} — can't find label/prob")
    else:
        print(f"  {name}: no predictions file at {ppath}")
    rows.append(row)

# Add best seg_mil config
if ALL_CONFIGS_PATH.exists():
    with open(ALL_CONFIGS_PATH) as f:
        all_cfgs = json.load(f)
    best = max(all_cfgs, key=lambda c: c.get("test_auroc", 0))
    best_name = f"SegMIL({best['frontend']}+{best['aggregator']})"
    seg_row = {"diarizer": best_name,
               "f1": best.get("test_f1"), "precision": best.get("test_precision"),
               "recall": best.get("test_recall"), "auroc": best.get("test_auroc"),
               "auprc": best.get("test_auprc"), "threshold": best.get("threshold")}
    # bootstrap for best seg_mil
    seg_preds_path = SEG_MIL_DIR / f"{best['frontend']}_{best['aggregator']}" / "test_predictions.csv"
    if seg_preds_path.exists():
        preds = pd.read_csv(seg_preds_path)
        label_col = next((c for c in preds.columns if c.lower() in ("label", "y_true", "gt")), None)
        prob_col  = next((c for c in preds.columns if c.lower() in ("prob", "score", "probability")), None)
        if label_col and prob_col:
            y_true  = preds[label_col].astype(int).values
            y_score = preds[prob_col].astype(float).values
            thr = float(seg_row["threshold"]) if seg_row["threshold"] is not None else 0.5
            cis = bootstrap_metric(y_true, y_score, thr)
            seg_row.update(cis)
            print(f"  {best_name}: F1={seg_row['f1']:.3f} [{seg_row.get('f1_ci_lo',float('nan')):.3f},{seg_row.get('f1_ci_hi',float('nan')):.3f}]")
    rows.append(seg_row)

# Auto-discover audio LLM baseline result folders
for llm_metrics_path in sorted(glob.glob(str(REPO / "baselines/audio_llm_baseline_runs/*/test_metrics_tuned.json"))):
    llm_path = Path(llm_metrics_path)
    model_slug = llm_path.parent.name
    try:
        with open(llm_path) as f:
            lm = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        print(f"SKIP audio_llm_{model_slug}: {e}")
        continue
    llm_row = {
        "diarizer": f"audio_llm_{model_slug}",
        "f1": lm.get("f1"), "precision": lm.get("precision"),
        "recall": lm.get("recall"), "auroc": lm.get("auroc"),
        "auprc": lm.get("auprc"), "threshold": lm.get("threshold"),
        "delta_f1_vs_babar": lm.get("delta_f1_vs_babar"),
        "delta_auroc_vs_babar": lm.get("delta_auroc_vs_babar"),
        "delta_auprc_vs_babar": lm.get("delta_auprc_vs_babar"),
    }
    llm_preds_path = llm_path.parent / "test_predictions.csv"
    if llm_preds_path.exists():
        preds = pd.read_csv(llm_preds_path)
        label_col = next((c for c in preds.columns if c.lower() in ("label", "y_true", "gt")), None)
        prob_col  = next((c for c in preds.columns if c.lower() in ("prob", "score", "probability")), None)
        if label_col and prob_col:
            y_true  = preds[label_col].dropna().astype(int).values
            y_score = preds[prob_col].dropna().astype(float).values
            if len(y_true) == len(y_score) and len(y_true) > 1:
                thr = float(llm_row["threshold"]) if llm_row["threshold"] is not None else 0.5
                cis = bootstrap_metric(y_true, y_score, thr)
                llm_row.update(cis)
                print(f"  audio_llm_{model_slug}: F1={llm_row['f1']:.3f} AUROC={llm_row['auroc']:.3f}")
    rows.append(llm_row)

df = pd.DataFrame(rows)
out_path = OUT_DIR / "cross_diarizer_master_table.csv"
df.to_csv(out_path, index=False, float_format="%.4f")
print(f"\nWrote {out_path}")
print(df[["diarizer","f1","f1_ci_lo","f1_ci_hi","auroc","auroc_ci_lo","auroc_ci_hi","auprc"]].to_string(index=False))
