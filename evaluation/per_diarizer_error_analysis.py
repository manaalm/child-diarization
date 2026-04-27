#!/usr/bin/env python3
"""
Cross-diarizer error analysis: FP/FN breakdown by task type, interaction, n_children, age group.
Extends the pyannote_error_analysis.py pattern to all diarizers simultaneously.
"""
import json, os, sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report

REPO = Path("/orcd/scratch/orcd/008/manaal/child-adult-diarization")
OUT_DIR = REPO / "evaluation" / "per_diarizer_error_analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)
ANNOTATIONS_CSV = "/orcd/scratch/bcs/001/sensein/sails/BIDS_data/anotated_processed.csv"

DIARIZERS = {
    "BabAR":       (REPO / "babar_ecapa_enrollment_runs",                         "enroll_test_predictions.csv"),
    "Pyannote":    (REPO / "pyannote/pyannote_enrollment_runs",                   "test_predictions.csv"),
    "USC-SAIL":    (REPO / "whisper-modeling/usc_sail_enrollment_runs",           "enroll_test_predictions.csv"),
    "VTC":         (REPO / "vtc_ecapa_enrollment_runs",                           "enroll_test_predictions.csv"),
    "VTC-KCHI":    (REPO / "vtc_kchi_ecapa_enrollment_runs",                      "enroll_test_predictions.csv"),
    "VBx":         (REPO / "vbx_ecapa_enrollment_runs",                           "enroll_test_predictions.csv"),
    "TalkNet-ASD": (REPO / "video_asd_ecapa_enrollment_runs/talknet_asd",         "enroll_test_predictions.csv"),
    "EEND-EDA":    (REPO / "eend_eda_ecapa_enrollment_runs",                      "enroll_test_predictions.csv"),
    "Sortformer":  (REPO / "sortformer_ecapa_enrollment_runs",                    "enroll_test_predictions.csv"),
    "WavLM-MIL":   (REPO / "mil/mil_results/wavlm_mil",                           "test_predictions.csv"),
    "Whisper-MIL": (REPO / "mil/mil_results/whisper_mil",                         "test_predictions.csv"),
}

def bids_to_audio_path(s):
    if pd.isna(s): return ""
    s = str(s).strip()
    suffix = "_desc-processed_beh.mp4"
    if s.endswith(suffix):
        return s[:-len(suffix)] + "_audio.wav"
    return ""

def extract_task(audio_path):
    fname = os.path.basename(str(audio_path))
    if "task-" in fname:
        return fname.split("task-")[1].split("_")[0]
    return "unknown"

# Load annotations once
print("Loading annotations...")
ann = pd.read_csv(ANNOTATIONS_CSV, low_memory=False)
ann["audio_path"] = ann["BidsProcessed"].apply(bids_to_audio_path)
for col in ["#_adults", "#_children", "#_people_background", "#_people_interacting"]:
    if col in ann.columns:
        ann[col] = pd.to_numeric(ann[col], errors="coerce")
if "Interaction_with_child" in ann.columns:
    ann["interaction"] = ann["Interaction_with_child"].astype(str).str.strip().str.lower()
    ann["has_interaction"] = ann["interaction"].isin(["yes", "1", "true"])
ann = ann[ann["audio_path"] != ""].drop_duplicates("audio_path")
print(f"Annotations: {len(ann)} rows")

summary_rows = []

for name, (d, preds_file) in DIARIZERS.items():
    ppath = d / preds_file
    if not ppath.exists():
        print(f"SKIP {name}: {ppath} not found")
        continue
    preds = pd.read_csv(ppath)
    label_col = next((c for c in preds.columns if c.lower() in ("label", "y_true", "gt")), None)
    prob_col  = next((c for c in preds.columns if c.lower() in ("prob", "score", "probability")), None)
    if not label_col or not prob_col:
        print(f"SKIP {name}: can't find label/prob columns in {list(preds.columns)[:6]}")
        continue

    # Load val-tuned threshold
    thr = 0.5
    for mname in ("enroll_test_metrics.json", "test_metrics_tuned.json", "test_metrics.json"):
        mp = d / mname
        if mp.exists():
            with open(mp) as f:
                m = json.load(f)
            if "threshold" in m:
                thr = float(m["threshold"])
            break

    preds = preds.rename(columns={label_col: "label", prob_col: "prob"})
    preds["pred_label"] = (preds["prob"] >= thr).astype(int)
    preds["outcome"] = preds.apply(
        lambda r: "TP" if r["label"]==1 and r["pred_label"]==1 else
                  "TN" if r["label"]==0 and r["pred_label"]==0 else
                  "FP" if r["label"]==0 and r["pred_label"]==1 else "FN", axis=1)

    # Merge with annotations
    if "audio_path" in preds.columns:
        merged = preds.merge(ann[["audio_path","#_adults","#_children","has_interaction"]],
                             on="audio_path", how="left")
    else:
        merged = preds.copy()

    # Add task type
    if "audio_path" in merged.columns:
        merged["task_type"] = merged["audio_path"].apply(extract_task)
    
    n = len(merged)
    fp = (merged["outcome"] == "FP").sum()
    fn = (merged["outcome"] == "FN").sum()
    tp = (merged["outcome"] == "TP").sum()
    tn = (merged["outcome"] == "TN").sum()
    
    row = {"diarizer": name, "n": n, "tp": tp, "tn": tn, "fp": fp, "fn": fn,
           "fp_rate": round(fp / max(tp+tn+fp+fn, 1), 4),
           "fn_rate": round(fn / max(tp+tn+fp+fn, 1), 4),
           "threshold": thr}
    
    # FP/FN by has_interaction
    if "has_interaction" in merged.columns:
        for val, label_str in [(True, "interactive"), (False, "non_interactive")]:
            sub = merged[merged["has_interaction"] == val]
            row[f"fp_{label_str}"] = (sub["outcome"] == "FP").sum()
            row[f"fn_{label_str}"] = (sub["outcome"] == "FN").sum()
    
    # FP/FN by timepoint
    if "timepoint_norm" in merged.columns:
        for tp_label in merged["timepoint_norm"].dropna().unique():
            sub = merged[merged["timepoint_norm"] == tp_label]
            clean = str(tp_label).replace("_", "")
            row[f"fp_{clean}"] = (sub["outcome"] == "FP").sum()
            row[f"fn_{clean}"] = (sub["outcome"] == "FN").sum()
    
    # FP/FN by n_children
    if "#_children" in merged.columns:
        for nc in [0, 1, 2]:
            sub = merged[merged["#_children"] == nc]
            row[f"fp_nc{nc}"] = (sub["outcome"] == "FP").sum()
            row[f"fn_nc{nc}"] = (sub["outcome"] == "FN").sum()
    
    summary_rows.append(row)
    
    # Save per-diarizer FP/FN files
    dout = OUT_DIR / name
    dout.mkdir(exist_ok=True)
    merged[merged["outcome"].isin(["FP","FN"])].to_csv(dout / "errors.csv", index=False)
    
    # Task breakdown
    if "task_type" in merged.columns:
        task_err = merged.groupby(["task_type","outcome"]).size().unstack(fill_value=0)
        task_err.to_csv(dout / "errors_by_task.csv")
    
    print(f"  {name}: FP={fp}, FN={fn} (threshold={thr:.3f})")

# Save summary
summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(OUT_DIR / "summary.csv", index=False)
print(f"\nWrote {OUT_DIR / 'summary.csv'}")
print(summary_df[["diarizer","n","tp","tn","fp","fn","fp_rate","fn_rate"]].to_string(index=False))
