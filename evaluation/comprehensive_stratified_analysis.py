#!/usr/bin/env python3
"""
Comprehensive stratified error analysis across all diarizers using SAILS annotations.
Stratifies by: task type, #adults, #children, interaction, context, location,
child constrained, video quality, vocalizations, gestures, body parts visible.
"""
import json, os, sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score, precision_score, recall_score

REPO = Path("/orcd/scratch/orcd/008/manaal/child-adult-diarization")
OUT_DIR = REPO / "evaluation" / "stratified_analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)
ANNOTATIONS_CSV = "/orcd/scratch/bcs/001/sensein/sails/BIDS_data/anotated_processed.csv"

DIARIZERS = {
    "BabAR":       (REPO / "babar_ecapa_enrollment_runs",                         "enroll_test_predictions.csv", "enroll_test_metrics.json"),
    "Pyannote":    (REPO / "pyannote/pyannote_enrollment_runs",                   "test_predictions.csv",        "test_metrics.json"),
    "USC-SAIL":    (REPO / "whisper-modeling/usc_sail_enrollment_runs",           "enroll_test_predictions.csv", "enroll_test_metrics.json"),
    "VTC":         (REPO / "vtc_ecapa_enrollment_runs",                           "enroll_test_predictions.csv", "enroll_test_metrics.json"),
    "VTC-KCHI":    (REPO / "vtc_kchi_ecapa_enrollment_runs",                      "enroll_test_predictions.csv", "enroll_test_metrics.json"),
    "VBx":         (REPO / "vbx_ecapa_enrollment_runs",                           "enroll_test_predictions.csv", "enroll_test_metrics.json"),
    "TalkNet-ASD": (REPO / "video_asd_ecapa_enrollment_runs/talknet_asd",         "enroll_test_predictions.csv", "enroll_test_metrics.json"),
    "EEND-EDA":    (REPO / "eend_eda_ecapa_enrollment_runs",                      "enroll_test_predictions.csv", "enroll_test_metrics.json"),
    "Sortformer":  (REPO / "sortformer_ecapa_enrollment_runs",                    "enroll_test_predictions.csv", "enroll_test_metrics.json"),
    "WavLM-MIL":   (REPO / "mil/mil_results/wavlm_mil",                           "test_predictions.csv",        "test_metrics_tuned.json"),
    "Whisper-MIL": (REPO / "mil/mil_results/whisper_mil",                         "test_predictions.csv",        "test_metrics_tuned.json"),
}

def bids_to_audio(s):
    if pd.isna(s): return ""
    s = str(s).strip()
    sfx = "_desc-processed_beh.mp4"
    return s[:-len(sfx)] + "_audio.wav" if s.endswith(sfx) else ""

def extract_task(path):
    fname = os.path.basename(str(path))
    return fname.split("task-")[1].split("_")[0] if "task-" in fname else "unknown"

def compute_metrics_safe(yt, ys, thr):
    yt, ys = np.asarray(yt, int), np.asarray(ys, float)
    if len(yt) < 5 or yt.sum() == 0 or yt.sum() == len(yt):
        return None
    yp = (ys >= thr).astype(int)
    try:
        return {
            "n": len(yt), "n_pos": int(yt.sum()),
            "f1": float(f1_score(yt, yp, zero_division=0)),
            "precision": float(precision_score(yt, yp, zero_division=0)),
            "recall": float(recall_score(yt, yp, zero_division=0)),
            "auroc": float(roc_auc_score(yt, ys)),
            "auprc": float(average_precision_score(yt, ys)),
            "fp": int(((yp==1)&(yt==0)).sum()),
            "fn": int(((yp==0)&(yt==1)).sum()),
        }
    except Exception:
        return None

# Load and prepare annotations
print("Loading annotations...")
ann = pd.read_csv(ANNOTATIONS_CSV, low_memory=False)
ann["audio_path"] = ann["BidsProcessed"].apply(bids_to_audio)
ann = ann[ann["audio_path"] != ""].drop_duplicates("audio_path")

# Clean up key columns
for col in ["#_adults","#_children","#_people_interacting"]:
    ann[col] = pd.to_numeric(ann[col], errors="coerce")
ann["has_interaction"] = ann["Interaction_with_child"].astype(str).str.strip().str.lower().isin(["yes","1","true"])
ann["has_vocalizations"] = ann["Vocalizations"].astype(str).str.strip().str.lower() == "yes"
ann["has_gestures"] = ann["Gestures"].astype(str).str.strip().str.lower() == "yes"
ann["n_adults_cat"] = ann["#_adults"].apply(lambda x: "0" if x==0 else "1" if x==1 else "2+" if pd.notna(x) and x>=2 else "unknown")
ann["n_children_cat"] = ann["#_children"].apply(lambda x: "1" if x==1 else "2+" if pd.notna(x) and x>=2 else "0" if x==0 else "unknown")
ann["face_vis_cat"] = ann["Video_Quality_Child_Face_Visibility"].apply(lambda x: "low(1-4)" if x<=4 else "mid(5-7)" if x<=7 else "high(8-10)" if pd.notna(x) else "unknown")
ann["lighting_cat"] = ann["Video_Quality_Lighting"].apply(lambda x: "low(1-5)" if x<=5 else "high(6-10)" if pd.notna(x) else "unknown")
ann["child_constrained"] = ann["Child_constrained"].astype(str).str.strip().str.lower().isin(["yes","constrained"])
print(f"Annotations: {len(ann)} rows")

STRATIFY_VARS = {
    "task_type":        ("extract_task", None),      # derived from audio path
    "timepoint_norm":   ("col", "timepoint_norm"),    # from preds (already there)
    "n_adults_cat":     ("ann", "n_adults_cat"),
    "n_children_cat":   ("ann", "n_children_cat"),
    "has_interaction":  ("ann", "has_interaction"),
    "has_vocalizations":("ann", "has_vocalizations"),
    "has_gestures":     ("ann", "has_gestures"),
    "context":          ("ann", "Context"),
    "location":         ("ann", "Location"),
    "child_constrained":("ann", "child_constrained"),
    "face_visibility":  ("ann", "face_vis_cat"),
    "lighting":         ("ann", "lighting_cat"),
    "body_parts":       ("ann", "Body_Parts_Visible"),
}

all_rows = []

for name, (d, preds_file, metrics_file) in DIARIZERS.items():
    ppath, mpath = d / preds_file, d / metrics_file
    if not ppath.exists():
        print(f"SKIP {name}")
        continue
    preds = pd.read_csv(ppath)
    with open(mpath) as f: m = json.load(f)
    thr = float(m.get("threshold", 0.5))

    label_col = next((c for c in preds.columns if c.lower() in ("label","y_true","gt")), None)
    prob_col  = next((c for c in preds.columns if c.lower() in ("prob","score","probability")), None)
    if not label_col or not prob_col: continue
    preds = preds.rename(columns={label_col: "label", prob_col: "prob"})

    # Merge with annotations
    if "audio_path" in preds.columns:
        cols_to_add = ["audio_path","n_adults_cat","n_children_cat","has_interaction",
                       "has_vocalizations","has_gestures","Context","Location",
                       "child_constrained","face_vis_cat","lighting_cat","Body_Parts_Visible"]
        merged = preds.merge(ann[cols_to_add], on="audio_path", how="left")
    else:
        merged = preds.copy()

    merged["task_type"] = merged.get("audio_path", pd.Series(["unknown"]*len(merged))).apply(extract_task)

    print(f"\n{name} (n={len(merged)}, thr={thr:.3f}):")

    for strat_name, (src, col) in STRATIFY_VARS.items():
        if src == "col" and col not in merged.columns: continue
        if src in ("ann","extract_task") and col not in merged.columns:
            if strat_name == "task_type":
                strat_col = "task_type"
            else:
                continue
        elif src == "col":
            strat_col = col
        else:
            strat_col = col if col in merged.columns else strat_name

        if strat_col not in merged.columns: continue

        for val, grp in merged.groupby(strat_col):
            yt = grp["label"].astype(int).values
            ys = grp["prob"].astype(float).values
            r = compute_metrics_safe(yt, ys, thr)
            if r:
                all_rows.append({
                    "diarizer": name, "stratify_by": strat_name,
                    "group": str(val), **r
                })

    # Quick print of task breakdown
    if "task_type" in merged.columns:
        for val, grp in merged.groupby("task_type"):
            yt = grp["label"].astype(int).values
            ys = grp["prob"].astype(float).values
            r = compute_metrics_safe(yt, ys, thr)
            if r:
                print(f"  task={val}: F1={r['f1']:.3f} AUROC={r['auroc']:.3f} n={r['n']}")

df = pd.DataFrame(all_rows)
out = OUT_DIR / "stratified_metrics_all_diarizers.csv"
df.to_csv(out, index=False, float_format="%.4f")
print(f"\nWrote {out} ({len(df)} rows)")

# Pivot: for each stratify_by × group, show F1 for all diarizers side by side
for strat_name in df["stratify_by"].unique():
    sub = df[df["stratify_by"]==strat_name][["diarizer","group","f1","auroc","n"]].copy()
    pivot = sub.pivot_table(index="group", columns="diarizer", values="f1", aggfunc="first")
    pivot_path = OUT_DIR / f"pivot_f1_by_{strat_name}.csv"
    pivot.to_csv(pivot_path, float_format="%.3f")
    print(f"  Pivot: {pivot_path.name}")

print("\nDone.")
