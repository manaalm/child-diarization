#!/usr/bin/env python3
"""
Double-stratification analysis: interaction effects between annotation factors.

Analyses:
  1. Vocalizations × Interaction: does adult-child interaction change error rates
     differently when the child is vocalizing vs. non-vocalizing?
  2. Face visibility × diarizer type: audio vs. video diarizers — does face
     visibility predict performance for video-only but not audio-only models?
  3. N_children × N_adults: combinatorial scene complexity
  4. Timepoint × Task: developmental × situational interaction

Outputs (evaluation/double_stratification/):
  vocalizations_x_interaction.csv
  face_visibility_x_diarizer_type.csv
  n_children_x_n_adults.csv
  timepoint_x_task.csv
  summary_interaction_effects.csv
"""
import json, os
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score

REPO = Path("/orcd/scratch/orcd/008/manaal/child-adult-diarization")
OUT_DIR = REPO / "evaluation" / "double_stratification"
OUT_DIR.mkdir(parents=True, exist_ok=True)
ANNOTATIONS_CSV = "/orcd/scratch/bcs/001/sensein/sails/BIDS_data/anotated_processed.csv"

DIARIZERS = {
    "BabAR":       (REPO / "babar_ecapa_enrollment_runs",
                    "enroll_test_predictions.csv", "enroll_test_metrics.json"),
    "Pyannote":    (REPO / "pyannote/pyannote_enrollment_runs",
                    "test_predictions.csv",        "test_metrics.json"),
    "USC-SAIL":    (REPO / "whisper-modeling/usc_sail_enrollment_runs",
                    "enroll_test_predictions.csv", "enroll_test_metrics.json"),
    "VTC":         (REPO / "vtc_ecapa_enrollment_runs",
                    "enroll_test_predictions.csv", "enroll_test_metrics.json"),
    "VTC-KCHI":    (REPO / "vtc_kchi_ecapa_enrollment_runs",
                    "enroll_test_predictions.csv", "enroll_test_metrics.json"),
    "VBx":         (REPO / "vbx_ecapa_enrollment_runs",
                    "enroll_test_predictions.csv", "enroll_test_metrics.json"),
    "TalkNet-ASD": (REPO / "video_asd_ecapa_enrollment_runs/talknet_asd",
                    "enroll_test_predictions.csv", "enroll_test_metrics.json"),
    "EEND-EDA":    (REPO / "eend_eda_ecapa_enrollment_runs",
                    "enroll_test_predictions.csv", "enroll_test_metrics.json"),
    "Sortformer":  (REPO / "sortformer_ecapa_enrollment_runs",
                    "enroll_test_predictions.csv", "enroll_test_metrics.json"),
    "WavLM-MIL":   (REPO / "mil/mil_results/wavlm_mil",
                    "test_predictions.csv",        "test_metrics_tuned.json"),
    "Whisper-MIL": (REPO / "mil/mil_results/whisper_mil",
                    "test_predictions.csv",        "test_metrics_tuned.json"),
}

# Diarizer type labels
DIARIZER_TYPE = {
    "BabAR": "audio", "Pyannote": "audio", "USC-SAIL": "audio",
    "VTC": "audio", "VTC-KCHI": "audio", "VBx": "audio",
    "TalkNet-ASD": "video", "EEND-EDA": "audio", "Sortformer": "audio",
    "WavLM-MIL": "audio_mil", "Whisper-MIL": "audio_mil",
}


def bids_to_audio(s):
    if pd.isna(s): return ""
    s = str(s).strip()
    sfx = "_desc-processed_beh.mp4"
    return s[:-len(sfx)] + "_audio.wav" if s.endswith(sfx) else ""


def safe_metrics(yt, ys, thr, min_n=5, require_diversity=True):
    """Compute metrics; when require_diversity=False, allow single-class cells."""
    yt, ys = np.asarray(yt, int), np.asarray(ys, float)
    if len(yt) < min_n:
        return None
    if require_diversity and (yt.sum() == 0 or yt.sum() == len(yt)):
        return None
    yp = (ys >= thr).astype(int)
    n_pos = int(yt.sum())
    n_neg = len(yt) - n_pos
    fp = int(((yp==1)&(yt==0)).sum())
    fn = int(((yp==0)&(yt==1)).sum())
    base = {
        "n": len(yt), "n_pos": n_pos, "n_neg": n_neg,
        "fp": fp, "fn": fn,
        "fp_rate": round(fp / max(n_neg, 1), 4),
        "fn_rate": round(fn / max(n_pos, 1), 4),
        "accuracy": round((len(yt) - fp - fn) / len(yt), 4),
    }
    if yt.sum() == 0 or yt.sum() == len(yt):
        # Can't compute rank metrics without both classes
        base.update({"f1": float("nan"), "auroc": float("nan")})
        return base
    try:
        base["f1"] = round(float(f1_score(yt, yp, zero_division=0)), 4)
        base["auroc"] = round(float(roc_auc_score(yt, ys)), 4)
    except Exception:
        base.update({"f1": float("nan"), "auroc": float("nan")})
    return base


# ------------------------------------------------------------------ load data --

print("Loading annotations...")
ann = pd.read_csv(ANNOTATIONS_CSV, low_memory=False)
ann["audio_path"] = ann["BidsProcessed"].apply(bids_to_audio)
ann = ann[ann["audio_path"] != ""].drop_duplicates("audio_path")
for col in ["#_adults", "#_children"]:
    if col in ann.columns:
        ann[col] = pd.to_numeric(ann[col], errors="coerce")

ann["has_interaction"] = ann["Interaction_with_child"].astype(str).str.strip().str.lower().isin(["yes","1","true"])
ann["has_vocalizations"] = ann["Vocalizations"].astype(str).str.strip().str.lower() == "yes"
ann["face_vis_cat"] = ann["Video_Quality_Child_Face_Visibility"].apply(
    lambda x: "low(1-4)" if pd.notna(x) and x<=4 else "mid(5-7)" if pd.notna(x) and x<=7 else "high(8-10)" if pd.notna(x) else "unknown"
)
ann["n_children_cat"] = ann["#_children"].apply(
    lambda x: "0" if x==0 else "1" if x==1 else "2+" if pd.notna(x) and x>=2 else "unknown"
)
ann["n_adults_cat"] = ann["#_adults"].apply(
    lambda x: "0" if x==0 else "1" if x==1 else "2+" if pd.notna(x) and x>=2 else "unknown"
)

ann["has_gestures"] = ann["Gestures"].astype(str).str.strip().str.lower() == "yes"
ann["context_type"] = ann["Context"].astype(str).str.strip().str.lower().apply(
    lambda x: "private" if "private" in x else "public" if "public" in x else "other"
)
ann_slim = ann[["audio_path","has_interaction","has_gestures","context_type","face_vis_cat",
                "n_children_cat","n_adults_cat"]].copy()
print(f"Annotations: {len(ann)} rows")

print("Loading predictions...")
all_loaded = {}
for name, (run_dir, preds_file, metrics_file) in DIARIZERS.items():
    ppath = run_dir / preds_file
    if not ppath.exists():
        continue
    df = pd.read_csv(ppath)
    col_map = {}
    for col in df.columns:
        lc = col.lower()
        if lc in ("label","y_true","gt") and "label" not in col_map: col_map[col] = "label"
        elif lc in ("prob","score","probability") and "prob" not in col_map: col_map[col] = "prob"
        elif lc in ("pred_label","prediction") and "pred_label" not in col_map: col_map[col] = "pred_label"
    df = df.rename(columns=col_map)
    thr = 0.5
    mpath = run_dir / metrics_file
    if mpath.exists():
        with open(mpath) as f: m = json.load(f)
        if "threshold" in m: thr = float(m["threshold"])
    if "pred_label" not in df.columns:
        df["pred_label"] = (df["prob"] >= thr).astype(int)
    df["label"] = df["label"].astype(int)
    df["prob"] = df["prob"].astype(float)
    df["threshold"] = thr
    df = df.merge(ann_slim, on="audio_path", how="left")
    all_loaded[name] = df

print(f"Loaded {len(all_loaded)} diarizers")


# ================================================================
# Analysis helper
# ================================================================

def double_stratify(dfs, factor_a, factor_b, diarizer_type_col=None, require_diversity=True):
    """Compute metrics for each (diarizer, factor_a_val, factor_b_val) cell."""
    rows = []
    for name, df in dfs.items():
        if factor_a not in df.columns or factor_b not in df.columns:
            continue
        thr = df["threshold"].iloc[0]
        for (va, vb), sub in df.groupby([factor_a, factor_b]):
            m = safe_metrics(sub["label"].values, sub["prob"].values, thr,
                             require_diversity=require_diversity)
            if m is None:
                continue
            row = {"diarizer": name, factor_a: va, factor_b: vb, **m}
            if diarizer_type_col:
                row["diarizer_type"] = DIARIZER_TYPE.get(name, "other")
            rows.append(row)
    return pd.DataFrame(rows)


# ================================================================
# 1. Vocalizations × Interaction
# ================================================================

print("\nAnalysis 1: Gestures × Interaction...")
gest_int = double_stratify(all_loaded, "has_gestures", "has_interaction")
gest_int.to_csv(OUT_DIR / "gestures_x_interaction.csv", index=False)

if not gest_int.empty and "f1" in gest_int.columns:
    print("\nGestures × Interaction (mean F1 across diarizers):")
    gest_summary = (
        gest_int.groupby(["has_gestures","has_interaction"])["f1"]
        .agg(["mean","std","min","max"])
        .round(4)
    )
    print(gest_summary.to_string())
    print("\nInteraction effect (does gesture/no-gesture gap differ by interaction?):")
    for inter in [True, False]:
        sub = gest_int[gest_int["has_interaction"] == inter]
        gest_mean = sub[sub["has_gestures"] == True]["f1"].mean()
        nogest_mean = sub[sub["has_gestures"] == False]["f1"].mean()
        print(f"  interaction={inter}: F1(gestures)={gest_mean:.3f}, "
              f"F1(no-gestures)={nogest_mean:.3f}, gap={gest_mean - nogest_mean:+.3f}")

print("\nAnalysis 1b: Context × Interaction...")
ctx_int = double_stratify(all_loaded, "context_type", "has_interaction")
ctx_int.to_csv(OUT_DIR / "context_x_interaction.csv", index=False)

if not ctx_int.empty and "f1" in ctx_int.columns:
    print("\nContext × Interaction (mean F1 across diarizers):")
    ctx_summary = (
        ctx_int.groupby(["context_type","has_interaction"])["f1"]
        .agg(["mean","std","count"])
        .round(4)
    )
    print(ctx_summary.to_string())


# ================================================================
# 2. Face visibility × diarizer type
# ================================================================

print("\nAnalysis 2: Face visibility × diarizer type...")
fv = double_stratify(all_loaded, "face_vis_cat", "has_interaction", diarizer_type_col=True, require_diversity=False)
fv.to_csv(OUT_DIR / "face_visibility_x_interaction.csv", index=False)

fv["diarizer_type"] = fv["diarizer"].map(DIARIZER_TYPE)
fv_type_summary = (
    fv.groupby(["diarizer_type","face_vis_cat"])["f1"]
    .agg(["mean","std","count"])
    .round(4)
)
print("\nFace visibility × diarizer type (mean F1):")
print(fv_type_summary.to_string())

# Direct face-vis × diarizer type pivot
fv_pivot = fv.pivot_table(
    index="face_vis_cat", columns="diarizer_type", values="f1", aggfunc="mean"
).round(4)
fv_pivot.to_csv(OUT_DIR / "face_visibility_x_diarizer_type.csv")
print("\nPivot (face_vis_cat × diarizer_type):")
print(fv_pivot.to_string())


# ================================================================
# 3. N_children × N_adults
# ================================================================

print("\nAnalysis 3: N_children × N_adults...")
nc_na = double_stratify(all_loaded, "n_children_cat", "n_adults_cat", require_diversity=False)
nc_na.to_csv(OUT_DIR / "n_children_x_n_adults.csv", index=False)

nc_na_summary = (
    nc_na.groupby(["n_children_cat","n_adults_cat"])["f1"]
    .agg(["mean","std","count"])
    .round(4)
)
print("\nN_children × N_adults (mean F1 across diarizers):")
print(nc_na_summary.to_string())


# ================================================================
# 4. Timepoint × Face visibility
# ================================================================

print("\nAnalysis 4: Timepoint × Face visibility...")
if "timepoint_norm" in list(all_loaded.values())[0].columns:
    tp_fv = []
    for name, df in all_loaded.items():
        if "timepoint_norm" not in df.columns or "face_vis_cat" not in df.columns:
            continue
        thr = df["threshold"].iloc[0]
        for (tp, fvc), sub in df.groupby(["timepoint_norm","face_vis_cat"]):
            m = safe_metrics(sub["label"].values, sub["prob"].values, thr)
            if m:
                tp_fv.append({"diarizer": name, "timepoint_norm": tp, "face_vis_cat": fvc, **m})
    tp_fv_df = pd.DataFrame(tp_fv)
    tp_fv_df.to_csv(OUT_DIR / "timepoint_x_face_visibility.csv", index=False)
    summary = tp_fv_df.groupby(["timepoint_norm","face_vis_cat"])["f1"].agg(["mean","std"]).round(4)
    print("\nTimepoint × face visibility (mean F1):")
    print(summary.to_string())


# ================================================================
# Summary interaction effects table
# ================================================================

summary_rows = []

# Gestures × Interaction effect size
if not gest_int.empty and "f1" in gest_int.columns:
    for diar in gest_int["diarizer"].unique():
        sub = gest_int[gest_int["diarizer"] == diar]
        for inter in [True, False]:
            s2 = sub[sub["has_interaction"] == inter]
            if len(s2) < 2: continue
            gest_f1 = s2[s2["has_gestures"] == True]["f1"].values
            nogest_f1 = s2[s2["has_gestures"] == False]["f1"].values
            if len(gest_f1) > 0 and len(nogest_f1) > 0:
                summary_rows.append({
                    "analysis": "gestures_x_interaction",
                    "diarizer": diar,
                    "condition": f"interaction={inter}",
                    "f1_cell_a": round(float(gest_f1[0]), 4),
                    "f1_cell_b": round(float(nogest_f1[0]), 4),
                    "effect_size": round(float(gest_f1[0]) - float(nogest_f1[0]), 4),
                })

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(OUT_DIR / "summary_interaction_effects.csv", index=False)

print(f"\nAll outputs written to {OUT_DIR}")
for f in sorted(OUT_DIR.iterdir()):
    print(f"  {f.name} ({f.stat().st_size // 1024} KB)")
