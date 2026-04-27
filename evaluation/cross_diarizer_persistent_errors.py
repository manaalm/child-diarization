#!/usr/bin/env python3
"""
Cross-diarizer persistent error analysis — enrollment-diarizer equivalent of
baselines/error_analysis.py's cross_experiment_analysis().

Covers:
  1. Persistent FPs/FNs — clips misclassified by multiple diarizers (inherently hard)
  2. Pairwise prediction agreement matrix (11×11)
  3. Unique contributions — what each diarizer gets right that majority miss
  4. Per-child error rates for all 11 diarizers
  5. Confidence calibration per diarizer
  6. Metadata patterns in persistent errors (SAILS annotations)
  7. Multi-child × interaction cross-tab per diarizer

Outputs (all under evaluation/cross_diarizer_errors/):
  persistent_false_positives.csv
  persistent_false_negatives.csv
  pairwise_agreement.csv           (long format)
  pairwise_agreement_matrix.csv    (wide pivot for easy reading)
  unique_contributions.csv
  per_child_error_rates_all_diarizers.csv
  per_child_mean_accuracy.csv
  confidence_calibration_all_diarizers.csv
  multi_child_interaction_crosstab_all_diarizers.csv
  cross_diarizer_report.txt
"""
import json, os, sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score

REPO = Path("/orcd/scratch/orcd/008/manaal/child-adult-diarization")
OUT_DIR = REPO / "evaluation" / "cross_diarizer_errors"
OUT_DIR.mkdir(parents=True, exist_ok=True)
ANNOTATIONS_CSV = "/orcd/scratch/bcs/001/sensein/sails/BIDS_data/anotated_processed.csv"

# ---- Diarizer registry ----
# (run_dir, predictions_file, metrics_file_for_threshold)
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


# ------------------------------------------------------------------ helpers --

def bids_to_audio(s):
    if pd.isna(s): return ""
    s = str(s).strip()
    sfx = "_desc-processed_beh.mp4"
    return s[:-len(sfx)] + "_audio.wav" if s.endswith(sfx) else ""


def extract_task(path):
    fname = os.path.basename(str(path))
    return fname.split("task-")[1].split("_")[0] if "task-" in fname else "unknown"


def load_preds(name, run_dir, preds_file, metrics_file):
    """Return standardised DataFrame with columns: audio_path, child_id,
    timepoint_norm, label, prob, pred_label, outcome, diarizer."""
    ppath = run_dir / preds_file
    if not ppath.exists():
        print(f"  SKIP {name}: {ppath} not found")
        return None

    df = pd.read_csv(ppath)

    # Normalise column names (MIL uses 'score'/'prediction')
    col_map = {}
    for col in df.columns:
        lc = col.lower()
        if lc in ("label", "y_true", "gt") and "label" not in col_map:
            col_map[col] = "label"
        elif lc in ("prob", "score", "probability", "enrollment_score") and "prob" not in col_map:
            col_map[col] = "prob"
        elif lc in ("pred_label", "prediction") and "pred_label" not in col_map:
            col_map[col] = "pred_label"
    df = df.rename(columns=col_map)

    if "label" not in df.columns or "prob" not in df.columns:
        print(f"  SKIP {name}: can't find label/prob columns ({list(df.columns)[:6]})")
        return None

    # Load val-tuned threshold (for confidence calibration; use stored pred_label for outcomes)
    thr = 0.5
    mpath = run_dir / metrics_file
    if mpath.exists():
        with open(mpath) as f:
            m = json.load(f)
        if "threshold" in m:
            thr = float(m["threshold"])

    if "pred_label" not in df.columns:
        df["pred_label"] = (df["prob"] >= thr).astype(int)

    df["label"] = df["label"].astype(int)
    df["prob"]  = df["prob"].astype(float)
    df["pred_label"] = df["pred_label"].astype(int)

    df["outcome"] = np.select(
        [(df["label"]==1) & (df["pred_label"]==1),
         (df["label"]==0) & (df["pred_label"]==0),
         (df["label"]==0) & (df["pred_label"]==1),
         (df["label"]==1) & (df["pred_label"]==0)],
        ["TP", "TN", "FP", "FN"], default="?"
    )
    df["diarizer"] = name
    df["threshold"] = thr
    return df


# ------------------------------------------------------------------ load all -

print("Loading annotations...")
ann = pd.read_csv(ANNOTATIONS_CSV, low_memory=False)
ann["audio_path"] = ann["BidsProcessed"].apply(bids_to_audio)
ann = ann[ann["audio_path"] != ""].drop_duplicates("audio_path")

for col in ["#_adults", "#_children", "#_people_background", "#_people_interacting"]:
    if col in ann.columns:
        ann[col] = pd.to_numeric(ann[col], errors="coerce")

ann["has_interaction"] = (
    ann["Interaction_with_child"].astype(str).str.strip().str.lower()
    .isin(["yes", "1", "true"])
)
ann["n_children_cat"] = ann["#_children"].apply(
    lambda x: "0" if x == 0 else "1" if x == 1 else "2+" if pd.notna(x) and x >= 2 else "unknown"
)
ann["n_adults_cat"] = ann["#_adults"].apply(
    lambda x: "0" if x == 0 else "1" if x == 1 else "2+" if pd.notna(x) and x >= 2 else "unknown"
)
print(f"Annotations: {len(ann)} rows")

META_COLS = ["audio_path", "#_adults", "#_children", "#_people_interacting",
             "has_interaction", "n_children_cat", "n_adults_cat",
             "Context", "Location"]
ann_slim = ann[[c for c in META_COLS if c in ann.columns]].copy()

print("Loading diarizer predictions...")
all_dfs = {}
for name, (run_dir, preds_file, metrics_file) in DIARIZERS.items():
    df = load_preds(name, run_dir, preds_file, metrics_file)
    if df is not None:
        all_dfs[name] = df
        print(f"  {name}: {len(df)} clips loaded")

diarizer_names = list(all_dfs.keys())
n_diarizers = len(diarizer_names)
print(f"\nLoaded {n_diarizers} diarizers.")

lines = []
def p(msg=""): lines.append(msg)


# ================================================================ 1. PERSISTENT FPs/FNs ==

p("=" * 80)
p(f"CROSS-DIARIZER PERSISTENT ERROR ANALYSIS ({n_diarizers} diarizers)")
p("=" * 80)

# Collect per-clip outcome across diarizers
p("\n" + "-" * 60)
p("1. PERSISTENT FALSE POSITIVES")
p("-" * 60)

fp_counts = {}
fp_diarizer_map = {}
fn_counts = {}
fn_diarizer_map = {}

for name, df in all_dfs.items():
    for path in df[df["outcome"] == "FP"]["audio_path"].tolist():
        fp_counts[path] = fp_counts.get(path, 0) + 1
        fp_diarizer_map.setdefault(path, []).append(name)
    for path in df[df["outcome"] == "FN"]["audio_path"].tolist():
        fn_counts[path] = fn_counts.get(path, 0) + 1
        fn_diarizer_map.setdefault(path, []).append(name)

# Build persistent-FP table
fp_rows = []
for path, count in fp_counts.items():
    fp_rows.append({
        "audio_path": path,
        "n_diarizers_fp": count,
        "pct_diarizers": round(count / n_diarizers, 3),
        "diarizers": ", ".join(sorted(fp_diarizer_map[path])),
    })
fp_persist = (
    pd.DataFrame(fp_rows)
    .sort_values("n_diarizers_fp", ascending=False)
    .reset_index(drop=True)
)

# Enrich with metadata
ref_df = list(all_dfs.values())[0]
fp_persist = fp_persist.merge(ann_slim, on="audio_path", how="left")
if "audio_path" in ref_df.columns:
    child_meta = ref_df[["audio_path", "child_id", "timepoint_norm"]].drop_duplicates()
    fp_persist = fp_persist.merge(child_meta, on="audio_path", how="left")
fp_persist["task"] = fp_persist["audio_path"].apply(extract_task)
fp_persist.to_csv(OUT_DIR / "persistent_false_positives.csv", index=False)

always_fp  = fp_persist[fp_persist["n_diarizers_fp"] == n_diarizers]
majority_fp = fp_persist[fp_persist["n_diarizers_fp"] >= n_diarizers / 2]
any_fp = fp_persist

p(f"\nTotal unique FP clips across all diarizers: {len(any_fp)}")
p(f"FP in ALL {n_diarizers} diarizers: {len(always_fp)}")
p(f"FP in >=half of diarizers (>={n_diarizers//2}): {len(majority_fp)}")

if len(majority_fp) > 0:
    p("\nPersistent FP metadata:")
    if "#_children" in majority_fp.columns:
        p(f"  Mean #_children: {majority_fp['#_children'].mean():.2f}")
        p(f"  Multi-child (>1): {(majority_fp['#_children'] > 1).sum()} / {len(majority_fp)}")
    if "#_adults" in majority_fp.columns:
        p(f"  Mean #_adults: {majority_fp['#_adults'].mean():.2f}")
    if "has_interaction" in majority_fp.columns:
        p(f"  Has interaction: {majority_fp['has_interaction'].sum()} / {len(majority_fp)}")
    if "timepoint_norm" in majority_fp.columns:
        p(f"  Timepoint distribution:\n    "
          + majority_fp["timepoint_norm"].value_counts().to_string().replace("\n", "\n    "))
    if "task" in majority_fp.columns:
        p(f"  Task distribution:\n    "
          + majority_fp["task"].value_counts().head(8).to_string().replace("\n", "\n    "))
    p(f"\n  Top 15 persistent FPs by diarizer count:")
    disp_cols = ["audio_path", "n_diarizers_fp", "diarizers"]
    for c in ["child_id", "timepoint_norm", "task", "#_children", "#_adults", "has_interaction"]:
        if c in majority_fp.columns:
            disp_cols.append(c)
    p(majority_fp[disp_cols].head(15).to_string(index=False))

# Build persistent-FN table
p("\n" + "-" * 60)
p("2. PERSISTENT FALSE NEGATIVES")
p("-" * 60)

fn_rows = []
for path, count in fn_counts.items():
    fn_rows.append({
        "audio_path": path,
        "n_diarizers_fn": count,
        "pct_diarizers": round(count / n_diarizers, 3),
        "diarizers": ", ".join(sorted(fn_diarizer_map[path])),
    })
fn_persist = (
    pd.DataFrame(fn_rows)
    .sort_values("n_diarizers_fn", ascending=False)
    .reset_index(drop=True)
)
fn_persist = fn_persist.merge(ann_slim, on="audio_path", how="left")
if "audio_path" in ref_df.columns:
    fn_persist = fn_persist.merge(child_meta, on="audio_path", how="left")
fn_persist["task"] = fn_persist["audio_path"].apply(extract_task)
fn_persist.to_csv(OUT_DIR / "persistent_false_negatives.csv", index=False)

always_fn   = fn_persist[fn_persist["n_diarizers_fn"] == n_diarizers]
majority_fn = fn_persist[fn_persist["n_diarizers_fn"] >= n_diarizers / 2]

p(f"\nTotal unique FN clips across all diarizers: {len(fn_persist)}")
p(f"FN in ALL {n_diarizers} diarizers: {len(always_fn)}")
p(f"FN in >=half of diarizers (>={n_diarizers//2}): {len(majority_fn)}")

if len(majority_fn) > 0:
    p("\nPersistent FN metadata:")
    if "#_children" in majority_fn.columns:
        p(f"  Mean #_children: {majority_fn['#_children'].mean():.2f}")
    if "#_adults" in majority_fn.columns:
        p(f"  Mean #_adults: {majority_fn['#_adults'].mean():.2f}")
    if "has_interaction" in majority_fn.columns:
        p(f"  Has interaction: {majority_fn['has_interaction'].sum()} / {len(majority_fn)}")
    if "timepoint_norm" in majority_fn.columns:
        p(f"  Timepoint distribution:\n    "
          + majority_fn["timepoint_norm"].value_counts().to_string().replace("\n", "\n    "))
    if "task" in majority_fn.columns:
        p(f"  Task distribution:\n    "
          + majority_fn["task"].value_counts().head(8).to_string().replace("\n", "\n    "))
    fn_disp_cols = ["audio_path", "n_diarizers_fn", "diarizers"]
    for c in ["child_id", "timepoint_norm", "task", "#_children", "#_adults", "has_interaction"]:
        if c in fn_persist.columns:
            fn_disp_cols.append(c)
    p(f"\n  Top 15 persistent FNs:")
    p(fn_persist[fn_disp_cols].head(15).to_string(index=False))


# ================================================================ 3. PAIRWISE AGREEMENT ==

p("\n" + "-" * 60)
p("3. PAIRWISE PREDICTION AGREEMENT")
p("-" * 60)
p("(Fraction of test clips where two diarizers agree on the binary prediction)")

agree_rows = []
for i, name_a in enumerate(diarizer_names):
    for name_b in diarizer_names[i + 1:]:
        df_a = all_dfs[name_a][["audio_path", "pred_label"]].rename(columns={"pred_label": "pa"})
        df_b = all_dfs[name_b][["audio_path", "pred_label"]].rename(columns={"pred_label": "pb"})
        merged = df_a.merge(df_b, on="audio_path")
        n = len(merged)
        n_agree = (merged["pa"] == merged["pb"]).sum()
        agree_rows.append({
            "diarizer_a": name_a,
            "diarizer_b": name_b,
            "n_clips": n,
            "n_agree": int(n_agree),
            "agreement": round(n_agree / n, 4) if n > 0 else float("nan"),
        })

agree_df = pd.DataFrame(agree_rows).sort_values("agreement")
agree_df.to_csv(OUT_DIR / "pairwise_agreement.csv", index=False)

# Also write a symmetric matrix for easy reading
matrix_data = {a: {} for a in diarizer_names}
for _, row in agree_df.iterrows():
    a, b, v = row["diarizer_a"], row["diarizer_b"], row["agreement"]
    matrix_data[a][b] = v
    matrix_data[b][a] = v
for name in diarizer_names:
    matrix_data[name][name] = 1.0
matrix_df = pd.DataFrame(matrix_data, index=diarizer_names, columns=diarizer_names)
matrix_df.to_csv(OUT_DIR / "pairwise_agreement_matrix.csv")

p("\nLowest agreement pairs (most diverse):")
p(agree_df.head(10).to_string(index=False))
p("\nHighest agreement pairs (most redundant):")
p(agree_df.tail(10).to_string(index=False))
p("\nAgreement matrix (rows = diarizer_a):")
p(matrix_df.round(3).to_string())


# ================================================================ 4. UNIQUE CONTRIBUTIONS ==

p("\n" + "-" * 60)
p("4. UNIQUE CONTRIBUTIONS")
p("-" * 60)
p("(Clips correct in THIS diarizer but wrong in majority of others)")

contrib_rows = []
for name in diarizer_names:
    df_me = all_dfs[name]
    correct_here = set(df_me[df_me["outcome"].isin(["TP", "TN"])]["audio_path"])

    # Count errors per clip from other diarizers
    other_names = [n for n in diarizer_names if n != name]
    error_count = {}
    for other in other_names:
        df_other = all_dfs[other]
        for path in df_other[df_other["outcome"].isin(["FP", "FN"])]["audio_path"]:
            error_count[path] = error_count.get(path, 0) + 1

    majority_wrong = {p for p, c in error_count.items() if c >= len(other_names) / 2}
    unique_correct = correct_here & majority_wrong

    p(f"\n  {name}: {len(unique_correct)} clips correct here but wrong in >=half of others")
    contrib_rows.append({
        "diarizer": name,
        "n_correct_here": len(correct_here),
        "n_unique_correct": len(unique_correct),
        "pct_unique": round(len(unique_correct) / max(len(correct_here), 1), 4),
    })

contrib_df = pd.DataFrame(contrib_rows).sort_values("n_unique_correct", ascending=False)
contrib_df.to_csv(OUT_DIR / "unique_contributions.csv", index=False)
p("\nUnique contributions summary:")
p(contrib_df.to_string(index=False))


# ================================================================ 5. PER-CHILD ERROR RATES ==

p("\n" + "-" * 60)
p("5. PER-CHILD ERROR RATES (all diarizers)")
p("-" * 60)

child_rows = []
for name, df in all_dfs.items():
    for child_id, sub in df.groupby("child_id"):
        n = len(sub)
        n_correct = sub["outcome"].isin(["TP", "TN"]).sum()
        n_fp = (sub["outcome"] == "FP").sum()
        n_fn = (sub["outcome"] == "FN").sum()
        tp_val = sub["timepoint_norm"].iloc[0] if "timepoint_norm" in sub.columns else "?"
        child_rows.append({
            "diarizer": name,
            "child_id": child_id,
            "timepoint_norm": tp_val,
            "n_clips": n,
            "n_fp": int(n_fp),
            "n_fn": int(n_fn),
            "accuracy": round(n_correct / n, 4),
        })

child_df = pd.DataFrame(child_rows)
child_df.to_csv(OUT_DIR / "per_child_error_rates_all_diarizers.csv", index=False)

# Mean accuracy across diarizers per child
child_mean = (
    child_df.groupby("child_id").agg(
        timepoint_norm=("timepoint_norm", "first"),
        n_clips=("n_clips", "first"),
        mean_accuracy=("accuracy", "mean"),
        std_accuracy=("accuracy", "std"),
        min_accuracy=("accuracy", "min"),
        max_accuracy=("accuracy", "max"),
        mean_n_fp=("n_fp", "mean"),
        mean_n_fn=("n_fn", "mean"),
    )
    .sort_values("mean_accuracy")
    .reset_index()
)
child_mean.to_csv(OUT_DIR / "per_child_mean_accuracy.csv", index=False)

p("\nHardest children (lowest mean accuracy across all diarizers):")
p(child_mean.head(15).to_string(index=False))
p("\nMost variable children (highest std):")
p(child_mean.sort_values("std_accuracy", ascending=False).head(10).to_string(index=False))


# ================================================================ 6. CONFIDENCE CALIBRATION ==

p("\n" + "-" * 60)
p("6. CONFIDENCE CALIBRATION")
p("-" * 60)

cal_rows = []
bins = [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0]
for name, df in all_dfs.items():
    df = df.copy()
    df["prob_bin"] = pd.cut(df["prob"], bins=bins, include_lowest=True)
    for bin_lbl, sub in df.groupby("prob_bin", observed=True):
        if len(sub) == 0:
            continue
        n_correct = sub["outcome"].isin(["TP", "TN"]).sum()
        cal_rows.append({
            "diarizer": name,
            "prob_bin": str(bin_lbl),
            "n": len(sub),
            "mean_prob": round(sub["prob"].mean(), 4),
            "actual_pos_rate": round(sub["label"].mean(), 4),
            "accuracy": round(n_correct / len(sub), 4),
        })

cal_df = pd.DataFrame(cal_rows)
cal_df.to_csv(OUT_DIR / "confidence_calibration_all_diarizers.csv", index=False)

p("\nCalibration summary (mean across diarizers per bin):")
cal_pivot = cal_df.pivot_table(
    index="prob_bin", columns="diarizer", values="actual_pos_rate"
).round(3)
p(cal_pivot.to_string())


# ================================================================ 7. MULTI-CHILD × INTERACTION ==

p("\n" + "-" * 60)
p("7. MULTI-CHILD × INTERACTION CROSS-TAB")
p("-" * 60)

ct_rows = []
for name, df in all_dfs.items():
    df_ann = df.merge(ann_slim[["audio_path", "#_children", "has_interaction", "n_children_cat"]],
                      on="audio_path", how="left")
    if "#_children" not in df_ann.columns:
        continue
    df_ann["multi_child"] = df_ann["#_children"] > 1

    for (mc, inter), sub in df_ann.groupby(["multi_child", "has_interaction"]):
        n = len(sub)
        n_correct = sub["outcome"].isin(["TP", "TN"]).sum()
        n_fp = (sub["outcome"] == "FP").sum()
        n_fn = (sub["outcome"] == "FN").sum()
        ct_rows.append({
            "diarizer": name,
            "multi_child": mc,
            "has_interaction": inter,
            "n": n,
            "accuracy": round(n_correct / n, 4),
            "fp_rate": round(n_fp / max((sub["label"]==0).sum(), 1), 4),
            "fn_rate": round(n_fn / max((sub["label"]==1).sum(), 1), 4),
        })

ct_df = pd.DataFrame(ct_rows)
ct_df.to_csv(OUT_DIR / "multi_child_interaction_crosstab_all_diarizers.csv", index=False)

# Summary: average across diarizers for each cell
ct_summary = (
    ct_df.groupby(["multi_child", "has_interaction"])[["accuracy", "fp_rate", "fn_rate"]]
    .mean()
    .round(4)
    .reset_index()
)
p("\nMulti-child × interaction cross-tab (averaged across diarizers):")
p(ct_summary.to_string(index=False))
p("\nPer-diarizer breakdown:")
p(ct_df.to_string(index=False))


# ================================================================ write report ==

report = "\n".join(lines)
report_path = OUT_DIR / "cross_diarizer_report.txt"
with open(report_path, "w") as f:
    f.write(report)
print(report)
print(f"\nAll outputs written to {OUT_DIR}")
print(f"  persistent_false_positives.csv     — {len(fp_persist)} rows")
print(f"  persistent_false_negatives.csv     — {len(fn_persist)} rows")
print(f"  pairwise_agreement.csv             — {len(agree_df)} pairs")
print(f"  pairwise_agreement_matrix.csv      — {n_diarizers}×{n_diarizers}")
print(f"  unique_contributions.csv           — {len(contrib_df)} rows")
print(f"  per_child_error_rates_all_diarizers.csv — {len(child_df)} rows")
print(f"  per_child_mean_accuracy.csv        — {len(child_mean)} children")
print(f"  confidence_calibration_all_diarizers.csv — {len(cal_df)} rows")
print(f"  multi_child_interaction_crosstab_all_diarizers.csv — {len(ct_df)} rows")
print(f"  cross_diarizer_report.txt")
