"""
Error analysis for USC-SAIL experiments (role-only and enrollment).

Mirrors the BabAR error_analysis.py output format for cross-experiment
comparison, and adds role-vs-enrollment comparison sections.

Usage:
    python error_analysis.py \
        --results-dir /home/manaal/orcd/scratch/child-adult-diarization/whisper-modeling/usc_sail_enrollment_runs \
        --output-dir /home/manaal/orcd/scratch/child-adult-diarization/whisper-modeling/usc_sail_enrollment_runs/error_analysis
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report


ANNOTATIONS_CSV = "/orcd/scratch/bcs/001/sensein/sails/BIDS_data/anotated_processed.csv"


def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def extract_task_type(audio_path):
    fname = os.path.basename(audio_path)
    if "task-" in fname:
        after_task = fname.split("task-")[1]
        return after_task.split("_")[0]
    return "unknown"


def extract_session(audio_path):
    fname = os.path.basename(audio_path)
    if "ses-" in fname:
        after_ses = fname.split("ses-")[1]
        return after_ses.split("_")[0]
    return "unknown"


def bidsprocessed_to_audio_path(bids_processed_path):
    if pd.isna(bids_processed_path):
        return ""
    s = str(bids_processed_path).strip()
    suffix = "_desc-processed_beh.mp4"
    if not s.endswith(suffix):
        return ""
    return s[:-len(suffix)] + "_audio.wav"


def load_annotations():
    """Load the annotations CSV and create an audio_path key for merging."""
    ann = pd.read_csv(ANNOTATIONS_CSV)
    ann["audio_path"] = ann["BidsProcessed"].apply(bidsprocessed_to_audio_path)

    for col in ["#_adults", "#_children", "#_people_background", "#_people_interacting"]:
        if col in ann.columns:
            ann[col] = pd.to_numeric(ann[col], errors="coerce")

    if "Interaction_with_child" in ann.columns:
        ann["interaction"] = ann["Interaction_with_child"].astype(str).str.strip().str.lower()
        ann["has_interaction"] = ann["interaction"].isin(["yes", "1", "true"])

    return ann


def prepare_df(pred_df, ann_dedup):
    """Add metadata, outcome labels, task/session extraction to a prediction df."""
    df = pred_df.copy()

    # Merge annotation metadata
    df = df.merge(ann_dedup, on="audio_path", how="left")

    # Fill missing metadata
    for col in ["#_adults", "#_children", "#_people_background", "#_people_interacting"]:
        if col in df.columns:
            df[col] = df[col].fillna(-1).astype(int)
    if "has_interaction" in df.columns:
        df["has_interaction"] = df["has_interaction"].fillna(False)

    # Classify outcomes
    df["outcome"] = "TN"
    df.loc[(df["label"] == 1) & (df["pred_label"] == 1), "outcome"] = "TP"
    df.loc[(df["label"] == 1) & (df["pred_label"] == 0), "outcome"] = "FN"
    df.loc[(df["label"] == 0) & (df["pred_label"] == 1), "outcome"] = "FP"

    # Extract task/session
    df["task"] = df["audio_path"].apply(extract_task_type)
    df["session"] = df["audio_path"].apply(extract_session)

    # Derived
    if "#_children" in df.columns:
        df["multi_child"] = df["#_children"] > 1

    return df


# =========================================================
# Shared analysis functions (mirror BabAR format)
# =========================================================

def analyze_model(df, model_name, output_dir, feature_cols):
    """
    Run the full error analysis for a single model's predictions.
    Output structure mirrors the BabAR error_analysis.py exactly.
    """
    model_dir = os.path.join(output_dir, model_name)
    os.makedirs(model_dir, exist_ok=True)

    fp_df = df[df["outcome"] == "FP"].copy()
    fn_df = df[df["outcome"] == "FN"].copy()

    # =========================================================
    # 1. Overall confusion matrix
    # =========================================================
    print("=" * 60)
    print(f"1. OVERALL PERFORMANCE — {model_name}")
    print("=" * 60)

    cm = confusion_matrix(df["label"], df["pred_label"])
    print(f"\nConfusion Matrix:")
    print(f"  TN={cm[0,0]}  FP={cm[0,1]}")
    print(f"  FN={cm[1,0]}  TP={cm[1,1]}")
    print(f"\nOutcome counts:")
    print(df["outcome"].value_counts().to_string())
    print(f"\nLabel balance: {df['label'].mean():.3f} positive")

    report = classification_report(df["label"], df["pred_label"], output_dict=True)
    save_json(report, os.path.join(model_dir, "classification_report.json"))

    print("\nPer timepoint:")
    for tp, sub in df.groupby("timepoint_norm"):
        n = len(sub)
        pos_rate = sub["label"].mean()
        fp_rate = (sub["outcome"] == "FP").sum() / max((sub["label"] == 0).sum(), 1)
        fn_rate = (sub["outcome"] == "FN").sum() / max((sub["label"] == 1).sum(), 1)
        print(f"  {tp} (n={n}, pos_rate={pos_rate:.2f}): "
              f"FP_rate={fp_rate:.3f}, FN_rate={fn_rate:.3f}")

    # =========================================================
    # 2. Feature distributions by outcome
    # =========================================================
    print("\n" + "=" * 60)
    print(f"2. FEATURE DISTRIBUTIONS BY OUTCOME — {model_name}")
    print("=" * 60)

    available_feats = [f for f in feature_cols if f in df.columns]

    feat_summary_rows = []
    for feat in available_feats:
        for outcome in ["TP", "FP", "FN", "TN"]:
            vals = df.loc[df["outcome"] == outcome, feat]
            if len(vals) == 0:
                continue
            feat_summary_rows.append({
                "feature": feat,
                "outcome": outcome,
                "n": len(vals),
                "mean": float(vals.mean()),
                "median": float(vals.median()),
                "std": float(vals.std()),
            })

    feat_summary = pd.DataFrame(feat_summary_rows)
    feat_summary.to_csv(
        os.path.join(model_dir, "feature_distributions_by_outcome.csv"),
        index=False,
    )

    # Print key features (use whatever is available)
    key_feats = [f for f in ["score_duration_sec", "prob"] if f in df.columns]
    for feat in key_feats:
        print(f"\n  {feat}:")
        for outcome in ["TP", "FP", "FN", "TN"]:
            row = feat_summary[
                (feat_summary["feature"] == feat) & (feat_summary["outcome"] == outcome)
            ]
            if len(row) == 0:
                continue
            r = row.iloc[0]
            print(f"    {outcome}: mean={r['mean']:.3f}  median={r['median']:.3f}  "
                  f"std={r['std']:.3f}  (n={int(r['n'])})")

    # =========================================================
    # 3. Number of children and false positives
    # =========================================================
    print("\n" + "=" * 60)
    print(f"3. NUMBER OF CHILDREN & FALSE POSITIVES — {model_name}")
    print("=" * 60)

    if "#_children" in df.columns:
        print("\nOutcome by #_children (negative clips only — where FP is possible):")
        neg_clips = df[df["label"] == 0].copy()
        for n_kids, sub in neg_clips.groupby("#_children"):
            n = len(sub)
            n_fp = (sub["outcome"] == "FP").sum()
            fp_rate = n_fp / n if n > 0 else 0
            print(f"  #_children={n_kids}: n={n}, FPs={n_fp}, FP_rate={fp_rate:.3f}")

        print("\nMulti-child clips (>1 child) vs single-child:")
        for multi, sub in neg_clips.groupby("multi_child"):
            n = len(sub)
            n_fp = (sub["outcome"] == "FP").sum()
            fp_rate = n_fp / n if n > 0 else 0
            label = "multi-child" if multi else "single-child"
            print(f"  {label}: n={n}, FPs={n_fp}, FP_rate={fp_rate:.3f}")

        if len(fp_df) > 0:
            print(f"\nFP mean #_children: {fp_df['#_children'].mean():.2f}")
            tn_df = df[df["outcome"] == "TN"]
            if len(tn_df) > 0:
                print(f"TN mean #_children: {tn_df['#_children'].mean():.2f}")

    # =========================================================
    # 4. Interaction with child and false negatives
    # =========================================================
    print("\n" + "=" * 60)
    print(f"4. INTERACTION WITH CHILD & FALSE NEGATIVES — {model_name}")
    print("=" * 60)

    if "has_interaction" in df.columns:
        print("\nOutcome by Interaction_with_child (positive clips only — where FN is possible):")
        pos_clips = df[df["label"] == 1].copy()
        for interact, sub in pos_clips.groupby("has_interaction"):
            n = len(sub)
            n_fn = (sub["outcome"] == "FN").sum()
            fn_rate = n_fn / n if n > 0 else 0
            label = "interaction=yes" if interact else "interaction=no/missing"
            print(f"  {label}: n={n}, FNs={n_fn}, FN_rate={fn_rate:.3f}")

        if len(fn_df) > 0:
            fn_interact = fn_df["has_interaction"].sum()
            fn_no_interact = len(fn_df) - fn_interact
            print(f"\nFN with interaction: {fn_interact}")
            print(f"FN without interaction: {fn_no_interact}")

    # =========================================================
    # 5. Number of adults and errors
    # =========================================================
    print("\n" + "=" * 60)
    print(f"5. NUMBER OF ADULTS & ERRORS — {model_name}")
    print("=" * 60)

    if "#_adults" in df.columns:
        print("\nOutcome by #_adults:")
        for outcome in ["TP", "TN", "FP", "FN"]:
            sub = df[df["outcome"] == outcome]
            if len(sub) > 0:
                mean_adults = sub["#_adults"].mean()
                print(f"  {outcome} (n={len(sub)}): mean #_adults={mean_adults:.2f}")

    if "#_people_interacting" in df.columns:
        print("\nOutcome by #_people_interacting:")
        for outcome in ["TP", "TN", "FP", "FN"]:
            sub = df[df["outcome"] == outcome]
            if len(sub) > 0:
                mean_pi = sub["#_people_interacting"].mean()
                print(f"  {outcome} (n={len(sub)}): mean #_people_interacting={mean_pi:.2f}")

    # =========================================================
    # 6. False positive deep dive
    # =========================================================
    print("\n" + "=" * 60)
    print(f"6. FALSE POSITIVE ANALYSIS — {model_name}")
    print("=" * 60)

    print(f"\n{len(fp_df)} false positives total")

    if len(fp_df) > 0:
        print("\nFP by timepoint:")
        print(fp_df["timepoint_norm"].value_counts().to_string())

        print("\nFP by task type:")
        print(fp_df["task"].value_counts().to_string())

        if "score_duration_sec" in fp_df.columns:
            print(f"\nFP mean CHI duration: {fp_df['score_duration_sec'].mean():.2f}s")
        print(f"FP mean prob: {fp_df['prob'].mean():.3f}")

        if "#_children" in fp_df.columns:
            print(f"FP mean #_children: {fp_df['#_children'].mean():.2f}")
            print(f"FP with >1 child: {(fp_df['#_children'] > 1).sum()} / {len(fp_df)}")

        if "#_people_background" in fp_df.columns:
            print(f"FP mean #_people_background: {fp_df['#_people_background'].mean():.2f}")

        fp_child_counts = fp_df["child_id"].value_counts()
        print(f"\nFP by child (top 10):")
        print(fp_child_counts.head(10).to_string())

        high_conf_fp = fp_df[fp_df["prob"] >= 0.7]
        print(f"\nHigh-confidence FPs (prob >= 0.7): {len(high_conf_fp)}")
        if len(high_conf_fp) > 0:
            if "score_duration_sec" in high_conf_fp.columns:
                print(f"  Mean CHI duration: {high_conf_fp['score_duration_sec'].mean():.2f}s")
            if "#_children" in high_conf_fp.columns:
                print(f"  Mean #_children: {high_conf_fp['#_children'].mean():.2f}")

    fp_df.to_csv(os.path.join(model_dir, "false_positives.csv"), index=False)
    with open(os.path.join(model_dir, "false_positive_files.txt"), "w") as f:
        for path in fp_df["audio_path"].tolist():
            f.write(path + "\n")

    # =========================================================
    # 7. False negative deep dive
    # =========================================================
    print("\n" + "=" * 60)
    print(f"7. FALSE NEGATIVE ANALYSIS — {model_name}")
    print("=" * 60)

    print(f"\n{len(fn_df)} false negatives total")

    dur_col = "score_duration_sec" if "score_duration_sec" in fn_df.columns else None

    if len(fn_df) > 0:
        print("\nFN by timepoint:")
        print(fn_df["timepoint_norm"].value_counts().to_string())

        print("\nFN by task type:")
        print(fn_df["task"].value_counts().to_string())

        if dur_col:
            print(f"\nFN mean CHI duration: {fn_df[dur_col].mean():.2f}s")
        print(f"FN mean prob: {fn_df['prob'].mean():.3f}")

        # Silent vs vocal FNs
        if dur_col:
            silent_fn = fn_df[fn_df[dur_col] == 0]
            vocal_fn = fn_df[fn_df[dur_col] > 0]
            print(f"\nFNs with zero CHI (silent child): {len(silent_fn)} / {len(fn_df)}")
            print(f"FNs with CHI > 0 (child speaks but missed): {len(vocal_fn)}")

            if len(vocal_fn) > 0:
                print(f"  Mean CHI duration: {vocal_fn[dur_col].mean():.2f}s")
                print(f"  Mean prob: {vocal_fn['prob'].mean():.3f}")

            if "has_interaction" in fn_df.columns and len(silent_fn) > 0:
                si_interact = silent_fn["has_interaction"].sum()
                print(f"\nSilent FNs with interaction: {si_interact} / {len(silent_fn)}")
                print(f"Silent FNs without interaction: {len(silent_fn) - si_interact}")

            if "#_children" in fn_df.columns and len(silent_fn) > 0:
                print(f"Silent FN mean #_children: {silent_fn['#_children'].mean():.2f}")

        if "#_adults" in fn_df.columns:
            print(f"\nFN mean #_adults: {fn_df['#_adults'].mean():.2f}")

        fn_child_counts = fn_df["child_id"].value_counts()
        print(f"\nFN by child (top 10):")
        print(fn_child_counts.head(10).to_string())

    fn_df.to_csv(os.path.join(model_dir, "false_negatives.csv"), index=False)
    with open(os.path.join(model_dir, "false_negative_files.txt"), "w") as f:
        for path in fn_df["audio_path"].tolist():
            f.write(path + "\n")

    # =========================================================
    # 8. Per-child error rates with metadata
    # =========================================================
    print("\n" + "=" * 60)
    print(f"8. PER-CHILD ERROR RATES — {model_name}")
    print("=" * 60)

    child_rows = []
    for child_id, sub in df.groupby("child_id"):
        n = len(sub)
        n_pos = (sub["label"] == 1).sum()
        n_neg = (sub["label"] == 0).sum()
        n_fp = (sub["outcome"] == "FP").sum()
        n_fn = (sub["outcome"] == "FN").sum()
        n_correct = ((sub["outcome"] == "TP") | (sub["outcome"] == "TN")).sum()
        accuracy = n_correct / n if n > 0 else 0

        row = {
            "child_id": child_id,
            "timepoint": sub["timepoint_norm"].iloc[0],
            "n_clips": n,
            "n_positive": int(n_pos),
            "n_negative": int(n_neg),
            "n_fp": int(n_fp),
            "n_fn": int(n_fn),
            "accuracy": accuracy,
        }

        if dur_col and dur_col in sub.columns:
            row["mean_chi_dur"] = sub[dur_col].mean()
        row["mean_prob"] = sub["prob"].mean()

        if "#_children" in sub.columns:
            row["mean_n_children"] = sub["#_children"].mean()
        if "#_adults" in sub.columns:
            row["mean_n_adults"] = sub["#_adults"].mean()
        if "has_interaction" in sub.columns:
            row["pct_interaction"] = sub["has_interaction"].mean()

        child_rows.append(row)

    child_df = pd.DataFrame(child_rows).sort_values("accuracy")
    child_df.to_csv(os.path.join(model_dir, "per_child_error_rates.csv"), index=False)

    display_cols = ["child_id", "timepoint", "n_clips", "accuracy", "n_fp", "n_fn"]
    if "mean_chi_dur" in child_df.columns:
        display_cols.append("mean_chi_dur")
    if "mean_n_children" in child_df.columns:
        display_cols.append("mean_n_children")
    if "pct_interaction" in child_df.columns:
        display_cols.append("pct_interaction")

    print(f"\nHardest children (lowest accuracy):")
    print(child_df.head(10)[display_cols].to_string(index=False))

    print(f"\nEasiest children (highest accuracy):")
    print(child_df.tail(5)[display_cols].to_string(index=False))

    # =========================================================
    # 9. Task-type analysis
    # =========================================================
    print("\n" + "=" * 60)
    print(f"9. TASK-TYPE BREAKDOWN — {model_name}")
    print("=" * 60)

    task_rows = []
    for task, sub in df.groupby("task"):
        n = len(sub)
        n_correct = ((sub["outcome"] == "TP") | (sub["outcome"] == "TN")).sum()
        accuracy = n_correct / n
        fp_rate = (sub["outcome"] == "FP").sum() / max((sub["label"] == 0).sum(), 1)
        fn_rate = (sub["outcome"] == "FN").sum() / max((sub["label"] == 1).sum(), 1)

        row = {
            "task": task,
            "n": n,
            "pos_rate": sub["label"].mean(),
            "accuracy": accuracy,
            "fp_rate": fp_rate,
            "fn_rate": fn_rate,
        }
        if dur_col and dur_col in sub.columns:
            row["mean_chi_dur"] = sub[dur_col].mean()
        if "#_children" in sub.columns:
            row["mean_n_children"] = sub["#_children"].mean()
        if "has_interaction" in sub.columns:
            row["pct_interaction"] = sub["has_interaction"].mean()

        task_rows.append(row)

    task_df = pd.DataFrame(task_rows).sort_values("accuracy")
    task_df.to_csv(os.path.join(model_dir, "task_type_breakdown.csv"), index=False)
    print(task_df.to_string(index=False))

    # =========================================================
    # 10. Confidence analysis
    # =========================================================
    print("\n" + "=" * 60)
    print(f"10. CONFIDENCE ANALYSIS — {model_name}")
    print("=" * 60)

    # For role-only, prob is raw duration — use quantile bins
    # For enrollment, prob is cosine sim in [0,1] — use fixed bins
    if model_name == "role_only":
        df["prob_bin"] = pd.qcut(df["prob"], q=6, duplicates="drop")
    else:
        bins = [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0]
        df["prob_bin"] = pd.cut(df["prob"], bins=bins, include_lowest=True)

    conf_rows = []
    for bin_label, sub in df.groupby("prob_bin", observed=True):
        if len(sub) == 0:
            continue
        n = len(sub)
        n_correct = ((sub["outcome"] == "TP") | (sub["outcome"] == "TN")).sum()
        conf_rows.append({
            "prob_bin": str(bin_label),
            "n": n,
            "mean_prob": sub["prob"].mean(),
            "actual_pos_rate": sub["label"].mean(),
            "accuracy": n_correct / n,
        })

    conf_df = pd.DataFrame(conf_rows)
    conf_df.to_csv(os.path.join(model_dir, "confidence_calibration.csv"), index=False)
    print("\nCalibration:")
    print(conf_df.to_string(index=False))

    # =========================================================
    # 11. Most confident mistakes
    # =========================================================
    print("\n" + "=" * 60)
    print(f"11. MOST CONFIDENT MISTAKES — {model_name}")
    print("=" * 60)

    errors = df[df["outcome"].isin(["FP", "FN"])].copy()
    errors["confidence"] = errors.apply(
        lambda r: r["prob"] if r["outcome"] == "FP" else (1 - r["prob"]),
        axis=1,
    )
    errors = errors.sort_values("confidence", ascending=False)

    print("\nTop 15 most confident errors:")
    for _, row in errors.head(15).iterrows():
        n_kids = row.get("#_children", "?")
        interact = row.get("interaction", "?")
        dur_str = ""
        if dur_col and dur_col in row.index:
            dur_str = f" | chi_dur={row[dur_col]:.2f}s"
        print(f"\n  {row['outcome']} | prob={row['prob']:.3f} | "
              f"child={row['child_id']} | tp={row['timepoint_norm']}")
        print(f"    task={row['task']}{dur_str}")
        print(f"    #_children={n_kids} | #_adults={row.get('#_adults', '?')} | "
              f"interaction={interact} | #_people_interacting={row.get('#_people_interacting', '?')}")

    errors.to_csv(os.path.join(model_dir, "all_errors_by_confidence.csv"), index=False)

    # =========================================================
    # 12. Cross-tabulation: multi-child x interaction x outcome
    # =========================================================
    print("\n" + "=" * 60)
    print(f"12. MULTI-CHILD x INTERACTION CROSS-TAB — {model_name}")
    print("=" * 60)

    if "multi_child" in df.columns and "has_interaction" in df.columns:
        ct_rows = []
        for (mc, inter), sub in df.groupby(["multi_child", "has_interaction"]):
            n = len(sub)
            n_correct = ((sub["outcome"] == "TP") | (sub["outcome"] == "TN")).sum()
            n_fp = (sub["outcome"] == "FP").sum()
            n_fn = (sub["outcome"] == "FN").sum()
            ct_rows.append({
                "multi_child": mc,
                "has_interaction": inter,
                "n": n,
                "accuracy": n_correct / n,
                "n_fp": n_fp,
                "n_fn": n_fn,
                "fp_rate": n_fp / max((sub["label"] == 0).sum(), 1),
                "fn_rate": n_fn / max((sub["label"] == 1).sum(), 1),
            })

        ct_df = pd.DataFrame(ct_rows)
        ct_df.to_csv(os.path.join(model_dir, "multi_child_interaction_crosstab.csv"), index=False)
        print(ct_df.to_string(index=False))

    # =========================================================
    # 13. Summary for thesis
    # =========================================================
    print("\n" + "=" * 60)
    print(f"13. SUMMARY FOR THESIS — {model_name}")
    print("=" * 60)

    n_fp = (df["outcome"] == "FP").sum()
    n_fn = (df["outcome"] == "FN").sum()
    total_errors = n_fp + n_fn

    silent_fn_count = 0
    if dur_col and len(fn_df) > 0:
        silent_fn_count = int((fn_df[dur_col] == 0).sum())

    if "#_children" in df.columns:
        neg_multi = df[(df["label"] == 0) & (df["#_children"] > 1)]
        neg_single = df[(df["label"] == 0) & (df["#_children"] <= 1)]
        fp_rate_multi = (neg_multi["outcome"] == "FP").sum() / max(len(neg_multi), 1)
        fp_rate_single = (neg_single["outcome"] == "FP").sum() / max(len(neg_single), 1)
    else:
        fp_rate_multi = fp_rate_single = float("nan")

    if "has_interaction" in df.columns:
        pos_interact = df[(df["label"] == 1) & (df["has_interaction"] == True)]
        pos_no_interact = df[(df["label"] == 1) & (df["has_interaction"] == False)]
        fn_rate_interact = (pos_interact["outcome"] == "FN").sum() / max(len(pos_interact), 1)
        fn_rate_no_interact = (pos_no_interact["outcome"] == "FN").sum() / max(len(pos_no_interact), 1)
    else:
        fn_rate_interact = fn_rate_no_interact = float("nan")

    summary = {
        "model": model_name,
        "total_test_clips": len(df),
        "total_errors": int(total_errors),
        "error_rate": total_errors / len(df),
        "n_false_positives": int(n_fp),
        "n_false_negatives": int(n_fn),
        "fn_silent_child": int(silent_fn_count),
        "fn_silent_child_pct": silent_fn_count / max(n_fn, 1),
        "fn_vocal_child": int(n_fn - silent_fn_count),
        "fp_rate_multi_child": float(fp_rate_multi),
        "fp_rate_single_child": float(fp_rate_single),
        "fn_rate_with_interaction": float(fn_rate_interact),
        "fn_rate_without_interaction": float(fn_rate_no_interact),
        "n_children_perfect": int((child_df["accuracy"] == 1.0).sum()),
        "n_children_below_70pct": int((child_df["accuracy"] < 0.7).sum()),
        "hardest_task": task_df.iloc[0]["task"] if len(task_df) > 0 else "N/A",
        "easiest_task": task_df.iloc[-1]["task"] if len(task_df) > 0 else "N/A",
    }

    if dur_col:
        summary["mean_fp_chi_dur"] = float(fp_df[dur_col].mean()) if len(fp_df) > 0 else 0
        summary["mean_fn_chi_dur"] = float(fn_df[dur_col].mean()) if len(fn_df) > 0 else 0

    save_json(summary, os.path.join(model_dir, "thesis_summary.json"))

    print(f"\n  Total test clips: {len(df)}")
    print(f"  Total errors: {total_errors} ({100*total_errors/len(df):.1f}%)")
    print(f"  False positives: {n_fp}")
    print(f"  False negatives: {n_fn}")
    if dur_col:
        print(f"    - Silent child: {silent_fn_count} ({100*silent_fn_count/max(n_fn,1):.0f}% of FNs)")
        print(f"    - Vocal child: {n_fn - silent_fn_count}")
    print(f"  FP rate multi-child clips: {fp_rate_multi:.3f}")
    print(f"  FP rate single-child clips: {fp_rate_single:.3f}")
    print(f"  FN rate with interaction: {fn_rate_interact:.3f}")
    print(f"  FN rate without interaction: {fn_rate_no_interact:.3f}")
    print(f"  Children with perfect accuracy: {(child_df['accuracy']==1.0).sum()}")
    print(f"  Children below 70% accuracy: {(child_df['accuracy']<0.7).sum()}")

    # Save full merged df
    df.to_csv(os.path.join(model_dir, "full_test_with_metadata.csv"), index=False)

    return summary, child_df, task_df


# =========================================================
# Role vs Enrollment comparison
# =========================================================

def compare_role_vs_enrollment(role_df, enroll_df, output_dir):
    """
    Compare role-only and enrollment predictions clip-by-clip.
    """
    comp_dir = os.path.join(output_dir, "role_vs_enrollment")
    os.makedirs(comp_dir, exist_ok=True)

    print("\n" + "#" * 70)
    print("ROLE-ONLY vs ENROLLMENT COMPARISON")
    print("#" * 70)

    # Align on audio_path
    merged = role_df[["audio_path", "child_id", "timepoint_norm", "label",
                       "pred_label", "prob", "outcome"]].copy()
    merged = merged.rename(columns={
        "pred_label": "role_pred",
        "prob": "role_prob",
        "outcome": "role_outcome",
    })

    enroll_sub = enroll_df[["audio_path", "pred_label", "prob", "outcome"]].copy()
    enroll_sub = enroll_sub.rename(columns={
        "pred_label": "enroll_pred",
        "prob": "enroll_prob",
        "outcome": "enroll_outcome",
    })

    merged = merged.merge(enroll_sub, on="audio_path", how="inner")

    # Agreement
    merged["agree"] = merged["role_pred"] == merged["enroll_pred"]
    n_agree = merged["agree"].sum()
    n_total = len(merged)
    print(f"\nClip-level agreement: {n_agree}/{n_total} ({100*n_agree/n_total:.1f}%)")

    # Transition matrix
    print("\nOutcome transitions (role → enrollment):")
    trans = pd.crosstab(
        merged["role_outcome"], merged["enroll_outcome"],
        margins=True,
    )
    print(trans.to_string())
    trans.to_csv(os.path.join(comp_dir, "outcome_transitions.csv"))

    # Clips fixed by enrollment (role wrong, enrollment right)
    role_wrong = merged[
        merged["role_outcome"].isin(["FP", "FN"]) &
        merged["enroll_outcome"].isin(["TP", "TN"])
    ]
    print(f"\nClips FIXED by enrollment (role wrong → enrollment right): {len(role_wrong)}")
    if len(role_wrong) > 0:
        print(f"  Role FP → Enrollment TN: {((role_wrong['role_outcome']=='FP') & (role_wrong['enroll_outcome']=='TN')).sum()}")
        print(f"  Role FN → Enrollment TP: {((role_wrong['role_outcome']=='FN') & (role_wrong['enroll_outcome']=='TP')).sum()}")

    # Clips broken by enrollment (role right, enrollment wrong)
    role_right = merged[
        merged["role_outcome"].isin(["TP", "TN"]) &
        merged["enroll_outcome"].isin(["FP", "FN"])
    ]
    print(f"\nClips BROKEN by enrollment (role right → enrollment wrong): {len(role_right)}")
    if len(role_right) > 0:
        print(f"  Role TN → Enrollment FP: {((role_right['role_outcome']=='TN') & (role_right['enroll_outcome']=='FP')).sum()}")
        print(f"  Role TP → Enrollment FN: {((role_right['role_outcome']=='TP') & (role_right['enroll_outcome']=='FN')).sum()}")

    # Net effect
    role_errors = merged["role_outcome"].isin(["FP", "FN"]).sum()
    enroll_errors = merged["enroll_outcome"].isin(["FP", "FN"]).sum()
    print(f"\nNet effect: role errors={role_errors}, enrollment errors={enroll_errors}, "
          f"delta={enroll_errors - role_errors} ({'worse' if enroll_errors > role_errors else 'better'})")

    # Per-timepoint comparison
    print("\nPer-timepoint error counts:")
    for tp, sub in merged.groupby("timepoint_norm"):
        r_err = sub["role_outcome"].isin(["FP", "FN"]).sum()
        e_err = sub["enroll_outcome"].isin(["FP", "FN"]).sum()
        print(f"  {tp}: role_errors={r_err}, enroll_errors={e_err}, delta={e_err - r_err}")

    # Per-child: who benefits / suffers from enrollment
    print("\nPer-child impact of enrollment:")
    child_impact_rows = []
    for child_id, sub in merged.groupby("child_id"):
        r_correct = sub["role_outcome"].isin(["TP", "TN"]).sum()
        e_correct = sub["enroll_outcome"].isin(["TP", "TN"]).sum()
        child_impact_rows.append({
            "child_id": child_id,
            "timepoint": sub["timepoint_norm"].iloc[0],
            "n_clips": len(sub),
            "role_accuracy": r_correct / len(sub),
            "enroll_accuracy": e_correct / len(sub),
            "accuracy_delta": (e_correct - r_correct) / len(sub),
        })

    child_impact = pd.DataFrame(child_impact_rows).sort_values("accuracy_delta")
    child_impact.to_csv(os.path.join(comp_dir, "per_child_impact.csv"), index=False)

    helped = child_impact[child_impact["accuracy_delta"] > 0]
    hurt = child_impact[child_impact["accuracy_delta"] < 0]
    unchanged = child_impact[child_impact["accuracy_delta"] == 0]
    print(f"  Children helped: {len(helped)}")
    print(f"  Children hurt: {len(hurt)}")
    print(f"  Children unchanged: {len(unchanged)}")

    if len(hurt) > 0:
        print(f"\n  Most hurt by enrollment:")
        print(hurt.head(5).to_string(index=False))

    if len(helped) > 0:
        print(f"\n  Most helped by enrollment:")
        print(helped.tail(5).to_string(index=False))

    # Disagreement analysis with metadata
    disagree = merged[~merged["agree"]].copy()
    disagree.to_csv(os.path.join(comp_dir, "disagreements.csv"), index=False)

    print(f"\nDisagreements by type:")
    if len(disagree) > 0:
        disagree["transition"] = disagree["role_outcome"] + " → " + disagree["enroll_outcome"]
        print(disagree["transition"].value_counts().to_string())

    merged.to_csv(os.path.join(comp_dir, "full_comparison.csv"), index=False)

    # Save comparison summary
    comp_summary = {
        "n_clips": n_total,
        "agreement_rate": n_agree / n_total,
        "role_total_errors": int(role_errors),
        "enroll_total_errors": int(enroll_errors),
        "clips_fixed_by_enrollment": int(len(role_wrong)),
        "clips_broken_by_enrollment": int(len(role_right)),
        "children_helped": int(len(helped)),
        "children_hurt": int(len(hurt)),
        "children_unchanged": int(len(unchanged)),
    }
    save_json(comp_summary, os.path.join(comp_dir, "comparison_summary.json"))

    print(f"\nComparison outputs saved to {comp_dir}")


# =========================================================
# Main
# =========================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load annotations
    ann = load_annotations()
    meta_cols = ["audio_path"]
    for col in ["#_adults", "#_children", "#_people_background",
                 "#_people_interacting", "interaction", "has_interaction"]:
        if col in ann.columns:
            meta_cols.append(col)
    ann_dedup = ann[meta_cols].drop_duplicates(subset=["audio_path"])

    # -------------------------------------------------------
    # Load role-only test predictions
    # -------------------------------------------------------
    role_pred_path = os.path.join(args.results_dir, "role_only_test_predictions.csv")
    role_pred_df = pd.read_csv(role_pred_path)

    # Feature columns for role-only (duration-based)
    role_feature_cols = ["score_duration_sec", "prob"]

    role_df = prepare_df(role_pred_df, ann_dedup)

    print("\n" + "#" * 70)
    print("ROLE-ONLY (USC-SAIL) ERROR ANALYSIS")
    print("#" * 70 + "\n")

    role_summary, role_child_df, role_task_df = analyze_model(
        role_df, "role_only", args.output_dir, role_feature_cols
    )

    # -------------------------------------------------------
    # Load enrollment test predictions
    # -------------------------------------------------------
    enroll_pred_path = os.path.join(args.results_dir, "enroll_test_predictions.csv")
    enroll_pred_df = pd.read_csv(enroll_pred_path)

    # Carry over the duration score from role predictions for analysis
    if "score_duration_sec" in role_pred_df.columns:
        dur_map = role_pred_df.set_index("audio_path")["score_duration_sec"].to_dict()
        enroll_pred_df["score_duration_sec"] = enroll_pred_df["audio_path"].map(dur_map)

    enroll_feature_cols = ["score_duration_sec", "prob"]

    enroll_df = prepare_df(enroll_pred_df, ann_dedup)

    print("\n" + "#" * 70)
    print("ENROLLMENT (USC-SAIL + ECAPA) ERROR ANALYSIS")
    print("#" * 70 + "\n")

    enroll_summary, enroll_child_df, enroll_task_df = analyze_model(
        enroll_df, "enrollment", args.output_dir, enroll_feature_cols
    )

    # -------------------------------------------------------
    # Role vs Enrollment comparison
    # -------------------------------------------------------
    compare_role_vs_enrollment(role_df, enroll_df, args.output_dir)

    # -------------------------------------------------------
    # Combined summary for easy cross-experiment comparison
    # -------------------------------------------------------
    combined = {
        "role_only": role_summary,
        "enrollment": enroll_summary,
    }
    save_json(combined, os.path.join(args.output_dir, "combined_thesis_summary.json"))

    print(f"\nAll outputs saved to {args.output_dir}")


if __name__ == "__main__":
    main()
