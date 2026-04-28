"""
Error analysis for the best model (pertp_logistic_diarizer_plus_phoneme).

Merges annotation metadata (#_children, #_adults, Interaction_with_child, etc.)
to explain false positives and false negatives.

Usage:
    python error_analysis.py \
        --results-dir /home/manaal/orcd/scratch/child-adult-diarization/babar_combined_runs \
        --output-dir /home/manaal/orcd/scratch/child-adult-diarization/babar_combined_runs/error_analysis
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

    # Clean up numeric columns
    for col in ["#_adults", "#_children", "#_people_background", "#_people_interacting"]:
        if col in ann.columns:
            ann[col] = pd.to_numeric(ann[col], errors="coerce")

    # Clean interaction column
    if "Interaction_with_child" in ann.columns:
        ann["interaction"] = ann["Interaction_with_child"].astype(str).str.strip().str.lower()
        ann["has_interaction"] = ann["interaction"].isin(["yes", "1", "true"])
    
    return ann


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load predictions and features
    pred_path = os.path.join(
        args.results_dir,
        "pertp_logistic_diarizer_plus_phoneme_test_predictions.csv",
    )
    feat_path = os.path.join(args.results_dir, "test_features.csv")

    pred_df = pd.read_csv(pred_path)
    feat_df = pd.read_csv(feat_path)

    # Merge predictions into features
    df = feat_df.copy()
    df["prob"] = pred_df["prob"].values
    df["pred_label"] = pred_df["pred_label"].values

    # Merge annotation metadata
    ann = load_annotations()
    meta_cols = ["audio_path"]
    for col in ["#_adults", "#_children", "#_people_background",
                 "#_people_interacting", "interaction", "has_interaction"]:
        if col in ann.columns:
            meta_cols.append(col)

    ann_dedup = ann[meta_cols].drop_duplicates(subset=["audio_path"])
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

    # Extract task/session from filenames
    df["task"] = df["audio_path"].apply(extract_task_type)
    df["session"] = df["audio_path"].apply(extract_session)

    # Derived: multi-child clips
    if "#_children" in df.columns:
        df["multi_child"] = df["#_children"] > 1

    feature_cols = [
        "kchi_total_dur", "kchi_n_segments", "kchi_mean_seg_dur",
        "kchi_max_seg_dur", "kchi_proportion",
        "phon_n_utterances", "phon_n_total", "phon_n_unique",
        "phon_n_consonants", "phon_n_vowels", "phon_cv_ratio",
        "phon_mean_per_utt", "phon_max_per_utt", "phon_unique_ratio",
        "sim_weighted_mean", "sim_max", "sim_top3_mean",
    ]

    # =========================================================
    # 1. Overall confusion matrix
    # =========================================================
    print("=" * 60)
    print("1. OVERALL PERFORMANCE")
    print("=" * 60)

    cm = confusion_matrix(df["label"], df["pred_label"])
    print(f"\nConfusion Matrix:")
    print(f"  TN={cm[0,0]}  FP={cm[0,1]}")
    print(f"  FN={cm[1,0]}  TP={cm[1,1]}")
    print(f"\nOutcome counts:")
    print(df["outcome"].value_counts().to_string())
    print(f"\nLabel balance: {df['label'].mean():.3f} positive")

    report = classification_report(df["label"], df["pred_label"], output_dict=True)
    save_json(report, os.path.join(args.output_dir, "classification_report.json"))

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
    print("2. FEATURE DISTRIBUTIONS BY OUTCOME")
    print("=" * 60)

    feat_summary_rows = []
    for feat in feature_cols:
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
        os.path.join(args.output_dir, "feature_distributions_by_outcome.csv"),
        index=False,
    )

    for feat in ["kchi_total_dur", "phon_n_unique", "sim_weighted_mean"]:
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
    print("3. NUMBER OF CHILDREN & FALSE POSITIVES")
    print("=" * 60)

    fp_df = df[df["outcome"] == "FP"].copy()
    fn_df = df[df["outcome"] == "FN"].copy()

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
    print("4. INTERACTION WITH CHILD & FALSE NEGATIVES")
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
    print("5. NUMBER OF ADULTS & ERRORS")
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
    print("6. FALSE POSITIVE ANALYSIS")
    print("=" * 60)

    print(f"\n{len(fp_df)} false positives total")

    if len(fp_df) > 0:
        print("\nFP by timepoint:")
        print(fp_df["timepoint_norm"].value_counts().to_string())

        print("\nFP by task type:")
        print(fp_df["task"].value_counts().to_string())

        print(f"\nFP mean KCHI duration: {fp_df['kchi_total_dur'].mean():.2f}s")
        print(f"FP mean KCHI segments: {fp_df['kchi_n_segments'].mean():.1f}")
        print(f"FP mean phonemes: {fp_df['phon_n_total'].mean():.1f}")
        print(f"FP mean unique phonemes: {fp_df['phon_n_unique'].mean():.1f}")
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
            print(f"  Mean KCHI duration: {high_conf_fp['kchi_total_dur'].mean():.2f}s")
            print(f"  Mean phonemes: {high_conf_fp['phon_n_total'].mean():.1f}")
            if "#_children" in high_conf_fp.columns:
                print(f"  Mean #_children: {high_conf_fp['#_children'].mean():.2f}")

    fp_df.to_csv(os.path.join(args.output_dir, "false_positives.csv"), index=False)
    with open(os.path.join(args.output_dir, "false_positive_files.txt"), "w") as f:
        for path in fp_df["audio_path"].tolist():
            f.write(path + "\n")

    # =========================================================
    # 7. False negative deep dive
    # =========================================================
    print("\n" + "=" * 60)
    print("7. FALSE NEGATIVE ANALYSIS")
    print("=" * 60)

    print(f"\n{len(fn_df)} false negatives total")

    if len(fn_df) > 0:
        print("\nFN by timepoint:")
        print(fn_df["timepoint_norm"].value_counts().to_string())

        print("\nFN by task type:")
        print(fn_df["task"].value_counts().to_string())

        print(f"\nFN mean KCHI duration: {fn_df['kchi_total_dur'].mean():.2f}s")
        print(f"FN mean KCHI segments: {fn_df['kchi_n_segments'].mean():.1f}")
        print(f"FN mean phonemes: {fn_df['phon_n_total'].mean():.1f}")
        print(f"FN mean prob: {fn_df['prob'].mean():.3f}")

        # Silent vs vocal FNs
        silent_fn = fn_df[fn_df["kchi_total_dur"] == 0]
        vocal_fn = fn_df[fn_df["kchi_total_dur"] > 0]
        print(f"\nFNs with zero KCHI (silent child): {len(silent_fn)} / {len(fn_df)}")
        print(f"FNs with KCHI > 0 (child speaks but missed): {len(vocal_fn)}")

        if len(vocal_fn) > 0:
            print(f"  Mean KCHI duration: {vocal_fn['kchi_total_dur'].mean():.2f}s")
            print(f"  Mean prob: {vocal_fn['prob'].mean():.3f}")

        # Do silent FNs have interaction?
        if "has_interaction" in silent_fn.columns and len(silent_fn) > 0:
            si_interact = silent_fn["has_interaction"].sum()
            print(f"\nSilent FNs with interaction: {si_interact} / {len(silent_fn)}")
            print(f"Silent FNs without interaction: {len(silent_fn) - si_interact}")

        # Do silent FNs have more children (maybe child is just listening)?
        if "#_children" in silent_fn.columns and len(silent_fn) > 0:
            print(f"Silent FN mean #_children: {silent_fn['#_children'].mean():.2f}")

        if "#_adults" in fn_df.columns:
            print(f"\nFN mean #_adults: {fn_df['#_adults'].mean():.2f}")

        fn_child_counts = fn_df["child_id"].value_counts()
        print(f"\nFN by child (top 10):")
        print(fn_child_counts.head(10).to_string())

    fn_df.to_csv(os.path.join(args.output_dir, "false_negatives.csv"), index=False)
    with open(os.path.join(args.output_dir, "false_negative_files.txt"), "w") as f:
        for path in fn_df["audio_path"].tolist():
            f.write(path + "\n")

    # =========================================================
    # 8. Per-child error rates with metadata
    # =========================================================
    print("\n" + "=" * 60)
    print("8. PER-CHILD ERROR RATES")
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
            "mean_kchi_dur": sub["kchi_total_dur"].mean(),
            "mean_phon_unique": sub["phon_n_unique"].mean(),
        }

        if "#_children" in sub.columns:
            row["mean_n_children"] = sub["#_children"].mean()
        if "#_adults" in sub.columns:
            row["mean_n_adults"] = sub["#_adults"].mean()
        if "has_interaction" in sub.columns:
            row["pct_interaction"] = sub["has_interaction"].mean()

        child_rows.append(row)

    child_df = pd.DataFrame(child_rows).sort_values("accuracy")
    child_df.to_csv(os.path.join(args.output_dir, "per_child_error_rates.csv"), index=False)

    display_cols = ["child_id", "timepoint", "n_clips", "accuracy", "n_fp", "n_fn",
                    "mean_kchi_dur", "mean_phon_unique"]
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
    print("9. TASK-TYPE BREAKDOWN")
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
            "mean_kchi_dur": sub["kchi_total_dur"].mean(),
        }
        if "#_children" in sub.columns:
            row["mean_n_children"] = sub["#_children"].mean()
        if "has_interaction" in sub.columns:
            row["pct_interaction"] = sub["has_interaction"].mean()

        task_rows.append(row)

    task_df = pd.DataFrame(task_rows).sort_values("accuracy")
    task_df.to_csv(os.path.join(args.output_dir, "task_type_breakdown.csv"), index=False)
    print(task_df.to_string(index=False))

    # =========================================================
    # 10. Confidence analysis
    # =========================================================
    print("\n" + "=" * 60)
    print("10. CONFIDENCE ANALYSIS")
    print("=" * 60)

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
    conf_df.to_csv(os.path.join(args.output_dir, "confidence_calibration.csv"), index=False)
    print("\nCalibration:")
    print(conf_df.to_string(index=False))

    uncertain = df[(df["prob"] >= 0.35) & (df["prob"] <= 0.65)]
    print(f"\nUncertain predictions (prob 0.35-0.65): {len(uncertain)} / {len(df)} "
          f"({100*len(uncertain)/len(df):.1f}%)")

    # =========================================================
    # 11. Most confident mistakes
    # =========================================================
    print("\n" + "=" * 60)
    print("11. MOST CONFIDENT MISTAKES")
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
        print(f"\n  {row['outcome']} | prob={row['prob']:.3f} | "
              f"child={row['child_id']} | tp={row['timepoint_norm']}")
        print(f"    task={row['task']} | kchi_dur={row['kchi_total_dur']:.2f}s | "
              f"kchi_segs={row['kchi_n_segments']} | phonemes={row['phon_n_total']} | "
              f"unique_phon={row['phon_n_unique']}")
        print(f"    #_children={n_kids} | #_adults={row.get('#_adults', '?')} | "
              f"interaction={interact} | #_people_interacting={row.get('#_people_interacting', '?')}")

    errors.to_csv(os.path.join(args.output_dir, "all_errors_by_confidence.csv"), index=False)

    # =========================================================
    # 12. Cross-tabulation: multi-child x interaction x outcome
    # =========================================================
    print("\n" + "=" * 60)
    print("12. MULTI-CHILD x INTERACTION CROSS-TAB")
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
        ct_df.to_csv(os.path.join(args.output_dir, "multi_child_interaction_crosstab.csv"), index=False)
        print(ct_df.to_string(index=False))

    # =========================================================
    # 13. Summary for thesis
    # =========================================================
    print("\n" + "=" * 60)
    print("13. SUMMARY FOR THESIS")
    print("=" * 60)

    n_fp = (df["outcome"] == "FP").sum()
    n_fn = (df["outcome"] == "FN").sum()
    total_errors = n_fp + n_fn
    silent_fn_count = len(fn_df[fn_df["kchi_total_dur"] == 0]) if len(fn_df) > 0 else 0

    # Multi-child FP rate
    if "#_children" in df.columns:
        neg_multi = df[(df["label"] == 0) & (df["#_children"] > 1)]
        neg_single = df[(df["label"] == 0) & (df["#_children"] <= 1)]
        fp_rate_multi = (neg_multi["outcome"] == "FP").sum() / max(len(neg_multi), 1)
        fp_rate_single = (neg_single["outcome"] == "FP").sum() / max(len(neg_single), 1)
    else:
        fp_rate_multi = fp_rate_single = float("nan")

    # Interaction FN rate
    if "has_interaction" in df.columns:
        pos_interact = df[(df["label"] == 1) & (df["has_interaction"] == True)]
        pos_no_interact = df[(df["label"] == 1) & (df["has_interaction"] == False)]
        fn_rate_interact = (pos_interact["outcome"] == "FN").sum() / max(len(pos_interact), 1)
        fn_rate_no_interact = (pos_no_interact["outcome"] == "FN").sum() / max(len(pos_no_interact), 1)
    else:
        fn_rate_interact = fn_rate_no_interact = float("nan")

    summary = {
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
        "mean_fp_kchi_dur": float(fp_df["kchi_total_dur"].mean()) if len(fp_df) > 0 else 0,
        "mean_fn_kchi_dur": float(fn_df["kchi_total_dur"].mean()) if len(fn_df) > 0 else 0,
    }

    save_json(summary, os.path.join(args.output_dir, "thesis_summary.json"))

    print(f"\n  Total test clips: {len(df)}")
    print(f"  Total errors: {total_errors} ({100*total_errors/len(df):.1f}%)")
    print(f"  False positives: {n_fp}")
    print(f"  False negatives: {n_fn}")
    print(f"    - Silent child: {silent_fn_count} ({100*silent_fn_count/max(n_fn,1):.0f}% of FNs)")
    print(f"    - Vocal child: {n_fn - silent_fn_count}")
    print(f"  FP rate multi-child clips: {fp_rate_multi:.3f}")
    print(f"  FP rate single-child clips: {fp_rate_single:.3f}")
    print(f"  FN rate with interaction: {fn_rate_interact:.3f}")
    print(f"  FN rate without interaction: {fn_rate_no_interact:.3f}")
    print(f"  Children with perfect accuracy: {(child_df['accuracy']==1.0).sum()}")
    print(f"  Children below 70% accuracy: {(child_df['accuracy']<0.7).sum()}")

    # Save full merged df for further exploration
    df.to_csv(os.path.join(args.output_dir, "full_test_with_metadata.csv"), index=False)

    print(f"\nAll outputs saved to {args.output_dir}")


if __name__ == "__main__":
    main()
