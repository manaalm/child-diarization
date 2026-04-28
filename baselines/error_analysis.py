"""
Error analysis across ALL baseline vocalization detection experiments.

Mirrors the structure of the BabAR error_analysis.py so results are directly
comparable, but adds cross-experiment persistence analysis.

Usage:
    python error_analysis.py \
        --results-dir /home/manaal/orcd/scratch/child-adult-diarization/baselines/baseline_results \
        --output-dir /home/manaal/orcd/scratch/child-adult-diarization/baselines/baseline_results/error_analysis
"""

import argparse
import glob
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


def discover_experiments(results_dir):
    """Find all experiment subdirectories that have test_predictions.csv."""
    experiments = {}
    for subdir in sorted(os.listdir(results_dir)):
        pred_path = os.path.join(results_dir, subdir, "test_predictions.csv")
        config_path = os.path.join(results_dir, subdir, "config.json")
        if os.path.isfile(pred_path):
            exp = {"name": subdir, "pred_path": pred_path}
            if os.path.isfile(config_path):
                with open(config_path) as f:
                    exp["config"] = json.load(f)
            else:
                exp["config"] = {}
            experiments[subdir] = exp
    return experiments


def load_and_annotate_predictions(exp, ann_dedup):
    """Load a single experiment's test predictions, merge metadata, classify outcomes."""
    df = pd.read_csv(exp["pred_path"])

    # Ensure pred_label exists (some runs may not have it)
    if "pred_label" not in df.columns:
        # Try to reconstruct from config threshold
        config = exp.get("config", {})
        threshold = config.get("threshold", 0.5)
        df["pred_label"] = (df["prob"] >= threshold).astype(int)

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

    # Normalize timepoint column name for consistency with BabAR analysis
    if "timepoint" in df.columns and "timepoint_norm" not in df.columns:
        df["timepoint_norm"] = df["timepoint"]

    # Derived
    if "#_children" in df.columns:
        df["multi_child"] = df["#_children"] > 1

    df["experiment"] = exp["name"]
    return df


# =============================================================
# Per-experiment analysis (mirrors BabAR error_analysis sections)
# =============================================================

def analyze_single_experiment(df, exp_name, exp_config, output_dir):
    """Run the full per-experiment analysis, consistent with BabAR error_analysis.py."""

    exp_dir = os.path.join(output_dir, "per_experiment", exp_name)
    os.makedirs(exp_dir, exist_ok=True)

    fp_df = df[df["outcome"] == "FP"].copy()
    fn_df = df[df["outcome"] == "FN"].copy()

    lines = []
    def p(msg=""):
        lines.append(msg)

    # ---- 1. Overall performance ----
    p("=" * 60)
    p(f"EXPERIMENT: {exp_name}")
    p("=" * 60)

    model_type = exp_config.get("model_type", "?")
    pooling = exp_config.get("pooling", "?")
    use_lw = exp_config.get("use_layer_weights", False)
    ptt = exp_config.get("per_timepoint_threshold", False)
    aug = exp_config.get("speed_perturb", False)
    p(f"  model={model_type} | pooling={pooling} | layer_weights={use_lw} | "
      f"per_tp_threshold={ptt} | augmentation={aug}")

    cm = confusion_matrix(df["label"], df["pred_label"])
    p(f"\nConfusion Matrix:")
    p(f"  TN={cm[0,0]}  FP={cm[0,1]}")
    p(f"  FN={cm[1,0]}  TP={cm[1,1]}")
    p(f"\nOutcome counts:")
    p(df["outcome"].value_counts().to_string())
    p(f"\nLabel balance: {df['label'].mean():.3f} positive")

    report = classification_report(df["label"], df["pred_label"], output_dict=True)
    save_json(report, os.path.join(exp_dir, "classification_report.json"))

    p("\nPer timepoint:")
    for tp, sub in df.groupby("timepoint_norm"):
        n = len(sub)
        pos_rate = sub["label"].mean()
        fp_rate = (sub["outcome"] == "FP").sum() / max((sub["label"] == 0).sum(), 1)
        fn_rate = (sub["outcome"] == "FN").sum() / max((sub["label"] == 1).sum(), 1)
        p(f"  {tp} (n={n}, pos_rate={pos_rate:.2f}): "
          f"FP_rate={fp_rate:.3f}, FN_rate={fn_rate:.3f}")

    # ---- 3. Number of children and false positives ----
    p("\n" + "=" * 60)
    p("NUMBER OF CHILDREN & FALSE POSITIVES")
    p("=" * 60)

    if "#_children" in df.columns:
        p("\nOutcome by #_children (negative clips only):")
        neg_clips = df[df["label"] == 0].copy()
        for n_kids, sub in neg_clips.groupby("#_children"):
            n = len(sub)
            n_fp = (sub["outcome"] == "FP").sum()
            fp_rate = n_fp / n if n > 0 else 0
            p(f"  #_children={n_kids}: n={n}, FPs={n_fp}, FP_rate={fp_rate:.3f}")

        p("\nMulti-child clips (>1 child) vs single-child:")
        for multi, sub in neg_clips.groupby("multi_child"):
            n = len(sub)
            n_fp = (sub["outcome"] == "FP").sum()
            fp_rate = n_fp / n if n > 0 else 0
            label = "multi-child" if multi else "single-child"
            p(f"  {label}: n={n}, FPs={n_fp}, FP_rate={fp_rate:.3f}")

    # ---- 4. Interaction with child and false negatives ----
    p("\n" + "=" * 60)
    p("INTERACTION WITH CHILD & FALSE NEGATIVES")
    p("=" * 60)

    if "has_interaction" in df.columns:
        p("\nOutcome by Interaction_with_child (positive clips only):")
        pos_clips = df[df["label"] == 1].copy()
        for interact, sub in pos_clips.groupby("has_interaction"):
            n = len(sub)
            n_fn = (sub["outcome"] == "FN").sum()
            fn_rate = n_fn / n if n > 0 else 0
            label = "interaction=yes" if interact else "interaction=no/missing"
            p(f"  {label}: n={n}, FNs={n_fn}, FN_rate={fn_rate:.3f}")

    # ---- 5. Number of adults and errors ----
    p("\n" + "=" * 60)
    p("NUMBER OF ADULTS & ERRORS")
    p("=" * 60)

    if "#_adults" in df.columns:
        p("\nOutcome by #_adults:")
        for outcome in ["TP", "TN", "FP", "FN"]:
            sub = df[df["outcome"] == outcome]
            if len(sub) > 0:
                mean_adults = sub["#_adults"].mean()
                p(f"  {outcome} (n={len(sub)}): mean #_adults={mean_adults:.2f}")

    if "#_people_interacting" in df.columns:
        p("\nOutcome by #_people_interacting:")
        for outcome in ["TP", "TN", "FP", "FN"]:
            sub = df[df["outcome"] == outcome]
            if len(sub) > 0:
                mean_pi = sub["#_people_interacting"].mean()
                p(f"  {outcome} (n={len(sub)}): mean #_people_interacting={mean_pi:.2f}")

    # ---- 6. False positive analysis ----
    p("\n" + "=" * 60)
    p("FALSE POSITIVE ANALYSIS")
    p("=" * 60)

    p(f"\n{len(fp_df)} false positives total")

    if len(fp_df) > 0:
        p("\nFP by timepoint:")
        p(fp_df["timepoint_norm"].value_counts().to_string())

        p("\nFP by task type:")
        p(fp_df["task"].value_counts().to_string())

        p(f"\nFP mean prob: {fp_df['prob'].mean():.3f}")

        if "#_children" in fp_df.columns:
            p(f"FP mean #_children: {fp_df['#_children'].mean():.2f}")
            p(f"FP with >1 child: {(fp_df['#_children'] > 1).sum()} / {len(fp_df)}")

        if "#_people_background" in fp_df.columns:
            p(f"FP mean #_people_background: {fp_df['#_people_background'].mean():.2f}")

        fp_child_counts = fp_df["child_id"].value_counts()
        p(f"\nFP by child (top 10):")
        p(fp_child_counts.head(10).to_string())

        high_conf_fp = fp_df[fp_df["prob"] >= 0.7]
        p(f"\nHigh-confidence FPs (prob >= 0.7): {len(high_conf_fp)}")
        if len(high_conf_fp) > 0 and "#_children" in high_conf_fp.columns:
            p(f"  Mean #_children: {high_conf_fp['#_children'].mean():.2f}")

    fp_df.to_csv(os.path.join(exp_dir, "false_positives.csv"), index=False)
    with open(os.path.join(exp_dir, "false_positive_files.txt"), "w") as f:
        for path in fp_df["audio_path"].tolist():
            f.write(path + "\n")

    # ---- 7. False negative analysis ----
    p("\n" + "=" * 60)
    p("FALSE NEGATIVE ANALYSIS")
    p("=" * 60)

    p(f"\n{len(fn_df)} false negatives total")

    if len(fn_df) > 0:
        p("\nFN by timepoint:")
        p(fn_df["timepoint_norm"].value_counts().to_string())

        p("\nFN by task type:")
        p(fn_df["task"].value_counts().to_string())

        p(f"\nFN mean prob: {fn_df['prob'].mean():.3f}")

        if "#_adults" in fn_df.columns:
            p(f"FN mean #_adults: {fn_df['#_adults'].mean():.2f}")

        fn_child_counts = fn_df["child_id"].value_counts()
        p(f"\nFN by child (top 10):")
        p(fn_child_counts.head(10).to_string())

    fn_df.to_csv(os.path.join(exp_dir, "false_negatives.csv"), index=False)
    with open(os.path.join(exp_dir, "false_negative_files.txt"), "w") as f:
        for path in fn_df["audio_path"].tolist():
            f.write(path + "\n")

    # ---- 8. Per-child error rates ----
    p("\n" + "=" * 60)
    p("PER-CHILD ERROR RATES")
    p("=" * 60)

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

        if "#_children" in sub.columns:
            row["mean_n_children"] = sub["#_children"].mean()
        if "#_adults" in sub.columns:
            row["mean_n_adults"] = sub["#_adults"].mean()
        if "has_interaction" in sub.columns:
            row["pct_interaction"] = sub["has_interaction"].mean()

        child_rows.append(row)

    child_df = pd.DataFrame(child_rows).sort_values("accuracy")
    child_df.to_csv(os.path.join(exp_dir, "per_child_error_rates.csv"), index=False)

    display_cols = ["child_id", "timepoint", "n_clips", "accuracy", "n_fp", "n_fn"]
    if "mean_n_children" in child_df.columns:
        display_cols.append("mean_n_children")
    if "pct_interaction" in child_df.columns:
        display_cols.append("pct_interaction")

    p(f"\nHardest children (lowest accuracy):")
    p(child_df.head(10)[display_cols].to_string(index=False))

    p(f"\nEasiest children (highest accuracy):")
    p(child_df.tail(5)[display_cols].to_string(index=False))

    # ---- 9. Task-type breakdown ----
    p("\n" + "=" * 60)
    p("TASK-TYPE BREAKDOWN")
    p("=" * 60)

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
        if "#_children" in sub.columns:
            row["mean_n_children"] = sub["#_children"].mean()
        if "has_interaction" in sub.columns:
            row["pct_interaction"] = sub["has_interaction"].mean()

        task_rows.append(row)

    task_df = pd.DataFrame(task_rows).sort_values("accuracy")
    task_df.to_csv(os.path.join(exp_dir, "task_type_breakdown.csv"), index=False)
    p(task_df.to_string(index=False))

    # ---- 10. Confidence analysis ----
    p("\n" + "=" * 60)
    p("CONFIDENCE ANALYSIS")
    p("=" * 60)

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
    conf_df.to_csv(os.path.join(exp_dir, "confidence_calibration.csv"), index=False)
    p("\nCalibration:")
    p(conf_df.to_string(index=False))

    uncertain = df[(df["prob"] >= 0.35) & (df["prob"] <= 0.65)]
    p(f"\nUncertain predictions (prob 0.35-0.65): {len(uncertain)} / {len(df)} "
      f"({100*len(uncertain)/len(df):.1f}%)")

    # ---- 11. Most confident mistakes ----
    p("\n" + "=" * 60)
    p("MOST CONFIDENT MISTAKES")
    p("=" * 60)

    errors = df[df["outcome"].isin(["FP", "FN"])].copy()
    errors["confidence"] = errors.apply(
        lambda r: r["prob"] if r["outcome"] == "FP" else (1 - r["prob"]),
        axis=1,
    )
    errors = errors.sort_values("confidence", ascending=False)

    p("\nTop 15 most confident errors:")
    for _, row in errors.head(15).iterrows():
        n_kids = row.get("#_children", "?")
        interact = row.get("interaction", "?")
        p(f"\n  {row['outcome']} | prob={row['prob']:.3f} | "
          f"child={row['child_id']} | tp={row['timepoint_norm']}")
        p(f"    task={row['task']} | "
          f"#_children={n_kids} | #_adults={row.get('#_adults', '?')} | "
          f"interaction={interact} | #_people_interacting={row.get('#_people_interacting', '?')}")

    errors.to_csv(os.path.join(exp_dir, "all_errors_by_confidence.csv"), index=False)

    # ---- 12. Cross-tabulation ----
    p("\n" + "=" * 60)
    p("MULTI-CHILD x INTERACTION CROSS-TAB")
    p("=" * 60)

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
        ct_df.to_csv(os.path.join(exp_dir, "multi_child_interaction_crosstab.csv"), index=False)
        p(ct_df.to_string(index=False))

    # ---- 13. Thesis summary ----
    n_fp = (df["outcome"] == "FP").sum()
    n_fn = (df["outcome"] == "FN").sum()
    total_errors = n_fp + n_fn

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
        "experiment": exp_name,
        "model_type": model_type,
        "pooling": pooling,
        "use_layer_weights": use_lw,
        "per_timepoint_threshold": ptt,
        "augmentation": aug,
        "total_test_clips": len(df),
        "total_errors": int(total_errors),
        "error_rate": total_errors / len(df),
        "n_false_positives": int(n_fp),
        "n_false_negatives": int(n_fn),
        "fp_rate_multi_child": float(fp_rate_multi),
        "fp_rate_single_child": float(fp_rate_single),
        "fn_rate_with_interaction": float(fn_rate_interact),
        "fn_rate_without_interaction": float(fn_rate_no_interact),
        "n_children_perfect": int((child_df["accuracy"] == 1.0).sum()),
        "n_children_below_70pct": int((child_df["accuracy"] < 0.7).sum()),
        "hardest_task": task_df.iloc[0]["task"] if len(task_df) > 0 else "N/A",
        "easiest_task": task_df.iloc[-1]["task"] if len(task_df) > 0 else "N/A",
    }

    save_json(summary, os.path.join(exp_dir, "thesis_summary.json"))

    # Save full merged df
    df.to_csv(os.path.join(exp_dir, "full_test_with_metadata.csv"), index=False)

    # Write report
    report_text = "\n".join(lines)
    with open(os.path.join(exp_dir, "report.txt"), "w") as f:
        f.write(report_text)

    return summary, df


# =============================================================
# Cross-experiment analysis
# =============================================================

def cross_experiment_analysis(all_dfs, all_summaries, output_dir):
    """Compare errors across experiments to find persistent hard cases."""

    cross_dir = os.path.join(output_dir, "cross_experiment")
    os.makedirs(cross_dir, exist_ok=True)

    lines = []
    def p(msg=""):
        lines.append(msg)

    exp_names = sorted(all_dfs.keys())
    n_experiments = len(exp_names)

    p("=" * 80)
    p(f"CROSS-EXPERIMENT ANALYSIS ({n_experiments} experiments)")
    p("=" * 80)

    # ---- 1. Summary comparison table ----
    p("\n" + "-" * 60)
    p("1. EXPERIMENT COMPARISON")
    p("-" * 60)

    summary_df = pd.DataFrame(all_summaries)
    summary_cols = [
        "experiment", "model_type", "pooling", "use_layer_weights",
        "per_timepoint_threshold", "augmentation",
        "total_errors", "error_rate", "n_false_positives", "n_false_negatives",
        "fp_rate_multi_child", "fp_rate_single_child",
        "fn_rate_with_interaction", "fn_rate_without_interaction",
    ]
    available_cols = [c for c in summary_cols if c in summary_df.columns]
    summary_df = summary_df[available_cols].sort_values("error_rate")
    summary_df.to_csv(os.path.join(cross_dir, "experiment_comparison.csv"), index=False)

    p("\nRanked by error rate:")
    p(summary_df.to_string(index=False))

    # ---- 2. Persistent false positives ----
    p("\n" + "-" * 60)
    p("2. PERSISTENT FALSE POSITIVES")
    p("-" * 60)
    p("(Files misclassified as FP across multiple experiments)")

    fp_counts = {}
    fp_exp_map = {}
    for exp_name, df in all_dfs.items():
        fps = df[df["outcome"] == "FP"]["audio_path"].tolist()
        for path in fps:
            fp_counts[path] = fp_counts.get(path, 0) + 1
            fp_exp_map.setdefault(path, []).append(exp_name)

    fp_persist = pd.DataFrame([
        {
            "audio_path": path,
            "n_experiments_fp": count,
            "pct_experiments": count / n_experiments,
            "experiments": ", ".join(sorted(exps)),
        }
        for path, count, exps in [
            (p, fp_counts[p], fp_exp_map[p]) for p in fp_counts
        ]
    ]).sort_values("n_experiments_fp", ascending=False)

    fp_persist.to_csv(os.path.join(cross_dir, "persistent_false_positives.csv"), index=False)
    with open(os.path.join(cross_dir, "persistent_false_positive_files.txt"), "w") as f:
        for path in fp_persist[fp_persist["n_experiments_fp"] >= 2]["audio_path"]:
            f.write(path + "\n")

    always_fp = fp_persist[fp_persist["n_experiments_fp"] == n_experiments]
    majority_fp = fp_persist[fp_persist["n_experiments_fp"] >= n_experiments / 2]

    p(f"\nTotal unique FP files across all experiments: {len(fp_persist)}")
    p(f"FP in ALL {n_experiments} experiments: {len(always_fp)}")
    p(f"FP in >= half of experiments: {len(majority_fp)}")

    if len(always_fp) > 0:
        # Enrich with metadata from any experiment
        ref_df = list(all_dfs.values())[0]
        p(f"\nFiles that are FP in EVERY experiment:")
        for _, row in always_fp.iterrows():
            ref_row = ref_df[ref_df["audio_path"] == row["audio_path"]]
            if len(ref_row) > 0:
                r = ref_row.iloc[0]
                n_kids = r.get("#_children", "?")
                interact = r.get("interaction", "?")
                p(f"  {row['audio_path']}")
                p(f"    child={r['child_id']} | tp={r['timepoint_norm']} | task={r['task']}")
                p(f"    #_children={n_kids} | #_adults={r.get('#_adults', '?')} | "
                  f"interaction={interact}")

    # ---- 3. Persistent false negatives ----
    p("\n" + "-" * 60)
    p("3. PERSISTENT FALSE NEGATIVES")
    p("-" * 60)
    p("(Files misclassified as FN across multiple experiments)")

    fn_counts = {}
    fn_exp_map = {}
    for exp_name, df in all_dfs.items():
        fns = df[df["outcome"] == "FN"]["audio_path"].tolist()
        for path in fns:
            fn_counts[path] = fn_counts.get(path, 0) + 1
            fn_exp_map.setdefault(path, []).append(exp_name)

    fn_persist = pd.DataFrame([
        {
            "audio_path": path,
            "n_experiments_fn": count,
            "pct_experiments": count / n_experiments,
            "experiments": ", ".join(sorted(exps)),
        }
        for path, count, exps in [
            (p, fn_counts[p], fn_exp_map[p]) for p in fn_counts
        ]
    ]).sort_values("n_experiments_fn", ascending=False)

    fn_persist.to_csv(os.path.join(cross_dir, "persistent_false_negatives.csv"), index=False)
    with open(os.path.join(cross_dir, "persistent_false_negative_files.txt"), "w") as f:
        for path in fn_persist[fn_persist["n_experiments_fn"] >= 2]["audio_path"]:
            f.write(path + "\n")

    always_fn = fn_persist[fn_persist["n_experiments_fn"] == n_experiments]
    majority_fn = fn_persist[fn_persist["n_experiments_fn"] >= n_experiments / 2]

    p(f"\nTotal unique FN files across all experiments: {len(fn_persist)}")
    p(f"FN in ALL {n_experiments} experiments: {len(always_fn)}")
    p(f"FN in >= half of experiments: {len(majority_fn)}")

    if len(always_fn) > 0:
        ref_df = list(all_dfs.values())[0]
        p(f"\nFiles that are FN in EVERY experiment:")
        for _, row in always_fn.iterrows():
            ref_row = ref_df[ref_df["audio_path"] == row["audio_path"]]
            if len(ref_row) > 0:
                r = ref_row.iloc[0]
                n_kids = r.get("#_children", "?")
                interact = r.get("interaction", "?")
                p(f"  {row['audio_path']}")
                p(f"    child={r['child_id']} | tp={r['timepoint_norm']} | task={r['task']}")
                p(f"    #_children={n_kids} | #_adults={r.get('#_adults', '?')} | "
                  f"interaction={interact}")

    # ---- 4. Persistent hard children ----
    p("\n" + "-" * 60)
    p("4. PERSISTENT HARD CHILDREN")
    p("-" * 60)

    child_accuracy_rows = []
    for exp_name, df in all_dfs.items():
        for child_id, sub in df.groupby("child_id"):
            n = len(sub)
            n_correct = ((sub["outcome"] == "TP") | (sub["outcome"] == "TN")).sum()
            child_accuracy_rows.append({
                "experiment": exp_name,
                "child_id": child_id,
                "timepoint": sub["timepoint_norm"].iloc[0],
                "n_clips": n,
                "accuracy": n_correct / n,
                "n_fp": (sub["outcome"] == "FP").sum(),
                "n_fn": (sub["outcome"] == "FN").sum(),
            })

    child_acc_df = pd.DataFrame(child_accuracy_rows)
    child_acc_df.to_csv(os.path.join(cross_dir, "per_child_per_experiment.csv"), index=False)

    # Mean accuracy across experiments per child
    child_mean = child_acc_df.groupby("child_id").agg(
        timepoint=("timepoint", "first"),
        n_clips=("n_clips", "first"),
        mean_accuracy=("accuracy", "mean"),
        std_accuracy=("accuracy", "std"),
        min_accuracy=("accuracy", "min"),
        max_accuracy=("accuracy", "max"),
        mean_n_fp=("n_fp", "mean"),
        mean_n_fn=("n_fn", "mean"),
    ).sort_values("mean_accuracy").reset_index()

    child_mean.to_csv(os.path.join(cross_dir, "per_child_mean_accuracy.csv"), index=False)

    p(f"\nHardest children (lowest mean accuracy across experiments):")
    p(child_mean.head(10).to_string(index=False))

    p(f"\nMost variable children (highest std in accuracy):")
    variable = child_mean.sort_values("std_accuracy", ascending=False)
    p(variable.head(10).to_string(index=False))

    # ---- 5. Persistent hard tasks ----
    p("\n" + "-" * 60)
    p("5. PERSISTENT HARD TASKS")
    p("-" * 60)

    task_accuracy_rows = []
    for exp_name, df in all_dfs.items():
        for task, sub in df.groupby("task"):
            n = len(sub)
            n_correct = ((sub["outcome"] == "TP") | (sub["outcome"] == "TN")).sum()
            task_accuracy_rows.append({
                "experiment": exp_name,
                "task": task,
                "n": n,
                "accuracy": n_correct / n,
                "fp_rate": (sub["outcome"] == "FP").sum() / max((sub["label"] == 0).sum(), 1),
                "fn_rate": (sub["outcome"] == "FN").sum() / max((sub["label"] == 1).sum(), 1),
            })

    task_acc_df = pd.DataFrame(task_accuracy_rows)

    task_mean = task_acc_df.groupby("task").agg(
        n=("n", "first"),
        mean_accuracy=("accuracy", "mean"),
        std_accuracy=("accuracy", "std"),
        mean_fp_rate=("fp_rate", "mean"),
        mean_fn_rate=("fn_rate", "mean"),
    ).sort_values("mean_accuracy").reset_index()

    task_mean.to_csv(os.path.join(cross_dir, "per_task_mean_accuracy.csv"), index=False)
    p(task_mean.to_string(index=False))

    # ---- 6. Pairwise experiment agreement ----
    p("\n" + "-" * 60)
    p("6. PAIRWISE EXPERIMENT AGREEMENT")
    p("-" * 60)
    p("(What fraction of predictions agree between each pair?)")

    agreement_rows = []
    for i, exp_a in enumerate(exp_names):
        for exp_b in exp_names[i+1:]:
            df_a = all_dfs[exp_a]
            df_b = all_dfs[exp_b]

            merged = df_a[["audio_path", "pred_label"]].merge(
                df_b[["audio_path", "pred_label"]],
                on="audio_path",
                suffixes=("_a", "_b"),
            )
            n = len(merged)
            agree = (merged["pred_label_a"] == merged["pred_label_b"]).sum()
            agreement_rows.append({
                "experiment_a": exp_a,
                "experiment_b": exp_b,
                "n_clips": n,
                "n_agree": int(agree),
                "agreement": agree / n if n > 0 else 0,
            })

    agreement_df = pd.DataFrame(agreement_rows).sort_values("agreement")
    agreement_df.to_csv(os.path.join(cross_dir, "pairwise_agreement.csv"), index=False)

    p(f"\nLowest agreement pairs:")
    p(agreement_df.head(10).to_string(index=False))
    p(f"\nHighest agreement pairs:")
    p(agreement_df.tail(5).to_string(index=False))

    # ---- 7. Error uniqueness: what does each experiment get right that others don't? ----
    p("\n" + "-" * 60)
    p("7. UNIQUE CONTRIBUTIONS")
    p("-" * 60)
    p("(Clips that only THIS experiment gets correct while majority get wrong)")

    for exp_name in exp_names:
        df_exp = all_dfs[exp_name]
        correct_here = set(
            df_exp[df_exp["outcome"].isin(["TP", "TN"])]["audio_path"]
        )

        # Find clips that majority of OTHER experiments get wrong
        other_exps = [e for e in exp_names if e != exp_name]
        error_counts = {}
        for other in other_exps:
            df_other = all_dfs[other]
            errors = df_other[df_other["outcome"].isin(["FP", "FN"])]["audio_path"]
            for path in errors:
                error_counts[path] = error_counts.get(path, 0) + 1

        majority_wrong = {
            path for path, count in error_counts.items()
            if count >= len(other_exps) / 2
        }

        unique_correct = correct_here & majority_wrong
        p(f"\n  {exp_name}: {len(unique_correct)} clips correct here but wrong in majority of others")

    # ---- 8. Metadata patterns in persistent errors ----
    p("\n" + "-" * 60)
    p("8. METADATA PATTERNS IN PERSISTENT ERRORS")
    p("-" * 60)

    ref_df = list(all_dfs.values())[0]

    if len(majority_fp) > 0 and "#_children" in ref_df.columns:
        persistent_fp_paths = set(majority_fp["audio_path"])
        persistent_fp_meta = ref_df[ref_df["audio_path"].isin(persistent_fp_paths)]

        p(f"\nPersistent FP metadata (FP in >= half of experiments, n={len(persistent_fp_meta)}):")
        if "#_children" in persistent_fp_meta.columns:
            p(f"  Mean #_children: {persistent_fp_meta['#_children'].mean():.2f}")
            p(f"  Multi-child: {(persistent_fp_meta['#_children'] > 1).sum()} / {len(persistent_fp_meta)}")
        if "#_adults" in persistent_fp_meta.columns:
            p(f"  Mean #_adults: {persistent_fp_meta['#_adults'].mean():.2f}")
        if "has_interaction" in persistent_fp_meta.columns:
            p(f"  Has interaction: {persistent_fp_meta['has_interaction'].sum()} / {len(persistent_fp_meta)}")
        p(f"  Timepoint distribution:")
        p("    " + persistent_fp_meta["timepoint_norm"].value_counts().to_string().replace("\n", "\n    "))

    if len(majority_fn) > 0:
        persistent_fn_paths = set(majority_fn["audio_path"])
        persistent_fn_meta = ref_df[ref_df["audio_path"].isin(persistent_fn_paths)]

        p(f"\nPersistent FN metadata (FN in >= half of experiments, n={len(persistent_fn_meta)}):")
        if "#_children" in persistent_fn_meta.columns:
            p(f"  Mean #_children: {persistent_fn_meta['#_children'].mean():.2f}")
        if "#_adults" in persistent_fn_meta.columns:
            p(f"  Mean #_adults: {persistent_fn_meta['#_adults'].mean():.2f}")
        if "has_interaction" in persistent_fn_meta.columns:
            p(f"  Has interaction: {persistent_fn_meta['has_interaction'].sum()} / {len(persistent_fn_meta)}")
        p(f"  Timepoint distribution:")
        p("    " + persistent_fn_meta["timepoint_norm"].value_counts().to_string().replace("\n", "\n    "))

    # Write cross-experiment report
    report_text = "\n".join(lines)
    with open(os.path.join(cross_dir, "cross_experiment_report.txt"), "w") as f:
        f.write(report_text)

    print(report_text)


# =============================================================
# Main
# =============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", required=True,
                        help="Path to baseline_results/ containing experiment subdirs")
    parser.add_argument("--output-dir", required=True,
                        help="Where to write error analysis outputs")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Discover experiments
    experiments = discover_experiments(args.results_dir)
    print(f"Found {len(experiments)} experiments with test predictions:")
    for name in sorted(experiments):
        print(f"  - {name}")

    if len(experiments) == 0:
        print("No experiments found. Check --results-dir path.")
        return

    # Load annotations once
    ann = load_annotations()
    meta_cols = ["audio_path"]
    for col in ["#_adults", "#_children", "#_people_background",
                 "#_people_interacting", "interaction", "has_interaction"]:
        if col in ann.columns:
            meta_cols.append(col)
    ann_dedup = ann[meta_cols].drop_duplicates(subset=["audio_path"])

    # Run per-experiment analysis
    all_summaries = []
    all_dfs = {}

    for exp_name, exp in sorted(experiments.items()):
        print(f"\nAnalyzing: {exp_name}")
        df = load_and_annotate_predictions(exp, ann_dedup)
        summary, df = analyze_single_experiment(
            df, exp_name, exp.get("config", {}), args.output_dir
        )
        all_summaries.append(summary)
        all_dfs[exp_name] = df

    # Run cross-experiment analysis
    print("\n" + "=" * 80)
    print("CROSS-EXPERIMENT ANALYSIS")
    print("=" * 80)
    cross_experiment_analysis(all_dfs, all_summaries, args.output_dir)

    print(f"\nAll outputs saved to {args.output_dir}")


if __name__ == "__main__":
    main()
