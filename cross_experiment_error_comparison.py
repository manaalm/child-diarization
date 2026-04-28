"""
Cross-experiment error comparison for child speaker identification.

Compares false positives and false negatives across all experiments
(baselines, BabAR, USC-SAIL role/enrollment, Pyannote) to find:
  - Which files are consistently misclassified across experiments
  - Which experiments share error patterns
  - Whether certain metadata (multi-child, task, interaction) predicts
    persistent errors

Usage:
    python cross_experiment_error_comparison.py \
        --base-dir /home/manaal/orcd/scratch/child-adult-diarization \
        --output-dir /home/manaal/orcd/scratch/child-adult-diarization/cross_experiment_error_analysis
"""

import argparse
import glob
import os

import numpy as np
import pandas as pd


# =========================================================
# Experiment discovery & loading
# =========================================================

def discover_experiments(base_dir):
    """
    Return a list of (experiment_name, fp_csv_path, fn_csv_path) tuples.
    """
    experiments = []

    # --- Baselines (glob over subdirectories) ---
    baseline_pattern = os.path.join(
        base_dir, "baselines", "baseline_results", "error_analysis",
        "per_experiment", "*",
    )
    for exp_dir in sorted(glob.glob(baseline_pattern)):
        if not os.path.isdir(exp_dir):
            continue
        exp_name = f"baseline_{os.path.basename(exp_dir)}"
        fp = os.path.join(exp_dir, "false_positives.csv")
        fn = os.path.join(exp_dir, "false_negatives.csv")
        if os.path.exists(fp) and os.path.exists(fn):
            experiments.append((exp_name, fp, fn))
        else:
            print(f"  WARNING: missing FP/FN csv in {exp_dir}, skipping")

    # --- BabAR ---
    babar_fp = os.path.join(base_dir, "babar_combined_runs", "error_analysis", "false_positives.csv")
    babar_fn = os.path.join(base_dir, "babar_combined_runs", "error_analysis", "false_negatives.csv")
    if os.path.exists(babar_fp) and os.path.exists(babar_fn):
        experiments.append(("babar", babar_fp, babar_fn))

    # --- USC-SAIL role_only ---
    usc_role_fp = os.path.join(base_dir, "whisper-modeling", "usc_sail_enrollment_runs", "error_analysis", "role_only", "false_positives.csv")
    usc_role_fn = os.path.join(base_dir, "whisper-modeling", "usc_sail_enrollment_runs", "error_analysis", "role_only", "false_negatives.csv")
    if os.path.exists(usc_role_fp) and os.path.exists(usc_role_fn):
        experiments.append(("usc_sail_role_only", usc_role_fp, usc_role_fn))

    # --- USC-SAIL enrollment ---
    usc_enroll_fp = os.path.join(base_dir, "whisper-modeling", "usc_sail_enrollment_runs", "error_analysis", "enrollment", "false_positives.csv")
    usc_enroll_fn = os.path.join(base_dir, "whisper-modeling", "usc_sail_enrollment_runs", "error_analysis", "enrollment", "false_negatives.csv")
    if os.path.exists(usc_enroll_fp) and os.path.exists(usc_enroll_fn):
        experiments.append(("usc_sail_enrollment", usc_enroll_fp, usc_enroll_fn))

    # --- Pyannote ---
    pyan_fp = os.path.join(base_dir, "pyannote", "pyannote_enrollment_runs", "error_analysis", "false_positives.csv")
    pyan_fn = os.path.join(base_dir, "pyannote", "pyannote_enrollment_runs", "error_analysis", "false_negatives.csv")
    if os.path.exists(pyan_fp) and os.path.exists(pyan_fn):
        experiments.append(("pyannote", pyan_fp, pyan_fn))

    return experiments


def load_error_files(experiments):
    """
    Load all FP/FN CSVs and return two DataFrames:
      fp_all: all false positives with an 'experiment' column
      fn_all: all false negatives with an 'experiment' column

    Normalises column names across the different CSV schemas.
    """
    fp_frames = []
    fn_frames = []

    # Columns we want to keep (union of useful metadata across schemas)
    keep_cols = [
        "audio_path", "child_id", "label", "prob", "pred_label", "outcome",
        "task", "session", "multi_child",
        "#_adults", "#_children", "#_people_background", "#_people_interacting",
        "interaction", "has_interaction",
        "experiment",
    ]

    # Some CSVs use 'timepoint', others 'timepoint_norm'
    tp_aliases = ["timepoint_norm", "timepoint"]

    for exp_name, fp_path, fn_path in experiments:
        for path, error_type, frames in [
            (fp_path, "FP", fp_frames),
            (fn_path, "FN", fn_frames),
        ]:
            try:
                df = pd.read_csv(path)
            except Exception as e:
                print(f"  WARNING: could not read {path}: {e}")
                continue

            df["experiment"] = exp_name

            # Normalise timepoint column
            for alias in tp_aliases:
                if alias in df.columns:
                    df["timepoint"] = df[alias]
                    break

            # Ensure multi_child exists
            if "multi_child" not in df.columns and "#_children" in df.columns:
                df["#_children"] = pd.to_numeric(df["#_children"], errors="coerce")
                df["multi_child"] = df["#_children"] > 1

            # Ensure task exists
            if "task" not in df.columns:
                df["task"] = df["audio_path"].apply(
                    lambda p: os.path.basename(p).split("task-")[1].split("_")[0]
                    if "task-" in os.path.basename(p) else "unknown"
                )

            # Keep only available columns
            available = [c for c in keep_cols + ["timepoint"] if c in df.columns]
            frames.append(df[available])

    fp_all = pd.concat(fp_frames, ignore_index=True) if fp_frames else pd.DataFrame()
    fn_all = pd.concat(fn_frames, ignore_index=True) if fn_frames else pd.DataFrame()

    return fp_all, fn_all


# =========================================================
# Analysis functions
# =========================================================

def build_file_experiment_matrix(error_df, error_type, experiments):
    """
    Build a matrix: rows = audio files, columns = experiments,
    values = 1 if the file is a FP/FN in that experiment, 0 otherwise.
    Also computes a count column.
    """
    if len(error_df) == 0:
        return pd.DataFrame()

    exp_names = [e[0] for e in experiments]

    # Pivot: for each audio_path, which experiments flagged it
    pivot = error_df.groupby(["audio_path", "experiment"]).size().unstack(fill_value=0)
    # Clip to 0/1 (shouldn't be >1 but just in case)
    pivot = (pivot > 0).astype(int)

    # Ensure all experiment columns exist
    for exp in exp_names:
        if exp not in pivot.columns:
            pivot[exp] = 0
    pivot = pivot[exp_names]

    pivot[f"{error_type}_count"] = pivot[exp_names].sum(axis=1)
    pivot = pivot.sort_values(f"{error_type}_count", ascending=False)

    # Merge in metadata from first occurrence
    meta_cols = ["audio_path", "child_id", "timepoint", "task", "session",
                 "#_children", "#_adults", "has_interaction", "multi_child"]
    available_meta = [c for c in meta_cols if c in error_df.columns]
    meta = error_df[available_meta].drop_duplicates(subset=["audio_path"])
    pivot = pivot.reset_index().merge(meta, on="audio_path", how="left")

    # Reorder: metadata first, then experiment columns, then count
    meta_first = [c for c in available_meta if c in pivot.columns]
    count_col = f"{error_type}_count"
    ordered = meta_first + [count_col] + exp_names
    pivot = pivot[[c for c in ordered if c in pivot.columns]]
    pivot = pivot.sort_values(count_col, ascending=False)

    return pivot


def persistent_error_analysis(matrix, error_type, n_experiments, output_dir):
    """
    Analyze files that are persistently misclassified across many experiments.
    """
    count_col = f"{error_type}_count"
    if count_col not in matrix.columns or len(matrix) == 0:
        return

    print(f"\n{'=' * 60}")
    print(f"PERSISTENT {error_type} ANALYSIS")
    print(f"{'=' * 60}")

    # Distribution of how many experiments each file appears in
    count_dist = matrix[count_col].value_counts().sort_index()
    print(f"\nDistribution of {error_type} count across experiments:")
    for count, n_files in count_dist.items():
        pct = 100 * n_files / len(matrix)
        bar = "█" * int(pct / 2)
        print(f"  {error_type} in {count:>2}/{n_experiments} experiments: "
              f"{n_files:>4} files ({pct:5.1f}%) {bar}")

    # Universal errors (appear in ALL experiments)
    universal = matrix[matrix[count_col] == n_experiments]
    print(f"\nUniversal {error_type}s (in ALL {n_experiments} experiments): {len(universal)}")
    if len(universal) > 0 and len(universal) <= 30:
        display = ["audio_path", "child_id", "timepoint", "task", count_col]
        if "#_children" in universal.columns:
            display.append("#_children")
        if "has_interaction" in universal.columns:
            display.append("has_interaction")
        available_display = [c for c in display if c in universal.columns]
        print(universal[available_display].to_string(index=False))

    # High-frequency errors (in >= 50% of experiments)
    threshold = max(2, n_experiments // 2)
    frequent = matrix[matrix[count_col] >= threshold]
    print(f"\nFrequent {error_type}s (in >= {threshold}/{n_experiments} experiments): "
          f"{len(frequent)}")

    # Metadata patterns in frequent errors
    if len(frequent) > 0:
        if "#_children" in frequent.columns:
            mc = frequent["#_children"]
            mc_valid = mc[mc >= 0]
            if len(mc_valid) > 0:
                print(f"  Mean #_children: {mc_valid.mean():.2f}")
                if "multi_child" in frequent.columns:
                    mc_rate = frequent["multi_child"].mean()
                    print(f"  Multi-child rate: {mc_rate:.3f}")

        if "has_interaction" in frequent.columns:
            interact_rate = frequent["has_interaction"].mean()
            print(f"  Has interaction rate: {interact_rate:.3f}")

        if "task" in frequent.columns:
            print(f"\n  Frequent {error_type}s by task:")
            print(frequent["task"].value_counts().to_string())

        if "timepoint" in frequent.columns:
            print(f"\n  Frequent {error_type}s by timepoint:")
            print(frequent["timepoint"].value_counts().to_string())

        if "child_id" in frequent.columns:
            child_counts = frequent["child_id"].value_counts()
            print(f"\n  Children with most frequent {error_type}s (top 10):")
            print(child_counts.head(10).to_string())


def per_child_cross_experiment(fp_all, fn_all, experiments, output_dir):
    """
    For each child, how many FP/FN across all experiments.
    Identifies children who are consistently hard.
    """
    print(f"\n{'=' * 60}")
    print("PER-CHILD CROSS-EXPERIMENT ERROR SUMMARY")
    print(f"{'=' * 60}")

    exp_names = [e[0] for e in experiments]
    n_exp = len(exp_names)

    # Count distinct experiments each child has FP/FN in
    child_rows = []

    all_children = set()
    if len(fp_all) > 0 and "child_id" in fp_all.columns:
        all_children |= set(fp_all["child_id"].unique())
    if len(fn_all) > 0 and "child_id" in fn_all.columns:
        all_children |= set(fn_all["child_id"].unique())

    for child_id in sorted(all_children):
        row = {"child_id": child_id}

        # FP: how many experiments have at least one FP for this child
        if len(fp_all) > 0:
            child_fp = fp_all[fp_all["child_id"] == child_id]
            row["n_fp_experiments"] = child_fp["experiment"].nunique()
            row["total_fp_instances"] = len(child_fp)
        else:
            row["n_fp_experiments"] = 0
            row["total_fp_instances"] = 0

        # FN: same
        if len(fn_all) > 0:
            child_fn = fn_all[fn_all["child_id"] == child_id]
            row["n_fn_experiments"] = child_fn["experiment"].nunique()
            row["total_fn_instances"] = len(child_fn)
        else:
            row["n_fn_experiments"] = 0
            row["total_fn_instances"] = 0

        row["total_error_experiments"] = max(row["n_fp_experiments"], row["n_fn_experiments"])

        # Timepoint from whichever df has it
        tp = "?"
        for df_check in [fp_all, fn_all]:
            if len(df_check) > 0 and "timepoint" in df_check.columns:
                child_sub = df_check[df_check["child_id"] == child_id]
                if len(child_sub) > 0:
                    tp = child_sub["timepoint"].iloc[0]
                    break
        row["timepoint"] = tp

        child_rows.append(row)

    child_df = pd.DataFrame(child_rows)
    child_df = child_df.sort_values("total_error_experiments", ascending=False)
    child_df.to_csv(os.path.join(output_dir, "per_child_cross_experiment.csv"), index=False)

    # Children with errors in ALL experiments
    always_error = child_df[child_df["total_error_experiments"] == n_exp]
    print(f"\nChildren with errors in ALL {n_exp} experiments: {len(always_error)}")
    if len(always_error) > 0:
        print(always_error.head(20).to_string(index=False))

    # Children with frequent FPs
    frequent_fp = child_df[child_df["n_fp_experiments"] >= max(2, n_exp // 2)]
    print(f"\nChildren with FPs in >= {max(2, n_exp // 2)} experiments: {len(frequent_fp)}")
    if len(frequent_fp) > 0:
        print(frequent_fp.sort_values("n_fp_experiments", ascending=False).head(15).to_string(index=False))

    # Children with frequent FNs
    frequent_fn = child_df[child_df["n_fn_experiments"] >= max(2, n_exp // 2)]
    print(f"\nChildren with FNs in >= {max(2, n_exp // 2)} experiments: {len(frequent_fn)}")
    if len(frequent_fn) > 0:
        print(frequent_fn.sort_values("n_fn_experiments", ascending=False).head(15).to_string(index=False))


def experiment_similarity(fp_matrix, fn_matrix, experiments, output_dir):
    """
    Compute Jaccard similarity between experiments based on shared errors.
    Which experiments make similar mistakes?
    """
    print(f"\n{'=' * 60}")
    print("EXPERIMENT ERROR SIMILARITY (JACCARD)")
    print(f"{'=' * 60}")

    exp_names = [e[0] for e in experiments]

    for error_type, matrix in [("FP", fp_matrix), ("FN", fn_matrix)]:
        if len(matrix) == 0:
            continue

        available_exps = [e for e in exp_names if e in matrix.columns]
        if len(available_exps) < 2:
            continue

        print(f"\n{error_type} Jaccard similarity:")

        jaccard_rows = []
        for i, exp_a in enumerate(available_exps):
            row = {"experiment": exp_a}
            set_a = set(matrix[matrix[exp_a] == 1].index)
            for exp_b in available_exps:
                set_b = set(matrix[matrix[exp_b] == 1].index)
                union = len(set_a | set_b)
                if union == 0:
                    row[exp_b] = 0.0
                else:
                    row[exp_b] = len(set_a & set_b) / union
            jaccard_rows.append(row)

        jaccard_df = pd.DataFrame(jaccard_rows).set_index("experiment")
        jaccard_df.to_csv(os.path.join(output_dir, f"{error_type.lower()}_jaccard_similarity.csv"))

        # Print in a readable format (truncate experiment names for display)
        display_df = jaccard_df.copy()
        display_df.index = [n[:20] for n in display_df.index]
        display_df.columns = [n[:20] for n in display_df.columns]
        print(display_df.round(3).to_string())

        # Find most/least similar pairs
        pairs = []
        for i, exp_a in enumerate(available_exps):
            for j, exp_b in enumerate(available_exps):
                if i < j:
                    pairs.append((exp_a, exp_b, jaccard_df.loc[exp_a, exp_b]))
        pairs.sort(key=lambda x: x[2], reverse=True)

        if pairs:
            print(f"\n  Most similar {error_type} pairs:")
            for a, b, sim in pairs[:5]:
                print(f"    {a} <-> {b}: {sim:.3f}")
            print(f"\n  Least similar {error_type} pairs:")
            for a, b, sim in pairs[-3:]:
                print(f"    {a} <-> {b}: {sim:.3f}")


def task_cross_experiment(fp_all, fn_all, experiments, output_dir):
    """
    Per-task error rates across experiments.
    """
    print(f"\n{'=' * 60}")
    print("PER-TASK CROSS-EXPERIMENT ERROR COUNTS")
    print(f"{'=' * 60}")

    for error_type, error_df in [("FP", fp_all), ("FN", fn_all)]:
        if len(error_df) == 0 or "task" not in error_df.columns:
            continue

        task_exp = error_df.groupby(["task", "experiment"]).size().unstack(fill_value=0)
        task_exp["total"] = task_exp.sum(axis=1)
        task_exp = task_exp.sort_values("total", ascending=False)
        task_exp.to_csv(os.path.join(output_dir, f"{error_type.lower()}_by_task_by_experiment.csv"))

        print(f"\n{error_type} counts by task (total across experiments):")
        print(task_exp["total"].to_string())


def timepoint_cross_experiment(fp_all, fn_all, experiments, output_dir):
    """
    Per-timepoint error counts across experiments.
    """
    print(f"\n{'=' * 60}")
    print("PER-TIMEPOINT CROSS-EXPERIMENT ERROR COUNTS")
    print(f"{'=' * 60}")

    for error_type, error_df in [("FP", fp_all), ("FN", fn_all)]:
        if len(error_df) == 0 or "timepoint" not in error_df.columns:
            continue

        tp_exp = error_df.groupby(["timepoint", "experiment"]).size().unstack(fill_value=0)
        tp_exp["total"] = tp_exp.sum(axis=1)
        tp_exp.to_csv(os.path.join(output_dir, f"{error_type.lower()}_by_timepoint_by_experiment.csv"))

        print(f"\n{error_type} by timepoint:")
        print(tp_exp.to_string())


def unique_vs_shared_errors(fp_matrix, fn_matrix, experiments, output_dir):
    """
    For each experiment, how many of its errors are unique to it
    vs shared with other experiments?
    """
    print(f"\n{'=' * 60}")
    print("UNIQUE vs SHARED ERRORS PER EXPERIMENT")
    print(f"{'=' * 60}")

    exp_names = [e[0] for e in experiments]

    rows = []
    for error_type, matrix in [("FP", fp_matrix), ("FN", fn_matrix)]:
        if len(matrix) == 0:
            continue

        count_col = f"{error_type}_count"
        if count_col not in matrix.columns:
            continue

        for exp in exp_names:
            if exp not in matrix.columns:
                continue
            exp_errors = matrix[matrix[exp] == 1]
            n_total = len(exp_errors)
            n_unique = len(exp_errors[exp_errors[count_col] == 1])
            n_shared = n_total - n_unique
            # Universal = in all experiments
            n_universal = len(exp_errors[exp_errors[count_col] == len(exp_names)])

            rows.append({
                "experiment": exp,
                "error_type": error_type,
                "total_errors": n_total,
                "unique_errors": n_unique,
                "shared_errors": n_shared,
                "universal_errors": n_universal,
                "unique_pct": 100 * n_unique / max(n_total, 1),
            })

    unique_df = pd.DataFrame(rows)
    unique_df.to_csv(os.path.join(output_dir, "unique_vs_shared_errors.csv"), index=False)

    for error_type in ["FP", "FN"]:
        sub = unique_df[unique_df["error_type"] == error_type]
        if len(sub) == 0:
            continue
        print(f"\n{error_type}:")
        print(sub.to_string(index=False))


def metadata_patterns_in_persistent_errors(fp_all, fn_all, experiments, output_dir):
    """
    For files that are persistent errors (in many experiments),
    what metadata patterns emerge?
    """
    print(f"\n{'=' * 60}")
    print("METADATA PATTERNS IN PERSISTENT ERRORS")
    print(f"{'=' * 60}")

    n_exp = len(experiments)
    threshold = max(2, n_exp // 2)

    for error_type, error_df in [("FP", fp_all), ("FN", fn_all)]:
        if len(error_df) == 0:
            continue

        # Count experiments per file
        file_counts = error_df.groupby("audio_path")["experiment"].nunique()
        persistent_files = set(file_counts[file_counts >= threshold].index)
        rare_files = set(file_counts[file_counts == 1].index)

        persistent = error_df[error_df["audio_path"].isin(persistent_files)].drop_duplicates("audio_path")
        rare = error_df[error_df["audio_path"].isin(rare_files)].drop_duplicates("audio_path")

        print(f"\n{error_type} — Persistent (>= {threshold} experiments) vs Rare (1 experiment):")
        print(f"  Persistent: {len(persistent)} files, Rare: {len(rare)} files")

        for label, sub in [("persistent", persistent), ("rare", rare)]:
            if len(sub) == 0:
                continue
            print(f"\n  {label.upper()} {error_type}s:")

            if "#_children" in sub.columns:
                valid = sub["#_children"][sub["#_children"] >= 0]
                if len(valid) > 0:
                    print(f"    Mean #_children: {valid.mean():.2f}")
            if "multi_child" in sub.columns:
                print(f"    Multi-child rate: {sub['multi_child'].mean():.3f}")
            if "has_interaction" in sub.columns:
                print(f"    Has interaction rate: {sub['has_interaction'].mean():.3f}")
            if "#_adults" in sub.columns:
                valid = sub["#_adults"][sub["#_adults"] >= 0]
                if len(valid) > 0:
                    print(f"    Mean #_adults: {valid.mean():.2f}")
            if "task" in sub.columns:
                print(f"    Task distribution:")
                for task, count in sub["task"].value_counts().head(5).items():
                    print(f"      {task}: {count}")
            if "timepoint" in sub.columns:
                print(f"    Timepoint distribution:")
                for tp, count in sub["timepoint"].value_counts().items():
                    print(f"      {tp}: {count}")


# =========================================================
# Main
# =========================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-dir",
        default="/home/manaal/orcd/scratch/child-adult-diarization",
    )
    parser.add_argument(
        "--output-dir",
        default="/home/manaal/orcd/scratch/child-adult-diarization/cross_experiment_error_analysis",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Discover all experiments
    experiments = discover_experiments(args.base_dir)
    exp_names = [e[0] for e in experiments]
    n_exp = len(experiments)

    print(f"Discovered {n_exp} experiments:")
    for name, fp, fn in experiments:
        print(f"  {name}")
    print()

    if n_exp == 0:
        print("ERROR: No experiments found. Check --base-dir.")
        return

    # Load all error files
    fp_all, fn_all = load_error_files(experiments)
    print(f"\nLoaded {len(fp_all)} FP instances across all experiments")
    print(f"Loaded {len(fn_all)} FN instances across all experiments")
    print(f"Unique FP files: {fp_all['audio_path'].nunique() if len(fp_all) > 0 else 0}")
    print(f"Unique FN files: {fn_all['audio_path'].nunique() if len(fn_all) > 0 else 0}")

    # -------------------------------------------------------
    # 1. File x Experiment matrices
    # -------------------------------------------------------
    print(f"\n{'#' * 70}")
    print("FILE x EXPERIMENT ERROR MATRICES")
    print(f"{'#' * 70}")

    fp_matrix = build_file_experiment_matrix(fp_all, "FP", experiments)
    fn_matrix = build_file_experiment_matrix(fn_all, "FN", experiments)

    if len(fp_matrix) > 0:
        fp_matrix.to_csv(os.path.join(args.output_dir, "fp_file_experiment_matrix.csv"), index=False)
        print(f"\nFP matrix: {len(fp_matrix)} unique files x {n_exp} experiments")
        print(f"  Files that are FP in ALL experiments: "
              f"{(fp_matrix['FP_count'] == n_exp).sum()}")
        print(f"  Files that are FP in only 1 experiment: "
              f"{(fp_matrix['FP_count'] == 1).sum()}")

    if len(fn_matrix) > 0:
        fn_matrix.to_csv(os.path.join(args.output_dir, "fn_file_experiment_matrix.csv"), index=False)
        print(f"\nFN matrix: {len(fn_matrix)} unique files x {n_exp} experiments")
        print(f"  Files that are FN in ALL experiments: "
              f"{(fn_matrix['FN_count'] == n_exp).sum()}")
        print(f"  Files that are FN in only 1 experiment: "
              f"{(fn_matrix['FN_count'] == 1).sum()}")

    # -------------------------------------------------------
    # 1b. Ranked most common error files
    # -------------------------------------------------------
    for error_type, matrix in [("FP", fp_matrix), ("FN", fn_matrix)]:
        if len(matrix) == 0:
            continue
        count_col = f"{error_type}_count"
        ranked = matrix.sort_values(count_col, ascending=False).reset_index(drop=True)

        display_cols = ["audio_path", count_col, "child_id", "timepoint", "task"]
        if "#_children" in ranked.columns:
            display_cols.append("#_children")
        if "has_interaction" in ranked.columns:
            display_cols.append("has_interaction")
        available = [c for c in display_cols if c in ranked.columns]

        # Print top 30 (or all if fewer)
        n_show = min(30, len(ranked))
        print(f"\n{'=' * 60}")
        print(f"TOP {n_show} MOST COMMON {error_type} FILES (ranked by # experiments)")
        print(f"{'=' * 60}")
        print(ranked.head(n_show)[available].to_string(index=False))

        # Save full ranked list
        ranked[available].to_csv(
            os.path.join(args.output_dir, f"{error_type.lower()}_ranked_by_frequency.csv"),
            index=False,
        )

    # -------------------------------------------------------
    # 2. Persistent error analysis
    # -------------------------------------------------------
    persistent_error_analysis(fp_matrix, "FP", n_exp, args.output_dir)
    persistent_error_analysis(fn_matrix, "FN", n_exp, args.output_dir)

    # -------------------------------------------------------
    # 3. Per-child cross-experiment
    # -------------------------------------------------------
    per_child_cross_experiment(fp_all, fn_all, experiments, args.output_dir)

    # -------------------------------------------------------
    # 4. Experiment similarity
    # -------------------------------------------------------
    experiment_similarity(fp_matrix, fn_matrix, experiments, args.output_dir)

    # -------------------------------------------------------
    # 5. Task breakdown
    # -------------------------------------------------------
    task_cross_experiment(fp_all, fn_all, experiments, args.output_dir)

    # -------------------------------------------------------
    # 6. Timepoint breakdown
    # -------------------------------------------------------
    timepoint_cross_experiment(fp_all, fn_all, experiments, args.output_dir)

    # -------------------------------------------------------
    # 7. Unique vs shared errors
    # -------------------------------------------------------
    unique_vs_shared_errors(fp_matrix, fn_matrix, experiments, args.output_dir)

    # -------------------------------------------------------
    # 8. Metadata patterns
    # -------------------------------------------------------
    metadata_patterns_in_persistent_errors(fp_all, fn_all, experiments, args.output_dir)

    # -------------------------------------------------------
    # 9. Grand summary
    # -------------------------------------------------------
    print(f"\n{'#' * 70}")
    print("GRAND SUMMARY")
    print(f"{'#' * 70}")

    summary = {
        "n_experiments": n_exp,
        "experiment_names": exp_names,
        "total_fp_instances": int(len(fp_all)),
        "total_fn_instances": int(len(fn_all)),
        "unique_fp_files": int(fp_all["audio_path"].nunique()) if len(fp_all) > 0 else 0,
        "unique_fn_files": int(fn_all["audio_path"].nunique()) if len(fn_all) > 0 else 0,
    }

    if len(fp_matrix) > 0:
        summary["fp_in_all_experiments"] = int((fp_matrix["FP_count"] == n_exp).sum())
        summary["fp_in_majority"] = int((fp_matrix["FP_count"] >= max(2, n_exp // 2)).sum())
        summary["fp_unique_to_one"] = int((fp_matrix["FP_count"] == 1).sum())

    if len(fn_matrix) > 0:
        summary["fn_in_all_experiments"] = int((fn_matrix["FN_count"] == n_exp).sum())
        summary["fn_in_majority"] = int((fn_matrix["FN_count"] >= max(2, n_exp // 2)).sum())
        summary["fn_unique_to_one"] = int((fn_matrix["FN_count"] == 1).sum())

    import json
    with open(os.path.join(args.output_dir, "grand_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    for k, v in summary.items():
        if k != "experiment_names":
            print(f"  {k}: {v}")

    print(f"\nAll outputs saved to {args.output_dir}")


if __name__ == "__main__":
    main()
