#!/usr/bin/env python3
"""
Error analysis for synthetic-augmentation experiments.

Compares the real-only (0×) model against the best-performing synthetic ratio
and categorises every test clip into one of eight failure modes:

    real_only_fp_fixed      – false positive under 0× that became correct
    real_only_fn_fixed      – false negative under 0× that became correct
    new_fp_introduced       – new false positive introduced by synthetic data
    new_fn_introduced       – new false negative introduced by synthetic data
    short_vocalization_err  – error in ≥1 model, clip has ≤1 child vocalization
                              (uses '#_children'==1 and 'Child_of_interest_clear'=='yes'
                              as a proxy for a short, target-child-only scene)
    overlap_error           – error in ≥1 model, clip has multiple speakers
                              (#_adults > 0 and #_children > 0)
    adult_background_fp     – false positive on a label=0 clip with adults present
                              but no target-child vocalization
    unchanged_error         – both models wrong (useful for completeness)

A clip can match more than one auxiliary category (short_vocalization_err,
overlap_error, adult_background_fp).  The primary category is the comparison
outcome (first 5 rows above).

Outputs
-------
{output_dir}/error_analysis.csv     – per-clip categorisation
{output_dir}/error_counts.json      – summary counts per category

Usage
-----
    python synth/scripts/error_analysis_synthetic.py \\
      --experiment-dir synth_results/augmentation_experiments/default_14_18mo/ \\
      --test-csv       whisper-modeling/seen_child_splits/test.csv \\
      --output-dir     synth_results/augmentation_experiments/default_14_18mo/
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


# ---------------------------------------------------------------------------
# Prediction loading (mirrors evaluate_synthetic_augmentation.py)
# ---------------------------------------------------------------------------

def _load_predictions(ratio_dir: Path) -> Optional[pd.DataFrame]:
    for fname in ("test_predictions.csv", "enroll_test_predictions.csv"):
        p = ratio_dir / fname
        if p.exists():
            df = pd.read_csv(p)
            # Normalise column names
            col_map = {}
            for c in df.columns:
                if c.lower() in ("label", "y_true", "gt"):
                    col_map[c] = "label"
                elif c.lower() in ("prob", "score", "probability", "y_score"):
                    col_map[c] = "prob"
                elif c.lower() in ("clip_id", "audio_path", "filename"):
                    col_map[c] = "clip_id"
            if col_map:
                df = df.rename(columns=col_map)
            if "label" in df.columns and "prob" in df.columns:
                return df
    return None


def _find_ratio_dirs(experiment_dir: Path):
    """Return a dict {ratio_str: Path} for all ratio_*x/ subdirectories."""
    pattern = re.compile(r"ratio_(.+)x$")
    dirs = {}
    for d in sorted(experiment_dir.iterdir()):
        m = pattern.match(d.name)
        if m and d.is_dir():
            dirs[m.group(1)] = d
    return dirs


def _ratio_float(s: str) -> float:
    try:
        return float(s)
    except ValueError:
        return float("inf")


def _best_ratio_dir(ratio_dirs: dict, metric: str = "auprc") -> Optional[tuple]:
    """Return (ratio_str, Path) for the ratio with the best test metric."""
    best_ratio, best_dir, best_val = None, None, -1.0
    for r_str, d in ratio_dirs.items():
        metrics_path = d / "test_metrics_tuned.json"
        if not metrics_path.exists():
            metrics_path = d / "enroll_test_metrics.json"
        if not metrics_path.exists():
            continue
        with open(metrics_path) as f:
            m = json.load(f)
        val = m.get(metric, m.get("auprc", m.get("auroc", -1.0)))
        if val > best_val:
            best_val, best_ratio, best_dir = val, r_str, d
    return (best_ratio, best_dir) if best_ratio is not None else None


# ---------------------------------------------------------------------------
# Merge helpers
# ---------------------------------------------------------------------------

def _merge_with_test_csv(preds: pd.DataFrame, test_csv: str) -> pd.DataFrame:
    """Left-join prediction rows with test-CSV metadata."""
    test_df = pd.read_csv(test_csv, low_memory=False)
    # Find a join key: prefer audio_path, else clip_id
    if "audio_path" in preds.columns and "audio_path" in test_df.columns:
        merged = preds.merge(
            test_df.drop(columns=["label"], errors="ignore"),
            on="audio_path", how="left",
        )
    elif "clip_id" in preds.columns and "audio_path" in test_df.columns:
        merged = preds.merge(
            test_df.drop(columns=["label"], errors="ignore"),
            left_on="clip_id", right_on="audio_path", how="left",
        )
    else:
        merged = preds.copy()
    return merged


# ---------------------------------------------------------------------------
# Classification logic
# ---------------------------------------------------------------------------

def _apply_threshold(prob: pd.Series, threshold: float = 0.5) -> pd.Series:
    return (prob >= threshold).astype(int)


def _classify_clips(
    base_df: pd.DataFrame,
    best_df: pd.DataFrame,
    base_threshold: float = 0.5,
    best_threshold: float = 0.5,
) -> pd.DataFrame:
    """
    Align base (0×) and best-ratio predictions row-by-row.
    Returns a merged DataFrame with classification columns.
    """
    base = base_df[["label", "prob"]].copy()
    base.columns = ["label", "prob_base"]

    best = best_df[["prob"]].copy()
    best.columns = ["prob_best"]

    # Reset indices to align row-by-row
    base = base.reset_index(drop=True)
    best = best.reset_index(drop=True)

    df = pd.concat([base, best], axis=1)

    df["pred_base"] = _apply_threshold(df["prob_base"], base_threshold)
    df["pred_best"] = _apply_threshold(df["prob_best"], best_threshold)
    df["y"] = df["label"].astype(int)

    # Primary outcome categories
    conditions = [
        (df["pred_base"] == 1) & (df["y"] == 0) & (df["pred_best"] == df["y"]),  # fp fixed
        (df["pred_base"] == 0) & (df["y"] == 1) & (df["pred_best"] == df["y"]),  # fn fixed
        (df["pred_base"] == df["y"]) & (df["pred_best"] == 1) & (df["y"] == 0),  # new fp
        (df["pred_base"] == df["y"]) & (df["pred_best"] == 0) & (df["y"] == 1),  # new fn
        (df["pred_base"] != df["y"]) & (df["pred_best"] != df["y"]),              # both wrong
    ]
    choices = [
        "real_only_fp_fixed",
        "real_only_fn_fixed",
        "new_fp_introduced",
        "new_fn_introduced",
        "unchanged_error",
    ]
    df["primary_category"] = np.select(conditions, choices, default="unchanged_correct")

    return df


def _add_auxiliary_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Add short_vocalization_err, overlap_error, adult_background_fp flags."""
    is_error = df["primary_category"].isin(
        ["real_only_fp_fixed", "real_only_fn_fixed",
         "new_fp_introduced", "new_fn_introduced", "unchanged_error"]
    )

    # short_vocalization proxy: only one child present, label=1 → short target-child vocalization
    has_short_proxy = pd.Series(False, index=df.index)
    if "#_children" in df.columns and "Child_of_interest_clear" in df.columns:
        has_short_proxy = (
            (pd.to_numeric(df["#_children"], errors="coerce").fillna(0) <= 1)
            & (df["Child_of_interest_clear"].astype(str).str.lower() == "yes")
            & (df["y"] == 1)
        )
    elif "#_children" in df.columns:
        has_short_proxy = (pd.to_numeric(df["#_children"], errors="coerce").fillna(0) <= 1) & (df["y"] == 1)
    df["short_vocalization_err"] = (is_error & has_short_proxy).astype(int)

    # overlap proxy: both adults and children present in scene
    has_overlap = pd.Series(False, index=df.index)
    if "#_adults" in df.columns and "#_children" in df.columns:
        has_overlap = (
            (pd.to_numeric(df["#_adults"], errors="coerce").fillna(0) > 0)
            & (pd.to_numeric(df["#_children"], errors="coerce").fillna(0) > 0)
        )
    df["overlap_error"] = (is_error & has_overlap).astype(int)

    # adult_background_fp: FP on a label=0 clip with adults present
    is_fp = df["primary_category"].isin(
        ["real_only_fp_fixed", "new_fp_introduced", "unchanged_error"]
    ) & (df["y"] == 0)
    has_adults = pd.Series(False, index=df.index)
    if "#_adults" in df.columns:
        has_adults = pd.to_numeric(df["#_adults"], errors="coerce").fillna(0) > 0
    df["adult_background_fp"] = (is_fp & has_adults).astype(int)

    return df


# ---------------------------------------------------------------------------
# Per-age-band summary
# ---------------------------------------------------------------------------

def _error_by_age_band(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    band_col = next(
        (c for c in ("timepoint_norm", "age_band", "timepoint") if c in df.columns),
        None,
    )
    if band_col is None:
        return None

    rows = []
    for band, grp in df.groupby(band_col):
        n = len(grp)
        n_error_base = (grp["pred_base"] != grp["y"]).sum()
        n_error_best = (grp["pred_best"] != grp["y"]).sum()
        rows.append({
            "age_band": band,
            "n_clips": n,
            "error_rate_base": n_error_base / n if n > 0 else 0.0,
            "error_rate_best": n_error_best / n if n > 0 else 0.0,
            "delta_error": (n_error_best - n_error_base) / n if n > 0 else 0.0,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Error analysis: real-only vs. best synthetic-ratio model."
    )
    parser.add_argument("--experiment-dir", required=True,
                        help="Root dir containing ratio_0x/, ratio_1x/, … subdirs.")
    parser.add_argument("--test-csv", required=True,
                        help="Real held-out test CSV (seen_child_splits/test.csv).")
    parser.add_argument("--output-dir", required=True,
                        help="Directory to write error_analysis.csv and counts_summary.json.")
    parser.add_argument("--base-ratio", default="0",
                        help="Ratio string for the baseline model (default: '0').")
    parser.add_argument("--metric", default="auprc",
                        help="Metric used to select best ratio (default: auprc).")
    args = parser.parse_args()

    exp_dir = Path(args.experiment_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ratio_dirs = _find_ratio_dirs(exp_dir)
    if not ratio_dirs:
        print(f"No ratio_*x/ subdirectories found in {exp_dir}")
        sys.exit(1)

    print(f"Found ratio directories: {list(ratio_dirs.keys())}")

    # Load base (0×) predictions
    base_dir = ratio_dirs.get(args.base_ratio)
    if base_dir is None:
        print(f"Base ratio '{args.base_ratio}' not found; available: {list(ratio_dirs.keys())}")
        sys.exit(1)
    base_preds = _load_predictions(base_dir)
    if base_preds is None:
        print(f"Could not load predictions from {base_dir}")
        sys.exit(1)
    print(f"Loaded {len(base_preds)} base (0×) predictions from {base_dir.name}")

    # Find best ratio
    non_base = {k: v for k, v in ratio_dirs.items() if k != args.base_ratio}
    best = _best_ratio_dir(non_base, metric=args.metric)
    if best is None:
        print("Could not determine best ratio (no metrics files found); using highest numeric ratio.")
        best_str = max(non_base.keys(), key=_ratio_float, default=None)
        best = (best_str, non_base[best_str]) if best_str else None
    if best is None:
        print("No non-base ratio directories available.")
        sys.exit(1)
    best_ratio_str, best_dir = best
    best_preds = _load_predictions(best_dir)
    if best_preds is None:
        print(f"Could not load predictions from {best_dir}")
        sys.exit(1)
    print(f"Best ratio: {best_ratio_str}× ({best_dir.name}), {len(best_preds)} predictions")

    # Align row counts
    n = min(len(base_preds), len(best_preds))
    if len(base_preds) != len(best_preds):
        print(f"  Warning: prediction counts differ ({len(base_preds)} vs {len(best_preds)}); "
              f"using first {n} rows of each.")
        base_preds = base_preds.iloc[:n].reset_index(drop=True)
        best_preds = best_preds.iloc[:n].reset_index(drop=True)

    # Tune thresholds (default 0.5; use val if available)
    base_threshold = 0.5
    best_threshold = 0.5
    for d, attr in [(base_dir, "base_threshold"), (best_dir, "best_threshold")]:
        for fname in ("test_metrics_tuned.json", "enroll_test_metrics.json"):
            mp = d / fname
            if mp.exists():
                with open(mp) as f:
                    m = json.load(f)
                t = m.get("threshold", 0.5)
                if attr == "base_threshold":
                    base_threshold = t
                else:
                    best_threshold = t
                break

    # Classify
    clf = _classify_clips(base_preds, best_preds, base_threshold, best_threshold)

    # Merge metadata from test CSV
    base_with_meta = _merge_with_test_csv(base_preds, args.test_csv)
    meta_cols = [c for c in base_with_meta.columns
                 if c not in ("label", "prob") and c in base_with_meta.columns]
    for c in meta_cols:
        if c not in clf.columns:
            clf[c] = base_with_meta[c].values[:len(clf)]

    # Add auxiliary flags
    clf = _add_auxiliary_flags(clf)

    # Add best-ratio info
    clf["best_ratio"] = best_ratio_str
    clf["base_threshold"] = base_threshold
    clf["best_threshold"] = best_threshold

    # Write per-clip CSV
    out_csv = out_dir / "error_analysis.csv"
    clf.to_csv(str(out_csv), index=False)
    print(f"\nWrote {len(clf)} rows → {out_csv}")

    # Counts summary
    primary_counts = clf["primary_category"].value_counts().to_dict()
    aux_counts = {
        "short_vocalization_err": int(clf["short_vocalization_err"].sum()),
        "overlap_error": int(clf["overlap_error"].sum()),
        "adult_background_fp": int(clf["adult_background_fp"].sum()),
    }
    summary = {
        "n_clips": len(clf),
        "base_ratio": args.base_ratio,
        "best_ratio": best_ratio_str,
        "base_threshold": base_threshold,
        "best_threshold": best_threshold,
        "primary_counts": primary_counts,
        "auxiliary_counts": aux_counts,
    }

    # Per-age-band breakdown
    age_band_df = _error_by_age_band(clf)
    if age_band_df is not None:
        age_band_out = out_dir / "error_by_age_band.csv"
        age_band_df.to_csv(str(age_band_out), index=False)
        summary["error_by_age_band"] = age_band_df.to_dict(orient="records")
        print(f"Wrote age-band breakdown → {age_band_out}")

    counts_out = out_dir / "error_counts.json"
    with open(counts_out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote counts summary → {counts_out}")

    print("\n--- Primary category counts ---")
    for cat, cnt in sorted(primary_counts.items(), key=lambda x: -x[1]):
        print(f"  {cat:<28} {cnt:>5}")
    print("\n--- Auxiliary counts ---")
    for cat, cnt in aux_counts.items():
        print(f"  {cat:<28} {cnt:>5}")
    print(f"\nDone.")


if __name__ == "__main__":
    main()
