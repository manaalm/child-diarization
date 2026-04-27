#!/usr/bin/env python3
"""
Evaluate synthetic augmentation experiments across all ratios.

For each ratio_{r}x/ subdirectory in --experiment-dir, loads the enrollment
predictions and computes F1, Precision, Recall, AUROC, AUPRC.

Outputs:
    {output_dir}/metrics_by_ratio.csv
    {output_dir}/metrics_by_age_band.csv
    {output_dir}/figures/synthetic_ratio_vs_auprc.png  (if --plot)
    {output_dir}/figures/synthetic_ratio_vs_f1.png     (if --plot)

Usage:
    python synth/scripts/evaluate_synthetic_augmentation.py \\
      --experiment-dir  synth_results/augmentation_experiments/default_14_18mo/ \\
      --test-csv        whisper-modeling/seen_child_splits/test.csv \\
      --output-dir      synth_results/augmentation_experiments/default_14_18mo/ \\
      --plot
"""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Reuse metric utilities from the AV fusion pipeline
from av_fusion.scripts.utils import compute_metrics, save_json, save_csv


def _discover_ratio_dirs(experiment_dir: Path) -> list:
    """Return (ratio_float, ratio_str, path) tuples sorted by ratio."""
    results = []
    for d in experiment_dir.iterdir():
        if not d.is_dir():
            continue
        m = re.match(r"ratio_([0-9.]+)x$", d.name)
        if not m:
            continue
        ratio_str = m.group(1)
        try:
            ratio = float(ratio_str)
        except ValueError:
            continue
        results.append((ratio, ratio_str, d))
    return sorted(results, key=lambda x: x[0])


def _load_predictions(ratio_dir: Path, test_csv: str) -> pd.DataFrame:
    """Load test predictions from a ratio experiment directory.

    Tries (in order):
        1. test_predictions.csv in the ratio dir
        2. Recompute from test_metrics_tuned.json (column 'threshold') +
           enroll_test_predictions.csv
    """
    # Option 1: direct predictions CSV
    preds_path = ratio_dir / "test_predictions.csv"
    if preds_path.exists():
        return pd.read_csv(preds_path)

    # Option 2: enroll CSV (used by pyannote/unified.py output format)
    enroll_path = ratio_dir / "enroll_test_predictions.csv"
    if enroll_path.exists():
        return pd.read_csv(enroll_path)

    return pd.DataFrame()


def _compute_ratio_metrics(preds: pd.DataFrame, threshold: float = 0.5) -> dict:
    """Compute overall metrics from a predictions DataFrame."""
    required = {"label", "prob"}
    if not required.issubset(preds.columns):
        # Try common column names
        alt_label = next((c for c in preds.columns
                          if c.lower() in ("label", "y_true", "gt")), None)
        alt_prob = next((c for c in preds.columns
                         if c.lower() in ("prob", "score", "probability",
                                          "enrollment_score")), None)
        if alt_label and alt_prob:
            preds = preds.rename(columns={alt_label: "label", alt_prob: "prob"})
        else:
            return {}

    return compute_metrics(
        y_true=preds["label"].astype(int).values,
        y_score=preds["prob"].astype(float).values,
        threshold=threshold,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate synthetic augmentation experiments by ratio."
    )
    parser.add_argument(
        "--experiment-dir",
        required=True,
        help="Directory containing ratio_{r}x/ subdirectories.",
    )
    parser.add_argument(
        "--test-csv",
        required=True,
        help="Real held-out test CSV.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory for metrics and figures.",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate ratio vs. metric line plots.",
    )
    args = parser.parse_args()

    exp_dir = Path(args.experiment_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ratio_dirs = _discover_ratio_dirs(exp_dir)
    if not ratio_dirs:
        print(f"No ratio_*x/ subdirectories found in {exp_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(ratio_dirs)} ratio experiments.")

    test_df = pd.read_csv(args.test_csv, low_memory=False)

    # ---- Overall metrics by ratio ----
    overall_rows = []
    age_band_rows = []

    for ratio, ratio_str, ratio_dir in ratio_dirs:
        preds = _load_predictions(ratio_dir, args.test_csv)
        if preds.empty:
            print(f"  [SKIP] No predictions found in {ratio_dir}")
            continue

        # Use val-tuned threshold if available
        threshold = 0.5
        for val_metrics_name in ("enroll_val_metrics.json", "val_metrics_tuned.json"):
            val_metrics_path = ratio_dir / val_metrics_name
            if val_metrics_path.exists():
                with open(val_metrics_path) as _f:
                    _vm = json.load(_f)
                if "threshold" in _vm:
                    threshold = float(_vm["threshold"])
                    break

        metrics = _compute_ratio_metrics(preds, threshold=threshold)
        if not metrics:
            print(f"  [SKIP] Could not compute metrics for ratio={ratio_str}x")
            continue

        row = {"ratio": ratio, **metrics}
        overall_rows.append(row)
        print(
            f"  ratio={ratio_str}x: "
            f"F1={metrics.get('f1', float('nan')):.3f}  "
            f"AUROC={metrics.get('auroc', float('nan')):.3f}  "
            f"AUPRC={metrics.get('auprc', float('nan')):.3f}"
        )

        # Per-age-band metrics (requires timepoint_norm or age_band in predictions)
        for tp_col in ("timepoint_norm", "age_band"):
            if tp_col in preds.columns:
                for band, band_preds in preds.groupby(tp_col):
                    band_metrics = _compute_ratio_metrics(band_preds, threshold=threshold)
                    if band_metrics:
                        age_band_rows.append(
                            {
                                "ratio": ratio,
                                "age_band": band,
                                **band_metrics,
                            }
                        )
                break

    # ---- Write outputs ----
    if overall_rows:
        metrics_df = pd.DataFrame(overall_rows)
        save_csv(metrics_df, str(out_dir / "metrics_by_ratio.csv"))
        print(f"\nWrote {out_dir / 'metrics_by_ratio.csv'}")

    if age_band_rows:
        ab_df = pd.DataFrame(age_band_rows)
        save_csv(ab_df, str(out_dir / "metrics_by_age_band.csv"))
        print(f"Wrote {out_dir / 'metrics_by_age_band.csv'}")

    # ---- Optional plots ----
    if args.plot and overall_rows:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig_dir = out_dir / "figures"
            fig_dir.mkdir(parents=True, exist_ok=True)
            metrics_df_plot = pd.DataFrame(overall_rows).sort_values("ratio")

            for metric, ylabel in [("auprc", "AUPRC"), ("f1", "F1 Score")]:
                if metric not in metrics_df_plot.columns:
                    continue
                fig, ax = plt.subplots(figsize=(7, 4))
                ax.plot(
                    metrics_df_plot["ratio"],
                    metrics_df_plot[metric],
                    marker="o",
                    linewidth=1.5,
                )
                ax.set_xlabel("Synthetic-to-Real Ratio")
                ax.set_ylabel(ylabel)
                ax.set_title(f"{ylabel} vs. Synthetic Augmentation Ratio")
                ax.grid(alpha=0.3)
                fig_path = fig_dir / f"synthetic_ratio_vs_{metric}.png"
                fig.savefig(str(fig_path), dpi=150, bbox_inches="tight")
                plt.close(fig)
                print(f"Wrote {fig_path}")

        except ImportError:
            print("matplotlib not available; skipping plots.")


if __name__ == "__main__":
    main()
