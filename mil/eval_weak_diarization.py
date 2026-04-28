"""Weakly-supervised frame-level evaluation of MIL attention weights.

Reads per-segment attention weight CSVs produced by the MIL sweep and evaluates
how well they correlate with ground-truth child speech from RTTM files.

Child-speaker lines are identified by labels containing CHI, KCHI, or CHILD
(case-insensitive). The ground-truth child fraction for a segment is:
    overlap(child_speech_intervals, [seg_start, seg_end]) / (seg_end - seg_start)

Metrics reported per (frontend, aggregator, timepoint):
    pearson_r      — Pearson correlation between attention weight and GT fraction
    pearson_pval
    spearman_rho   — Spearman rank correlation
    spearman_pval
    auroc_ranking  — AUROC treating GT fraction >= 0.5 as positive, weight as score
    n_segments     — number of segments evaluated
    n_clips        — number of unique clips

Usage:
    python mil/eval_weak_diarization.py \\
        --results-dir  mil/mil_results/seg_mil \\
        --split-csv    whisper-modeling/seen_child_splits/test.csv \\
        --rttm-cache   whisper-modeling/usc_sail_rttm_cache \\
        --output       mil/mil_results/seg_mil/weak_diarization_eval.csv

Exit codes:
    0 = success
    1 = no attention weight files found
"""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_CHILD_LABELS = {"chi", "kchi", "child"}


def _rttm_cache_path(audio_path: str, rttm_cache_dir: str) -> str:
    stem = Path(audio_path).stem
    md5 = hashlib.md5(audio_path.encode("utf-8")).hexdigest()
    return os.path.join(rttm_cache_dir, f"{stem}__{md5}.rttm")


def _parse_rttm(rttm_path: str) -> List[Tuple[float, float, str]]:
    """Return list of (start, end, speaker_label) from RTTM file."""
    segments = []
    with open(rttm_path) as f:
        for line in f:
            line = line.strip()
            if not line or not line.startswith("SPEAKER"):
                continue
            parts = line.split()
            if len(parts) < 9:
                continue
            try:
                start = float(parts[3])
                dur = float(parts[4])
                label = parts[7]
            except (ValueError, IndexError):
                continue
            segments.append((start, start + dur, label))
    return segments


def _child_fraction(
    seg_start: float,
    seg_end: float,
    rttm_segments: List[Tuple[float, float, str]],
) -> float:
    """Fraction of [seg_start, seg_end] covered by child-speaker RTTM segments."""
    seg_dur = seg_end - seg_start
    if seg_dur <= 0:
        return 0.0
    overlap = 0.0
    for rttm_start, rttm_end, label in rttm_segments:
        if label.lower() not in _CHILD_LABELS:
            continue
        ov = max(0.0, min(rttm_end, seg_end) - max(rttm_start, seg_start))
        overlap += ov
    return min(1.0, overlap / seg_dur)


def _compute_correlations(
    weights: np.ndarray,
    gt_fractions: np.ndarray,
) -> Dict[str, float]:
    """Pearson, Spearman, and AUROC for (weights, gt_fractions) arrays."""
    result: Dict[str, float] = {}
    if len(weights) < 5:
        return {k: float("nan") for k in ("pearson_r", "pearson_pval", "spearman_rho", "spearman_pval", "auroc_ranking")}

    r, pval = stats.pearsonr(weights, gt_fractions)
    result["pearson_r"] = float(r)
    result["pearson_pval"] = float(pval)

    rho, spval = stats.spearmanr(weights, gt_fractions)
    result["spearman_rho"] = float(rho)
    result["spearman_pval"] = float(spval)

    binary_labels = (gt_fractions >= 0.5).astype(int)
    if binary_labels.sum() > 0 and binary_labels.sum() < len(binary_labels):
        result["auroc_ranking"] = float(roc_auc_score(binary_labels, weights))
    else:
        result["auroc_ranking"] = float("nan")

    return result


def evaluate_attention_weights(
    sw_csv: str,
    rttm_cache_dir: str,
    split_df: pd.DataFrame,
) -> pd.DataFrame:
    """Evaluate one test_segment_weights.csv against RTTM ground truth.

    Returns a DataFrame with per-(timepoint) rows of correlation metrics.
    """
    sw_df = pd.read_csv(sw_csv)
    if sw_df.empty:
        return pd.DataFrame()

    # Extract frontend/aggregator from directory name
    config_dir = os.path.basename(os.path.dirname(sw_csv))
    parts = config_dir.rsplit("_", 1)
    frontend = parts[0] if len(parts) == 2 else config_dir
    aggregator = parts[1] if len(parts) == 2 else "unknown"

    # Join with split to get timepoint
    if "timepoint_norm" not in split_df.columns and "timepoint" in split_df.columns:
        split_df = split_df.rename(columns={"timepoint": "timepoint_norm"})
    audio_to_tp = dict(zip(split_df["audio_path"], split_df["timepoint_norm"]))

    # Add timepoint; skip clips not in split
    sw_df["timepoint_norm"] = sw_df["audio_path"].map(audio_to_tp)
    sw_df = sw_df.dropna(subset=["timepoint_norm"])

    # Compute GT child fraction per segment
    gt_fracs = []
    valid_mask = []
    rttm_cache: Dict[str, List] = {}

    for _, row in sw_df.iterrows():
        audio_path = row["audio_path"]
        if audio_path not in rttm_cache:
            rttm_path = _rttm_cache_path(audio_path, rttm_cache_dir)
            if os.path.exists(rttm_path):
                rttm_cache[audio_path] = _parse_rttm(rttm_path)
            else:
                rttm_cache[audio_path] = None

        segs = rttm_cache[audio_path]
        if segs is None:
            gt_fracs.append(float("nan"))
            valid_mask.append(False)
        else:
            frac = _child_fraction(float(row["seg_start"]), float(row["seg_end"]), segs)
            gt_fracs.append(frac)
            valid_mask.append(True)

    sw_df = sw_df.copy()
    sw_df["gt_child_fraction"] = gt_fracs
    sw_df = sw_df[valid_mask]

    rows = []
    for tp, grp in sw_df.groupby("timepoint_norm"):
        w = grp["attention_weight"].values.astype(float)
        gt = grp["gt_child_fraction"].values.astype(float)
        corr = _compute_correlations(w, gt)
        rows.append({
            "frontend": frontend,
            "aggregator": aggregator,
            "timepoint": tp,
            **corr,
            "n_segments": len(grp),
            "n_clips": grp["audio_path"].nunique(),
        })

    # Also compute aggregate (all timepoints)
    if len(sw_df) >= 5:
        w_all = sw_df["attention_weight"].values.astype(float)
        gt_all = sw_df["gt_child_fraction"].values.astype(float)
        corr_all = _compute_correlations(w_all, gt_all)
        rows.append({
            "frontend": frontend,
            "aggregator": aggregator,
            "timepoint": "all",
            **corr_all,
            "n_segments": len(sw_df),
            "n_clips": sw_df["audio_path"].nunique(),
        })

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate MIL attention weights against RTTM ground truth."
    )
    parser.add_argument("--results-dir", required=True,
                        help="Directory containing per-config subdirs (mil/mil_results/seg_mil)")
    parser.add_argument("--split-csv", required=True,
                        help="Test split CSV with audio_path and timepoint columns")
    parser.add_argument("--rttm-cache", required=True,
                        help="RTTM cache directory (e.g. whisper-modeling/usc_sail_rttm_cache)")
    parser.add_argument("--output", required=True,
                        help="Output CSV path for evaluation results")
    parser.add_argument("--fallback-rttm-cache", default=None,
                        help="Fallback RTTM cache if primary cache is missing for a clip")
    args = parser.parse_args()

    results_dir = os.path.join(_REPO, args.results_dir) if not os.path.isabs(args.results_dir) else args.results_dir
    rttm_cache_dir = os.path.join(_REPO, args.rttm_cache) if not os.path.isabs(args.rttm_cache) else args.rttm_cache
    split_csv = os.path.join(_REPO, args.split_csv) if not os.path.isabs(args.split_csv) else args.split_csv
    output = os.path.join(_REPO, args.output) if not os.path.isabs(args.output) else args.output

    split_df = pd.read_csv(split_csv)

    # Discover all test_segment_weights.csv files
    sw_files = []
    for subdir in sorted(os.listdir(results_dir)):
        sw_path = os.path.join(results_dir, subdir, "test_segment_weights.csv")
        if os.path.exists(sw_path):
            sw_files.append(sw_path)

    if not sw_files:
        print("ERROR: No test_segment_weights.csv files found.", file=sys.stderr)
        print(f"  Searched under: {results_dir}", file=sys.stderr)
        print("  Only attention, gated_attention, and transformer configs produce these files.",
              file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(sw_files)} attention-variant config(s) to evaluate.", flush=True)

    all_rows = []
    for sw_csv in sw_files:
        config_name = os.path.basename(os.path.dirname(sw_csv))
        print(f"  Evaluating: {config_name}", flush=True)
        result_df = evaluate_attention_weights(sw_csv, rttm_cache_dir, split_df.copy())
        if not result_df.empty:
            all_rows.append(result_df)
        else:
            print(f"    WARNING: no valid segments found for {config_name}", file=sys.stderr)

    if not all_rows:
        print("ERROR: No results produced — check RTTM cache path.", file=sys.stderr)
        sys.exit(1)

    out_df = pd.concat(all_rows, ignore_index=True)
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    out_df.to_csv(output, index=False)

    print(f"\nResults written to: {output}", flush=True)
    print(out_df[["frontend", "aggregator", "timepoint", "pearson_r", "spearman_rho", "auroc_ranking", "n_segments"]].to_string(index=False))


if __name__ == "__main__":
    main()
