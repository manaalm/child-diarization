"""Apply temporal smoothing to raw clip-level predictions within recording sessions.

Smoothing is always scoped within (child_id, session_id) groups and is applied
to raw probabilities (not binary predictions) to preserve threshold flexibility.

Methods:
  gaussian      — scipy.ndimage.gaussian_filter1d; param = bandwidth (sigma)
  majority_vote — rolling mode vote; param = window size (int, odd recommended)
  moving_average — pd.Series.rolling mean; param = window size (int)

Parameter selection:
  If --param None (default), bandwidth/window is auto-tuned on the val set
  by grid-searching values from av_extensions.yaml and maximising val F1.

Usage:
    python av_fusion/scripts/smooth_predictions.py \\
        --predictions     av_fusion/av_results/run1/predictions_test.csv \\
        --val-predictions av_fusion/av_results/run1/predictions_val.csv \\
        --output          av_fusion/av_results/run1/predictions_test_smoothed.csv \\
        --method          gaussian \\
        --param           None \\
        --group-cols      child_id,timepoint_norm
"""

import argparse
import os
import sys
from statistics import mode as _mode
from typing import List, Optional

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from sklearn.metrics import f1_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import get_repo_root

_REPO = get_repo_root()

_GAUSSIAN_GRID = [0.5, 1.0, 1.5, 2.0, 3.0]
_WINDOW_GRID = [3, 5, 7]


def _load_grids(method: str):
    try:
        import yaml
        cfg_path = os.path.join(_REPO, "av_fusion", "configs", "av_extensions.yaml")
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)
        sm = cfg.get("temporal_smoothing", {})
        if method == "gaussian":
            return sm.get("gaussian_bandwidth_grid", _GAUSSIAN_GRID)
        elif method == "majority_vote":
            return sm.get("majority_vote_window_grid", _WINDOW_GRID)
        else:
            return sm.get("moving_average_window_grid", _WINDOW_GRID)
    except Exception:
        return _GAUSSIAN_GRID if method == "gaussian" else _WINDOW_GRID


def _smooth_group(probs: np.ndarray, method: str, param: float) -> np.ndarray:
    """Apply smoothing to a 1-D probability array for one session group."""
    if len(probs) == 1:
        return probs.copy()

    if method == "gaussian":
        return gaussian_filter1d(probs, sigma=float(param))

    window = max(1, int(param))
    if method == "moving_average":
        s = pd.Series(probs).rolling(window=window, center=True, min_periods=1).mean()
        return s.values

    if method == "majority_vote":
        out = probs.copy()
        half = window // 2
        for i in range(len(probs)):
            lo, hi = max(0, i - half), min(len(probs), i + half + 1)
            window_vals = probs[lo:hi]
            preds = (window_vals >= 0.5).astype(int)
            try:
                vote = float(_mode(preds))
            except Exception:
                vote = float(np.mean(preds) >= 0.5)
            # Preserve magnitude: use mean of window probs for the winning class
            mask_win = preds == int(vote)
            out[i] = float(window_vals[mask_win].mean()) if mask_win.any() else probs[i]
        return out

    raise ValueError(f"Unknown smoothing method: {method}")


def _apply_smoothing(df: pd.DataFrame, group_cols: List[str], method: str, param: float) -> np.ndarray:
    """Apply smoothing within each group; return smoothed prob array aligned with df index."""
    prob_col = "prob" if "prob" in df.columns else df.columns[df.columns.str.contains("prob")][0]
    probs = df[prob_col].fillna(0.5).values.copy()
    out = probs.copy()

    sort_col = "clip_position" if "clip_position" in df.columns else "clip_id"
    available_groups = [c for c in group_cols if c in df.columns]

    if not available_groups:
        print("  WARNING: group columns not found in predictions CSV; applying global smoothing", file=sys.stderr)
        out = _smooth_group(probs, method, param)
        return out

    for _, group_idx in df.groupby(available_groups).groups.items():
        group_df = df.loc[group_idx].copy()
        if sort_col in group_df.columns:
            group_df = group_df.sort_values(sort_col)
        group_probs = group_df[prob_col].fillna(0.5).values
        smoothed = _smooth_group(group_probs, method, param)
        # Write back in original order
        for orig_idx, s_val in zip(group_df.index, smoothed):
            out[df.index.get_loc(orig_idx)] = s_val

    return out


def _tune_param(val_df: pd.DataFrame, group_cols: List[str], method: str, grid: List[float]) -> float:
    """Grid-search param on val set maximising F1."""
    y = val_df["label"].values.astype(int)
    prob_col = "prob" if "prob" in val_df.columns else "prob_raw"
    val_df = val_df.copy()
    if prob_col not in val_df.columns:
        # Use first column containing "prob"
        candidates = [c for c in val_df.columns if "prob" in c.lower()]
        prob_col = candidates[0] if candidates else val_df.columns[-1]

    best_f1, best_param = -1.0, grid[0]
    for param in grid:
        smoothed = _apply_smoothing(val_df.rename(columns={prob_col: "prob"}) if prob_col != "prob" else val_df,
                                    group_cols, method, param)
        # Tune final threshold on val
        best_t = 0.5
        for t in [i / 20 for i in range(1, 20)]:
            preds = (smoothed >= t).astype(int)
            f = float(f1_score(y, preds, zero_division=0))
            if f > best_f1:
                best_f1 = f
                best_t = t
                best_param = param

    return best_param


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apply temporal smoothing to raw clip-level predictions."
    )
    parser.add_argument("--predictions", required=True,
                        help="Path to predictions CSV (must have 'prob' and 'label' columns)")
    parser.add_argument("--output", required=True,
                        help="Output path for smoothed predictions CSV")
    parser.add_argument("--method", default="gaussian",
                        choices=["gaussian", "majority_vote", "moving_average"],
                        help="Smoothing method (default: gaussian)")
    parser.add_argument("--param", default=None,
                        help="Bandwidth (gaussian) or window size (other). 'None' = auto-tune on val.")
    parser.add_argument("--val-predictions", default=None,
                        help="Val predictions CSV for auto-tuning (required if --param None)")
    parser.add_argument("--group-cols", default="child_id,timepoint_norm",
                        help="Comma-separated columns defining session groups (default: child_id,timepoint_norm)")
    args = parser.parse_args()

    pred_path = args.predictions if os.path.isabs(args.predictions) else os.path.join(_REPO, args.predictions)
    out_path = args.output if os.path.isabs(args.output) else os.path.join(_REPO, args.output)

    if not os.path.exists(pred_path):
        print(f"ERROR: predictions CSV not found: {pred_path}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(pred_path, low_memory=False)
    group_cols = [c.strip() for c in args.group_cols.split(",")]

    # Determine param
    if args.param is None or args.param.lower() == "none":
        if args.val_predictions is None:
            print("ERROR: --val-predictions is required when --param is None (auto-tuning)", file=sys.stderr)
            sys.exit(1)
        val_path = (args.val_predictions if os.path.isabs(args.val_predictions)
                    else os.path.join(_REPO, args.val_predictions))
        if not os.path.exists(val_path):
            print(f"ERROR: val predictions CSV not found: {val_path}", file=sys.stderr)
            sys.exit(1)
        val_df = pd.read_csv(val_path, low_memory=False)
        grid = _load_grids(args.method)
        print(f"Auto-tuning {args.method} param on val set (grid: {grid})...")
        param = _tune_param(val_df, group_cols, args.method, grid)
        print(f"  Best param: {param}")

        # Report val F1 before/after smoothing
        prob_col_val = "prob" if "prob" in val_df.columns else "prob_raw"
        y_val = val_df["label"].values.astype(int) if "label" in val_df.columns else None
        if y_val is not None:
            raw_probs_val = val_df[prob_col_val].fillna(0.5).values
            raw_t = 0.5
            best_raw_f1 = max(f1_score(y_val, (raw_probs_val >= t).astype(int), zero_division=0)
                              for t in [i / 20 for i in range(1, 20)])
            smoothed_val = _apply_smoothing(
                val_df.rename(columns={prob_col_val: "prob"}) if prob_col_val != "prob" else val_df,
                group_cols, args.method, param
            )
            best_smooth_f1 = max(f1_score(y_val, (smoothed_val >= t).astype(int), zero_division=0)
                                 for t in [i / 20 for i in range(1, 20)])
            print(f"  Val F1 raw={best_raw_f1:.4f} → smoothed={best_smooth_f1:.4f} ({args.method}, param={param})")
    else:
        param = float(args.param)

    # Apply smoothing to test predictions
    smoothed = _apply_smoothing(df, group_cols, args.method, param)

    # Build output dataframe
    prob_col = "prob" if "prob" in df.columns else [c for c in df.columns if "prob" in c.lower()][0]
    out_df = df.copy()
    out_df["prob_raw"] = df[prob_col].values
    out_df["prob_smoothed"] = np.clip(smoothed, 0.0, 1.0)
    out_df["smoothing_method"] = args.method
    out_df["smoothing_param"] = float(param)

    # Add session_id and clip_position if missing
    if "session_id" not in out_df.columns:
        available_group = [c for c in group_cols if c in out_df.columns]
        if available_group:
            out_df["session_id"] = out_df[available_group].astype(str).agg("_".join, axis=1)
    if "clip_position" not in out_df.columns and "clip_id" in out_df.columns:
        out_df["clip_position"] = (
            out_df.groupby([c for c in group_cols if c in out_df.columns]).cumcount()
        )

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    out_df.to_csv(out_path, index=False)
    print(f"\nSmoothed predictions written to: {out_path}")
    print(f"  Method: {args.method}, param: {param}")
    print(f"  Clips: {len(out_df)}")

    # Count isolated sign changes (SC-002 metric)
    if "label" in out_df.columns:
        raw_binary = (out_df["prob_raw"].fillna(0.5).values >= 0.5).astype(int)
        smooth_binary = (out_df["prob_smoothed"].values >= 0.5).astype(int)

        def _isolated_flips(arr: np.ndarray) -> int:
            if len(arr) < 3:
                return 0
            return int(np.sum(
                (arr[1:-1] != arr[:-2]) & (arr[1:-1] != arr[2:])
            ))

        raw_flips = _isolated_flips(raw_binary)
        smooth_flips = _isolated_flips(smooth_binary)
        print(f"  Isolated single-clip sign changes: raw={raw_flips} → smoothed={smooth_flips}")
        if raw_flips > 0:
            reduction = (raw_flips - smooth_flips) / raw_flips
            print(f"  Reduction: {reduction:.1%}")


if __name__ == "__main__":
    main()
