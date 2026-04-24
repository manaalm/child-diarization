"""Tune cascade thresholds on the validation set and save stage configuration.

Three-stage cascade:
  Stage 1 — VAD gate: if kchi_total_dur < vad_threshold → no child speech → final_prob=0.0
  Stage 2 — Child ID gate: if enrollment score < child_id_threshold → not target child → final_prob=child_id_score
  Stage 3 — AV fusion: pass through to GatedAVModel (existing fusion probability)

Thresholds are grid-searched on the val set maximising F1. Test set is never
used during threshold selection.

Usage:
    python av_fusion/scripts/train_cascaded_pipeline.py \\
        --feature-dir   av_fusion/av_results/manual_only/ \\
        --output-dir    av_fusion/av_results/manual_only/models/ \\
        [--vad-feature  kchi_total_dur] \\
        [--child-id-feature  prob] \\
        [--seed 42]

Outputs:
    models/cascade_thresholds.json
    cascade_val_stage_breakdown.csv
"""

import argparse
import os
import sys
from itertools import product
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import get_repo_root, save_json

_REPO = get_repo_root()

VAD_GRID = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0, 1.5, 2.0]
CHILD_ID_GRID = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def _load_config_grids(feature_dir: str) -> Tuple[List[float], List[float]]:
    """Try to load grids from av_extensions.yaml; fall back to module defaults."""
    try:
        import yaml
        cfg_path = os.path.join(_REPO, "av_fusion", "configs", "av_extensions.yaml")
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)
        return (
            cfg["cascade"]["vad_threshold_grid"],
            cfg["cascade"]["child_id_threshold_grid"],
        )
    except Exception:
        return VAD_GRID, CHILD_ID_GRID


def assign_cascade_stages(
    val_df: pd.DataFrame,
    vad_feature: str,
    child_id_feature: str,
    vad_threshold: float,
    child_id_threshold: float,
    fusion_col: str = "proba_gated_av",
) -> pd.DataFrame:
    """Assign per-clip cascade stage and final_prob.

    Stage 1: vad_speech_detected = False (kchi_total_dur < vad_threshold)
             → final_prob = 0.0
    Stage 2: child_id_score < child_id_threshold
             → final_prob = child_id_score (normalized to [0,1])
    Stage 3: reached AV fusion
             → final_prob = fusion probability (or child_id_score if fusion unavailable)

    Returns a copy of val_df with added columns:
        vad_speech_detected, vad_child_dur_sec, child_id_score,
        av_fusion_prob, cascade_stage, final_prob,
        vad_threshold, child_id_threshold
    """
    out = val_df.copy()

    # VAD signal
    if vad_feature in val_df.columns:
        vad_dur = val_df[vad_feature].fillna(0.0).values
    else:
        print(f"  WARNING: vad_feature '{vad_feature}' not found; treating all clips as speech-present",
              file=sys.stderr)
        vad_dur = np.ones(len(val_df))

    vad_speech = vad_dur >= vad_threshold  # True = speech detected

    # Child ID signal
    if child_id_feature in val_df.columns:
        child_id_score = val_df[child_id_feature].fillna(0.0).values
    else:
        print(f"  WARNING: child_id_feature '{child_id_feature}' not found; treating all clips as stage 3",
              file=sys.stderr)
        child_id_score = np.ones(len(val_df))

    # Fusion prob
    if fusion_col in val_df.columns:
        fusion_prob = val_df[fusion_col].fillna(np.nan).values
    else:
        fusion_prob = np.full(len(val_df), np.nan)

    n = len(val_df)
    cascade_stage = np.full(n, 3, dtype=int)
    final_prob = np.empty(n)
    av_fusion_prob = fusion_prob.copy()

    for i in range(n):
        if not vad_speech[i]:
            cascade_stage[i] = 1
            final_prob[i] = 0.0
        elif child_id_score[i] < child_id_threshold:
            cascade_stage[i] = 2
            final_prob[i] = float(child_id_score[i])
        else:
            cascade_stage[i] = 3
            # Use fusion prob if available, else fall back to child_id_score
            if not np.isnan(fusion_prob[i]):
                final_prob[i] = float(fusion_prob[i])
            else:
                final_prob[i] = float(child_id_score[i])

    out["vad_speech_detected"] = vad_speech
    out["vad_child_dur_sec"] = vad_dur
    out["child_id_score"] = child_id_score
    out["av_fusion_prob"] = av_fusion_prob
    out["cascade_stage"] = cascade_stage
    out["final_prob"] = final_prob
    out["vad_threshold"] = vad_threshold
    out["child_id_threshold"] = child_id_threshold
    return out


def _tune_threshold(y_true: np.ndarray, y_score: np.ndarray, grid: List[float]) -> float:
    """Return threshold from grid maximising F1."""
    best_f1, best_t = -1.0, 0.5
    for t in grid:
        preds = (y_score >= t).astype(int)
        f = f1_score(y_true, preds, zero_division=0)
        if f > best_f1:
            best_f1, best_t = f, t
    return best_t


def grid_search_thresholds(
    val_df: pd.DataFrame,
    vad_feature: str,
    child_id_feature: str,
    vad_grid: List[float],
    child_id_grid: List[float],
    fusion_col: str = "proba_gated_av",
) -> Tuple[float, float, float, float]:
    """Grid search over (vad_threshold × child_id_threshold) maximising val F1.

    Returns (best_vad_threshold, best_child_id_threshold, best_val_f1, best_val_auroc).
    """
    y_true = val_df["label"].values.astype(int)
    best_f1, best_vad_t, best_child_t = -1.0, vad_grid[0], child_id_grid[0]

    for vad_t, child_t in product(vad_grid, child_id_grid):
        staged = assign_cascade_stages(
            val_df, vad_feature, child_id_feature, vad_t, child_t, fusion_col
        )
        probs = staged["final_prob"].values
        # Tune final classification threshold
        final_t = _tune_threshold(y_true, probs, [i / 20 for i in range(1, 20)])
        preds = (probs >= final_t).astype(int)
        f = f1_score(y_true, preds, zero_division=0)
        if f > best_f1:
            best_f1 = f
            best_vad_t = vad_t
            best_child_t = child_t

    # Compute AUROC at best thresholds
    staged = assign_cascade_stages(
        val_df, vad_feature, child_id_feature, best_vad_t, best_child_t, fusion_col
    )
    probs = staged["final_prob"].values
    try:
        auroc = float(roc_auc_score(y_true, probs))
    except Exception:
        auroc = float("nan")

    return best_vad_t, best_child_t, float(best_f1), auroc


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tune cascade thresholds on val set; write cascade_thresholds.json."
    )
    parser.add_argument("--feature-dir", required=True,
                        help="Directory containing av_train.csv, av_val.csv, av_test.csv")
    parser.add_argument("--output-dir", required=True,
                        help="Directory to write models/cascade_thresholds.json")
    parser.add_argument("--vad-feature", default="kchi_total_dur",
                        help="Column in av_val.csv for VAD signal (default: kchi_total_dur)")
    parser.add_argument("--child-id-feature", default="prob",
                        help="Column in av_val.csv for enrollment score (default: prob)")
    parser.add_argument("--fusion-col", default="proba_gated_av",
                        help="Column in av_val.csv for existing fusion probability (default: proba_gated_av)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)

    feature_dir = args.feature_dir if os.path.isabs(args.feature_dir) else os.path.join(_REPO, args.feature_dir)
    output_dir = args.output_dir if os.path.isabs(args.output_dir) else os.path.join(_REPO, args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    val_csv = os.path.join(feature_dir, "av_val.csv")
    if not os.path.exists(val_csv):
        print(f"ERROR: av_val.csv not found in {feature_dir}", file=sys.stderr)
        sys.exit(1)

    val_df = pd.read_csv(val_csv, low_memory=False)
    print(f"Loaded {len(val_df)} val clips from {val_csv}")

    vad_grid, child_id_grid = _load_config_grids(feature_dir)

    print(f"Grid search: {len(vad_grid)} VAD thresholds × {len(child_id_grid)} child-ID thresholds "
          f"= {len(vad_grid) * len(child_id_grid)} combinations")

    best_vad_t, best_child_t, best_f1, best_auroc = grid_search_thresholds(
        val_df,
        args.vad_feature,
        args.child_id_feature,
        vad_grid,
        child_id_grid,
        args.fusion_col,
    )

    print(f"Best thresholds: vad_threshold={best_vad_t}, child_id_threshold={best_child_t}")
    print(f"Val F1={best_f1:.4f}  AUROC={best_auroc:.4f}")

    # Save threshold config
    thresholds = {
        "vad_threshold": best_vad_t,
        "child_id_threshold": best_child_t,
        "val_f1": best_f1,
        "val_auroc": best_auroc,
        "vad_feature": args.vad_feature,
        "child_id_feature": args.child_id_feature,
        "fusion_col": args.fusion_col,
        "seed": args.seed,
    }
    thresh_path = os.path.join(output_dir, "cascade_thresholds.json")
    save_json(thresholds, thresh_path)
    print(f"Thresholds saved to: {thresh_path}")

    # Generate val stage breakdown
    val_staged = assign_cascade_stages(
        val_df,
        args.vad_feature,
        args.child_id_feature,
        best_vad_t,
        best_child_t,
        args.fusion_col,
    )
    stage_cols = [
        "clip_id", "label", "vad_speech_detected", "vad_child_dur_sec",
        "child_id_score", "av_fusion_prob", "cascade_stage", "final_prob",
        "vad_threshold", "child_id_threshold",
    ]
    # Only keep columns that exist
    out_cols = [c for c in stage_cols if c in val_staged.columns]
    breakdown_path = os.path.join(feature_dir, "cascade_val_stage_breakdown.csv")
    val_staged[out_cols].to_csv(breakdown_path, index=False)
    print(f"Val stage breakdown saved to: {breakdown_path}")

    # Print stage distribution
    stage_counts = val_staged["cascade_stage"].value_counts().sort_index()
    print("\nVal stage distribution:")
    for stage, count in stage_counts.items():
        frac = count / len(val_staged)
        print(f"  Stage {stage}: {count} clips ({frac:.1%})")


if __name__ == "__main__":
    main()
