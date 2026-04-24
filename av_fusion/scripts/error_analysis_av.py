"""Categorize test clips by audio-vs-AV failure mode for thesis error analysis.

Error mode categories:
  av_helped_fp:       audio pred=1 (FP), gated_av pred=0 (TN) — video corrects false alarm
  av_helped_fn:       audio pred=0 (FN), gated_av pred=1 (TP) — video finds missed child
  av_hurt_fp:         audio pred=0 or TN, gated_av pred=1 (FP) — video introduces false alarm
  av_hurt_fn:         audio pred=1 or TP, gated_av pred=0 (FN) — video misses found child
  off_camera_miss:    label=1 AND high off_camera_likely_score AND gated_av wrong
  multi_face_ambiguous: multiple faces AND both models wrong on positive clip

Usage:
    python av_fusion/scripts/error_analysis_av.py \\
        --predictions-csv av_fusion/av_results/run1/predictions_test.csv \\
        --feature-dir     av_fusion/av_results/run1/ \\
        --output-dir      av_fusion/av_results/run1/ \\
        [--n-examples     20]

Exit codes:
    0 = success (even if some categories have 0 examples)
    1 = predictions CSV not found
"""

import argparse
import os
import sys
from typing import Dict, List

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import get_repo_root, save_json

_REPO = get_repo_root()

_FEATURE_COLS = [
    "visual_eligible",
    "visual_eligibility_score",
    "off_camera_likely_score",
    "multi_person_clip",
    "child_visible_score",
    "manual_face_visibility_norm",
    "manual_quality_norm",
    "child_of_interest_clear_binary",
    "n_face_tracks",
    "existing_audio_score",
]


def _delta(row: pd.Series) -> float:
    """Absolute difference between audio-only and gated AV probability (salience proxy)."""
    a = row.get("proba_audio_only", 0.5)
    g = row.get("proba_gated_av", 0.5)
    return abs(float(a) - float(g)) if pd.notna(a) and pd.notna(g) else 0.0


def assign_error_modes(df: pd.DataFrame) -> pd.DataFrame:
    """Assign error mode label(s) to each test clip. Returns copy with 'error_mode' column."""
    df = df.copy()

    a_pred = df.get("pred_audio_only", pd.Series([-1] * len(df)))
    g_pred = df.get("pred_gated_av", pd.Series([-1] * len(df)))
    y = df["label"].astype(int)

    modes: List[str] = []
    for i in range(len(df)):
        row_modes = []
        ap, gp, yi = int(a_pred.iloc[i]), int(g_pred.iloc[i]), int(y.iloc[i])

        # AV-helped: audio wrong, gated correct
        if ap == 1 and yi == 0 and gp == 0:
            row_modes.append("av_helped_fp")
        if ap == 0 and yi == 1 and gp == 1:
            row_modes.append("av_helped_fn")

        # AV-hurt: audio right, gated wrong
        if ap == 0 and yi == 0 and gp == 1:
            row_modes.append("av_hurt_fp")
        if ap == 1 and yi == 1 and gp == 0:
            row_modes.append("av_hurt_fn")

        # Off-camera miss
        off_cam = float(df.iloc[i].get("off_camera_likely_score", 0.0) or 0.0)
        if yi == 1 and off_cam > 0.7 and gp == 0:
            row_modes.append("off_camera_miss")

        # Multi-face ambiguous
        multi = int(df.iloc[i].get("multi_person_clip", 0) or 0)
        if multi == 1 and yi == 1 and ap != yi and gp != yi:
            row_modes.append("multi_face_ambiguous")

        modes.append("|".join(row_modes) if row_modes else "correct")

    df["error_mode"] = modes
    return df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Categorize test clips by audio-vs-AV failure mode."
    )
    parser.add_argument("--predictions-csv", required=True)
    parser.add_argument("--feature-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--n-examples", type=int, default=20)
    args = parser.parse_args()

    preds_path = args.predictions_csv if os.path.isabs(args.predictions_csv) else os.path.join(_REPO, args.predictions_csv)
    feat_dir = args.feature_dir if os.path.isabs(args.feature_dir) else os.path.join(_REPO, args.feature_dir)
    out_dir = args.output_dir if os.path.isabs(args.output_dir) else os.path.join(_REPO, args.output_dir)
    os.makedirs(out_dir, exist_ok=True)

    if not os.path.exists(preds_path):
        print(f"ERROR: predictions CSV not found: {preds_path}", file=sys.stderr)
        sys.exit(1)

    pred_df = pd.read_csv(preds_path, low_memory=False)

    # Merge feature context from av_test.csv
    feat_csv = os.path.join(feat_dir, "av_test.csv")
    if os.path.exists(feat_csv):
        feat_df = pd.read_csv(feat_csv, low_memory=False)
        available_feat_cols = [c for c in _FEATURE_COLS if c in feat_df.columns and c not in pred_df.columns]
        if "clip_id" in feat_df.columns and "clip_id" in pred_df.columns:
            pred_df = pred_df.merge(feat_df[["clip_id"] + available_feat_cols], on="clip_id", how="left")
        else:
            for col in available_feat_cols:
                pred_df[col] = feat_df[col].values[:len(pred_df)] if len(feat_df) == len(pred_df) else float("nan")

    # Assign error modes
    pred_df = assign_error_modes(pred_df)
    pred_df["proba_delta"] = pred_df.apply(_delta, axis=1)

    # Collect examples per mode
    all_categories = [
        "av_helped_fp", "av_helped_fn", "av_hurt_fp", "av_hurt_fn",
        "off_camera_miss", "multi_face_ambiguous",
    ]
    output_cols = [
        "clip_id", "child_id", "age_band", "error_mode", "label",
        "proba_audio_only", "proba_gated_av", "proba_delta",
    ] + [c for c in _FEATURE_COLS if c in pred_df.columns]
    output_cols = [c for c in output_cols if c in pred_df.columns]

    summary: Dict[str, Dict] = {}
    example_rows = []

    for mode in all_categories:
        subset = pred_df[pred_df["error_mode"].str.contains(mode, regex=False, na=False)]
        n = len(subset)
        if n > 0:
            top = subset.nlargest(args.n_examples, "proba_delta")
            example_rows.append(top[output_cols])
        summary[mode] = {
            "n_clips": n,
            "mean_proba_delta": float(subset["proba_delta"].mean()) if n > 0 else float("nan"),
            "mean_visual_eligibility": float(subset["visual_eligibility_score"].mean()) if n > 0 and "visual_eligibility_score" in subset.columns else float("nan"),
        }
        status = "✓" if n > 0 else "–"
        print(f"  {status} {mode}: {n} clips")

    if example_rows:
        examples_df = pd.concat(example_rows, ignore_index=True)
    else:
        examples_df = pd.DataFrame(columns=output_cols)

    examples_df.to_csv(os.path.join(out_dir, "error_analysis_examples.csv"), index=False)
    save_json(summary, os.path.join(out_dir, "error_analysis_summary.json"))

    correct = int((pred_df["error_mode"] == "correct").sum())
    print(f"\n  Correctly classified (no error mode): {correct}/{len(pred_df)}")
    print(f"\nError analysis written to: {out_dir}")


if __name__ == "__main__":
    main()
