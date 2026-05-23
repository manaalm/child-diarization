"""Group-stratified 3-fold evaluation for zero-shot / non-trained systems.

For each (system, fold) pair: load the system's val/test predictions, filter to
the fold's val children for threshold tuning, evaluate on the fold's test
children. No retraining — the score column is constant across folds (zero-shot
or pre-trained pipeline). Output per-fold metrics + predictions in a fold dir
named `<system>_groupstrat3_f<fold>`.

Covers: audio LLMs (Qwen2-Audio, Qwen2.5-Omni, Qwen3-Omni-Thinking), scene-
analysis baselines (YAMNet, AST), joint_asr_diar, CLAP/PANNs (if present), and
the manual-only AV fusion late-stack scores.

Usage:
    python evaluation/groupstrat_zero_shot_eval.py
"""
from __future__ import annotations

import json
import os
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


_REPO = Path("/orcd/scratch/orcd/008/manaal/child-adult-diarization")
_SPLIT_ROOT = _REPO / "whisper-modeling" / "seen_child_splits_groupstrat_3fold"
_OUT_ROOT_FOR = {
    "baselines/scene_analysis_runs/yamnet": _REPO / "baselines" / "scene_analysis_runs" / "yamnet",
    "baselines/scene_analysis_runs/ast": _REPO / "baselines" / "scene_analysis_runs" / "ast",
    "baselines/audio_llm_baseline_runs/qwen2_audio_7b": _REPO / "baselines" / "audio_llm_baseline_runs" / "qwen2_audio_7b",
    "baselines/audio_llm_baseline_runs/qwen25_omni_7b": _REPO / "baselines" / "audio_llm_baseline_runs" / "qwen25_omni_7b",
    "baselines/audio_llm_baseline_runs/qwen3_omni_30b_thinking": _REPO / "baselines" / "audio_llm_baseline_runs" / "qwen3_omni_30b_thinking",
    "joint_asr_diar_ecapa_enrollment_runs": _REPO / "joint_asr_diar_ecapa_enrollment_runs",
}


def _compute_metrics(y_true, y_prob, threshold):
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "auroc": float(roc_auc_score(y_true, y_prob)) if y_true.sum() > 0 and y_true.sum() < len(y_true) else float("nan"),
        "auprc": float(average_precision_score(y_true, y_prob)) if y_true.sum() > 0 else float("nan"),
        "threshold": float(threshold),
        "n": int(len(y_true)),
    }


def _tune_threshold_ba(y_true, y_prob):
    """Sweep thresholds 0.05–0.95 step 0.05, pick the one with max balanced accuracy."""
    if y_true.sum() == 0 or y_true.sum() == len(y_true):
        return 0.5
    best_thr, best_ba = 0.5, -1.0
    for thr in np.arange(0.05, 1.0, 0.05):
        y_pred = (y_prob >= thr).astype(int)
        ba = balanced_accuracy_score(y_true, y_pred)
        if ba > best_ba or (ba == best_ba and abs(thr - 0.5) < abs(best_thr - 0.5)):
            best_thr, best_ba = float(thr), float(ba)
    return best_thr


def _load_predictions(system_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Try several conventional names for val/test predictions."""
    val_candidates = [
        system_dir / "val_predictions.csv",
        system_dir / "enroll_val_predictions.csv",
    ]
    test_candidates = [
        system_dir / "test_predictions.csv",
        system_dir / "enroll_test_predictions.csv",
    ]
    val_path = next((p for p in val_candidates if p.exists()), None)
    test_path = next((p for p in test_candidates if p.exists()), None)
    if val_path is None or test_path is None:
        return None, None
    return pd.read_csv(val_path), pd.read_csv(test_path)


def _fold_child_sets(fold: int) -> tuple[set, set]:
    fold_dir = _SPLIT_ROOT / f"fold_{fold}"
    val_df = pd.read_csv(fold_dir / "val.csv")
    test_df = pd.read_csv(fold_dir / "test.csv")
    return (
        set(val_df["child_id"].astype(str)),
        set(test_df["child_id"].astype(str)),
    )


def _score_col(df: pd.DataFrame) -> str:
    for c in ("score", "prob", "y_prob", "prediction_prob"):
        if c in df.columns:
            return c
    raise ValueError(f"no score column among: {list(df.columns)}")


def evaluate_system(system_rel: str, out_root: Path, n_folds: int = 3) -> list[dict]:
    system_dir = _REPO / system_rel
    val_df, test_df = _load_predictions(system_dir)
    if val_df is None:
        print(f"  SKIP {system_rel}: no val/test_predictions.csv")
        return []
    if "child_id" not in val_df.columns or "child_id" not in test_df.columns:
        print(f"  SKIP {system_rel}: missing child_id column")
        return []
    if "label" not in val_df.columns or "label" not in test_df.columns:
        print(f"  SKIP {system_rel}: missing label column")
        return []

    val_score_col = _score_col(val_df)
    test_score_col = _score_col(test_df)

    results = []
    for fold in range(n_folds):
        val_kids, test_kids = _fold_child_sets(fold)
        val_sub = val_df[val_df["child_id"].astype(str).isin(val_kids)]
        test_sub = test_df[test_df["child_id"].astype(str).isin(test_kids)]
        if len(val_sub) == 0 or len(test_sub) == 0:
            # System's predictions might be on a different split (e.g., cross-child).
            # Pool val and test sources, then filter.
            pool = pd.concat([val_df, test_df], ignore_index=True)
            val_sub = pool[pool["child_id"].astype(str).isin(val_kids)]
            test_sub = pool[pool["child_id"].astype(str).isin(test_kids)]
        if len(val_sub) == 0 or len(test_sub) == 0:
            print(f"  fold {fold} for {system_rel}: empty after filter — skipping")
            continue

        thr = _tune_threshold_ba(
            val_sub["label"].astype(int).to_numpy(),
            val_sub[val_score_col].astype(float).to_numpy(),
        )
        metrics = _compute_metrics(
            test_sub["label"].astype(int).to_numpy(),
            test_sub[test_score_col].astype(float).to_numpy(),
            thr,
        )

        out_dir = out_root.parent / f"{out_root.name}_groupstrat3_f{fold}"
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "test_metrics_tuned.json", "w") as f:
            json.dump(metrics, f, indent=2)

        test_out = test_sub.copy()
        test_out["score"] = test_sub[test_score_col].astype(float)
        test_out["prediction"] = (test_out["score"] >= thr).astype(int)
        test_out[["audio_path", "child_id", "timepoint_norm", "label", "score", "prediction"]
                 ].to_csv(out_dir / "test_predictions.csv", index=False) if all(
            c in test_out.columns for c in ("audio_path", "child_id", "timepoint_norm", "label")
        ) else test_out.to_csv(out_dir / "test_predictions.csv", index=False)

        results.append({"system": system_rel, "fold": fold, **metrics})
        print(f"  fold {fold} of {system_rel}: thr={thr:.2f} F1={metrics['f1']:.3f} "
              f"BA={metrics['balanced_accuracy']:.3f} AUROC={metrics['auroc']:.3f}  n={metrics['n']}")
    return results


def main():
    all_rows = []
    for system_rel, out_root in _OUT_ROOT_FOR.items():
        print(f"\n=== {system_rel} ===")
        rows = evaluate_system(system_rel, out_root)
        all_rows.extend(rows)

    summary_path = _REPO / "evaluation" / "groupstrat3_zero_shot_summary.csv"
    pd.DataFrame(all_rows).to_csv(summary_path, index=False)
    print(f"\nWrote summary: {summary_path} ({len(all_rows)} rows)")


if __name__ == "__main__":
    main()
