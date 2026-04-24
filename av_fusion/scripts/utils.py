"""Shared utilities for the AV fusion pipeline.

Provides metric computation, threshold tuning, split integrity checks, and
file I/O helpers used across all av_fusion scripts.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

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


def compute_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float,
) -> Dict[str, float]:
    """Compute AUROC, AUPRC, F1, precision, recall, balanced accuracy.

    Args:
        y_true: Binary ground-truth labels (0/1).
        y_score: Predicted probabilities.
        threshold: Binary decision threshold.

    Returns:
        Dict with keys: auroc, auprc, f1, precision, recall, balanced_accuracy, threshold.
        All NaN if insufficient class diversity or sample count.
    """
    nan_result = {k: float("nan") for k in ("auroc", "auprc", "f1", "precision", "recall", "balanced_accuracy")}
    nan_result["threshold"] = threshold

    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)

    if len(y_true) < 5 or y_true.sum() == 0 or y_true.sum() == len(y_true):
        return nan_result

    y_pred = (y_score >= threshold).astype(int)

    try:
        return {
            "auroc": float(roc_auc_score(y_true, y_score)),
            "auprc": float(average_precision_score(y_true, y_score)),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
            "threshold": threshold,
        }
    except Exception:
        return nan_result


def tune_threshold_f1(
    y_true: np.ndarray,
    y_score: np.ndarray,
    n_thresholds: int = 100,
) -> Tuple[float, float]:
    """Find threshold that maximises F1 on the provided data.

    Returns:
        (best_threshold, best_f1)
    """
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    thresholds = np.linspace(0.01, 0.99, n_thresholds)
    best_thresh, best_f1 = 0.5, 0.0
    for t in thresholds:
        f = f1_score(y_true, (y_score >= t).astype(int), zero_division=0)
        if f > best_f1:
            best_f1 = f
            best_thresh = float(t)
    return best_thresh, best_f1


def tune_threshold_balanced_acc(
    y_true: np.ndarray,
    y_score: np.ndarray,
    n_thresholds: int = 100,
) -> Tuple[float, float]:
    """Find threshold that maximises balanced accuracy on the provided data.

    Returns:
        (best_threshold, best_balanced_acc)
    """
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    thresholds = np.linspace(0.01, 0.99, n_thresholds)
    best_thresh, best_bacc = 0.5, 0.0
    for t in thresholds:
        ba = balanced_accuracy_score(y_true, (y_score >= t).astype(int))
        if ba > best_bacc:
            best_bacc = ba
            best_thresh = float(t)
    return best_thresh, best_bacc


def tune_late_fusion_alpha(
    y_true: np.ndarray,
    audio_score: np.ndarray,
    visual_score: np.ndarray,
    n_alphas: int = 21,
) -> Tuple[float, float]:
    """Find late-fusion mixing weight alpha that maximises AUROC on val.

    Combined score = alpha * audio_score + (1-alpha) * visual_score.

    Returns:
        (best_alpha, best_auroc)
    """
    y_true = np.asarray(y_true, dtype=int)
    if y_true.sum() == 0 or y_true.sum() == len(y_true):
        return 0.5, float("nan")

    alphas = np.linspace(0.0, 1.0, n_alphas)
    best_alpha, best_auroc = 0.5, 0.0
    for a in alphas:
        combined = a * np.asarray(audio_score, dtype=float) + (1.0 - a) * np.asarray(visual_score, dtype=float)
        try:
            au = float(roc_auc_score(y_true, combined))
        except Exception:
            continue
        if au > best_auroc:
            best_auroc = au
            best_alpha = float(a)
    return best_alpha, best_auroc


def assert_split_integrity(df: pd.DataFrame) -> None:
    """Raise ValueError if any clip_id appears in more than one split.

    Uses clip_id (not child_id) because the seen-child split intentionally
    has the same 109 children in train/val/test with different clips per split.
    What must never happen is the same clip appearing in multiple splits.

    Args:
        df: DataFrame with 'clip_id' and 'split' columns.
    """
    if "clip_id" not in df.columns or "split" not in df.columns:
        return
    clip_splits = df.groupby("clip_id")["split"].nunique()
    violators = clip_splits[clip_splits > 1].index.tolist()
    if violators:
        raise ValueError(
            f"Split integrity violation: {len(violators)} clip(s) appear in multiple splits: "
            f"{violators[:5]}{'...' if len(violators) > 5 else ''}"
        )


def save_json(data: Any, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_feature_csv(path: str, nan_fill: Optional[float] = None) -> pd.DataFrame:
    """Load a feature CSV; optionally fill numeric NaN values."""
    df = pd.read_csv(path, low_memory=False)
    if nan_fill is not None:
        num_cols = df.select_dtypes(include=[np.number]).columns
        df[num_cols] = df[num_cols].fillna(nan_fill)
    return df


def get_repo_root() -> str:
    """Return absolute path to the repository root (parent of av_fusion/)."""
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
