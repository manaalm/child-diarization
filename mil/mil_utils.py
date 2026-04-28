"""Shared metric helpers for the MIL workflow."""

import json
import os
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_metrics(y_true: List[int], y_score: List[float], threshold: float = 0.5) -> Dict[str, float]:
    """Compute F1, precision, recall, AUROC, AUPRC at a fixed threshold."""
    y_true = np.array(y_true, dtype=int)
    y_score = np.array(y_score, dtype=float)
    y_pred = (y_score >= threshold).astype(int)

    n_pos = y_true.sum()
    if n_pos == 0 or n_pos == len(y_true):
        auroc = float("nan")
        auprc = float("nan")
    else:
        auroc = float(roc_auc_score(y_true, y_score))
        auprc = float(average_precision_score(y_true, y_score))

    return {
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "auroc": auroc,
        "auprc": auprc,
    }


def tune_threshold(val_labels: List[int], val_scores: List[float]) -> float:
    """Sweep thresholds 0.05–0.95 and return the one maximising val F1."""
    best_thresh, best_f1 = 0.5, -1.0
    for t in np.arange(0.05, 0.96, 0.05):
        m = compute_metrics(val_labels, val_scores, threshold=float(t))
        if m["f1"] > best_f1:
            best_f1, best_thresh = m["f1"], float(t)
    return round(best_thresh, 4)


def per_timepoint_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-timepoint metrics from a predictions DataFrame.

    Expected columns: timepoint_norm, label, score, prediction.
    Returns DataFrame with columns: timepoint, f1, precision, recall, auroc, auprc, n.
    """
    rows = []
    for tp, grp in df.groupby("timepoint_norm"):
        m = compute_metrics(grp["label"].tolist(), grp["score"].tolist(),
                            threshold=grp["prediction"].mean())  # implicit threshold
        # Recompute with explicit threshold from prediction column
        y_pred = grp["prediction"].tolist()
        y_true = grp["label"].tolist()
        m = {
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "auroc": float(roc_auc_score(y_true, grp["score"].tolist()))
                     if len(set(y_true)) > 1 else float("nan"),
            "auprc": float(average_precision_score(y_true, grp["score"].tolist()))
                     if len(set(y_true)) > 1 else float("nan"),
        }
        rows.append({"timepoint": tp, **m, "n": len(grp)})
    return pd.DataFrame(rows)


def save_json(d: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(d, f, indent=2)


def save_csv(df: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
