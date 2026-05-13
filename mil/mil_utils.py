"""Shared metric helpers for the MIL workflow."""

import json
import os
from typing import Dict, List

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


def compute_metrics(y_true: List[int], y_score: List[float], threshold: float = 0.5) -> Dict[str, float]:
    """Compute F1, precision, recall, AUROC, AUPRC, plus imbalance-aware extensions
    (f1_macro, f1_weighted, balanced_accuracy) at a fixed threshold.

    Extended 2026-05-12 (spec 022 US2 / FR-007). Existing keys preserved verbatim;
    f1 remains binary-positive-class F1 to keep legacy headline tables stable.
    """
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
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "auroc": auroc,
        "auprc": auprc,
    }


def tune_threshold(val_labels: List[int], val_scores: List[float],
                   objective: str = "balanced_accuracy") -> float:
    """Sweep thresholds 0.05–0.95 and return the one maximising the named
    objective on val.

    spec 022 (2026-05-13): default flipped from `"f1"` to `"balanced_accuracy"`
    per advisor directive — the F1-max threshold systematically picks
    recall≈0.99 operating points on this 76%-positive split, masking imbalance
    behavior. Balanced-accuracy max gives a more deployable operating point.
    Pass `objective="f1"` to recover the legacy behavior.

    Tie-break: closest threshold to 0.5 wins (matches the
    `retune_all_by_ba.py` tie-break to keep the two pipelines consistent).
    """
    if objective not in {"balanced_accuracy", "f1"}:
        raise ValueError(f"unknown objective: {objective!r}; expected 'balanced_accuracy' or 'f1'")

    best_thresh, best_score = 0.5, -1.0
    for t in np.arange(0.05, 0.96, 0.05):
        t = float(t)
        m = compute_metrics(val_labels, val_scores, threshold=t)
        s = m[objective]
        if s > best_score or (s == best_score and abs(t - 0.5) < abs(best_thresh - 0.5)):
            best_score, best_thresh = s, t
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
