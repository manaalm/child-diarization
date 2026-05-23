"""Calibrator family for spec-021 US7 (FR-060).

Per R7.1: fit (a) global Platt, (b) global isotonic, (c) per-cohort Platt,
(d) per-child Platt where >= 5 positive clips. Pick lowest Brier on val per
system; apply to test.

Brier and Expected Calibration Error (ECE) are reported pre- and post-cal
in evaluation/calibration/per_system_pre_post.csv.
"""
from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


PER_CHILD_MIN_POSITIVES = 5


def brier(probs: np.ndarray, labels: np.ndarray) -> float:
    return float(np.mean((probs - labels) ** 2))


def ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    """Expected calibration error with equal-width bins."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ids = np.digitize(probs, bins[1:-1])
    total = len(probs)
    err = 0.0
    for b in range(n_bins):
        mask = ids == b
        if not mask.any():
            continue
        w = mask.sum() / total
        err += w * abs(probs[mask].mean() - labels[mask].mean())
    return float(err)


# ---------- Calibrator implementations ----------

@dataclass
class GlobalPlatt:
    a: float = 1.0
    b: float = 0.0

    def fit(self, scores, labels):
        # Logit Platt: scores assumed already in [0, 1]; fit on logits.
        eps = 1e-6
        s = np.clip(scores, eps, 1 - eps)
        logits = np.log(s / (1 - s)).reshape(-1, 1)
        lr = LogisticRegression(C=1e6, solver="lbfgs", max_iter=200)
        lr.fit(logits, labels)
        self.a = float(lr.coef_[0, 0])
        self.b = float(lr.intercept_[0])
        return self

    def transform(self, scores):
        eps = 1e-6
        s = np.clip(scores, eps, 1 - eps)
        logits = np.log(s / (1 - s))
        return 1.0 / (1.0 + np.exp(-(self.a * logits + self.b)))


@dataclass
class GlobalIsotonic:
    iso: IsotonicRegression | None = None

    def fit(self, scores, labels):
        self.iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        self.iso.fit(np.asarray(scores), np.asarray(labels))
        return self

    def transform(self, scores):
        return self.iso.transform(np.asarray(scores))


@dataclass
class GlobalTemperature:
    T: float = 1.0

    def fit(self, scores, labels):
        # Temperature on logits via 1-d log-likelihood minimization.
        eps = 1e-6
        s = np.clip(np.asarray(scores), eps, 1 - eps)
        z = np.log(s / (1 - s))
        y = np.asarray(labels).astype(float)
        # Sweep T over a log range; pick min cross-entropy (cheap, stable).
        best, best_ce = 1.0, None
        for T in np.geomspace(0.1, 10.0, 40):
            p = 1.0 / (1.0 + np.exp(-z / T))
            p = np.clip(p, eps, 1 - eps)
            ce = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
            if best_ce is None or ce < best_ce:
                best, best_ce = T, ce
        self.T = float(best)
        return self

    def transform(self, scores):
        eps = 1e-6
        s = np.clip(np.asarray(scores), eps, 1 - eps)
        z = np.log(s / (1 - s))
        return 1.0 / (1.0 + np.exp(-z / self.T))


@dataclass
class PerCohortPlatt:
    """Separate Platt per timepoint cohort; falls back to global for unseen cohorts."""
    cohort_models: dict[str, GlobalPlatt] | None = None
    fallback: GlobalPlatt | None = None

    def fit(self, scores, labels, cohorts):
        cohorts = np.asarray(cohorts)
        labels = np.asarray(labels)
        models = {}
        for c in np.unique(cohorts):
            mask = cohorts == c
            cohort_labels = labels[mask]
            if mask.sum() < 5 or len(np.unique(cohort_labels)) < 2:
                continue
            models[str(c)] = GlobalPlatt().fit(scores[mask], cohort_labels)
        self.cohort_models = models
        self.fallback = GlobalPlatt().fit(scores, labels)
        return self

    def transform(self, scores, cohorts):
        cohorts = np.asarray(cohorts)
        out = np.empty_like(scores, dtype=float)
        for i, c in enumerate(cohorts):
            m = self.cohort_models.get(str(c), self.fallback)
            out[i] = m.transform(np.array([scores[i]]))[0]
        return out


@dataclass
class PerChildPlatt:
    """Per-child Platt where >=5 positive clips on val; per-cohort fallback."""
    child_models: dict[str, GlobalPlatt] | None = None
    cohort_fallback: PerCohortPlatt | None = None

    def fit(self, scores, labels, child_ids, cohorts):
        child_ids = np.asarray(child_ids)
        labels = np.asarray(labels)
        models = {}
        for c in np.unique(child_ids):
            mask = child_ids == c
            child_labels = labels[mask]
            n_pos = int(child_labels.sum())
            n_neg = int(len(child_labels) - n_pos)
            # Need both classes for LR fit, plus >= PER_CHILD_MIN_POSITIVES.
            if n_pos < PER_CHILD_MIN_POSITIVES or n_neg < 2:
                continue
            models[str(c)] = GlobalPlatt().fit(scores[mask], child_labels)
        self.child_models = models
        self.cohort_fallback = PerCohortPlatt().fit(scores, labels, cohorts)
        return self

    def transform(self, scores, child_ids, cohorts):
        child_ids = np.asarray(child_ids)
        cohorts = np.asarray(cohorts)
        out = np.empty_like(scores, dtype=float)
        for i, c in enumerate(child_ids):
            m = self.child_models.get(str(c))
            if m is not None:
                out[i] = m.transform(np.array([scores[i]]))[0]
            else:
                out[i] = self.cohort_fallback.transform(np.array([scores[i]]), np.array([cohorts[i]]))[0]
        return out


# ---------- Per-system calibration runner ----------

CANDIDATE_PRED_COLS = ["prob", "score"]
CANDIDATE_TP_COLS = ["timepoint_norm", "timepoint"]


def detect_cols(df: pd.DataFrame) -> tuple[str, str]:
    score_col = next((c for c in CANDIDATE_PRED_COLS if c in df.columns), None)
    tp_col = next((c for c in CANDIDATE_TP_COLS if c in df.columns), None)
    if score_col is None or tp_col is None:
        raise ValueError(f"could not detect score/timepoint cols in {df.columns.tolist()}")
    return score_col, tp_col


def calibrate_system(val_csv: Path, test_csv: Path, system_name: str) -> dict:
    val = pd.read_csv(val_csv)
    test = pd.read_csv(test_csv)
    sc, tc = detect_cols(val)

    val_s, val_y, val_c, val_id = (
        val[sc].to_numpy(), val["label"].to_numpy(), val[tc].to_numpy(), val["child_id"].to_numpy()
    )
    test_s, test_y, test_c, test_id = (
        test[sc].to_numpy(), test["label"].to_numpy(), test[tc].to_numpy(), test["child_id"].to_numpy()
    )

    candidates = {
        "global_platt":       lambda: GlobalPlatt().fit(val_s, val_y).transform(test_s),
        "global_isotonic":    lambda: GlobalIsotonic().fit(val_s, val_y).transform(test_s),
        "global_temperature": lambda: GlobalTemperature().fit(val_s, val_y).transform(test_s),
        "per_cohort_platt":   lambda: PerCohortPlatt().fit(val_s, val_y, val_c).transform(test_s, test_c),
        "per_child_platt":    lambda: PerChildPlatt().fit(val_s, val_y, val_id, val_c).transform(test_s, test_id, test_c),
    }

    val_calibrators = {
        "global_platt":       GlobalPlatt().fit(val_s, val_y),
        "global_isotonic":    GlobalIsotonic().fit(val_s, val_y),
        "global_temperature": GlobalTemperature().fit(val_s, val_y),
    }
    val_briers = {n: brier(c.transform(val_s), val_y) for n, c in val_calibrators.items()}
    val_briers["per_cohort_platt"] = brier(PerCohortPlatt().fit(val_s, val_y, val_c).transform(val_s, val_c), val_y)
    val_briers["per_child_platt"] = brier(PerChildPlatt().fit(val_s, val_y, val_id, val_c).transform(val_s, val_id, val_c), val_y)

    pre_brier = brier(test_s, test_y)
    pre_ece = ece(test_s, test_y)

    best_name = min(val_briers, key=val_briers.get)
    test_post = candidates[best_name]()
    post_brier = brier(test_post, test_y)
    post_ece = ece(test_post, test_y)

    return {
        "system": system_name,
        "val_csv": str(val_csv),
        "test_csv": str(test_csv),
        "score_col": sc,
        "n_val": int(len(val)),
        "n_test": int(len(test)),
        "selected_calibrator": best_name,
        "val_briers": val_briers,
        "pre_brier": pre_brier,
        "post_brier": post_brier,
        "pre_ece": pre_ece,
        "post_ece": post_ece,
        "test_calibrated_probs": test_post.tolist(),
        "test_labels": test_y.tolist(),
        "test_child_ids": test_id.tolist(),
        "test_cohorts": test_c.tolist(),
    }


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--val", required=True, type=Path)
    ap.add_argument("--test", required=True, type=Path)
    ap.add_argument("--name", required=True)
    args = ap.parse_args()
    res = calibrate_system(args.val, args.test, args.name)
    print(json.dumps({k: v for k, v in res.items() if not isinstance(v, list)}, indent=2))
