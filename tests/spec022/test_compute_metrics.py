"""Smoke tests for mil/mil_utils.compute_metrics extended dict (spec 022 US2 / FR-007)."""

import os
import sys

import pytest

REPO_ROOT = "/orcd/scratch/orcd/008/manaal/child-adult-diarization"
sys.path.insert(0, os.path.join(REPO_ROOT, "mil"))

from mil_utils import compute_metrics  # noqa: E402


def test_extended_keys_present():
    """compute_metrics must return the extended imbalance-aware keys."""
    y_true = [0, 0, 1, 1, 1, 1]
    y_score = [0.1, 0.4, 0.6, 0.7, 0.8, 0.2]
    m = compute_metrics(y_true, y_score, threshold=0.5)
    expected_keys = {"f1", "f1_macro", "f1_weighted", "balanced_accuracy",
                     "precision", "recall", "auroc", "auprc"}
    assert set(m.keys()) == expected_keys


def test_legacy_f1_unchanged():
    """Existing `f1` key must be positive-class binary F1, unchanged."""
    y_true = [0, 0, 1, 1, 1, 1]
    y_score = [0.0, 0.0, 1.0, 1.0, 1.0, 1.0]
    m = compute_metrics(y_true, y_score, threshold=0.5)
    assert m["f1"] == pytest.approx(1.0)
    assert m["precision"] == pytest.approx(1.0)
    assert m["recall"] == pytest.approx(1.0)


def test_balanced_accuracy_is_chance_for_predict_all():
    """A constant predict-all-positive predictor has balanced_accuracy = 0.5 by construction."""
    y_true = [0, 0, 0, 1, 1, 1, 1, 1, 1]  # 67% positive
    y_score = [0.9] * 9  # always positive at threshold 0.5
    m = compute_metrics(y_true, y_score, threshold=0.5)
    assert m["balanced_accuracy"] == pytest.approx(0.5)
    # The legacy F1 looks "good" because precision = pos_rate and recall = 1.0
    assert m["f1"] > 0.7
    # But class-weighted F1 already deflates this somewhat
    assert m["f1_weighted"] < m["f1"]


def test_imbalance_aware_gap():
    """Show the F1 vs balanced_accuracy gap on a realistic SAILS-like imbalance."""
    # 76% positive (SAILS-like), classifier mostly predicts positive
    y_true = [1] * 76 + [0] * 24
    y_score = [0.9] * 100  # always positive
    m = compute_metrics(y_true, y_score, threshold=0.5)
    assert m["f1"] == pytest.approx(0.864, abs=0.005)        # trivial floor F1
    assert m["balanced_accuracy"] == pytest.approx(0.5)      # but BA reveals the truth


def test_auroc_returns_nan_on_single_class():
    """AUROC and AUPRC must be NaN when only one class present (not crash)."""
    import math
    m = compute_metrics([1, 1, 1, 1], [0.1, 0.2, 0.3, 0.4])
    assert math.isnan(m["auroc"])
    assert math.isnan(m["auprc"])
