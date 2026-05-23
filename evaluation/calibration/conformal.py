"""Split-conformal prediction at alpha = 0.10 for spec-021 US7 FR-061.

Per R7.2: val set is the calibration set, test set is the test set; report
empirical coverage on a held-out 10% rotated fold of test predictions to
verify SC-061 (within +/- 0.02 of nominal 0.90).

For binary classification with calibrated probabilities, the conformity score
follows Romano et al. 2020 ("APS"-style) using the predicted-class probability:
    s_i = 1 - p_yi(x_i)   for the true class y_i
The threshold q is the (1-alpha)*(1+1/n) quantile of the calibration s_i's.
At test time, the prediction set includes class y iff (1 - p_y(x)) <= q.
Coverage = fraction of test points whose true label is in the predicted set.
"""
from __future__ import annotations
import argparse
from dataclasses import dataclass

import numpy as np


def conformity(probs_pos: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """s_i = 1 - p_{y_i}(x_i) for binary task."""
    p_true = np.where(labels == 1, probs_pos, 1.0 - probs_pos)
    return 1.0 - p_true


def quantile_threshold(s_calib: np.ndarray, alpha: float) -> float:
    n = len(s_calib)
    # (1-alpha)*(1+1/n) quantile, clipped to [0, 1].
    k = int(np.ceil((1 - alpha) * (n + 1)))
    k = max(1, min(n, k))
    return float(np.sort(s_calib)[k - 1])


def predict_sets(probs_pos: np.ndarray, q: float) -> np.ndarray:
    """Return a (n, 2) bool array: columns = [class0_in_set, class1_in_set]."""
    n = len(probs_pos)
    s0 = 1.0 - (1.0 - probs_pos)  # s if true label = 0 -> s0 = probs_pos
    s1 = 1.0 - probs_pos          # s if true label = 1 -> s1 = 1 - probs_pos
    # NB: using true-label-conditional formulation; recompute symmetrically.
    in0 = s0 <= q
    in1 = s1 <= q
    return np.stack([in0, in1], axis=1)


def empirical_coverage(probs_pos: np.ndarray, labels: np.ndarray, q: float) -> float:
    sets = predict_sets(probs_pos, q)
    in_set = sets[np.arange(len(labels)), labels.astype(int)]
    return float(in_set.mean())


@dataclass
class ConformalResult:
    alpha: float
    q: float
    coverage_test: float
    coverage_holdout_mean: float
    coverage_holdout_std: float
    set_size_mean: float
    n_calib: int
    n_test: int


def split_conformal(probs_calib: np.ndarray, labels_calib: np.ndarray,
                    probs_test: np.ndarray, labels_test: np.ndarray,
                    alpha: float = 0.10,
                    n_holdout_rotations: int = 10,
                    seed: int = 42) -> ConformalResult:
    s_calib = conformity(probs_calib, labels_calib)
    q = quantile_threshold(s_calib, alpha)
    cov_full = empirical_coverage(probs_test, labels_test, q)

    rng = np.random.default_rng(seed)
    n = len(probs_test)
    fold = max(1, n // n_holdout_rotations)
    covs = []
    idxs = np.arange(n)
    for r in range(n_holdout_rotations):
        rng.shuffle(idxs)
        held = idxs[:fold]
        covs.append(empirical_coverage(probs_test[held], labels_test[held], q))
    covs = np.array(covs)

    sets = predict_sets(probs_test, q)
    set_sizes = sets.sum(axis=1)

    return ConformalResult(
        alpha=alpha,
        q=q,
        coverage_test=cov_full,
        coverage_holdout_mean=float(covs.mean()),
        coverage_holdout_std=float(covs.std()),
        set_size_mean=float(set_sizes.mean()),
        n_calib=int(len(probs_calib)),
        n_test=int(len(probs_test)),
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--probs-calib", required=True)
    ap.add_argument("--labels-calib", required=True)
    ap.add_argument("--probs-test", required=True)
    ap.add_argument("--labels-test", required=True)
    ap.add_argument("--alpha", type=float, default=0.10)
    args = ap.parse_args()

    pc = np.loadtxt(args.probs_calib)
    yc = np.loadtxt(args.labels_calib).astype(int)
    pt = np.loadtxt(args.probs_test)
    yt = np.loadtxt(args.labels_test).astype(int)

    r = split_conformal(pc, yc, pt, yt, alpha=args.alpha)
    print(r)
