"""Vickers & Elkin 2006 net-benefit decision-curve analysis for spec-021 US7 FR-062.

Net benefit at threshold p_t:
    NB(p_t) = TP/N - FP/N * (p_t / (1 - p_t))

where p_t is the threshold probability that determines whether to "treat"
(i.e., predict positive). The miss-cost ratio (cost_FN / cost_FP) implies
p_t via p_t = 1 / (1 + cost_FN/cost_FP). Common cost ratios { 1:1, 1:5, 1:10, 1:25 }
correspond to p_t in {0.5, 0.1667, 0.0909, 0.0385}.

Per R7.3: report net benefit at miss-cost ratios {1:1, 1:5, 1:10, 1:25}
across top-band systems and identify the regime (if any) where the metadata
stacker net-benefit dominates the audio-only Whisper-MIL.
"""
from __future__ import annotations
from dataclasses import dataclass

import numpy as np


def threshold_from_ratio(cost_ratio: float) -> float:
    """cost_ratio = cost_FN / cost_FP. Returns p_t for net-benefit formula."""
    return 1.0 / (1.0 + cost_ratio)


def net_benefit(probs: np.ndarray, labels: np.ndarray, p_t: float) -> float:
    n = len(labels)
    pred = (probs >= p_t).astype(int)
    tp = int(((pred == 1) & (labels == 1)).sum())
    fp = int(((pred == 1) & (labels == 0)).sum())
    return float(tp / n - fp / n * (p_t / (1.0 - p_t)))


def treat_all_net_benefit(labels: np.ndarray, p_t: float) -> float:
    """Reference 'treat all' strategy."""
    n = len(labels)
    tp = int(labels.sum())
    fp = n - tp
    return float(tp / n - fp / n * (p_t / (1.0 - p_t)))


def treat_none_net_benefit() -> float:
    return 0.0


# Standard miss-cost ratios from spec FR-062.
RATIOS = {
    "1:1":  1.0,
    "1:5":  5.0,
    "1:10": 10.0,
    "1:25": 25.0,
}
