"""
Turn-taking simulator for synthetic parent-child conversation scenes.

Implements a discrete alternating speaker sequence (ADULT → TARGET_CHILD →
ADULT → …) with configurable pause and overlap distributions, following
the design decisions in ``specs/008-synthetic-child-scenes/research.md`` §D2.
"""

from __future__ import annotations

from typing import List, Dict, Optional
import numpy as np


# Age-band-specific defaults drawn from CHILDES turn-taking literature (D2).
_AGE_BAND_CHILD_DEFAULTS: Dict[str, Dict[str, float]] = {
    "14_18_months": {"dur_mean": 0.6, "dur_std": 0.3},
    "34_38_months": {"dur_mean": 1.8, "dur_std": 0.8},
}
_DEFAULT_CHILD_DUR = {"dur_mean": 1.2, "dur_std": 0.5}


class TurnTakingSimulator:
    """Simulate alternating adult-child turn sequences.

    The speaker sequence always starts with ``ADULT`` and alternates with
    ``TARGET_CHILD``.  Each turn's duration is drawn from a truncated normal
    distribution (minimum 0.1 s).  Pauses between turns are drawn from a
    normal distribution; with probability ``overlap_prob`` the pause becomes
    negative (i.e. overlap), drawn from the absolute value of a normal
    centred on ``overlap_dur_mean``.

    Parameters
    ----------
    age_band : str
        One of ``14_18_months``, ``34_38_months``, or any other string
        (fallback defaults apply).
    overlap_prob : float
        Probability that any given inter-turn pause is an overlap event
        (i.e., the next speaker starts before the current one finishes).
    n_turns_min : int
        Minimum number of turns (inclusive).
    n_turns_max : int
        Maximum number of turns (inclusive).
    child_dur_mean : float, optional
        Mean child-turn duration in seconds.  If None, age-band defaults are
        used.
    child_dur_std : float, optional
        Std of child-turn duration in seconds.  If None, age-band defaults
        are used.
    adult_dur_mean : float
        Mean adult-turn duration in seconds.
    adult_dur_std : float
        Std of adult-turn duration in seconds.
    pause_mean : float
        Mean pause duration between turns in seconds.
    pause_std : float
        Std of pause duration in seconds.
    overlap_dur_mean : float
        Mean overlap duration (seconds) when an overlap event fires.
    overlap_dur_std : float
        Std of overlap duration in seconds.
    """

    def __init__(
        self,
        age_band: str,
        overlap_prob: float = 0.15,
        n_turns_min: int = 2,
        n_turns_max: int = 20,
        child_dur_mean: Optional[float] = None,
        child_dur_std: Optional[float] = None,
        adult_dur_mean: float = 3.5,
        adult_dur_std: float = 1.5,
        pause_mean: float = 0.8,
        pause_std: float = 0.3,
        overlap_dur_mean: float = 0.4,
        overlap_dur_std: float = 0.2,
    ) -> None:
        self.age_band = age_band
        self.overlap_prob = float(overlap_prob)
        self.n_turns_min = int(n_turns_min)
        self.n_turns_max = int(n_turns_max)

        # Resolve child duration defaults from age band if not provided
        if child_dur_mean is None or child_dur_std is None:
            band_defaults = _AGE_BAND_CHILD_DEFAULTS.get(age_band, _DEFAULT_CHILD_DUR)
            if child_dur_mean is None:
                child_dur_mean = band_defaults["dur_mean"]
            if child_dur_std is None:
                child_dur_std = band_defaults["dur_std"]

        self.child_dur_mean = float(child_dur_mean)
        self.child_dur_std = float(child_dur_std)
        self.adult_dur_mean = float(adult_dur_mean)
        self.adult_dur_std = float(adult_dur_std)
        self.pause_mean = float(pause_mean)
        self.pause_std = float(pause_std)
        self.overlap_dur_mean = float(overlap_dur_mean)
        self.overlap_dur_std = float(overlap_dur_std)

    def sample_turns(self, rng: np.random.Generator) -> List[Dict[str, float]]:
        """Sample an ordered list of turns for one scene.

        Parameters
        ----------
        rng : np.random.Generator
            NumPy random generator.

        Returns
        -------
        list of dict
            Each dict has the keys:
              - ``speaker_role`` (str): ``"ADULT"`` or ``"TARGET_CHILD"``
              - ``duration_sec`` (float): turn duration in seconds (≥ 0.1)
              - ``pause_before_sec`` (float): pause before this turn starts;
                negative values indicate overlap with the previous turn.
            The first turn always has ``pause_before_sec = 0.0``.
        """
        n_turns = int(rng.integers(self.n_turns_min, self.n_turns_max + 1))

        turns: List[Dict[str, float]] = []
        # Alternate starting with ADULT
        for i in range(n_turns):
            role = "ADULT" if i % 2 == 0 else "TARGET_CHILD"

            # Sample turn duration
            if role == "ADULT":
                dur = max(0.1, float(rng.normal(self.adult_dur_mean, self.adult_dur_std)))
            else:
                dur = max(0.1, float(rng.normal(self.child_dur_mean, self.child_dur_std)))

            # Sample pause before this turn
            if i == 0:
                pause = 0.0
            else:
                if rng.random() < self.overlap_prob:
                    # Overlap: negative pause; use absolute value of normal sample
                    overlap_dur = abs(
                        float(rng.normal(self.overlap_dur_mean, self.overlap_dur_std))
                    )
                    pause = -max(0.0, overlap_dur)
                else:
                    pause = max(
                        -self.overlap_dur_mean,  # floor to avoid extreme negative
                        float(rng.normal(self.pause_mean, self.pause_std)),
                    )

            turns.append(
                {
                    "speaker_role": role,
                    "duration_sec": dur,
                    "pause_before_sec": pause,
                }
            )

        return turns
