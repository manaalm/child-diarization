"""
Segment manifest loading, validation, and sampling utilities.

Enforces the integrity constraints defined in
``specs/008-synthetic-child-scenes/contracts/segment-manifest.md``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import List, Optional

# Minimum set of columns required by the segment manifest contract.
REQUIRED_COLUMNS: List[str] = [
    "segment_id",
    "source_dataset",
    "source_recording_id",
    "speaker_id",
    "speaker_role",
    "age_band",
    "start_time_sec",
    "end_time_sec",
    "duration_sec",
    "audio_path",
    "sample_rate",
    "split",
    "usable_for_training",
]


def _coerce_bool(series: pd.Series) -> pd.Series:
    """Coerce a column that may contain booleans, strings, or ints to bool."""
    if series.dtype == bool:
        return series
    # Handle string representations
    return series.map(
        lambda v: (
            v
            if isinstance(v, bool)
            else str(v).strip().lower() not in ("false", "0", "no", "")
        )
    ).astype(bool)


def load_manifest(csv_path: str) -> pd.DataFrame:
    """Load and validate a segment manifest CSV.

    Parameters
    ----------
    csv_path : str
        Path to the manifest CSV file.

    Returns
    -------
    pd.DataFrame
        Validated manifest with ``usable_for_training`` coerced to ``bool``.

    Raises
    ------
    ValueError
        If required columns are missing, if a speaker_id appears in both
        ``train`` and ``test`` splits, or if any ``split='test'`` row has
        ``usable_for_training=True``.
    """
    df = pd.read_csv(csv_path, low_memory=False)

    # --- Check required columns ---
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Segment manifest is missing required columns: {missing}"
        )

    # --- Coerce usable_for_training to bool ---
    df["usable_for_training"] = _coerce_bool(df["usable_for_training"])

    # --- Split integrity: no speaker_id in both train and test ---
    train_speakers = set(df.loc[df["split"] == "train", "speaker_id"].dropna())
    test_speakers = set(df.loc[df["split"] == "test", "speaker_id"].dropna())
    overlap = train_speakers & test_speakers
    if overlap:
        raise ValueError(
            f"Split integrity violation: the following speaker_ids appear in "
            f"both 'train' and 'test' splits: {sorted(overlap)}"
        )

    # --- All test rows must have usable_for_training=False ---
    test_usable = df[(df["split"] == "test") & (df["usable_for_training"] == True)]
    if not test_usable.empty:
        bad_ids = test_usable["speaker_id"].tolist()
        raise ValueError(
            f"Integrity violation: {len(test_usable)} row(s) with split='test' "
            f"have usable_for_training=True.  Offending speaker_ids: {bad_ids}"
        )

    return df


def filter_usable(
    df: pd.DataFrame,
    age_band: Optional[str] = None,
    source_datasets: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Filter manifest to usable training segments.

    Parameters
    ----------
    df : pd.DataFrame
        Segment manifest (as returned by :func:`load_manifest`).
    age_band : str, optional
        If provided, restrict to segments with this age_band value.
    source_datasets : list of str, optional
        If provided, restrict to segments whose ``source_dataset`` is in this
        list.

    Returns
    -------
    pd.DataFrame
        Filtered manifest (subset of rows from ``df``).

    Raises
    ------
    ValueError
        If the filtered result is empty.
    """
    mask = df["usable_for_training"] == True
    if age_band is not None:
        mask &= df["age_band"] == age_band
    if source_datasets is not None:
        mask &= df["source_dataset"].isin(source_datasets)

    result = df[mask].copy()
    if result.empty:
        parts = ["usable_for_training=True"]
        if age_band is not None:
            parts.append(f"age_band='{age_band}'")
        if source_datasets is not None:
            parts.append(f"source_dataset in {source_datasets}")
        raise ValueError(
            f"filter_usable: no segments match the criteria: "
            + ", ".join(parts)
        )

    return result


def sample_segment(df: pd.DataFrame, rng: np.random.Generator) -> dict:
    """Sample one segment row uniformly at random.

    Parameters
    ----------
    df : pd.DataFrame
        Segment manifest (filtered to usable rows, typically).
    rng : np.random.Generator
        NumPy random generator for reproducible sampling.

    Returns
    -------
    dict
        A single row from ``df`` as a plain Python dict.
    """
    idx = rng.integers(0, len(df))
    return df.iloc[idx].to_dict()
