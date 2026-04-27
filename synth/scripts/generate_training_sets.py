#!/usr/bin/env python3
"""
Generate augmented training manifests at multiple synthetic-to-real ratios.

For each ratio r, produces:
    synth_results/manifests/train_{r}x_manifest.csv

where the manifest contains all real training rows plus
round(r * len(real_rows)) synthetic rows sampled from the synthetic manifest,
stratified by age_band to match the real distribution.

The set of real rows is identical across all ratio files.

Usage:
    python synth/scripts/generate_training_sets.py \\
      --real-train-csv        whisper-modeling/seen_child_splits/train.csv \\
      --synthetic-manifest    synth_results/manifests/synthetic_manifest.csv \\
      --ratios                0 0.5 1 2 5 10 \\
      --output-dir            synth_results/manifests/ \\
      --seed                  42
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


# Columns required in the output per contracts/training-manifest.md
_OUTPUT_COLUMNS = [
    "audio_path",
    "rttm_path",
    "label",
    "child_id",
    "timepoint_norm",
    "split",
    "is_synthetic",
    "source_config",
    "age_band",
]


def _build_real_rows(real_train_csv: str) -> pd.DataFrame:
    """Load the real training CSV and annotate it for the training manifest schema."""
    df = pd.read_csv(real_train_csv, low_memory=False)
    required = {"audio_path", "child_id", "timepoint_norm", "label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Real train CSV is missing required columns: {sorted(missing)}"
        )

    # The seen-child splits use timepoint_norm (14_month, 36_month);
    # map to age_band (14_18_months, 34_38_months) for schema consistency.
    timepoint_to_age_band = {
        "14_month": "14_18_months",
        "36_month": "34_38_months",
    }

    rows = df.copy()
    rows["is_synthetic"] = False
    rows["source_config"] = ""
    rows["split"] = "train"
    # Infer age_band from timepoint_norm if not already present
    if "age_band" not in rows.columns:
        rows["age_band"] = rows["timepoint_norm"].map(
            lambda t: timepoint_to_age_band.get(str(t), str(t))
        )
    # rttm_path may not exist in the real clips (they don't have per-clip RTTMs)
    if "rttm_path" not in rows.columns:
        rows["rttm_path"] = None

    return rows


def _build_synthetic_rows(synthetic_manifest_csv: str) -> pd.DataFrame:
    """Load the synthetic clip-labels manifest and prepare for training manifest schema."""
    df = pd.read_csv(synthetic_manifest_csv, low_memory=False)
    required = {"audio_path", "rttm_path", "target_child_vocalized",
                "age_band", "synthetic_scene_id", "generation_config_name"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Synthetic manifest is missing required columns: {sorted(missing)}"
        )

    rows = df.copy()
    rows["is_synthetic"] = True
    rows["label"] = rows["target_child_vocalized"].astype(int)
    rows["child_id"] = "synthetic_" + rows["synthetic_scene_id"].astype(str)
    rows["source_config"] = rows["generation_config_name"].astype(str)
    rows["split"] = "train"

    # Map age_band to timepoint_norm for compatibility
    age_band_to_timepoint = {
        "14_18_months": "14_month",
        "34_38_months": "36_month",
    }
    rows["timepoint_norm"] = rows["age_band"].map(
        lambda b: age_band_to_timepoint.get(str(b), str(b))
    )

    return rows


def _stratified_sample(
    df: pd.DataFrame,
    n: int,
    stratify_col: str,
    rng: np.random.Generator,
    target_distribution: dict,
) -> pd.DataFrame:
    """Sample n rows from df, stratified by stratify_col.

    Parameters
    ----------
    df : pd.DataFrame
        Pool to sample from.
    n : int
        Total rows to sample.
    stratify_col : str
        Column to stratify by.
    rng : np.random.Generator
        Random generator for reproducibility.
    target_distribution : dict
        Desired fraction per stratum (e.g. {"14_18_months": 0.6}).
    """
    if n == 0:
        return df.iloc[:0]

    sampled_parts = []
    remaining = n

    # Sort strata for deterministic ordering
    strata = sorted(target_distribution.keys())

    for i, stratum in enumerate(strata):
        frac = target_distribution[stratum]
        if i == len(strata) - 1:
            n_stratum = remaining
        else:
            n_stratum = round(n * frac)
            remaining -= n_stratum

        pool = df[df[stratify_col] == stratum]
        if pool.empty:
            continue
        if n_stratum <= 0:
            continue

        # Sample with replacement if pool is smaller than needed
        replace = n_stratum > len(pool)
        idx = rng.choice(len(pool), size=n_stratum, replace=replace)
        sampled_parts.append(pool.iloc[idx])

    if not sampled_parts:
        return df.iloc[:0]
    return pd.concat(sampled_parts, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate augmented training manifests at multiple synth ratios."
    )
    parser.add_argument(
        "--real-train-csv",
        required=True,
        help="Path to the real training CSV (whisper-modeling/seen_child_splits/train.csv).",
    )
    parser.add_argument(
        "--synthetic-manifest",
        required=True,
        help="Path to the synthetic clip-labels manifest CSV.",
    )
    parser.add_argument(
        "--ratios",
        type=float,
        nargs="+",
        default=[0, 0.5, 1, 2, 5, 10],
        help="List of synthetic-to-real ratios (default: 0 0.5 1 2 5 10).",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory for train_{ratio}x_manifest.csv files.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for stratified sampling.",
    )
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading real training CSV: {args.real_train_csv}")
    real_rows = _build_real_rows(args.real_train_csv)
    n_real = len(real_rows)
    print(f"  {n_real} real training rows.")

    print(f"Loading synthetic manifest: {args.synthetic_manifest}")
    synth_rows = _build_synthetic_rows(args.synthetic_manifest)
    n_synth_pool = len(synth_rows)
    print(f"  {n_synth_pool} synthetic rows in pool.")

    # Compute real age_band distribution for stratification
    age_band_counts = real_rows["age_band"].value_counts()
    total_real = len(real_rows)
    target_dist = {
        band: count / total_real
        for band, count in age_band_counts.items()
    }
    print(f"  Real age_band distribution: {dict(age_band_counts)}")

    print(f"\nGenerating manifests for ratios: {args.ratios}")
    for ratio in args.ratios:
        n_synth = round(ratio * n_real)
        if n_synth > n_synth_pool:
            print(
                f"  [WARN] ratio={ratio}x requests {n_synth} synthetic rows "
                f"but pool has only {n_synth_pool}; will sample with replacement."
            )

        if n_synth > 0:
            synth_sample = _stratified_sample(
                synth_rows, n_synth, "age_band", rng, target_dist
            )
        else:
            synth_sample = synth_rows.iloc[:0]

        combined = pd.concat([real_rows, synth_sample], ignore_index=True)

        # Ensure all required output columns are present
        for col in _OUTPUT_COLUMNS:
            if col not in combined.columns:
                combined[col] = None

        # Format ratio for filename: 0.5 → "0.5", 1 → "1", 10 → "10"
        ratio_str = str(int(ratio)) if ratio == int(ratio) else str(ratio)
        out_path = out_dir / f"train_{ratio_str}x_manifest.csv"
        combined[_OUTPUT_COLUMNS + [c for c in combined.columns
                                    if c not in _OUTPUT_COLUMNS]].to_csv(
            out_path, index=False
        )

        n_is_synth = int(combined["is_synthetic"].sum())
        n_is_real = len(combined) - n_is_synth
        print(
            f"  ratio={ratio_str}x → {len(combined)} rows "
            f"(real={n_is_real}, synthetic={n_is_synth}) → {out_path.name}"
        )


if __name__ == "__main__":
    main()
