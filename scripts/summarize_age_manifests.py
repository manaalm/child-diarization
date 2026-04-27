"""
Summarize age-group counts across dataset manifests.

Prints per-dataset, per-age-group statistics and exits non-zero if any
age group has fewer than 500 child segments (across all datasets combined).

Usage:
    python scripts/summarize_age_manifests.py
    python scripts/summarize_age_manifests.py --min-segments 500
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
DATASETS = ["playlogue", "providence", "seedlings"]
AGE_GROUPS = ["12_16m", "34_38m"]


def load_manifest(dataset: str) -> pd.DataFrame | None:
    path = REPO_ROOT / dataset / "manifest.csv"
    if not path.exists():
        print(f"  [MISSING] {path}")
        return None
    df = pd.read_csv(path)
    print(f"  [OK] {path} ({len(df)} records)")
    return df


def main():
    parser = argparse.ArgumentParser(description="Summarize age manifests")
    parser.add_argument("--min-segments", type=int, default=500,
                        help="Minimum labeled child segments per age group (default: 500)")
    args = parser.parse_args()

    all_dfs = []
    print("Loading manifests...")
    for ds in DATASETS:
        df = load_manifest(ds)
        if df is not None:
            all_dfs.append(df)

    if not all_dfs:
        sys.exit("ERROR: No manifests found. Run prepare_age_manifests.py first.")

    # Per-dataset, per-age-group breakdown
    print("\n" + "=" * 70)
    print("Per-dataset breakdown")
    print("=" * 70)
    for ds, df in zip(DATASETS, all_dfs):
        print(f"\n{ds}:")
        grp = df.groupby("age_group").agg(
            n_recordings=("recording_id", "count"),
            has_rttm=("has_rttm", "sum"),
        ).reset_index()
        print(grp.to_string(index=False))

    # Combined counts across all datasets
    combined = pd.concat(all_dfs, ignore_index=True)
    print("\n" + "=" * 70)
    print("Combined totals")
    print("=" * 70)
    combined_grp = combined.groupby("age_group").agg(
        n_recordings=("recording_id", "count"),
        with_rttm=("has_rttm", "sum"),
    ).reset_index()
    print(combined_grp.to_string(index=False))

    # RTTM-bearing recordings (available for synthesis extraction)
    rttm_df = combined[combined["has_rttm"]]
    print(f"\nRecordings with RTTM (usable for synthesis extraction):")
    rttm_grp = rttm_df.groupby("age_group").agg(
        n_recordings=("recording_id", "count"),
    ).reset_index()
    print(rttm_grp.to_string(index=False))

    # Validation: check minimum per age group
    print("\n" + "=" * 70)
    print(f"Validation (min_segments={args.min_segments})")
    print("=" * 70)
    failed = False
    for age in AGE_GROUPS:
        n = int(combined_grp[combined_grp["age_group"] == age]["n_recordings"].sum())
        status = "PASS" if n >= args.min_segments else "FAIL"
        print(f"  {age}: {n} recordings [{status}]")
        if n < args.min_segments:
            failed = True

    if failed:
        sys.exit(1)

    print("\nAll age groups meet minimum threshold.")


if __name__ == "__main__":
    main()
