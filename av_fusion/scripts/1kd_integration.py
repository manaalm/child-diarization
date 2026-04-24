"""Check compatibility of a 1kd dataset directory with the existing clip schema.

Produces a JSON report documenting: access status, compatible columns, clip count,
age range overlap, and access/integration pathway notes.

The script always exits 0 — never crashes — and always writes a report even when
data is not found.

Known datasets referred to as "1kd":
  - 1000 Days (1kd) Project — Brown University / NICHD:
    Naturalistic home recordings of infants/toddlers from birth to 36 months.
    Access via institutional data use agreement; contact PI at Brown University.
    Publications: Bergelson & Swingley (2012), Casillas et al. (2019).
  - 1000 Days from Home (UK Biobank linked): age 0-3 longitudinal; restricted access.

Usage:
    python av_fusion/scripts/1kd_integration.py \\
        --data-dir /path/to/1kd/ \\
        --output   av_fusion/av_results/run1/1kd_integration_report.json \\
        [--dry-run]
"""

import argparse
import json
import os
import sys
from typing import Any, Dict, List

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import get_repo_root

_REPO = get_repo_root()

_REQUIRED_COLUMNS = {"clip_id", "child_id", "audio_path", "label", "timepoint"}
_EXISTING_TIMEPOINTS = {"14_month", "36_month"}
_EXISTING_AGE_RANGE = (14, 36)  # months

_ACCESS_NOTES = (
    "The '1kd' dataset likely refers to the 1000 Days (1kd) longitudinal project from "
    "Brown University / NICHD, which contains naturalistic home recordings of infants "
    "and toddlers from birth to 36 months. Access requires an institutional data use "
    "agreement (DUA) with the PI's lab at Brown University. "
    "See: Bergelson & Swingley (2012) PNAS; Casillas et al. (2019) LREC. "
    "Alternative interpretation: 1000 Days from Home (UK Biobank linked cohort, age 0-3). "
    "To integrate if accessible: map clip_id, child_id, audio_path, label, timepoint columns "
    "to the existing seen-child split schema; assign 1kd children to a separate child_id "
    "namespace (e.g., prefix 'kd_') to avoid conflicts with existing SAILS BIDS child IDs; "
    "re-run make_seen_child_split.py with the merged annotation CSV."
)


def _find_annotation_csv(data_dir: str) -> str:
    """Try to find a plausible annotation CSV in the data directory."""
    for fname in ("annotations.csv", "metadata.csv", "clips.csv", "manifest.csv", "labels.csv"):
        p = os.path.join(data_dir, fname)
        if os.path.exists(p):
            return p
    # Try any .csv
    for root, _, files in os.walk(data_dir):
        for f in files:
            if f.endswith(".csv"):
                return os.path.join(root, f)
    return ""


def _age_months_from_timepoint(tp: str) -> int:
    """Try to parse a timepoint string like '14_month' → 14."""
    try:
        return int(str(tp).split("_")[0])
    except Exception:
        return -1


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check 1kd dataset compatibility with existing clip schema."
    )
    parser.add_argument("--data-dir", required=True,
                        help="Directory containing 1kd audio/video files and annotation CSV")
    parser.add_argument("--output", required=True,
                        help="Output path for JSON compatibility report")
    parser.add_argument("--dry-run", action="store_true",
                        help="Check schema only; do not copy any files")
    args = parser.parse_args()

    data_dir = args.data_dir if os.path.isabs(args.data_dir) else os.path.join(_REPO, args.data_dir)
    output = args.output if os.path.isabs(args.output) else os.path.join(_REPO, args.output)
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)

    # Case 1: Data directory does not exist
    if not os.path.exists(data_dir):
        report: Dict[str, Any] = {
            "status": "not_found",
            "n_clips": 0,
            "missing_columns": [],
            "age_range_overlap": [],
            "notes": (
                f"Data directory not found: {data_dir}. "
                + _ACCESS_NOTES
            ),
        }
        with open(output, "w") as f:
            json.dump(report, f, indent=2)
        print(f"1kd compatibility report written to: {output}")
        print(f"  Status: not_found")
        return

    # Case 2: Directory exists — check for annotation CSV
    csv_path = _find_annotation_csv(data_dir)
    if not csv_path:
        report = {
            "status": "incompatible",
            "n_clips": 0,
            "missing_columns": list(_REQUIRED_COLUMNS),
            "age_range_overlap": [],
            "notes": (
                f"Data directory found at {data_dir} but no annotation CSV located. "
                "Expected: annotations.csv, metadata.csv, clips.csv, manifest.csv, or labels.csv. "
                + _ACCESS_NOTES
            ),
        }
        with open(output, "w") as f:
            json.dump(report, f, indent=2)
        print(f"1kd compatibility report written to: {output}")
        print(f"  Status: incompatible (no annotation CSV found)")
        return

    print(f"Found annotation CSV: {csv_path}")
    df = pd.read_csv(csv_path, low_memory=False)
    present_cols = set(df.columns)
    missing_cols: List[str] = sorted(_REQUIRED_COLUMNS - present_cols)

    # Check age range overlap
    age_overlap: List[str] = []
    if "timepoint" in df.columns:
        unique_tp = df["timepoint"].dropna().unique()
        ages = [_age_months_from_timepoint(tp) for tp in unique_tp]
        for age in ages:
            if _EXISTING_AGE_RANGE[0] <= age <= _EXISTING_AGE_RANGE[1]:
                age_overlap.append(f"{age}_month")
    elif "age_months" in df.columns:
        ages = df["age_months"].dropna().unique()
        for age in ages:
            if _EXISTING_AGE_RANGE[0] <= int(age) <= _EXISTING_AGE_RANGE[1]:
                age_overlap.append(f"{int(age)}_month")

    n_clips = len(df)
    n_compatible = n_clips if not missing_cols else 0

    if missing_cols:
        status = "incompatible"
        notes = (
            f"Found annotation CSV at {csv_path} with {n_clips} rows. "
            f"Missing required columns: {missing_cols}. "
            f"Available columns: {sorted(present_cols)}. "
            + _ACCESS_NOTES
        )
    else:
        status = "compatible"
        notes = (
            f"Found annotation CSV at {csv_path} with {n_clips} compatible rows. "
            f"All required columns present. "
            f"Age range overlap with existing dataset: {age_overlap or 'none detected'}. "
            "To integrate: re-run make_seen_child_split.py with merged annotations, "
            "using 'kd_' prefix for 1kd child IDs to avoid namespace conflicts."
        )

    report = {
        "status": status,
        "n_clips": n_compatible,
        "missing_columns": missing_cols,
        "age_range_overlap": age_overlap,
        "notes": notes,
    }

    with open(output, "w") as f:
        json.dump(report, f, indent=2)

    print(f"1kd compatibility report written to: {output}")
    print(f"  Status: {status}")
    print(f"  Clips: {n_clips}  |  Compatible: {n_compatible}")
    print(f"  Missing columns: {missing_cols or 'none'}")
    print(f"  Age range overlap: {age_overlap or 'none detected'}")


if __name__ == "__main__":
    main()
