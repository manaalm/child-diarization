#!/usr/bin/env python3
"""
Train (run enrollment) at each synthetic augmentation ratio.

For each ratio manifest in --manifest-dir, this script substitutes the
augmented CSV for the real training split and calls the BabAR enrollment
pipeline (pyannote/unified.py --diarizer babar) with the synthetic-augmented
training data.

The enrolled ECAPA prototype and evaluation predictions are saved to:
    {output_dir}/{config_name}/ratio_{r}x/

Usage:
    python synth/scripts/train_with_synthetic.py \\
      --manifest-dir  synth_results/manifests/ \\
      --ratios        0 0.5 1 2 5 10 \\
      --output-dir    synth_results/augmentation_experiments/default_14_18mo/
"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# The BabAR enrollment script to invoke.
# pyannote/unified.py supports --train-csv to substitute a custom training CSV.
_UNIFIED_SCRIPT = _REPO_ROOT / "pyannote" / "unified.py"
_PYTHON = sys.executable


def _ratio_to_str(ratio: float) -> str:
    return str(int(ratio)) if ratio == int(ratio) else str(ratio)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run BabAR enrollment for each synthetic augmentation ratio."
    )
    parser.add_argument(
        "--manifest-dir",
        required=True,
        help="Directory containing train_{ratio}x_manifest.csv files.",
    )
    parser.add_argument(
        "--ratios",
        type=float,
        nargs="+",
        default=[0, 0.5, 1, 2, 5, 10],
        help="Ratios to run (must match files in manifest-dir).",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Base directory for experiment outputs.",
    )
    parser.add_argument(
        "--val-csv",
        default=str(_REPO_ROOT / "whisper-modeling" / "seen_child_splits" / "val.csv"),
        help="Validation CSV (default: seen_child_splits/val.csv).",
    )
    parser.add_argument(
        "--test-csv",
        default=str(_REPO_ROOT / "whisper-modeling" / "seen_child_splits" / "test.csv"),
        help="Test CSV (default: seen_child_splits/test.csv).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the commands that would be run without executing them.",
    )
    args = parser.parse_args()

    manifest_dir = Path(args.manifest_dir)
    out_base = Path(args.output_dir)

    for ratio in args.ratios:
        ratio_str = _ratio_to_str(ratio)
        manifest_path = manifest_dir / f"train_{ratio_str}x_manifest.csv"

        if not manifest_path.exists():
            print(
                f"  [SKIP] Manifest not found for ratio={ratio_str}x: "
                f"{manifest_path}",
                file=sys.stderr,
            )
            continue

        ratio_out = out_base / f"ratio_{ratio_str}x"
        ratio_out.mkdir(parents=True, exist_ok=True)

        # Check if already done (resume-safe)
        done_marker = ratio_out / "test_metrics_tuned.json"
        if done_marker.exists():
            print(f"  [SKIP] ratio={ratio_str}x already complete ({done_marker})")
            continue

        # Invoke pyannote/unified.py with the augmented training CSV.
        #
        # pyannote/unified.py accepts --train-csv to override the default
        # seen_child_splits/train.csv used for enrollment prototype building.
        # The diarizer (babar) produces RTTMs; the enrollment wrapper then
        # builds ECAPA prototypes from the augmented training set.
        cmd = [
            _PYTHON,
            str(_UNIFIED_SCRIPT),
            "--diarizer", "babar",
            "--train-csv", str(manifest_path),
            "--val-csv", args.val_csv,
            "--test-csv", args.test_csv,
            "--output-dir", str(ratio_out),
        ]

        print(f"\nratio={ratio_str}x: {' '.join(cmd)}")

        if args.dry_run:
            continue

        result = subprocess.run(cmd, cwd=str(_REPO_ROOT))
        if result.returncode != 0:
            print(
                f"  [ERROR] ratio={ratio_str}x enrollment failed "
                f"(exit code {result.returncode}). See output above.",
                file=sys.stderr,
            )
            # Continue with other ratios rather than aborting
            continue

        print(f"  ratio={ratio_str}x complete → {ratio_out}")

    print("\nAll ratio enrollments finished.")


if __name__ == "__main__":
    main()
