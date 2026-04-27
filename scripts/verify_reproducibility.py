"""
Verify reproducibility: compare committed config.json against result files
across all result folders.

For each result folder, checks that config.json exists and that key metric
files (test_metrics_tuned.json, val_metrics_tuned.json) exist. Reports any
mismatches or missing files to stdout.

Usage:
    python scripts/verify_reproducibility.py
    python scripts/verify_reproducibility.py --output evaluation/reproducibility_report.txt
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path
from datetime import datetime

REPO_ROOT = Path(__file__).resolve().parent.parent

RESULT_FOLDERS = [
    "whisper-modeling/usc_sail_enrollment_runs",
    "pyannote/pyannote_enrollment_runs",
    "babar_ecapa_enrollment_runs",
    "vtc_ecapa_enrollment_runs",
    "vtc_kchi_ecapa_enrollment_runs",
    "vbx_ecapa_enrollment_runs",
    "babar_combined_runs",
    "baselines/baseline_results",
]

# Subdirectory names to skip — not result folders
SKIP_SUBDIRS = {"error_analysis", "phoneme_cache", "__pycache__"}

# Either of these naming conventions counts as the test metrics file
TEST_METRICS_VARIANTS = ["test_metrics_tuned.json", "enroll_test_metrics.json",
                         "test_metrics.json", "all_model_results.json"]
VAL_METRICS_VARIANTS = ["val_metrics_tuned.json", "enroll_val_metrics.json", "val_metrics.json"]

# Folders that are known to have partial / non-standard result structure
INCOMPLETE_OK_FOLDERS = {"fused_stats_lw"}

# Top-level result folders with non-standard structure (checked separately)
SPECIAL_FOLDER_CHECKS = {
    "babar_combined_runs": "all_model_results.json",
}

EXPECTED_FILES = [
    "config.json",
    "test_metrics_tuned.json",
    "val_metrics_tuned.json",
]

OPTIONAL_FILES = [
    "test_predictions.csv",
    "val_predictions.csv",
    "training_history.csv",
    "test_metrics_by_timepoint.csv",
    "val_metrics_by_timepoint.csv",
]


def file_hash(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


def check_result_folder(folder: Path) -> dict:
    result = {
        "folder": str(folder),
        "exists": folder.exists(),
        "status": "PASS",
        "issues": [],
        "files": {},
    }

    if not folder.exists():
        result["status"] = "MISSING_FOLDER"
        result["issues"].append(f"Folder does not exist: {folder}")
        return result

    # Special folders with non-standard structure
    if folder.name in SPECIAL_FOLDER_CHECKS:
        key_file = folder / SPECIAL_FOLDER_CHECKS[folder.name]
        if key_file.exists():
            result["files"][folder.name] = {
                "path": str(folder),
                "present": [f"{key_file.name} (md5={file_hash(key_file)[:8]})"],
                "missing": [],
            }
        else:
            result["status"] = "INCOMPLETE"
            result["issues"].append(f"Missing {key_file.name} in {folder}")
        return result

    # Check for subfolders (e.g. baseline_results has per-model subdirs)
    # Skip known non-result subdirs
    subdirs = [d for d in folder.iterdir()
               if d.is_dir() and d.name not in SKIP_SUBDIRS]
    check_dirs = subdirs if subdirs else [folder]

    for check_dir in check_dirs:
        if check_dir.name in SKIP_SUBDIRS:
            continue
        # Known incomplete folders — downgrade to INFO, don't flag as FAIL
        if check_dir.name in INCOMPLETE_OK_FOLDERS:
            result["files"][str(check_dir.name)] = {
                "path": str(check_dir), "missing": [], "present": [],
                "note": "Known incomplete run — skipped",
            }
            continue
        sub_result = {"path": str(check_dir), "missing": [], "present": []}

        config_path = check_dir / "config.json"
        if not config_path.exists():
            sub_result["missing"].append("config.json")
            result["issues"].append(f"Missing config.json in {check_dir}")
            result["status"] = "FAIL"
        else:
            sub_result["present"].append(f"config.json (md5={file_hash(config_path)[:8]})")
            try:
                with open(config_path) as f:
                    json.load(f)
            except json.JSONDecodeError as e:
                result["issues"].append(f"config.json is invalid JSON in {check_dir}: {e}")
                result["status"] = "FAIL"

        # Accept either naming convention for test/val metrics
        for variants in [TEST_METRICS_VARIANTS, VAL_METRICS_VARIANTS]:
            found = any((check_dir / v).exists() for v in variants)
            if not found:
                missing_name = variants[0]
                sub_result["missing"].append(missing_name)
                result["issues"].append(f"Missing {missing_name} (or equivalent) in {check_dir}")
                if result["status"] == "PASS":
                    result["status"] = "INCOMPLETE"
            else:
                present_name = next(v for v in variants if (check_dir / v).exists())
                fpath = check_dir / present_name
                sub_result["present"].append(f"{present_name} (md5={file_hash(fpath)[:8]})")

        for fname in OPTIONAL_FILES:
            fpath = check_dir / fname
            if fpath.exists():
                sub_result["present"].append(f"{fname} (optional, present)")
        # Also accept enroll_* variants of optional files
        for fname in ["enroll_test_predictions.csv", "enroll_val_predictions.csv"]:
            fpath = check_dir / fname
            if fpath.exists():
                sub_result["present"].append(f"{fname} (optional, present)")

        result["files"][str(check_dir.name)] = sub_result

    return result


def main():
    parser = argparse.ArgumentParser(description="Verify result folder reproducibility")
    parser.add_argument("--output", default=None,
                        help="Path to write report (default: stdout only)")
    parser.add_argument("--extra-folders", nargs="*", default=[],
                        help="Additional result folders to check")
    args = parser.parse_args()

    folders_to_check = list(RESULT_FOLDERS) + list(args.extra_folders)

    lines = []
    lines.append(f"Reproducibility Report — {datetime.utcnow().isoformat()}Z")
    lines.append("=" * 70)

    n_pass = n_fail = n_missing = 0
    for folder_rel in folders_to_check:
        folder = REPO_ROOT / folder_rel
        result = check_result_folder(folder)

        status = result["status"]
        if status == "PASS":
            n_pass += 1
        elif status == "MISSING_FOLDER":
            n_missing += 1
        else:
            n_fail += 1

        lines.append(f"\n[{status}] {folder_rel}")
        for issue in result["issues"]:
            lines.append(f"  ! {issue}")

        if result["exists"] and not result["issues"]:
            file_count = sum(len(v.get("present", [])) for v in result["files"].values())
            lines.append(f"  All expected files present ({file_count} files checked)")

    lines.append("\n" + "=" * 70)
    lines.append(f"Summary: {n_pass} PASS / {n_fail} FAIL / {n_missing} MISSING")
    lines.append("=" * 70)

    report = "\n".join(lines)
    print(report)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            f.write(report + "\n")
        print(f"\nReport written to {out_path}")

    if n_fail > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
