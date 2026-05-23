#!/usr/bin/env python3
"""Validate a spec-021 ResultDir against contracts/result_json_schema.md.

Exits 0 on PASS, 1 on FAIL with a diagnostic message to stderr.

Usage:
    python validate_result_dir.py <result-dir>
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

REQUIRED_METRIC_KEYS = [
    "split", "n_clips", "tuned_threshold", "val_f1", "f1", "precision",
    "recall", "auroc", "auprc", "balanced_accuracy", "by_timepoint",
]
REQUIRED_PRED_COLS = ["clip_path", "child_id", "timepoint_norm", "label", "score", "prob", "pred"]
TIMEPOINT_KEYS = ["14_month", "36_month"]


def find_metric_json(d: Path) -> Path:
    for name in ("test_metrics_tuned.json", "enroll_test_metrics.json"):
        if (d / name).exists():
            return d / name
    raise FileNotFoundError(f"no test_metrics_tuned.json or enroll_test_metrics.json in {d}")


def find_pred_csv(d: Path) -> Path:
    for name in ("test_predictions.csv", "enroll_test_predictions.csv"):
        if (d / name).exists():
            return d / name
    raise FileNotFoundError(f"no test_predictions.csv or enroll_test_predictions.csv in {d}")


def validate_metrics(d: Path, errors: list) -> dict | None:
    try:
        path = find_metric_json(d)
    except FileNotFoundError as e:
        errors.append(str(e))
        return None
    with path.open() as f:
        m = json.load(f)
    for k in REQUIRED_METRIC_KEYS:
        if k not in m:
            errors.append(f"{path}: missing required key '{k}'")
    if "by_timepoint" in m:
        for tp in TIMEPOINT_KEYS:
            if tp not in m["by_timepoint"]:
                errors.append(f"{path}: by_timepoint missing key '{tp}'")
    return m


def validate_predictions(d: Path, errors: list) -> None:
    try:
        path = find_pred_csv(d)
    except FileNotFoundError as e:
        errors.append(str(e))
        return
    df = pd.read_csv(path)
    for col in REQUIRED_PRED_COLS:
        if col not in df.columns:
            errors.append(f"{path}: missing required column '{col}'")


def validate_readme(d: Path, errors: list) -> None:
    p = d / "README.md"
    if not p.exists():
        errors.append(f"{d}: README.md missing (VR-6 verdict line required)")
        return
    lines = p.read_text().splitlines()
    if len(lines) < 3:
        errors.append(f"{p}: README has fewer than 3 lines (verdict block expected)")
        return
    verdict_line = next((ln for ln in lines[:5] if "**Verdict**" in ln), None)
    if verdict_line is None:
        errors.append(f"{p}: no '**Verdict**' line in first 5 lines")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("result_dir", type=Path)
    ap.add_argument("--skip-readme", action="store_true",
                    help="Skip README.md verdict-line check (use during early scaffolding)")
    args = ap.parse_args()

    d = args.result_dir
    if not d.is_dir():
        print(f"FAIL: not a directory: {d}", file=sys.stderr)
        return 1

    errors: list[str] = []
    validate_metrics(d, errors)
    validate_predictions(d, errors)
    if not args.skip_readme:
        validate_readme(d, errors)

    if errors:
        print(f"FAIL ({len(errors)} issues) in {d}:", file=sys.stderr)
        for e in errors:
            print(f"  - {e}", file=sys.stderr)
        return 1
    print(f"PASS {d}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
