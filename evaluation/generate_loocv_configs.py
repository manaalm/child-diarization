"""generate_loocv_configs.py — emit per-fold YAML configs for LOOCV runs.

For each (system, fold) pair, writes a new config that overrides:
  - split_dir = whisper-modeling/seen_child_splits_loocv/fold_<f>
  - variant_name = <system>_loocv_f<f>

so each LOOCV fold writes to its own results folder. Original configs are
untouched. Run AFTER evaluation/loocv_split.py has produced the 130 fold
directories.

Output:
  mil/configs/loocv/<system>_fold<f>.yaml
  pseudo_frame/configs/loocv/<system>_fold<f>.yaml

Usage:
  python evaluation/generate_loocv_configs.py --systems whisper_mil whisper_medium_mil
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List

import yaml

_REPO = Path(__file__).resolve().parent.parent

SYSTEMS: Dict[str, Dict[str, str]] = {
    "wavlm_mil":               {"base": "mil/configs/wavlm_mil.yaml",                "family": "mil"},
    "whisper_mil":             {"base": "mil/configs/whisper_mil.yaml",              "family": "mil"},
    "whisper_medium_mil":      {"base": "mil/configs/whisper_medium_mil.yaml",       "family": "mil"},
    "whisper_mil_acmil_max":   {"base": "mil/configs/whisper_mil_acmil_max.yaml",    "family": "mil"},
    "whisper_mil_tsmil_concat": {"base": "mil/configs/whisper_mil_tsmil_concat.yaml", "family": "mil"},
    "whisper_pseudo_frame":    {"base": "pseudo_frame/configs/whisper_pseudo.yaml",   "family": "pseudo_frame"},
    "wavlm_pseudo_frame":      {"base": "pseudo_frame/configs/wavlm_pseudo.yaml",     "family": "pseudo_frame"},
}


def _write(system: str, n_folds: int) -> List[str]:
    info = SYSTEMS[system]
    base_path = _REPO / info["base"]
    if not base_path.exists():
        print(f"  SKIP {system}: base config missing at {base_path}", file=sys.stderr)
        return []

    family = info["family"]
    out_dir = _REPO / f"{family}/configs/loocv"
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(base_path) as f:
        base_cfg = yaml.safe_load(f)

    written = []
    for fold in range(n_folds):
        cfg = dict(base_cfg)
        cfg["split_dir"] = f"whisper-modeling/seen_child_splits_loocv/fold_{fold}"
        cfg["variant_name"] = f"{system}_loocv_f{fold}"
        dst = out_dir / f"{system}_fold{fold}.yaml"
        with open(dst, "w") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)
        written.append(str(dst.relative_to(_REPO)))
    return written


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--systems", nargs="+", default=list(SYSTEMS.keys()))
    ap.add_argument(
        "--n-folds", type=int, default=130,
        help="Number of LOOCV folds (default 130 = #children in BIDS seen-child master).",
    )
    args = ap.parse_args()

    split_root = _REPO / "whisper-modeling/seen_child_splits_loocv"
    if not split_root.exists():
        print(f"ERROR: LOOCV split dir not found: {split_root}", file=sys.stderr)
        print("Run: python evaluation/loocv_split.py", file=sys.stderr)
        return 1

    total = 0
    for system in args.systems:
        if system not in SYSTEMS:
            print(f"  SKIP {system}: not in registry", file=sys.stderr)
            continue
        written = _write(system, args.n_folds)
        print(f"  {system}: wrote {len(written)} configs")
        total += len(written)

    print(f"\nTotal: {total} LOOCV config files.")
    print("Dispatch with: sbatch --array=0-129%25 mil/slurm/train_mil_loocv.sh <system>")
    return 0


if __name__ == "__main__":
    sys.exit(main())
