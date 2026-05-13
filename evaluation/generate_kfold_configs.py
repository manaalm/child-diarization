"""generate_kfold_configs.py — emit per-fold YAML configs from base configs.

For each (base_config, fold) pair, writes a new config that overrides
`split_dir` and `variant_name` (so each fold writes to its own results
folder). Original configs are untouched.

Output directory structure:
  mil/configs/kfold_<K>fold/<base_name>_fold<k>.yaml
  pseudo_frame/configs/kfold_<K>fold/<base_name>_fold<k>.yaml

Run *after* make_kfold_seen_child_split.py. Then submit the SLURM driver
scripts that consume these configs.

Usage:
  python evaluation/generate_kfold_configs.py --k 3
  python evaluation/generate_kfold_configs.py --k 5 --systems wavlm_mil whisper_mil
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List

import yaml

_REPO = Path(__file__).resolve().parent.parent

# Map system name → (base config path, results subdir name).
# results subdir is appended under mil/mil_results/ or pseudo_frame/results/.
SYSTEMS: Dict[str, Dict[str, str]] = {
    # MIL family — `mil/mil_train.py --config <path>`
    "wavlm_mil":            {"base": "mil/configs/wavlm_mil.yaml",         "family": "mil"},
    "whisper_mil":          {"base": "mil/configs/whisper_mil.yaml",       "family": "mil"},
    "whisper_mil_tsmil_concat": {"base": "mil/configs/whisper_mil_tsmil_concat.yaml", "family": "mil"},
    # Pseudo-frame — `pseudo_frame/pseudo_train.py --config <path>`
    "wavlm_pseudo_frame":   {"base": "pseudo_frame/configs/wavlm_pseudo.yaml", "family": "pseudo_frame"},
}


def _kfold_split_dir(k: int, fold: int, variant: str = "within_child") -> str:
    """Path used by the training scripts to find {train,val,test}.csv.

    spec-022 polish: `variant` selects which k-fold split to point at:
      within_child  -> seen_child_splits_kfold_{k}fold/fold_{f}        (legacy)
      within_child_bids -> seen_child_splits_kfold_{k}fold_bids/fold_{f}  (BIDS-corrected within-child)
      groupstrat    -> seen_child_splits_groupstrat_{k}fold/fold_{f}    (children disjoint per fold, BIDS)
    """
    suffix = {
        "within_child":     f"seen_child_splits_kfold_{k}fold/fold_{fold}",
        "within_child_bids": f"seen_child_splits_kfold_{k}fold_bids/fold_{fold}",
        "groupstrat":       f"seen_child_splits_groupstrat_{k}fold/fold_{fold}",
    }[variant]
    return f"whisper-modeling/{suffix}"


def _variant_tag(variant: str, k: int) -> str:
    """Tag appended to result-dir names so variants don't collide."""
    return {
        "within_child":     f"kfold{k}",
        "within_child_bids": f"kfold{k}bids",
        "groupstrat":       f"groupstrat{k}",
    }[variant]


def _fold_config(base_path: Path, k: int, fold: int, system_name: str, variant: str = "within_child") -> dict:
    with open(base_path) as f:
        cfg = yaml.safe_load(f)
    cfg = dict(cfg)  # shallow copy
    cfg["split_dir"] = _kfold_split_dir(k, fold, variant)
    cfg["variant_name"] = f"{system_name}_{_variant_tag(variant, k)}_f{fold}"
    return cfg


def _write_configs(k: int, systems: List[str], variant: str = "within_child") -> List[str]:
    written: List[str] = []
    tag = _variant_tag(variant, k)
    for sys_name in systems:
        if sys_name not in SYSTEMS:
            print(f"  SKIP {sys_name}: not in registry", file=sys.stderr)
            continue
        info = SYSTEMS[sys_name]
        base_path = _REPO / info["base"]
        if not base_path.exists():
            print(f"  SKIP {sys_name}: base config missing at {base_path}",
                  file=sys.stderr)
            continue

        family = info["family"]
        out_dir = _REPO / f"{family}/configs/{tag}"
        if family == "mil":
            out_dir = _REPO / f"mil/configs/{tag}"
        out_dir.mkdir(parents=True, exist_ok=True)

        for fold in range(k):
            cfg = _fold_config(base_path, k, fold, sys_name, variant)
            dst = out_dir / f"{sys_name}_fold{fold}.yaml"
            with open(dst, "w") as f:
                yaml.safe_dump(cfg, f, sort_keys=False)
            written.append(str(dst.relative_to(_REPO)))
            print(f"  wrote {dst.relative_to(_REPO)}")
    return written


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--k", type=int, default=3,
                    help="Number of folds (must match the split builder).")
    ap.add_argument("--variant", choices=["within_child", "within_child_bids", "groupstrat"],
                    default="within_child",
                    help="Which split paradigm to point at (spec-022 US2).")
    ap.add_argument(
        "--systems", nargs="+",
        default=list(SYSTEMS.keys()),
        help="Subset of systems to generate configs for. Default: all.",
    )
    args = ap.parse_args()

    split_root_map = {
        "within_child":     _REPO / f"whisper-modeling/seen_child_splits_kfold_{args.k}fold",
        "within_child_bids": _REPO / f"whisper-modeling/seen_child_splits_kfold_{args.k}fold_bids",
        "groupstrat":       _REPO / f"whisper-modeling/seen_child_splits_groupstrat_{args.k}fold",
    }
    split_root = split_root_map[args.variant]
    if not split_root.exists():
        print(f"ERROR: split dir not found: {split_root}", file=sys.stderr)
        return 1

    print(f"Generating {args.k}-fold ({args.variant}) configs for {len(args.systems)} systems:")
    written = _write_configs(args.k, args.systems, variant=args.variant)
    print(f"\nWrote {len(written)} config files. Next:")
    print(f"  sbatch --array=0-{args.k-1} mil/slurm/train_mil_kfold.sh wavlm_mil")
    print(f"  sbatch --array=0-{args.k-1} pseudo_frame/slurm/train_pseudo_kfold.sh wavlm_pseudo_frame")
    return 0


if __name__ == "__main__":
    sys.exit(main())
