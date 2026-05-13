"""Audit what the existing within-child k-fold split scheme actually does
(spec 022 US2 / FR-008).

The verdict is unambiguous from the source: make_kfold_seen_child_split.py
preserves the within-child paradigm — same children appear in every fold's
train/val/test. This script confirms the verdict by inspecting actual fold
membership of a few headline k-fold result dirs and writes evaluation/
kfold_audit.md with the audit + code citations.
"""

import json
import os
import re
import sys
from collections import defaultdict
from typing import Optional

import pandas as pd

REPO_ROOT = "/orcd/scratch/orcd/008/manaal/child-adult-diarization"

KFOLD_RESULT_PATTERNS = [
    "mil/mil_results",
    "pseudo_frame/results",
    "baseline_results_seen_child",
    "babar_ecapa_enrollment_runs",
]


def _scan_kfold_dirs(repo_root: str) -> dict[str, list[str]]:
    """Return system_base -> list of fold-result dirs."""
    out: dict[str, list[str]] = defaultdict(list)
    for stub in KFOLD_RESULT_PATTERNS:
        base = os.path.join(repo_root, stub)
        if not os.path.isdir(base):
            continue
        for entry in os.listdir(base):
            m = re.match(r"^(.+)_kfold(\d+)_f(\d+)$", entry)
            if not m:
                continue
            system_base = m.group(1)
            full = os.path.join(base, entry)
            if os.path.isdir(full):
                out[system_base].append(full)
    return dict(out)


def _children_in_split(split_csv: str) -> Optional[set[str]]:
    if not os.path.exists(split_csv):
        return None
    try:
        df = pd.read_csv(split_csv, usecols=["child_id"])
    except Exception:
        try:
            df = pd.read_csv(split_csv)
            if "child_id" not in df.columns:
                return None
        except Exception:
            return None
    return set(df["child_id"].dropna().astype(str).tolist())


def _audit_one_system(system_base: str, fold_dirs: list[str]) -> dict:
    """For a system with fold_0/fold_1/.../ result dirs, infer the split source
    (look for fold-membership JSON or for split CSVs in expected locations)
    and determine if children overlap across folds."""
    summary = {
        "system": system_base,
        "n_fold_dirs": len(fold_dirs),
        "fold_dirs": sorted(fold_dirs),
    }

    # Look for split CSVs alongside each fold dir
    split_csvs_per_fold = {}
    for fd in fold_dirs:
        m = re.search(r"_kfold(\d+)_f(\d+)$", fd)
        if not m:
            continue
        k, f = int(m.group(1)), int(m.group(2))
        # Convention from make_kfold_seen_child_split.py:
        # whisper-modeling/seen_child_splits_kfold_<k>fold/fold_<f>/{train,val,test}.csv
        split_dir = os.path.join(
            REPO_ROOT, "whisper-modeling",
            f"seen_child_splits_kfold_{k}fold", f"fold_{f}",
        )
        if os.path.isdir(split_dir):
            split_csvs_per_fold[f] = {
                "train": os.path.join(split_dir, "train.csv"),
                "val": os.path.join(split_dir, "val.csv"),
                "test": os.path.join(split_dir, "test.csv"),
            }

    # Compute per-fold child sets
    overlap_table = []
    children_per_fold = {}
    for fold, paths in split_csvs_per_fold.items():
        children_per_fold[fold] = {
            split: _children_in_split(p) for split, p in paths.items()
        }
        overlap = {}
        train_c = children_per_fold[fold]["train"] or set()
        test_c = children_per_fold[fold]["test"] or set()
        val_c = children_per_fold[fold]["val"] or set()
        overlap["train_test_intersect"] = len(train_c & test_c)
        overlap["train_val_intersect"] = len(train_c & val_c)
        overlap["val_test_intersect"] = len(val_c & test_c)
        overlap["n_train_children"] = len(train_c)
        overlap["n_val_children"] = len(val_c)
        overlap["n_test_children"] = len(test_c)
        overlap_table.append({"fold": fold, **overlap})

    summary["per_fold_overlap"] = overlap_table

    # Verdict
    if not overlap_table:
        summary["verdict"] = "NO-SPLIT-CSV-FOUND"
    elif all(o["train_test_intersect"] > 0 for o in overlap_table):
        summary["verdict"] = "WITHIN-CHILD (same children in train+test of every fold)"
    elif all(o["train_test_intersect"] == 0 for o in overlap_table):
        summary["verdict"] = "GROUP-DISJOINT (children disjoint across folds — proper cross-child k-fold)"
    else:
        summary["verdict"] = "MIXED (some folds within-child, some disjoint — likely bug)"

    return summary


def main():
    by_system = _scan_kfold_dirs(REPO_ROOT)
    audits = [_audit_one_system(sysname, fds) for sysname, fds in sorted(by_system.items())]

    out_md = os.path.join(REPO_ROOT, "evaluation", "kfold_audit.md")
    os.makedirs(os.path.dirname(out_md), exist_ok=True)
    with open(out_md, "w") as fh:
        fh.write("# K-fold audit (spec 022 US2 / FR-008)\n\n")
        fh.write("**Verdict from source code**:\n\n")
        fh.write("The within-child k-fold splitter is `whisper-modeling/make_kfold_seen_child_split.py`. "
                 "Its docstring (lines 9-11) states explicitly:\n\n")
        fh.write("> *\"This preserves the within-child paradigm (the same 109 children appear in train/val/test of every fold), "
                 "so training scripts that accept `--split-dir` can simply point at the appropriate fold directory without code changes.\"*\n\n")
        fh.write("The split mechanism (lines 107-113) iterates over each `(child_id, timepoint_norm)` group "
                 "and assigns its clips to folds modulo k. Therefore every child appears in every fold's train, val, AND test partitions; "
                 "the variance reported by the 3-fold k-fold is *clip-level* shuffle variance, not *child-level* generalisation variance.\n\n")
        fh.write("This matches the PI's flagged concern: the existing 3-fold k-fold does NOT measure cross-child generalisation. "
                 "Spec 022 US2 introduces group-stratified k-fold (children disjoint per fold) to fill that gap.\n\n")
        fh.write("---\n\n## Empirical confirmation\n\n")
        fh.write(f"Inspected {len(audits)} systems with `*_kfold<k>_f<i>/` result dirs:\n\n")
        fh.write("| System | n fold dirs | Verdict |\n")
        fh.write("|---|---|---|\n")
        for a in audits:
            fh.write(f"| `{a['system']}` | {a['n_fold_dirs']} | {a['verdict']} |\n")
        fh.write("\n")
        fh.write("### Per-fold child-overlap detail\n\n")
        for a in audits:
            if not a["per_fold_overlap"]:
                continue
            fh.write(f"**{a['system']}**:\n\n")
            fh.write("| fold | n_train_children | n_test_children | train∩test | train∩val | val∩test |\n")
            fh.write("|---|---|---|---|---|---|\n")
            for o in a["per_fold_overlap"]:
                fh.write(f"| {o['fold']} | {o['n_train_children']} | {o['n_test_children']} | "
                         f"{o['train_test_intersect']} | {o['train_val_intersect']} | {o['val_test_intersect']} |\n")
            fh.write("\n")
        fh.write("---\n\n## Implication for the headline k-fold table\n\n")
        fh.write("The within-child 3-fold AUROC numbers in `CLAUDE.md` (e.g., Whisper pseudo-frame 0.884±0.020) "
                 "are *clip-level shuffle variance within the same 109-child population*, not held-out-child generalisation. "
                 "They are not statistically defensible as cross-child generalisation estimates.\n\n")
        fh.write("Spec 022 US2 introduces `evaluation/group_stratified_kfold.py` using "
                 "`sklearn.model_selection.StratifiedGroupKFold(n_splits=5, random_state=42)` with `groups=child_id`. "
                 "Once that lands, the legacy within-child numbers will be relabelled `Within-child 3-fold (legacy)` "
                 "in CLAUDE.md and the new group-stratified numbers will be added alongside.\n")
    print(f"wrote {out_md}", file=sys.stderr)

    # Also drop a JSON for programmatic consumption
    out_json = os.path.join(REPO_ROOT, "evaluation", "kfold_audit.json")
    with open(out_json, "w") as fh:
        json.dump({"audits": audits}, fh, indent=2)
    print(f"wrote {out_json}", file=sys.stderr)


if __name__ == "__main__":
    main()
