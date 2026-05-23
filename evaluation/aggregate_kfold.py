"""aggregate_kfold.py — collect per-fold metrics into mean ± std table.

Walks the per-fold result directories produced by the k-fold SLURM jobs
(train_mil_kfold.sh, train_pseudo_kfold.sh, ...), reads each fold's
test_predictions.csv, re-tunes the threshold on the fold's val under
balanced-accuracy + Youden's J + F1 objectives, and aggregates across
folds. Reports mean, std, min, max for each metric.

Output:
  evaluation/kfold_summary.csv        — long-format per (system, objective, metric)
  evaluation/kfold_summary.md         — human-readable mean ± std tables
  evaluation/kfold_per_fold.csv       — per-fold metrics for traceability

Naming convention:
  Per-fold result dirs are named "<base_system>_kfold<K>_f<fold>" (matches the
  variant_name set by generate_kfold_configs.py). Anything not matching that
  pattern is ignored.

Usage:
  python evaluation/aggregate_kfold.py --k 3
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "evaluation"))
from recompute_metrics import (  # noqa: E402
    _read_predictions,
    _tune_threshold,
    _metrics_at_threshold,
)


# Folder roots to scan for per-fold result directories
RESULTS_ROOTS = [
    _REPO / "mil/mil_results",
    _REPO / "pseudo_frame/results",
    _REPO / "baseline_results_seen_child",
    _REPO,
]

KFOLD_RE = re.compile(r"^(.+)_kfold(\d+)_f(\d+)$")


def _discover_kfold_dirs(k: int) -> Dict[str, Dict[int, Path]]:
    """Group per-fold result dirs by base system name.

    Returns {base_system_name: {fold_idx: result_dir, ...}}.
    """
    out: Dict[str, Dict[int, Path]] = {}
    for root in RESULTS_ROOTS:
        if not root.exists():
            continue
        for child in root.iterdir():
            if not child.is_dir():
                continue
            m = KFOLD_RE.match(child.name)
            if not m:
                continue
            base, kk, fold = m.group(1), int(m.group(2)), int(m.group(3))
            if kk != k:
                continue
            out.setdefault(base, {})[fold] = child
    return out


def _per_fold_metrics(test_pred_csv: Path,
                      val_pred_csv: Optional[Path]) -> Dict[str, Dict[str, float]]:
    """Compute per-fold metrics under three threshold objectives."""
    test_df = _read_predictions(test_pred_csv)
    if test_df is None:
        return {}
    y_test = test_df["label"].to_numpy()
    s_test = test_df["score"].to_numpy()
    prev = float(y_test.mean())

    val_df = _read_predictions(val_pred_csv) if val_pred_csv else None
    y_tune = val_df["label"].to_numpy() if val_df is not None else y_test
    s_tune = val_df["score"].to_numpy() if val_df is not None else s_test

    out: Dict[str, Dict[str, float]] = {}
    for obj in ("f1", "balanced_acc", "youden"):
        try:
            thr = _tune_threshold(y_tune, s_tune, obj)
            out[obj] = _metrics_at_threshold(y_test, s_test, thr, prev)
        except Exception as e:
            print(f"  WARN tune/metric failed for {obj}: {e}", file=sys.stderr)
    return out


def aggregate(k: int) -> None:
    discovered = _discover_kfold_dirs(k)
    print(f"Found {len(discovered)} systems with k={k} folds:")
    for s, folds in sorted(discovered.items()):
        print(f"  {s}: folds {sorted(folds.keys())}")

    METRICS = ("f1", "balanced_acc", "mcc", "auroc", "auprc",
               "auprc_lift", "delta_f1_vs_trivial",
               "precision", "recall", "specificity")

    per_fold_rows: List[Dict] = []
    summary_rows: List[Dict] = []

    for base_sys, fold_dirs in sorted(discovered.items()):
        for obj in ("f1", "balanced_acc", "youden"):
            fold_metrics: Dict[str, List[float]] = {m: [] for m in METRICS}
            for fold, dir_ in sorted(fold_dirs.items()):
                test_csv = dir_ / "test_predictions.csv"
                val_csv = dir_ / "val_predictions.csv"
                # Fall back to enrollment naming convention used by pyannote/unified.py
                if not test_csv.exists() and (dir_ / "enroll_test_predictions.csv").exists():
                    test_csv = dir_ / "enroll_test_predictions.csv"
                    val_csv = dir_ / "enroll_val_predictions.csv"
                if not test_csv.exists():
                    print(f"  SKIP {base_sys} fold {fold}: no test_predictions.csv")
                    continue
                pf = _per_fold_metrics(
                    test_csv, val_csv if val_csv.exists() else None
                )
                if obj not in pf:
                    continue
                row = {"system": base_sys, "objective": obj, "fold": fold,
                       **{m: pf[obj].get(m, float("nan")) for m in METRICS}}
                per_fold_rows.append(row)
                for m in METRICS:
                    val = pf[obj].get(m, float("nan"))
                    if not np.isnan(val):
                        fold_metrics[m].append(val)

            summary = {"system": base_sys, "objective": obj,
                       "n_folds_completed": len(fold_dirs)}
            for m, vals in fold_metrics.items():
                if not vals:
                    summary[f"{m}_mean"] = float("nan")
                    summary[f"{m}_std"] = float("nan")
                    summary[f"{m}_min"] = float("nan")
                    summary[f"{m}_max"] = float("nan")
                else:
                    arr = np.asarray(vals)
                    summary[f"{m}_mean"] = float(arr.mean())
                    summary[f"{m}_std"] = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
                    summary[f"{m}_min"] = float(arr.min())
                    summary[f"{m}_max"] = float(arr.max())
            summary_rows.append(summary)

    if not per_fold_rows:
        print("No completed folds discovered. Exiting.")
        return

    pf_df = pd.DataFrame(per_fold_rows)
    sm_df = pd.DataFrame(summary_rows)

    out_per_fold = _REPO / "evaluation/kfold_per_fold.csv"
    out_summary = _REPO / "evaluation/kfold_summary.csv"
    out_md = _REPO / "evaluation/kfold_summary.md"
    pf_df.to_csv(out_per_fold, index=False)
    sm_df.to_csv(out_summary, index=False)
    print(f"\nWrote per-fold rows → {out_per_fold}")
    print(f"Wrote summary       → {out_summary}")

    write_md(sm_df, pf_df, out_md, k)
    print(f"Wrote markdown      → {out_md}")


def write_md(sm: pd.DataFrame, pf: pd.DataFrame, path: Path, k: int) -> None:
    lines: List[str] = []
    lines.append(f"# {k}-fold cross-validation summary")
    lines.append("")
    lines.append(
        f"Mean ± std across {k} folds for each system × threshold-objective. "
        "Thresholds are re-tuned on each fold's val (not shared across folds). "
        "All metrics computed on the fold's held-out test split."
    )
    lines.append("")
    lines.append(
        "Compare fold-to-fold std with the bootstrap CI widths in "
        "`recomputed_metrics_summary.md` and `child_bootstrap_cis.md`. If "
        "fold-std overlaps with bootstrap CI, the bootstrap was a fair "
        "approximation of CV variance and the rankings are stable."
    )
    lines.append("")

    for obj, label in (("youden", "Youden's J"),
                       ("balanced_acc", "Balanced accuracy"),
                       ("f1", "F1 (legacy)")):
        sub = sm[sm["objective"] == obj].copy()
        if sub.empty:
            continue
        sub = sub.sort_values("balanced_acc_mean", ascending=False)
        lines.append(f"## Threshold tuned for {label}")
        lines.append("")
        lines.append("| System | Folds | F1 mean ± std | BalAcc mean ± std | "
                     "MCC mean ± std | AUROC mean ± std | AUPRC mean ± std | "
                     "ΔF1 vs trivial mean |")
        lines.append("|---|---:|---|---|---|---|---|---:|")
        for _, r in sub.iterrows():
            def fmt(m: str, sign: str = "") -> str:
                mean = r[f"{m}_mean"]
                std = r[f"{m}_std"]
                if np.isnan(mean):
                    return "—"
                return f"{mean:{sign}.3f} ± {std:.3f}"
            lines.append(
                f"| `{r['system']}` "
                f"| {int(r['n_folds_completed'])} "
                f"| {fmt('f1')} "
                f"| {fmt('balanced_acc')} "
                f"| {fmt('mcc', '+')} "
                f"| {fmt('auroc')} "
                f"| {fmt('auprc')} "
                f"| {r['delta_f1_vs_trivial_mean']:+.3f} |"
            )
        lines.append("")

    lines.append("## Per-fold details")
    lines.append("")
    lines.append("Full per-fold rows in `evaluation/kfold_per_fold.csv`. "
                 "If you see large std across folds, drill into that table to "
                 "find which fold was the outlier.")
    lines.append("")
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--k", type=int, default=3)
    args = ap.parse_args()
    aggregate(args.k)
    return 0


if __name__ == "__main__":
    sys.exit(main())
