"""child_bootstrap.py — child-stratified bootstrap CIs (CV substitute).

Why this script exists
----------------------
Proper k-fold cross-validation requires retraining each model k times, which
isn't feasible across our 12+ systems. The next-best alternative — and the
one reviewers in this literature accept as a CV substitute — is the
**hierarchical / cluster bootstrap**: resample the unit of independence
(here, children), not individual clips, then recompute metrics.

This gives a 95% CI that captures *between-child* variance, which is the
relevant uncertainty for "how would this model do on a different sample of
children?" — exactly what k-fold CV tries to answer.

For systems where the same children appear in train and test (seen-child
split), the CI tells you "how stable is performance under resampling of
the same population" — looser than per-clip CIs but still meaningful.

For systems where train and test children are disjoint (cross-child split),
the CI is a direct estimate of how much your reported number would move
under a different held-out cohort — i.e. fold-to-fold variance.

Output
------
evaluation/child_bootstrap_cis.csv with rows keyed on
  (system, objective)
and columns: f1_lo/hi, balanced_acc_lo/hi, mcc_lo/hi, auroc_lo/hi, auprc_lo/hi.

Run *after* recompute_metrics.py — re-uses the val-tuned thresholds via
the same threshold-selection logic on val_predictions.csv.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)

# Re-use schema helpers from the per-clip script
import sys
_THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _THIS)
from recompute_metrics import _discover_pairs, _read_predictions, _tune_threshold  # noqa: E402

CHILD_COL_CANDIDATES = ("child_id", "child", "subject", "speaker_id")


def _read_with_child(csv_path: Path) -> Optional[pd.DataFrame]:
    """Read a predictions CSV and keep label/score plus a child grouping column."""
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return None
    label_col = next((c for c in ("label", "y_true", "true") if c in df.columns), None)
    score_col = next((c for c in ("score", "prob", "probability") if c in df.columns), None)
    child_col = next((c for c in CHILD_COL_CANDIDATES if c in df.columns), None)
    if label_col is None or score_col is None:
        return None
    out = pd.DataFrame({
        "label": pd.to_numeric(df[label_col], errors="coerce"),
        "score": pd.to_numeric(df[score_col], errors="coerce"),
        "child_id": df[child_col].astype(str) if child_col else "_ALL",
    }).dropna(subset=["label", "score"])
    if out.empty or out["label"].nunique() < 2:
        return None
    return out


def _metric_set(y: np.ndarray, scores: np.ndarray, threshold: float) -> Dict[str, float]:
    pred = (scores >= threshold).astype(int)
    out: Dict[str, float] = {}
    try:
        out["f1"] = f1_score(y, pred, zero_division=0)
    except Exception:
        out["f1"] = float("nan")
    try:
        out["balanced_acc"] = balanced_accuracy_score(y, pred)
    except Exception:
        out["balanced_acc"] = float("nan")
    try:
        out["mcc"] = matthews_corrcoef(y, pred)
    except Exception:
        out["mcc"] = float("nan")
    try:
        out["auroc"] = float(roc_auc_score(y, scores)) if len(np.unique(y)) > 1 else float("nan")
    except Exception:
        out["auroc"] = float("nan")
    try:
        out["auprc"] = float(average_precision_score(y, scores)) if len(np.unique(y)) > 1 else float("nan")
    except Exception:
        out["auprc"] = float("nan")
    return out


def _child_bootstrap(
    df: pd.DataFrame,
    threshold: float,
    n_bootstrap: int,
    seed: int,
) -> Dict[str, Tuple[float, float]]:
    """Resample children with replacement; for each resample take all clips
    of the chosen children, then compute metrics. Returns {metric: (lo, hi)}.
    """
    children = df["child_id"].unique()
    rng = np.random.default_rng(seed)
    by_child = {c: df[df["child_id"] == c] for c in children}

    samples: Dict[str, List[float]] = {k: [] for k in
                                        ("f1", "balanced_acc", "mcc", "auroc", "auprc")}
    for _ in range(n_bootstrap):
        chosen = rng.choice(children, size=children.size, replace=True)
        rows = pd.concat([by_child[c] for c in chosen], ignore_index=True)
        if rows["label"].nunique() < 2:
            continue
        m = _metric_set(rows["label"].to_numpy(),
                        rows["score"].to_numpy(),
                        threshold)
        for k, v in m.items():
            if not np.isnan(v):
                samples[k].append(v)

    out: Dict[str, Tuple[float, float]] = {}
    for k, vals in samples.items():
        if not vals:
            out[k] = (float("nan"), float("nan"))
        else:
            arr = np.asarray(vals)
            out[k] = (float(np.percentile(arr, 2.5)),
                      float(np.percentile(arr, 97.5)))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=".")
    ap.add_argument("--out", default="evaluation/child_bootstrap_cis.csv")
    ap.add_argument("--n-bootstrap", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--objectives", nargs="+",
                    default=["f1", "balanced_acc", "youden"])
    args = ap.parse_args()

    root = Path(args.root).resolve()
    pairs = _discover_pairs(root)
    print(f"Found {len(pairs)} candidate (test, val) prediction pairs.")

    rows: List[Dict] = []
    for sys_name, test_csv, val_csv in pairs:
        test_df = _read_with_child(test_csv)
        if test_df is None:
            continue
        n_children = test_df["child_id"].nunique()
        if n_children < 2:
            continue  # bootstrap is meaningless with one child

        val_df = _read_with_child(val_csv) if val_csv else None
        y_tune = val_df["label"].to_numpy() if val_df is not None else test_df["label"].to_numpy()
        s_tune = val_df["score"].to_numpy() if val_df is not None else test_df["score"].to_numpy()

        for obj in args.objectives:
            try:
                thr = _tune_threshold(y_tune, s_tune, obj)
            except Exception:
                continue
            cis = _child_bootstrap(test_df, thr, args.n_bootstrap, args.seed)
            row = {
                "system": sys_name,
                "objective": obj,
                "threshold": thr,
                "n_children_test": n_children,
                "n_clips_test": len(test_df),
            }
            for k, (lo, hi) in cis.items():
                row[f"{k}_child_ci_low"] = lo
                row[f"{k}_child_ci_high"] = hi
            rows.append(row)

    if not rows:
        print("No predictions with usable child_id columns.")
        return 1

    out_df = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out, index=False)
    print(f"Wrote {len(out_df)} rows → {args.out}")

    # Compact summary: one block per objective, sorted by AUROC CI lower bound.
    md_path = Path(args.out).with_suffix(".md")
    write_md(out_df, md_path)
    print(f"Wrote summary → {md_path}")
    return 0


def write_md(df: pd.DataFrame, path: Path) -> None:
    lines: List[str] = []
    lines.append("# Child-stratified bootstrap CIs")
    lines.append("")
    lines.append("Each row = one (system, threshold-objective) pair. CIs are 95% percentile from "
                 "1000 bootstrap iterations resampling **children** with replacement (then taking "
                 "all clips of the chosen children). This is the cluster-bootstrap CV substitute "
                 "— it captures the variance you'd expect under resampling the cohort, which is "
                 "what k-fold CV tries to estimate.")
    lines.append("")
    lines.append("Wider CIs than per-clip bootstrap is expected and meaningful: clips within a "
                 "child are correlated (same speaker, same recording conditions), so the per-clip "
                 "bootstrap underestimates uncertainty.")
    lines.append("")
    for obj, label in (("youden", "Youden's J"),
                       ("balanced_acc", "Balanced accuracy"),
                       ("f1", "F1 (legacy)")):
        sub = df[df["objective"] == obj].copy()
        if sub.empty:
            continue
        # Sort by midpoint of AUROC CI (best discriminator at top).
        sub["_mid"] = (sub["auroc_child_ci_low"] + sub["auroc_child_ci_high"]) / 2
        sub = sub.sort_values("_mid", ascending=False)
        lines.append(f"## Threshold tuned for {label}")
        lines.append("")
        lines.append("| System | n_children | n_clips | Thr | F1 95% CI | BalAcc 95% CI | MCC 95% CI | AUROC 95% CI | AUPRC 95% CI |")
        lines.append("|---|---:|---:|---:|---|---|---|---|---|")
        for _, r in sub.iterrows():
            lines.append(
                f"| `{r['system']}` "
                f"| {int(r['n_children_test'])} "
                f"| {int(r['n_clips_test'])} "
                f"| {r['threshold']:.3f} "
                f"| [{r['f1_child_ci_low']:.3f}, {r['f1_child_ci_high']:.3f}] "
                f"| [{r['balanced_acc_child_ci_low']:.3f}, {r['balanced_acc_child_ci_high']:.3f}] "
                f"| [{r['mcc_child_ci_low']:+.3f}, {r['mcc_child_ci_high']:+.3f}] "
                f"| [{r['auroc_child_ci_low']:.3f}, {r['auroc_child_ci_high']:.3f}] "
                f"| [{r['auprc_child_ci_low']:.3f}, {r['auprc_child_ci_high']:.3f}] |"
            )
        lines.append("")
    path.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    sys.exit(main())
