"""recompute_metrics.py — re-evaluate every model's saved predictions.

Walks the repo for predictions CSVs (test + matching val), then for each system:
  1. Re-tunes the decision threshold on val under three objectives:
       (a) F1            — what was originally used
       (b) balanced_acc  — robust to class imbalance
       (c) Youden's J    — TPR − FPR; picks point of maximum lift
  2. Computes a richer metric set on test using each tuned threshold.
  3. Reports lift over the trivial "predict all positive" baseline.
  4. Adds stratified bootstrap CIs (resamples positives and negatives
     independently to preserve the test-set prevalence).

Outputs:
  evaluation/recomputed_metrics.csv      — one row per (system, threshold-objective)
  evaluation/recomputed_metrics_summary.md — readable summary

Usage:
  python evaluation/recompute_metrics.py [--n-bootstrap 1000] [--seed 42]
                                         [--root /path/to/repo]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


# ---------------------------------------------------------------------------
# Schema normalization
# ---------------------------------------------------------------------------

LABEL_COL_CANDIDATES = ("label", "y_true", "true")
SCORE_COL_CANDIDATES = ("score", "prob", "probability", "logit_prob")


def _read_predictions(csv_path: Path) -> Optional[pd.DataFrame]:
    """Read a predictions CSV and return a DataFrame with columns
    ``label`` (int 0/1) and ``score`` (float). Returns None if the file
    cannot be parsed in the expected form.
    """
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return None

    label_col = next((c for c in LABEL_COL_CANDIDATES if c in df.columns), None)
    score_col = next((c for c in SCORE_COL_CANDIDATES if c in df.columns), None)
    if label_col is None or score_col is None:
        return None

    out = pd.DataFrame({
        "label": pd.to_numeric(df[label_col], errors="coerce"),
        "score": pd.to_numeric(df[score_col], errors="coerce"),
    }).dropna()

    if out.empty or out["label"].nunique() < 2:
        return None
    return out


def _discover_pairs(root: Path) -> List[Tuple[str, Path, Optional[Path]]]:
    """Walk the repo and return (system_name, test_csv, val_csv_or_None) tuples.

    Recognizes both the standard `test_predictions.csv` filename and the
    enrollment-style `enroll_test_predictions.csv`. Skips
    `role_only_*.csv` (duration-only baselines — kept implicitly via the
    enrollment files since they hold the embedding scores).
    """
    skip_dirs = {"node_modules", ".git", "__pycache__", ".cache", "wandb"}
    pairs: List[Tuple[str, Path, Optional[Path]]] = []

    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in skip_dirs]
        candidates = []
        for fname in filenames:
            if fname.endswith("test_predictions.csv") and "role_only" not in fname:
                candidates.append(fname)
        for fname in candidates:
            test = Path(dirpath) / fname
            val_name = fname.replace("test", "val")
            val_path = Path(dirpath) / val_name
            if not val_path.exists():
                val_path = None
            rel = test.relative_to(root)
            # Strip the trailing "test_predictions.csv"; keep the parent dir
            # plus any prefix on the filename ("enroll_") for disambiguation.
            sys_name = str(rel.parent)
            prefix = fname.replace("test_predictions.csv", "").rstrip("_")
            if prefix:
                sys_name = f"{sys_name}::{prefix}"
            pairs.append((sys_name, test, val_path))

    pairs.sort(key=lambda x: x[0])
    return pairs


# ---------------------------------------------------------------------------
# Threshold tuning
# ---------------------------------------------------------------------------

def _candidate_thresholds(scores: np.ndarray, n_max: int = 200) -> np.ndarray:
    """Sorted unique scores (subsampled if there are too many)."""
    uniq = np.unique(scores)
    if uniq.size > n_max:
        idx = np.linspace(0, uniq.size - 1, n_max).astype(int)
        uniq = uniq[idx]
    # Add 0 and 1 endpoints to cover degenerate cases.
    return np.unique(np.concatenate([[0.0, 1.0], uniq]))


def _tune_threshold(y: np.ndarray, scores: np.ndarray, objective: str) -> float:
    """Pick the threshold maximizing the objective on (y, scores)."""
    cands = _candidate_thresholds(scores)
    best_t, best_v = 0.5, -np.inf
    for t in cands:
        pred = (scores >= t).astype(int)
        if objective == "f1":
            v = f1_score(y, pred, zero_division=0)
        elif objective == "balanced_acc":
            v = balanced_accuracy_score(y, pred)
        elif objective == "youden":
            tp = ((pred == 1) & (y == 1)).sum()
            fn = ((pred == 0) & (y == 1)).sum()
            fp = ((pred == 1) & (y == 0)).sum()
            tn = ((pred == 0) & (y == 0)).sum()
            tpr = tp / max(tp + fn, 1)
            fpr = fp / max(fp + tn, 1)
            v = tpr - fpr
        else:
            raise ValueError(f"Unknown objective: {objective}")
        if v > best_v:
            best_v, best_t = v, float(t)
    return best_t


# ---------------------------------------------------------------------------
# Metric computation (single sample, given threshold)
# ---------------------------------------------------------------------------

def _metrics_at_threshold(y: np.ndarray, scores: np.ndarray,
                          threshold: float, prevalence: float) -> Dict[str, float]:
    pred = (scores >= threshold).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    bal_acc = 0.5 * (recall + specificity)
    accuracy = (tp + tn) / max(tp + fn + fp + tn, 1)
    try:
        mcc = matthews_corrcoef(y, pred)
    except Exception:
        mcc = float("nan")

    try:
        auroc = float(roc_auc_score(y, scores))
    except Exception:
        auroc = float("nan")
    try:
        auprc = float(average_precision_score(y, scores))
    except Exception:
        auprc = float("nan")

    # Trivial-baseline lift (predict all positive)
    trivial_f1 = 2 * prevalence / (1 + prevalence) if prevalence > 0 else 0.0
    trivial_bal_acc = 0.5  # always-positive: TPR=1, TNR=0 → balanced acc = 0.5
    trivial_mcc = 0.0
    trivial_auprc = prevalence  # random ranker AUPRC = prevalence

    return {
        "threshold": float(threshold),
        "n": int(len(y)),
        "n_pos": int((y == 1).sum()),
        "n_neg": int((y == 0).sum()),
        "prevalence": float(prevalence),
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "accuracy": float(accuracy),
        "f1": float(f1),
        "balanced_acc": float(bal_acc),
        "mcc": float(mcc),
        "auroc": auroc,
        "auprc": auprc,
        "auprc_lift": auprc - trivial_auprc if not np.isnan(auprc) else float("nan"),
        "delta_f1_vs_trivial": float(f1 - trivial_f1),
        "delta_bal_acc_vs_random": float(bal_acc - trivial_bal_acc),
        "delta_mcc_vs_trivial": float(mcc - trivial_mcc) if not np.isnan(mcc) else float("nan"),
    }


# ---------------------------------------------------------------------------
# Stratified bootstrap CIs
# ---------------------------------------------------------------------------

def _stratified_bootstrap_cis(
    y: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    n_bootstrap: int,
    seed: int,
    metric_keys: Tuple[str, ...] = ("f1", "balanced_acc", "mcc", "auroc", "auprc"),
) -> Dict[str, Tuple[float, float]]:
    """Return 95% percentile CIs for each metric. Stratified resampling
    samples positives and negatives independently with replacement,
    preserving the test-set prevalence.
    """
    rng = np.random.default_rng(seed)
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    if pos_idx.size == 0 or neg_idx.size == 0:
        return {k: (float("nan"), float("nan")) for k in metric_keys}

    samples: Dict[str, List[float]] = {k: [] for k in metric_keys}
    prev = float(y.mean())
    for _ in range(n_bootstrap):
        sp = rng.choice(pos_idx, size=pos_idx.size, replace=True)
        sn = rng.choice(neg_idx, size=neg_idx.size, replace=True)
        idx = np.concatenate([sp, sn])
        m = _metrics_at_threshold(y[idx], scores[idx], threshold, prev)
        for k in metric_keys:
            samples[k].append(m.get(k, float("nan")))

    out: Dict[str, Tuple[float, float]] = {}
    for k, vals in samples.items():
        arr = np.asarray(vals, dtype=float)
        arr = arr[~np.isnan(arr)]
        if arr.size == 0:
            out[k] = (float("nan"), float("nan"))
        else:
            out[k] = (float(np.percentile(arr, 2.5)),
                      float(np.percentile(arr, 97.5)))
    return out


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=".", help="Repo root to walk.")
    ap.add_argument("--out-csv", default="evaluation/recomputed_metrics.csv")
    ap.add_argument("--out-md",  default="evaluation/recomputed_metrics_summary.md")
    ap.add_argument("--n-bootstrap", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    pairs = _discover_pairs(root)
    print(f"Found {len(pairs)} candidate (test, val) prediction pairs under {root}")

    rows: List[Dict] = []
    for sys_name, test_csv, val_csv in pairs:
        test_df = _read_predictions(test_csv)
        if test_df is None:
            if args.verbose:
                print(f"  SKIP (unparseable test): {sys_name}")
            continue

        y_test = test_df["label"].astype(int).to_numpy()
        s_test = test_df["score"].astype(float).to_numpy()
        prev = float(y_test.mean())

        # Re-tune threshold on val if available; otherwise reuse the val-tuned
        # threshold by treating test as both (least defensible — flag it).
        val_df = _read_predictions(val_csv) if val_csv else None
        threshold_source = "val" if val_df is not None else "test_self"
        y_tune = val_df["label"].to_numpy() if val_df is not None else y_test
        s_tune = val_df["score"].to_numpy() if val_df is not None else s_test

        for objective in ("f1", "balanced_acc", "youden"):
            try:
                thr = _tune_threshold(y_tune, s_tune, objective)
            except Exception as e:
                print(f"  WARN ({sys_name}, {objective}): tune failed: {e}")
                continue

            metrics = _metrics_at_threshold(y_test, s_test, thr, prev)
            cis = _stratified_bootstrap_cis(
                y_test, s_test, thr,
                n_bootstrap=args.n_bootstrap,
                seed=args.seed,
            )
            row = {
                "system": sys_name,
                "objective": objective,
                "threshold_source": threshold_source,
                **metrics,
            }
            for k, (lo, hi) in cis.items():
                row[f"{k}_ci_low"] = lo
                row[f"{k}_ci_high"] = hi
            rows.append(row)

        if args.verbose:
            base = f"  {sys_name}: prev={prev:.3f}"
            print(base)

    if not rows:
        print("No parseable predictions found.")
        return 1

    out_df = pd.DataFrame(rows)
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out_csv, index=False)
    print(f"\nWrote {len(out_df)} rows → {args.out_csv}")

    # ---- markdown summary (Youden-J ranking) ----
    write_summary_md(out_df, Path(args.out_md))
    print(f"Wrote summary → {args.out_md}")
    return 0


def write_summary_md(df: pd.DataFrame, path: Path) -> None:
    """One readable table per (split, objective). Sorted by balanced acc."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: List[str] = []
    lines.append("# Recomputed metrics (imbalance-aware)")
    lines.append("")
    lines.append("Generated by `evaluation/recompute_metrics.py`. ")
    lines.append("Thresholds re-tuned on val under each objective; metrics + CIs computed on test. ")
    lines.append("CIs are 95% percentile from 1000 stratified bootstraps "
                 "(positives and negatives resampled independently to preserve prevalence).")
    lines.append("")
    lines.append("**Trivial-baseline reference** (predict-all-positive on a test set with prevalence p):")
    lines.append("- F1 = 2·p / (1+p)")
    lines.append("- Balanced acc = 0.5")
    lines.append("- MCC = 0")
    lines.append("- Random AUPRC = p (so `auprc_lift` = AUPRC − p; only positive lift means real ranking signal)")
    lines.append("")

    for objective, label in (("youden", "Youden's J"),
                             ("balanced_acc", "Balanced accuracy"),
                             ("f1", "F1 (legacy)")):
        sub = df[df["objective"] == objective].copy()
        if sub.empty:
            continue
        sub = sub.sort_values("balanced_acc", ascending=False)
        lines.append(f"## Threshold tuned for {label}")
        lines.append("")
        lines.append("| System | Prev | Thr | F1 | ΔF1 vs trivial | Bal. Acc | ΔBalAcc | MCC | "
                     "AUROC | AUPRC | AUPRC lift | F1 95% CI | BalAcc 95% CI |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|")
        for _, r in sub.iterrows():
            lines.append(
                f"| `{r['system']}` "
                f"| {r['prevalence']:.3f} "
                f"| {r['threshold']:.3f} "
                f"| {r['f1']:.3f} "
                f"| {r['delta_f1_vs_trivial']:+.3f} "
                f"| {r['balanced_acc']:.3f} "
                f"| {r['delta_bal_acc_vs_random']:+.3f} "
                f"| {r['mcc']:+.3f} "
                f"| {r['auroc']:.3f} "
                f"| {r['auprc']:.3f} "
                f"| {r['auprc_lift']:+.3f} "
                f"| [{r.get('f1_ci_low', float('nan')):.3f}, {r.get('f1_ci_high', float('nan')):.3f}] "
                f"| [{r.get('balanced_acc_ci_low', float('nan')):.3f}, "
                f"{r.get('balanced_acc_ci_high', float('nan')):.3f}] |"
            )
        lines.append("")

    path.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    sys.exit(main())
