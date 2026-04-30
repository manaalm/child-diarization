"""Cluster-bootstrap 95% CIs for the metadata stacker (and key comparators).

The seen-child split shares 179/180 children across val/test, so per-clip
bootstrap would underestimate variance (clips of the same child are not
independent). Cluster-bootstrap by child_id: resample child_ids with
replacement, then take all of each resampled child's test clips. Repeat
B=2000 times.

Reports point estimate + 2.5/97.5 percentile CI for:
  - AUROC
  - AUPRC
  - F1 (val-tuned threshold from each system's config)
  - AUROC stratified by timepoint_norm (14_month vs 36_month)

Usage:
  python evaluation/bootstrap_metadata_stacker.py
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    roc_auc_score,
)

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
B = 2000
SEED = 42
CI_LO, CI_HI = 2.5, 97.5

MASTER = os.path.join(_REPO, "whisper-modeling/seen_child_splits/master_with_split.csv")


def load_master():
    df = pd.read_csv(MASTER)
    df = df[df["split"] == "test"][["audio_path", "child_id", "timepoint_norm", "label"]]
    return df.reset_index(drop=True)


def load_system_scores():
    """Return dict {system: DataFrame[audio_path, score, threshold]}."""
    systems = {}

    # Metadata stacker (the headline)
    df = pd.read_csv(os.path.join(_REPO, "ensemble_runs/metadata_stack/test_predictions.csv"))
    cfg = json.load(open(os.path.join(_REPO, "ensemble_runs/metadata_stack/test_metrics_tuned.json")))
    systems["metadata_stacker"] = {"df": df.rename(columns={"score": "score"})[["audio_path", "score"]],
                                   "thr": cfg["threshold"]}

    # Baseline: best_audio_mil_mean ensemble (column in ensemble_runs/test_predictions.csv).
    # No val_predictions.csv exists for this baseline; threshold=0.5 is the natural choice
    # for an averaged probability ensemble. AUROC/AUPRC are threshold-free, F1 uses 0.5.
    df = pd.read_csv(os.path.join(_REPO, "ensemble_runs/test_predictions.csv"))
    df_b = df[["audio_path", "best_audio_mil_mean"]].rename(columns={"best_audio_mil_mean": "score"})
    systems["best_audio_mil_mean (baseline, thr=0.5)"] = {"df": df_b, "thr": 0.5}

    # Best single audio system: Whisper-MIL
    df = pd.read_csv(os.path.join(_REPO, "mil/mil_results/whisper_mil/test_predictions.csv"))
    cfg = json.load(open(os.path.join(_REPO, "mil/mil_results/whisper_mil/val_metrics_tuned.json")))
    systems["whisper_mil"] = {"df": df[["audio_path", "score"]], "thr": cfg["threshold"]}

    # Best non-MIL audio diarizer: BabAR enrollment
    df = pd.read_csv(os.path.join(_REPO, "babar_ecapa_enrollment_runs/enroll_test_predictions.csv"))
    cfg = json.load(open(os.path.join(_REPO, "babar_ecapa_enrollment_runs/enroll_val_metrics.json")))
    systems["babar"] = {"df": df[["audio_path", "prob"]].rename(columns={"prob": "score"}),
                        "thr": cfg["threshold"]}

    return systems


def _safe_metric(y, p, fn, **kw):
    """Compute metric, returning NaN if degenerate."""
    if len(y) < 2 or y.sum() == 0 or y.sum() == len(y):
        return float("nan")
    try:
        return float(fn(y, p, **kw))
    except Exception:
        return float("nan")


def metrics(y, p, thr):
    pred = (p >= thr).astype(int)
    return {
        "auroc": _safe_metric(y, p, roc_auc_score),
        "auprc": _safe_metric(y, p, average_precision_score),
        "f1":    float(f1_score(y, pred, zero_division=0)) if len(y) else float("nan"),
        "n":     int(len(y)),
    }


def cluster_bootstrap(joined: pd.DataFrame, score_col: str, thr: float,
                      B: int, seed: int, age: str | None = None):
    """Resample child_ids with replacement; compute metrics per resample.

    Returns dict {metric: (point, lo, hi)}.
    """
    rng = np.random.default_rng(seed)

    # Optional age filter (apply to both point estimate and bootstrap samples)
    df = joined if age is None else joined[joined["timepoint_norm"] == age]
    df = df.reset_index(drop=True)

    children = df["child_id"].unique()
    # Pre-bucket: child_id → row indices
    by_child = {c: df.index[df["child_id"] == c].to_numpy() for c in children}

    # Point estimate (no resampling)
    point = metrics(df["label"].to_numpy(dtype=int),
                    df[score_col].to_numpy(dtype=float),
                    thr)

    # Bootstrap
    aurocs, auprcs, f1s = [], [], []
    n_kids = len(children)
    for b in range(B):
        sampled = rng.choice(children, size=n_kids, replace=True)
        idx = np.concatenate([by_child[c] for c in sampled])
        y = df.loc[idx, "label"].to_numpy(dtype=int)
        p = df.loc[idx, score_col].to_numpy(dtype=float)
        m = metrics(y, p, thr)
        aurocs.append(m["auroc"])
        auprcs.append(m["auprc"])
        f1s.append(m["f1"])

    def ci(arr):
        a = np.array(arr, dtype=float)
        a = a[~np.isnan(a)]
        if len(a) == 0:
            return float("nan"), float("nan")
        return float(np.percentile(a, CI_LO)), float(np.percentile(a, CI_HI))

    auroc_lo, auroc_hi = ci(aurocs)
    auprc_lo, auprc_hi = ci(auprcs)
    f1_lo, f1_hi       = ci(f1s)

    return {
        "auroc": (point["auroc"], auroc_lo, auroc_hi),
        "auprc": (point["auprc"], auprc_lo, auprc_hi),
        "f1":    (point["f1"],    f1_lo,    f1_hi),
        "n_clips": point["n"],
        "n_children": int(n_kids),
        "threshold": thr,
    }


def fmt(v, lo, hi, ndigits=4):
    if np.isnan(v):
        return f"{'n/a':>23}"
    return f"{v:.{ndigits}f}  [{lo:.{ndigits}f}, {hi:.{ndigits}f}]"


def main():
    print(f"=== Cluster-bootstrap (resample by child_id, B={B}, seed={SEED}, 95% CI) ===\n", flush=True)

    master = load_master()
    print(f"test set: {len(master)} clips, {master['child_id'].nunique()} children, "
          f"timepoints={master['timepoint_norm'].value_counts().to_dict()}\n", flush=True)

    systems = load_system_scores()

    rows = []
    for name, info in systems.items():
        df = master.merge(info["df"], on="audio_path", how="left")
        if df["score"].isna().any():
            n_missing = df["score"].isna().sum()
            print(f"  WARN {name}: {n_missing} clips missing scores; dropping", flush=True)
            df = df.dropna(subset=["score"])

        thr = info["thr"]
        all_m = cluster_bootstrap(df, "score", thr, B=B, seed=SEED)
        m_14  = cluster_bootstrap(df, "score", thr, B=B, seed=SEED, age="14_month")
        m_36  = cluster_bootstrap(df, "score", thr, B=B, seed=SEED, age="36_month")

        print(f"--- {name}  (threshold={thr:.3f},  n_clips={all_m['n_clips']},  n_children={all_m['n_children']}) ---", flush=True)
        print(f"  AUROC               {fmt(*all_m['auroc'])}", flush=True)
        print(f"  AUPRC               {fmt(*all_m['auprc'])}", flush=True)
        print(f"  F1                  {fmt(*all_m['f1'])}", flush=True)
        print(f"  AUROC (14_month, n={m_14['n_clips']}/{m_14['n_children']}c)  {fmt(*m_14['auroc'])}", flush=True)
        print(f"  AUROC (36_month, n={m_36['n_clips']}/{m_36['n_children']}c)  {fmt(*m_36['auroc'])}", flush=True)
        print("", flush=True)

        rows.append({
            "system": name,
            "threshold": thr,
            "n_clips": all_m["n_clips"],
            "n_children": all_m["n_children"],
            "auroc": all_m["auroc"][0], "auroc_lo": all_m["auroc"][1], "auroc_hi": all_m["auroc"][2],
            "auprc": all_m["auprc"][0], "auprc_lo": all_m["auprc"][1], "auprc_hi": all_m["auprc"][2],
            "f1":    all_m["f1"][0],    "f1_lo":    all_m["f1"][1],    "f1_hi":    all_m["f1"][2],
            "auroc_14m": m_14["auroc"][0], "auroc_14m_lo": m_14["auroc"][1], "auroc_14m_hi": m_14["auroc"][2],
            "auroc_36m": m_36["auroc"][0], "auroc_36m_lo": m_36["auroc"][1], "auroc_36m_hi": m_36["auroc"][2],
        })

    out_path = os.path.join(_REPO, "ensemble_runs/metadata_stack/bootstrap_ci.json")
    out = {
        "config": {"B": B, "seed": SEED, "ci_low": CI_LO, "ci_high": CI_HI,
                   "method": "cluster bootstrap by child_id"},
        "results": rows,
    }
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote → {out_path}", flush=True)


if __name__ == "__main__":
    main()
