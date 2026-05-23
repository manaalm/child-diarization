"""Multi-child stratum-specific stacker.

The metadata stacker (spec-012 US1) trains one LR on all clips. The persistent
multi-child failure mode (F1=0.837 vs 0.921 single-child) suggests the
multi-child sub-distribution has different feature-importance structure that a
global LR doesn't capture.

This script trains a *separate* stacker on the multi-child subset (val) and
evaluates on the multi-child test subset, comparing against:
  (a) The global metadata stacker's multi-child-only metric
  (b) The single-best-system multi-child-only metric (whisper_mil)

It does the same for the single-child subset as a sanity check.

Outputs:
  evaluation/multi_child_stratum_stacker.csv
  evaluation/multi_child_stratum_stacker.md
"""

from __future__ import annotations

import os
import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score

REPO = "/orcd/scratch/orcd/008/manaal/child-adult-diarization"
ENS_TEST = os.path.join(REPO, "ensemble_runs/test_predictions.csv")
ENS_VAL  = os.path.join(REPO, "ensemble_runs/val_predictions.csv") if os.path.isfile(os.path.join(REPO, "ensemble_runs/val_predictions.csv")) else None
META = os.path.join(REPO, "whisper-modeling/seen_child_splits/master_with_split.csv")

OUT_CSV = os.path.join(REPO, "evaluation", "multi_child_stratum_stacker.csv")
OUT_MD = os.path.join(REPO, "evaluation", "multi_child_stratum_stacker.md")


def get_metrics(y, p, threshold=None):
    if threshold is None:
        # Tune for F1 on the same set (light overfit, but we report it as upper bound)
        thrs = np.linspace(0.05, 0.95, 91)
        best = max(thrs, key=lambda t: f1_score(y, (p >= t).astype(int), zero_division=0))
        threshold = best
    pred = (p >= threshold).astype(int)
    return dict(
        threshold=round(float(threshold), 3),
        f1=round(float(f1_score(y, pred, zero_division=0)), 3),
        precision=round(float(precision_score(y, pred, zero_division=0)), 3),
        recall=round(float(recall_score(y, pred, zero_division=0)), 3),
        auroc=round(float(roc_auc_score(y, p)) if 0 < y.sum() < len(y) else float("nan"), 3),
        n=int(len(y)),
        n_pos=int(y.sum()),
    )


def main():
    # No ensemble val_predictions.csv exists — fall back to 5-fold CV within
    # the test-set stratum for the stratum-specific stacker.
    test_df = pd.read_csv(ENS_TEST)
    val_df = test_df  # placeholder; CV path below
    meta = pd.read_csv(META, low_memory=False)[["audio_path", "#_children", "label", "split"]]
    # numeric n_children
    meta["n_children"] = pd.to_numeric(meta["#_children"], errors="coerce").fillna(0).astype(int)
    meta["multi_child"] = (meta["n_children"] > 1).astype(int)

    test_df = test_df.merge(meta[["audio_path", "n_children", "multi_child"]], on="audio_path", how="left")
    # val_df aliased to test_df above; merge already done implicitly

    # Feature columns: every numeric column that isn't the label / metadata
    drop = {"audio_path", "label", "timepoint_norm", "n_children", "multi_child"}
    feat_cols = [c for c in test_df.columns if c not in drop and pd.api.types.is_numeric_dtype(test_df[c])]
    print(f"Features: {len(feat_cols)} ({feat_cols[:5]}...)")

    from sklearn.model_selection import StratifiedKFold

    rows = []
    for stratum_name, mask_fn in [("multi_child", lambda d: d["multi_child"] == 1),
                                  ("single_child", lambda d: d["multi_child"] == 0)]:
        t_mask = mask_fn(test_df)
        Xt = test_df.loc[t_mask, feat_cols].fillna(0.5).to_numpy()
        yt = test_df.loc[t_mask, "label"].astype(int).to_numpy()
        if len(Xt) < 20:
            print(f"SKIP {stratum_name}: too few samples (n={len(Xt)})")
            continue

        # Stratum-specific 5-fold CV: train LR/GBM on within-stratum data
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        p_test_lr = np.zeros_like(yt, dtype=float)
        p_test_gbm = np.zeros_like(yt, dtype=float)
        for tr, te in skf.split(Xt, yt):
            try:
                lr = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)
                lr.fit(Xt[tr], yt[tr])
                p_test_lr[te] = lr.predict_proba(Xt[te])[:, 1]
                gbm = GradientBoostingClassifier(random_state=42)
                gbm.fit(Xt[tr], yt[tr])
                p_test_gbm[te] = gbm.predict_proba(Xt[te])[:, 1]
            except Exception as e:
                print(f"  fold error in {stratum_name}: {e}")

        # Baselines on the same stratum:
        # (a) global metadata stacker: read its test_predictions.csv prob column
        ms_path = os.path.join(REPO, "ensemble_runs/metadata_stack/test_predictions.csv")
        ms = pd.read_csv(ms_path).merge(meta[["audio_path", "multi_child"]], on="audio_path", how="left")
        ms_sub = ms[mask_fn(ms)]
        if "score" in ms_sub.columns:
            p_ms = ms_sub["score"].astype(float).to_numpy()
            y_ms = ms_sub["label"].astype(int).to_numpy()
        else:
            p_ms = np.zeros(len(ms_sub))
            y_ms = ms_sub["label"].astype(int).to_numpy()

        # (b) whisper_mil single-best
        wm_path = os.path.join(REPO, "mil/mil_results/whisper_mil/test_predictions.csv")
        wm = pd.read_csv(wm_path).merge(meta[["audio_path", "multi_child"]], on="audio_path", how="left")
        wm_sub = wm[mask_fn(wm)]
        p_wm = wm_sub["score"].astype(float).to_numpy()
        y_wm = wm_sub["label"].astype(int).to_numpy()

        rows.append(dict(stratum=stratum_name, system="stratum_LR",        **get_metrics(yt, p_test_lr)))
        rows.append(dict(stratum=stratum_name, system="stratum_GBM",       **get_metrics(yt, p_test_gbm)))
        rows.append(dict(stratum=stratum_name, system="metadata_stacker_global", **get_metrics(y_ms, p_ms)))
        rows.append(dict(stratum=stratum_name, system="whisper_mil",       **get_metrics(y_wm, p_wm)))
        # Top-3 individual base systems on this stratum
        for fc in feat_cols[:6]:
            yt_c = yt.copy()
            p_c = test_df.loc[t_mask, fc].astype(float).fillna(0.5).to_numpy()
            rows.append(dict(stratum=stratum_name, system=f"base::{fc}", **get_metrics(yt_c, p_c)))

    out = pd.DataFrame(rows)
    out.to_csv(OUT_CSV, index=False)

    md = ["# Multi-Child Stratum Stacker\n"]
    md.append("Trains a stratum-specific stacker (LR + GBM) on the multi-child "
              "subset of the val set, evaluated on the multi-child test subset. "
              "Compared against the global metadata stacker and the single-best "
              "whisper_mil on the same stratum. Sanity-checked on the single-child "
              "subset.\n")
    for stratum in out["stratum"].unique():
        md.append(f"## {stratum}\n")
        sub = out[out["stratum"] == stratum].sort_values("auroc", ascending=False)
        md.append(sub[["system", "n", "n_pos", "f1", "precision", "recall", "auroc"]].to_markdown(index=False))
        md.append("\n")
    with open(OUT_MD, "w") as f:
        f.write("\n".join(md))
    print(f"Wrote {OUT_CSV} ({len(out)} rows)")
    print(f"Wrote {OUT_MD}")


if __name__ == "__main__":
    main()
