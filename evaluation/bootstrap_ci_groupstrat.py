"""Bootstrap 95% CIs for group-stratified 3-fold catalog systems.

For each system that has all 3 group-strat fold test_predictions.csv:
  1. Concatenate per-fold test predictions into one pooled set (n ~= 3145
     across all 3 folds since group-strat partitions the 130 children into
     3 disjoint groups).
  2. Take the BA-tuned threshold (from each fold's val_metrics_tuned.json
     or fall back to median across folds) — for catalog consistency we
     use the median val-tuned threshold across the 3 folds.
  3. Cluster-bootstrap resample by child_id (B=2000), recompute F1 / BA /
     AUROC / AUPRC, report mean ± 95% CI.

Cluster bootstrap (not row bootstrap) because child is the natural unit of
non-exchangeability: clips from the same child are dependent.

Output:
  evaluation/bootstrap_ci_groupstrat3.csv
  evaluation/bootstrap_ci_groupstrat3.md (pretty-printed table)
"""
from __future__ import annotations

import json
import os
import sys
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)

_REPO = Path("/orcd/scratch/orcd/008/manaal/child-adult-diarization")
B = 2000  # bootstrap resamples
SEED = 42


def _collect_fold_predictions(system_dir_glob: str) -> tuple[pd.DataFrame, float]:
    """Return (pooled_predictions_df, median_val_tuned_threshold)."""
    dirs = sorted(glob(str(_REPO / system_dir_glob)))
    if len(dirs) < 3:
        return None, None
    preds = []
    thresholds = []
    for d in dirs:
        pred_path = os.path.join(d, "test_predictions.csv")
        if not os.path.exists(pred_path):
            pred_path = os.path.join(d, "enroll_test_predictions.csv")
        if not os.path.exists(pred_path):
            continue
        df = pd.read_csv(pred_path)
        if "score" not in df.columns:
            for c in ("prob", "y_prob"):
                if c in df.columns:
                    df = df.rename(columns={c: "score"})
                    break
        if "score" not in df.columns or "label" not in df.columns or "child_id" not in df.columns:
            continue
        preds.append(df)
        val_path = os.path.join(d, "val_metrics_tuned.json")
        if not os.path.exists(val_path):
            val_path = os.path.join(d, "enroll_val_metrics.json")
        if os.path.exists(val_path):
            with open(val_path) as f:
                v = json.load(f)
            t = v.get("threshold", 0.5)
            if isinstance(t, (int, float)):
                thresholds.append(float(t))
    if not preds:
        return None, None
    pooled = pd.concat(preds, ignore_index=True)
    thr = float(np.median(thresholds)) if thresholds else 0.5
    return pooled, thr


def _metrics(y_true, y_prob, threshold):
    y_pred = (y_prob >= threshold).astype(int)
    try:
        auroc = float(roc_auc_score(y_true, y_prob))
    except Exception:
        auroc = float("nan")
    try:
        auprc = float(average_precision_score(y_true, y_prob))
    except Exception:
        auprc = float("nan")
    return {
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "auroc": auroc,
        "auprc": auprc,
    }


def _cluster_bootstrap(pooled: pd.DataFrame, threshold: float, n_resamples: int = B, seed: int = SEED):
    """Vectorised cluster bootstrap.

    Precompute per-child (label, score) numpy arrays once, then for each
    resample concatenate the chosen children's arrays via np.concatenate
    (much faster than pd.concat in a 2000-iteration loop).
    """
    rng = np.random.default_rng(seed)
    children = pooled["child_id"].astype(str).unique()
    n_children = len(children)
    child_to_idx = {c: i for i, c in enumerate(children)}

    # Precompute per-child label / score arrays (lists indexed by child idx).
    labels_by_child = [None] * n_children
    scores_by_child = [None] * n_children
    groups = pooled.groupby(pooled["child_id"].astype(str))
    for c, g in groups:
        i = child_to_idx[c]
        labels_by_child[i] = g["label"].astype(int).to_numpy()
        scores_by_child[i] = g["score"].astype(float).to_numpy()

    rows = []
    for _ in range(n_resamples):
        sampled_idx = rng.integers(0, n_children, size=n_children)
        labels = np.concatenate([labels_by_child[i] for i in sampled_idx])
        scores = np.concatenate([scores_by_child[i] for i in sampled_idx])
        if labels.sum() == 0 or labels.sum() == len(labels):
            continue
        rows.append(_metrics(labels, scores, threshold))
    return pd.DataFrame(rows)


def _summarize(boot_df: pd.DataFrame, point_metrics: dict) -> dict:
    out = {}
    for metric in ("f1", "balanced_accuracy", "auroc", "auprc"):
        col = boot_df[metric].dropna()
        if len(col) == 0:
            out[f"{metric}_point"] = point_metrics.get(metric, float("nan"))
            out[f"{metric}_ci_lo"] = float("nan")
            out[f"{metric}_ci_hi"] = float("nan")
            continue
        out[f"{metric}_point"] = point_metrics[metric]
        out[f"{metric}_ci_lo"] = float(np.percentile(col, 2.5))
        out[f"{metric}_ci_hi"] = float(np.percentile(col, 97.5))
    return out


# Catalog: name → glob pattern for the 3 fold dirs
CATALOG = {
    # MIL family
    "whisper_medium_mil": "mil/mil_results/whisper_medium_mil_groupstrat3_f*",
    "whisper_mil": "mil/mil_results/whisper_mil_groupstrat3_f*",
    "whisper_mil_acmil_max": "mil/mil_results/whisper_mil_acmil_max_groupstrat3_f*",
    "whisper_mil_tsmil_concat": "mil/mil_results/whisper_mil_tsmil_concat_groupstrat3_f*",
    "wavlm_mil": "mil/mil_results/wavlm_mil_groupstrat3_f*",
    # Pseudo-frame
    "whisper_pseudo_frame": "pseudo_frame/results/whisper_pseudo_frame_groupstrat3_f*",
    "wavlm_pseudo_frame": "pseudo_frame/results/wavlm_pseudo_frame_groupstrat3_f*",
    # Enrollment diarizers: excluded from group-strat cross-child analysis.
    # The ECAPA enrollment paradigm requires per-child prototypes from train data,
    # which group-strat (child-disjoint train/test) makes impossible by construction.
    # All test scores collapse to 0.0 because no test child has a prototype.
    # Their canonical evaluation remains the single-split + within-child k-fold;
    # cross-child generalisation is fundamentally not applicable.
    # "babar":         "babar_ecapa_enrollment_runs_groupstrat3_f*",  # excluded
    # "vtc_kchi":      "vtc_kchi_ecapa_enrollment_runs_groupstrat3_f*",  # excluded
    # "vtc":           "vtc_ecapa_enrollment_runs_groupstrat3_f*",  # excluded
    # "usc_sail":      "usc_sail_ecapa_enrollment_runs_groupstrat3_f*",  # excluded
    # "vbx":           "vbx_ecapa_enrollment_runs_groupstrat3_f*",  # excluded
    # "pyannote":      "pyannote_ecapa_enrollment_runs_groupstrat3_f*",  # excluded
    # "sortformer":    "sortformer_ecapa_enrollment_runs_groupstrat3_f*",  # excluded
    # "eend_eda":      "eend_eda_ecapa_enrollment_runs_groupstrat3_f*",  # excluded
    # "joint_asr_diar":"joint_asr_diar_ecapa_enrollment_runs_groupstrat3_f*",  # excluded
    # Zero-shot
    "yamnet": "baselines/scene_analysis_runs/yamnet_groupstrat3_f*",
    "ast": "baselines/scene_analysis_runs/ast_groupstrat3_f*",
    "qwen2_audio_7b": "baselines/audio_llm_baseline_runs/qwen2_audio_7b_groupstrat3_f*",
    "qwen25_omni_7b": "baselines/audio_llm_baseline_runs/qwen25_omni_7b_groupstrat3_f*",
    "qwen3_omni_30b_thinking": "baselines/audio_llm_baseline_runs/qwen3_omni_30b_thinking_groupstrat3_f*",
}


def main():
    rows = []
    for system, pattern in CATALOG.items():
        pooled, thr = _collect_fold_predictions(pattern)
        if pooled is None:
            print(f"  SKIP {system}: insufficient fold predictions")
            continue
        if thr is None or np.isnan(thr):
            thr = 0.5
        point = _metrics(
            pooled["label"].astype(int).to_numpy(),
            pooled["score"].astype(float).to_numpy(),
            thr,
        )
        boot = _cluster_bootstrap(pooled, thr, n_resamples=B, seed=SEED)
        summary = _summarize(boot, point)
        n_clips = len(pooled)
        n_children = pooled["child_id"].nunique()
        print(f"  {system:>30}  thr={thr:.2f}  n={n_clips}  kids={n_children}  "
              f"BA={summary['balanced_accuracy_point']:.3f} [{summary['balanced_accuracy_ci_lo']:.3f}, "
              f"{summary['balanced_accuracy_ci_hi']:.3f}]  AUROC={summary['auroc_point']:.3f} "
              f"[{summary['auroc_ci_lo']:.3f}, {summary['auroc_ci_hi']:.3f}]")
        rows.append({
            "system": system,
            "n_clips": n_clips,
            "n_children": n_children,
            "threshold": thr,
            **summary,
        })

    out_csv = _REPO / "evaluation" / "bootstrap_ci_groupstrat3.csv"
    out_md = _REPO / "evaluation" / "bootstrap_ci_groupstrat3.md"
    df = pd.DataFrame(rows).sort_values("balanced_accuracy_point", ascending=False)
    df.to_csv(out_csv, index=False)

    with open(out_md, "w") as f:
        f.write("# Bootstrap CIs on group-stratified 3-fold predictions\n\n")
        f.write(f"Cluster bootstrap by child_id, B={B} resamples, seed={SEED}.\n")
        f.write("Pooled predictions across 3 group-stratified folds (children disjoint per fold).\n\n")
        f.write("| System | n | kids | thr | F1 [95% CI] | BA [95% CI] | AUROC [95% CI] | AUPRC [95% CI] |\n")
        f.write("|---|---|---|---|---|---|---|---|\n")
        for _, r in df.iterrows():
            f.write(
                f"| {r['system']} | {r['n_clips']} | {r['n_children']} | {r['threshold']:.2f} | "
                f"{r['f1_point']:.3f} [{r['f1_ci_lo']:.3f}, {r['f1_ci_hi']:.3f}] | "
                f"{r['balanced_accuracy_point']:.3f} [{r['balanced_accuracy_ci_lo']:.3f}, {r['balanced_accuracy_ci_hi']:.3f}] | "
                f"{r['auroc_point']:.3f} [{r['auroc_ci_lo']:.3f}, {r['auroc_ci_hi']:.3f}] | "
                f"{r['auprc_point']:.3f} [{r['auprc_ci_lo']:.3f}, {r['auprc_ci_hi']:.3f}] |\n"
            )

    print(f"\nWrote: {out_csv}")
    print(f"Wrote: {out_md}")
    print(f"({len(df)} systems audited)")


if __name__ == "__main__":
    sys.exit(main() or 0)
