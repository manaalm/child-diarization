"""Top-k LR-stacker ensembles over the headline-table systems.

For every system that appears in Tab:headline of the thesis (within-speaker,
BIDS-corrected, n=635), load val + test predictions, rank by validation AUROC,
and fit a logistic-regression stacker on the top-k systems for several k.

Fit on val only (BA-threshold tune on val), evaluate once on test. Reports
F1 / balanced accuracy / AUROC / AUPRC and writes a CSV to
evaluation/topk_headline_ensembles.csv.

Usage: python evaluation/topk_headline_ensembles.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)

REPO = Path(__file__).resolve().parent.parent
MASTER_CSV = REPO / "whisper-modeling/seen_child_splits/master_with_split.csv"
OUT_CSV = REPO / "evaluation/topk_headline_ensembles.csv"
OUT_LADDER = REPO / "evaluation/topk_headline_ensembles_ladder.csv"
SEED = 42

# (display_name, val_csv, test_csv, score_col)
# Paths relative to REPO. Predictions all keyed on `audio_path` (master CSV key)
# except av_fusion rows which use `clip_id` (master row index, `Unnamed: 0`).
SYSTEMS = [
    # Diarization + enrollment
    ("USC-SAIL",          "whisper-modeling/usc_sail_enrollment_runs/enroll_val_predictions.csv",
                          "whisper-modeling/usc_sail_enrollment_runs/enroll_test_predictions.csv",  "prob",      "audio_path"),
    ("Pyannote",          "pyannote/pyannote_enrollment_runs/val_predictions.csv",
                          "pyannote/pyannote_enrollment_runs/test_predictions.csv",                  "prob",      "audio_path"),
    ("BabAR",             "babar_ecapa_enrollment_runs/enroll_val_predictions.csv",
                          "babar_ecapa_enrollment_runs/enroll_test_predictions.csv",                 "prob",      "audio_path"),
    ("VTC-KCHI+OCH",      "vtc_ecapa_enrollment_runs/enroll_val_predictions.csv",
                          "vtc_ecapa_enrollment_runs/enroll_test_predictions.csv",                   "prob",      "audio_path"),
    ("VTC-KCHI",          "vtc_kchi_ecapa_enrollment_runs/enroll_val_predictions.csv",
                          "vtc_kchi_ecapa_enrollment_runs/enroll_test_predictions.csv",              "prob",      "audio_path"),
    ("VBx",               "vbx_ecapa_enrollment_runs/enroll_val_predictions.csv",
                          "vbx_ecapa_enrollment_runs/enroll_test_predictions.csv",                   "prob",      "audio_path"),
    ("EEND-EDA",          "eend_eda_ecapa_enrollment_runs/enroll_val_predictions.csv",
                          "eend_eda_ecapa_enrollment_runs/enroll_test_predictions.csv",              "prob",      "audio_path"),
    ("Sortformer",        "sortformer_ecapa_enrollment_runs/enroll_val_predictions.csv",
                          "sortformer_ecapa_enrollment_runs/enroll_test_predictions.csv",            "prob",      "audio_path"),
    ("Joint-ASR-diar",    "joint_asr_diar_ecapa_enrollment_runs/enroll_val_predictions.csv",
                          "joint_asr_diar_ecapa_enrollment_runs/enroll_test_predictions.csv",        "prob",      "audio_path"),
    # Direct supervised encoders
    ("Fused-small-frozen","baselines/baseline_results/fused_attn/val_predictions.csv",
                          "baselines/baseline_results/fused_attn/test_predictions.csv",              "prob",      "audio_path"),
    ("Fused-small-PU2",   "baselines/baseline_results/fused_attn_unfreeze2/val_predictions.csv",
                          "baselines/baseline_results/fused_attn_unfreeze2/test_predictions.csv",    "prob",      "audio_path"),
    ("whisper_attn",      "baselines/baseline_results/whisper_attn/val_predictions.csv",
                          "baselines/baseline_results/whisper_attn/test_predictions.csv",            "prob",      "audio_path"),
    ("whisper_attn_lw",   "baselines/baseline_results/whisper_attn_lw/val_predictions.csv",
                          "baselines/baseline_results/whisper_attn_lw/test_predictions.csv",         "prob",      "audio_path"),
    ("wavlm_attn",        "baselines/baseline_results/wavlm_attn/val_predictions.csv",
                          "baselines/baseline_results/wavlm_attn/test_predictions.csv",              "prob",      "audio_path"),
    ("wavlm_stats_lw",    "baselines/baseline_results/wavlm_stats_lw/val_predictions.csv",
                          "baselines/baseline_results/wavlm_stats_lw/test_predictions.csv",          "prob",      "audio_path"),
    # MIL
    ("WavLM-MIL",         "mil/mil_results/wavlm_mil/val_predictions.csv",
                          "mil/mil_results/wavlm_mil/test_predictions.csv",                          "score",     "audio_path"),
    ("Whisper-MIL",       "mil/mil_results/whisper_mil/val_predictions.csv",
                          "mil/mil_results/whisper_mil/test_predictions.csv",                        "score",     "audio_path"),
    ("Whisper-medium-MIL","mil/mil_results/whisper_medium_mil/val_predictions.csv",
                          "mil/mil_results/whisper_medium_mil/test_predictions.csv",                 "score",     "audio_path"),
    ("Whisper-MIL-TS",    "mil/mil_results/whisper_mil_tsmil_concat/val_predictions.csv",
                          "mil/mil_results/whisper_mil_tsmil_concat/test_predictions.csv",           "score",     "audio_path"),
    ("Whisper-MIL-ACMIL", "mil/mil_results/whisper_mil_acmil_max/val_predictions.csv",
                          "mil/mil_results/whisper_mil_acmil_max/test_predictions.csv",              "score",     "audio_path"),
    # Pseudo-frame (smaller n=438 subset; will inner-join shrink the universe)
    ("WavLM-pseudo-frame","pseudo_frame/results/wavlm_pseudo_frame/val_predictions.csv",
                          "pseudo_frame/results/wavlm_pseudo_frame/test_predictions.csv",            "score",     "audio_path"),
    ("Whisper-pseudo-frame","pseudo_frame/results/whisper_pseudo_frame/val_predictions.csv",
                          "pseudo_frame/results/whisper_pseudo_frame/test_predictions.csv",          "score",     "audio_path"),
    # Audio LLM
    ("Qwen2.5-Omni-7B",   "baselines/audio_llm_baseline_runs/qwen25_omni_7b/val_predictions.csv",
                          "baselines/audio_llm_baseline_runs/qwen25_omni_7b/test_predictions.csv",   "prob",      "audio_path"),
    # AV fusion (manual features); always-fuse + gated + audio-only are columns of one CSV
    ("AV-audio-only",     "av_fusion/av_results/manual_only/predictions_val.csv",
                          "av_fusion/av_results/manual_only/predictions_test.csv",                   "proba_audio_only",  "clip_id"),
    ("AV-always-fuse",    "av_fusion/av_results/manual_only/predictions_val.csv",
                          "av_fusion/av_results/manual_only/predictions_test.csv",                   "proba_always_fuse", "clip_id"),
    ("AV-gated",          "av_fusion/av_results/manual_only/predictions_val.csv",
                          "av_fusion/av_results/manual_only/predictions_test.csv",                   "proba_gated_av",    "clip_id"),
]


def load_master() -> pd.DataFrame:
    df = pd.read_csv(MASTER_CSV)
    df = df.rename(columns={"Unnamed: 0": "clip_id"})
    return df[["clip_id", "audio_path", "child_id", "label", "split", "timepoint_norm"]]


def load_one(name: str, val_path: str, test_path: str, score_col: str,
             key: str, master: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (val_df, test_df) each with columns [audio_path, <name>]."""
    if not (REPO / val_path).exists() or not (REPO / test_path).exists():
        print(f"  SKIP {name}: missing prediction CSV", flush=True)
        return None, None
    v = pd.read_csv(REPO / val_path)
    t = pd.read_csv(REPO / test_path)
    if score_col not in v.columns:
        print(f"  SKIP {name}: column {score_col!r} not in val csv ({list(v.columns)})", flush=True)
        return None, None
    if key == "clip_id":
        v = v[["clip_id", score_col]].merge(master[["clip_id", "audio_path"]], on="clip_id", how="left")
        t = t[["clip_id", score_col]].merge(master[["clip_id", "audio_path"]], on="clip_id", how="left")
    v = v[["audio_path", score_col]].rename(columns={score_col: name}).dropna()
    t = t[["audio_path", score_col]].rename(columns={score_col: name}).dropna()
    # Keep one row per clip in case of duplicates
    v = v.drop_duplicates(subset=["audio_path"], keep="first").reset_index(drop=True)
    t = t.drop_duplicates(subset=["audio_path"], keep="first").reset_index(drop=True)
    return v, t


def tune_threshold_ba(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Threshold on a 0.01 grid that maximises balanced accuracy."""
    grid = np.arange(0.01, 1.0, 0.01)
    best_t, best_ba = 0.5, -1.0
    for t in grid:
        ba = balanced_accuracy_score(y_true, (y_prob >= t).astype(int))
        if ba > best_ba:
            best_ba, best_t = ba, float(t)
    return best_t


def metrics(y: np.ndarray, p: np.ndarray, thr: float) -> dict:
    yhat = (p >= thr).astype(int)
    return {
        "threshold": round(thr, 3),
        "f1":        round(float(f1_score(y, yhat, average="weighted", zero_division=0)), 4),
        "bal_acc":   round(float(balanced_accuracy_score(y, yhat)), 4),
        "auroc":     round(float(roc_auc_score(y, p)),   4) if len(np.unique(y)) > 1 else float("nan"),
        "auprc":     round(float(average_precision_score(y, p)), 4) if len(np.unique(y)) > 1 else float("nan"),
    }


def main() -> None:
    master = load_master()
    val_labels  = master[master["split"] == "val"][["audio_path", "label"]].reset_index(drop=True)
    test_labels = master[master["split"] == "test"][["audio_path", "label"]].reset_index(drop=True)
    print(f"Master split sizes: val={len(val_labels)}, test={len(test_labels)}")

    # Load every system into wide val and test frames
    val_wide  = val_labels.copy()
    test_wide = test_labels.copy()
    loaded = []
    for name, vpath, tpath, col, key in SYSTEMS:
        v, t = load_one(name, vpath, tpath, col, key, master)
        if v is None: continue
        val_wide  = val_wide.merge(v,  on="audio_path", how="left")
        test_wide = test_wide.merge(t, on="audio_path", how="left")
        loaded.append(name)
    print(f"Loaded {len(loaded)} systems: {loaded}")

    # Two universes: (1) "full" = clips with all systems present (inner join);
    # we report top-k on this for fair k comparison.
    full_val  = val_wide.dropna(subset=loaded).reset_index(drop=True)
    full_test = test_wide.dropna(subset=loaded).reset_index(drop=True)
    print(f"Inner-join universe: val n={len(full_val)} / test n={len(full_test)}")

    # Rank systems by val AUROC on full universe
    ranking = []
    y_val_full = full_val["label"].to_numpy(int)
    for name in loaded:
        p = full_val[name].to_numpy(float)
        if len(np.unique(y_val_full)) < 2: continue
        ranking.append((name, float(roc_auc_score(y_val_full, p))))
    ranking.sort(key=lambda r: -r[1])
    print("\nSystem ranking on val AUROC (inner-join universe):")
    for i, (name, au) in enumerate(ranking, 1):
        print(f"  {i:2d}. {name:25s}  val_AUROC={au:.4f}")

    # Per-k stacker
    rows = []
    y_test_full = full_test["label"].to_numpy(int)
    k_values = sorted({1, 2, 3, 5, 7, 10, 13, 15, len(ranking)})
    for k in k_values:
        if k > len(ranking): continue
        top_names = [n for n, _ in ranking[:k]]
        X_val  = full_val[top_names].to_numpy(float)
        X_test = full_test[top_names].to_numpy(float)
        if k == 1:
            # No fit — use the single system's raw probability
            p_val  = X_val[:, 0]
            p_test = X_test[:, 0]
            descr  = top_names[0]
        else:
            clf = LogisticRegression(C=1.0, max_iter=500, random_state=SEED)
            clf.fit(X_val, y_val_full)
            p_val  = clf.predict_proba(X_val)[:, 1]
            p_test = clf.predict_proba(X_test)[:, 1]
            descr  = "LR-stack(top-%d)" % k
        thr = tune_threshold_ba(y_val_full, p_val)
        v_m = metrics(y_val_full,  p_val,  thr); v_m["split"] = "val"
        t_m = metrics(y_test_full, p_test, thr); t_m["split"] = "test"
        for m in (v_m, t_m):
            m.update({"k": k, "model": descr, "systems": ",".join(top_names)})
            rows.append(m)
        print(f"\n  k={k:2d}  {descr}")
        print(f"     val : F1={v_m['f1']:.4f}  BA={v_m['bal_acc']:.4f}  AUROC={v_m['auroc']:.4f}  thr={thr:.2f}")
        print(f"     test: F1={t_m['f1']:.4f}  BA={t_m['bal_acc']:.4f}  AUROC={t_m['auroc']:.4f}  AUPRC={t_m['auprc']:.4f}")

    out = pd.DataFrame(rows)[["k", "model", "split", "f1", "bal_acc", "auroc", "auprc", "threshold", "systems"]]
    out.to_csv(OUT_CSV, index=False)
    print(f"\nWrote {OUT_CSV} ({len(out)} rows)")

    # Compact ladder: test-only rows, ranked
    ladder = out[out["split"] == "test"].copy().sort_values("k").reset_index(drop=True)
    ladder.to_csv(OUT_LADDER, index=False)
    print(f"Wrote {OUT_LADDER}")


if __name__ == "__main__":
    main()
