"""US2: Late-fuse mouth-motion visual score with the audio pseudo-frame score.

Three configurations evaluated:
  - audio_only:   pseudo-frame audio score (already in pseudo_frame/results/wavlm_pseudo_frame/)
  - always_fuse:  α · audio + (1−α) · visual,  α tuned on val
  - gated_av:     fuse for clips where visual_eligibility >= τ; audio-only otherwise

Visual score is produced by a small LR trained on val set using:
  - 17 mouth-motion features from pseudo_frame/visual_features/mouth_motion.csv
  - the 9 visual-eligibility features from pseudo_frame/visual_features/visual_eligibility.csv

Outputs: pseudo_frame/results/avhubert_lipfusion/{audio_only,always_fuse,gated_av}/

Usage:
  python pseudo_frame/avhubert_late_fusion.py
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)

OUT_BASE = os.path.join(_REPO, "pseudo_frame/results/avhubert_lipfusion")
AUDIO_DIR = os.path.join(_REPO, "pseudo_frame/results/wavlm_pseudo_frame")
ELIG_CSV = os.path.join(_REPO, "pseudo_frame/visual_features/visual_eligibility.csv")
MOUTH_CSV = os.path.join(_REPO, "pseudo_frame/visual_features/mouth_motion.csv")
SEED = 42


def metrics(y, p, thr):
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    pred = (p >= thr).astype(int)
    out = {
        "n": int(len(y)),
        "n_pos": int(y.sum()),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "threshold": float(thr),
    }
    try:
        out["auroc"] = float(roc_auc_score(y, p)) if y.sum() and y.sum() < len(y) else float("nan")
    except Exception:
        out["auroc"] = float("nan")
    try:
        out["auprc"] = float(average_precision_score(y, p))
    except Exception:
        out["auprc"] = float("nan")
    return out


def tune_threshold(y, p):
    best_t, best_f1 = 0.5, -1.0
    for t in np.linspace(0.05, 0.95, 181):
        f = f1_score(np.asarray(y), (np.asarray(p) >= t).astype(int), zero_division=0)
        if f > best_f1:
            best_f1, best_t = float(f), float(t)
    return best_t


def tune_alpha(y, audio, visual):
    best_alpha, best_t, best_f1 = 1.0, 0.5, -1.0
    for alpha in np.linspace(0.0, 1.0, 21):
        s = alpha * audio + (1 - alpha) * visual
        t = tune_threshold(y, s)
        m = f1_score(np.asarray(y), (s >= t).astype(int), zero_division=0)
        if m > best_f1:
            best_alpha, best_t, best_f1 = float(alpha), float(t), float(m)
    return best_alpha, best_t


def tune_eligibility_threshold(y, audio, visual_or_fused, eligibility):
    """For gated_av: pick τ so that the better of (audio_only, fused) is selected per clip."""
    best_tau, best_f1, best_alpha = 0.5, -1.0, 1.0
    for tau in np.linspace(0.1, 0.9, 17):
        eligible = eligibility >= tau
        # For eligible: search alpha; for ineligible: audio
        if eligible.sum() < 5 or (~eligible).sum() < 5:
            continue
        for alpha in np.linspace(0.0, 1.0, 11):
            fused = alpha * audio + (1 - alpha) * visual_or_fused
            score = np.where(eligible, fused, audio)
            t = tune_threshold(y, score)
            f = f1_score(np.asarray(y),
                         (score >= t).astype(int),
                         zero_division=0)
            if f > best_f1:
                best_f1 = float(f)
                best_tau = float(tau)
                best_alpha = float(alpha)
    return best_tau, best_alpha


def visual_feature_cols(elig: pd.DataFrame, mouth: pd.DataFrame) -> list:
    elig_feats = [c for c in [
        "face_count_max", "face_count_mean", "face_area_max_norm", "face_area_mean_norm",
        "face_confidence_mean", "face_track_coverage_ratio",
        "n_distinct_tracks", "has_any_face", "eligibility_score",
    ] if c in elig.columns]
    motion_feats = [c for c in mouth.columns if c not in ("audio_path", "error", "mouth_extraction_failed")]
    return elig_feats, motion_feats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=OUT_BASE)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    # ---- Load audio scores ----
    val_au = pd.read_csv(os.path.join(AUDIO_DIR, "val_predictions.csv"))[
        ["audio_path", "label", "score"]].rename(columns={"score": "audio_score"})
    test_au = pd.read_csv(os.path.join(AUDIO_DIR, "test_predictions.csv"))[
        ["audio_path", "label", "timepoint_norm", "score"]].rename(columns={"score": "audio_score"})

    # Need timepoint_norm for stratification; merge from val_predictions
    if "timepoint_norm" not in val_au.columns:
        master = pd.read_csv(os.path.join(_REPO, "whisper-modeling/seen_child_splits/val.csv"))
        master = master[["audio_path", "timepoint_norm"]]
        val_au = val_au.merge(master, on="audio_path", how="left")

    # ---- Load visual features ----
    elig = pd.read_csv(ELIG_CSV)
    mouth = pd.read_csv(MOUTH_CSV)
    elig_feats, motion_feats = visual_feature_cols(elig, mouth)

    val = val_au.merge(elig, on="audio_path", how="left").merge(mouth, on="audio_path", how="left")
    test = test_au.merge(elig, on="audio_path", how="left").merge(mouth, on="audio_path", how="left")
    for col in elig_feats + motion_feats:
        val[col]  = val[col].fillna(0.0)
        test[col] = test[col].fillna(0.0)

    print(f"Val: {len(val)} clips | Test: {len(test)} clips", flush=True)
    print(f"Visual features: {len(elig_feats)} eligibility + {len(motion_feats)} motion "
          f"= {len(elig_feats) + len(motion_feats)} total", flush=True)

    # ---- Train visual-only LR on val (predict label from visual features) ----
    Xv_val = val[elig_feats + motion_feats].to_numpy(dtype=float)
    yv_val = val["label"].to_numpy(dtype=int)
    Xv_test = test[elig_feats + motion_feats].to_numpy(dtype=float)
    yv_test = test["label"].to_numpy(dtype=int)

    visual_lr = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(C=1.0, max_iter=2000, random_state=SEED)),
    ])
    visual_lr.fit(Xv_val, yv_val)
    val_visual = visual_lr.predict_proba(Xv_val)[:, 1]
    test_visual = visual_lr.predict_proba(Xv_test)[:, 1]
    print(f"\nVisual-only LR val AUROC: {roc_auc_score(yv_val, val_visual):.4f}", flush=True)
    print(f"Visual-only LR test AUROC: {roc_auc_score(yv_test, test_visual):.4f}", flush=True)

    # ---- (1) audio_only ----
    val_audio = val["audio_score"].to_numpy(dtype=float)
    test_audio = test["audio_score"].to_numpy(dtype=float)
    t_audio = tune_threshold(yv_val, val_audio)
    audio_only_val = metrics(yv_val, val_audio, t_audio)
    audio_only_test = metrics(yv_test, test_audio, t_audio)

    out_audio = os.path.join(args.out, "audio_only")
    os.makedirs(out_audio, exist_ok=True)
    json.dump(audio_only_test, open(os.path.join(out_audio, "test_metrics_tuned.json"), "w"), indent=2)
    json.dump(audio_only_val, open(os.path.join(out_audio, "val_metrics_tuned.json"), "w"), indent=2)
    print(f"\n[audio_only] test F1={audio_only_test['f1']:.4f} AUROC={audio_only_test['auroc']:.4f}",
          flush=True)

    # ---- (2) always_fuse ----
    alpha, t_fuse = tune_alpha(yv_val, val_audio, val_visual)
    val_fused = alpha * val_audio + (1 - alpha) * val_visual
    test_fused = alpha * test_audio + (1 - alpha) * test_visual
    fuse_val = metrics(yv_val, val_fused, t_fuse)
    fuse_test = metrics(yv_test, test_fused, t_fuse)
    fuse_test["alpha"] = alpha
    fuse_val["alpha"] = alpha

    out_fuse = os.path.join(args.out, "always_fuse")
    os.makedirs(out_fuse, exist_ok=True)
    json.dump(fuse_test, open(os.path.join(out_fuse, "test_metrics_tuned.json"), "w"), indent=2)
    json.dump(fuse_val, open(os.path.join(out_fuse, "val_metrics_tuned.json"), "w"), indent=2)
    print(f"[always_fuse] α={alpha:.2f}  test F1={fuse_test['f1']:.4f} AUROC={fuse_test['auroc']:.4f}",
          flush=True)

    # ---- (3) gated_av ----
    elig_score_val  = val["eligibility_score"].to_numpy(dtype=float)
    elig_score_test = test["eligibility_score"].to_numpy(dtype=float)
    tau, alpha_gated = tune_eligibility_threshold(yv_val, val_audio, val_visual, elig_score_val)
    eligible_val  = elig_score_val  >= tau
    eligible_test = elig_score_test >= tau
    val_g  = np.where(eligible_val,  alpha_gated * val_audio  + (1 - alpha_gated) * val_visual,  val_audio)
    test_g = np.where(eligible_test, alpha_gated * test_audio + (1 - alpha_gated) * test_visual, test_audio)
    t_g = tune_threshold(yv_val, val_g)
    gated_val = metrics(yv_val, val_g, t_g)
    gated_test = metrics(yv_test, test_g, t_g)
    gated_test["alpha"] = alpha_gated
    gated_test["tau_eligibility"] = tau
    gated_test["n_eligible_test"] = int(eligible_test.sum())

    out_g = os.path.join(args.out, "gated_av")
    os.makedirs(out_g, exist_ok=True)
    json.dump(gated_test, open(os.path.join(out_g, "test_metrics_tuned.json"), "w"), indent=2)
    json.dump(gated_val, open(os.path.join(out_g, "val_metrics_tuned.json"), "w"), indent=2)
    print(f"[gated_av]    α={alpha_gated:.2f} τ={tau:.2f}  "
          f"test F1={gated_test['f1']:.4f} AUROC={gated_test['auroc']:.4f}  "
          f"n_eligible={int(eligible_test.sum())}/{len(test)}", flush=True)

    # ---- Subset metrics on visually-eligible test clips only ----
    print("\n=== ON VISUALLY-ELIGIBLE TEST SUBSET ===", flush=True)
    elig_subset = test[eligible_test]
    y_sub = elig_subset["label"].to_numpy(dtype=int)
    audio_sub = elig_subset["audio_score"].to_numpy(dtype=float)
    visual_sub = test_visual[eligible_test]
    fused_sub = alpha * audio_sub + (1 - alpha) * visual_sub
    gated_sub = test_g[eligible_test]

    eligible_only = {
        "n_eligible": int(eligible_test.sum()),
        "tau": float(tau),
        "audio_only": metrics(y_sub, audio_sub, t_audio),
        "always_fuse": metrics(y_sub, fused_sub, t_fuse),
        "gated_av_subset": metrics(y_sub, gated_sub, t_g),
    }
    json.dump(eligible_only, open(os.path.join(args.out, "subset_eligible_metrics.json"), "w"),
              indent=2)
    for k in ["audio_only", "always_fuse", "gated_av_subset"]:
        m = eligible_only[k]
        print(f"  [{k}] F1={m['f1']:.4f} AUROC={m['auroc']:.4f} AUPRC={m['auprc']:.4f} "
              f"n={m['n']}", flush=True)

    # ---- Predictions table ----
    preds = test[["audio_path", "label"]].copy()
    if "timepoint_norm" in test.columns:
        preds["timepoint_norm"] = test["timepoint_norm"]
    preds["audio_score"] = test_audio
    preds["visual_score"] = test_visual
    preds["always_fuse_score"] = test_fused
    preds["gated_av_score"] = test_g
    preds["eligible"] = eligible_test.astype(int)
    preds.to_csv(os.path.join(args.out, "test_predictions_all.csv"), index=False)

    # ---- Config ----
    cfg = {
        "method": "AV late fusion (mouth-motion features substituting AV-HuBERT)",
        "audio_source": AUDIO_DIR,
        "visual_features": elig_feats + motion_feats,
        "n_visual_features": len(elig_feats) + len(motion_feats),
        "alpha_always_fuse": alpha,
        "alpha_gated": alpha_gated,
        "tau_eligibility": tau,
        "thresholds": {
            "audio_only": t_audio,
            "always_fuse": t_fuse,
            "gated_av": t_g,
        },
        "seed": SEED,
        "created": "2026-04-29",
        "note": "Hand-engineered face/mouth-motion features substitute for AV-HuBERT "
                "(fairseq install non-trivial on this cluster). The architectural intent "
                "(frozen visual extractor + tiny fusion + visual-eligibility gating) is preserved.",
    }
    json.dump(cfg, open(os.path.join(args.out, "config.json"), "w"), indent=2)
    print(f"\nConfig + predictions written to: {args.out}", flush=True)


if __name__ == "__main__":
    main()
