"""US1 (spec-017): real AV-HuBERT-Large embeddings late-fused with audio pseudo-frame score.

Drop-in variant of `avhubert_late_fusion.py` that swaps the hand-engineered
mouth-motion features for real AV-HuBERT-Large embeddings (T120 output).

Visual feature vector per clip = mean + std + max + p95 over frame embeddings
(D=1024 each → 4096-d), concatenated with the 9 visual-eligibility features
from spec-015. Same 3 fusion configs (audio_only / always_fuse / gated_av) and
same val/test splits as the substitute baseline.

Outputs: pseudo_frame/results/avhubert_real_lipfusion/{audio_only,always_fuse,gated_av}/
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

OUT_BASE = os.path.join(_REPO, "pseudo_frame/results/avhubert_real_lipfusion")
AUDIO_DIR = os.path.join(_REPO, "pseudo_frame/results/wavlm_pseudo_frame")
ELIG_CSV = os.path.join(_REPO, "pseudo_frame/visual_features/visual_eligibility.csv")
AVH_POOLED_CSV = os.path.join(_REPO, "pseudo_frame/visual_features/avhubert_pooled.csv")
SEED = 42


def metrics(y, p, thr):
    y = np.asarray(y, dtype=int); p = np.asarray(p, dtype=float)
    pred = (p >= thr).astype(int)
    out = {"n": int(len(y)), "n_pos": int(y.sum()),
           "f1": float(f1_score(y, pred, zero_division=0)),
           "precision": float(precision_score(y, pred, zero_division=0)),
           "recall": float(recall_score(y, pred, zero_division=0)),
           "threshold": float(thr)}
    try: out["auroc"] = float(roc_auc_score(y, p)) if y.sum() and y.sum() < len(y) else float("nan")
    except Exception: out["auroc"] = float("nan")
    try: out["auprc"] = float(average_precision_score(y, p))
    except Exception: out["auprc"] = float("nan")
    return out


def tune_threshold(y, p):
    best_t, best_f1 = 0.5, -1.0
    for t in np.linspace(0.05, 0.95, 181):
        f = f1_score(np.asarray(y), (np.asarray(p) >= t).astype(int), zero_division=0)
        if f > best_f1: best_f1, best_t = float(f), float(t)
    return best_t


def tune_alpha(y, audio, visual):
    best_alpha, best_t, best_f1 = 1.0, 0.5, -1.0
    for alpha in np.linspace(0.0, 1.0, 21):
        s = alpha * audio + (1 - alpha) * visual
        t = tune_threshold(y, s)
        m = f1_score(np.asarray(y), (s >= t).astype(int), zero_division=0)
        if m > best_f1: best_alpha, best_t, best_f1 = float(alpha), float(t), float(m)
    return best_alpha, best_t


def tune_eligibility_threshold(y, audio, visual, eligibility):
    best_tau, best_alpha, best_f1 = 0.5, 1.0, -1.0
    for tau in np.linspace(0.1, 0.9, 17):
        eligible = eligibility >= tau
        if eligible.sum() < 5 or (~eligible).sum() < 5: continue
        for alpha in np.linspace(0.0, 1.0, 11):
            fused = alpha * audio + (1 - alpha) * visual
            score = np.where(eligible, fused, audio)
            t = tune_threshold(y, score)
            f = f1_score(np.asarray(y), (score >= t).astype(int), zero_division=0)
            if f > best_f1: best_f1, best_tau, best_alpha = float(f), float(tau), float(alpha)
    return best_tau, best_alpha


def avh_feature_cols(df: pd.DataFrame) -> list:
    return [c for c in df.columns if c.startswith("avh_")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=OUT_BASE)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    val_au = pd.read_csv(os.path.join(AUDIO_DIR, "val_predictions.csv"))[
        ["audio_path", "label", "score"]].rename(columns={"score": "audio_score"})
    test_au = pd.read_csv(os.path.join(AUDIO_DIR, "test_predictions.csv"))[
        ["audio_path", "label", "timepoint_norm", "score"]].rename(columns={"score": "audio_score"})

    elig = pd.read_csv(ELIG_CSV)
    avh = pd.read_csv(AVH_POOLED_CSV)

    elig_feats = [c for c in [
        "face_count_max", "face_count_mean", "face_area_max_norm", "face_area_mean_norm",
        "face_confidence_mean", "face_track_coverage_ratio",
        "n_distinct_tracks", "has_any_face", "eligibility_score",
    ] if c in elig.columns]
    avh_feats = avh_feature_cols(avh)
    print(f"AV-HuBERT pooled features: {len(avh_feats)}; eligibility features: {len(elig_feats)}", flush=True)

    val = val_au.merge(elig, on="audio_path", how="left").merge(avh, on="audio_path", how="left")
    test = test_au.merge(elig, on="audio_path", how="left").merge(avh, on="audio_path", how="left")
    for col in elig_feats + avh_feats:
        val[col] = val[col].fillna(0.0); test[col] = test[col].fillna(0.0)

    print(f"Val: {len(val)} clips | Test: {len(test)} clips", flush=True)

    Xv_val  = val[elig_feats + avh_feats].to_numpy(dtype=float)
    Xv_test = test[elig_feats + avh_feats].to_numpy(dtype=float)
    yv_val  = val["label"].to_numpy(dtype=int)
    yv_test = test["label"].to_numpy(dtype=int)

    visual_lr = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(C=1.0, max_iter=2000, random_state=SEED)),
    ])
    visual_lr.fit(Xv_val, yv_val)
    val_visual  = visual_lr.predict_proba(Xv_val)[:, 1]
    test_visual = visual_lr.predict_proba(Xv_test)[:, 1]
    print(f"Visual-only LR val AUROC: {roc_auc_score(yv_val, val_visual):.4f}", flush=True)
    print(f"Visual-only LR test AUROC: {roc_auc_score(yv_test, test_visual):.4f}", flush=True)

    val_audio  = val["audio_score"].to_numpy(dtype=float)
    test_audio = test["audio_score"].to_numpy(dtype=float)

    # (1) audio_only  (kept for layout symmetry; identical numbers to substitute audio_only)
    t_audio = tune_threshold(yv_val, val_audio)
    out_a = os.path.join(args.out, "audio_only"); os.makedirs(out_a, exist_ok=True)
    json.dump(metrics(yv_test, test_audio, t_audio), open(os.path.join(out_a, "test_metrics_tuned.json"), "w"), indent=2)
    json.dump(metrics(yv_val,  val_audio,  t_audio), open(os.path.join(out_a, "val_metrics_tuned.json"), "w"), indent=2)
    print(f"\n[audio_only]  test F1={f1_score(yv_test, (test_audio>=t_audio).astype(int), zero_division=0):.4f} "
          f"AUROC={roc_auc_score(yv_test, test_audio):.4f}", flush=True)

    # (2) always_fuse
    alpha, t_fuse = tune_alpha(yv_val, val_audio, val_visual)
    val_fused  = alpha * val_audio  + (1 - alpha) * val_visual
    test_fused = alpha * test_audio + (1 - alpha) * test_visual
    fuse_test = metrics(yv_test, test_fused, t_fuse); fuse_test["alpha"] = alpha
    out_f = os.path.join(args.out, "always_fuse"); os.makedirs(out_f, exist_ok=True)
    json.dump(fuse_test, open(os.path.join(out_f, "test_metrics_tuned.json"), "w"), indent=2)
    json.dump(metrics(yv_val, val_fused, t_fuse) | {"alpha": alpha},
              open(os.path.join(out_f, "val_metrics_tuned.json"), "w"), indent=2)
    print(f"[always_fuse] α={alpha:.2f}  test F1={fuse_test['f1']:.4f} AUROC={fuse_test['auroc']:.4f}", flush=True)

    # (3) gated_av
    elig_score_val  = val["eligibility_score"].to_numpy(dtype=float)
    elig_score_test = test["eligibility_score"].to_numpy(dtype=float)
    tau, alpha_g = tune_eligibility_threshold(yv_val, val_audio, val_visual, elig_score_val)
    eligible_val  = elig_score_val  >= tau
    eligible_test = elig_score_test >= tau
    val_g  = np.where(eligible_val,  alpha_g * val_audio  + (1 - alpha_g) * val_visual,  val_audio)
    test_g = np.where(eligible_test, alpha_g * test_audio + (1 - alpha_g) * test_visual, test_audio)
    t_g = tune_threshold(yv_val, val_g)
    gated_test = metrics(yv_test, test_g, t_g)
    gated_test["alpha"] = alpha_g; gated_test["tau_eligibility"] = tau
    gated_test["n_eligible_test"] = int(eligible_test.sum())
    out_g = os.path.join(args.out, "gated_av"); os.makedirs(out_g, exist_ok=True)
    json.dump(gated_test, open(os.path.join(out_g, "test_metrics_tuned.json"), "w"), indent=2)
    json.dump(metrics(yv_val, val_g, t_g) | {"alpha": alpha_g, "tau_eligibility": tau},
              open(os.path.join(out_g, "val_metrics_tuned.json"), "w"), indent=2)
    print(f"[gated_av]    α={alpha_g:.2f} τ={tau:.2f}  test F1={gated_test['f1']:.4f} "
          f"AUROC={gated_test['auroc']:.4f}  n_eligible={int(eligible_test.sum())}/{len(test)}", flush=True)

    # Eligible-subset diagnostics
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
        "audio_only":     metrics(y_sub, audio_sub, t_audio),
        "always_fuse":    metrics(y_sub, fused_sub, t_fuse),
        "gated_av_subset": metrics(y_sub, gated_sub, t_g),
    }
    json.dump(eligible_only, open(os.path.join(args.out, "subset_eligible_metrics.json"), "w"), indent=2)
    for k in ["audio_only", "always_fuse", "gated_av_subset"]:
        m = eligible_only[k]
        print(f"  [{k}] F1={m['f1']:.4f} AUROC={m['auroc']:.4f} AUPRC={m['auprc']:.4f} n={m['n']}", flush=True)

    # config snapshot
    json.dump({
        "spec": "017",
        "us": "US1",
        "feature_source": "real AV-HuBERT-Large embeddings (T120 output, mean+std+max+p95 pooled per clip)",
        "n_avh_features": len(avh_feats),
        "n_elig_features": len(elig_feats),
        "fusion_configs": ["audio_only", "always_fuse", "gated_av"],
        "alpha_always_fuse": alpha,
        "alpha_gated": alpha_g,
        "tau_eligibility": tau,
        "seed": SEED,
        "audio_baseline_dir": AUDIO_DIR,
    }, open(os.path.join(args.out, "config.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
