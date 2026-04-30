"""US3: Speaker-Embedding-Informed AV fusion (simplified Clarke et al. 2025).

Clarke et al. show that adding speaker-comparison info to ASD improves Ego4D
mAP by 14.5%/10.3%. The full Clarke design needs per-face-track ASD scores AND
per-track audio-ECAPA-vs-prototype scores. This project's existing ECAPA
prototype cache is at the clip level (one cosine per clip via the babar/vtc
enrollment pipeline), not per-face-track, so we implement a simplified version:

  joint_score = audio_speaker_score × visual_speaking_score

where:
  - audio_speaker_score = babar_prob (clip-level ECAPA cosine to target-child prototype)
  - visual_speaking_score = LR over mouth-motion features (US2 visual head)

Hypothesis: multiplicative fusion is more discriminative on n_children≥2
clips than additive fusion (US2), because both modalities must agree before
a high score is produced — exactly the failure mode where mean-ensemble
suppressors fail (spec-012 US3 multi-child suppressor was a null result).

Outputs: pseudo_frame/results/speaker_informed_asd/{test_metrics_tuned.json,
test_predictions.csv, val_metrics_tuned.json, multi_child_test_metrics.json,
config.json}.

Usage:
  python pseudo_frame/speaker_informed_asd.py
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

DEFAULT_OUT_DIR = os.path.join(_REPO, "pseudo_frame/results/speaker_informed_asd")
PER_TRACK_OUT_DIR = os.path.join(_REPO, "pseudo_frame/results/speaker_informed_asd_per_track")
SEED = 42

# Audio "is the target child speaking" score: BabAR ECAPA cosine
# (best non-MIL audio diarizer; the literature's "speaker-comparison cue").
BABAR_VAL = os.path.join(_REPO, "babar_ecapa_enrollment_runs/enroll_val_predictions.csv")
BABAR_TEST = os.path.join(_REPO, "babar_ecapa_enrollment_runs/enroll_test_predictions.csv")
WAVLM_PSEUDO_VAL = os.path.join(_REPO, "pseudo_frame/results/wavlm_pseudo_frame/val_predictions.csv")
WAVLM_PSEUDO_TEST = os.path.join(_REPO, "pseudo_frame/results/wavlm_pseudo_frame/test_predictions.csv")
ELIG_CSV = os.path.join(_REPO, "pseudo_frame/visual_features/visual_eligibility.csv")
MOUTH_CSV = os.path.join(_REPO, "pseudo_frame/visual_features/mouth_motion.csv")
PER_TRACK_CSV = os.path.join(_REPO, "pseudo_frame/visual_features/per_track_speaker_score.csv")
MASTER = os.path.join(_REPO, "whisper-modeling/seen_child_splits/master_with_split.csv")


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


def load_split_with_meta(split: str) -> pd.DataFrame:
    """Load metadata stratifiers (n_children, child_of_interest_clear) for a split."""
    df = pd.read_csv(MASTER)
    df = df[df["split"] == split].reset_index(drop=True)

    def _to_int(val, default):
        try:
            return int(str(val).strip().split("+")[0])
        except Exception:
            return default

    df["n_children_int"] = df["#_children"].apply(lambda v: _to_int(v, 1))
    df["coi_norm"] = df["Child_of_interest_clear"].astype(str).str.strip().str.lower()
    return df[["audio_path", "n_children_int", "coi_norm", "timepoint_norm"]]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-track", action="store_true",
                        help="Use per-face-track ECAPA cosines (max-track) as the audio_speaker_score "
                             "instead of the clip-level BabAR enrollment prob. Falls back to BabAR "
                             "for clips with no face track. Output goes to "
                             "pseudo_frame/results/speaker_informed_asd_per_track/.")
    parser.add_argument("--per-track-agg", choices=["max", "top2_mean", "mean"], default="max",
                        help="How to aggregate per-track cosines to a single clip score (default: max).")
    args = parser.parse_args()

    OUT_DIR = PER_TRACK_OUT_DIR if args.per_track else DEFAULT_OUT_DIR
    os.makedirs(OUT_DIR, exist_ok=True)

    babar_val = pd.read_csv(BABAR_VAL)[["audio_path", "label", "prob"]] \
        .rename(columns={"prob": "audio_speaker_prob"})
    babar_test = pd.read_csv(BABAR_TEST)[["audio_path", "label", "prob"]] \
        .rename(columns={"prob": "audio_speaker_prob"})

    # Per-track override (Clarke 2025 full design): replace clip-level cosine
    # with max-track cosine; clips without face tracks keep the BabAR fallback
    if args.per_track:
        if not os.path.exists(PER_TRACK_CSV):
            raise FileNotFoundError(f"Per-track scores not found: {PER_TRACK_CSV}. "
                                    f"Run pseudo_frame/per_track_speaker_score.py first.")
        pt = pd.read_csv(PER_TRACK_CSV)
        col = {"max": "max_track_cosine",
               "top2_mean": "top2_mean_track_cosine",
               "mean": "mean_track_cosine"}[args.per_track_agg]
        # Map cosine in [-1, 1] to [0, 1] (ECAPA cosines are non-negative for same-speaker
        # but can dip below 0 for adult vs child child prototypes); clamp & rescale.
        pt["per_track_score"] = ((pt[col].clip(-1, 1) + 1.0) / 2.0)
        pt_lookup = pt.set_index("audio_path")[["has_any_track", "per_track_score"]]

        def _override(row):
            ap = row["audio_path"]
            if ap in pt_lookup.index and int(pt_lookup.loc[ap, "has_any_track"]) == 1:
                return float(pt_lookup.loc[ap, "per_track_score"])
            return float(row["audio_speaker_prob"])  # fallback

        babar_val["audio_speaker_prob"] = babar_val.apply(_override, axis=1)
        babar_test["audio_speaker_prob"] = babar_test.apply(_override, axis=1)
        n_val_overridden = int(babar_val["audio_path"].apply(
            lambda ap: ap in pt_lookup.index and int(pt_lookup.loc[ap, "has_any_track"]) == 1).sum())
        n_test_overridden = int(babar_test["audio_path"].apply(
            lambda ap: ap in pt_lookup.index and int(pt_lookup.loc[ap, "has_any_track"]) == 1).sum())
        print(f"Per-track override active (agg={args.per_track_agg}): "
              f"{n_val_overridden}/{len(babar_val)} val, "
              f"{n_test_overridden}/{len(babar_test)} test", flush=True)

    pseudo_val = pd.read_csv(WAVLM_PSEUDO_VAL)[["audio_path", "score"]] \
        .rename(columns={"score": "audio_pseudo_prob"})
    pseudo_test = pd.read_csv(WAVLM_PSEUDO_TEST)[["audio_path", "score"]] \
        .rename(columns={"score": "audio_pseudo_prob"})

    elig = pd.read_csv(ELIG_CSV)
    mouth = pd.read_csv(MOUTH_CSV)
    elig_feats = [c for c in [
        "face_count_max", "face_count_mean", "face_area_max_norm", "face_area_mean_norm",
        "face_confidence_mean", "face_track_coverage_ratio",
        "n_distinct_tracks", "has_any_face", "eligibility_score",
    ] if c in elig.columns]
    motion_feats = [c for c in mouth.columns if c not in ("audio_path", "error", "mouth_extraction_failed")]

    val_meta = load_split_with_meta("val")
    test_meta = load_split_with_meta("test")

    val = babar_val.merge(pseudo_val, on="audio_path", how="left") \
                   .merge(elig, on="audio_path", how="left") \
                   .merge(mouth, on="audio_path", how="left") \
                   .merge(val_meta, on="audio_path", how="left")
    test = babar_test.merge(pseudo_test, on="audio_path", how="left") \
                     .merge(elig, on="audio_path", how="left") \
                     .merge(mouth, on="audio_path", how="left") \
                     .merge(test_meta, on="audio_path", how="left")
    for col in elig_feats + motion_feats + ["audio_pseudo_prob"]:
        val[col]  = val[col].fillna(0.0)
        test[col] = test[col].fillna(0.0)

    print(f"Val: {len(val)}  Test: {len(test)}", flush=True)

    # Train visual LR on val
    X_val = val[elig_feats + motion_feats].to_numpy(dtype=float)
    y_val = val["label"].to_numpy(dtype=int)
    X_test = test[elig_feats + motion_feats].to_numpy(dtype=float)
    y_test = test["label"].to_numpy(dtype=int)
    visual_lr = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(C=1.0, max_iter=2000, random_state=SEED)),
    ])
    visual_lr.fit(X_val, y_val)
    val["visual_prob"] = visual_lr.predict_proba(X_val)[:, 1]
    test["visual_prob"] = visual_lr.predict_proba(X_test)[:, 1]

    # ---- Audio-only baseline (pseudo-frame from 11c) ----
    val_audio_pf = val["audio_pseudo_prob"].to_numpy()
    test_audio_pf = test["audio_pseudo_prob"].to_numpy()
    t_pf = tune_threshold(y_val, val_audio_pf)
    audio_pf_test = metrics(y_test, test_audio_pf, t_pf)

    # ---- Speaker-only baseline (BabAR ECAPA) ----
    val_speaker = val["audio_speaker_prob"].to_numpy()
    test_speaker = test["audio_speaker_prob"].to_numpy()
    t_sp = tune_threshold(y_val, val_speaker)
    speaker_only_test = metrics(y_test, test_speaker, t_sp)

    # ---- Multiplicative speaker-informed AV: speaker × visual ----
    val_joint = val_speaker * val["visual_prob"].to_numpy()
    test_joint = test_speaker * test["visual_prob"].to_numpy()
    t_joint = tune_threshold(y_val, val_joint)
    joint_test = metrics(y_test, test_joint, t_joint)

    # ---- Late additive fuse: pseudo-frame + (speaker × visual) ----
    best_alpha = 1.0
    best_f1 = -1.0
    best_t = 0.5
    for alpha in np.linspace(0.0, 1.0, 21):
        s = alpha * val_audio_pf + (1 - alpha) * val_joint
        t = tune_threshold(y_val, s)
        f = f1_score(y_val, (s >= t).astype(int), zero_division=0)
        if f > best_f1:
            best_f1 = float(f)
            best_alpha = float(alpha)
            best_t = float(t)
    fused_test = best_alpha * test_audio_pf + (1 - best_alpha) * test_joint
    fused_test_m = metrics(y_test, fused_test, best_t)
    fused_test_m["alpha"] = best_alpha

    print(f"\n=== TEST METRICS ===")
    print(f"  Pseudo-frame audio only:        F1={audio_pf_test['f1']:.4f} AUROC={audio_pf_test['auroc']:.4f}")
    print(f"  BabAR speaker-only (audio):     F1={speaker_only_test['f1']:.4f} AUROC={speaker_only_test['auroc']:.4f}")
    print(f"  Joint  (speaker × visual):      F1={joint_test['f1']:.4f} AUROC={joint_test['auroc']:.4f}")
    print(f"  Fused  (α·pseudo + (1−α)·joint, α={best_alpha:.2f}): "
          f"F1={fused_test_m['f1']:.4f} AUROC={fused_test_m['auroc']:.4f}")

    # ---- Stratified on n_children≥2 (the spec-012 hard stratum) ----
    print("\n=== ON n_children≥2 TEST SUBSET (Clarke hypothesis stratum) ===")
    multi_mask = test["n_children_int"] >= 2
    yt_multi = test.loc[multi_mask, "label"].to_numpy(dtype=int)
    test_audio_multi = test_audio_pf[multi_mask.to_numpy()]
    test_joint_multi = test_joint[multi_mask.to_numpy()]
    test_speaker_multi = test_speaker[multi_mask.to_numpy()]
    test_fused_multi = fused_test[multi_mask.to_numpy()]
    multi_metrics = {
        "n": int(multi_mask.sum()),
        "audio_pseudo": metrics(yt_multi, test_audio_multi, t_pf),
        "audio_speaker_only": metrics(yt_multi, test_speaker_multi, t_sp),
        "speaker_x_visual": metrics(yt_multi, test_joint_multi, t_joint),
        "fused":            metrics(yt_multi, test_fused_multi, best_t),
    }
    for k, m in multi_metrics.items():
        if isinstance(m, dict):
            print(f"  [{k:<22s}] F1={m['f1']:.4f} AUROC={m['auroc']:.4f} AUPRC={m['auprc']:.4f} n={m['n']}")
    json.dump(multi_metrics, open(os.path.join(OUT_DIR, "multi_child_test_metrics.json"), "w"),
              indent=2)

    # ---- Save core artifacts ----
    json.dump(fused_test_m, open(os.path.join(OUT_DIR, "test_metrics_tuned.json"), "w"), indent=2)
    json.dump({
        "audio_pseudo_only": audio_pf_test,
        "audio_speaker_only": speaker_only_test,
        "joint_speaker_visual": joint_test,
        "fused_pseudo_plus_joint": fused_test_m,
    }, open(os.path.join(OUT_DIR, "all_test_metrics.json"), "w"), indent=2)

    val_t_for_fused = best_t
    val_fused = best_alpha * val_audio_pf + (1 - best_alpha) * val_joint
    val_fused_m = metrics(y_val, val_fused, val_t_for_fused)
    val_fused_m["alpha"] = best_alpha
    json.dump(val_fused_m, open(os.path.join(OUT_DIR, "val_metrics_tuned.json"), "w"), indent=2)

    preds = test[["audio_path", "label", "n_children_int", "timepoint_norm"]].copy()
    preds["audio_pseudo_score"] = test_audio_pf
    preds["audio_speaker_score"] = test_speaker
    preds["visual_score"] = test["visual_prob"].to_numpy()
    preds["joint_score"] = test_joint
    preds["fused_score"] = fused_test
    preds["prob"] = fused_test
    preds["prediction"] = (fused_test >= best_t).astype(int)
    preds.to_csv(os.path.join(OUT_DIR, "test_predictions.csv"), index=False)

    val_preds = val[["audio_path", "label", "n_children_int", "timepoint_norm"]].copy()
    val_preds["audio_pseudo_score"] = val_audio_pf
    val_preds["audio_speaker_score"] = val_speaker
    val_preds["visual_score"] = val["visual_prob"].to_numpy()
    val_preds["joint_score"] = val_joint
    val_preds["fused_score"] = val_fused
    val_preds["prob"] = val_fused
    val_preds["prediction"] = (val_fused >= best_t).astype(int)
    val_preds.to_csv(os.path.join(OUT_DIR, "val_predictions.csv"), index=False)

    speaker_source = ("Per-face-track ECAPA cosine to target prototype "
                      f"(agg={args.per_track_agg}; falls back to BabAR clip score when no face track)"
                      if args.per_track else
                      "babar_ecapa_enrollment_runs/enroll_<split>_predictions.csv (clip-level ECAPA cosine)")
    cfg = {
        "method": "Speaker-embedding-informed AV (simplified Clarke et al. 2025)",
        "per_track_mode": bool(args.per_track),
        "per_track_aggregation": args.per_track_agg if args.per_track else None,
        "audio_speaker_source": speaker_source,
        "audio_pseudo_source": "pseudo_frame/results/wavlm_pseudo_frame/<split>_predictions.csv",
        "visual_features": elig_feats + motion_feats,
        "fusion": "joint = speaker × visual; final = α·pseudo + (1−α)·joint",
        "alpha": best_alpha,
        "threshold_fused": best_t,
        "seed": SEED,
        "created": "2026-04-29",
    }
    json.dump(cfg, open(os.path.join(OUT_DIR, "config.json"), "w"), indent=2)
    print(f"\nWrote: {OUT_DIR}/")


if __name__ == "__main__":
    main()
