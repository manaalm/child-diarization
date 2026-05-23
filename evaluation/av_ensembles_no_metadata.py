"""AV ensembles without BIDS metadata (timepoint allowed).

Target: beat the no-metadata audio ensemble (per_child_offset, AUROC 0.900) by
incorporating automatically-extracted visual signals — without using any of the
manual BIDS labels (#_children, Child_of_interest_clear, video quality flags).

Available auto AV resources (all face-detection-derived, no annotation):
  - pseudo_frame/visual_features/visual_eligibility.csv  (9 features)
  - pseudo_frame/visual_features/mouth_motion.csv        (22 features)
  - pseudo_frame/results/speaker_informed_asd/{val,test}_predictions.csv         (US3)
  - pseudo_frame/results/speaker_informed_asd_per_track/{val,test}_predictions.csv (US3-per-track)
  - mil/mil_results/whisper_mil/{val,test}_predictions.csv  (audio reference)

Variants:
  av_pure_visual          — 13 audio probs + timepoint + 9 visual eligibility features
  av_pure_visual_motion   — + 22 mouth-motion features (L2 regularized)
  av_per_track_added      — adds US3-per-track as a 14th base system
  av_visual_per_child     — av_pure_visual + per-child offset (best audio trick)
  av_full                 — system probs + visual + per_track + per-child offset
  audio_per_child (ref)   — winner from advanced_ensembles.py (no AV)

Outputs to ensemble_runs/advanced_av/{variant}/ + leaderboard.

Usage:
  python evaluation/av_ensembles_no_metadata.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from evaluation.advanced_ensembles import (
    SCORE_FEATS,
    load_system_scores,
    load_labels_split,
    assemble,
    save_variant,
    MASTER_CSV,
    SEED, BASELINE_F1, BASELINE_AUROC,
)
from evaluation.metadata_router import compute_metrics, tune_threshold

OUT_ROOT = _REPO / "ensemble_runs/advanced_av"
OUT_ROOT.mkdir(parents=True, exist_ok=True)

# ── Auto AV feature loaders ────────────────────────────────────────────────

VISUAL_ELIG = _REPO / "pseudo_frame/visual_features/visual_eligibility.csv"
MOUTH_MOTION = _REPO / "pseudo_frame/visual_features/mouth_motion.csv"
US3_PER_TRACK = _REPO / "pseudo_frame/results/speaker_informed_asd_per_track/{split}_predictions.csv"

VISUAL_FEATS = [
    "face_count_max", "face_count_mean",
    "face_area_max_norm", "face_area_mean_norm",
    "face_confidence_mean", "face_track_coverage_ratio",
    "n_distinct_tracks", "has_any_face", "eligibility_score",
]

# Mouth-motion: dropping `mouth_extraction_failed` (binary; redundant with mouth_extraction_rate)
# and `track_n_frames`/`n_mouth_frames` (already encoded in extraction_rate).
MOUTH_MOTION_FEATS = [
    "face_intensity_std_mean", "face_intensity_std_std", "face_intensity_std_max",
    "face_intensity_std_p95",
    "mouth_intensity_std_mean", "mouth_intensity_std_std", "mouth_intensity_std_max",
    "mouth_intensity_std_p95",
    "face_motion_energy_mean", "face_motion_energy_std", "face_motion_energy_max",
    "face_motion_energy_p95",
    "mouth_region_motion_energy_mean", "mouth_region_motion_energy_std",
    "mouth_region_motion_energy_max", "mouth_region_motion_energy_p95",
    "mouth_extraction_rate", "mouth_motion_variance", "face_motion_log_max",
]


def load_visual_eligibility() -> pd.DataFrame:
    df = pd.read_csv(VISUAL_ELIG)
    keep = ["audio_path"] + [c for c in VISUAL_FEATS if c in df.columns]
    return df[keep].copy()


def load_mouth_motion() -> pd.DataFrame:
    df = pd.read_csv(MOUTH_MOTION)
    keep = ["audio_path"] + [c for c in MOUTH_MOTION_FEATS if c in df.columns]
    return df[keep].copy()


def load_per_track_av(split: str) -> pd.DataFrame:
    p = Path(str(US3_PER_TRACK).replace("{split}", split))
    if not p.exists():
        return pd.DataFrame(columns=["audio_path", "us3_per_track_prob"])
    df = pd.read_csv(p)
    return df[["audio_path", "prob"]].rename(columns={"prob": "us3_per_track_prob"})


# ── Per-child offset helper ─────────────────────────────────────────────────

def per_child_offset_correction(val_y, val_base, val_df, test_base, test_df,
                                shrink_prior: float = 2.0):
    """Compute per-child mean residual on val, shrink, apply at test."""
    master = pd.read_csv(MASTER_CSV)[["audio_path", "child_id"]]
    val_with  = val_df.merge(master,  on="audio_path", how="left").reset_index(drop=True)
    test_with = test_df.merge(master, on="audio_path", how="left").reset_index(drop=True)
    resid = val_y - val_base
    offsets = (
        pd.DataFrame({"child_id": val_with["child_id"], "resid": resid})
        .groupby("child_id")["resid"].mean()
    )
    n_per = val_with["child_id"].value_counts()
    shrink = n_per / (n_per + shrink_prior)
    offsets = offsets * shrink.reindex(offsets.index).fillna(0.0)
    val_off  = val_with["child_id"].map(offsets).fillna(0.0).to_numpy()
    test_off = test_with["child_id"].map(offsets).fillna(0.0).to_numpy()
    return np.clip(val_base + val_off, 0, 1), np.clip(test_base + test_off, 0, 1)


# ── Variant builders ────────────────────────────────────────────────────────

def _stacker_lr(X_val, y_val, X_test, C=1.0):
    clf = LogisticRegression(C=C, max_iter=500, random_state=SEED)
    clf.fit(X_val, y_val)
    return clf.predict_proba(X_val)[:, 1], clf.predict_proba(X_test)[:, 1], clf


def variant_audio_pure(val_df, test_df, val_y, test_y):
    """Reference: audio-only pure stacker (13 systems + timepoint), no AV."""
    feats = list(SCORE_FEATS) + ["timepoint_is_36m"]
    Xv = val_df[feats].to_numpy(dtype=float); Xt = test_df[feats].to_numpy(dtype=float)
    v, t, _ = _stacker_lr(Xv, val_y, Xt)
    return v, t, {"features": feats}


def variant_audio_per_child(val_df, test_df, val_y, test_y):
    """Reference: audio per_child_offset (winner from advanced_ensembles.py)."""
    base_v, base_t, _ = variant_audio_pure(val_df, test_df, val_y, test_y)
    v, t = per_child_offset_correction(val_y, base_v, val_df, base_t, test_df)
    return v, t, {"first_stage": "audio_pure", "shrink_prior": 2.0}


def variant_av_pure_visual(val_df, test_df, val_y, test_y):
    """13 audio probs + timepoint + 9 auto visual features."""
    feats = list(SCORE_FEATS) + ["timepoint_is_36m"] + VISUAL_FEATS
    Xv = val_df[feats].fillna(0.0).to_numpy(dtype=float)
    Xt = test_df[feats].fillna(0.0).to_numpy(dtype=float)
    v, t, _ = _stacker_lr(Xv, val_y, Xt)
    return v, t, {"features": feats}


def variant_av_pure_visual_motion(val_df, test_df, val_y, test_y):
    """+ mouth motion features. L2 regularized harder (more features → overfit risk)."""
    feats = list(SCORE_FEATS) + ["timepoint_is_36m"] + VISUAL_FEATS + MOUTH_MOTION_FEATS
    Xv = val_df[feats].fillna(0.0).to_numpy(dtype=float)
    Xt = test_df[feats].fillna(0.0).to_numpy(dtype=float)
    # C=0.3 = stronger L2 to handle 22 extra features
    v, t, _ = _stacker_lr(Xv, val_y, Xt, C=0.3)
    return v, t, {"features": feats, "C": 0.3}


def variant_av_per_track_added(val_df, test_df, val_y, test_y):
    """Add US3-per-track as a 14th base system + visual features."""
    feats = list(SCORE_FEATS) + ["timepoint_is_36m", "us3_per_track_prob"] + VISUAL_FEATS
    Xv = val_df[feats].fillna(0.0).to_numpy(dtype=float)
    Xt = test_df[feats].fillna(0.0).to_numpy(dtype=float)
    v, t, _ = _stacker_lr(Xv, val_y, Xt)
    return v, t, {"features": feats}


def variant_av_visual_per_child(val_df, test_df, val_y, test_y):
    """av_pure_visual + per-child offset on top."""
    base_v, base_t, _ = variant_av_pure_visual(val_df, test_df, val_y, test_y)
    v, t = per_child_offset_correction(val_y, base_v, val_df, base_t, test_df)
    return v, t, {"first_stage": "av_pure_visual"}


def variant_av_full(val_df, test_df, val_y, test_y):
    """All AV signals: 13 probs + timepoint + visual + us3_per_track + per-child offset."""
    feats = list(SCORE_FEATS) + ["timepoint_is_36m", "us3_per_track_prob"] + VISUAL_FEATS
    Xv = val_df[feats].fillna(0.0).to_numpy(dtype=float)
    Xt = test_df[feats].fillna(0.0).to_numpy(dtype=float)
    base_v, base_t, _ = _stacker_lr(Xv, val_y, Xt)
    v, t = per_child_offset_correction(val_y, base_v, val_df, base_t, test_df)
    return v, t, {"features": feats, "first_stage": "av_per_track_added", "offset": True}


def variant_av_eligibility_only(val_df, test_df, val_y, test_y):
    """Minimal AV: 13 probs + timepoint + just `eligibility_score` (1 visual feature)."""
    feats = list(SCORE_FEATS) + ["timepoint_is_36m", "eligibility_score"]
    Xv = val_df[feats].fillna(0.0).to_numpy(dtype=float)
    Xt = test_df[feats].fillna(0.0).to_numpy(dtype=float)
    v, t, _ = _stacker_lr(Xv, val_y, Xt)
    return v, t, {"features": feats}


# ── Main ───────────────────────────────────────────────────────────────────

VARIANTS = [
    ("audio_pure",             variant_audio_pure),
    ("audio_per_child",        variant_audio_per_child),
    ("av_pure_visual",         variant_av_pure_visual),
    ("av_pure_visual_motion",  variant_av_pure_visual_motion),
    ("av_per_track_added",     variant_av_per_track_added),
    ("av_visual_per_child",    variant_av_visual_per_child),
    ("av_full",                variant_av_full),
    ("av_eligibility_only",    variant_av_eligibility_only),
]


def main() -> None:
    val_scores  = load_system_scores("val")
    test_scores = load_system_scores("test")
    labels      = load_labels_split()
    val_df  = assemble("val",  val_scores,  labels)
    test_df = assemble("test", test_scores, labels)

    # Add visual features
    vis = load_visual_eligibility()
    val_df  = val_df.merge(vis, on="audio_path", how="left")
    test_df = test_df.merge(vis, on="audio_path", how="left")
    print(f"  Visual eligibility joined: val coverage={val_df['has_any_face'].notna().mean():.3f}  "
          f"test coverage={test_df['has_any_face'].notna().mean():.3f}")

    mm = load_mouth_motion()
    val_df  = val_df.merge(mm, on="audio_path", how="left")
    test_df = test_df.merge(mm, on="audio_path", how="left")
    mm_cols = [c for c in MOUTH_MOTION_FEATS if c in val_df.columns]
    print(f"  Mouth motion joined: {len(mm_cols)} features available")

    pt_v = load_per_track_av("val")
    pt_t = load_per_track_av("test")
    val_df  = val_df.merge(pt_v, on="audio_path", how="left")
    test_df = test_df.merge(pt_t, on="audio_path", how="left")
    val_df["us3_per_track_prob"]  = val_df["us3_per_track_prob"].fillna(0.5)
    test_df["us3_per_track_prob"] = test_df["us3_per_track_prob"].fillna(0.5)
    print(f"  US3-per-track joined: val mean={val_df['us3_per_track_prob'].mean():.3f}  "
          f"test mean={test_df['us3_per_track_prob'].mean():.3f}")

    val_y  = val_df["label"].to_numpy(dtype=int)
    test_y = test_df["label"].to_numpy(dtype=int)
    print(f"\nval n={len(val_df)}  test n={len(test_df)}  audio_systems={len(SCORE_FEATS)}\n")

    rows = []
    for name, fn in VARIANTS:
        val_p, test_p, extra = fn(val_df, test_df, val_y, test_y)
        m = save_variant(name, val_p, val_y, test_p, test_y, test_df, extra,
                         out_root=OUT_ROOT)
        rows.append({
            "variant": name, "F1": round(m["f1"], 4),
            "AUROC": round(m["auroc"], 4), "AUPRC": round(m["auprc"], 4),
            "threshold": round(m["threshold"], 3),
            "delta_F1": m["delta_f1"], "delta_AUROC": m["delta_auroc"],
        })

    leaderboard = pd.DataFrame(rows).sort_values("AUROC", ascending=False)
    leaderboard.to_csv(OUT_ROOT / "leaderboard.csv", index=False)
    print("\n=== AV LEADERBOARD ===")
    print(leaderboard.to_string(index=False))
    print(f"\nReference (audio-only winner): per_child_offset  F1=0.9041 AUROC=0.9002 AUPRC=0.9626")
    print(f"Reference (with metadata):     12-sys metadata stacker  F1=0.9053 AUROC=0.9044 AUPRC=0.9663")
    print(f"Reference (project ceiling):   12-sys + visual stacker  F1=0.8977 AUROC=0.9052 AUPRC=0.9677")


if __name__ == "__main__":
    main()
