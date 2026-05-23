"""No-BIDS-metadata ensembles — prototype for §22.x discussion.

Two variants over the same 12-system base set:
  --variant pure        12 system probs + timepoint only.
  --variant audio_cov   12 system probs + timepoint + audio-derived covariates.

Audio covariates (all computable from artifacts already in the repo):
  - n_speakers_{diar}    : unique SPEAKER labels in cached RTTMs from
                           eend_eda / sortformer / vbx / vtc / pyannote
  - mean_n_speakers      : mean across the 5 above
  - max_n_speakers       : max across the 5 above
  - silero_speech_score  : silero VAD baseline prob (proxy for speech rate)
  - vad_energy_score     : energy VAD baseline prob (proxy for non-silence)
  - system_disagreement  : std-dev of the 12 system probs for the clip

Outputs (parallel layout to ensemble_runs/metadata_stack/):
  ensemble_runs/no_metadata_stack/         (variant=pure)
  ensemble_runs/no_metadata_stack_audio/   (variant=audio_cov)

Each contains: test_metrics_tuned.json, val_metrics_tuned.json, test_predictions.csv,
feature_importances.json, config.json.

Usage:
  python evaluation/no_metadata_ensemble.py --variant pure
  python evaluation/no_metadata_ensemble.py --variant audio_cov
  python evaluation/no_metadata_ensemble.py --variant all
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from evaluation.metadata_router import (
    _SYSTEM_PATHS,
    SCORE_FEATS,
    load_system_scores,
    compute_metrics,
    tune_threshold,
)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier

BASELINE_F1, BASELINE_AUROC, SEED = 0.893, 0.878, 42
MASTER_CSV = _REPO / "whisper-modeling/seen_child_splits/master_with_split.csv"

# ── RTTM-derived n_speakers ─────────────────────────────────────────────────

DIARIZER_RTTM_CACHES = {
    "eend_eda":   _REPO / "pyannote/eend_eda_rttm_cache",
    "sortformer": _REPO / "pyannote/sortformer_rttm_cache",
    "vbx":        _REPO / "pyannote/vbx_rttm_cache",
    "vtc":        _REPO / "pyannote/vtc_rttm_cache",
    "pyannote":   _REPO / "pyannote/pyannote_rttm_cache",
}


def _rttm_cache_path(cache_dir: Path, audio_path: str) -> Path:
    """Reproduce pyannote/unified.py's cache convention: {stem}__{md5}.rttm."""
    stem = Path(audio_path).stem
    md5 = hashlib.md5(audio_path.encode("utf-8")).hexdigest()
    # Most caches use 12-char prefix, some use full md5 — try both
    for n in (12, 32):
        p = cache_dir / f"{stem}__{md5[:n]}.rttm"
        if p.exists():
            return p
    return cache_dir / f"{stem}__{md5[:12]}.rttm"  # default


def _count_unique_speakers(rttm_path: Path) -> int:
    if not rttm_path.exists():
        return 0
    speakers = set()
    try:
        with open(rttm_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 9 and parts[0] == "SPEAKER":
                    speakers.add(parts[7])
    except Exception:
        return 0
    return len(speakers)


def build_n_speakers(audio_paths: list[str]) -> pd.DataFrame:
    rows = []
    for ap in audio_paths:
        row = {"audio_path": ap}
        counts = []
        for name, cdir in DIARIZER_RTTM_CACHES.items():
            n = _count_unique_speakers(_rttm_cache_path(cdir, ap))
            row[f"n_speakers_{name}"] = n
            if n > 0:
                counts.append(n)
        row["mean_n_speakers"] = float(np.mean(counts)) if counts else 0.0
        row["max_n_speakers"]  = float(np.max(counts))  if counts else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


# ── VAD covariates from baseline prediction CSVs ───────────────────────────

VAD_PRED_PATHS = {
    "silero_speech_score": _REPO / "baselines/vad_baseline_runs/silero/{split}_predictions.csv",
    "vad_energy_score":    _REPO / "baselines/vad_baseline_runs/energy/{split}_predictions.csv",
}


def load_vad_covariates(split: str) -> pd.DataFrame:
    out = None
    for name, tmpl in VAD_PRED_PATHS.items():
        p = Path(str(tmpl).replace("{split}", split))
        if not p.exists():
            print(f"  [warn] {name}: {p} missing — filling 0.5")
            continue
        df = pd.read_csv(p)
        prob_col = next((c for c in ["prob", "score"] if c in df.columns), None)
        if prob_col is None:
            continue
        d = df[["audio_path", prob_col]].rename(columns={prob_col: name})
        out = d if out is None else out.merge(d, on="audio_path", how="outer")
    return out if out is not None else pd.DataFrame(columns=["audio_path"])


# ── Label / split source ────────────────────────────────────────────────────

def load_labels_split() -> pd.DataFrame:
    df = pd.read_csv(MASTER_CSV)
    df["timepoint_is_36m"] = (df["timepoint_norm"] == "36_month").astype(int)
    return df[["audio_path", "split", "label", "timepoint_norm",
               "timepoint_is_36m"]].reset_index(drop=True)


# ── Build feature matrix per variant ────────────────────────────────────────

AUDIO_COV_FEATS = [
    "n_speakers_eend_eda", "n_speakers_sortformer", "n_speakers_vbx",
    "n_speakers_vtc", "n_speakers_pyannote",
    "mean_n_speakers", "max_n_speakers",
    "silero_speech_score", "vad_energy_score",
    "system_disagreement",
]


def assemble(split: str, scores: pd.DataFrame, labels: pd.DataFrame,
             n_spk: pd.DataFrame | None, vad: pd.DataFrame | None) -> pd.DataFrame:
    base = labels[labels["split"] == split].merge(scores, on="audio_path", how="left")
    base[SCORE_FEATS] = base[SCORE_FEATS].fillna(0.5)
    if n_spk is not None:
        base = base.merge(n_spk, on="audio_path", how="left")
    if vad is not None:
        base = base.merge(vad, on="audio_path", how="left")
    # System-disagreement (std across the 12 probs, computed regardless of variant)
    base["system_disagreement"] = base[SCORE_FEATS].std(axis=1)
    # Fill audio cov NaNs with 0
    for c in AUDIO_COV_FEATS:
        if c in base.columns:
            base[c] = base[c].fillna(0.0)
    return base.reset_index(drop=True)


def feature_list(variant: str) -> list[str]:
    base = list(SCORE_FEATS) + ["timepoint_is_36m"]
    if variant == "audio_cov":
        base += AUDIO_COV_FEATS
    return base


# ── Train + evaluate ────────────────────────────────────────────────────────

def run_variant(variant: str, val_df: pd.DataFrame, test_df: pd.DataFrame,
                out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    feats = feature_list(variant)
    X_val  = val_df[feats].fillna(0.0).to_numpy(dtype=float)
    X_test = test_df[feats].fillna(0.0).to_numpy(dtype=float)
    y_val  = val_df["label"].to_numpy(dtype=int)
    y_test = test_df["label"].to_numpy(dtype=int)

    print(f"\n=== variant={variant!r}  n_features={len(feats)} ===")
    results = {}
    for name, clf in [
        ("lr",  LogisticRegression(C=1.0, max_iter=500, random_state=SEED)),
        ("gbm", HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05,
                                               max_leaf_nodes=15, min_samples_leaf=20,
                                               random_state=SEED)),
    ]:
        clf.fit(X_val, y_val)
        v_prob = clf.predict_proba(X_val)[:, 1]
        t = tune_threshold(y_val, v_prob)
        v_m = compute_metrics(y_val, v_prob, threshold=t); v_m["threshold"] = t
        t_prob = clf.predict_proba(X_test)[:, 1]
        t_m = compute_metrics(y_test, t_prob, threshold=t); t_m["threshold"] = t
        results[name] = {"val_f1": v_m["f1"], "val_m": v_m, "test_m": t_m,
                         "test_prob": t_prob, "clf": clf}
        print(f"  {name}: val_F1={v_m['f1']:.4f}  test_F1={t_m['f1']:.4f}  "
              f"test_AUROC={t_m['auroc']:.4f}  test_AUPRC={t_m['auprc']:.4f}")

    best_name = max(results, key=lambda k: results[k]["val_f1"]
                    if results[k]["val_f1"] < 0.99 else 0.0)
    best = results[best_name]

    # Persist
    test_m = dict(best["test_m"])
    test_m["baseline_f1"] = BASELINE_F1
    test_m["baseline_auroc"] = BASELINE_AUROC
    test_m["delta_f1"]    = round(test_m["f1"] - BASELINE_F1, 4)
    test_m["delta_auroc"] = round(test_m["auroc"] - BASELINE_AUROC, 4)
    test_m["n"] = int(len(test_df))

    with open(out_dir / "test_metrics_tuned.json", "w") as f:
        json.dump(test_m, f, indent=2)
    with open(out_dir / "val_metrics_tuned.json", "w") as f:
        json.dump(best["val_m"], f, indent=2)

    preds = test_df[["audio_path", "label", "timepoint_norm"]].copy()
    preds["score"] = best["test_prob"]
    preds["prediction"] = (best["test_prob"] >= test_m["threshold"]).astype(int)
    preds.to_csv(out_dir / "test_predictions.csv", index=False)

    importances = {
        "lr_coefficients": dict(zip(feats, results["lr"]["clf"].coef_[0].tolist())),
    }
    with open(out_dir / "feature_importances.json", "w") as f:
        json.dump(importances, f, indent=2)

    cfg = {
        "variant": variant,
        "model_type": best_name,
        "features": feats,
        "score_features": list(SCORE_FEATS),
        "audio_covariates": AUDIO_COV_FEATS if variant == "audio_cov" else [],
        "seed": SEED, "n_val": int(len(val_df)), "n_test": int(len(test_df)),
    }
    with open(out_dir / "config.json", "w") as f:
        json.dump(cfg, f, indent=2)

    print(f"  → {out_dir}  best={best_name}  test F1={test_m['f1']:.4f}  "
          f"AUROC={test_m['auroc']:.4f}  AUPRC={test_m['auprc']:.4f}  "
          f"ΔF1={test_m['delta_f1']:+.4f}  ΔAUROC={test_m['delta_auroc']:+.4f}")
    return test_m


# ── Main ───────────────────────────────────────────────────────────────────

VARIANT_DIRS = {
    "pure":       _REPO / "ensemble_runs/no_metadata_stack",
    "audio_cov":  _REPO / "ensemble_runs/no_metadata_stack_audio",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["pure", "audio_cov", "all"], default="all")
    args = ap.parse_args()
    variants = ["pure", "audio_cov"] if args.variant == "all" else [args.variant]

    # Load common artifacts once
    val_scores  = load_system_scores("val")
    test_scores = load_system_scores("test")
    labels      = load_labels_split()

    # Audio covariates only needed for audio_cov; cheap to compute always
    print("Building n_speakers covariates from RTTM caches …")
    all_paths = sorted(set(val_scores["audio_path"]) | set(test_scores["audio_path"]))
    n_spk = build_n_speakers(all_paths)
    val_vad  = load_vad_covariates("val")
    test_vad = load_vad_covariates("test")

    val_df  = assemble("val",  val_scores,  labels, n_spk, val_vad)
    test_df = assemble("test", test_scores, labels, n_spk, test_vad)
    print(f"  val n={len(val_df)}  test n={len(test_df)}  "
          f"system_disagreement test mean={test_df['system_disagreement'].mean():.3f}")

    # Quick sanity print of n_speakers stats
    print(f"  n_speakers (test): "
          f"eend_eda mean={test_df['n_speakers_eend_eda'].mean():.2f}  "
          f"sortformer mean={test_df['n_speakers_sortformer'].mean():.2f}  "
          f"vbx mean={test_df['n_speakers_vbx'].mean():.2f}  "
          f"max(any) mean={test_df['max_n_speakers'].mean():.2f}")

    summary = {}
    for v in variants:
        summary[v] = run_variant(v, val_df, test_df, VARIANT_DIRS[v])

    print("\n=== SUMMARY ===")
    for v, m in summary.items():
        print(f"  {v:11s}  F1={m['f1']:.4f}  AUROC={m['auroc']:.4f}  "
              f"AUPRC={m['auprc']:.4f}  ΔF1={m['delta_f1']:+.4f}  "
              f"ΔAUROC={m['delta_auroc']:+.4f}")
    print(f"\nReference: best_audio_mil mean ensemble  F1=0.8930  AUROC=0.8780")
    print(f"Reference: 12-sys metadata stacker        F1=0.9053  AUROC=0.9044")


if __name__ == "__main__":
    main()
