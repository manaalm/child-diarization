"""
Late-fusion ensemble across all trained systems.

Trains a logistic regression meta-stacker on val-set predictions from
multiple systems, evaluates on the held-out test set.

Also reports simple mean-ensemble and per-system AUROCs for comparison.

Systems included (by default):
  - BabAR ECAPA enrollment
  - VTC ECAPA enrollment
  - VBx ECAPA enrollment
  - USC-SAIL ECAPA enrollment
  - Pyannote ECAPA enrollment
  - MIL: babar_vtc gated_attention (best MIL config)
  - MIL: vbx_max (second-best MIL config)
  - VTC combined features (diarizer+embedding, best if vtc_combined_runs/ exists)
  - BabAR combined features (best per-timepoint LR if babar_combined_runs/ exists)

Usage:
    cd /orcd/scratch/orcd/008/manaal/child-adult-diarization/pyannote
    python ensemble_combined.py
    python ensemble_combined.py --results-dir ensemble_runs/ --subsets best3 all

Output (in --results-dir):
    ensemble_results.json   — AUROC/F1/AUPRC for every combination
    test_predictions.csv    — test probabilities for all ensembles
    val_predictions.csv     — val probabilities for all ensembles
"""

import argparse
import json
import os
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

# ---------------------------------------------------------------------------
# Paths (relative to repo root, one directory up from pyannote/)
# ---------------------------------------------------------------------------

ROOT = str(Path(__file__).resolve().parent.parent)


def _rp(*parts):
    return os.path.join(ROOT, *parts)


SYSTEM_DEFS = {
    "babar_enroll": {
        "val":  _rp("babar_ecapa_enrollment_runs/enroll_val_predictions.csv"),
        "test": _rp("babar_ecapa_enrollment_runs/enroll_test_predictions.csv"),
        "prob_col": "prob",
    },
    "vtc_enroll": {
        "val":  _rp("vtc_ecapa_enrollment_runs/enroll_val_predictions.csv"),
        "test": _rp("vtc_ecapa_enrollment_runs/enroll_test_predictions.csv"),
        "prob_col": "prob",
    },
    "vbx_enroll": {
        "val":  _rp("vbx_ecapa_enrollment_runs/enroll_val_predictions.csv"),
        "test": _rp("vbx_ecapa_enrollment_runs/enroll_test_predictions.csv"),
        "prob_col": "prob",
    },
    "usc_sail_enroll": {
        "val":  _rp("whisper-modeling/usc_sail_enrollment_runs/enroll_val_predictions.csv"),
        "test": _rp("whisper-modeling/usc_sail_enrollment_runs/enroll_test_predictions.csv"),
        "prob_col": "prob",
    },
    "pyannote_enroll": {
        "val":  _rp("pyannote/pyannote_enrollment_runs/val_predictions.csv"),
        "test": _rp("pyannote/pyannote_enrollment_runs/test_predictions.csv"),
        "prob_col": "prob",
    },
    "mil_babar_gated": {
        "val":  _rp("mil/mil_results/seg_mil/babar_vtc_gated_attention/val_predictions.csv"),
        "test": _rp("mil/mil_results/seg_mil/babar_vtc_gated_attention/test_predictions.csv"),
        "prob_col": "prob",
    },
    "mil_vbx_max": {
        "val":  _rp("mil/mil_results/seg_mil/vbx_max/val_predictions.csv"),
        "test": _rp("mil/mil_results/seg_mil/vbx_max/test_predictions.csv"),
        "prob_col": "prob",
    },
    "vtc_combined": {
        "val":  _rp("vtc_combined_runs/logistic_diarizer_plus_embedding_val_predictions.csv"),
        "test": _rp("vtc_combined_runs/logistic_diarizer_plus_embedding_test_predictions.csv"),
        "prob_col": "prob",
    },
    "babar_combined": {
        "val":  _rp("babar_combined_runs/lr_diarizer_plus_phoneme_val_predictions.csv"),
        "test": _rp("babar_combined_runs/lr_diarizer_plus_phoneme_test_predictions.csv"),
        "prob_col": "prob",
    },
    "eend_eda_enroll": {
        "val":  _rp("eend_eda_ecapa_enrollment_runs/enroll_val_predictions.csv"),
        "test": _rp("eend_eda_ecapa_enrollment_runs/enroll_test_predictions.csv"),
        "prob_col": "prob",
    },
    "sortformer_enroll": {
        "val":  _rp("sortformer_ecapa_enrollment_runs/enroll_val_predictions.csv"),
        "test": _rp("sortformer_ecapa_enrollment_runs/enroll_test_predictions.csv"),
        "prob_col": "prob",
    },
}

# Curated subsets to report
ENSEMBLE_SUBSETS = {
    "best3": ["babar_enroll", "vtc_enroll", "mil_babar_gated"],
    "top4_enroll": ["babar_enroll", "vtc_enroll", "vbx_enroll", "usc_sail_enroll"],
    "babar_mil": ["babar_enroll", "mil_babar_gated"],
    "vtc_mil": ["vtc_enroll", "mil_babar_gated"],
    "best_audio_mil": ["babar_enroll", "vtc_enroll", "mil_babar_gated", "mil_vbx_max"],
    "with_vtc_combined": ["babar_enroll", "vtc_combined", "mil_babar_gated"],
    "with_babar_combined": ["babar_combined", "vtc_enroll", "mil_babar_gated"],
    "neural_diar": ["eend_eda_enroll", "sortformer_enroll", "mil_babar_gated"],
    "with_eend_eda": ["babar_enroll", "eend_eda_enroll", "mil_babar_gated"],
    "with_sortformer": ["babar_enroll", "sortformer_enroll", "mil_babar_gated"],
    "all_available": None,  # filled at runtime with all loaded systems
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=lambda x: float(x) if isinstance(x, (np.floating, np.integer)) else x)


def compute_metrics(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    m = {
        "f1":       float(f1_score(y_true, y_pred, zero_division=0)),
        "auroc":    float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else float("nan"),
        "auprc":    float(average_precision_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else float("nan"),
        "threshold": float(threshold),
        "n": int(len(y_true)),
    }
    return m


def tune_threshold(y_true, y_prob):
    best_t, best_f1 = 0.5, -1.0
    for t in np.linspace(0.05, 0.95, 91):
        f = float(f1_score(y_true, (y_prob >= t).astype(int), zero_division=0))
        if f > best_f1:
            best_f1, best_t = f, float(t)
    return best_t


def per_timepoint_auroc(df, prob_col):
    rows = []
    for tp, sub in df.groupby("timepoint_norm"):
        if len(sub["label"].unique()) < 2:
            continue
        rows.append({
            "timepoint": tp,
            "auroc": float(roc_auc_score(sub["label"].values, sub[prob_col].values)),
            "n": len(sub),
        })
    return rows


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_systems(system_defs):
    """Load val and test prediction DataFrames for each available system."""
    loaded = {}
    for name, cfg in system_defs.items():
        val_path, test_path = cfg["val"], cfg["test"]
        if not os.path.exists(val_path):
            print(f"  [skip] {name}: val predictions missing ({val_path})")
            continue
        if not os.path.exists(test_path):
            print(f"  [skip] {name}: test predictions missing ({test_path})")
            continue
        val_df  = pd.read_csv(val_path)[["audio_path", "label", cfg["prob_col"]]].rename(columns={cfg["prob_col"]: name})
        test_df = pd.read_csv(test_path)[["audio_path", "label", cfg["prob_col"]]].rename(columns={cfg["prob_col"]: name})
        loaded[name] = {"val": val_df, "test": test_df}
        print(f"  [ok]   {name}")
    return loaded


def build_matrix(loaded, system_names, split):
    """Join predictions from multiple systems on audio_path."""
    dfs = [loaded[n][split] for n in system_names if n in loaded]
    if not dfs:
        return None
    base = dfs[0][["audio_path", "label"]].copy()
    for df in dfs:
        name = [c for c in df.columns if c not in ("audio_path", "label")][0]
        base = base.merge(df[["audio_path", name]], on="audio_path", how="inner")
    return base


# ---------------------------------------------------------------------------
# Ensemble training
# ---------------------------------------------------------------------------

def run_ensemble(val_mat, test_mat, system_names, ensemble_name):
    """Train val-calibrated LR stacker, evaluate on test."""
    feat_cols = [c for c in system_names if c in val_mat.columns]
    if not feat_cols:
        return None

    X_val  = val_mat[feat_cols].fillna(0.5).values
    y_val  = val_mat["label"].values
    X_test = test_mat[feat_cols].fillna(0.5).values
    y_test = test_mat["label"].values

    # Individual system AUROCs on test
    individual = {}
    for col in feat_cols:
        if col in test_mat.columns:
            individual[col] = float(roc_auc_score(y_test, test_mat[col].values))

    # Simple mean ensemble
    mean_val  = val_mat[feat_cols].fillna(0.5).mean(axis=1).values
    mean_test = test_mat[feat_cols].fillna(0.5).mean(axis=1).values
    mean_t    = tune_threshold(y_val, mean_val)
    mean_metrics = compute_metrics(y_test, mean_test, mean_t)

    # LR meta-stacker (trained on val, evaluated on test)
    lr = LogisticRegression(C=0.1, max_iter=1000, solver="lbfgs")
    lr.fit(X_val, y_val)
    lr_val_proba  = lr.predict_proba(X_val)[:, 1]
    lr_test_proba = lr.predict_proba(X_test)[:, 1]
    lr_t = tune_threshold(y_val, lr_val_proba)
    lr_metrics = compute_metrics(y_test, lr_test_proba, lr_t)

    return {
        "ensemble_name": ensemble_name,
        "systems": feat_cols,
        "n_systems": len(feat_cols),
        "individual_test_auroc": individual,
        "mean_ensemble": mean_metrics,
        "lr_stack": lr_metrics,
        "lr_weights": dict(zip(feat_cols, lr.coef_[0].tolist())),
        "lr_test_proba": lr_test_proba.tolist(),
        "mean_test_proba": mean_test.tolist(),
        # spec-022 polish: also expose val-side probabilities so balanced-
        # accuracy tuning (or any val-tuned recalibration) can be done
        # without re-running the ensemble pipeline.
        "lr_val_proba": lr_val_proba.tolist(),
        "mean_val_proba": mean_val.tolist(),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default=_rp("ensemble_runs"))
    args = parser.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)

    print("Loading system predictions...")
    loaded = load_systems(SYSTEM_DEFS)
    available = list(loaded.keys())
    print(f"\nLoaded {len(available)} systems: {available}")

    if len(available) < 2:
        print("ERROR: need at least 2 systems to ensemble")
        return

    # Fill "all_available" subset
    ENSEMBLE_SUBSETS["all_available"] = available

    all_results = {}
    test_proba_cols = {}  # name → array for test predictions CSV
    val_proba_cols  = {}  # name → array for val  predictions CSV (spec-022)

    print("\n" + "=" * 60)
    print("ENSEMBLE RESULTS")
    print("=" * 60)

    for subset_name, system_list in ENSEMBLE_SUBSETS.items():
        # Filter to only available systems
        systems = [s for s in (system_list or available) if s in loaded]
        if len(systems) < 2:
            print(f"\n[{subset_name}] skipped — fewer than 2 systems available: {systems}")
            continue

        val_mat  = build_matrix(loaded, systems, "val")
        test_mat = build_matrix(loaded, systems, "test")
        if val_mat is None or test_mat is None or len(val_mat) == 0:
            continue

        result = run_ensemble(val_mat, test_mat, systems, subset_name)
        if result is None:
            continue

        all_results[subset_name] = {
            "systems": result["systems"],
            "individual_test_auroc": result["individual_test_auroc"],
            "mean_ensemble": result["mean_ensemble"],
            "lr_stack": result["lr_stack"],
            "lr_weights": result["lr_weights"],
        }

        # Store proba for predictions CSV
        test_proba_cols[f"{subset_name}_mean"] = result["mean_test_proba"]
        test_proba_cols[f"{subset_name}_lr"]   = result["lr_test_proba"]
        val_proba_cols[f"{subset_name}_mean"]  = result["mean_val_proba"]
        val_proba_cols[f"{subset_name}_lr"]    = result["lr_val_proba"]

        print(f"\n[{subset_name}] {result['systems']}")
        for sname, sauc in result["individual_test_auroc"].items():
            print(f"  {sname}: AUROC={sauc:.3f}")
        m = result["mean_ensemble"]
        print(f"  Mean ensemble: AUROC={m['auroc']:.3f}  F1={m['f1']:.3f}  AUPRC={m['auprc']:.3f}")
        m = result["lr_stack"]
        print(f"  LR stack:      AUROC={m['auroc']:.3f}  F1={m['f1']:.3f}  AUPRC={m['auprc']:.3f}")
        print(f"  LR weights: {result['lr_weights']}")

    # Save results JSON
    save_json(all_results, os.path.join(args.results_dir, "ensemble_results.json"))

    # Build test predictions CSV
    # Use the first available system's test CSV as the base
    base_sys = available[0]
    test_base = loaded[base_sys]["test"][["audio_path", "label"]].copy()
    # add timepoint_norm if available
    test_full_path = _rp("babar_ecapa_enrollment_runs/enroll_test_predictions.csv")
    if os.path.exists(test_full_path):
        tp_df = pd.read_csv(test_full_path)[["audio_path", "timepoint_norm"]]
        test_base = test_base.merge(tp_df, on="audio_path", how="left")

    for col_name, proba_arr in test_proba_cols.items():
        if len(proba_arr) == len(test_base):
            test_base[col_name] = proba_arr

    test_base.to_csv(os.path.join(args.results_dir, "test_predictions.csv"), index=False)

    # spec-022 polish: write val_predictions.csv with the same multi-column
    # structure so per-candidate balanced-accuracy tuning is reproducible.
    val_base = loaded[base_sys]["val"][["audio_path", "label"]].copy()
    val_full_path = _rp("babar_ecapa_enrollment_runs/enroll_val_predictions.csv")
    if os.path.exists(val_full_path):
        tp_df = pd.read_csv(val_full_path)[["audio_path", "timepoint_norm"]]
        val_base = val_base.merge(tp_df, on="audio_path", how="left")
    for col_name, proba_arr in val_proba_cols.items():
        if len(proba_arr) == len(val_base):
            val_base[col_name] = proba_arr
    val_base.to_csv(os.path.join(args.results_dir, "val_predictions.csv"), index=False)

    print(f"\nResults saved to {args.results_dir}")

    # Final summary table
    print("\n" + "=" * 60)
    print("SUMMARY — LR Stack AUROC by ensemble subset")
    print("=" * 60)
    print(f"{'Subset':<35} {'N systems':>9} {'Mean AUROC':>11} {'LR AUROC':>9}")
    print("-" * 65)
    for name, r in sorted(all_results.items(), key=lambda x: x[1]["lr_stack"]["auroc"], reverse=True):
        print(
            f"{name:<35} {len(r['systems']):>9} "
            f"{r['mean_ensemble']['auroc']:>11.3f} "
            f"{r['lr_stack']['auroc']:>9.3f}"
        )


if __name__ == "__main__":
    main()
