"""Advanced ensemble prototypes — beat the no-metadata pure stacker.

Six variants over the same 12 base systems, no BIDS metadata beyond timepoint.
All fit on val, tune threshold on val, evaluate on test.

Variants:
  1. mean              — simple mean of 12 raw probs (sanity)
  2. calibrated_mean   — Platt-calibrate each system on val, then mean
  3. isotonic_weighted — isotonic-calibrate, weight by val_AUROC^2, sum
  4. rank_stacker      — LR on rank-transformed probs + timepoint
  5. per_timepoint     — separate LR per timepoint, merge by clip's age
  6. confidence_weighted — per-clip soft attention; weight = val_AUROC * (1 - entropy(prob))
  7. pair_disagreement — LR on probs + top-K (system_i - system_j) pair features (lasso-selected)
  +. pure (reference)  — 12 probs + timepoint_is_36m

Outputs to ensemble_runs/advanced/{variant}/ and a leaderboard CSV.

Usage:
  python evaluation/advanced_ensembles.py
"""

from __future__ import annotations

import json
import os
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from evaluation.metadata_router import (
    SCORE_FEATS as _BASE_SCORE_FEATS,
    load_system_scores as _load_base_scores,
    compute_metrics,
    tune_threshold,
)

from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.isotonic import IsotonicRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from scipy.stats import rankdata


# Extend base systems with one specialist that has val predictions:
# pseudo_frame_wavlm — frame-level WavLM distill (AUROC=0.831 standalone).
# Independent error pattern from the other 12 (frame-level vs clip-level training).
EXTRA_SYSTEM_PATHS = {
    "pseudo_frame_wavlm": ("pseudo_frame/results/wavlm_pseudo_frame/{split}_predictions.csv", "score"),
}
# Build extended SCORE_FEATS in module scope so all variants see them
SCORE_FEATS = list(_BASE_SCORE_FEATS) + [f"{n}_prob" for n in EXTRA_SYSTEM_PATHS]


def load_system_scores(split: str) -> pd.DataFrame:
    """Wraps metadata_router.load_system_scores and joins the extra specialist systems."""
    df = _load_base_scores(split)
    for name, (tmpl, col) in EXTRA_SYSTEM_PATHS.items():
        path = _REPO / tmpl.format(split=split)
        if not path.exists():
            print(f"  WARNING: {name} {split} predictions missing: {path}", flush=True)
            df[f"{name}_prob"] = 0.5
            continue
        e = pd.read_csv(path)[["audio_path", col]].rename(columns={col: f"{name}_prob"})
        df = df.merge(e, on="audio_path", how="outer")
    prob_cols = [c for c in df.columns if c.endswith("_prob")]
    df[prob_cols] = df[prob_cols].fillna(0.5)
    return df.reset_index(drop=True)

BASELINE_F1, BASELINE_AUROC, SEED = 0.893, 0.878, 42
MASTER_CSV = _REPO / "whisper-modeling/seen_child_splits/master_with_split.csv"
OUT_ROOT = _REPO / "ensemble_runs/advanced"
OUT_ROOT.mkdir(parents=True, exist_ok=True)


def load_labels_split() -> pd.DataFrame:
    df = pd.read_csv(MASTER_CSV)
    df["timepoint_is_36m"] = (df["timepoint_norm"] == "36_month").astype(int)
    return df[["audio_path", "split", "label", "timepoint_norm",
               "timepoint_is_36m"]].reset_index(drop=True)


def assemble(split: str, scores: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    base = labels[labels["split"] == split].merge(scores, on="audio_path", how="left")
    base[SCORE_FEATS] = base[SCORE_FEATS].fillna(0.5)
    return base.reset_index(drop=True)


# ── Calibration helpers ────────────────────────────────────────────────────

def platt_calibrate(val_prob: np.ndarray, val_y: np.ndarray, target_prob: np.ndarray) -> np.ndarray:
    """Sigmoid (Platt) calibration. Fits LR(prob → label) on val, applies to target."""
    lr = LogisticRegression(C=1e3, max_iter=500, random_state=SEED)
    lr.fit(val_prob.reshape(-1, 1), val_y)
    return lr.predict_proba(target_prob.reshape(-1, 1))[:, 1]


def isotonic_calibrate(val_prob: np.ndarray, val_y: np.ndarray, target_prob: np.ndarray) -> np.ndarray:
    """Non-parametric isotonic calibration."""
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(val_prob, val_y)
    return iso.predict(target_prob)


def normalize_probs(arr: np.ndarray) -> np.ndarray:
    """Rank-transform within each column to [0, 1]."""
    n = arr.shape[0]
    return np.stack([rankdata(arr[:, j], method="average") / n for j in range(arr.shape[1])], axis=1)


def entropy(p: np.ndarray) -> np.ndarray:
    """Binary entropy in [0, 1] (max=1 at p=0.5)."""
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return -p * np.log2(p) - (1 - p) * np.log2(1 - p)


# ── Persist + summarize ────────────────────────────────────────────────────

def save_variant(name: str, val_prob: np.ndarray, val_y: np.ndarray,
                 test_prob: np.ndarray, test_y: np.ndarray,
                 test_df: pd.DataFrame, cfg_extra: dict,
                 out_root: Path | None = None,
                 val_df: pd.DataFrame | None = None) -> dict:
    root = out_root if out_root is not None else OUT_ROOT
    out_dir = root / name
    out_dir.mkdir(parents=True, exist_ok=True)
    t = tune_threshold(val_y, val_prob)
    val_m = compute_metrics(val_y, val_prob, threshold=t); val_m["threshold"] = t
    test_m = compute_metrics(test_y, test_prob, threshold=t); test_m["threshold"] = t
    test_m["baseline_f1"] = BASELINE_F1
    test_m["baseline_auroc"] = BASELINE_AUROC
    test_m["delta_f1"] = round(test_m["f1"] - BASELINE_F1, 4)
    test_m["delta_auroc"] = round(test_m["auroc"] - BASELINE_AUROC, 4)
    test_m["n"] = int(len(test_df))
    with open(out_dir / "test_metrics_tuned.json", "w") as f:
        json.dump(test_m, f, indent=2)
    with open(out_dir / "val_metrics_tuned.json", "w") as f:
        json.dump(val_m, f, indent=2)
    preds = test_df[["audio_path", "label", "timepoint_norm"]].copy()
    preds["score"] = test_prob
    preds["prediction"] = (test_prob >= t).astype(int)
    preds.to_csv(out_dir / "test_predictions.csv", index=False)
    # spec-022 polish: dump val_predictions.csv too so balanced-accuracy
    # threshold retuning (or any val-tuned recalibration) is reproducible
    # without rerunning the LR/GBM stacker pipeline.
    if val_df is not None:
        val_preds = val_df[["audio_path", "label", "timepoint_norm"]].copy()
        val_preds["score"] = val_prob
        val_preds["prediction"] = (val_prob >= t).astype(int)
        val_preds.to_csv(out_dir / "val_predictions.csv", index=False)
    cfg = {"variant": name, "seed": SEED, **cfg_extra}
    with open(out_dir / "config.json", "w") as f:
        json.dump(cfg, f, indent=2)
    return test_m


# ── Variants ────────────────────────────────────────────────────────────────

def variant_pure(val_df, test_df, val_y, test_y):
    """Pure stacker: 12 probs + timepoint. Reference baseline."""
    feats = list(SCORE_FEATS) + ["timepoint_is_36m"]
    X_val = val_df[feats].to_numpy(dtype=float)
    X_test = test_df[feats].to_numpy(dtype=float)
    clf = LogisticRegression(C=1.0, max_iter=500, random_state=SEED)
    clf.fit(X_val, val_y)
    return clf.predict_proba(X_val)[:, 1], clf.predict_proba(X_test)[:, 1], {"features": feats}


def variant_mean(val_df, test_df, val_y, test_y):
    """Naive mean of 12 raw probs."""
    P_val = val_df[SCORE_FEATS].to_numpy(dtype=float)
    P_test = test_df[SCORE_FEATS].to_numpy(dtype=float)
    return P_val.mean(axis=1), P_test.mean(axis=1), {}


def variant_calibrated_mean(val_df, test_df, val_y, test_y):
    """Platt-calibrate each system on val, then mean."""
    P_val = val_df[SCORE_FEATS].to_numpy(dtype=float)
    P_test = test_df[SCORE_FEATS].to_numpy(dtype=float)
    cal_val = np.zeros_like(P_val)
    cal_test = np.zeros_like(P_test)
    for j in range(len(SCORE_FEATS)):
        cal_val[:, j]  = platt_calibrate(P_val[:, j], val_y, P_val[:, j])
        cal_test[:, j] = platt_calibrate(P_val[:, j], val_y, P_test[:, j])
    return cal_val.mean(axis=1), cal_test.mean(axis=1), {"calibration": "platt"}


def variant_isotonic_weighted(val_df, test_df, val_y, test_y):
    """Isotonic-calibrate, weight by val_AUROC^2, sum."""
    P_val = val_df[SCORE_FEATS].to_numpy(dtype=float)
    P_test = test_df[SCORE_FEATS].to_numpy(dtype=float)
    cal_val = np.zeros_like(P_val); cal_test = np.zeros_like(P_test)
    aurocs = np.zeros(len(SCORE_FEATS))
    for j in range(len(SCORE_FEATS)):
        cal_val[:, j]  = isotonic_calibrate(P_val[:, j], val_y, P_val[:, j])
        cal_test[:, j] = isotonic_calibrate(P_val[:, j], val_y, P_test[:, j])
        try:
            aurocs[j] = max(roc_auc_score(val_y, P_val[:, j]), 0.5)
        except Exception:
            aurocs[j] = 0.5
    weights = (aurocs - 0.5) ** 2  # quadratic emphasis on above-chance systems
    weights /= weights.sum()
    val_prob  = (cal_val  * weights).sum(axis=1)
    test_prob = (cal_test * weights).sum(axis=1)
    return val_prob, test_prob, {"calibration": "isotonic",
                                  "system_weights": dict(zip(SCORE_FEATS, weights.round(4).tolist())),
                                  "system_val_auroc": dict(zip(SCORE_FEATS, aurocs.round(4).tolist()))}


def variant_rank_stacker(val_df, test_df, val_y, test_y):
    """LR on rank-transformed probs + timepoint."""
    P_val = val_df[SCORE_FEATS].to_numpy(dtype=float)
    P_test = test_df[SCORE_FEATS].to_numpy(dtype=float)
    # Rank within val; for test, rank within test (preserves monotone calibration)
    R_val = normalize_probs(P_val)
    R_test = normalize_probs(P_test)
    feats = list(SCORE_FEATS) + ["timepoint_is_36m"]
    tp_val = val_df[["timepoint_is_36m"]].to_numpy(dtype=float)
    tp_test = test_df[["timepoint_is_36m"]].to_numpy(dtype=float)
    X_val = np.hstack([R_val, tp_val])
    X_test = np.hstack([R_test, tp_test])
    clf = LogisticRegression(C=1.0, max_iter=500, random_state=SEED)
    clf.fit(X_val, val_y)
    return clf.predict_proba(X_val)[:, 1], clf.predict_proba(X_test)[:, 1], {"features": feats}


def variant_per_timepoint(val_df, test_df, val_y, test_y):
    """Separate LR per timepoint, merge."""
    feats = list(SCORE_FEATS)
    val_prob = np.zeros(len(val_df))
    test_prob = np.zeros(len(test_df))
    importances = {}
    for tp_val, tp_str in [(0, "14mo"), (1, "36mo")]:
        v_mask = (val_df["timepoint_is_36m"] == tp_val).to_numpy()
        t_mask = (test_df["timepoint_is_36m"] == tp_val).to_numpy()
        if v_mask.sum() < 30:
            continue
        Xv = val_df.loc[v_mask, feats].to_numpy(dtype=float)
        yv = val_y[v_mask]
        Xt = test_df.loc[t_mask, feats].to_numpy(dtype=float)
        clf = LogisticRegression(C=1.0, max_iter=500, random_state=SEED)
        clf.fit(Xv, yv)
        val_prob[v_mask]  = clf.predict_proba(Xv)[:, 1]
        test_prob[t_mask] = clf.predict_proba(Xt)[:, 1]
        importances[tp_str] = {"n_val": int(v_mask.sum()), "n_test": int(t_mask.sum())}
    return val_prob, test_prob, {"features": feats, "per_tp": importances}


def variant_confidence_weighted(val_df, test_df, val_y, test_y):
    """Per-clip soft attention: weight = val_AUROC^2 * (1 - entropy(prob))."""
    P_val = val_df[SCORE_FEATS].to_numpy(dtype=float)
    P_test = test_df[SCORE_FEATS].to_numpy(dtype=float)
    aurocs = np.zeros(len(SCORE_FEATS))
    for j in range(len(SCORE_FEATS)):
        try:
            aurocs[j] = max(roc_auc_score(val_y, P_val[:, j]), 0.5)
        except Exception:
            aurocs[j] = 0.5
    base_w = (aurocs - 0.5) ** 2 + 1e-6  # global per-system weight
    # Per-clip confidence = 1 - entropy(prob); scale into [0,1]
    conf_val  = 1 - entropy(P_val)
    conf_test = 1 - entropy(P_test)
    # Combined weight: w_ij = base_w_j * conf_ij; normalize per-clip
    W_val  = base_w[None, :] * conf_val
    W_test = base_w[None, :] * conf_test
    W_val  /= W_val.sum(axis=1, keepdims=True)
    W_test /= W_test.sum(axis=1, keepdims=True)
    val_prob  = (P_val  * W_val ).sum(axis=1)
    test_prob = (P_test * W_test).sum(axis=1)
    return val_prob, test_prob, {"system_val_auroc": dict(zip(SCORE_FEATS, aurocs.round(4).tolist()))}


def variant_pair_disagreement(val_df, test_df, val_y, test_y, top_k: int = 8):
    """LR on 12 probs + top-K (system_i - system_j) pair features, lasso-selected."""
    P_val = val_df[SCORE_FEATS].to_numpy(dtype=float)
    P_test = test_df[SCORE_FEATS].to_numpy(dtype=float)
    pair_names, pair_vals_v, pair_vals_t = [], [], []
    for i, j in combinations(range(len(SCORE_FEATS)), 2):
        pair_names.append(f"{SCORE_FEATS[i]}_minus_{SCORE_FEATS[j]}")
        pair_vals_v.append(P_val[:, i] - P_val[:, j])
        pair_vals_t.append(P_test[:, i] - P_test[:, j])
    pair_v = np.stack(pair_vals_v, axis=1)
    pair_t = np.stack(pair_vals_t, axis=1)

    # Select top-K pairs by L1-regularized LR coefficient mass on val
    selector = LogisticRegression(penalty="l1", C=0.1, solver="liblinear",
                                  max_iter=2000, random_state=SEED)
    selector.fit(pair_v, val_y)
    coefs = np.abs(selector.coef_[0])
    top_idx = np.argsort(-coefs)[:top_k]
    selected_pairs = [pair_names[i] for i in top_idx]

    feats = list(SCORE_FEATS) + ["timepoint_is_36m"] + selected_pairs
    tp_v = val_df[["timepoint_is_36m"]].to_numpy(dtype=float)
    tp_t = test_df[["timepoint_is_36m"]].to_numpy(dtype=float)
    X_val = np.hstack([P_val,  tp_v, pair_v[:,  top_idx]])
    X_test = np.hstack([P_test, tp_t, pair_t[:, top_idx]])
    clf = LogisticRegression(C=1.0, max_iter=500, random_state=SEED)
    clf.fit(X_val, val_y)
    return clf.predict_proba(X_val)[:, 1], clf.predict_proba(X_test)[:, 1], {
        "features": feats, "selected_pairs": selected_pairs,
        "selected_pair_lasso_coefs": coefs[top_idx].round(4).tolist(),
    }


# ── Main ───────────────────────────────────────────────────────────────────

def variant_topk_systems(val_df, test_df, val_y, test_y, k: int = 6):
    """LR on the top-K base systems by val AUROC + timepoint. Sheds noisy systems."""
    P_val = val_df[SCORE_FEATS].to_numpy(dtype=float)
    P_test = test_df[SCORE_FEATS].to_numpy(dtype=float)
    aurocs = []
    for j in range(len(SCORE_FEATS)):
        try: aurocs.append(roc_auc_score(val_y, P_val[:, j]))
        except Exception: aurocs.append(0.5)
    top_idx = np.argsort(-np.array(aurocs))[:k]
    selected = [SCORE_FEATS[i] for i in top_idx]
    feats = selected + ["timepoint_is_36m"]
    tp_v = val_df[["timepoint_is_36m"]].to_numpy(dtype=float)
    tp_t = test_df[["timepoint_is_36m"]].to_numpy(dtype=float)
    X_val = np.hstack([P_val[:, top_idx], tp_v])
    X_test = np.hstack([P_test[:, top_idx], tp_t])
    clf = LogisticRegression(C=1.0, max_iter=500, random_state=SEED)
    clf.fit(X_val, val_y)
    return clf.predict_proba(X_val)[:, 1], clf.predict_proba(X_test)[:, 1], {
        "k": k, "features": feats,
        "selected_val_aurocs": [round(aurocs[i], 4) for i in top_idx],
    }


def variant_cv_stacked(val_df, test_df, val_y, test_y, n_splits: int = 5):
    """Proper stacked generalization: K-fold CV on val to get OOF predictions
    from each first-stage variant, train second-stage LR on OOF, predict on test.
    First-stage methods: pure, isotonic_weighted, rank_stacker, pair_disagreement.
    """
    first_stage_fns = [
        ("pure",              variant_pure),
        ("isotonic_weighted", variant_isotonic_weighted),
        ("rank_stacker",      variant_rank_stacker),
        ("pair_disagreement", variant_pair_disagreement),
    ]
    n_val, n_test = len(val_df), len(test_df)
    oof_val  = np.zeros((n_val,  len(first_stage_fns)))
    test_acc = np.zeros((n_test, len(first_stage_fns)))

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    for fold_idx, (tr_idx, va_idx) in enumerate(skf.split(np.arange(n_val), val_y)):
        v_tr = val_df.iloc[tr_idx].reset_index(drop=True)
        v_va = val_df.iloc[va_idx].reset_index(drop=True)
        y_tr = val_y[tr_idx]
        y_va_dummy = val_y[va_idx]
        for k, (name, fn) in enumerate(first_stage_fns):
            _, fold_pred, _ = fn(v_tr, v_va, y_tr, y_va_dummy)
            oof_val[va_idx, k] = fold_pred
        # Each first-stage's test predictions accumulated across folds (avg)
        for k, (name, fn) in enumerate(first_stage_fns):
            _, t_pred, _ = fn(v_tr, test_df, y_tr, test_y)
            test_acc[:, k] += t_pred / n_splits

    # Second-stage: small LR on OOF predictions
    second = LogisticRegression(C=1.0, max_iter=500, random_state=SEED)
    second.fit(oof_val, val_y)
    val_prob = second.predict_proba(oof_val)[:, 1]
    test_prob = second.predict_proba(test_acc)[:, 1]
    return val_prob, test_prob, {
        "n_splits": n_splits,
        "first_stage_models": [n for n, _ in first_stage_fns],
        "second_stage_coefs": dict(zip([n for n, _ in first_stage_fns],
                                        second.coef_[0].round(4).tolist())),
    }


def variant_blend_topk(val_df, test_df, val_y, test_y):
    """Blend of the top-3 advanced variants (pair_disagreement + isotonic_weighted + pure)
    with weights tuned on val by grid search."""
    pure_v, pure_t, _ = variant_pure(val_df, test_df, val_y, test_y)
    iso_v, iso_t, _   = variant_isotonic_weighted(val_df, test_df, val_y, test_y)
    pair_v, pair_t, _ = variant_pair_disagreement(val_df, test_df, val_y, test_y)
    best_w, best_auroc = (1/3, 1/3, 1/3), -1.0
    grid = np.linspace(0.0, 1.0, 11)
    for a in grid:
        for b in grid:
            c = 1 - a - b
            if c < -1e-6 or c > 1 + 1e-6: continue
            blend_val = a * pure_v + b * iso_v + c * pair_v
            try:
                au = roc_auc_score(val_y, blend_val)
                if au > best_auroc:
                    best_auroc = au; best_w = (a, b, c)
            except Exception:
                pass
    a, b, c = best_w
    val_prob  = a * pure_v + b * iso_v + c * pair_v
    test_prob = a * pure_t + b * iso_t + c * pair_t
    return val_prob, test_prob, {"weights": {"pure": a, "isotonic_weighted": b,
                                             "pair_disagreement": c},
                                  "val_auroc_at_grid_best": round(best_auroc, 4)}


def variant_per_child_offset(val_df, test_df, val_y, test_y):
    """Pure stacker, then per-child mean-residual correction.
    Exploits the seen-child split structure: same children in val + test.
    For each child, compute on val: offset_c = mean(label - stacker_prob).
    At test: prob_test_c = clip(stacker_prob + offset_c, 0, 1).
    """
    feats = list(SCORE_FEATS) + ["timepoint_is_36m"]
    X_val = val_df[feats].to_numpy(dtype=float)
    X_test = test_df[feats].to_numpy(dtype=float)
    clf = LogisticRegression(C=1.0, max_iter=500, random_state=SEED)
    clf.fit(X_val, val_y)
    base_val  = clf.predict_proba(X_val)[:, 1]
    base_test = clf.predict_proba(X_test)[:, 1]

    # Per-child offset on val. Need child_id — read from master split csv.
    master = pd.read_csv(MASTER_CSV)[["audio_path", "child_id"]]
    val_with = val_df.merge(master, on="audio_path", how="left").reset_index(drop=True)
    test_with = test_df.merge(master, on="audio_path", how="left").reset_index(drop=True)
    val_resid = val_y - base_val
    offsets = (
        pd.DataFrame({"child_id": val_with["child_id"], "resid": val_resid})
        .groupby("child_id")["resid"].mean()
    )
    # Shrink offsets toward 0 by k=2 prior weight to reduce overfit on tiny per-child counts
    n_per_child = val_with["child_id"].value_counts()
    shrink = n_per_child / (n_per_child + 2.0)
    offsets = offsets * shrink.reindex(offsets.index).fillna(0.0)
    val_with["offset"]  = val_with["child_id"].map(offsets).fillna(0.0)
    test_with["offset"] = test_with["child_id"].map(offsets).fillna(0.0)
    val_prob  = np.clip(base_val  + val_with["offset"].to_numpy(),  0.0, 1.0)
    test_prob = np.clip(base_test + test_with["offset"].to_numpy(), 0.0, 1.0)
    return val_prob, test_prob, {"offset_mean": float(offsets.mean()),
                                  "offset_std":  float(offsets.std()),
                                  "n_children_with_offset": int(offsets.notna().sum())}


def variant_fp_focused(val_df, test_df, val_y, test_y, fp_weight: float = 5.0):
    """Pure stacker but with class_weight={0: fp_weight, 1: 1.0} to penalize FPs harder.
    Targets the dominant failure mode (median FP=18 vs median FN=4).
    """
    feats = list(SCORE_FEATS) + ["timepoint_is_36m"]
    X_val = val_df[feats].to_numpy(dtype=float)
    X_test = test_df[feats].to_numpy(dtype=float)
    clf = LogisticRegression(C=1.0, max_iter=500, random_state=SEED,
                              class_weight={0: fp_weight, 1: 1.0})
    clf.fit(X_val, val_y)
    return clf.predict_proba(X_val)[:, 1], clf.predict_proba(X_test)[:, 1], {
        "class_weight": {0: fp_weight, 1: 1.0},
    }


def variant_bagged_stacker(val_df, test_df, val_y, test_y, n_boot: int = 100):
    """Bag the pure stacker: 100 bootstrap resamples of val, average their predictions.
    Reduces variance from the small val set (431 clips)."""
    feats = list(SCORE_FEATS) + ["timepoint_is_36m"]
    X_val = val_df[feats].to_numpy(dtype=float)
    X_test = test_df[feats].to_numpy(dtype=float)
    rng = np.random.default_rng(SEED)
    n = len(val_df)
    val_acc = np.zeros(n)
    test_acc = np.zeros(len(test_df))
    n_used = 0
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        if len(np.unique(val_y[idx])) < 2:  # skip degenerate samples
            continue
        clf = LogisticRegression(C=1.0, max_iter=500, random_state=SEED + b)
        clf.fit(X_val[idx], val_y[idx])
        val_acc  += clf.predict_proba(X_val)[:, 1]
        test_acc += clf.predict_proba(X_test)[:, 1]
        n_used += 1
    return val_acc / n_used, test_acc / n_used, {"n_bootstrap": n_used}


VARIANTS = [
    ("pure",                 variant_pure),
    ("mean",                 variant_mean),
    ("calibrated_mean",      variant_calibrated_mean),
    ("isotonic_weighted",    variant_isotonic_weighted),
    ("rank_stacker",         variant_rank_stacker),
    ("per_timepoint",        variant_per_timepoint),
    ("confidence_weighted",  variant_confidence_weighted),
    ("pair_disagreement",    variant_pair_disagreement),
    ("topk_systems",         variant_topk_systems),
    ("cv_stacked",           variant_cv_stacked),
    ("blend_topk",           variant_blend_topk),
    ("bagged_stacker",       variant_bagged_stacker),
    ("per_child_offset",     variant_per_child_offset),
    ("fp_focused",           variant_fp_focused),
]


# ── Best-of-best combo ──────────────────────────────────────────────────────

def variant_cv_stacked_then_offset(val_df, test_df, val_y, test_y, n_splits: int = 5):
    """Take cv_stacked output and apply per-child offset correction on top."""
    cv_v, cv_t, _ = variant_cv_stacked(val_df, test_df, val_y, test_y, n_splits=n_splits)
    # Per-child offset on cv_stacked predictions
    master = pd.read_csv(MASTER_CSV)[["audio_path", "child_id"]]
    val_with  = val_df.merge(master,  on="audio_path", how="left").reset_index(drop=True)
    test_with = test_df.merge(master, on="audio_path", how="left").reset_index(drop=True)
    val_resid = val_y - cv_v
    offsets = (
        pd.DataFrame({"child_id": val_with["child_id"], "resid": val_resid})
        .groupby("child_id")["resid"].mean()
    )
    n_per_child = val_with["child_id"].value_counts()
    shrink = n_per_child / (n_per_child + 2.0)
    offsets = offsets * shrink.reindex(offsets.index).fillna(0.0)
    val_off  = val_with["child_id"].map(offsets).fillna(0.0).to_numpy()
    test_off = test_with["child_id"].map(offsets).fillna(0.0).to_numpy()
    return np.clip(cv_v + val_off, 0, 1), np.clip(cv_t + test_off, 0, 1), {
        "n_children_with_offset": int(offsets.notna().sum()),
        "first_stage": "cv_stacked",
    }


VARIANTS.append(("cv_stacked_then_offset", variant_cv_stacked_then_offset))


def main() -> None:
    val_scores  = load_system_scores("val")
    test_scores = load_system_scores("test")
    labels      = load_labels_split()
    val_df  = assemble("val",  val_scores,  labels)
    test_df = assemble("test", test_scores, labels)
    val_y  = val_df["label"].to_numpy(dtype=int)
    test_y = test_df["label"].to_numpy(dtype=int)
    print(f"val n={len(val_df)}  test n={len(test_df)}  systems={len(SCORE_FEATS)}")

    rows = []
    for name, fn in VARIANTS:
        val_p, test_p, extra = fn(val_df, test_df, val_y, test_y)
        m = save_variant(name, val_p, val_y, test_p, test_y, test_df, extra, val_df=val_df)
        rows.append({
            "variant": name, "F1": round(m["f1"], 4),
            "AUROC": round(m["auroc"], 4), "AUPRC": round(m["auprc"], 4),
            "threshold": round(m["threshold"], 3),
            "delta_F1": m["delta_f1"], "delta_AUROC": m["delta_auroc"],
        })

    leaderboard = pd.DataFrame(rows)
    # Sort by AUROC desc but keep pure on top for visual reference
    pure_row = leaderboard[leaderboard["variant"] == "pure"]
    rest = leaderboard[leaderboard["variant"] != "pure"].sort_values("AUROC", ascending=False)
    leaderboard = pd.concat([pure_row, rest], ignore_index=True)
    leaderboard.to_csv(OUT_ROOT / "leaderboard.csv", index=False)

    print("\n=== LEADERBOARD ===")
    print(leaderboard.to_string(index=False))
    print(f"\nReference: best_audio_mil mean F1={BASELINE_F1:.3f} AUROC={BASELINE_AUROC:.3f}")
    print(f"Reference: 12-sys metadata stacker F1=0.9053 AUROC=0.9044 AUPRC=0.9663")
    print(f"Reference: 12-sys + visual stacker F1=0.8977 AUROC=0.9052 AUPRC=0.9677  (project ceiling)")
    print(f"Reference: pure (no metadata)      F1=0.9094 AUROC=0.8969 AUPRC=0.9622")


if __name__ == "__main__":
    main()
