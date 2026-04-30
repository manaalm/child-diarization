"""Metadata-conditioned routing and ensemble extensions.

Sub-features:
  A (--mode router):  rule-based + learned metadata router
  B (--mode stack):   metadata-augmented LR/GBM stacker
  --verify:           check all prediction files are loadable

Usage:
  python evaluation/metadata_router.py --verify
  python evaluation/metadata_router.py --mode stack
  python evaluation/metadata_router.py --mode router
  python evaluation/metadata_router.py --mode all
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, average_precision_score

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

BASELINE_F1 = 0.893
BASELINE_AUROC = 0.878
SEED = 42

# ── System prediction file registry ─────────────────────────────────────────

_SYSTEM_PATHS = {
    "babar":       ("babar_ecapa_enrollment_runs/enroll_{split}_predictions.csv", "prob"),
    "vtc":         ("vtc_ecapa_enrollment_runs/enroll_{split}_predictions.csv", "prob"),
    "vtc_kchi":    ("vtc_kchi_ecapa_enrollment_runs/enroll_{split}_predictions.csv", "prob"),
    "vbx":         ("vbx_ecapa_enrollment_runs/enroll_{split}_predictions.csv", "prob"),
    "usc_sail":    ("whisper-modeling/usc_sail_enrollment_runs/enroll_{split}_predictions.csv", "prob"),
    "pyannote":    ("pyannote/pyannote_enrollment_runs/{split}_predictions.csv", "prob"),
    "eend_eda":    ("eend_eda_ecapa_enrollment_runs/enroll_{split}_predictions.csv", "prob"),
    "sortformer":  ("sortformer_ecapa_enrollment_runs/enroll_{split}_predictions.csv", "prob"),
    "wavlm_mil":   ("mil/mil_results/wavlm_mil/{split}_predictions.csv", "score"),
    "whisper_mil": ("mil/mil_results/whisper_mil/{split}_predictions.csv", "score"),
    "audio_llm":   ("baselines/audio_llm_baseline_runs/qwen2_audio_7b/{split}_predictions.csv", "prob"),
    "speaker_informed_av": ("pseudo_frame/results/speaker_informed_asd/{split}_predictions.csv", "prob"),
}

MASTER_CSV = os.path.join(_REPO, "whisper-modeling/seen_child_splits/master_with_split.csv")
ENSEMBLE_TEST_CSV = os.path.join(_REPO, "ensemble_runs/test_predictions.csv")

# ── Data loading ─────────────────────────────────────────────────────────────

def load_system_scores(split: str) -> pd.DataFrame:
    """Load all 12 systems and join on audio_path. score→prob renamed. NaN→0.5."""
    dfs = []
    for name, (tmpl, col) in _SYSTEM_PATHS.items():
        path = os.path.join(_REPO, tmpl.format(split=split))
        if not os.path.exists(path):
            print(f"  WARNING: {name} {split} predictions missing: {path}", flush=True)
            continue
        df = pd.read_csv(path)[["audio_path", col]].rename(columns={col: f"{name}_prob"})
        dfs.append(df)

    if not dfs:
        raise FileNotFoundError("No system predictions found.")

    merged = dfs[0]
    for df in dfs[1:]:
        merged = merged.merge(df, on="audio_path", how="outer")

    # Impute missing scores with neutral prior
    prob_cols = [c for c in merged.columns if c.endswith("_prob")]
    merged[prob_cols] = merged[prob_cols].fillna(0.5)
    return merged.reset_index(drop=True)


def load_metadata() -> pd.DataFrame:
    """Load BIDS metadata from master_with_split.csv; parse #_adults, #_children."""
    df = pd.read_csv(MASTER_CSV)

    def _to_int(val, default):
        try:
            return int(str(val).strip().split("+")[0])
        except Exception:
            return default

    df["n_adults_int"] = df["#_adults"].apply(lambda v: _to_int(v, 0))
    df["n_children_int"] = df["#_children"].apply(lambda v: _to_int(v, 1))
    df["n_adults_ge2"] = (df["n_adults_int"] >= 2).astype(int)
    df["n_children_ge2"] = (df["n_children_int"] >= 2).astype(int)
    df["context_unknown"] = df["Context"].str.lower().str.contains("unknown", na=False).astype(int)
    df["has_interaction"] = (df["Interaction_with_child"].str.lower() == "yes").astype(int)
    df["timepoint_is_36m"] = (df["timepoint_norm"] == "36_month").astype(int)

    keep = ["audio_path", "split", "label", "timepoint_norm",
            "n_adults_int", "n_children_int", "n_adults_ge2", "n_children_ge2",
            "context_unknown", "has_interaction", "timepoint_is_36m"]
    return df[keep].reset_index(drop=True)


def load_split(scores: pd.DataFrame, meta: pd.DataFrame, split: str):
    """Join scores + metadata for a given split; return merged DataFrame."""
    split_meta = meta[meta["split"] == split].copy()
    merged = split_meta.merge(scores, on="audio_path", how="inner")
    if len(merged) == 0:
        raise ValueError(f"Empty merge for split={split}")
    return merged.reset_index(drop=True)


# ── Metrics & helpers ────────────────────────────────────────────────────────

def compute_metrics(y_true, y_prob, threshold=0.5):
    y_true = np.array(y_true, dtype=int)
    y_prob = np.array(y_prob, dtype=float)
    y_pred = (y_prob >= threshold).astype(int)
    m = {
        "f1":        float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall":    float(recall_score(y_true, y_pred, zero_division=0)),
    }
    try:    m["auroc"] = float(roc_auc_score(y_true, y_prob))
    except: m["auroc"] = float("nan")
    try:    m["auprc"] = float(average_precision_score(y_true, y_prob))
    except: m["auprc"] = float("nan")
    return m


def tune_threshold(y_true, y_prob):
    best_t, best_f1 = 0.5, -1.0
    for t in np.linspace(0.05, 0.95, 181):
        f1 = float(f1_score(np.array(y_true), (np.array(y_prob) >= t).astype(int), zero_division=0))
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return best_t


def save_results(out_dir: str, val_m: dict, test_m: dict, preds: pd.DataFrame, cfg: dict):
    os.makedirs(out_dir, exist_ok=True)
    test_m["baseline_f1"] = BASELINE_F1
    test_m["baseline_auroc"] = BASELINE_AUROC
    test_m["delta_f1"] = round(test_m["f1"] - BASELINE_F1, 4)
    test_m["delta_auroc"] = round(test_m.get("auroc", float("nan")) - BASELINE_AUROC, 4)
    test_m["n"] = len(preds)
    with open(os.path.join(out_dir, "test_metrics_tuned.json"), "w") as f:
        json.dump(test_m, f, indent=2)
    with open(os.path.join(out_dir, "val_metrics_tuned.json"), "w") as f:
        json.dump(val_m, f, indent=2)
    preds.to_csv(os.path.join(out_dir, "test_predictions.csv"), index=False)
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"  → {out_dir}  F1={test_m['f1']:.4f} AUROC={test_m.get('auroc', float('nan')):.4f} "
          f"delta_F1={test_m['delta_f1']:+.4f} delta_AUROC={test_m['delta_auroc']:+.4f}", flush=True)


# ── Sub-feature B: Metadata-augmented stacker ────────────────────────────────

SCORE_FEATS = [f"{s}_prob" for s in _SYSTEM_PATHS]
META_FEATS = ["n_adults_int", "n_children_int", "n_adults_ge2", "n_children_ge2",
              "context_unknown", "has_interaction", "timepoint_is_36m"]
VISUAL_FEATS = [
    "face_count_max", "face_count_mean",
    "face_area_max_norm", "face_area_mean_norm",
    "face_confidence_mean", "face_track_coverage_ratio",
    "n_distinct_tracks", "has_any_face", "eligibility_score",
]


def load_visual_features(path: str) -> pd.DataFrame:
    """Load the per-clip visual eligibility CSV produced by
    pseudo_frame/visual_eligibility.py. Returns a DataFrame with audio_path +
    the columns in VISUAL_FEATS (missing columns are filled with 0)."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Visual feature CSV not found: {path}")
    df = pd.read_csv(path)
    keep = ["audio_path"] + [c for c in VISUAL_FEATS if c in df.columns]
    return df[keep].reset_index(drop=True)


def build_feature_matrix(df: pd.DataFrame, include_visual: bool = False) -> np.ndarray:
    feats = SCORE_FEATS + META_FEATS + (VISUAL_FEATS if include_visual else [])
    available = [f for f in feats if f in df.columns]
    X = df[available].fillna(0.0).to_numpy(dtype=float)
    return X, available


def run_metadata_stack(val_df, test_df, out_dir, seed=SEED, include_visual=False):
    label = "Metadata + Visual" if include_visual else "Metadata"
    print(f"\n=== Sub-feature B: {label}-Augmented Stacker ===", flush=True)
    X_val, feats = build_feature_matrix(val_df, include_visual=include_visual)
    y_val = val_df["label"].to_numpy(dtype=int)
    X_test, _ = build_feature_matrix(test_df, include_visual=include_visual)
    y_test = test_df["label"].to_numpy(dtype=int)

    results = {}
    for name, clf in [
        ("lr",  LogisticRegression(C=1.0, max_iter=500, random_state=seed)),
        ("gbm", HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05,
                                               max_leaf_nodes=15, min_samples_leaf=20,
                                               random_state=seed)),
    ]:
        clf.fit(X_val, y_val)
        val_prob = clf.predict_proba(X_val)[:, 1]
        t = tune_threshold(y_val, val_prob)
        val_m = compute_metrics(y_val, val_prob, threshold=t)
        val_m["threshold"] = t

        test_prob = clf.predict_proba(X_test)[:, 1]
        test_m = compute_metrics(y_test, test_prob, threshold=t)
        test_m["threshold"] = t
        results[name] = {"val_f1": val_m["f1"], "test_f1": test_m["f1"],
                         "test_auroc": test_m.get("auroc"), "clf": clf,
                         "val_prob": val_prob, "test_prob": test_prob,
                         "val_m": val_m, "test_m": test_m}
        print(f"  {name}: val_F1={val_m['f1']:.4f} test_F1={test_m['f1']:.4f}", flush=True)

    # Penalize overfit models (val_F1 >= 0.99 on 431 training clips = likely overfit)
    best_name = max(results, key=lambda k: results[k]["val_f1"] if results[k]["val_f1"] < 0.99 else 0.0)
    best = results[best_name]
    preds = test_df[["audio_path", "label", "timepoint_norm"]].copy()
    preds["score"] = best["test_prob"]
    preds["prediction"] = (best["test_prob"] >= best["test_m"]["threshold"]).astype(int)

    # Feature importances
    lr_clf, gbm_clf = results["lr"]["clf"], results["gbm"]["clf"]
    importances = {
        "lr_coefficients":        dict(zip(feats, lr_clf.coef_[0].tolist())),
        "gbm_feature_importances": None,  # HistGradientBoosting doesn't expose feature_importances_
    }
    with open(os.path.join(out_dir, "feature_importances.json"), "w") as f:
        json.dump(importances, f, indent=2)

    # Print top metadata importance
    meta_importance_lr = {k: abs(v) for k, v in importances["lr_coefficients"].items()
                          if k in META_FEATS}
    print(f"  Top metadata features (|LR coef|): "
          f"{sorted(meta_importance_lr.items(), key=lambda x: -x[1])[:3]}", flush=True)

    cfg = {"sub_feature": "B", "model_type": best_name, "features": feats,
           "score_features": SCORE_FEATS, "meta_features": META_FEATS,
           "visual_features": VISUAL_FEATS if include_visual else [],
           "include_visual": bool(include_visual),
           "seed": seed, "val_threshold": best["test_m"]["threshold"],
           "created": "2026-04-29" if include_visual else "2026-04-28"}
    save_results(out_dir, best["val_m"], best["test_m"], preds, cfg)


# ── Sub-feature A: Router ────────────────────────────────────────────────────

def _default_score(row):
    """Fallback: mean of babar, vtc, wavlm_mil, whisper_mil."""
    cols = ["babar_prob", "vtc_prob", "wavlm_mil_prob", "whisper_mil_prob"]
    available = [c for c in cols if c in row.index and not pd.isna(row[c])]
    return float(np.mean([row[c] for c in available])) if available else 0.5


def apply_rule_router(row) -> tuple:
    if row["context_unknown"] == 1:
        return float(row.get("sortformer_prob", 0.5)), "context_unknown→sortformer"
    if row["n_adults_int"] >= 2:
        s = np.mean([row.get("wavlm_mil_prob", 0.5), row.get("eend_eda_prob", 0.5)])
        return float(s), "n_adults_ge2→mean(wavlm_mil,eend_eda)"
    if row["n_children_int"] >= 2:
        return float(row.get("whisper_mil_prob", 0.5)), "n_children_ge2→whisper_mil"
    if row["n_children_int"] == 1:
        return float(row.get("vtc_prob", 0.5)), "n_children_eq1→vtc"
    return _default_score(row), "default→mean(babar,vtc,wavlm_mil,whisper_mil)"


def run_rule_router(val_df, test_df, out_dir):
    print("\n=== Sub-feature A (rule-based): Metadata Router ===", flush=True)

    def _scores(df):
        results = [apply_rule_router(row) for _, row in df.iterrows()]
        return np.array([r[0] for r in results]), [r[1] for r in results]

    val_scores, _ = _scores(val_df)
    t = tune_threshold(val_df["label"].to_numpy(), val_scores)
    val_m = compute_metrics(val_df["label"].to_numpy(), val_scores, threshold=t)
    val_m["threshold"] = t

    test_scores, test_rules = _scores(test_df)
    test_m = compute_metrics(test_df["label"].to_numpy(), test_scores, threshold=t)
    test_m["threshold"] = t

    rule_counts = pd.Series(test_rules).value_counts().to_dict()
    print(f"  Rule distribution: {rule_counts}", flush=True)

    preds = test_df[["audio_path", "label", "timepoint_norm"]].copy()
    preds["score"] = test_scores
    preds["routed_system"] = test_rules
    preds["prediction"] = (test_scores >= t).astype(int)

    cfg = {"sub_feature": "A_rule", "rule_distribution": rule_counts,
           "val_threshold": t, "seed": SEED, "created": "2026-04-28",
           "rules": ["context_unknown→sortformer", "n_adults_ge2→mean(wavlm_mil,eend_eda)",
                     "n_children_ge2→whisper_mil", "n_children_eq1→vtc",
                     "default→mean(babar,vtc,wavlm_mil,whisper_mil)"]}
    save_results(out_dir, val_m, test_m, preds, cfg)


def run_learned_router(val_df, test_df, out_dir, seed=SEED):
    print("\n=== Sub-feature A (learned): Metadata Router ===", flush=True)
    # For each val clip, find which single system achieved lowest binary cross-entropy.
    # Train a classifier on metadata→best_system_index, then at test time pick that system's score.
    prob_cols = [c for c in SCORE_FEATS if c in val_df.columns]
    y_val_labels = val_df["label"].to_numpy(dtype=int)

    best_system_idx = []
    for i, row in val_df.iterrows():
        label = int(row["label"])
        losses = []
        for col in prob_cols:
            p = float(np.clip(row.get(col, 0.5), 1e-7, 1 - 1e-7))
            loss = -(label * np.log(p) + (1 - label) * np.log(1 - p))
            losses.append(loss)
        best_system_idx.append(int(np.argmin(losses)))

    X_val = val_df[META_FEATS].fillna(0).to_numpy(dtype=float)
    clf = LogisticRegression(C=1.0, max_iter=1000, random_state=seed,
                             multi_class="multinomial")
    clf.fit(X_val, best_system_idx)

    degenerate = len(set(clf.predict(X_val))) == 1

    X_test = test_df[META_FEATS].fillna(0).to_numpy(dtype=float)
    predicted_sys_test = clf.predict(X_test)
    test_scores = np.array([
        float(test_df.iloc[i].get(prob_cols[s], 0.5))
        for i, s in enumerate(predicted_sys_test)
    ])

    val_pred_sys = clf.predict(X_val)
    val_scores = np.array([
        float(val_df.iloc[i].get(prob_cols[s], 0.5))
        for i, s in enumerate(val_pred_sys)
    ])
    t = tune_threshold(y_val_labels, val_scores)
    val_m = compute_metrics(y_val_labels, val_scores, threshold=t)
    val_m["threshold"] = t

    test_m = compute_metrics(test_df["label"].to_numpy(), test_scores, threshold=t)
    test_m["threshold"] = t

    sys_counts = pd.Series([prob_cols[s] for s in predicted_sys_test]).value_counts().to_dict()
    print(f"  System distribution (test): {sys_counts}", flush=True)
    if degenerate:
        print("  WARNING: learned router degenerated (always picks one system)", flush=True)

    preds = test_df[["audio_path", "label", "timepoint_norm"]].copy()
    preds["score"] = test_scores
    preds["routed_system"] = [prob_cols[s] for s in predicted_sys_test]
    preds["prediction"] = (test_scores >= t).astype(int)

    cfg = {"sub_feature": "A_learned", "system_distribution_test": sys_counts,
           "degenerate": degenerate, "val_threshold": t, "seed": seed, "created": "2026-04-28"}
    save_results(out_dir, val_m, test_m, preds, cfg)


# ── Verification ─────────────────────────────────────────────────────────────

def run_verify():
    print("=== Verifying system predictions ===", flush=True)
    ok = True
    for split, exp_rows in [("val", 431), ("test", 441)]:
        scores = load_system_scores(split)
        prob_cols = [c for c in scores.columns if c.endswith("_prob")]
        print(f"\n{split}: {len(scores)} rows, {len(prob_cols)} systems", flush=True)
        if len(scores) < exp_rows:
            print(f"  WARNING: expected {exp_rows} rows, got {len(scores)}", flush=True)
            ok = False
        for col in prob_cols:
            n_missing = scores[col].isna().sum()
            if n_missing > 0:
                print(f"  {col}: {n_missing} NaN (will impute 0.5)", flush=True)

    meta = load_metadata()
    for col in ["n_adults_int", "n_children_int", "context_unknown", "has_interaction"]:
        if col not in meta.columns:
            print(f"  MISSING metadata col: {col}", flush=True)
            ok = False
    print(f"\nMetadata: {len(meta)} rows, splits={meta['split'].value_counts().to_dict()}", flush=True)
    print(f"\n{'PASS' if ok else 'FAIL'}", flush=True)
    return ok


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Metadata-conditioned routing and stacking")
    parser.add_argument("--mode", choices=["stack", "router", "all"], default="all")
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--visual-features",
                        help="Path to a per-clip visual eligibility CSV "
                             "(e.g. pseudo_frame/visual_features/visual_eligibility.csv). "
                             "When set, augments the stacker with computed visual-quality features "
                             "and writes results to ensemble_runs/metadata_stack_av/ instead.")
    args = parser.parse_args()

    if args.verify:
        ok = run_verify()
        sys.exit(0 if ok else 1)

    print("Loading data ...", flush=True)
    val_scores = load_system_scores("val")
    test_scores = load_system_scores("test")
    meta = load_metadata()

    val_df  = load_split(val_scores,  meta, "val")
    test_df = load_split(test_scores, meta, "test")
    print(f"Val: {len(val_df)} clips | Test: {len(test_df)} clips", flush=True)

    include_visual = bool(args.visual_features)
    if include_visual:
        vis_path = args.visual_features if os.path.isabs(args.visual_features) \
                   else os.path.join(_REPO, args.visual_features)
        vis = load_visual_features(vis_path)
        val_df  = val_df.merge(vis, on="audio_path", how="left")
        test_df = test_df.merge(vis, on="audio_path", how="left")
        # Fill any clips missing from the visual table with 0
        for col in [c for c in VISUAL_FEATS if c in val_df.columns]:
            val_df[col]  = val_df[col].fillna(0.0)
            test_df[col] = test_df[col].fillna(0.0)
        print(f"  Merged visual features: {len(vis)} rows; using {len(VISUAL_FEATS)} features",
              flush=True)

    if args.mode in ("stack", "all"):
        out = os.path.join(_REPO,
                           "ensemble_runs/metadata_stack_av" if include_visual
                           else "ensemble_runs/metadata_stack")
        run_metadata_stack(val_df, test_df, out, seed=args.seed,
                           include_visual=include_visual)

    if args.mode in ("router", "all"):
        out_rule = os.path.join(_REPO, "ensemble_runs/metadata_router_rule")
        run_rule_router(val_df, test_df, out_rule)
        out_learned = os.path.join(_REPO, "ensemble_runs/metadata_router_learned")
        run_learned_router(val_df, test_df, out_learned, seed=args.seed)

    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
