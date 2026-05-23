"""
Cross-speaker BIDS LR-stacker ensemble sweep.

Auto-discovers every available BIDS cross-child component system, builds val
(n=444) / test (n=742) score matrices, and trains 9 LR-stacker variants on
val + evaluates on test --- the cross-speaker analogue of
``evaluation/advanced_ensembles.py`` for the within-speaker grid.

Variants run (subset of the within-speaker advanced/ family):
    mean             — unweighted arithmetic mean (no learning)
    pure             — LR over all system probs + timepoint_is_36m
    cv_stacked       — 5-fold cross-validated LR (avoids val leakage)
    fp_focused       — pure with class_weight={0: 5.0, 1: 1.0}, demotes FPs
    topk_systems     — LR on the top-k systems by val AUROC
    blend_topk       — convex blend of (pure, isotonic_weighted, rank_stacker)
    bagged_stacker   — 100-bootstrap bag of pure stackers
    per_timepoint    — separate pure stacker per timepoint cohort
    calibrated_mean  — isotonic-calibrate each system, then average

Outputs:
    ensemble_runs/cross_child_bids/<variant>/{test,val}_metrics_tuned.json
    ensemble_runs/cross_child_bids/<variant>/test_predictions.csv
    ensemble_runs/cross_child_bids/leaderboard.csv

Component sources (auto-discovered; missing ones logged then skipped):
    role-only diarizers       — evaluation/cross_child_{babar,vtc,vtc_kchi}_role_only_bids/
    audio LLMs                — baselines/audio_llm_baseline_runs/{qwen2_audio,qwen25_omni,qwen3_omni_30b_thinking}_cross_child_bids/
    zero-shot scene           — baselines/scene_analysis_runs/{yamnet,ast}_cross_child_bids/
    AV fusion (always-fuse)   — av_fusion/av_results/manual_only_cross_child_bids/predictions_*.csv
    pseudo-frame              — pseudo_frame/results/{wavlm,whisper}_pseudo_frame_cross_child/
    MIL                       — mil/mil_results/*_cross_child/   (when SLURM 14191408 lands)
    encoders                  — baselines/baseline_results_cross_child_bids/*/  (when 14194134 lands)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import StratifiedKFold

REPO = Path("/orcd/scratch/orcd/008/manaal/child-adult-diarization")
SPLITS = REPO / "baselines/splits"
OUT_ROOT = REPO / "ensemble_runs/cross_child_bids"
OUT_ROOT.mkdir(parents=True, exist_ok=True)
SEED = 42


# ── system discovery ───────────────────────────────────────────────────────

def candidate_sources() -> list[tuple[str, Path, str]]:
    """List (system_name, predictions_dir, prob_column) tuples to probe."""
    role_dirs = [
        ("babar_role",          REPO / "evaluation/cross_child_babar_role_only_bids",    "prob"),
        ("vtc_role",            REPO / "evaluation/cross_child_vtc_role_only_bids",      "prob"),
        ("vtc_kchi_role",       REPO / "evaluation/cross_child_vtc_kchi_role_only_bids", "prob"),
    ]
    audio_llm_dirs = [
        ("qwen2_audio_7b",                REPO / "baselines/audio_llm_baseline_runs/qwen2_audio_7b_cross_child_bids", "prob"),
        ("qwen25_omni_7b",                REPO / "baselines/audio_llm_baseline_runs/qwen25_omni_7b_cross_child_bids", "prob"),
        ("qwen3_omni_30b_thinking",       REPO / "baselines/audio_llm_baseline_runs/qwen3_omni_30b_thinking_cross_child_bids", "prob"),
    ]
    scene_dirs = [
        ("yamnet", REPO / "baselines/scene_analysis_runs/yamnet_cross_child_bids", "prob"),
        ("ast",    REPO / "baselines/scene_analysis_runs/ast_cross_child_bids",    "prob"),
    ]
    pseudo_dirs = [
        ("wavlm_pseudo_frame",   REPO / "pseudo_frame/results/wavlm_pseudo_frame_cross_child",   "score"),
        ("whisper_pseudo_frame", REPO / "pseudo_frame/results/whisper_pseudo_frame_cross_child", "score"),
    ]
    mil_dirs = [
        ("wavlm_mil",                   REPO / "mil/mil_results/wavlm_mil_cross_child",                   "score"),
        ("whisper_mil",                 REPO / "mil/mil_results/whisper_mil_cross_child",                 "score"),
        ("whisper_medium_mil",          REPO / "mil/mil_results/whisper_medium_mil_cross_child",          "score"),
        ("whisper_mil_acmil_max",       REPO / "mil/mil_results/whisper_mil_acmil_max_cross_child",       "score"),
        ("whisper_mil_tsmil_concat",    REPO / "mil/mil_results/whisper_mil_tsmil_concat_cross_child",    "score"),
    ]
    encoder_variants = [
        "whisper_mean", "whisper_attn", "wavlm_mean", "wavlm_attn",
        "whisper_attn_lw", "wavlm_attn_lw",
        "fused_attn", "whisper_attn_unfreeze2", "fused_attn_unfreeze2",
        "whisper_attn_ptt", "whisper_attn_aug", "whisper_attn_aug_ptt",
    ]
    encoder_dirs = [
        (v, REPO / f"baselines/baseline_results_cross_child_bids/{v}", "prob")
        for v in encoder_variants
    ]
    return (
        role_dirs + audio_llm_dirs + scene_dirs + pseudo_dirs + mil_dirs + encoder_dirs
    )


def _read_predictions(d: Path, col: str, split: str) -> pd.Series | None:
    """Returns a Series indexed by audio_path with the named score column."""
    candidates = [
        d / f"{split}_predictions.csv",
        d / f"predictions_{split}.csv",  # AV pipeline naming
    ]
    csv = next((c for c in candidates if c.exists()), None)
    if csv is None:
        return None
    df = pd.read_csv(csv)
    if "audio_path" not in df.columns or col not in df.columns:
        # try synonyms
        if "prob" in df.columns and col == "score":
            col = "prob"
        elif "score" in df.columns and col == "prob":
            col = "score"
        else:
            return None
    return df.set_index("audio_path")[col].rename(d.name)


def discover_available_systems() -> dict[str, tuple[Path, str]]:
    """Return {system_name: (dir, col)} for every system that has BOTH val and test predictions."""
    avail = {}
    for name, d, col in candidate_sources():
        v = _read_predictions(d, col, "val")
        t = _read_predictions(d, col, "test")
        if v is not None and t is not None:
            avail[name] = (d, col)
            print(f"  ✓ {name:38s} val={len(v.dropna())} test={len(t.dropna())}  ({d.name})")
        else:
            missing_split = "val" if v is None else "test"
            print(f"  ✗ {name:38s} ({missing_split} predictions missing under {d})")
    return avail


# ── data assembly ──────────────────────────────────────────────────────────

def load_labels(split: str) -> pd.DataFrame:
    csv = SPLITS / f"{split}.csv"
    df = pd.read_csv(csv)
    if "audio_exists" in df.columns:
        df = df[df.audio_exists.astype(bool)]
    df["timepoint_is_36m"] = (df["timepoint_norm"] == "36_month").astype(int)
    return df[["audio_path", "label", "timepoint_norm", "timepoint_is_36m", "child_id"]].reset_index(drop=True)


def build_score_matrix(systems: dict[str, tuple[Path, str]], split: str,
                       ref: pd.DataFrame) -> pd.DataFrame:
    base = ref.copy()
    for name, (d, col) in systems.items():
        s = _read_predictions(d, col, split)
        if s is None:
            continue
        base = base.merge(s.rename(f"{name}_prob").reset_index(),
                          on="audio_path", how="left")
    prob_cols = [c for c in base.columns if c.endswith("_prob")]
    base[prob_cols] = base[prob_cols].fillna(0.5)
    return base, prob_cols


# ── metrics + threshold ───────────────────────────────────────────────────

def ba_tune(y, p):
    best_thr, best_ba = 0.5, -1.0
    for t in np.arange(0.05, 0.991, 0.025):
        preds = (p >= t).astype(int)
        tp = ((preds == 1) & (y == 1)).sum(); tn = ((preds == 0) & (y == 0)).sum()
        fp = ((preds == 1) & (y == 0)).sum(); fn = ((preds == 0) & (y == 1)).sum()
        tpr = tp / max(tp + fn, 1); tnr = tn / max(tn + fp, 1)
        ba = 0.5 * (tpr + tnr)
        if ba > best_ba: best_ba, best_thr = ba, float(t)
    return best_thr


def metrics(y, p, thr):
    y = y.astype(int); preds = (p >= thr).astype(int)
    tp = ((preds == 1) & (y == 1)).sum(); tn = ((preds == 0) & (y == 0)).sum()
    fp = ((preds == 1) & (y == 0)).sum(); fn = ((preds == 0) & (y == 1)).sum()
    prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    tnr = tn / max(tn + fp, 1); ba = 0.5 * (rec + tnr)
    prec_n = tn / max(tn + fn, 1); rec_n = tn / max(tn + fp, 1)
    f1_n = 2 * prec_n * rec_n / max(prec_n + rec_n, 1e-9)
    n = len(y); npos = int(y.sum()); nneg = n - npos
    weighted = (npos / n) * f1 + (nneg / n) * f1_n
    try: auroc = float(roc_auc_score(y, p))
    except Exception: auroc = float("nan")
    try: auprc = float(average_precision_score(y, p))
    except Exception: auprc = float("nan")
    return dict(f1=float(f1), f1_weighted=float(weighted), balanced_accuracy=float(ba),
                precision=float(prec), recall=float(rec),
                auroc=auroc, auprc=auprc, threshold=float(thr),
                n=int(n), n_pos=npos, n_neg=nneg)


def save_variant(name: str, val_p, val_y, test_p, test_y, test_df, val_df, extra: dict):
    out = OUT_ROOT / name
    out.mkdir(parents=True, exist_ok=True)
    thr = ba_tune(val_y, val_p)
    val_m = metrics(val_y, val_p, thr)
    test_m = metrics(test_y, test_p, thr)
    test_m.update(extra)
    (out / "val_metrics_tuned.json").write_text(json.dumps(val_m, indent=2))
    (out / "test_metrics_tuned.json").write_text(json.dumps(test_m, indent=2))
    tp = test_df[["audio_path", "label", "timepoint_norm"]].copy()
    tp["score"] = test_p; tp["prediction"] = (test_p >= thr).astype(int)
    tp.to_csv(out / "test_predictions.csv", index=False)
    vp = val_df[["audio_path", "label", "timepoint_norm"]].copy()
    vp["score"] = val_p; vp["prediction"] = (val_p >= thr).astype(int)
    vp.to_csv(out / "val_predictions.csv", index=False)
    print(f"    [{name}] val BA={val_m['balanced_accuracy']:.4f}  "
          f"test AUROC={test_m['auroc']:.4f}  BA={test_m['balanced_accuracy']:.4f}  "
          f"F1={test_m['f1']:.4f}  thr={thr:.3f}")
    return test_m


# ── variant implementations ────────────────────────────────────────────────

def normalize_probs(arr):
    n = arr.shape[0]
    return np.stack([rankdata(arr[:, j], method="average") / n for j in range(arr.shape[1])], axis=1)


def isotonic_calibrate(val_p, val_y, target_p):
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(val_p, val_y)
    return iso.predict(target_p)


def variant_mean(val_df, test_df, prob_cols):
    return val_df[prob_cols].mean(axis=1).values, test_df[prob_cols].mean(axis=1).values, {"method": "unweighted_mean"}


def variant_pure(val_df, test_df, val_y, prob_cols):
    feats = prob_cols + ["timepoint_is_36m"]
    Xv, Xt = val_df[feats].values, test_df[feats].values
    lr = LogisticRegression(C=1.0, max_iter=500, random_state=SEED)
    lr.fit(Xv, val_y)
    return lr.predict_proba(Xv)[:, 1], lr.predict_proba(Xt)[:, 1], {"method": "lr_over_probs+tp"}


def variant_calibrated_mean(val_df, test_df, val_y, prob_cols):
    cal_v = np.zeros((len(val_df), len(prob_cols)))
    cal_t = np.zeros((len(test_df), len(prob_cols)))
    for j, c in enumerate(prob_cols):
        cal_v[:, j] = isotonic_calibrate(val_df[c].values, val_y, val_df[c].values)
        cal_t[:, j] = isotonic_calibrate(val_df[c].values, val_y, test_df[c].values)
    return cal_v.mean(axis=1), cal_t.mean(axis=1), {"method": "isotonic_per_system_then_mean"}


def variant_rank_stacker(val_df, test_df, val_y, prob_cols):
    # rank-transform each column to [0,1] for both splits independently
    Xv = normalize_probs(val_df[prob_cols].values)
    Xt = normalize_probs(test_df[prob_cols].values)
    Xv = np.hstack([Xv, val_df[["timepoint_is_36m"]].values])
    Xt = np.hstack([Xt, test_df[["timepoint_is_36m"]].values])
    lr = LogisticRegression(C=1.0, max_iter=500, random_state=SEED)
    lr.fit(Xv, val_y)
    return lr.predict_proba(Xv)[:, 1], lr.predict_proba(Xt)[:, 1], {"method": "lr_on_rank_transformed_probs+tp"}


def variant_fp_focused(val_df, test_df, val_y, prob_cols):
    feats = prob_cols + ["timepoint_is_36m"]
    lr = LogisticRegression(C=1.0, max_iter=500, random_state=SEED,
                            class_weight={0: 5.0, 1: 1.0})
    lr.fit(val_df[feats].values, val_y)
    return (lr.predict_proba(val_df[feats].values)[:, 1],
            lr.predict_proba(test_df[feats].values)[:, 1],
            {"method": "lr_with_fp_class_weight_5"})


def variant_topk_systems(val_df, test_df, val_y, prob_cols, k=6):
    # rank systems by val AUROC
    aurocs = {c: roc_auc_score(val_y, val_df[c].values) for c in prob_cols}
    top = [c for c, _ in sorted(aurocs.items(), key=lambda kv: -kv[1])[:k]]
    feats = top + ["timepoint_is_36m"]
    lr = LogisticRegression(C=1.0, max_iter=500, random_state=SEED)
    lr.fit(val_df[feats].values, val_y)
    return (lr.predict_proba(val_df[feats].values)[:, 1],
            lr.predict_proba(test_df[feats].values)[:, 1],
            {"method": f"lr_on_top{k}_systems+tp", "selected": top})


def variant_cv_stacked(val_df, test_df, val_y, prob_cols):
    feats = prob_cols + ["timepoint_is_36m"]
    lr = LogisticRegressionCV(Cs=10, cv=5, max_iter=500, random_state=SEED, scoring="roc_auc")
    lr.fit(val_df[feats].values, val_y)
    return (lr.predict_proba(val_df[feats].values)[:, 1],
            lr.predict_proba(test_df[feats].values)[:, 1],
            {"method": "lrcv_5fold"})


def variant_bagged_stacker(val_df, test_df, val_y, prob_cols, n_boot=100):
    feats = prob_cols + ["timepoint_is_36m"]
    Xv, Xt = val_df[feats].values, test_df[feats].values
    rng = np.random.default_rng(SEED)
    test_preds = np.zeros(len(test_df))
    val_preds = np.zeros(len(val_df))
    for i in range(n_boot):
        idx = rng.integers(0, len(val_y), size=len(val_y))
        lr = LogisticRegression(C=1.0, max_iter=500, random_state=SEED + i)
        lr.fit(Xv[idx], val_y[idx])
        test_preds += lr.predict_proba(Xt)[:, 1] / n_boot
        val_preds += lr.predict_proba(Xv)[:, 1] / n_boot
    return val_preds, test_preds, {"method": f"bagged_lr_n_boot={n_boot}"}


def variant_per_timepoint(val_df, test_df, val_y, prob_cols):
    """Separate LR per timepoint cohort (14m, 36m)."""
    val_p = np.full(len(val_df), 0.5); test_p = np.full(len(test_df), 0.5)
    for tp in val_df["timepoint_norm"].unique():
        m_v = val_df["timepoint_norm"] == tp
        m_t = test_df["timepoint_norm"] == tp
        if m_v.sum() < 20 or m_t.sum() == 0: continue
        lr = LogisticRegression(C=1.0, max_iter=500, random_state=SEED)
        lr.fit(val_df.loc[m_v, prob_cols].values, val_y[m_v.values])
        val_p[m_v.values] = lr.predict_proba(val_df.loc[m_v, prob_cols].values)[:, 1]
        if m_t.sum() > 0:
            test_p[m_t.values] = lr.predict_proba(test_df.loc[m_t, prob_cols].values)[:, 1]
    return val_p, test_p, {"method": "lr_per_timepoint"}


def variant_blend_topk(val_df, test_df, val_y, prob_cols):
    """Convex blend of (pure, rank_stacker, calibrated_mean) tuned on val AUROC."""
    pure_v, pure_t, _ = variant_pure(val_df, test_df, val_y, prob_cols)
    rank_v, rank_t, _ = variant_rank_stacker(val_df, test_df, val_y, prob_cols)
    cal_v, cal_t, _ = variant_calibrated_mean(val_df, test_df, val_y, prob_cols)
    best, best_auroc = (1, 0, 0), -1.0
    for a in np.arange(0, 1.01, 0.05):
        for b in np.arange(0, 1.01 - a, 0.05):
            c = 1 - a - b
            if c < 0: continue
            blend = a * pure_v + b * rank_v + c * cal_v
            auroc = roc_auc_score(val_y, blend)
            if auroc > best_auroc: best_auroc, best = auroc, (a, b, c)
    a, b, c = best
    return (a * pure_v + b * rank_v + c * cal_v,
            a * pure_t + b * rank_t + c * cal_t,
            {"method": "convex_blend(pure,rank,calibrated_mean)", "weights": {"pure": a, "rank": b, "cal_mean": c}})


# ── main ───────────────────────────────────────────────────────────────────

VARIANTS = [
    ("mean",            "mean of all system probs"),
    ("pure",            "LR over all probs + timepoint_is_36m"),
    ("calibrated_mean", "isotonic per-system → mean"),
    ("rank_stacker",    "LR on rank-transformed probs + tp"),
    ("fp_focused",      "pure with FP class weight 5.0"),
    ("topk_systems",    "LR on top-6 systems by val AUROC"),
    ("cv_stacked",      "LRCV-5fold"),
    ("bagged_stacker",  "100-bootstrap bag of pure"),
    ("per_timepoint",   "separate LR per timepoint cohort"),
    ("blend_topk",      "convex blend of (pure, rank, cal_mean)"),
]


def main():
    print(f"=== Cross-speaker BIDS LR-stacker ensemble sweep ===")
    print(f"=== Discovering available systems ===")
    systems = discover_available_systems()
    print(f"\n  -> {len(systems)} systems available\n")
    if len(systems) < 3:
        print("ERROR: fewer than 3 systems with val+test BIDS cross-child predictions; aborting.")
        sys.exit(1)

    val_lab = load_labels("val")
    test_lab = load_labels("test")
    print(f"  BIDS cross-child: val={len(val_lab)} (pos={int(val_lab.label.sum())}), "
          f"test={len(test_lab)} (pos={int(test_lab.label.sum())})\n")

    print("=== Building score matrices ===")
    val_df, prob_cols = build_score_matrix(systems, "val", val_lab)
    test_df, _ = build_score_matrix(systems, "test", test_lab)
    print(f"  val_df shape={val_df.shape}, test_df shape={test_df.shape}, "
          f"n_systems={len(prob_cols)}\n")

    val_y = val_df.label.astype(int).values
    test_y = test_df.label.astype(int).values

    # Trivial baseline
    print("=== Running variants ===")
    save_variant("trivial_positive",
                 np.full(len(val_df), 0.99), val_y,
                 np.full(len(test_df), 0.99), test_y,
                 test_df, val_df,
                 {"method": "predict_all_positive"})

    rows = []
    for name, desc in VARIANTS:
        print(f"  ----- {name} -----  ({desc})")
        if name == "mean":
            vp, tp, extra = variant_mean(val_df, test_df, prob_cols)
        elif name == "pure":
            vp, tp, extra = variant_pure(val_df, test_df, val_y, prob_cols)
        elif name == "calibrated_mean":
            vp, tp, extra = variant_calibrated_mean(val_df, test_df, val_y, prob_cols)
        elif name == "rank_stacker":
            vp, tp, extra = variant_rank_stacker(val_df, test_df, val_y, prob_cols)
        elif name == "fp_focused":
            vp, tp, extra = variant_fp_focused(val_df, test_df, val_y, prob_cols)
        elif name == "topk_systems":
            vp, tp, extra = variant_topk_systems(val_df, test_df, val_y, prob_cols)
        elif name == "cv_stacked":
            vp, tp, extra = variant_cv_stacked(val_df, test_df, val_y, prob_cols)
        elif name == "bagged_stacker":
            vp, tp, extra = variant_bagged_stacker(val_df, test_df, val_y, prob_cols)
        elif name == "per_timepoint":
            vp, tp, extra = variant_per_timepoint(val_df, test_df, val_y, prob_cols)
        elif name == "blend_topk":
            vp, tp, extra = variant_blend_topk(val_df, test_df, val_y, prob_cols)
        else:
            raise ValueError(name)
        extra["systems_used"] = list(prob_cols)
        extra["n_systems"] = len(prob_cols)
        m = save_variant(name, vp, val_y, tp, test_y, test_df, val_df, extra)
        rows.append({"variant": name, **{k: m[k] for k in ["auroc", "balanced_accuracy", "f1", "f1_weighted", "precision", "recall", "threshold"]}})

    leaderboard = pd.DataFrame(rows).sort_values("auroc", ascending=False)
    leaderboard.to_csv(OUT_ROOT / "leaderboard.csv", index=False)
    print(f"\n=== Leaderboard (sorted by AUROC) ===")
    print(leaderboard.to_string(index=False))
    print(f"\nWrote {OUT_ROOT / 'leaderboard.csv'}")
    # Manifest of available systems
    (OUT_ROOT / "systems_manifest.json").write_text(json.dumps({
        "n_systems": len(prob_cols),
        "systems": list(prob_cols),
        "val_n": int(len(val_df)), "test_n": int(len(test_df)),
        "val_pos": int(val_y.sum()), "test_pos": int(test_y.sum()),
    }, indent=2))


if __name__ == "__main__":
    main()
