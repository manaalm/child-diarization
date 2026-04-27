"""Evaluate trained AV fusion models on the held-out test split.

Produces metrics_overall.json, stratified CSVs, predictions_test.csv,
and (optionally) thesis-ready figures.

Loads val-tuned thresholds from the models directory and applies them
unchanged to the test set. No threshold re-tuning occurs here.

Usage:
    python av_fusion/scripts/evaluate_av_fusion.py \\
        --feature-dir  av_fusion/av_results/manual_only/ \\
        --model-dir    av_fusion/av_results/manual_only/models/ \\
        --output-dir   av_fusion/av_results/manual_only/ \\
        [--plot]

Exit codes:
    0 = success
    1 = model pkl not found
    2 = test feature CSV not found
"""

import argparse
import json
import os
import pickle
import sys
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import compute_metrics, get_repo_root, save_json
# Register model classes in __main__ namespace so pickle.load can find them
# (models were saved when train_av_fusion.py ran as __main__)
from train_av_fusion import AudioOnlyModel, VisualXGBModel, GatedAVModel  # noqa: F401
from train_cascaded_pipeline import assign_cascade_stages  # noqa: F401

_REPO = get_repo_root()

_MODEL_NAMES = ["audio_only", "video_only", "always_fuse", "gated_av"]


def _load_model(model_dir: str, name: str):
    pkl = os.path.join(model_dir, f"{name}.pkl")
    if not os.path.exists(pkl):
        return None
    with open(pkl, "rb") as f:
        return pickle.load(f)


def _get_probas(model, name: str, test_df: pd.DataFrame, elig_scores: np.ndarray) -> Optional[np.ndarray]:
    audio = test_df["existing_audio_score"].fillna(0.5).values

    if name == "audio_only" or (hasattr(model, "model_type") and model.model_type == "audio_only"):
        return model.predict_proba_1d(audio)

    if name == "video_only" or (hasattr(model, "model_type") and model.model_type == "xgb"):
        X = test_df[model.feature_cols].copy()
        return model.predict_proba_1d(X)

    # GatedAVModel (always_fuse or gated_av)
    X = test_df[model.visual_model.feature_cols].copy()
    if model.always_fuse:
        return model.predict_proba_1d(X, audio)
    else:
        return model.predict_proba_1d(X, audio, elig_scores)


def _metrics_for_subset(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> Dict[str, Any]:
    return {**compute_metrics(y_true, y_score, threshold), "n_clips": int(len(y_true))}


def evaluate_strata(
    test_df: pd.DataFrame,
    probas: Dict[str, np.ndarray],
    thresholds: Dict[str, float],
) -> Dict[str, pd.DataFrame]:
    """Compute metrics for standard strata. Returns dict of DataFrame per strata type."""
    results: Dict[str, List[Dict]] = {
        "age_band": [], "visual_eligibility": [], "strata": []
    }
    y = test_df["label"].values.astype(int)

    for age_band in test_df["age_band"].unique():
        mask = (test_df["age_band"] == age_band).values
        if mask.sum() < 5:
            continue
        for name, proba in probas.items():
            m = _metrics_for_subset(y[mask], proba[mask], thresholds.get(name, 0.5))
            results["age_band"].append({"model": name, "age_band": age_band, **m})

    # Visual eligibility
    if "visual_eligible" in test_df.columns:
        for eligible in [0, 1]:
            mask = (test_df["visual_eligible"] == eligible).values
            if mask.sum() < 5:
                continue
            for name, proba in probas.items():
                m = _metrics_for_subset(y[mask], proba[mask], thresholds.get(name, 0.5))
                results["visual_eligibility"].append({
                    "model": name, "visual_eligible": eligible, **m
                })

    # Ad-hoc strata
    strata_defs = {
        "off_camera_likely": lambda df: df.get("off_camera_likely_score", pd.Series([0] * len(df))) > 0.7,
        "multi_person": lambda df: df.get("multi_person_clip", pd.Series([0] * len(df))) == 1,
        "low_quality": lambda df: df.get("manual_quality_norm", pd.Series([1.0] * len(df))) < 0.5,
        "low_face_visibility": lambda df: df.get("manual_face_visibility_norm", pd.Series([1.0] * len(df))) < 0.5,
    }
    for stratum_name, mask_fn in strata_defs.items():
        try:
            mask = mask_fn(test_df).values
        except Exception:
            continue
        if mask.sum() < 5:
            continue
        for name, proba in probas.items():
            m = _metrics_for_subset(y[mask], proba[mask], thresholds.get(name, 0.5))
            results["strata"].append({"model": name, "stratum": stratum_name, **m})

    return {k: pd.DataFrame(v) for k, v in results.items()}


def generate_plots(
    test_df: pd.DataFrame,
    probas: Dict[str, np.ndarray],
    thresholds: Dict[str, float],
    output_dir: str,
) -> None:
    """Generate PR curve, ROC curve, stratified bar chart, and eligibility histogram."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.metrics import precision_recall_curve, roc_curve
    except ImportError:
        print("WARNING: matplotlib not available; skipping plots", file=sys.stderr)
        return

    figures_dir = os.path.join(output_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)
    y = test_df["label"].values.astype(int)

    colors = {"audio_only": "navy", "video_only": "green", "always_fuse": "orange", "gated_av": "red"}

    # PR Curve
    fig, ax = plt.subplots(figsize=(7, 5))
    for name, proba in probas.items():
        if np.isnan(proba).any():
            continue
        try:
            prec, rec, _ = precision_recall_curve(y, proba)
            from sklearn.metrics import average_precision_score
            ap = average_precision_score(y, proba)
            ax.plot(rec, prec, label=f"{name} (AP={ap:.3f})", color=colors.get(name))
        except Exception:
            pass
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(figures_dir, "pr_curve.png"), dpi=150)
    plt.close(fig)

    # ROC Curve
    fig, ax = plt.subplots(figsize=(7, 5))
    for name, proba in probas.items():
        if np.isnan(proba).any():
            continue
        try:
            fpr, tpr, _ = roc_curve(y, proba)
            from sklearn.metrics import roc_auc_score
            auc = roc_auc_score(y, proba)
            ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})", color=colors.get(name))
        except Exception:
            pass
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(figures_dir, "roc_curve.png"), dpi=150)
    plt.close(fig)

    # Stratified bar chart — AUROC by age band
    strata_results = evaluate_strata(test_df, probas, thresholds)
    if not strata_results["age_band"].empty:
        age_df = strata_results["age_band"]
        fig, ax = plt.subplots(figsize=(8, 5))
        bands = age_df["age_band"].unique()
        x = np.arange(len(list(probas.keys())))
        width = 0.35
        for i, band in enumerate(bands):
            sub = age_df[age_df["age_band"] == band].set_index("model")
            vals = [sub.loc[n, "auroc"] if n in sub.index else float("nan") for n in probas.keys()]
            ax.bar(x + i * width, vals, width, label=band, alpha=0.8)
        ax.set_xticks(x + width / 2)
        ax.set_xticklabels(list(probas.keys()), rotation=15)
        ax.set_ylabel("AUROC")
        ax.set_title("AUROC by Age Band and Model")
        ax.set_ylim(0, 1)
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(figures_dir, "stratified_bar_metrics.png"), dpi=150)
        plt.close(fig)

    # Visual eligibility histogram
    if "visual_eligibility_score" in test_df.columns:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for ax, label_val, label_name in [(axes[0], 0, "No vocalization"), (axes[1], 1, "Vocalization")]:
            scores = test_df[test_df["label"] == label_val]["visual_eligibility_score"].dropna()
            ax.hist(scores, bins=20, alpha=0.7, color="steelblue")
            ax.set_title(f"{label_name}")
            ax.set_xlabel("Visual Eligibility Score")
            ax.set_ylabel("Count")
            ax.grid(alpha=0.3)
        fig.suptitle("Visual Eligibility Score Distribution by Label")
        fig.tight_layout()
        fig.savefig(os.path.join(figures_dir, "visual_eligibility_histogram.png"), dpi=150)
        plt.close(fig)

    print(f"  Figures saved to {figures_dir}", flush=True)


def _evaluate_cascade(
    test_df: pd.DataFrame,
    model_dir: str,
    output_dir: str,
    feature_dir: str,
) -> Optional[np.ndarray]:
    """Apply cascade to test set using val-tuned thresholds.

    Writes cascade_stage_breakdown.csv and metrics_cascade_by_stage.csv.
    Returns final_prob array for inclusion in overall metrics, or None on failure.
    """
    thresh_path = os.path.join(model_dir, "cascade_thresholds.json")
    if not os.path.exists(thresh_path):
        print(f"  WARNING: cascade_thresholds.json not found in {model_dir}; skipping cascade", file=sys.stderr)
        return None

    with open(thresh_path) as f:
        thresholds = json.load(f)

    vad_feature = thresholds.get("vad_feature", "kchi_total_dur")
    child_id_feature = thresholds.get("child_id_feature", "prob")
    fusion_col = thresholds.get("fusion_col", "proba_gated_av")
    vad_t = thresholds["vad_threshold"]
    child_t = thresholds["child_id_threshold"]

    staged = assign_cascade_stages(
        test_df, vad_feature, child_id_feature, vad_t, child_t, fusion_col
    )

    # Write full breakdown
    stage_cols = [
        "clip_id", "label", "vad_speech_detected", "vad_child_dur_sec",
        "child_id_score", "av_fusion_prob", "cascade_stage", "final_prob",
        "vad_threshold", "child_id_threshold",
    ]
    out_cols = [c for c in stage_cols if c in staged.columns]
    breakdown_path = os.path.join(output_dir, "cascade_stage_breakdown.csv")
    staged[out_cols].to_csv(breakdown_path, index=False)
    print(f"  Cascade stage breakdown → {breakdown_path}")

    # Per-stage metrics
    y_true = test_df["label"].values.astype(int)
    final_prob = staged["final_prob"].values
    stage_rows = []
    for stage in [1, 2, 3]:
        mask = staged["cascade_stage"].values == stage
        if mask.sum() < 2:
            continue
        try:
            auroc = float(roc_auc_score(y_true[mask], final_prob[mask]))
        except Exception:
            auroc = float("nan")
        from sklearn.metrics import f1_score as _f1
        preds = (final_prob[mask] >= 0.5).astype(int)
        f1 = float(_f1(y_true[mask], preds, zero_division=0))
        stage_rows.append({
            "cascade_stage": stage,
            "n_clips": int(mask.sum()),
            "auroc": auroc,
            "f1": f1,
        })
    if stage_rows:
        by_stage_path = os.path.join(output_dir, "metrics_cascade_by_stage.csv")
        pd.DataFrame(stage_rows).to_csv(by_stage_path, index=False)
        print(f"  Cascade by-stage metrics → {by_stage_path}")
        for row in stage_rows:
            print(f"    Stage {row['cascade_stage']}: n={row['n_clips']}  AUROC={row['auroc']:.3f}  F1={row['f1']:.3f}")

    return final_prob


def _evaluate_smoothed(
    test_df: pd.DataFrame,
    smoothed_csv: str,
    output_dir: str,
) -> None:
    """Compare raw vs smoothed prediction metrics; write metrics_smoothed.csv."""
    if not os.path.exists(smoothed_csv):
        print(f"  WARNING: smoothed predictions file not found: {smoothed_csv}", file=sys.stderr)
        return

    smooth_df = pd.read_csv(smoothed_csv, low_memory=False)
    if "prob_smoothed" not in smooth_df.columns:
        print(
            "  WARNING: --smoothed-predictions CSV has no 'prob_smoothed' column. "
            "Run smooth_predictions.py first.",
            file=sys.stderr,
        )
        return

    # Merge on clip_id if available
    if "clip_id" in test_df.columns and "clip_id" in smooth_df.columns:
        merged = test_df[["clip_id", "label"]].merge(
            smooth_df[["clip_id", "prob_smoothed"]], on="clip_id", how="left"
        )
    else:
        merged = smooth_df.copy()
        if "label" not in merged.columns and "label" in test_df.columns:
            merged["label"] = test_df["label"].values

    y = merged["label"].values.astype(int)
    prob_raw = merged.get("prob_raw", merged.get("prob", pd.Series([float("nan")] * len(merged)))).values
    prob_smoothed = merged["prob_smoothed"].values

    rows = []
    for label, probs in [("raw", prob_raw), ("smoothed", prob_smoothed)]:
        valid = ~np.isnan(probs)
        if valid.sum() < 2:
            continue
        try:
            auroc = float(roc_auc_score(y[valid], probs[valid]))
        except Exception:
            auroc = float("nan")
        from sklearn.metrics import f1_score as _f1
        preds = (probs[valid] >= 0.5).astype(int)
        f1 = float(_f1(y[valid], preds, zero_division=0))
        rows.append({"type": label, "auroc": auroc, "f1": f1, "n_clips": int(valid.sum())})

    if rows:
        out_path = os.path.join(output_dir, "metrics_smoothed.csv")
        pd.DataFrame(rows).to_csv(out_path, index=False)
        print(f"  Smoothed metrics → {out_path}")
        for row in rows:
            print(f"    {row['type']}: AUROC={row['auroc']:.3f}  F1={row['f1']:.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate AV fusion models on the held-out test split."
    )
    parser.add_argument("--feature-dir", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--models",
                        default="audio_only,video_only,always_fuse,gated_av")
    parser.add_argument("--cascade-breakdown", default=None,
                        help="If provided, evaluate cascade model using cascade_thresholds.json from model-dir")
    parser.add_argument("--smoothed-predictions", default=None,
                        help="Path to smoothed predictions CSV (output of smooth_predictions.py)")
    parser.add_argument("--eval-val", action="store_true",
                        help="Also run evaluation on av_val.csv and write predictions_val.csv")
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()

    feature_dir = args.feature_dir if os.path.isabs(args.feature_dir) else os.path.join(_REPO, args.feature_dir)
    model_dir = args.model_dir if os.path.isabs(args.model_dir) else os.path.join(_REPO, args.model_dir)
    output_dir = args.output_dir if os.path.isabs(args.output_dir) else os.path.join(_REPO, args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    test_csv = os.path.join(feature_dir, "av_test.csv")
    if not os.path.exists(test_csv):
        print(f"ERROR: av_test.csv not found in {feature_dir}", file=sys.stderr)
        sys.exit(2)

    test_df = pd.read_csv(test_csv, low_memory=False)
    y = test_df["label"].values.astype(int)
    models_to_eval = [m.strip() for m in args.models.split(",")]

    # Load visual eligibility threshold
    elig_thresh = 0.5
    elig_path = os.path.join(model_dir, "visual_eligibility_threshold.json")
    if os.path.exists(elig_path):
        with open(elig_path) as f:
            elig_thresh = json.load(f)["threshold"]

    elig_scores = test_df.get("visual_eligibility_score", pd.Series([0.0] * len(test_df))).fillna(0.0).values
    test_df["visual_eligible"] = (elig_scores >= elig_thresh).astype(int)

    # Load models and compute probabilities
    probas: Dict[str, np.ndarray] = {}
    thresholds: Dict[str, float] = {}

    for name in models_to_eval:
        model = _load_model(model_dir, name if name != "always_fuse" else "always_fuse_av")
        if model is None:
            print(f"  WARNING: model not found for '{name}'; skipping", file=sys.stderr)
            continue
        try:
            p = _get_probas(model, name, test_df, elig_scores)
        except Exception as e:
            print(f"  WARNING: failed to get probas for '{name}': {e}", file=sys.stderr)
            continue
        probas[name] = p if p is not None else np.full(len(test_df), float("nan"))
        thresholds[name] = getattr(model, "threshold", 0.5)

    if not probas:
        print("ERROR: no models could produce predictions", file=sys.stderr)
        sys.exit(1)

    # Overall metrics
    overall: Dict[str, Any] = {}
    for name, proba in probas.items():
        overall[name] = compute_metrics(y, proba, thresholds[name])
        overall[name]["n_clips"] = int(len(y))
    save_json(overall, os.path.join(output_dir, "metrics_overall.json"))

    print("\nTest metrics (overall):")
    for name, m in overall.items():
        print(f"  {name}: AUROC={m.get('auroc', float('nan')):.3f}  AUPRC={m.get('auprc', float('nan')):.3f}  F1={m.get('f1', float('nan')):.3f}")

    # Predictions CSV
    pred_df = test_df[["clip_id", "child_id", "age_band", "split", "label"]].copy() if "clip_id" in test_df.columns else test_df[["child_id", "age_band", "split", "label"]].copy()
    pred_df["visual_eligible"] = test_df["visual_eligible"].values
    for name, proba in probas.items():
        pred_df[f"proba_{name}"] = proba
        pred_df[f"pred_{name}"] = (proba >= thresholds[name]).astype(int)
    pred_df.to_csv(os.path.join(output_dir, "predictions_test.csv"), index=False)

    # Optional val predictions (needed by smooth_predictions.py)
    if args.eval_val:
        val_csv = os.path.join(feature_dir, "av_val.csv")
        if os.path.exists(val_csv):
            val_df = pd.read_csv(val_csv, low_memory=False)
            val_elig = val_df.get("visual_eligibility_score", pd.Series([0.0] * len(val_df))).fillna(0.0).values
            val_df["visual_eligible"] = (val_elig >= elig_thresh).astype(int)
            val_pred_df = val_df[["clip_id", "child_id", "age_band", "split", "label"]].copy() if "clip_id" in val_df.columns else val_df[["child_id", "age_band", "split", "label"]].copy()
            val_pred_df["visual_eligible"] = val_df["visual_eligible"].values
            for name in models_to_eval:
                pkl_name = name if name != "always_fuse" else "always_fuse_av"
                model = _load_model(model_dir, pkl_name)
                if model is None:
                    continue
                try:
                    p = _get_probas(model, name, val_df, val_elig)
                except Exception:
                    p = None
                val_pred_df[f"proba_{name}"] = p if p is not None else float("nan")
                val_pred_df[f"pred_{name}"] = (val_pred_df[f"proba_{name}"] >= thresholds.get(name, 0.5)).astype(int)
            val_pred_df.to_csv(os.path.join(output_dir, "predictions_val.csv"), index=False)
            print(f"Val predictions written: {len(val_pred_df)} clips")
        else:
            print(f"  WARNING: --eval-val set but {val_csv} not found; skipping", file=sys.stderr)

    # Stratified metrics
    strata_results = evaluate_strata(test_df, probas, thresholds)
    if not strata_results["age_band"].empty:
        strata_results["age_band"].to_csv(os.path.join(output_dir, "metrics_by_age_band.csv"), index=False)
    if not strata_results["visual_eligibility"].empty:
        strata_results["visual_eligibility"].to_csv(os.path.join(output_dir, "metrics_by_visual_eligibility.csv"), index=False)
    if not strata_results["strata"].empty:
        strata_results["strata"].to_csv(os.path.join(output_dir, "metrics_by_strata.csv"), index=False)

    # Cascade evaluation (007 extension)
    if args.cascade_breakdown is not None:
        print("\nEvaluating cascade pipeline...")
        # Add proba_gated_av to test_df if not already present
        if "proba_gated_av" not in test_df.columns and "gated_av" in probas:
            test_df = test_df.copy()
            test_df["proba_gated_av"] = probas["gated_av"]
        cascade_prob = _evaluate_cascade(test_df, model_dir, output_dir, feature_dir)
        if cascade_prob is not None:
            probas["cascaded_av"] = cascade_prob
            from sklearn.metrics import f1_score as _f1
            cascade_thresh = 0.5
            try:
                from sklearn.metrics import roc_auc_score as _auroc
                cascade_auroc = float(_auroc(y, cascade_prob))
            except Exception:
                cascade_auroc = float("nan")
            cascade_preds = (cascade_prob >= cascade_thresh).astype(int)
            cascade_f1 = float(_f1(y, cascade_preds, zero_division=0))
            overall["cascaded_av"] = {
                "auroc": cascade_auroc, "f1": cascade_f1, "n_clips": int(len(y))
            }
            save_json(overall, os.path.join(output_dir, "metrics_overall.json"))
            print(f"  cascaded_av: AUROC={cascade_auroc:.3f}  F1={cascade_f1:.3f}")

    # Smoothed predictions evaluation (007 extension)
    if args.smoothed_predictions is not None:
        smoothed_path = (args.smoothed_predictions if os.path.isabs(args.smoothed_predictions)
                         else os.path.join(_REPO, args.smoothed_predictions))
        print("\nEvaluating smoothed predictions...")
        _evaluate_smoothed(test_df, smoothed_path, output_dir)

    # Plots
    if args.plot:
        generate_plots(test_df, probas, thresholds, output_dir)

    print(f"\nResults written to: {output_dir}")


if __name__ == "__main__":
    main()
