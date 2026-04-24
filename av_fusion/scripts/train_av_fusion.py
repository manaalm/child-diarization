"""Train audio-only, video-only, always-fuse AV, and gated AV fusion models.

Architecture (late fusion):
  - audio_only: thresholds BabAR enrollment probability on val; no trained model.
  - video_only: XGBoost on visual features from av_train.csv.
  - always_fuse: XGBoost on visual features; combined with audio via late-fusion alpha tuned on val.
  - gated_av: same visual model as always_fuse; at inference switches to audio-only when
              visual_eligibility_score < visual_eligibility_threshold.

Train-set audio scores are not available (BabAR was trained on the same train split),
so audio is combined at inference via a val-tuned alpha, not as a training feature.
This is standard late fusion and avoids data leakage.

Usage:
    python av_fusion/scripts/train_av_fusion.py \\
        --feature-dir av_fusion/av_results/manual_only/ \\
        --output-dir  av_fusion/av_results/manual_only/models/ \\
        --config      av_fusion/configs/av_fusion.yaml \\
        --seed        42

Exit codes:
    0 = success
    1 = feature CSV not found
    2 = no training examples after filtering
"""

import argparse
import os
import pickle
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import (
    assert_split_integrity,
    compute_metrics,
    get_repo_root,
    save_json,
    tune_late_fusion_alpha,
    tune_threshold_balanced_acc,
    tune_threshold_f1,
)

_REPO = get_repo_root()


# ---------------------------------------------------------------------------
# Model wrappers
# ---------------------------------------------------------------------------

class AudioOnlyModel:
    """Wraps pre-computed audio score with a val-tuned threshold. No fitting."""

    def __init__(self, threshold: float = 0.5) -> None:
        self.threshold = threshold
        self.model_type = "audio_only"

    def predict_proba_1d(self, audio_scores: np.ndarray) -> np.ndarray:
        return np.asarray(audio_scores, dtype=float)

    def predict(self, audio_scores: np.ndarray) -> np.ndarray:
        return (self.predict_proba_1d(audio_scores) >= self.threshold).astype(int)


class VisualXGBModel:
    """XGBoost classifier on visual features with StandardScaler preprocessing."""

    def __init__(self, xgb_params: Dict[str, Any], seed: int = 42) -> None:
        from xgboost import XGBClassifier
        self.scaler = StandardScaler()
        self.model_type = "xgb"
        self.seed = seed
        # Compute scale_pos_weight lazily at fit time
        self.xgb_params = {k: v for k, v in xgb_params.items() if k not in ("use_label_encoder", "eval_metric")}
        self.xgb_params.setdefault("random_state", seed)
        self.xgb_params.setdefault("eval_metric", "logloss")
        self.clf: Optional[Any] = None
        self.threshold: float = 0.5
        self.feature_cols: List[str] = []

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> None:
        from xgboost import XGBClassifier
        self.feature_cols = list(X.columns)
        n_pos = int(y.sum())
        n_neg = len(y) - n_pos
        scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0

        Xf = X.values.astype(float)
        self.scaler.fit(Xf)
        Xs = self.scaler.transform(Xf)

        self.clf = XGBClassifier(
            scale_pos_weight=scale_pos_weight,
            **self.xgb_params,
        )
        self.clf.fit(Xs, y)

    def predict_proba_1d(self, X: pd.DataFrame) -> np.ndarray:
        Xf = X[self.feature_cols].values.astype(float)
        Xs = self.scaler.transform(Xf)
        return self.clf.predict_proba(Xs)[:, 1]

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return (self.predict_proba_1d(X) >= self.threshold).astype(int)


class GatedAVModel:
    """Visual model with late-fusion alpha and visual eligibility gating.

    At inference:
    - If visual_eligible == 1: final_prob = alpha * audio_prob + (1-alpha) * visual_prob
    - If visual_eligible == 0: final_prob = audio_prob
    """

    def __init__(
        self,
        visual_model: VisualXGBModel,
        alpha: float,
        eligibility_threshold: float,
        threshold: float = 0.5,
        always_fuse: bool = False,
    ) -> None:
        self.visual_model = visual_model
        self.alpha = alpha
        self.eligibility_threshold = eligibility_threshold
        self.threshold = threshold
        self.always_fuse = always_fuse
        self.model_type = "gated_av" if not always_fuse else "always_fuse"

    def predict_proba_1d(
        self,
        X: pd.DataFrame,
        audio_scores: np.ndarray,
        visual_eligibility_scores: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        visual_proba = self.visual_model.predict_proba_1d(X)
        audio_scores = np.asarray(audio_scores, dtype=float)
        combined = self.alpha * audio_scores + (1.0 - self.alpha) * visual_proba

        if self.always_fuse or visual_eligibility_scores is None:
            return combined

        eligible = (np.asarray(visual_eligibility_scores, dtype=float) >= self.eligibility_threshold).astype(float)
        return eligible * combined + (1.0 - eligible) * audio_scores

    def predict(
        self,
        X: pd.DataFrame,
        audio_scores: np.ndarray,
        visual_eligibility_scores: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        return (self.predict_proba_1d(X, audio_scores, visual_eligibility_scores) >= self.threshold).astype(int)


# ---------------------------------------------------------------------------
# Feature selection helpers
# ---------------------------------------------------------------------------

def _get_feature_cols(cfg: Dict[str, Any], model_class: str, df: pd.DataFrame) -> List[str]:
    wanted = cfg.get("feature_columns", {}).get(model_class, [])
    available = [c for c in wanted if c in df.columns and not df[c].isna().all()]
    missing = [c for c in wanted if c not in available]
    if missing:
        print(f"  NOTE: {len(missing)} feature(s) missing/all-NaN for {model_class}: {missing[:5]}", flush=True)
    return available


def _prepare_X(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    X = df[cols].copy()
    for c in X.columns:
        if X[c].dtype == object:
            X[c] = pd.to_numeric(X[c], errors="coerce")
    return X


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train AV fusion models (audio-only, video-only, always-fuse, gated AV)."
    )
    parser.add_argument("--feature-dir", required=True,
                        help="Directory containing av_train.csv and av_val.csv")
    parser.add_argument("--output-dir", required=True,
                        help="Directory to save models and metrics")
    parser.add_argument("--config", default="av_fusion/configs/av_fusion.yaml")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--models", default="audio_only,video_only,always_fuse,gated_av",
                        help="Comma-separated list of model classes to train")
    args = parser.parse_args()

    feature_dir = args.feature_dir if os.path.isabs(args.feature_dir) else os.path.join(_REPO, args.feature_dir)
    output_dir = args.output_dir if os.path.isabs(args.output_dir) else os.path.join(_REPO, args.output_dir)
    config_path = args.config if os.path.isabs(args.config) else os.path.join(_REPO, args.config)

    os.makedirs(output_dir, exist_ok=True)

    for fname in ("av_train.csv", "av_val.csv"):
        if not os.path.exists(os.path.join(feature_dir, fname)):
            print(f"ERROR: {fname} not found in {feature_dir}", file=sys.stderr)
            sys.exit(1)

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    np.random.seed(args.seed)
    models_to_train = [m.strip() for m in args.models.split(",")]

    train_df = pd.read_csv(os.path.join(feature_dir, "av_train.csv"), low_memory=False)
    val_df = pd.read_csv(os.path.join(feature_dir, "av_val.csv"), low_memory=False)

    if len(train_df) == 0:
        print("ERROR: training set is empty", file=sys.stderr)
        sys.exit(2)

    y_train = train_df["label"].values.astype(int)
    y_val = val_df["label"].values.astype(int)

    xgb_params = cfg.get("xgboost", {})
    xgb_params["random_state"] = args.seed

    val_metrics: Dict[str, Any] = {}
    saved_models: Dict[str, str] = {}

    # ----- Audio-only -----
    if "audio_only" in models_to_train:
        print("Training: audio_only", flush=True)
        val_audio = val_df["existing_audio_score"].fillna(0.5).values
        has_val_audio = val_df["existing_audio_score"].notna().sum()
        print(f"  Val clips with audio scores: {has_val_audio}/{len(val_df)}", flush=True)

        thresh, val_f1 = tune_threshold_f1(y_val, val_audio)
        model = AudioOnlyModel(threshold=thresh)
        m_path = os.path.join(output_dir, "audio_only.pkl")
        with open(m_path, "wb") as f:
            pickle.dump(model, f)
        saved_models["audio_only"] = m_path

        val_preds = model.predict(val_audio)
        val_metrics["audio_only"] = compute_metrics(y_val, val_audio, thresh)
        print(f"  Val AUROC: {val_metrics['audio_only']['auroc']:.3f}, F1: {val_f1:.3f}", flush=True)

    # ----- Video-only -----
    if "video_only" in models_to_train:
        print("Training: video_only", flush=True)
        feat_cols = _get_feature_cols(cfg, "visual_only", train_df)
        if not feat_cols:
            print("  WARNING: no visual features available; skipping video_only", flush=True)
        else:
            X_train = _prepare_X(train_df, feat_cols)
            X_val = _prepare_X(val_df, feat_cols)
            model = VisualXGBModel(xgb_params, seed=args.seed)
            model.fit(X_train, y_train)
            val_proba = model.predict_proba_1d(X_val)
            thresh, _ = tune_threshold_f1(y_val, val_proba)
            model.threshold = thresh
            m_path = os.path.join(output_dir, "video_only.pkl")
            with open(m_path, "wb") as f:
                pickle.dump(model, f)
            saved_models["video_only"] = m_path
            val_metrics["video_only"] = compute_metrics(y_val, val_proba, thresh)
            print(f"  Val AUROC: {val_metrics['video_only']['auroc']:.3f}", flush=True)

    # ----- Always-fuse AV (late fusion) -----
    if "always_fuse" in models_to_train or "gated_av" in models_to_train:
        print("Training: always_fuse / gated_av visual model", flush=True)
        feat_cols = _get_feature_cols(cfg, "always_fuse", train_df)
        if not feat_cols:
            feat_cols = _get_feature_cols(cfg, "visual_only", train_df)
        X_train = _prepare_X(train_df, feat_cols)
        X_val = _prepare_X(val_df, feat_cols)
        vis_model = VisualXGBModel(xgb_params, seed=args.seed)
        vis_model.fit(X_train, y_train)
        vis_val_proba = vis_model.predict_proba_1d(X_val)

        val_audio = val_df["existing_audio_score"].fillna(0.5).values

        # Tune late-fusion alpha on val
        alpha, alpha_auroc = tune_late_fusion_alpha(y_val, val_audio, vis_val_proba)
        print(f"  Late-fusion alpha={alpha:.2f} (val AUROC={alpha_auroc:.3f})", flush=True)

        # Tune visual eligibility threshold
        elig_scores = val_df["visual_eligibility_score"].fillna(0.0).values
        elig_proxy = val_df["child_of_interest_clear_binary"].fillna(0.0).values
        elig_thresh, elig_bacc = tune_threshold_balanced_acc(elig_proxy.astype(int), elig_scores)
        print(f"  Visual eligibility threshold={elig_thresh:.2f} (val balanced_acc={elig_bacc:.3f})", flush=True)

        save_json(
            {"threshold": elig_thresh, "val_balanced_acc": elig_bacc},
            os.path.join(output_dir, "visual_eligibility_threshold.json"),
        )

        # Always-fuse model
        if "always_fuse" in models_to_train:
            fuse_proba = alpha * val_audio + (1.0 - alpha) * vis_val_proba
            thresh, _ = tune_threshold_f1(y_val, fuse_proba)
            always_fuse_model = GatedAVModel(
                vis_model, alpha=alpha, eligibility_threshold=elig_thresh,
                threshold=thresh, always_fuse=True,
            )
            m_path = os.path.join(output_dir, "always_fuse_av.pkl")
            with open(m_path, "wb") as f:
                pickle.dump(always_fuse_model, f)
            saved_models["always_fuse"] = m_path
            val_metrics["always_fuse"] = compute_metrics(y_val, fuse_proba, thresh)
            print(f"  always_fuse val AUROC: {val_metrics['always_fuse']['auroc']:.3f}", flush=True)

        # Gated AV model
        if "gated_av" in models_to_train:
            elig_mask = (elig_scores >= elig_thresh).astype(float)
            combined = elig_mask * (alpha * val_audio + (1.0 - alpha) * vis_val_proba) + (1.0 - elig_mask) * val_audio
            thresh, _ = tune_threshold_f1(y_val, combined)
            gated_model = GatedAVModel(
                vis_model, alpha=alpha, eligibility_threshold=elig_thresh,
                threshold=thresh, always_fuse=False,
            )
            m_path = os.path.join(output_dir, "gated_av.pkl")
            with open(m_path, "wb") as f:
                pickle.dump(gated_model, f)
            saved_models["gated_av"] = m_path
            val_metrics["gated_av"] = compute_metrics(y_val, combined, thresh)
            print(f"  gated_av val AUROC: {val_metrics['gated_av']['auroc']:.3f}", flush=True)

    # Save val metrics and config
    save_json(val_metrics, os.path.join(output_dir, "val_metrics.json"))

    config_out = {
        "run": {
            "seed": args.seed,
            "feature_dir": feature_dir,
            "models_trained": models_to_train,
        },
        "xgboost": xgb_params,
        "feature_columns": cfg.get("feature_columns", {}),
        "late_fusion_alpha": alpha if ("always_fuse" in models_to_train or "gated_av" in models_to_train) else None,
        "models_saved": saved_models,
    }
    save_json(config_out, os.path.join(output_dir, "config.json"))

    print(f"\nModels saved to: {output_dir}")
    print("Val AUROC summary:")
    for name, m in val_metrics.items():
        print(f"  {name}: {m.get('auroc', float('nan')):.3f}")


if __name__ == "__main__":
    main()
