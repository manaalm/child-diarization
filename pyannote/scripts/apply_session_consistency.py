"""Session-level consistency constraint for BabAR/ECAPA enrollment (spec-021 US4 T079).

For each BIDS session (= unique (child_id, timepoint_norm) pair), consult the
face re-id cluster assignments from `face_reid_session.py` and require the
target-child cluster to match across all clips in that session.

The constraint operates *post-hoc* on existing test predictions: for any clip
whose target-child face cluster differs from the *modal* target-child cluster
within its session, we down-weight (or zero) its positive prediction to suppress
within-session false positives.

Output: a constrained predictions CSV + recomputed metrics JSON.

CLI:
    python pyannote/scripts/apply_session_consistency.py \
        --predictions pyannote/babar_ecapa_child_enrollment_runs/enroll_test_predictions.csv \
        --face-clusters av_fusion/face_reid/session_clusters.csv \
        --out-predictions pyannote/babar_ecapa_child_enrollment_runs/enroll_test_predictions_session_constrained.csv \
        --out-metrics pyannote/babar_ecapa_child_enrollment_runs/enroll_test_metrics_session_constrained.json
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def session_id(row: pd.Series) -> str:
    return f"{row['child_id']}__{row['timepoint_norm']}"


def apply_constraint(predictions: pd.DataFrame, clusters: pd.DataFrame,
                     mode: str = "zero") -> pd.DataFrame:
    """For each session, find the modal target-child cluster across clips. Any
    clip whose face cluster mismatches the session mode gets its positive score
    zeroed (mode='zero') or halved (mode='halve').

    Predictions must have: child_id, timepoint_norm, audio_path, label, score.
    Clusters must have:    audio_path, target_child_cluster.
    """
    if "session" not in predictions.columns:
        predictions = predictions.copy()
        predictions["session"] = predictions.apply(session_id, axis=1)
    merged = predictions.merge(
        clusters[["audio_path", "target_child_cluster"]],
        on="audio_path", how="left",
    )
    # Modal cluster per session, ignoring nan.
    session_mode = (
        merged.dropna(subset=["target_child_cluster"])
        .groupby("session")["target_child_cluster"]
        .agg(lambda s: s.mode().iat[0] if not s.mode().empty else np.nan)
        .rename("session_modal_cluster")
        .reset_index()
    )
    merged = merged.merge(session_mode, on="session", how="left")
    consistent = (
        (merged["target_child_cluster"] == merged["session_modal_cluster"])
        | merged["target_child_cluster"].isna()
    )
    if mode == "zero":
        factor = consistent.astype(float)
    elif mode == "halve":
        factor = consistent.astype(float) + (~consistent).astype(float) * 0.5
    else:
        raise ValueError(f"unknown mode: {mode}")
    out = merged.copy()
    out["score_constrained"] = out["score"] * factor
    out["constraint_applied"] = (~consistent).astype(int)
    return out


def recompute_metrics(df: pd.DataFrame, threshold: float = 0.5) -> dict:
    y = df["label"].astype(int).values
    s = df["score_constrained"].values
    yhat = (s >= threshold).astype(int)
    return {
        "f1": float(f1_score(y, yhat, zero_division=0)),
        "precision": float(precision_score(y, yhat, zero_division=0)),
        "recall": float(recall_score(y, yhat, zero_division=0)),
        "auroc": float(roc_auc_score(y, s)) if len(set(y)) > 1 else None,
        "auprc": float(average_precision_score(y, s)) if len(set(y)) > 1 else None,
        "threshold": threshold,
        "n_clips": int(len(df)),
        "n_constraint_applied": int(df["constraint_applied"].sum()),
    }


def per_stratum(df: pd.DataFrame, threshold: float = 0.5) -> dict:
    out = {}
    if "n_children" in df.columns:
        multi = df[df["n_children"] > 1]
        if len(multi) > 0 and multi["label"].nunique() > 1:
            yhat = (multi["score_constrained"] >= threshold).astype(int)
            out["multi_child"] = {
                "n": int(len(multi)),
                "f1": float(f1_score(multi["label"], yhat, zero_division=0)),
                "auroc": float(roc_auc_score(multi["label"], multi["score_constrained"])),
                "auprc": float(average_precision_score(multi["label"], multi["score_constrained"])),
            }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", required=True, type=Path)
    ap.add_argument("--face-clusters", required=True, type=Path)
    ap.add_argument("--out-predictions", required=True, type=Path)
    ap.add_argument("--out-metrics", required=True, type=Path)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--mode", default="zero", choices=("zero", "halve"))
    args = ap.parse_args()

    preds = pd.read_csv(args.predictions)
    clusters = pd.read_csv(args.face_clusters)

    constrained = apply_constraint(preds, clusters, mode=args.mode)
    args.out_predictions.parent.mkdir(parents=True, exist_ok=True)
    constrained.to_csv(args.out_predictions, index=False)

    pre_metrics = recompute_metrics(
        constrained.assign(score_constrained=constrained["score"]), args.threshold
    )
    post_metrics = recompute_metrics(constrained, args.threshold)
    delta = {k: (post_metrics[k] - pre_metrics[k]) if (post_metrics[k] is not None and pre_metrics[k] is not None) else None
             for k in ("f1", "precision", "recall", "auroc", "auprc")}
    out = {
        "pre": pre_metrics,
        "post": post_metrics,
        "delta": delta,
        "by_stratum_post": per_stratum(constrained, args.threshold),
    }
    args.out_metrics.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
