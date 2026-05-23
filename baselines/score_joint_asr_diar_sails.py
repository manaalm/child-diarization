"""Clip-level SAILS scorer for the joint ASR+diar predictions.

Reads per-clip RTTMs from `<results-dir>/per_file_predictions/`, computes a
continuous clip-level score = sum(CHI segment duration) / clip duration,
val-tunes the threshold for maximum F1, and writes out the standard
`{val,test}_predictions.csv`, `{val,test}_metrics_tuned.json`, and
`enroll_test_metrics_by_timepoint.csv` files used by every other diarizer
in this project.

Usage:
    python baselines/score_joint_asr_diar_sails.py \
        --results-dir pyannote/eval_results/joint_asr_diar_sails \
        --val-csv whisper-modeling/seen_child_splits/val.csv \
        --test-csv whisper-modeling/seen_child_splits/test.csv \
        --output-dir joint_asr_diar_sails_runs
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import librosa
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def parse_rttm_chi_duration(rttm_path: str) -> float:
    """Return total seconds of CHI-labeled segments in the RTTM, or 0.0 if
    the file is missing / malformed."""
    if not os.path.isfile(rttm_path):
        return 0.0
    total = 0.0
    with open(rttm_path) as f:
        for line in f:
            parts = line.split()
            if not parts or parts[0] != "SPEAKER" or len(parts) < 9:
                continue
            if parts[7] != "CHI":
                continue
            try:
                total += float(parts[4])
            except ValueError:
                continue
    return total


def score_split(df: pd.DataFrame, pred_dir: str) -> pd.DataFrame:
    """Add continuous `score` column = chi_dur / clip_dur."""
    out = []
    for _, row in df.iterrows():
        audio_path = row["audio_path"]
        stem = Path(audio_path).stem
        rttm_path = os.path.join(pred_dir, f"{stem}_pred.rttm")
        chi_dur = parse_rttm_chi_duration(rttm_path)
        try:
            clip_dur = librosa.get_duration(path=audio_path)
        except Exception:
            clip_dur = 0.0
        score = chi_dur / clip_dur if clip_dur > 0 else 0.0
        out.append({
            "audio_path": audio_path,
            "child_id": row["child_id"],
            "timepoint_norm": row["timepoint_norm"],
            "label": int(row["label"]),
            "score": float(score),
            "chi_duration_sec": float(chi_dur),
            "clip_duration_sec": float(clip_dur),
            "rttm_exists": os.path.isfile(rttm_path),
        })
    return pd.DataFrame(out)


def tune_threshold(scores: np.ndarray, labels: np.ndarray) -> float:
    """Return the threshold in (0, 1) that maximizes F1 on (scores, labels)."""
    candidates = np.unique(np.concatenate([[0.0], scores, [1.0]]))
    best_t, best_f1 = 0.5, -1.0
    for t in candidates:
        preds = (scores >= t).astype(int)
        if preds.sum() == 0 and labels.sum() == 0:
            f = 0.0
        else:
            f = f1_score(labels, preds, zero_division=0)
        if f > best_f1:
            best_f1, best_t = f, float(t)
    return best_t


def metrics_at_threshold(scores: np.ndarray, labels: np.ndarray, t: float) -> dict:
    preds = (scores >= t).astype(int)
    n_pos = int(labels.sum())
    n_neg = int(len(labels) - n_pos)
    out = {
        "n": int(len(labels)),
        "n_pos": n_pos,
        "n_neg": n_neg,
        "threshold": float(t),
        "f1": float(f1_score(labels, preds, zero_division=0)),
        "precision": float(precision_score(labels, preds, zero_division=0)),
        "recall": float(recall_score(labels, preds, zero_division=0)),
    }
    if 0 < labels.sum() < len(labels):
        out["auroc"] = float(roc_auc_score(labels, scores))
        out["auprc"] = float(average_precision_score(labels, scores))
    else:
        out["auroc"] = float("nan")
        out["auprc"] = float("nan")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", required=True,
                    help="Dir containing per_file_predictions/<stem>_pred.rttm")
    ap.add_argument("--val-csv", required=True)
    ap.add_argument("--test-csv", required=True)
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()

    pred_dir = os.path.join(args.results_dir, "per_file_predictions")
    if not os.path.isdir(pred_dir):
        raise FileNotFoundError(f"missing predictions dir: {pred_dir}")
    os.makedirs(args.output_dir, exist_ok=True)

    val_df = pd.read_csv(args.val_csv)
    test_df = pd.read_csv(args.test_csv)
    val_df = val_df[val_df.get("audio_exists", True).astype(bool)]
    test_df = test_df[test_df.get("audio_exists", True).astype(bool)]

    print(f"scoring val (n={len(val_df)}) ...", flush=True)
    val_scored = score_split(val_df, pred_dir)
    print(f"scoring test (n={len(test_df)}) ...", flush=True)
    test_scored = score_split(test_df, pred_dir)

    n_val_missing = int((~val_scored["rttm_exists"]).sum())
    n_test_missing = int((~test_scored["rttm_exists"]).sum())
    print(f"missing RTTMs: val={n_val_missing}/{len(val_scored)}, "
          f"test={n_test_missing}/{len(test_scored)}", flush=True)

    val_scored.to_csv(os.path.join(args.output_dir, "val_predictions.csv"), index=False)
    test_scored.to_csv(os.path.join(args.output_dir, "test_predictions.csv"), index=False)

    threshold = tune_threshold(
        val_scored["score"].to_numpy(), val_scored["label"].to_numpy()
    )
    val_metrics = metrics_at_threshold(
        val_scored["score"].to_numpy(), val_scored["label"].to_numpy(), threshold
    )
    test_metrics = metrics_at_threshold(
        test_scored["score"].to_numpy(), test_scored["label"].to_numpy(), threshold
    )

    with open(os.path.join(args.output_dir, "val_metrics_tuned.json"), "w") as f:
        json.dump(val_metrics, f, indent=2)
    with open(os.path.join(args.output_dir, "test_metrics_tuned.json"), "w") as f:
        json.dump(test_metrics, f, indent=2)

    by_tp = []
    for tp, sub in test_scored.groupby("timepoint_norm"):
        if len(sub) == 0:
            continue
        m = metrics_at_threshold(
            sub["score"].to_numpy(), sub["label"].to_numpy(), threshold
        )
        m["timepoint"] = tp
        by_tp.append(m)
    pd.DataFrame(by_tp).to_csv(
        os.path.join(args.output_dir, "enroll_test_metrics_by_timepoint.csv"),
        index=False,
    )

    config = {
        "system": "joint_asr_diar_sails",
        "model": "AlexXu811/child-adult-joint-asr-diarization",
        "score_type": "chi_duration_fraction",
        "predictions_dir": pred_dir,
        "val_csv": args.val_csv,
        "test_csv": args.test_csv,
        "n_val": int(len(val_scored)),
        "n_test": int(len(test_scored)),
        "n_val_missing_rttm": n_val_missing,
        "n_test_missing_rttm": n_test_missing,
        "tuned_threshold": float(threshold),
    }
    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    print("=== val metrics ===")
    print(json.dumps(val_metrics, indent=2))
    print("=== test metrics ===")
    print(json.dumps(test_metrics, indent=2))
    print(f"DONE: outputs in {args.output_dir}")


if __name__ == "__main__":
    main()
