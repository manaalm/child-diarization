#!/usr/bin/env python3
"""Augment a MIL/baseline ResultDir to match spec-021 contracts/result_json_schema.md.

The legacy MIL pipeline emits {f1, precision, recall, auroc, auprc, threshold} in
test_metrics_tuned.json and {audio_path, child_id, timepoint_norm, label, score,
prediction} in test_predictions.csv. The spec-021 contract requires additional
keys (split, n_clips, tuned_threshold, val_f1, balanced_accuracy, by_timepoint)
and additional columns (clip_path, prob, pred).

This adapter is *additive*: it never removes existing keys/columns. It reads
test_metrics_by_timepoint.csv for the by_timepoint block and val_metrics_tuned.json
for val_f1, and computes balanced_accuracy from the predictions CSV.

Usage: python conform_result_dir.py <result-dir>
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import balanced_accuracy_score


def conform_metrics(d: Path) -> None:
    test_path = d / "test_metrics_tuned.json"
    if not test_path.exists():
        return
    with test_path.open() as f:
        m = json.load(f)

    pred_path = d / "test_predictions.csv"
    df_pred = pd.read_csv(pred_path) if pred_path.exists() else None

    if "split" not in m:
        m["split"] = "test"
    if "n_clips" not in m and df_pred is not None:
        m["n_clips"] = int(len(df_pred))
    if "tuned_threshold" not in m and "threshold" in m:
        m["tuned_threshold"] = m["threshold"]
    if "val_f1" not in m:
        val_path = d / "val_metrics_tuned.json"
        if val_path.exists():
            with val_path.open() as f:
                v = json.load(f)
            m["val_f1"] = v.get("f1")
    if "balanced_accuracy" not in m and df_pred is not None:
        thr = m.get("tuned_threshold", m.get("threshold", 0.5))
        score_col = "score" if "score" in df_pred.columns else "prob"
        y_pred = (df_pred[score_col] >= thr).astype(int)
        m["balanced_accuracy"] = float(balanced_accuracy_score(df_pred["label"], y_pred))
    if "by_timepoint" not in m:
        tp_csv = d / "test_metrics_by_timepoint.csv"
        if tp_csv.exists():
            tp_df = pd.read_csv(tp_csv)
            m["by_timepoint"] = {}
            for _, r in tp_df.iterrows():
                m["by_timepoint"][r["timepoint"]] = {
                    "f1": float(r["f1"]),
                    "precision": float(r["precision"]),
                    "recall": float(r["recall"]),
                    "auroc": float(r["auroc"]),
                    "auprc": float(r["auprc"]),
                    "n": int(r["n"]),
                }
    test_path.write_text(json.dumps(m, indent=2))


def conform_predictions(d: Path) -> None:
    p = d / "test_predictions.csv"
    if not p.exists():
        return
    df = pd.read_csv(p)
    changed = False
    if "clip_path" not in df.columns and "audio_path" in df.columns:
        df["clip_path"] = df["audio_path"]
        changed = True
    if "prob" not in df.columns and "score" in df.columns:
        df["prob"] = df["score"]
        changed = True
    if "pred" not in df.columns:
        if "prediction" in df.columns:
            df["pred"] = df["prediction"].astype(int)
        elif "prob" in df.columns or "score" in df.columns:
            score_col = "prob" if "prob" in df.columns else "score"
            tj = d / "test_metrics_tuned.json"
            thr = 0.5
            if tj.exists():
                with tj.open() as f:
                    mj = json.load(f)
                thr = mj.get("tuned_threshold", mj.get("threshold", 0.5))
            df["pred"] = (df[score_col] >= thr).astype(int)
        changed = True
    if changed:
        df.to_csv(p, index=False)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("result_dir", type=Path)
    args = ap.parse_args()
    if not args.result_dir.is_dir():
        print(f"not a directory: {args.result_dir}", file=sys.stderr)
        return 1
    conform_metrics(args.result_dir)
    conform_predictions(args.result_dir)
    print(f"CONFORMED {args.result_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
