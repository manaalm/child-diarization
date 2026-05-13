"""Split ensemble_runs/test_predictions.csv (multi-score candidates packed as
columns) into per-candidate balanced-metrics rows, append to
evaluation/balanced_metrics_summary.csv (spec 022 polish — closes the 1
SCHEMA_FAIL from balanced_metrics.py's standard scanner).

The top-level ensemble_runs CSV holds 16 candidate scores from the spec-012
US1 ensemble sweep (best3_mean, best3_lr, ..., best_audio_mil_mean, ...,
all_available_lr). Each candidate is a score column in [0, 1] already; no
sibling val_predictions.csv exists for tuning, so the trivial threshold=0.5
is applied uniformly. Existing dedicated subdirs (`ensemble_runs/advanced/*`,
`ensemble_runs/metadata_*`) keep their own val-tuned rows from the standard
balanced_metrics scan; this script only fills the gap left by the top-level
multi-column file.
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone

import pandas as pd

REPO_ROOT = "/orcd/scratch/orcd/008/manaal/child-adult-diarization"
sys.path.insert(0, os.path.join(REPO_ROOT, "mil"))
from mil_utils import compute_metrics  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", default=os.path.join(REPO_ROOT, "ensemble_runs", "test_predictions.csv"))
    ap.add_argument("--summary", default=os.path.join(REPO_ROOT, "evaluation", "balanced_metrics_summary.csv"))
    ap.add_argument("--threshold", type=float, default=0.5)
    args = ap.parse_args()

    if not os.path.exists(args.input):
        print(f"input missing: {args.input}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(args.input)
    required = {"audio_path", "label"}
    missing = required - set(df.columns)
    if missing:
        print(f"missing required columns: {missing}", file=sys.stderr)
        sys.exit(2)

    # Score columns: every non-meta column
    meta_cols = {"audio_path", "label", "timepoint_norm", "child_id", "clip_id"}
    score_cols = [c for c in df.columns if c not in meta_cols]
    print(f"found {len(score_cols)} ensemble candidates")

    summary = pd.read_csv(args.summary)
    new_rows = []
    for col in score_cols:
        y_true = df["label"].astype(int).tolist()
        y_score = df[col].astype(float).tolist()
        m = compute_metrics(y_true, y_score, threshold=args.threshold)
        trivial = compute_metrics(y_true, [1.0] * len(y_true), threshold=0.5)
        sys_name = f"ensemble_runs/candidate_{col}"
        new_rows.append({
            "system_name": sys_name,
            "split": "seen_child_test",
            "n_clips": int(len(y_true)),
            "pos_rate": round(sum(y_true) / len(y_true), 4),
            "threshold_source": f"untuned-{args.threshold}",
            "tuned_threshold": args.threshold,
            "f1": round(m["f1"], 4),
            "f1_macro": round(m["f1_macro"], 4),
            "f1_weighted": round(m["f1_weighted"], 4),
            "balanced_accuracy": round(m["balanced_accuracy"], 4),
            "precision": round(m["precision"], 4),
            "recall": round(m["recall"], 4),
            "auroc": round(m["auroc"], 4) if m["auroc"] == m["auroc"] else None,
            "auprc": round(m["auprc"], 4) if m["auprc"] == m["auprc"] else None,
            "trivial_f1": round(trivial["f1"], 4),
            "trivial_f1_macro": round(trivial["f1_macro"], 4),
            "trivial_balanced_accuracy": round(trivial["balanced_accuracy"], 4),
            "predictions_path": args.input,
            "metrics_json_path": "",
            "status": "OK",
            "regenerated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        })

    # Drop existing candidate rows from prior runs (idempotent re-run)
    keep = ~summary["system_name"].str.startswith("ensemble_runs/candidate_")
    summary = summary[keep].copy()

    out = pd.concat([summary, pd.DataFrame(new_rows)], ignore_index=True)
    out = out.sort_values(["system_name", "split"])
    out.to_csv(args.summary, index=False)
    print(f"appended {len(new_rows)} candidate rows; summary now has {len(out)} total rows")

    # Headline preview
    print("\n=== top 5 candidates by balanced_accuracy ===")
    nr = pd.DataFrame(new_rows).sort_values("balanced_accuracy", ascending=False)
    print(nr[["system_name", "f1", "balanced_accuracy", "auroc"]].head(5).to_string(index=False))


if __name__ == "__main__":
    main()
