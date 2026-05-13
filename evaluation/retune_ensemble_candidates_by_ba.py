"""Re-tune per-candidate thresholds on val to maximize balanced accuracy
(spec-022 polish, supersedes the earlier threshold=0.5 fallback in
split_ensemble_candidates.py).

For every score column in ensemble_runs/{val,test}_predictions.csv:
  1. Sweep thresholds 0.05–0.95 on val
  2. Pick the threshold that maximizes balanced_accuracy_score(y_val, y_pred_val)
  3. Apply that threshold to test → recompute the extended metric set
  4. Emit one row per candidate to evaluation/balanced_metrics_summary.csv
     (idempotent: replaces existing ensemble_runs/candidate_* rows)

Constitution IV compliant: threshold is val-tuned, applied to test once.
"""

import argparse
import os
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

REPO_ROOT = "/orcd/scratch/orcd/008/manaal/child-adult-diarization"
sys.path.insert(0, os.path.join(REPO_ROOT, "mil"))
from mil_utils import compute_metrics  # noqa: E402


def _tune_threshold_by_ba(y_val: list, y_score: list) -> float:
    """Sweep 0.05..0.95 step 0.05; pick threshold that maximises balanced
    accuracy on val. Ties broken toward 0.5 (closer to standard operating point)."""
    y_val_np = np.asarray(y_val, dtype=int)
    y_score_np = np.asarray(y_score, dtype=float)
    best_t = 0.5
    best_ba = -1.0
    for t in np.arange(0.05, 0.96, 0.05):
        y_pred = (y_score_np >= t).astype(int)
        ba = balanced_accuracy_score(y_val_np, y_pred)
        if ba > best_ba or (ba == best_ba and abs(t - 0.5) < abs(best_t - 0.5)):
            best_ba, best_t = ba, float(t)
    return round(best_t, 4)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--val",  default=os.path.join(REPO_ROOT, "ensemble_runs", "val_predictions.csv"))
    ap.add_argument("--test", default=os.path.join(REPO_ROOT, "ensemble_runs", "test_predictions.csv"))
    ap.add_argument("--summary", default=os.path.join(REPO_ROOT, "evaluation", "balanced_metrics_summary.csv"))
    args = ap.parse_args()

    if not os.path.exists(args.val) or not os.path.exists(args.test):
        sys.exit(f"missing {args.val} or {args.test}; re-run pyannote/ensemble_combined.py first")

    val = pd.read_csv(args.val)
    test = pd.read_csv(args.test)
    meta = {"audio_path", "label", "timepoint_norm", "child_id", "clip_id"}
    score_cols = [c for c in test.columns if c not in meta and c in val.columns]
    print(f"found {len(score_cols)} candidate score columns")

    summary = pd.read_csv(args.summary)
    # Drop any existing ensemble candidate rows (idempotent re-run)
    summary = summary[~summary["system_name"].str.startswith("ensemble_runs/candidate_")].copy()

    new_rows = []
    for col in score_cols:
        y_val = val["label"].astype(int).tolist()
        y_val_score = val[col].astype(float).tolist()
        threshold = _tune_threshold_by_ba(y_val, y_val_score)

        y_test = test["label"].astype(int).tolist()
        y_test_score = test[col].astype(float).tolist()
        m = compute_metrics(y_test, y_test_score, threshold=threshold)
        trivial = compute_metrics(y_test, [1.0] * len(y_test), threshold=0.5)

        # Reference: BA at threshold=0.5 (the prior split_ensemble_candidates default)
        m_at_half = compute_metrics(y_test, y_test_score, threshold=0.5)

        new_rows.append({
            "system_name": f"ensemble_runs/candidate_{col}",
            "split": "seen_child_test",
            "n_clips": int(len(y_test)),
            "pos_rate": round(sum(y_test) / len(y_test), 4),
            "threshold_source": "val-tuned-by-balanced-accuracy",
            "tuned_threshold": threshold,
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
            "predictions_path": args.test,
            "metrics_json_path": "",
            "status": "OK",
            "regenerated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        })

    out = pd.concat([summary, pd.DataFrame(new_rows)], ignore_index=True)
    out = out.sort_values(["system_name", "split"])
    out.to_csv(args.summary, index=False)
    print(f"wrote {len(new_rows)} candidate rows (BA-tuned); summary now has {len(out)} total rows")

    nr = pd.DataFrame(new_rows).sort_values("balanced_accuracy", ascending=False)
    print("\n=== top 8 candidates by balanced_accuracy (val-tuned) ===")
    print(nr[["system_name", "tuned_threshold", "f1", "balanced_accuracy", "auroc"]].head(8).to_string(index=False))


if __name__ == "__main__":
    main()
