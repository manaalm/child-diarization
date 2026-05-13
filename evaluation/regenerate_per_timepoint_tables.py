"""Regenerate every test_metrics_by_timepoint.csv from cached predictions, using
the BIDS-corrected timepoint mapping (spec 022 US1 / FR-004).

Workflow:
  1. Load the new BIDS-corrected master_with_split.csv as the canonical
     audio_path -> timepoint_norm mapping.
  2. For every test_predictions.csv (or enroll_test_predictions.csv) under the
     repo, join on audio_path, overwrite the file's existing timepoint column
     with the BIDS-corrected value, recompute per-timepoint metrics via
     mil/mil_utils.compute_metrics() (which now also returns f1_weighted +
     balanced_accuracy), and write the result to the sibling
     test_metrics_by_timepoint.csv (or enroll_test_metrics_by_timepoint.csv).
  3. Back up any pre-existing per-timepoint CSV to .legacy_pre_bids_022 before
     overwriting (Constitution VI: no silent overwrites).

No model is re-run. Predictions made by each system on the legacy 441-row test
set are reused; rows in the new BIDS-derived test set that no system has
predicted yet (the +194 rows recovered by the BIDS correction) are simply
absent from each system's prediction file and therefore from its per-timepoint
table. Those rows are picked up later by US3's universal-coverage evaluation.
"""

import argparse
import json
import os
import shutil
import sys
from typing import Optional, Tuple

import pandas as pd

REPO_ROOT = "/orcd/scratch/orcd/008/manaal/child-adult-diarization"
DEFAULT_MASTER = os.path.join(
    REPO_ROOT, "whisper-modeling", "seen_child_splits", "master_with_split.csv"
)

sys.path.insert(0, os.path.join(REPO_ROOT, "mil"))
from mil_utils import compute_metrics  # noqa: E402


def _detect_columns(df: pd.DataFrame) -> Tuple[str, str, str, str]:
    """Return (audio_path_col, label_col, score_col, prediction_col)."""
    audio_col = "audio_path" if "audio_path" in df.columns else None
    label_col = "label" if "label" in df.columns else None
    score_col = None
    for cand in ("score", "prob", "fused_score", "p_child_voc", "joint_score"):
        if cand in df.columns:
            score_col = cand
            break
    prediction_col = None
    for cand in ("prediction", "pred_label", "predicted", "pred"):
        if cand in df.columns:
            prediction_col = cand
            break
    return audio_col, label_col, score_col, prediction_col


def _per_timepoint_metrics(preds: pd.DataFrame, score_col: str, prediction_col: str) -> pd.DataFrame:
    """Compute per-timepoint metrics, applying each row's existing prediction
    column verbatim (no re-thresholding — the original threshold tuning is
    preserved). Returns a DataFrame matching the legacy schema plus the new
    imbalance-aware columns."""
    rows = []
    for tp, grp in preds.groupby("timepoint_norm", dropna=True):
        # mil_utils.compute_metrics takes score+threshold; here predictions are
        # already binarised. Pass threshold=0.5 and use the prediction column
        # by encoding it into the score channel — but for AUROC/AUPRC we still
        # need the raw scores. So compute prediction-based metrics manually and
        # AUROC/AUPRC from raw scores via compute_metrics.
        y_true = grp[label_col_global].astype(int).tolist()
        y_score = grp[score_col].astype(float).tolist()
        y_pred = grp[prediction_col].astype(int).tolist()

        # Use compute_metrics to get auroc/auprc + imbalance-aware metrics
        # from the prediction column, by faking the score=prediction.
        m_pred_based = compute_metrics(y_true, [float(p) for p in y_pred], threshold=0.5)
        m_score_based = compute_metrics(y_true, y_score, threshold=0.5)

        rows.append({
            "f1": m_pred_based["f1"],
            "f1_macro": m_pred_based["f1_macro"],
            "f1_weighted": m_pred_based["f1_weighted"],
            "balanced_accuracy": m_pred_based["balanced_accuracy"],
            "precision": m_pred_based["precision"],
            "recall": m_pred_based["recall"],
            "auroc": m_score_based["auroc"],
            "auprc": m_score_based["auprc"],
            "timepoint": tp,
            "n": int(len(grp)),
        })
    return pd.DataFrame(rows)


label_col_global = "label"  # set per file in the loop


def _regen_one(pred_path: str, master_df: pd.DataFrame, dry_run: bool = False) -> Optional[dict]:
    global label_col_global
    out_dir = os.path.dirname(pred_path)
    # Determine the sibling per-timepoint filename
    if pred_path.endswith("/enroll_test_predictions.csv"):
        out_path = os.path.join(out_dir, "enroll_test_metrics_by_timepoint.csv")
    else:
        out_path = os.path.join(out_dir, "test_metrics_by_timepoint.csv")

    try:
        preds = pd.read_csv(pred_path)
    except Exception as e:
        return {"path": pred_path, "status": "READ_FAIL", "error": str(e)}

    audio_col, label_col, score_col, prediction_col = _detect_columns(preds)
    if not all([audio_col, label_col, score_col, prediction_col]):
        return {
            "path": pred_path,
            "status": "SCHEMA_FAIL",
            "missing": [name for name, val in zip(
                ["audio_path", "label", "score", "prediction"],
                [audio_col, label_col, score_col, prediction_col]
            ) if val is None],
        }
    label_col_global = label_col

    # Join: bring BIDS-corrected timepoint_norm from master onto predictions
    merge = preds.merge(
        master_df[[audio_col, "timepoint_norm"]].rename(columns={"timepoint_norm": "_bids_tp"}),
        on=audio_col, how="left",
    )
    matched = merge["_bids_tp"].notna().sum()
    n_total = len(merge)
    # Overwrite the timepoint column with BIDS-corrected values; drop rows we couldn't map
    merge["timepoint_norm"] = merge["_bids_tp"]
    merge = merge.dropna(subset=["timepoint_norm"])

    if merge.empty:
        return {"path": pred_path, "status": "EMPTY_AFTER_JOIN", "n_total": n_total, "matched": int(matched)}

    table = _per_timepoint_metrics(merge, score_col, prediction_col)

    # Backup existing per-timepoint file if present
    if os.path.exists(out_path) and not dry_run:
        backup = out_path + ".legacy_pre_bids_022"
        if not os.path.exists(backup):
            shutil.copyfile(out_path, backup)

    if not dry_run:
        table.to_csv(out_path, index=False)

    return {
        "path": pred_path,
        "out": out_path,
        "status": "OK",
        "n_predictions": int(n_total),
        "n_matched_to_bids": int(matched),
        "per_timepoint": table.to_dict(orient="records"),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--master", default=DEFAULT_MASTER,
                   help=f"path to BIDS-corrected master_with_split.csv (default {DEFAULT_MASTER})")
    p.add_argument("--root", default=REPO_ROOT,
                   help="repo root to scan for test_predictions.csv")
    p.add_argument("--dry-run", action="store_true",
                   help="report what would change without writing")
    p.add_argument("--summary-out",
                   default=os.path.join(REPO_ROOT, "specs", "022-pi-thesis-revisions",
                                        "regenerate_per_timepoint_summary.json"))
    args = p.parse_args()

    master = pd.read_csv(args.master)
    print(f"loaded master: {len(master)} rows", file=sys.stderr)

    # Find prediction files: both test_predictions.csv and enroll_test_predictions.csv
    pred_files = []
    for root, _, files in os.walk(args.root):
        for name in files:
            if name in ("test_predictions.csv", "enroll_test_predictions.csv"):
                pred_files.append(os.path.join(root, name))
    print(f"discovered {len(pred_files)} prediction files", file=sys.stderr)

    results = []
    for pf in pred_files:
        r = _regen_one(pf, master, dry_run=args.dry_run)
        if r is not None:
            results.append(r)

    # Aggregate summary
    summary = {
        "n_prediction_files": len(pred_files),
        "n_regenerated_ok": sum(1 for r in results if r["status"] == "OK"),
        "n_schema_fail": sum(1 for r in results if r["status"] == "SCHEMA_FAIL"),
        "n_read_fail": sum(1 for r in results if r["status"] == "READ_FAIL"),
        "n_empty_after_join": sum(1 for r in results if r["status"] == "EMPTY_AFTER_JOIN"),
        "dry_run": args.dry_run,
        "master": args.master,
        "failures": [r for r in results if r["status"] != "OK"],
    }
    os.makedirs(os.path.dirname(args.summary_out), exist_ok=True)
    with open(args.summary_out, "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps({k: v for k, v in summary.items() if k != "failures"}, indent=2))
    if summary["failures"]:
        print(f"\n{len(summary['failures'])} failures — see {args.summary_out}")


if __name__ == "__main__":
    main()
