"""Re-tune EVERY system's threshold by balanced accuracy on val and apply
on test, instead of the val-tuned-F1 threshold that's currently used
across the catalog. Writes a parallel summary at
`evaluation/balanced_metrics_ba_tuned_summary.csv` so the F1-tuned
view (`balanced_metrics_summary.csv`) is preserved for cross-reference.

Per Constitution IV: threshold tuned on val only, applied to test once.

Scope: every directory under repo root that has a (val, test) prediction
pair — variants:
  - val_predictions.csv + test_predictions.csv  (MIL, pseudo_frame, ensembles, scene_analysis, audio_llm)
  - enroll_val_predictions.csv + enroll_test_predictions.csv (enrollment runs)
  - + universal-coverage test_all_predictions.csv where val_predictions exists

Score column auto-detected from {score, prob, fused_score, p_child_voc, joint_score}.
Threshold sweep: 0.05..0.95 step 0.05. Tie-break: closest to 0.5.

Output schema mirrors balanced_metrics_summary.csv plus a new column
`f1_at_f1_tuned_threshold` (the legacy F1 value) so the F1→BA tradeoff
is auditable in one row.
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

REPO_ROOT = "/orcd/scratch/orcd/008/manaal/child-adult-diarization"
sys.path.insert(0, os.path.join(REPO_ROOT, "mil"))
from mil_utils import compute_metrics  # noqa: E402

SCORE_COL_PREFS = ("score", "prob", "fused_score", "p_child_voc", "joint_score")
META = {"audio_path", "label", "timepoint_norm", "child_id", "clip_id"}


def _find_score_col(df: pd.DataFrame) -> str | None:
    for c in SCORE_COL_PREFS:
        if c in df.columns:
            return c
    return None


def _tune_by_ba(y_val: np.ndarray, y_score: np.ndarray) -> float:
    best_t, best_ba = 0.5, -1.0
    for t in np.arange(0.05, 0.96, 0.05):
        y_pred = (y_score >= t).astype(int)
        ba = balanced_accuracy_score(y_val, y_pred)
        if ba > best_ba or (ba == best_ba and abs(t - 0.5) < abs(best_t - 0.5)):
            best_ba, best_t = ba, float(t)
    return round(best_t, 4)


def _load_legacy_f1_threshold(sys_dir: str, is_enroll: bool) -> float | None:
    # Enrollment runs store the val-tuned threshold under either
    # enroll_val_metrics_tuned.json or just enroll_val_metrics.json.
    # Non-enrollment systems use val_metrics_tuned.json (or rarely val_metrics.json).
    candidates = (("enroll_val_metrics_tuned.json", "enroll_val_metrics.json",
                   "val_metrics_tuned.json", "val_metrics.json")
                  if is_enroll
                  else ("val_metrics_tuned.json", "val_metrics.json"))
    for fname in candidates:
        p = os.path.join(sys_dir, fname)
        if os.path.exists(p):
            try:
                d = json.load(open(p))
                if "threshold" in d:
                    return float(d["threshold"])
            except Exception:
                pass
    return None


def _process(val_path: str, test_path: str, is_enroll: bool, split_tag: str) -> dict | None:
    sys_dir = os.path.dirname(val_path)
    try:
        val_df = pd.read_csv(val_path)
        test_df = pd.read_csv(test_path)
    except Exception as e:
        return {"system_name": os.path.relpath(sys_dir, REPO_ROOT),
                "split": split_tag, "status": "READ_FAIL", "error": str(e)}

    sc_val = _find_score_col(val_df)
    sc_test = _find_score_col(test_df)
    if sc_val is None or sc_test is None or "label" not in val_df.columns or "label" not in test_df.columns:
        return {"system_name": os.path.relpath(sys_dir, REPO_ROOT),
                "split": split_tag, "status": "SCHEMA_FAIL",
                "val_score_col": sc_val, "test_score_col": sc_test}

    y_val = val_df["label"].astype(int).to_numpy()
    y_val_score = val_df[sc_val].astype(float).to_numpy()
    threshold_ba = _tune_by_ba(y_val, y_val_score)

    y_test = test_df["label"].astype(int).to_numpy()
    y_test_score = test_df[sc_test].astype(float).to_numpy()
    m_ba = compute_metrics(y_test.tolist(), y_test_score.tolist(), threshold=threshold_ba)

    # Reference: the legacy F1-tuned threshold + its F1
    threshold_f1 = _load_legacy_f1_threshold(sys_dir, is_enroll)
    m_f1 = None
    if threshold_f1 is not None:
        m_f1 = compute_metrics(y_test.tolist(), y_test_score.tolist(), threshold=threshold_f1)

    trivial = compute_metrics(y_test.tolist(), [1.0] * len(y_test), threshold=0.5)

    return {
        "system_name": os.path.relpath(sys_dir, REPO_ROOT),
        "split": split_tag,
        "n_clips": int(len(y_test)),
        "pos_rate": round(float(y_test.mean()), 4),
        "threshold_source": "val-tuned-by-balanced-accuracy",
        "ba_tuned_threshold": threshold_ba,
        "f1_tuned_threshold": threshold_f1 if threshold_f1 is not None else "",
        # Metrics at BA-tuned threshold (this is the new headline)
        "f1": round(m_ba["f1"], 4),
        "f1_macro": round(m_ba["f1_macro"], 4),
        "f1_weighted": round(m_ba["f1_weighted"], 4),
        "balanced_accuracy": round(m_ba["balanced_accuracy"], 4),
        "precision": round(m_ba["precision"], 4),
        "recall": round(m_ba["recall"], 4),
        "auroc": round(m_ba["auroc"], 4) if m_ba["auroc"] == m_ba["auroc"] else None,
        "auprc": round(m_ba["auprc"], 4) if m_ba["auprc"] == m_ba["auprc"] else None,
        # Legacy comparison: F1 at F1-tuned threshold (audit of what's lost)
        "f1_at_f1_tuned_threshold": round(m_f1["f1"], 4) if m_f1 is not None else "",
        "balanced_accuracy_at_f1_tuned_threshold": round(m_f1["balanced_accuracy"], 4) if m_f1 is not None else "",
        # Trivial floor
        "trivial_f1": round(trivial["f1"], 4),
        "trivial_balanced_accuracy": round(trivial["balanced_accuracy"], 4),
        "predictions_path": test_path,
        "status": "OK",
        "regenerated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=REPO_ROOT)
    ap.add_argument("--output",
                    default=os.path.join(REPO_ROOT, "evaluation", "balanced_metrics_ba_tuned_summary.csv"))
    ap.add_argument("--failures-output",
                    default=os.path.join(REPO_ROOT, "evaluation", "balanced_metrics_ba_tuned_failures.json"))
    args = ap.parse_args()

    pairs = []  # (val_path, test_path, is_enroll, split_tag)
    for root, _, files in os.walk(args.root):
        if any(seg in root for seg in (".git/", "yamnet-eval", "__pycache__", "/.venv/")):
            continue
        if "val_predictions.csv" in files and "test_predictions.csv" in files:
            pairs.append((
                os.path.join(root, "val_predictions.csv"),
                os.path.join(root, "test_predictions.csv"),
                False, "seen_child_test",
            ))
            if "test_all_predictions.csv" in files:
                pairs.append((
                    os.path.join(root, "val_predictions.csv"),
                    os.path.join(root, "test_all_predictions.csv"),
                    False, "all_children_coverage",
                ))
        if "enroll_val_predictions.csv" in files and "enroll_test_predictions.csv" in files:
            pairs.append((
                os.path.join(root, "enroll_val_predictions.csv"),
                os.path.join(root, "enroll_test_predictions.csv"),
                True, "seen_child_test",
            ))
    print(f"discovered {len(pairs)} (val, test) prediction pairs", file=sys.stderr)

    rows, failures = [], []
    for v, t, is_enroll, split_tag in pairs:
        r = _process(v, t, is_enroll, split_tag)
        if r is None:
            continue
        if r["status"] == "OK":
            rows.append(r)
        else:
            failures.append(r)

    if not rows:
        sys.exit("no rows produced")
    df = pd.DataFrame(rows).sort_values(["system_name", "split"])
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"wrote {len(df)} BA-tuned rows to {args.output}", file=sys.stderr)

    with open(args.failures_output, "w") as f:
        json.dump({"n_failures": len(failures), "failures": failures}, f, indent=2)
    if failures:
        print(f"{len(failures)} failures recorded at {args.failures_output}", file=sys.stderr)


if __name__ == "__main__":
    main()
