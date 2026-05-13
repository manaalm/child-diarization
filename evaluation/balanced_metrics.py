"""Recompute imbalance-aware metrics from every cached test_predictions.csv
(spec 022 US2 / FR-006). Writes one row per (system, split) to
evaluation/balanced_metrics_summary.csv.

Threshold-source contract:
  - Each system has a sibling val_metrics_tuned.json with a `threshold` field
    (val-tuned per Constitution IV). We re-binarise test scores using that
    threshold and recompute the extended metric set.
  - Systems missing val_metrics_tuned.json fall back to threshold=0.5 with a
    `threshold_source=default-0.5` annotation.

Output schema matches contracts/balanced_metrics_summary.schema.md.
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from typing import Optional

import pandas as pd

REPO_ROOT = "/orcd/scratch/orcd/008/manaal/child-adult-diarization"
sys.path.insert(0, os.path.join(REPO_ROOT, "mil"))
from mil_utils import compute_metrics  # noqa: E402


def _detect_columns(df: pd.DataFrame) -> dict:
    out = {"label": None, "score": None, "audio_path": None}
    if "label" in df.columns:
        out["label"] = "label"
    if "audio_path" in df.columns:
        out["audio_path"] = "audio_path"
    for cand in ("score", "prob", "fused_score", "p_child_voc", "joint_score"):
        if cand in df.columns:
            out["score"] = cand
            break
    return out


def _load_threshold(system_dir: str) -> tuple[float, str]:
    """Return (threshold, source). Source is one of:
      val-tuned   — from sibling val_metrics_tuned.json
      val-tuned-alt — from val_metrics.json (older convention)
      default-0.5 — fallback when no val artefact found
    """
    for fname, source in (
        ("val_metrics_tuned.json", "val-tuned"),
        ("enroll_val_metrics_tuned.json", "val-tuned"),
        ("val_metrics.json", "val-tuned-alt"),
    ):
        p = os.path.join(system_dir, fname)
        if os.path.exists(p):
            try:
                d = json.load(open(p))
                if "threshold" in d:
                    return float(d["threshold"]), source
            except Exception:
                pass
    return 0.5, "default-0.5"


def _process_one(pred_path: str, system_root: str = REPO_ROOT) -> Optional[dict]:
    sys_dir = os.path.dirname(pred_path)
    is_enroll = pred_path.endswith("enroll_test_predictions.csv")
    is_test_all = pred_path.endswith("test_all_predictions.csv")
    split = "all_children_coverage" if is_test_all else "seen_child_test"

    try:
        df = pd.read_csv(pred_path)
    except Exception as e:
        return {"system_name": os.path.relpath(sys_dir, system_root),
                "split": split, "status": "READ_FAIL", "error": str(e)}

    cols = _detect_columns(df)
    if cols["label"] is None or cols["score"] is None:
        return {"system_name": os.path.relpath(sys_dir, system_root),
                "split": split, "status": "SCHEMA_FAIL", "missing": cols}

    threshold, threshold_source = _load_threshold(sys_dir)

    y_true = df[cols["label"]].astype(int).tolist()
    y_score = df[cols["score"]].astype(float).tolist()
    m = compute_metrics(y_true, y_score, threshold=threshold)

    # Trivial floor: constant predict-all-positive
    trivial_floor = compute_metrics(y_true, [1.0] * len(y_true), threshold=0.5)

    system_name = os.path.relpath(sys_dir, system_root)
    if is_enroll:
        sibling_metrics = os.path.join(sys_dir, "enroll_test_metrics_tuned.json")
    elif is_test_all:
        sibling_metrics = os.path.join(sys_dir, "test_all_metrics_tuned.json")
    else:
        sibling_metrics = os.path.join(sys_dir, "test_metrics_tuned.json")
    if not os.path.exists(sibling_metrics):
        sibling_metrics = ""

    return {
        "system_name": system_name,
        "split": split,
        "n_clips": len(y_true),
        "pos_rate": round(float(sum(y_true) / len(y_true)), 4),
        "threshold_source": threshold_source,
        "tuned_threshold": threshold,
        "f1": round(m["f1"], 4),
        "f1_macro": round(m["f1_macro"], 4),
        "f1_weighted": round(m["f1_weighted"], 4),
        "balanced_accuracy": round(m["balanced_accuracy"], 4),
        "precision": round(m["precision"], 4),
        "recall": round(m["recall"], 4),
        "auroc": round(m["auroc"], 4) if m["auroc"] == m["auroc"] else None,
        "auprc": round(m["auprc"], 4) if m["auprc"] == m["auprc"] else None,
        "trivial_f1": round(trivial_floor["f1"], 4),
        "trivial_f1_macro": round(trivial_floor["f1_macro"], 4),
        "trivial_balanced_accuracy": round(trivial_floor["balanced_accuracy"], 4),
        "predictions_path": pred_path,
        "metrics_json_path": sibling_metrics,
        "status": "OK",
        "regenerated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=REPO_ROOT)
    ap.add_argument("--output", default=os.path.join(REPO_ROOT, "evaluation", "balanced_metrics_summary.csv"))
    ap.add_argument("--failures-output", default=os.path.join(REPO_ROOT, "evaluation", "balanced_metrics_failures.json"))
    args = ap.parse_args()

    pred_files = []
    for root, _, files in os.walk(args.root):
        for name in files:
            if name in ("test_predictions.csv", "enroll_test_predictions.csv", "test_all_predictions.csv"):
                pred_files.append(os.path.join(root, name))
    print(f"discovered {len(pred_files)} prediction files", file=sys.stderr)

    rows, failures = [], []
    for pf in pred_files:
        r = _process_one(pf)
        if r is None:
            continue
        if r["status"] == "OK":
            rows.append(r)
        else:
            failures.append(r)

    if not rows:
        print("no rows produced", file=sys.stderr)
        sys.exit(1)

    df = pd.DataFrame(rows)
    df = df.sort_values(["system_name", "split"])
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"wrote {len(df)} rows to {args.output}", file=sys.stderr)

    with open(args.failures_output, "w") as f:
        json.dump({"n_failures": len(failures), "failures": failures}, f, indent=2)
    if failures:
        print(f"{len(failures)} failures recorded at {args.failures_output}", file=sys.stderr)


if __name__ == "__main__":
    main()
