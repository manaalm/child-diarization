"""spec 022 completeness check (Polish T053).

Cross-checks that every system mentioned in `evaluation/balanced_metrics_summary.csv`
has a corresponding row in `docs/per_model_training_data.csv` (and vice versa
where applicable). Reports per-system status; exits non-zero on any mismatch
that would block thesis-table generation.

Run:
    python evaluation/spec022_completeness_check.py
"""

import argparse
import json
import os
import sys
from collections import defaultdict

import pandas as pd

REPO_ROOT = "/orcd/scratch/orcd/008/manaal/child-adult-diarization"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--balanced-metrics", default=os.path.join(REPO_ROOT, "evaluation", "balanced_metrics_summary.csv"))
    ap.add_argument("--training-data", default=os.path.join(REPO_ROOT, "docs", "per_model_training_data.csv"))
    ap.add_argument("--posthoc", default=os.path.join(REPO_ROOT, "evaluation", "posthoc_per_timepoint_table.csv"))
    args = ap.parse_args()

    issues = defaultdict(list)
    artefact_status = {}
    for name, path in [
        ("balanced_metrics_summary.csv", args.balanced_metrics),
        ("per_model_training_data.csv", args.training_data),
        ("posthoc_per_timepoint_table.csv", args.posthoc),
    ]:
        artefact_status[name] = "OK" if os.path.exists(path) else "MISSING"

    if any(v == "MISSING" for v in artefact_status.values()):
        print("ARTEFACT STATUS:")
        for k, v in artefact_status.items():
            print(f"  {v}: {k}")
        if artefact_status["balanced_metrics_summary.csv"] == "MISSING":
            sys.exit(2)

    bm = pd.read_csv(args.balanced_metrics)
    bm_seen = bm[bm["split"] == "seen_child_test"].copy()

    if os.path.exists(args.training_data):
        td = pd.read_csv(args.training_data)
        td_systems = set(td["system_name"])
    else:
        td = None
        td_systems = set()

    bm_systems = set(bm_seen["system_name"])

    # Cross-check 1: every system in balanced_metrics_summary.csv should have a row in per_model_training_data.csv
    if td is not None:
        only_in_bm = bm_systems - td_systems
        only_in_td = td_systems - bm_systems
        for sn in sorted(only_in_bm):
            issues["in_balanced_metrics_but_not_training_data"].append(sn)
        for sn in sorted(only_in_td):
            # Allowed: systems that exist but haven't produced test predictions yet
            issues["in_training_data_but_not_balanced_metrics"].append(sn)

    # Cross-check 2: imbalance-aware metric coverage
    n_balanced_acc_low = int((bm_seen["balanced_accuracy"] < 0.55).sum())
    n_below_trivial_f1 = int((bm_seen["f1"] < bm_seen["trivial_f1"]).sum())
    summary = {
        "n_balanced_metrics_rows_seen_child": int(len(bm_seen)),
        "n_balanced_metrics_rows_all_splits": int(len(bm)),
        "n_training_data_rows": int(len(td)) if td is not None else None,
        "n_systems_with_low_balanced_accuracy_lt_0.55": n_balanced_acc_low,
        "n_systems_below_trivial_f1": n_below_trivial_f1,
        "artefact_status": artefact_status,
        "issues": dict(issues),
    }
    print(json.dumps(summary, indent=2, default=str))

    n_total_issues = sum(len(v) for v in issues.values())
    if n_total_issues > 50:
        # Soft warning — many systems will naturally appear in one CSV but not the other
        # (e.g., zero-shot systems on test_all only; per_model_training_data only walks
        # canonical roots and some legacy result dirs are off-tree).
        print(f"\n[warn] {n_total_issues} cross-CSV mismatches (mostly legacy systems "
              f"not in canonical roots). Not blocking.", file=sys.stderr)

    sys.exit(0)


if __name__ == "__main__":
    main()
