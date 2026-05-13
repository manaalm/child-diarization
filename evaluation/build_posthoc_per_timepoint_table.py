"""Consolidate per-timepoint metrics across all systems into a single posthoc
analysis table (spec 022 US5 / FR-021).

Joins:
  - evaluation/balanced_metrics_summary.csv (combined-timepoint reference)
  - each system's */test_metrics_by_timepoint.csv (per-timepoint breakdown,
    BIDS-corrected per US1 regeneration)

Output:
  evaluation/posthoc_per_timepoint_table.md  (human-readable thesis appendix)
  evaluation/posthoc_per_timepoint_table.csv (machine-readable for sorting)

For each system, columns include:
  combined_auroc  (from balanced_metrics_summary)
  combined_balanced_accuracy
  14m_auroc
  14m_balanced_accuracy   (recomputed below from per-timepoint preds if available)
  36m_auroc
  delta_36m_minus_14m
  flagged_large_delta  (|delta| > 0.05)
"""

import argparse
import csv
import json
import os
import sys

import pandas as pd

REPO_ROOT = "/orcd/scratch/orcd/008/manaal/child-adult-diarization"
BALANCED_SUMMARY = os.path.join(REPO_ROOT, "evaluation", "balanced_metrics_summary.csv")


def _load_per_timepoint(result_dir: str) -> dict:
    """Load test_metrics_by_timepoint.csv and return per-timepoint metric dict."""
    candidates = [
        os.path.join(result_dir, "test_metrics_by_timepoint.csv"),
        os.path.join(result_dir, "enroll_test_metrics_by_timepoint.csv"),
    ]
    for path in candidates:
        if os.path.exists(path):
            try:
                df = pd.read_csv(path)
                out = {}
                for _, row in df.iterrows():
                    tp = row.get("timepoint")
                    if pd.isna(tp):
                        continue
                    out[str(tp)] = row.to_dict()
                return out
            except Exception:
                continue
    return {}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", default=BALANCED_SUMMARY)
    ap.add_argument("--output-md", default=os.path.join(REPO_ROOT, "evaluation", "posthoc_per_timepoint_table.md"))
    ap.add_argument("--output-csv", default=os.path.join(REPO_ROOT, "evaluation", "posthoc_per_timepoint_table.csv"))
    ap.add_argument("--delta-flag", type=float, default=0.05)
    args = ap.parse_args()

    summary = pd.read_csv(args.input)
    # Only consider seen-child test rows (per-timepoint breakdowns live on that split)
    summary = summary[summary["split"] == "seen_child_test"].copy()

    rows = []
    for _, sys_row in summary.iterrows():
        system_name = sys_row["system_name"]
        result_dir = os.path.join(REPO_ROOT, system_name)
        per_tp = _load_per_timepoint(result_dir)

        if not per_tp:
            continue

        auroc_14 = per_tp.get("14_month", {}).get("auroc")
        auroc_36 = per_tp.get("36_month", {}).get("auroc")
        ba_14 = per_tp.get("14_month", {}).get("balanced_accuracy")
        ba_36 = per_tp.get("36_month", {}).get("balanced_accuracy")
        f1w_14 = per_tp.get("14_month", {}).get("f1_weighted")
        f1w_36 = per_tp.get("36_month", {}).get("f1_weighted")

        delta = None
        flagged = False
        if auroc_14 is not None and auroc_36 is not None \
                and not pd.isna(auroc_14) and not pd.isna(auroc_36):
            delta = float(auroc_36) - float(auroc_14)
            flagged = abs(delta) > args.delta_flag

        rows.append({
            "system_name": system_name,
            "combined_f1": sys_row["f1"],
            "combined_balanced_accuracy": sys_row["balanced_accuracy"],
            "combined_auroc": sys_row["auroc"],
            "14m_auroc": auroc_14,
            "14m_balanced_accuracy": ba_14,
            "14m_f1_weighted": f1w_14,
            "36m_auroc": auroc_36,
            "36m_balanced_accuracy": ba_36,
            "36m_f1_weighted": f1w_36,
            "delta_36m_minus_14m_auroc": delta,
            "flagged_large_delta": flagged,
            "n_14m": per_tp.get("14_month", {}).get("n"),
            "n_36m": per_tp.get("36_month", {}).get("n"),
        })

    if not rows:
        print("no rows produced — check that per-timepoint CSVs exist for systems in balanced_metrics_summary.csv", file=sys.stderr)
        sys.exit(1)

    df = pd.DataFrame(rows).sort_values("combined_auroc", ascending=False)
    df.to_csv(args.output_csv, index=False)

    # Markdown rendering
    md_lines = []
    md_lines.append("# Posthoc: per-timepoint stratification (spec 022 US5)\n")
    md_lines.append(f"Generated from `{os.path.relpath(args.input, REPO_ROOT)}` + each "
                    f"system's `test_metrics_by_timepoint.csv` (BIDS-corrected per US1).\n")
    md_lines.append(f"**Rows**: {len(df)} systems with per-timepoint breakdowns available.\n")
    md_lines.append(f"**Flag threshold**: |Δ AUROC 36m−14m| > {args.delta_flag} → flagged.\n")
    md_lines.append(f"**Flagged systems**: {int(df['flagged_large_delta'].sum())}.\n\n")

    md_lines.append("## Headline table — combined-timepoint metrics (primary)\n\n")
    md_lines.append("| System | F1 | Balanced Acc | AUROC |\n|---|---|---|---|\n")
    for _, r in df.head(30).iterrows():
        md_lines.append(f"| `{r['system_name']}` | {r['combined_f1']:.3f} | "
                        f"{r['combined_balanced_accuracy']:.3f} | {r['combined_auroc']:.3f} |\n")

    md_lines.append("\n## Per-timepoint posthoc breakdown (full sort by combined AUROC)\n\n")
    md_lines.append("| System | 14m AUROC | 14m BA | 14m n | 36m AUROC | 36m BA | 36m n | Δ AUROC | flagged |\n")
    md_lines.append("|---|---|---|---|---|---|---|---|---|\n")
    for _, r in df.iterrows():
        def _fmt(x):
            if x is None or (isinstance(x, float) and pd.isna(x)):
                return "—"
            if isinstance(x, float):
                return f"{x:.3f}"
            return str(x)
        flag = "**FLAG**" if r["flagged_large_delta"] else ""
        md_lines.append(f"| `{r['system_name']}` | {_fmt(r['14m_auroc'])} | "
                        f"{_fmt(r['14m_balanced_accuracy'])} | {_fmt(r['n_14m'])} | "
                        f"{_fmt(r['36m_auroc'])} | {_fmt(r['36m_balanced_accuracy'])} | "
                        f"{_fmt(r['n_36m'])} | {_fmt(r['delta_36m_minus_14m_auroc'])} | {flag} |\n")

    md_lines.append("\n## Flagged systems (|Δ AUROC| > " f"{args.delta_flag})\n\n")
    flagged = df[df["flagged_large_delta"]].sort_values("delta_36m_minus_14m_auroc", ascending=False)
    if len(flagged) == 0:
        md_lines.append("(no flagged systems)\n")
    else:
        md_lines.append("| System | 14m AUROC | 36m AUROC | Δ |\n|---|---|---|---|\n")
        for _, r in flagged.iterrows():
            md_lines.append(f"| `{r['system_name']}` | {r['14m_auroc']:.3f} | "
                            f"{r['36m_auroc']:.3f} | {r['delta_36m_minus_14m_auroc']:+.3f} |\n")
        md_lines.append("\n*Interpretation prose to be added by chapter author (FR-022).*\n")

    with open(args.output_md, "w") as f:
        f.write("".join(md_lines))

    print(f"wrote {args.output_md}")
    print(f"wrote {args.output_csv}")
    print(f"  {len(df)} systems with per-timepoint data; "
          f"{int(df['flagged_large_delta'].sum())} flagged at |Δ| > {args.delta_flag}")


if __name__ == "__main__":
    main()
