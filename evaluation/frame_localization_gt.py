"""Frame-level GT localization evaluation on Playlogue and Providence.

Each system in pyannote/eval_results/<sys>_<dataset>/ has:
  - aggregate_metrics.json       (corpus-level micro/macro F1, AUROC)
  - per_file_metrics.csv         (per-file F1, AUROC, AUPRC vs GT RTTM)
  - per_file_predictions/*.rttm  (binary segment predictions)

This script aggregates per_file metrics into:
  - mean ± std per-file AUROC and F1 per (system, dataset)
  - distribution of per-file AUROC (5/25/50/75/95 percentiles)

Outputs:
  evaluation/frame_localization_gt.csv  (one row per (system, dataset))
  evaluation/frame_localization_gt.md   (summary tables)
"""

from __future__ import annotations

import os
import json

import numpy as np
import pandas as pd

REPO = "/orcd/scratch/orcd/008/manaal/child-adult-diarization"
EVAL_DIR = os.path.join(REPO, "pyannote/eval_results")
OUT_CSV = os.path.join(REPO, "evaluation", "frame_localization_gt.csv")
OUT_MD = os.path.join(REPO, "evaluation", "frame_localization_gt.md")


def main():
    rows = []
    for d in sorted(os.listdir(EVAL_DIR)):
        full = os.path.join(EVAL_DIR, d)
        if not os.path.isdir(full):
            continue
        per_file = os.path.join(full, "per_file_metrics.csv")
        agg = os.path.join(full, "aggregate_metrics.json")
        if not os.path.isfile(per_file) or not os.path.isfile(agg):
            continue
        # parse <system>_<dataset>
        if d.endswith("_playlogue"):
            system, dataset = d[:-len("_playlogue")], "playlogue"
        elif d.endswith("_providence"):
            system, dataset = d[:-len("_providence")], "providence"
        elif d.endswith("_synth_holdout"):
            system, dataset = d[:-len("_synth_holdout")], "synth_holdout"
        else:
            continue
        df = pd.read_csv(per_file)
        with open(agg) as f:
            ag = json.load(f)
        # Per-file AUROC distribution
        auroc_col = next((c for c in df.columns if "auroc" in c.lower()), None)
        f1_col = next((c for c in df.columns if c.lower() == "f1"), None)
        if auroc_col is None or f1_col is None:
            continue
        au = df[auroc_col].dropna().to_numpy()
        f1 = df[f1_col].dropna().to_numpy()
        if len(au) == 0:
            continue
        rows.append(dict(
            system=system, dataset=dataset, n_files=int(len(df)),
            agg_micro_f1=ag.get("micro_f1"),
            agg_macro_f1=ag.get("macro_f1"),
            agg_macro_auroc=ag.get("macro_auroc"),
            agg_macro_auprc=ag.get("macro_auprc"),
            agg_miss_rate=ag.get("miss_rate"),
            agg_false_alarm_rate=ag.get("false_alarm_rate"),
            agg_binary_der=ag.get("binary_der"),
            per_file_auroc_mean=round(float(np.mean(au)), 3),
            per_file_auroc_std=round(float(np.std(au)), 3),
            per_file_auroc_p25=round(float(np.percentile(au, 25)), 3),
            per_file_auroc_p50=round(float(np.percentile(au, 50)), 3),
            per_file_auroc_p75=round(float(np.percentile(au, 75)), 3),
            per_file_f1_mean=round(float(np.mean(f1)), 3),
            per_file_f1_std=round(float(np.std(f1)), 3),
            files_with_auroc_above_07=int(np.sum(au > 0.7)),
            files_with_auroc_above_08=int(np.sum(au > 0.8)),
        ))

    out = pd.DataFrame(rows).sort_values(["dataset", "agg_macro_auroc"], ascending=[True, False])
    out.to_csv(OUT_CSV, index=False)

    md = ["# Frame-Level GT Localization (Playlogue + Providence)\n"]
    md.append("Per-file frame-level metrics (10 ms binary mask vs. GT RTTM) for "
              "12 systems × 2 long-form datasets, sourced from "
              "`pyannote/eval_results/<system>_<dataset>/per_file_metrics.csv`.\n")

    md.append("## Headline (corpus-level)\n")
    md.append("F1 / AUROC / AUPRC + miss & false-alarm rates per system. "
              "AUROC=0.5 means random temporal localization; miss=fraction of "
              "GT child speech the system never flagged.\n")
    headline_cols = ["system", "agg_micro_f1", "agg_macro_auroc",
                     "agg_macro_auprc", "agg_miss_rate",
                     "agg_false_alarm_rate", "agg_binary_der"]
    for ds in ("playlogue", "providence", "synth_holdout"):
        sub = out[out["dataset"] == ds][headline_cols].copy()
        if sub.empty:
            continue
        for c in sub.columns:
            if c == "system":
                continue
            sub[c] = sub[c].apply(lambda v: round(float(v), 3) if v is not None else None)
        md.append(f"### {ds.replace('_', ' ').title()}\n")
        md.append(sub.to_markdown(index=False))
        md.append("\n")

    md.append("## Full per-file distribution\n")
    md.append("Columns: corpus-level micro/macro F1 + macro AUROC + AUPRC + miss/FA + DER; "
              "per-file AUROC distribution (mean ± std and 25/50/75 percentiles); "
              "count of files with per-file AUROC above 0.7 / 0.8.\n")
    for ds in ("playlogue", "providence", "synth_holdout"):
        sub = out[out["dataset"] == ds].drop(columns=["dataset"])
        if sub.empty:
            continue
        md.append(f"### {ds.replace('_', ' ').title()}\n")
        md.append(sub.to_markdown(index=False))
        md.append("\n")
    md.append("\n## Reading these numbers\n")
    md.append("- **agg_macro_auroc** is the corpus-level binary-AUROC over all "
              "frames concatenated. The reported §8 numbers come from this.\n")
    md.append("- **per_file_auroc_mean** is the unweighted average of per-recording "
              "AUROC. When this is much *higher* than agg_macro_auroc, the system "
              "performs well on most files but a few large recordings dominate the "
              "corpus-level metric. When much *lower*, the system has a few high-AUROC "
              "files that pull the corpus number up while most files are weak.\n")
    md.append("- **files_with_auroc_above_07**: a useful sanity metric for "
              "deployment — how many of the test recordings would the system be "
              "actually useful on, at a per-file level.\n")
    md.append("- Frame-level AUROC of 0.5 = random temporal localization (the "
              "system might still get high *segment*-level F1 if it predicts the "
              "right total amount of speech, just not at the right times).")
    with open(OUT_MD, "w") as f:
        f.write("\n".join(md))
    print(f"Wrote {OUT_CSV}  ({len(out)} rows)")
    print(f"Wrote {OUT_MD}")


if __name__ == "__main__":
    main()
