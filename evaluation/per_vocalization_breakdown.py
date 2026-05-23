"""Per-vocalization stratification for the headline systems.

The BIDS metadata has a single binary `Vocalizations` column (yes/no = label).
There is no annotation of vocalization *type* (laugh / cry / babble / word).
This script therefore stratifies by the available proxies that correlate with
vocalization sub-character:

  - Activity context (toy play / general interaction / motor play / other)
  - Locomotion present (yes/no)
  - Gestures present (yes/no)
  - Interaction with adult (yes/no)
  - Number of children (1 / >1)

For each top system, computes per-stratum F1 / AUROC.

Outputs:
  evaluation/per_vocalization_proxy.csv
  evaluation/per_vocalization_proxy.md
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score

REPO = "/orcd/scratch/orcd/008/manaal/child-adult-diarization"
MASTER = os.path.join(REPO, "whisper-modeling/seen_child_splits/master_with_split.csv")
OUT_CSV = os.path.join(REPO, "evaluation", "per_vocalization_proxy.csv")
OUT_MD = os.path.join(REPO, "evaluation", "per_vocalization_proxy.md")

SYSTEMS = [
    ("whisper_mil",              "mil/mil_results/whisper_mil/test_predictions.csv",                      "score"),
    ("whisper_mil_tsmil_concat", "mil/mil_results/whisper_mil_tsmil_concat/test_predictions.csv",         "score"),
    ("babar_enrollment",         "babar_ecapa_enrollment_runs/enroll_test_predictions.csv",               "prob"),
    ("metadata_stacker",         "ensemble_runs/metadata_stack/test_predictions.csv",                     "score"),
    ("wavlm_pseudo_frame",       "pseudo_frame/results/wavlm_pseudo_frame/test_predictions.csv",          "score"),
]

STRATA = [
    ("Activity",              lambda df: df["Activity"].fillna("unknown")),
    ("Locomotion",            lambda df: df["Locomotion"].fillna("no")),
    ("Gestures",              lambda df: df["Gestures"].fillna("no")),
    ("Interaction_with_child", lambda df: df["Interaction_with_child"].fillna("no")),
    ("n_children_grp",        lambda df: pd.to_numeric(df["#_children"], errors="coerce").fillna(0).apply(lambda x: "1" if x == 1 else (">1" if x > 1 else "0"))),
]
META_COLS = ["Activity", "Locomotion", "Gestures", "Interaction_with_child", "#_children"]


def main():
    master = pd.read_csv(MASTER, low_memory=False)
    master = master[master["split"] == "test"][["audio_path", "Activity", "Locomotion", "Gestures",
                                                 "Interaction_with_child", "#_children", "label"]]

    rows = []
    for name, csv, col in SYSTEMS:
        path = os.path.join(REPO, csv)
        if not os.path.isfile(path):
            print(f"SKIP {name}")
            continue
        df = pd.read_csv(path)
        if col not in df.columns:
            continue
        df = df.merge(master[["audio_path"] + META_COLS], on="audio_path", how="left")

        for sname, sfn in STRATA:
            df[sname] = sfn(df)
            for level, sub in df.groupby(sname):
                if len(sub) < 10:
                    continue
                y = sub["label"].astype(int).to_numpy()
                p = sub[col].astype(float).clip(0, 1).to_numpy()
                if y.sum() == 0 or y.sum() == len(y):
                    auc = float("nan")
                else:
                    auc = float(roc_auc_score(y, p))
                pred = (p >= 0.5).astype(int)
                f1 = float(f1_score(y, pred, zero_division=0))
                rows.append(dict(
                    system=name, stratum=sname, level=str(level),
                    n=int(len(sub)), n_pos=int(y.sum()),
                    f1=round(f1, 3), auroc=round(auc, 3) if auc == auc else None,
                ))

    out = pd.DataFrame(rows)
    out.to_csv(OUT_CSV, index=False)

    md = ["# Per-Vocalization-Type Proxy Stratification\n"]
    md.append("**Note**: BIDS metadata has no annotation of vocalization *type* "
              "(laugh / cry / babble / word). The single `Vocalizations` column is "
              "binary yes/no and equals the prediction target itself. We therefore "
              "stratify by available proxies that correlate with vocalization "
              "sub-character: activity context, locomotion, gestures, interaction, "
              "and number of children.\n")
    md.append("Per-stratum F1 / AUROC for top 5 systems below; full table in "
              "`per_vocalization_proxy.csv`.\n")
    for sys, sub in out.groupby("system"):
        md.append(f"## {sys}\n")
        md.append(sub[["stratum", "level", "n", "n_pos", "f1", "auroc"]].to_markdown(index=False))
        md.append("\n")
    with open(OUT_MD, "w") as f:
        f.write("\n".join(md))
    print(f"Wrote {OUT_CSV} ({len(out)} rows)")
    print(f"Wrote {OUT_MD}")


if __name__ == "__main__":
    main()
