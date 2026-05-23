"""Onset-tolerance F1 for child-vocalization detection.

Frame-Pearson and frame-F1 over-penalize boundary jitter — clinicians and the
diarization literature typically report event-based metrics that match
predicted segments to GT segments under a tolerance window on the onset (and
optionally the offset) time.

For each (system, dataset), we read the prediction RTTMs from
`pyannote/eval_results/<system>_<dataset>/per_file_predictions/` and the GT
RTTMs from `playlogue/rttm/` or `providence/rttm/` (filtered to CHI), and
compute precision/recall/F1 at several onset tolerances.

Matching rule (one-to-one greedy):
  - Sort predictions by onset; sort GT by onset.
  - For each prediction, find the closest unmatched GT whose |onset_pred -
    onset_gt| <= tolerance. If found: TP, mark GT used; else: FP.
  - Any unmatched GT after the sweep: FN.

Outputs:
  evaluation/onset_tolerance_f1.csv  (one row per (system, dataset, tolerance))
  evaluation/onset_tolerance_f1.md   (summary tables at 100/250/500 ms)
"""

from __future__ import annotations

import os
from collections import defaultdict
from typing import Iterable

import numpy as np
import pandas as pd

REPO = "/orcd/scratch/orcd/008/manaal/child-adult-diarization"
EVAL_DIR = os.path.join(REPO, "pyannote/eval_results")
GT_DIRS = {
    "playlogue": os.path.join(REPO, "playlogue/rttm"),
    "providence": os.path.join(REPO, "providence/rttm"),
    "synth_holdout": os.path.join(
        REPO, "synth_results/synthetic_scenes_v2/holdout_eval_200/rttm"),
}
GT_CHILD_LABELS_BY_DATASET = {
    "playlogue": frozenset({"CHI"}),
    "providence": frozenset({"CHI"}),
    "synth_holdout": frozenset({"TARGET_CHILD"}),
}
OUT_CSV = os.path.join(REPO, "evaluation", "onset_tolerance_f1.csv")
OUT_MD = os.path.join(REPO, "evaluation", "onset_tolerance_f1.md")

TOLERANCES_MS = (100, 250, 500, 1000)
HEADLINE_TOL_MS = 250  # field-standard


CHILD_LABELS = frozenset({"CHI", "KCHI", "OCH", "TARGET_CHILD"})


def parse_rttm(path: str, allowed_labels: frozenset[str] | None = CHILD_LABELS) -> list[tuple[float, float]]:
    """Return [(onset, offset), ...] for segments whose label is in allowed_labels.

    If allowed_labels is None, returns all segments regardless of label.
    BabAR/VTC predictions use 'KCHI'; USC-SAIL/Sortformer/Pyannote/EEND-EDA use
    'CHI'; the wider 'vtc' variant additionally emits 'OCH'. GT RTTMs use 'CHI'."""
    segs: list[tuple[float, float]] = []
    if not os.path.isfile(path):
        return segs
    with open(path) as f:
        for line in f:
            parts = line.split()
            if not parts or parts[0] != "SPEAKER":
                continue
            if len(parts) < 9:
                continue
            try:
                onset = float(parts[3])
                duration = float(parts[4])
            except ValueError:
                continue
            label = parts[7]
            if allowed_labels is None or label in allowed_labels:
                segs.append((onset, onset + duration))
    segs.sort()
    return segs


def match_onsets(
    pred: list[tuple[float, float]],
    gt: list[tuple[float, float]],
    tolerance_sec: float,
) -> tuple[int, int, int]:
    """Greedy one-to-one onset matching within tolerance.

    Returns (tp, fp, fn)."""
    pred_sorted = sorted(pred)
    gt_sorted = sorted(gt)
    used_gt = [False] * len(gt_sorted)
    tp = 0
    fp = 0
    for p_on, _ in pred_sorted:
        best_idx = -1
        best_diff = float("inf")
        for j, (g_on, _) in enumerate(gt_sorted):
            if used_gt[j]:
                continue
            diff = abs(p_on - g_on)
            if diff <= tolerance_sec and diff < best_diff:
                best_diff = diff
                best_idx = j
            elif g_on > p_on + tolerance_sec:
                break  # gt is sorted; further gts are too late
        if best_idx >= 0:
            tp += 1
            used_gt[best_idx] = True
        else:
            fp += 1
    fn = used_gt.count(False)
    return tp, fp, fn


def gt_path_for_pred(pred_filename: str, dataset: str) -> str | None:
    """Map a prediction filename to its GT RTTM path.

    Pred files are named `<base>_pred.rttm`; GT files are `<base>.rttm` but
    Playlogue GT uses lowercase `aae` while pred uses `AAE`."""
    base = pred_filename.replace("_pred.rttm", "")
    gt_dir = GT_DIRS[dataset]
    direct = os.path.join(gt_dir, f"{base}.rttm")
    if os.path.isfile(direct):
        return direct
    # Case-insensitive fallback
    target_lower = f"{base.lower()}.rttm"
    for fn in os.listdir(gt_dir):
        if fn.lower() == target_lower:
            return os.path.join(gt_dir, fn)
    return None


def evaluate_pair(system: str, dataset: str) -> list[dict]:
    pred_dir = os.path.join(EVAL_DIR, f"{system}_{dataset}", "per_file_predictions")
    if not os.path.isdir(pred_dir):
        return []

    # Aggregate counts per tolerance over all files
    counts: dict[int, list[int]] = {tol: [0, 0, 0] for tol in TOLERANCES_MS}
    n_files = 0
    n_files_with_gt = 0
    for fn in sorted(os.listdir(pred_dir)):
        if not fn.endswith("_pred.rttm"):
            continue
        n_files += 1
        gt_path = gt_path_for_pred(fn, dataset)
        if gt_path is None:
            continue
        n_files_with_gt += 1
        pred_segs = parse_rttm(os.path.join(pred_dir, fn))
        gt_segs = parse_rttm(
            gt_path,
            allowed_labels=GT_CHILD_LABELS_BY_DATASET.get(
                dataset, frozenset({"CHI"})),
        )
        for tol in TOLERANCES_MS:
            tp, fp, fn_ = match_onsets(pred_segs, gt_segs, tol / 1000.0)
            counts[tol][0] += tp
            counts[tol][1] += fp
            counts[tol][2] += fn_

    rows = []
    for tol, (tp, fp, fn_) in counts.items():
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn_) if (tp + fn_) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        rows.append(dict(
            system=system,
            dataset=dataset,
            tolerance_ms=tol,
            tp=tp,
            fp=fp,
            fn=fn_,
            precision=round(prec, 4),
            recall=round(rec, 4),
            f1=round(f1, 4),
            n_files=n_files,
            n_files_with_gt=n_files_with_gt,
        ))
    return rows


def main():
    all_rows = []
    for d in sorted(os.listdir(EVAL_DIR)):
        if d.endswith("_playlogue"):
            system, dataset = d[: -len("_playlogue")], "playlogue"
        elif d.endswith("_providence"):
            system, dataset = d[: -len("_providence")], "providence"
        elif d.endswith("_synth_holdout"):
            system, dataset = d[: -len("_synth_holdout")], "synth_holdout"
        else:
            continue
        rows = evaluate_pair(system, dataset)
        all_rows.extend(rows)
        if rows:
            r = next(r for r in rows if r["tolerance_ms"] == HEADLINE_TOL_MS)
            print(f"{system:24s} {dataset:11s} F1@{HEADLINE_TOL_MS}ms = {r['f1']:.3f} "
                  f"P={r['precision']:.3f} R={r['recall']:.3f}")

    df = pd.DataFrame(all_rows)
    df.to_csv(OUT_CSV, index=False)

    md = ["# Onset-Tolerance F1 (Playlogue + Providence)\n"]
    md.append(
        "Event-based localization metric: each predicted segment is matched "
        "one-to-one (greedy by onset) to the closest unmatched GT child segment "
        "within ±tolerance ms on the onset time. TP=matched, FP=unmatched "
        "prediction, FN=unmatched GT.\n"
    )
    md.append(
        "This penalizes systems that produce the right total amount of speech "
        "at the wrong times, and rewards systems that put the *start* of each "
        "vocalization in approximately the right place — which is what "
        "downstream applications (clinical transcription, language sampling) "
        "actually need.\n"
    )
    for ds in ("playlogue", "providence", "synth_holdout"):
        sub = df[df["dataset"] == ds]
        if sub.empty:
            continue
        md.append(f"## {ds.replace('_', ' ').title()}\n")
        for tol in TOLERANCES_MS:
            tsub = sub[sub["tolerance_ms"] == tol].sort_values("f1", ascending=False)
            md.append(f"### Tolerance ±{tol} ms\n")
            cols = ["system", "precision", "recall", "f1", "tp", "fp", "fn"]
            md.append(tsub[cols].to_markdown(index=False))
            md.append("\n")
        md.append("\n")
    md.append("## Reading these numbers\n")
    md.append(
        "- A system with high frame-level F1 but low onset-tolerance F1 "
        "predicts roughly the right amount of speech but at the wrong "
        "times (or fragments long vocalizations into many short ones).\n"
    )
    md.append(
        "- The 100 / 250 / 500 / 1000 ms tolerances span the field-standard "
        "range. 250 ms is roughly the duration of a syllable; 500 ms is "
        "comparable to typical NIST-style scoring with a forgiveness collar; "
        "1000 ms is a loose 'did we put the segment in the right "
        "neighborhood' check.\n"
    )
    md.append(
        "- One-to-one greedy matching means a system that emits 10 short "
        "predictions all within tolerance of one GT segment gets 1 TP and "
        "9 FPs — the right behavior, since each prediction is supposed to "
        "correspond to a real vocalization event.\n"
    )

    with open(OUT_MD, "w") as f:
        f.write("\n".join(md))
    print(f"Wrote {OUT_CSV}  ({len(df)} rows)")
    print(f"Wrote {OUT_MD}")


if __name__ == "__main__":
    main()
