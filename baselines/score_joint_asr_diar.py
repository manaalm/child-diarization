"""General-purpose scorer for joint ASR+Diar predictions vs an RTTM GT dir.

Generalization of `score_joint_asr_diar_synth.py`:
- supports configurable GT child label (`--gt-child-labels CHI` for Providence,
  `TARGET_CHILD` for synth, etc.),
- reads actual audio durations rather than assuming a fixed clip length,
- writes the same `aggregate_metrics.json` + `per_file_metrics.csv` schema as
  `pyannote/unified_rttm.py` so `frame_localization_gt.py` and
  `onset_tolerance_f1.py` ingest it.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np
import librosa
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    roc_auc_score, average_precision_score,
)


def parse_rttm(path: str) -> list[tuple[float, float, str]]:
    segs = []
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
                start = float(parts[3])
                dur = float(parts[4])
            except ValueError:
                continue
            segs.append((start, start + dur, parts[7]))
    return segs


def to_mask(segs, child_labels, total_dur, frame_step=0.01):
    n = int(round(total_dur / frame_step))
    mask = np.zeros(n, dtype=np.uint8)
    for start, end, lbl in segs:
        if lbl not in child_labels:
            continue
        s = max(0, int(round(start / frame_step)))
        e = min(n, int(round(end / frame_step)))
        if e > s:
            mask[s:e] = 1
    return mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", required=True)
    ap.add_argument("--gt-dir", required=True)
    ap.add_argument("--audio-dir", required=True,
                    help="Used to look up actual file duration (matched by stem).")
    ap.add_argument("--gt-child-labels", default="CHI",
                    help="Comma-separated GT child labels (e.g. 'CHI' for "
                         "Providence, 'TARGET_CHILD' for synth, 'CHI,KCHI,OCH' "
                         "if using a wider child class.)")
    ap.add_argument("--frame-step", type=float, default=0.01)
    args = ap.parse_args()

    pred_dir = os.path.join(args.results_dir, "per_file_predictions")
    child_labels = set(s.strip() for s in args.gt_child_labels.split(",") if s.strip())
    pred_child_labels = {"CHI"}  # written by joint_asr_diar_batch

    # Index audio files by stem for duration lookup
    audio_by_stem = {}
    for ext in (".wav", ".mp3", ".flac"):
        for p in Path(args.audio_dir).rglob(f"*{ext}"):
            audio_by_stem.setdefault(p.stem.lower(), str(p))

    rows = []
    all_y_true = []
    all_y_pred = []
    total_audio_sec = 0.0
    total_gt = 0.0
    total_pred = 0.0

    for f in sorted(Path(pred_dir).glob("*_pred.rttm")):
        stem = f.stem.replace("_pred", "")
        gt_path = os.path.join(args.gt_dir, f"{stem}.rttm")
        if not os.path.isfile(gt_path):
            # Case-insensitive fallback
            for fn in os.listdir(args.gt_dir):
                if fn.lower() == f"{stem.lower()}.rttm":
                    gt_path = os.path.join(args.gt_dir, fn)
                    break
            if not os.path.isfile(gt_path):
                continue
        audio_path = audio_by_stem.get(stem.lower())
        if audio_path is None:
            continue
        try:
            dur = librosa.get_duration(path=audio_path)
        except Exception:
            continue
        pred_segs = parse_rttm(str(f))
        gt_segs = parse_rttm(gt_path)
        gt_mask = to_mask(gt_segs, child_labels, dur, args.frame_step)
        pred_mask = to_mask(pred_segs, pred_child_labels, dur, args.frame_step)

        gt_dur_sec = float(gt_mask.sum()) * args.frame_step
        pred_dur_sec = float(pred_mask.sum()) * args.frame_step
        total_audio_sec += dur
        total_gt += gt_dur_sec
        total_pred += pred_dur_sec
        all_y_true.append(gt_mask)
        all_y_pred.append(pred_mask)

        if gt_mask.sum() == 0 and pred_mask.sum() == 0:
            f1 = 1.0
        elif gt_mask.sum() == 0 or pred_mask.sum() == 0:
            f1 = 0.0
        else:
            f1 = f1_score(gt_mask, pred_mask, zero_division=0)
        prec = precision_score(gt_mask, pred_mask, zero_division=0)
        rec = recall_score(gt_mask, pred_mask, zero_division=0)
        try:
            auroc = roc_auc_score(gt_mask, pred_mask) if len(np.unique(gt_mask)) > 1 else 0.5
        except ValueError:
            auroc = 0.5
        try:
            auprc = average_precision_score(gt_mask, pred_mask)
        except ValueError:
            auprc = float(gt_mask.mean())
        tp = int(((gt_mask == 1) & (pred_mask == 1)).sum())
        fp = int(((gt_mask == 0) & (pred_mask == 1)).sum())
        fn = int(((gt_mask == 1) & (pred_mask == 0)).sum())
        tn = int(((gt_mask == 0) & (pred_mask == 0)).sum())
        miss = fn / max(1, tp + fn)
        fa = fp / max(1, fp + tn)
        rows.append({
            "file": stem,
            "total_dur_sec": round(dur, 2),
            "gt_child_dur_sec": round(gt_dur_sec, 2),
            "pred_child_dur_sec": round(pred_dur_sec, 2),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "accuracy": round((tp + tn) / max(1, tp + fp + fn + tn), 4),
            "miss_rate": round(miss, 4),
            "false_alarm_rate": round(fa, 4),
            "der": round((fn + fp) / max(1, tp + fn), 4),
            "auroc": round(auroc, 4),
            "auprc": round(auprc, 4),
            "tp_frames": tp, "fp_frames": fp, "fn_frames": fn, "tn_frames": tn,
        })

    if not rows:
        raise SystemExit(f"No matched (pred, gt, audio) triples under {pred_dir}")

    y_true = np.concatenate(all_y_true)
    y_pred = np.concatenate(all_y_pred)
    micro_f1 = f1_score(y_true, y_pred, zero_division=0)
    micro_p = precision_score(y_true, y_pred, zero_division=0)
    micro_r = recall_score(y_true, y_pred, zero_division=0)
    miss_micro = float(((y_true == 1) & (y_pred == 0)).sum() /
                       max(1, (y_true == 1).sum()))
    fa_micro = float(((y_true == 0) & (y_pred == 1)).sum() /
                     max(1, (y_true == 0).sum()))
    macro_f1 = float(np.mean([r["f1"] for r in rows]))
    macro_auroc = float(np.mean([r["auroc"] for r in rows]))
    macro_auprc = float(np.mean([r["auprc"] for r in rows]))

    agg = {
        "micro_precision": round(micro_p, 4),
        "micro_recall": round(micro_r, 4),
        "micro_f1": round(micro_f1, 4),
        "micro_accuracy": round(float((y_true == y_pred).mean()), 4),
        "miss_rate": round(miss_micro, 4),
        "false_alarm_rate": round(fa_micro, 4),
        "binary_der": round(miss_micro + fa_micro, 4),
        "macro_f1": round(macro_f1, 4),
        "n_files": len(rows),
        "total_gt_child_dur_sec": round(total_gt, 2),
        "total_pred_child_dur_sec": round(total_pred, 2),
        "total_audio_dur_sec": round(total_audio_sec, 2),
        "macro_auroc": round(macro_auroc, 4),
        "macro_auprc": round(macro_auprc, 4),
    }
    with open(os.path.join(args.results_dir, "aggregate_metrics.json"), "w") as f:
        json.dump(agg, f, indent=2)
    with open(os.path.join(args.results_dir, "per_file_metrics.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print(f"Wrote: {args.results_dir}/aggregate_metrics.json")
    print(json.dumps(agg, indent=2))


if __name__ == "__main__":
    main()
