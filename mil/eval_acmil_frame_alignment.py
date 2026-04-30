"""Per-branch weak-diarization alignment for ACMIL frame-window MIL.

Reads branch_attention_{split}.csv from an ACMIL run dir and computes per-branch
Pearson/Spearman correlations against ground-truth child speech fraction per
2s window. The GT child-speech fraction comes from RTTM files in a chosen cache
(typically `whisper-modeling/usc_sail_rttm_cache` since USC-SAIL annotations
align with Playlogue/Providence/SAILS frame-level GT).

Usage:
    python mil/eval_acmil_frame_alignment.py \\
        --results-dir mil/mil_results/wavlm_mil_acmil \\
        --rttm-cache  whisper-modeling/usc_sail_rttm_cache \\
        --split test
    # writes branch_alignment_{split}.csv

Output columns:
    branch, n_windows, n_clips, pearson_r, pearson_pval, spearman_rho,
    spearman_pval, auroc_ranking
"""

import argparse
import hashlib
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

_CHILD_LABELS = {"chi", "kchi", "child"}


def _rttm_path(audio_path: str, cache_dir: str) -> str:
    stem = Path(audio_path).stem
    md5 = hashlib.md5(audio_path.encode("utf-8")).hexdigest()
    return os.path.join(cache_dir, f"{stem}__{md5}.rttm")


def _child_intervals(rttm_path: str):
    out = []
    if not os.path.isfile(rttm_path):
        return out
    with open(rttm_path) as f:
        for line in f:
            line = line.strip()
            if not line.startswith("SPEAKER"):
                continue
            parts = line.split()
            if len(parts) < 8:
                continue
            label = parts[7].lower()
            if label not in _CHILD_LABELS:
                continue
            start = float(parts[3])
            dur = float(parts[4])
            if dur > 0:
                out.append((start, start + dur))
    return out


def _frac_overlap(intervals, w_start: float, w_end: float) -> float:
    total = 0.0
    for s, e in intervals:
        ov = max(0.0, min(e, w_end) - max(s, w_start))
        total += ov
    return total / max(1e-9, (w_end - w_start))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", required=True)
    ap.add_argument("--rttm-cache", default="whisper-modeling/usc_sail_rttm_cache")
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--window-sec", type=float, default=None,
                    help="Window length sec (default: read from config.json)")
    ap.add_argument("--stride-sec", type=float, default=None,
                    help="Stride sec (default: read from config.json)")
    args = ap.parse_args()

    results_dir = args.results_dir
    cfg_path = os.path.join(results_dir, "config.json")
    if os.path.isfile(cfg_path):
        import json
        cfg = json.load(open(cfg_path))
        window_sec = args.window_sec or cfg.get("window_sec", 2.0)
        stride_sec = args.stride_sec or cfg.get("stride_sec", 2.0)
    else:
        window_sec = args.window_sec or 2.0
        stride_sec = args.stride_sec or 2.0

    csv_in = os.path.join(results_dir, f"branch_attention_{args.split}.csv")
    if not os.path.isfile(csv_in):
        print(f"ERROR: {csv_in} not found", file=sys.stderr)
        sys.exit(2)
    df = pd.read_csv(csv_in)

    branch_cols = [c for c in df.columns if c.startswith("branch_") and c.endswith("_weight")]
    if not branch_cols:
        print("ERROR: no branch_*_weight columns found", file=sys.stderr)
        sys.exit(2)

    # Compute window start/end from instance_idx + window_sec/stride_sec
    df = df.copy()
    df["w_start"] = df["instance_idx"] * stride_sec
    df["w_end"] = df["w_start"] + window_sec

    # Per-clip GT
    rttm_dir = args.rttm_cache if os.path.isabs(args.rttm_cache) else os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), args.rttm_cache)

    intervals_by_clip = {}
    gt_frac = np.zeros(len(df), dtype=np.float32)
    n_clips_with_rttm = 0
    for clip_path, sub in df.groupby("audio_path"):
        rttm = _rttm_path(clip_path, rttm_dir)
        intervals = _child_intervals(rttm)
        if intervals:
            n_clips_with_rttm += 1
        intervals_by_clip[clip_path] = intervals
        for idx, row in sub.iterrows():
            gt_frac[idx] = _frac_overlap(intervals, row["w_start"], row["w_end"])
    df["gt_frac"] = gt_frac

    # Per-branch correlations
    rows = []
    for col in branch_cols + ["mean_weight"]:
        x = df[col].values
        y = df["gt_frac"].values
        # Constant-x guard
        if np.std(x) < 1e-12 or np.std(y) < 1e-12:
            pearson_r = np.nan
            pearson_p = np.nan
            spearman_rho = np.nan
            spearman_p = np.nan
        else:
            pr = stats.pearsonr(x, y)
            sr = stats.spearmanr(x, y)
            pearson_r, pearson_p = float(pr.statistic), float(pr.pvalue)
            spearman_rho, spearman_p = float(sr.statistic), float(sr.pvalue)
        # AUROC: positive = gt_frac >= 0.5
        bin_y = (y >= 0.5).astype(int)
        if bin_y.sum() == 0 or bin_y.sum() == len(bin_y):
            auroc = np.nan
        else:
            auroc = float(roc_auc_score(bin_y, x))
        rows.append({
            "branch": col,
            "n_windows": len(df),
            "n_clips": df["audio_path"].nunique(),
            "n_clips_with_child_rttm": n_clips_with_rttm,
            "pearson_r": pearson_r,
            "pearson_pval": pearson_p,
            "spearman_rho": spearman_rho,
            "spearman_pval": spearman_p,
            "auroc_ranking": auroc,
        })

    out_df = pd.DataFrame(rows)
    out_path = os.path.join(results_dir, f"branch_alignment_{args.split}.csv")
    out_df.to_csv(out_path, index=False)
    print(f"Wrote {out_path}")
    print(out_df.to_string(index=False))


if __name__ == "__main__":
    main()
