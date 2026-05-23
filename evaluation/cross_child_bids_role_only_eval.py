"""
Cross-child BIDS role-only evaluation for BabAR/VTC/VTC-KCHI diarizers.

Same algorithm as cross_child_diarizer_eval.py but
  - uses Vid_duration column from the splits CSV (no torchaudio dep) and
    falls back to wave / stdlib for any clips where Vid_duration is missing
    or malformed,
  - writes outputs to evaluation/cross_child_<diarizer>_role_only_bids/
    so the legacy n=496 dirs are preserved.

If a clip is in the BIDS split but its RTTM is missing from the cache, it
gets score=0 (predicted negative). This biases the role-only number on the
missing-RTTM subset; the missing count is logged in config.json for
transparency.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import wave
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

REPO = Path("/orcd/scratch/orcd/008/manaal/child-adult-diarization")
BABAR_RTTM = REPO / "babar/babar_output/rttm"
VTC_RTTM = REPO / "pyannote/vtc_rttm_cache"
SPLITS_DIR = REPO / "baselines/splits"


def cid(p): return hashlib.md5(p.encode()).hexdigest()
def rttm_babar(ap): return BABAR_RTTM / f"{Path(ap).stem}__{cid(ap)}.rttm"
def rttm_vtc(ap): return VTC_RTTM / f"{Path(ap).stem}__{cid(ap)}.rttm"


def parse_rttm_dur(p: Path, child_labels: set) -> float:
    if not p.exists() or p.stat().st_size == 0:
        return 0.0
    total = 0.0
    with open(p) as f:
        for line in f:
            line = line.strip()
            if not line.startswith("SPEAKER"):
                continue
            parts = line.split()
            if len(parts) < 8:
                continue
            try:
                dur = float(parts[4])
            except ValueError:
                continue
            if parts[7] in child_labels:
                total += dur
    return total


def parse_vid_duration(s) -> float:
    """Vid_duration like '00:20.9' is MM:SS.s."""
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return float("nan")
    s = str(s).strip()
    if ":" in s:
        m, ss = s.split(":")
        try:
            return float(m) * 60.0 + float(ss)
        except ValueError:
            return float("nan")
    try:
        return float(s)
    except ValueError:
        return float("nan")


def wav_duration(ap: str) -> float:
    try:
        with wave.open(ap) as w:
            return w.getnframes() / w.getframerate()
    except Exception:
        return 30.0


def ba_tune(labels, probs, grid=None):
    if grid is None:
        grid = np.arange(0.0, 1.0001, 0.05)
    best_thr, best_ba = 0.5, -1.0
    for t in grid:
        preds = (probs >= t).astype(int)
        tp = ((preds == 1) & (labels == 1)).sum()
        tn = ((preds == 0) & (labels == 0)).sum()
        fp = ((preds == 1) & (labels == 0)).sum()
        fn = ((preds == 0) & (labels == 1)).sum()
        tpr = tp / max(tp + fn, 1); tnr = tn / max(tn + fp, 1)
        ba = 0.5 * (tpr + tnr)
        if ba > best_ba: best_ba, best_thr = ba, float(t)
    return best_thr


def metrics(labels, probs, thr):
    labels = labels.astype(int); preds = (probs >= thr).astype(int)
    tp = ((preds == 1) & (labels == 1)).sum(); tn = ((preds == 0) & (labels == 0)).sum()
    fp = ((preds == 1) & (labels == 0)).sum(); fn = ((preds == 0) & (labels == 1)).sum()
    prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    tnr = tn / max(tn + fp, 1); ba = 0.5 * (rec + tnr)
    prec_neg = tn / max(tn + fn, 1); rec_neg = tn / max(tn + fp, 1)
    f1_neg = 2 * prec_neg * rec_neg / max(prec_neg + rec_neg, 1e-9)
    macro = 0.5 * (f1 + f1_neg)
    n = len(labels); npos = int(labels.sum()); nneg = n - npos
    weighted = (npos / n) * f1 + (nneg / n) * f1_neg
    try: auroc = roc_auc_score(labels, probs)
    except Exception: auroc = float("nan")
    try: auprc = average_precision_score(labels, probs)
    except Exception: auprc = float("nan")
    return dict(f1=float(f1), f1_macro=float(macro), f1_weighted=float(weighted),
                balanced_accuracy=float(ba), precision=float(prec), recall=float(rec),
                auroc=float(auroc), auprc=float(auprc), threshold=float(thr),
                n=int(n), n_pos=npos, n_neg=nneg)


def score(df: pd.DataFrame, rttm_fn, child_labels: set) -> tuple[np.ndarray, int]:
    audio_paths = df["audio_path"].tolist()
    if "Vid_duration" in df.columns:
        durations = df["Vid_duration"].apply(parse_vid_duration).tolist()
    else:
        durations = [float("nan")] * len(audio_paths)
    probs = np.zeros(len(audio_paths))
    miss = 0
    for i, ap in enumerate(audio_paths):
        p = rttm_fn(ap)
        if not p.exists():
            miss += 1; probs[i] = 0.0; continue
        cd = parse_rttm_dur(p, child_labels)
        cl_dur = durations[i]
        if np.isnan(cl_dur) or cl_dur <= 0:
            cl_dur = wav_duration(ap)
        probs[i] = min(cd / max(cl_dur, 1e-3), 1.0)
    return probs, miss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--diarizer", choices=["babar", "vtc", "vtc_kchi"], required=True)
    args = ap.parse_args()

    if args.diarizer == "babar":
        rttm_fn, labels = rttm_babar, {"KCHI"}
    elif args.diarizer == "vtc":
        rttm_fn, labels = rttm_vtc, {"KCHI", "OCH"}
    else:
        rttm_fn, labels = rttm_vtc, {"KCHI"}

    out = REPO / f"evaluation/cross_child_{args.diarizer}_role_only_bids"
    out.mkdir(parents=True, exist_ok=True)

    val = pd.read_csv(SPLITS_DIR / "val.csv")
    test = pd.read_csv(SPLITS_DIR / "test.csv")
    val = val[val.audio_exists.astype(bool)].reset_index(drop=True)
    test = test[test.audio_exists.astype(bool)].reset_index(drop=True)
    print(f"BIDS cross-child: val n={len(val)} (pos={int(val.label.sum())}), "
          f"test n={len(test)} (pos={int(test.label.sum())})")

    val_p, val_miss = score(val, rttm_fn, labels)
    thr = ba_tune(val.label.astype(int).values, val_p)
    val_m = metrics(val.label.astype(int).values, val_p, thr)

    test_p, test_miss = score(test, rttm_fn, labels)
    test_m = metrics(test.label.astype(int).values, test_p, thr)

    print(f"  [{args.diarizer}]  thr={thr:.3f}  "
          f"val_BA={val_m['balanced_accuracy']:.4f} val_AUROC={val_m['auroc']:.4f} "
          f"val_RTTM_miss={val_miss}/{len(val)}; "
          f"test_BA={test_m['balanced_accuracy']:.4f} test_AUROC={test_m['auroc']:.4f} "
          f"test_RTTM_miss={test_miss}/{len(test)}")

    (out / "val_metrics_tuned.json").write_text(json.dumps(val_m, indent=2))
    (out / "test_metrics_tuned.json").write_text(json.dumps(test_m, indent=2))

    vp = val[["audio_path", "child_id", "timepoint_norm", "label"]].copy()
    vp["prob"] = val_p; vp["pred"] = (val_p >= thr).astype(int)
    vp.to_csv(out / "val_predictions.csv", index=False)
    tp = test[["audio_path", "child_id", "timepoint_norm", "label"]].copy()
    tp["prob"] = test_p; tp["pred"] = (test_p >= thr).astype(int)
    tp.to_csv(out / "test_predictions.csv", index=False)
    (out / "config.json").write_text(json.dumps({
        "diarizer": args.diarizer, "split": "cross_child_bids",
        "val_n": int(len(val)), "test_n": int(len(test)),
        "val_rttm_missing": int(val_miss), "test_rttm_missing": int(test_miss),
        "ba_tuned_threshold": float(thr),
        "note": ("RTTM cache miss = score 0.0; missing fraction logged here. "
                 "BabAR RTTM cache built on within-child SAILS pool; "
                 "BIDS-recovered cross-child test clips may miss RTTM.")
    }, indent=2))
    print(f"  Wrote -> {out}")


if __name__ == "__main__":
    main()
