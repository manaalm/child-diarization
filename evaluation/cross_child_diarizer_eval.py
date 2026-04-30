"""
Cross-child role-only evaluation for BabAR and VTC diarizers.

Since test children are disjoint from train in the cross-child split,
ECAPA enrollment is not possible. Instead, we score each clip by the
fraction of child-labeled speech detected by the diarizer:

  score = total_child_duration / clip_duration

For BabAR: child labels = KCHI
For VTC:   child labels = KCHI + OCH

RTTMs are read from the existing seen-child caches (babar/babar_output/rttm/
and pyannote/vtc_rttm_cache/). For the ~51 clips not yet cached, we run VTC
inline (requires BabAR/VTC uv env).

Usage:
    python evaluation/cross_child_diarizer_eval.py --diarizer babar
    python evaluation/cross_child_diarizer_eval.py --diarizer vtc
    python evaluation/cross_child_diarizer_eval.py --diarizer vtc_kchi
"""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torchaudio

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from mil.mil_utils import compute_metrics, tune_threshold, save_json, save_csv

BABAR_RTTM_DIR = _REPO / "babar/babar_output/rttm"
VTC_RTTM_DIR   = _REPO / "pyannote/vtc_rttm_cache"
VTC_DIR        = _REPO / "BabAR/VTC"
VTC_STAGING    = _REPO / "pyannote/vtc_input_staging"

SPLITS_DIR     = _REPO / "baselines/splits"

BABAR_BASELINE = {"f1": 0.874, "auroc": 0.820, "auprc": 0.918}


# ---------------------------------------------------------------------------

def audio_to_cache_id(path: str) -> str:
    return hashlib.md5(path.encode()).hexdigest()


def rttm_path_vtc(audio_path: str) -> Path:
    stem = Path(audio_path).stem
    cid  = audio_to_cache_id(audio_path)
    return VTC_RTTM_DIR / f"{stem}__{cid}.rttm"


def rttm_path_babar(audio_path: str) -> Path:
    stem = Path(audio_path).stem
    cid  = audio_to_cache_id(audio_path)
    return BABAR_RTTM_DIR / f"{stem}__{cid}.rttm"


def parse_rttm_child_duration(rttm_path: Path, child_labels: set) -> float:
    total = 0.0
    if not rttm_path.exists() or rttm_path.stat().st_size == 0:
        return 0.0
    with open(rttm_path) as f:
        for line in f:
            line = line.strip()
            if not line.startswith("SPEAKER"):
                continue
            parts = line.split()
            if len(parts) < 8:
                continue
            dur, label = float(parts[4]), parts[7]
            if label in child_labels:
                total += dur
    return total


def get_clip_duration(audio_path: str) -> float:
    info = torchaudio.info(audio_path)
    return info.num_frames / info.sample_rate


def stage_and_run_vtc(missing_paths: list) -> None:
    """Run VTC on uncached clips, caching results into VTC_RTTM_DIR."""
    VTC_STAGING.mkdir(parents=True, exist_ok=True)
    staged = []
    for ap in missing_paths:
        stem = Path(ap).stem
        cid  = audio_to_cache_id(ap)
        sp   = VTC_STAGING / f"{stem}__{cid}.wav"
        if not sp.exists():
            wav, sr = torchaudio.load(ap)
            if wav.shape[0] > 1:
                wav = wav.mean(dim=0, keepdim=True)
            if sr != 16000:
                wav = torchaudio.functional.resample(wav, sr, 16000)
            torchaudio.save(str(sp), wav, 16000)
        staged.append((ap, sp))

    with tempfile.TemporaryDirectory() as tmp:
        input_dir = Path(tmp) / "wavs"
        input_dir.mkdir()
        for ap, sp in staged:
            dst = input_dir / sp.name
            if not dst.exists():
                os.symlink(str(sp), str(dst))

        ckpt   = VTC_DIR / "VTC-2.0/model/best.ckpt"
        config = VTC_DIR / "VTC-2.0/model/config.yml"
        cmd = [
            "uv", "run", "python", "scripts/infer.py",
            "--wavs",       str(input_dir),
            "--output",     tmp,
            "--config",     str(config),
            "--checkpoint", str(ckpt),
            "--device",     "cuda",
            "--batch_size", "4",
            "--min_duration_on_s",  "0.1",
            "--min_duration_off_s", "0.1",
        ]
        print(f"  Running VTC on {len(staged)} clips ...")
        subprocess.run(cmd, cwd=str(VTC_DIR), check=True)

        # Copy results to VTC cache
        for ap, sp in staged:
            src = Path(tmp) / "rttm" / f"{sp.stem}.rttm"
            dst = rttm_path_vtc(ap)
            if src.exists():
                shutil.copy2(str(src), str(dst))
            else:
                open(str(dst), "w").close()


def score_clips(audio_paths: list, rttm_fn, child_labels: set,
                run_vtc_for_missing: bool) -> pd.Series:
    # Identify missing
    missing = [ap for ap in audio_paths if not rttm_fn(ap).exists()]
    if missing:
        print(f"  {len(missing)} clips missing from cache", end="")
        if run_vtc_for_missing:
            print(f" — running VTC ...")
            stage_and_run_vtc(missing)
        else:
            print(f" — assigning score=0.0 (BabAR cache only, no live inference)")

    scores = {}
    n_zero = 0
    for ap in audio_paths:
        rttm = rttm_fn(ap)
        child_dur = parse_rttm_child_duration(rttm, child_labels)
        try:
            clip_dur = get_clip_duration(ap)
        except Exception:
            clip_dur = 30.0
        scores[ap] = min(child_dur / max(clip_dur, 1e-3), 1.0)
        if scores[ap] == 0.0:
            n_zero += 1
    print(f"  Scored {len(scores)} clips ({n_zero} zero-score, "
          f"{len(scores)-n_zero} non-zero)")
    return pd.Series(scores)


# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--diarizer", choices=["babar", "vtc", "vtc_kchi"],
                   default="babar")
    p.add_argument("--splits-dir", default=str(SPLITS_DIR))
    p.add_argument("--output-dir", default=None)
    p.add_argument("--run-vtc-for-missing", action="store_true",
                   help="Run VTC to fill missing cache entries (requires GPU + uv env)")
    args = p.parse_args()

    if args.diarizer == "babar":
        rttm_fn      = rttm_path_babar
        child_labels = {"KCHI"}
        run_vtc      = False
    elif args.diarizer == "vtc":
        rttm_fn      = rttm_path_vtc
        child_labels = {"KCHI", "OCH"}
        run_vtc      = args.run_vtc_for_missing
    else:  # vtc_kchi
        rttm_fn      = rttm_path_vtc
        child_labels = {"KCHI"}
        run_vtc      = args.run_vtc_for_missing

    out_dir = Path(args.output_dir) if args.output_dir else (
        _REPO / f"evaluation/cross_child_{args.diarizer}_role_only"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    splits_dir = Path(args.splits_dir)
    val_df  = pd.read_csv(splits_dir / "val.csv")
    test_df = pd.read_csv(splits_dir / "test.csv")
    if "audio_exists" in val_df.columns:
        val_df  = val_df[val_df["audio_exists"].astype(bool)]
        test_df = test_df[test_df["audio_exists"].astype(bool)]
    val_df  = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    print(f"=== Cross-child role-only: {args.diarizer} ===")
    print(f"  Val:  {len(val_df)} clips ({val_df['label'].sum()} pos)")
    print(f"  Test: {len(test_df)} clips ({test_df['label'].sum()} pos)")

    # --- Val: tune threshold ---
    print("\n[Val] Scoring ...")
    val_scores = score_clips(list(val_df["audio_path"]), rttm_fn, child_labels, run_vtc)
    val_df["prob"] = val_scores.values

    threshold = tune_threshold(val_df["label"].values, val_df["prob"].values)
    val_metrics = compute_metrics(val_df["label"].values, val_df["prob"].values, threshold)
    val_metrics.update({"threshold": threshold, "diarizer": args.diarizer,
                        "n": len(val_df), "split_type": "cross_child"})
    save_json(val_metrics, str(out_dir / "val_metrics_tuned.json"))
    print(f"  Val  F1={val_metrics['f1']:.4f}  AUROC={val_metrics['auroc']:.4f}"
          f"  thr={threshold:.3f}")

    # --- Test: apply val threshold ---
    print("\n[Test] Scoring ...")
    test_scores = score_clips(list(test_df["audio_path"]), rttm_fn, child_labels, run_vtc)
    test_df["prob"] = test_scores.values

    test_metrics = compute_metrics(test_df["label"].values, test_df["prob"].values, threshold)
    test_metrics.update({
        "threshold":    threshold,
        "diarizer":     args.diarizer,
        "n":            len(test_df),
        "split_type":   "cross_child",
        "delta_f1":     round(test_metrics["f1"]    - BABAR_BASELINE["f1"],    4),
        "delta_auroc":  round(test_metrics["auroc"] - BABAR_BASELINE["auroc"], 4),
    })
    save_json(test_metrics, str(out_dir / "test_metrics_tuned.json"))
    print(f"  Test F1={test_metrics['f1']:.4f}  AUROC={test_metrics['auroc']:.4f}"
          f"  AUPRC={test_metrics['auprc']:.4f}")
    print(f"  vs BabAR seen-child: ΔF1={test_metrics['delta_f1']:+.4f}"
          f"  ΔAUROC={test_metrics['delta_auroc']:+.4f}")

    # Per-timepoint
    if "timepoint_norm" in test_df.columns:
        tp_rows = []
        for tp, grp in test_df.groupby("timepoint_norm"):
            m = compute_metrics(grp["label"].values, grp["prob"].values, threshold)
            tp_rows.append({"timepoint_norm": tp, **m})
        if tp_rows:
            save_csv(pd.DataFrame(tp_rows), str(out_dir / "test_metrics_by_timepoint.csv"))

    # Predictions CSV
    pred_df = test_df[["audio_path", "child_id", "timepoint_norm", "label"]].copy() \
        if "child_id" in test_df.columns else test_df[["audio_path", "label"]].copy()
    pred_df["prob"] = test_df["prob"].values
    pred_df["pred"] = (test_df["prob"].values >= threshold).astype(int)
    save_csv(pred_df, str(out_dir / "test_predictions.csv"))

    # Val predictions (for downstream ensemble)
    val_pred_df = val_df[["audio_path", "child_id", "timepoint_norm", "label"]].copy() \
        if "child_id" in val_df.columns else val_df[["audio_path", "label"]].copy()
    val_pred_df["prob"] = val_df["prob"].values
    val_pred_df["pred"] = (val_df["prob"].values >= threshold).astype(int)
    save_csv(val_pred_df, str(out_dir / "val_predictions.csv"))

    config = {"diarizer": args.diarizer, "child_labels": list(child_labels),
              "split_type": "cross_child", "splits_dir": str(splits_dir),
              "threshold": threshold}
    save_json(config, str(out_dir / "config.json"))

    print(f"\nOutputs written to: {out_dir}")


if __name__ == "__main__":
    main()
