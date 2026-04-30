"""
Cross-child ensemble: mean of available system scores on the cross-child split.

Mirrors the best_audio_mil ensemble logic but uses cross-child predictions.
Requires the following prediction files to exist:
  - evaluation/cross_child_babar_role_only/val_predictions.csv   (prob col)
  - evaluation/cross_child_vtc_role_only/val_predictions.csv     (prob col)
  - mil/mil_results/wavlm_mil_cross_child/val_predictions.csv    (score col)
  - mil/mil_results/whisper_mil_cross_child/val_predictions.csv  (score col)

Plus optional:
  - baselines/audio_llm_baseline_runs/qwen2_audio_7b_cross_child/val_predictions.csv (prob col)

Usage:
    python evaluation/cross_child_ensemble.py
    python evaluation/cross_child_ensemble.py --output-dir ensemble_runs/cross_child_ensemble
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from mil.mil_utils import compute_metrics, tune_threshold, save_json, save_csv

SPLITS_DIR = _REPO / "baselines/splits"

SYSTEMS = {
    "babar_role":      (_REPO / "evaluation/cross_child_babar_role_only",   "prob"),
    "vtc_role":        (_REPO / "evaluation/cross_child_vtc_role_only",     "prob"),
    "wavlm_mil":       (_REPO / "mil/mil_results/wavlm_mil_cross_child",    "score"),
    "whisper_mil":     (_REPO / "mil/mil_results/whisper_mil_cross_child",  "score"),
    "audio_llm":       (_REPO / "baselines/audio_llm_baseline_runs/qwen2_audio_7b_cross_child", "prob"),
    "clap":            (_REPO / "baselines/clap_baseline_runs/clap_htsat_fused_cross_child", "prob"),
}


def load_preds(result_dir: Path, score_col: str, split: str) -> pd.Series | None:
    csv = result_dir / f"{split}_predictions.csv"
    if not csv.exists():
        return None
    df = pd.read_csv(csv)
    df = df.set_index("audio_path")
    return df[score_col].rename(result_dir.name)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", default=str(_REPO / "ensemble_runs/cross_child_best_audio_mil"))
    p.add_argument("--splits-dir", default=str(SPLITS_DIR))
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    splits_dir = Path(args.splits_dir)
    val_df  = pd.read_csv(splits_dir / "val.csv").set_index("audio_path")
    test_df = pd.read_csv(splits_dir / "test.csv").set_index("audio_path")
    if "audio_exists" in val_df.columns:
        val_df  = val_df[val_df["audio_exists"].astype(bool)]
        test_df = test_df[test_df["audio_exists"].astype(bool)]

    print("=== Cross-child ensemble ===")
    available = {}
    for name, (rdir, col) in SYSTEMS.items():
        s = load_preds(rdir, col, "val")
        if s is not None:
            available[name] = (rdir, col)
            print(f"  ✓ {name}")
        else:
            print(f"  ✗ {name}  (missing {rdir}/val_predictions.csv)")

    if len(available) < 2:
        print("ERROR: fewer than 2 systems available — cannot form ensemble.", file=sys.stderr)
        sys.exit(1)

    def assemble(split: str, ref_df: pd.DataFrame) -> pd.DataFrame:
        cols = {}
        for name, (rdir, col) in available.items():
            s = load_preds(rdir, col, split)
            if s is not None:
                cols[name] = s.reindex(ref_df.index)
        score_df = pd.DataFrame(cols)
        # Row-wise mean, ignoring NaN
        mean_score = score_df.mean(axis=1)
        result = ref_df[["label"]].copy()
        result["prob"]  = mean_score.values
        return result

    # Val: tune threshold
    val_scores = assemble("val", val_df)
    threshold = tune_threshold(val_scores["label"].values, val_scores["prob"].values)
    val_metrics = compute_metrics(val_scores["label"].values, val_scores["prob"].values, threshold)
    val_metrics.update({"threshold": threshold, "systems": list(available.keys()),
                        "n_systems": len(available), "n": len(val_scores),
                        "split_type": "cross_child"})
    save_json(val_metrics, str(out_dir / "val_metrics_tuned.json"))
    print(f"\nVal  F1={val_metrics['f1']:.4f}  AUROC={val_metrics['auroc']:.4f}  thr={threshold:.3f}")

    # Test
    test_scores = assemble("test", test_df)
    test_metrics = compute_metrics(test_scores["label"].values, test_scores["prob"].values, threshold)
    test_metrics.update({"threshold": threshold, "systems": list(available.keys()),
                         "n_systems": len(available), "n": len(test_scores),
                         "split_type": "cross_child"})
    save_json(test_metrics, str(out_dir / "test_metrics_tuned.json"))
    print(f"Test F1={test_metrics['f1']:.4f}  AUROC={test_metrics['auroc']:.4f}"
          f"  AUPRC={test_metrics['auprc']:.4f}")

    # Predictions
    pred_df = test_df[["label"]].copy()
    pred_df["prob"] = test_scores["prob"].values
    pred_df["pred"] = (test_scores["prob"].values >= threshold).astype(int)
    pred_df = pred_df.reset_index()
    save_csv(pred_df, str(out_dir / "test_predictions.csv"))

    save_json({"systems": list(available.keys()), "threshold": threshold,
               "splits_dir": str(splits_dir)}, str(out_dir / "config.json"))
    print(f"\nOutputs: {out_dir}")


if __name__ == "__main__":
    main()
