"""
Re-evaluate zero-shot systems (Qwen audio LLMs, YAMNet, AST) on the
BIDS-corrected cross-child split (baselines/splits/, n=444 val / n=742 test
audio_exists==True; 105 train / 23 val / 23 test child-disjoint children).

These systems all already have test_all_predictions.csv with scores for every
SAILS clip; we therefore re-index the per-clip score onto the BIDS val/test
partition, BA-tune the threshold on val, and evaluate on test --- no new
inference required, no GPU.

Writes outputs to <system_dir>_cross_child_bids/ alongside the existing
*_cross_child legacy-pre-BIDS dirs.
"""
from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd

REPO = Path("/orcd/scratch/orcd/008/manaal/child-adult-diarization")
sys.path.insert(0, str(REPO))
from mil.mil_utils import compute_metrics, tune_threshold  # noqa: E402

SPLITS_DIR = REPO / "baselines/splits"


def ba_tune(labels, probs, grid=None):
    if grid is None:
        grid = np.arange(0.05, 1.0, 0.05)
    best_thr, best_ba = 0.5, -1.0
    for t in grid:
        preds = (probs >= t).astype(int)
        tp = ((preds == 1) & (labels == 1)).sum()
        tn = ((preds == 0) & (labels == 0)).sum()
        fp = ((preds == 1) & (labels == 0)).sum()
        fn = ((preds == 0) & (labels == 1)).sum()
        tpr = tp / max(tp + fn, 1)
        tnr = tn / max(tn + fp, 1)
        ba = 0.5 * (tpr + tnr)
        if ba > best_ba:
            best_ba, best_thr = ba, float(t)
    return best_thr


def metrics_at_threshold(labels, probs, thr):
    labels = labels.astype(int)
    preds = (probs >= thr).astype(int)
    tp = ((preds == 1) & (labels == 1)).sum()
    tn = ((preds == 0) & (labels == 0)).sum()
    fp = ((preds == 1) & (labels == 0)).sum()
    fn = ((preds == 0) & (labels == 1)).sum()
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    tnr = tn / max(tn + fp, 1)
    ba = 0.5 * (recall + tnr)
    # Macro and weighted F1
    f1_pos = f1
    prec_neg = tn / max(tn + fn, 1)
    rec_neg = tn / max(tn + fp, 1)
    f1_neg = 2 * prec_neg * rec_neg / max(prec_neg + rec_neg, 1e-9)
    macro = 0.5 * (f1_pos + f1_neg)
    n = len(labels); npos = int(labels.sum()); nneg = n - npos
    weighted = (npos / n) * f1_pos + (nneg / n) * f1_neg
    # AUROC, AUPRC
    from sklearn.metrics import roc_auc_score, average_precision_score
    try:
        auroc = roc_auc_score(labels, probs)
    except Exception:
        auroc = float("nan")
    try:
        auprc = average_precision_score(labels, probs)
    except Exception:
        auprc = float("nan")
    return dict(
        f1=f1, f1_macro=macro, f1_weighted=weighted,
        balanced_accuracy=ba, precision=precision, recall=recall,
        auroc=auroc, auprc=auprc, threshold=thr,
        n=int(n), n_pos=int(npos), n_neg=int(nneg),
    )


SYSTEMS = {
    "qwen2_audio_7b": REPO / "baselines/audio_llm_baseline_runs/qwen2_audio_7b",
    "qwen25_omni_7b": REPO / "baselines/audio_llm_baseline_runs/qwen25_omni_7b",
    "qwen3_omni_30b_thinking": REPO / "baselines/audio_llm_baseline_runs/qwen3_omni_30b_thinking",
    "yamnet": REPO / "baselines/scene_analysis_runs/yamnet",
    "ast": REPO / "baselines/scene_analysis_runs/ast",
}


def main():
    val = pd.read_csv(SPLITS_DIR / "val.csv")
    test = pd.read_csv(SPLITS_DIR / "test.csv")
    val = val[val.audio_exists.astype(bool)].reset_index(drop=True)
    test = test[test.audio_exists.astype(bool)].reset_index(drop=True)
    print(f"BIDS cross-child: val={len(val)} (pos={int(val.label.sum())}), "
          f"test={len(test)} (pos={int(test.label.sum())})\n")

    summary = []
    for name, src in SYSTEMS.items():
        all_preds = src / "test_all_predictions.csv"
        if not all_preds.exists():
            print(f"[{name}] SKIP: no {all_preds}")
            continue
        df = pd.read_csv(all_preds)
        if "prob" not in df.columns:
            print(f"[{name}] SKIP: no 'prob' column in {all_preds}")
            continue
        df = df[["audio_path", "prob"]].drop_duplicates(subset=["audio_path"])

        val_m = val.merge(df, on="audio_path", how="left")
        test_m = test.merge(df, on="audio_path", how="left")
        n_val_missing = val_m.prob.isna().sum()
        n_test_missing = test_m.prob.isna().sum()
        if n_val_missing or n_test_missing:
            print(f"[{name}] WARN: {n_val_missing} val + {n_test_missing} test clips "
                  f"missing from {all_preds}; dropping them")
            val_m = val_m.dropna(subset=["prob"]).reset_index(drop=True)
            test_m = test_m.dropna(subset=["prob"]).reset_index(drop=True)

        val_y = val_m.label.astype(int).values; val_p = val_m.prob.astype(float).values
        thr = ba_tune(val_y, val_p)
        val_metrics = metrics_at_threshold(val_y, val_p, thr)
        test_y = test_m.label.astype(int).values; test_p = test_m.prob.astype(float).values
        test_metrics = metrics_at_threshold(test_y, test_p, thr)

        out = REPO / f"{src.relative_to(REPO)}_cross_child_bids"
        out.mkdir(parents=True, exist_ok=True)
        (out / "val_metrics_tuned.json").write_text(json.dumps(val_metrics, indent=2))
        (out / "test_metrics_tuned.json").write_text(json.dumps(test_metrics, indent=2))
        vp = val_m[["audio_path", "child_id", "timepoint_norm", "label"]].copy()
        vp["prob"] = val_p; vp["pred"] = (val_p >= thr).astype(int)
        vp.to_csv(out / "val_predictions.csv", index=False)
        tp = test_m[["audio_path", "child_id", "timepoint_norm", "label"]].copy()
        tp["prob"] = test_p; tp["pred"] = (test_p >= thr).astype(int)
        tp.to_csv(out / "test_predictions.csv", index=False)
        (out / "config.json").write_text(json.dumps({
            "system": name,
            "split": "cross_child_bids (baselines/splits/, audio_exists=True)",
            "source_all_predictions": str(all_preds),
            "val_n": len(val_m), "test_n": len(test_m),
            "ba_tuned_threshold": thr,
        }, indent=2))
        print(f"[{name}] BIDS cross-child: val BA={val_metrics['balanced_accuracy']:.4f} "
              f"AUROC={val_metrics['auroc']:.4f}  "
              f"test BA={test_metrics['balanced_accuracy']:.4f} "
              f"AUROC={test_metrics['auroc']:.4f}  thr={thr:.2f}  -> {out}")

        summary.append({
            "system": name,
            "val_n": len(val_m), "test_n": len(test_m),
            "thr_ba": thr,
            "val_ba": round(val_metrics["balanced_accuracy"], 4),
            "val_auroc": round(val_metrics["auroc"], 4),
            "test_f1": round(test_metrics["f1"], 4),
            "test_f1_weighted": round(test_metrics["f1_weighted"], 4),
            "test_ba": round(test_metrics["balanced_accuracy"], 4),
            "test_precision": round(test_metrics["precision"], 4),
            "test_recall": round(test_metrics["recall"], 4),
            "test_auroc": round(test_metrics["auroc"], 4),
            "test_auprc": round(test_metrics["auprc"], 4),
        })

    sum_df = pd.DataFrame(summary)
    print("\n=== SUMMARY ===")
    print(sum_df.to_string(index=False))
    sum_csv = REPO / "evaluation/cross_child_bids_zero_shot_summary.csv"
    sum_df.to_csv(sum_csv, index=False)
    print(f"\nWrote {sum_csv}")


if __name__ == "__main__":
    main()
