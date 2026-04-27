"""
Child vocalization detection using nvidia/parakeet-tdt-0.6b-v2.

Strategy: ASR word-level timestamps expose what the adult-trained model
*recognizes*. Child speech and vocalizations leave unrecognized gaps.
Score = gap_ratio = 1 - (covered_word_seconds / clip_duration).
High gap_ratio → more unrecognized audio → more likely child is present.

Usage:
    python baselines/parakeet_baseline.py --split val
    python baselines/parakeet_baseline.py --split test   # val must run first
    python baselines/parakeet_baseline.py --split val --dry-run --max-clips 5
"""

import argparse
import hashlib
import json
import os
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchaudio

REPO_ROOT = Path(__file__).resolve().parent.parent
BABAR_BASELINES = {"f1": 0.874, "auroc": 0.820, "auprc": 0.918}
AUDIO_LLM_BASELINES = {"f1": 0.871, "auroc": 0.725, "auprc": 0.853}
MODEL_NAME = "nvidia/parakeet-tdt-0.6b-v2"
MODEL_SLUG = "parakeet_tdt_0.6b_v2"


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def _cache_path(audio_path: str, cache_dir: str) -> str:
    stem = Path(audio_path).stem
    md5 = hashlib.md5(audio_path.encode()).hexdigest()[:12]
    return str(Path(cache_dir) / f"{stem}__{md5}.json")


def _load_cache(path: str):
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def _save_cache(path: str, entry: dict):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(entry, f)
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# Audio duration helper
# ---------------------------------------------------------------------------

def _clip_duration(audio_path: str) -> float:
    try:
        info = torchaudio.info(audio_path)
        return info.num_frames / info.sample_rate
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# Word timestamp parsing
# ---------------------------------------------------------------------------

def _word_coverage(hypothesis, clip_duration: float, frame_stride_sec: float = 0.04) -> dict:
    """
    Extract word-level coverage from a NeMo Hypothesis.
    Returns gap_ratio (1 - covered / duration) and auxiliary features.

    NeMo 2.x timestamp format:
      hypothesis.timestamp['word'] = [
          {'word': str, 'start_offset': int, 'end_offset': int},
          ...
      ]
    Offsets are in encoder-output frames. FastConformer TDT with 8x
    subsampling at 10ms frame shift = 80ms per stride (0.08s).
    For safety we detect second-valued offsets vs frame-index offsets.
    """
    word_stamps = []
    if hasattr(hypothesis, "timestamp") and hypothesis.timestamp:
        word_stamps = hypothesis.timestamp.get("word", [])

    if not word_stamps:
        # No recognized speech → maximum gap score
        return {"gap_ratio": 1.0, "word_count": 0,
                "covered_sec": 0.0, "words_per_sec": 0.0,
                "clip_duration": clip_duration}

    # Determine if offsets are already in seconds or frame indices
    max_offset = max(w.get("end_offset", 0) for w in word_stamps)
    if max_offset > clip_duration * 10:
        # Frame indices — convert using frame_stride_sec
        scale = frame_stride_sec
    else:
        # Already in seconds
        scale = 1.0

    covered = 0.0
    for w in word_stamps:
        start = w.get("start_offset", 0) * scale
        end = w.get("end_offset", 0) * scale
        if end > start:
            covered += min(end, clip_duration) - max(start, 0.0)

    covered = min(covered, clip_duration)
    gap_ratio = 1.0 - (covered / clip_duration) if clip_duration > 0 else 1.0
    words_per_sec = len(word_stamps) / clip_duration if clip_duration > 0 else 0.0

    return {
        "gap_ratio": float(gap_ratio),
        "word_count": len(word_stamps),
        "covered_sec": float(covered),
        "words_per_sec": float(words_per_sec),
        "clip_duration": float(clip_duration),
    }


# ---------------------------------------------------------------------------
# Metrics & threshold
# ---------------------------------------------------------------------------

def _compute_metrics(y_true, y_score, threshold: float) -> dict:
    from sklearn.metrics import (f1_score, precision_score, recall_score,
                                 roc_auc_score, average_precision_score)
    y_pred = (y_score >= threshold).astype(int)
    return {
        "threshold": float(threshold),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "auroc": float(roc_auc_score(y_true, y_score)),
        "auprc": float(average_precision_score(y_true, y_score)),
        "n_pos": int(y_true.sum()),
        "n_total": int(len(y_true)),
    }


def _tune_threshold(y_true, y_score) -> float:
    from sklearn.metrics import f1_score
    best_t, best_f1 = 0.5, 0.0
    for t in np.linspace(0.01, 0.99, 199):
        f1 = f1_score(y_true, (y_score >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return best_t


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="val", choices=["val", "test"])
    parser.add_argument("--model-name", default=MODEL_NAME)
    parser.add_argument("--model-slug", default=MODEL_SLUG)
    parser.add_argument("--splits-dir", default=str(REPO_ROOT / "whisper-modeling/seen_child_splits"))
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-clips", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true",
                        help="Print first 3 clips with features and exit")
    args = parser.parse_args()

    np.random.seed(args.seed)

    if args.output_dir is None:
        args.output_dir = str(REPO_ROOT / f"baselines/parakeet_baseline_runs/{args.model_slug}")
    if args.cache_dir is None:
        args.cache_dir = str(REPO_ROOT / f"baselines/parakeet_cache/{args.model_slug}")

    out_dir = Path(args.output_dir)
    cache_dir = Path(args.cache_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Guard: test requires val threshold
    val_metrics_path = out_dir / "val_metrics_tuned.json"
    if args.split == "test" and not val_metrics_path.exists():
        print(f"ERROR: val_metrics_tuned.json not found at {val_metrics_path}. "
              f"Run --split val first.", file=sys.stderr)
        sys.exit(2)

    # Load split CSV
    split_csv = Path(args.splits_dir) / f"{args.split}.csv"
    df = pd.read_csv(split_csv)
    if args.max_clips:
        df = df.head(args.max_clips)

    print(f"Split: {args.split}  |  {len(df)} clips  |  model: {args.model_name}")

    if args.dry_run:
        print("\n[DRY RUN] First 3 clips (no model loaded):")
        for _, row in df.head(3).iterrows():
            dur = _clip_duration(str(row["audio_path"]))
            print(f"  {Path(row['audio_path']).name}  dur={dur:.1f}s  label={row['label']}")
        print("Dry run complete — no model loaded.")
        return

    # Load model
    print("Loading Parakeet TDT model...")
    import nemo.collections.asr as nemo_asr
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    model = nemo_asr.models.ASRModel.from_pretrained(
        model_name=args.model_name,
        map_location=device,
    )
    model.eval()
    if device == "cuda":
        model = model.cuda()
    print("Model loaded.")

    # Inference
    rows = []
    audio_paths = df["audio_path"].tolist()
    labels = df["label"].tolist()
    child_ids = df.get("child_id", pd.Series([None] * len(df))).tolist()
    timepoints = df.get("timepoint_norm", pd.Series([None] * len(df))).tolist()

    # Process in batches
    n = len(audio_paths)
    for batch_start in range(0, n, args.batch_size):
        batch_paths = audio_paths[batch_start: batch_start + args.batch_size]
        batch_labels = labels[batch_start: batch_start + args.batch_size]
        batch_children = child_ids[batch_start: batch_start + args.batch_size]
        batch_tp = timepoints[batch_start: batch_start + args.batch_size]

        # Check cache
        uncached_indices = []
        cached_features = {}
        for i, p in enumerate(batch_paths):
            cp = _cache_path(p, str(cache_dir))
            cached = _load_cache(cp)
            if cached is not None:
                cached_features[i] = cached
            else:
                uncached_indices.append(i)

        # Run model on uncached clips
        if uncached_indices:
            paths_to_run = [batch_paths[i] for i in uncached_indices]
            try:
                with torch.no_grad():
                    hypotheses = model.transcribe(
                        paths_to_run,
                        timestamps=True,
                        batch_size=len(paths_to_run),
                        verbose=False,
                    )
                # hypotheses may be a list or nested list
                if hypotheses and isinstance(hypotheses[0], list):
                    hypotheses = [h[0] for h in hypotheses]

                for idx, (orig_i, hyp) in enumerate(zip(uncached_indices, hypotheses)):
                    dur = _clip_duration(batch_paths[orig_i])
                    feat = _word_coverage(hyp, dur)
                    feat["text"] = hyp.text if hasattr(hyp, "text") else ""
                    cp = _cache_path(batch_paths[orig_i], str(cache_dir))
                    _save_cache(cp, feat)
                    cached_features[orig_i] = feat

            except Exception as e:
                warnings.warn(f"Batch {batch_start}-{batch_start+len(paths_to_run)} error: {e}")
                for orig_i in uncached_indices:
                    dur = _clip_duration(batch_paths[orig_i])
                    cached_features[orig_i] = {
                        "gap_ratio": 1.0, "word_count": 0,
                        "covered_sec": 0.0, "words_per_sec": 0.0,
                        "clip_duration": dur, "text": "", "error": str(e),
                    }

        for i, (p, lbl, cid, tp) in enumerate(
                zip(batch_paths, batch_labels, batch_children, batch_tp)):
            feat = cached_features.get(i, {"gap_ratio": 1.0, "word_count": 0,
                                           "covered_sec": 0.0, "words_per_sec": 0.0,
                                           "clip_duration": 0.0})
            rows.append({
                "audio_path": p,
                "child_id": cid,
                "timepoint_norm": tp,
                "label": int(lbl),
                "score": feat["gap_ratio"],  # primary score
                "gap_ratio": feat["gap_ratio"],
                "word_count": feat.get("word_count", 0),
                "covered_sec": feat.get("covered_sec", 0.0),
                "words_per_sec": feat.get("words_per_sec", 0.0),
                "clip_duration": feat.get("clip_duration", 0.0),
                "text": feat.get("text", ""),
            })

        done = min(batch_start + args.batch_size, n)
        print(f"  [{done}/{n}] done", flush=True)

    pred_df = pd.DataFrame(rows)
    y_true = pred_df["label"].values
    y_score = pred_df["score"].values

    # Save predictions
    pred_df.to_csv(out_dir / f"{args.split}_predictions.csv", index=False)

    if args.split == "val":
        threshold = _tune_threshold(y_true, y_score)
        metrics = _compute_metrics(y_true, y_score, threshold)
        metrics["val_f1"] = metrics["f1"]
        metrics["model"] = args.model_name
        metrics["split"] = "val"
        metrics["n_clips"] = len(pred_df)
        metrics["timestamp"] = datetime.now(timezone.utc).isoformat()
        print(f"\nVal metrics: F1={metrics['f1']:.3f} AUROC={metrics['auroc']:.3f} "
              f"AUPRC={metrics['auprc']:.3f} threshold={threshold:.3f}")
        print(f"  vs BabAR:    F1={BABAR_BASELINES['f1']:.3f} "
              f"AUROC={BABAR_BASELINES['auroc']:.3f} AUPRC={BABAR_BASELINES['auprc']:.3f}")
        with open(out_dir / "val_metrics_tuned.json", "w") as f:
            json.dump(metrics, f, indent=2)

    else:
        with open(val_metrics_path) as f:
            val_m = json.load(f)
        threshold = val_m["threshold"]
        metrics = _compute_metrics(y_true, y_score, threshold)
        metrics["model"] = args.model_name
        metrics["split"] = "test"
        metrics["val_f1"] = val_m.get("val_f1", val_m.get("f1"))
        metrics["delta_f1_vs_babar"] = metrics["f1"] - BABAR_BASELINES["f1"]
        metrics["delta_auroc_vs_babar"] = metrics["auroc"] - BABAR_BASELINES["auroc"]
        metrics["delta_auprc_vs_babar"] = metrics["auprc"] - BABAR_BASELINES["auprc"]
        metrics["delta_f1_vs_audio_llm"] = metrics["f1"] - AUDIO_LLM_BASELINES["f1"]
        metrics["delta_auroc_vs_audio_llm"] = metrics["auroc"] - AUDIO_LLM_BASELINES["auroc"]
        metrics["delta_auprc_vs_audio_llm"] = metrics["auprc"] - AUDIO_LLM_BASELINES["auprc"]
        print(f"\nTest metrics: F1={metrics['f1']:.3f} AUROC={metrics['auroc']:.3f} "
              f"AUPRC={metrics['auprc']:.3f}")
        print(f"  vs BabAR:    F1={BABAR_BASELINES['f1']:.3f} "
              f"AUROC={BABAR_BASELINES['auroc']:.3f} AUPRC={BABAR_BASELINES['auprc']:.3f}")
        print(f"  vs AudioLLM: F1={AUDIO_LLM_BASELINES['f1']:.3f} "
              f"AUROC={AUDIO_LLM_BASELINES['auroc']:.3f} AUPRC={AUDIO_LLM_BASELINES['auprc']:.3f}")
        with open(out_dir / "test_metrics_tuned.json", "w") as f:
            json.dump(metrics, f, indent=2)

        # Per-timepoint metrics
        tp_rows = []
        for tp_val, grp in pred_df.groupby("timepoint_norm"):
            vmask = grp["label"].notna()
            if vmask.sum() < 2:
                continue
            tm = _compute_metrics(grp.loc[vmask, "label"].values,
                                  grp.loc[vmask, "score"].values, threshold)
            tm["timepoint_norm"] = tp_val
            tp_rows.append(tm)
        if tp_rows:
            pd.DataFrame(tp_rows).to_csv(
                out_dir / "test_metrics_by_timepoint.csv", index=False)

    # Config
    cfg = {
        "model": args.model_name,
        "model_slug": args.model_slug,
        "split": args.split,
        "scoring": "gap_ratio = 1 - (word_covered_seconds / clip_duration)",
        "batch_size": args.batch_size,
        "seed": args.seed,
        "splits_dir": args.splits_dir,
        "cache_dir": str(cache_dir),
        "output_dir": str(out_dir),
        "n_clips": len(pred_df),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    with open(out_dir / "config.json", "w") as f:
        json.dump(cfg, f, indent=2)

    print(f"\nOutputs: {out_dir}")


if __name__ == "__main__":
    main()
