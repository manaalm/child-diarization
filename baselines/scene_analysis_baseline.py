"""Audio-scene-analysis baselines for child vocalisation detection (spec 022 US3).

Supports two backbones:
  --model ast    : MIT/ast-finetuned-audioset-10-10-0.4593, HuggingFace transformers,
                   runs in-process (PyTorch).
  --model yamnet : google/yamnet on TFHub. Runs in a sibling tensorflow env
                   (yamnet-eval/.venv) via subprocess bridge to encoders/yamnet_worker.py.

Both produce a per-clip child-vocalisation probability:
  p_child_voc = max(P[Child speech], P[Babbling], P[Baby cry], P[Children shouting])

Auxiliary class probabilities are written as separate columns for posthoc
analysis. AudioSet ontology IDs are documented in baselines/scene_analysis_runs/<model>/README.md.

Splits:
  --split val          -> whisper-modeling/seen_child_splits/val.csv
  --split test         -> whisper-modeling/seen_child_splits/test.csv
  --split test_all     -> whisper-modeling/all_children_splits/test_all.csv (US3 FR-014)

Threshold tuning happens on `val`; `test` and `test_all` reuse the val-tuned
threshold (Constitution IV — no test-set tuning).
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torchaudio

REPO_ROOT = "/orcd/scratch/orcd/008/manaal/child-adult-diarization"
SAMPLE_RATE = 16000

# AudioSet child-vocalisation ontology IDs.
# Documented in baselines/scene_analysis_runs/<model>/README.md.
CHILD_LABEL_NAMES = [
    "Child speech, kid speaking",  # /m/02zsn
    "Babbling",                    # /m/0463cq4
    "Baby cry, infant cry",        # /t/dd00002
    "Children shouting",           # /m/02p0sh1
]

SPLIT_TO_PATH = {
    "val":      os.path.join(REPO_ROOT, "whisper-modeling", "seen_child_splits", "val.csv"),
    "test":     os.path.join(REPO_ROOT, "whisper-modeling", "seen_child_splits", "test.csv"),
    "test_all": os.path.join(REPO_ROOT, "whisper-modeling", "all_children_splits", "test_all.csv"),
}

sys.path.insert(0, os.path.join(REPO_ROOT, "mil"))
from mil_utils import compute_metrics, tune_threshold  # noqa: E402


# ---------------------------------------------------------------------------
# AST (in-process)
# ---------------------------------------------------------------------------

def _load_audio(path: str) -> np.ndarray:
    wav, sr = torchaudio.load(path)
    if wav.dim() > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != SAMPLE_RATE:
        wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE)
    return wav.squeeze(0).numpy()


def _ast_score_clips(audio_paths: List[str], model_name: str, device: str) -> pd.DataFrame:
    """Return per-clip per-class probabilities. The model emits multi-label
    sigmoid scores over the 527-class AudioSet ontology."""
    from transformers import AutoFeatureExtractor, ASTForAudioClassification

    print(f"Loading AST: {model_name}", flush=True)
    fe = AutoFeatureExtractor.from_pretrained(model_name)
    model = ASTForAudioClassification.from_pretrained(model_name).to(device).eval()

    id2label = model.config.id2label  # dict[int, str]
    label2id = {v: k for k, v in id2label.items()}
    target_ids: Dict[str, int] = {}
    for name in CHILD_LABEL_NAMES:
        if name in label2id:
            target_ids[name] = label2id[name]
        else:
            print(f"  [warn] AudioSet label {name!r} not in AST id2label; using NaN col", flush=True)

    rows = []
    n = len(audio_paths)
    for i, p in enumerate(audio_paths):
        if i % 50 == 0:
            print(f"  AST {i+1}/{n}: {Path(p).name}", flush=True)
        try:
            wav = _load_audio(p)
            inputs = fe(wav, sampling_rate=SAMPLE_RATE, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                logits = model(**inputs).logits.squeeze(0)
            probs = torch.sigmoid(logits).cpu().numpy()
            row = {"audio_path": p}
            for name in CHILD_LABEL_NAMES:
                col = f"p_{name.split(',')[0].lower().replace(' ', '_')}"
                row[col] = float(probs[target_ids[name]]) if name in target_ids else float("nan")
            row["p_child_voc"] = float(np.nanmax([row[c] for c in row if c.startswith("p_")]))
            rows.append(row)
        except Exception as e:
            print(f"  [error] {p}: {e}", flush=True)
            rows.append({
                "audio_path": p,
                **{f"p_{n.split(',')[0].lower().replace(' ', '_')}": float("nan") for n in CHILD_LABEL_NAMES},
                "p_child_voc": float("nan"),
            })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# YAMNet (subprocess bridge)
# ---------------------------------------------------------------------------

def _yamnet_score_clips(audio_paths: List[str], yamnet_env_python: str) -> pd.DataFrame:
    """Bridge to encoders/yamnet_worker.py inside the yamnet-eval sibling env."""
    worker = os.path.join(REPO_ROOT, "encoders", "yamnet_worker.py")
    if not os.path.exists(yamnet_env_python):
        raise SystemExit(
            f"yamnet-eval env python not found at {yamnet_env_python}. "
            "Set up via: uv venv yamnet-eval/.venv && "
            "source yamnet-eval/.venv/bin/activate && "
            "uv pip install tensorflow==2.16 tensorflow-hub==0.16 soundfile==0.12"
        )

    # Pipe audio paths to the worker via stdin (one path per line)
    payload = "\n".join(audio_paths)
    print(f"Bridging to YAMNet worker at {worker} ({len(audio_paths)} clips)", flush=True)
    res = subprocess.run(
        [yamnet_env_python, worker],
        input=payload, text=True, capture_output=True,
        env={**os.environ, "TF_CPP_MIN_LOG_LEVEL": "2"},
    )
    if res.returncode != 0:
        raise SystemExit(f"yamnet_worker failed: {res.stderr}")

    # Worker emits CSV to stdout
    from io import StringIO
    df = pd.read_csv(StringIO(res.stdout))
    return df


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def _detect_audio_column(df: pd.DataFrame) -> str:
    for cand in ("audio_path", "wav_path"):
        if cand in df.columns:
            return cand
    raise SystemExit(f"split CSV missing audio_path column; got {list(df.columns)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", choices=["ast", "yamnet"], required=True)
    ap.add_argument("--split", choices=["val", "test", "test_all"], required=True)
    ap.add_argument("--out-dir", default=None,
                    help="default: baselines/scene_analysis_runs/<model>/")
    ap.add_argument("--ast-model-name", default="MIT/ast-finetuned-audioset-10-10-0.4593")
    ap.add_argument("--yamnet-env-python",
                    default=os.path.join(REPO_ROOT, "yamnet-eval", ".venv", "bin", "python"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--max-clips", type=int, default=None,
                    help="cap clips for smoke tests")
    args = ap.parse_args()

    out_dir = args.out_dir or os.path.join(REPO_ROOT, "baselines", "scene_analysis_runs", args.model)
    os.makedirs(out_dir, exist_ok=True)

    split_csv = SPLIT_TO_PATH[args.split]
    df = pd.read_csv(split_csv)
    if "label" not in df.columns:
        raise SystemExit(f"split CSV missing `label` column: {split_csv}")

    audio_col = _detect_audio_column(df)
    if args.max_clips:
        df = df.head(args.max_clips).copy()
    print(f"loaded {len(df)} clips from {split_csv}", flush=True)

    # Score clips
    if args.model == "ast":
        scores_df = _ast_score_clips(df[audio_col].tolist(), args.ast_model_name, args.device)
    else:
        scores_df = _yamnet_score_clips(df[audio_col].tolist(), args.yamnet_env_python)

    # Join scores back to the metadata
    merged = df.merge(scores_df, on=audio_col, how="left")
    if "child_id" not in merged.columns and "ID" in merged.columns:
        merged["child_id"] = merged["ID"]

    # Threshold: tune on val; reuse for test/test_all
    val_metrics_path = os.path.join(out_dir, "val_metrics_tuned.json")
    if args.split == "val":
        y_true = merged["label"].astype(int).tolist()
        y_score = merged["p_child_voc"].astype(float).fillna(0.5).tolist()
        threshold = tune_threshold(y_true, y_score)
        m = compute_metrics(y_true, y_score, threshold=threshold)
        m["threshold"] = threshold
        merged["prob"] = merged["p_child_voc"]
        merged["prediction"] = (merged["p_child_voc"].fillna(0.5) >= threshold).astype(int)
    else:
        if not os.path.exists(val_metrics_path):
            sys.exit(2)  # spec contract: exit 2 if val tuning missing
        threshold = float(json.load(open(val_metrics_path))["threshold"])
        y_true = merged["label"].astype(int).tolist()
        y_score = merged["p_child_voc"].astype(float).fillna(0.5).tolist()
        m = compute_metrics(y_true, y_score, threshold=threshold)
        m["threshold"] = threshold
        merged["prob"] = merged["p_child_voc"]
        merged["prediction"] = (merged["p_child_voc"].fillna(0.5) >= threshold).astype(int)

    # Write predictions
    keep_cols = [audio_col, "child_id", "timepoint_norm", "label", "prob", "prediction",
                 "p_child_voc"] + [c for c in scores_df.columns if c.startswith("p_") and c != "p_child_voc"]
    keep_cols = [c for c in keep_cols if c in merged.columns]
    preds_path = os.path.join(out_dir, f"{args.split}_predictions.csv")
    merged[keep_cols].to_csv(preds_path, index=False)

    metrics_path = os.path.join(out_dir, f"{args.split}_metrics_tuned.json")
    with open(metrics_path, "w") as f:
        json.dump({**m, "n": int(len(merged)),
                   "model": args.model,
                   "split": args.split,
                   "split_csv": split_csv,
                   "regenerated_at": datetime.now(timezone.utc).isoformat(timespec="seconds")},
                  f, indent=2)

    config_path = os.path.join(out_dir, "config.json")
    if not os.path.exists(config_path):
        with open(config_path, "w") as f:
            json.dump({
                "model": args.model,
                "ast_model_name": args.ast_model_name if args.model == "ast" else None,
                "yamnet_env_python": args.yamnet_env_python if args.model == "yamnet" else None,
                "audioset_labels": CHILD_LABEL_NAMES,
                "aggregation": "p_child_voc = max(P[label]) across child-vocalisation labels",
            }, f, indent=2)

    print(f"\n=== {args.model} on {args.split} ===")
    print(json.dumps(m, indent=2))
    print(f"wrote {preds_path}")
    print(f"wrote {metrics_path}")


if __name__ == "__main__":
    main()
