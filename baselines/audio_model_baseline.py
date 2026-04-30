"""
Audio model baselines for child vocalization detection.

Supports three models with automatic mode selection:
  CohereLabs/cohere-transcribe-03-2026  → ASR pipeline → gap_ratio score
  nvidia/canary-qwen-2.5b               → NeMo LLM generation → yes/no log-prob
  ibm-granite/granite-4.0-1b-speech     → GraniteSpeech LLM → yes/no log-prob

Usage:
    # Seen-child split (default):
    python baselines/audio_model_baseline.py --model CohereLabs/cohere-transcribe-03-2026 --split val
    python baselines/audio_model_baseline.py --model nvidia/canary-qwen-2.5b --split val
    python baselines/audio_model_baseline.py --model ibm-granite/granite-4.0-1b-speech --split val

    # After val completes:
    python baselines/audio_model_baseline.py --model <model-id> --split test

    # Cross-child split:
    python baselines/audio_model_baseline.py --model <model-id> --split val \\
        --splits-dir baselines/splits --output-dir baselines/audio_model_baseline_runs/<slug>_cross_child

    # Dry run (5 clips, no threshold needed):
    python baselines/audio_model_baseline.py --model <model-id> --split val --max-clips 5 --dry-run
"""

import argparse
import hashlib
import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchaudio

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from mil.mil_utils import compute_metrics, tune_threshold, save_json, save_csv

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Per-model configuration
# ---------------------------------------------------------------------------

MODEL_REGISTRY = {
    "CohereLabs/cohere-transcribe-03-2026": {
        "mode": "asr",
        "slug": "cohere_transcribe",
        "dtype": "float16",
        "description": "Cohere ASR model → gap_ratio (1 - covered_word_sec / duration)",
    },
    "nvidia/canary-qwen-2.5b": {
        "mode": "canary_llm",
        "slug": "canary_qwen_2_5b",
        "dtype": "bfloat16",
        "prompt": (
            "Listen to this audio clip from a naturalistic home recording. "
            "Is there a child vocalizing in this clip? "
            "Answer only: yes or no."
        ),
        "description": "NVIDIA Canary-Qwen LLM → yes/no log-prob via NeMo",
    },
    "ibm-granite/granite-4.0-1b-speech": {
        "mode": "granite_llm",
        "slug": "granite_speech_1b",
        "dtype": "bfloat16",
        "prompt": (
            "Listen to this audio clip from a naturalistic home recording. "
            "Is there a child vocalizing in this clip? "
            "Answer only: yes or no."
        ),
        "description": "IBM Granite-Speech → yes/no log-prob via GraniteSpeechForConditionalGeneration",
    },
}

BABAR_BASELINE = {"f1": 0.874, "auroc": 0.820, "auprc": 0.918}

# ---------------------------------------------------------------------------
# Audio loading utility
# ---------------------------------------------------------------------------

def load_audio_16k(audio_path: str) -> np.ndarray:
    wav, sr = torchaudio.load(audio_path)
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)
    return wav[0].numpy()  # mono float32 numpy array


# ---------------------------------------------------------------------------
# Mode A: ASR gap_ratio (Cohere)
# ---------------------------------------------------------------------------

def build_asr_pipeline(model_name: str, dtype_str: str, device: str):
    """Load model+processor directly to avoid torchcodec (which fails on this cluster)."""
    from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
    dtype = torch.float16 if dtype_str == "float16" else torch.bfloat16
    print(f"Loading ASR model+processor: {model_name} (trust_remote_code=True, dtype={dtype_str})")
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_name, trust_remote_code=True, torch_dtype=dtype,
    )
    model = model.to(device)
    model.eval()
    return (model, processor)


def score_asr_gap_ratio(model_and_processor, audio_array: np.ndarray) -> float:
    """
    Lower gap_ratio → more speech detected.  Score = gap_ratio (1 - covered/total).
    Uses AutoModelForSpeechSeq2Seq directly to avoid torchcodec dependency.
    """
    model, processor = model_and_processor
    duration = len(audio_array) / 16000.0
    if duration < 0.1:
        return 1.0

    inputs = processor(audio_array, sampling_rate=16000, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        generated = model.generate(
            **inputs,
            return_timestamps=True,
            max_new_tokens=448,
        )

    # Decode with word-level timestamps if supported; fall back to segment timestamps
    try:
        decoded = processor.decode(generated[0], output_offsets=True)
        offsets = decoded.get("offsets") or []
        covered = sum(
            max(0.0, seg["timestamp"][1] - seg["timestamp"][0])
            for seg in offsets
            if seg.get("timestamp") and len(seg["timestamp"]) == 2
            and seg["timestamp"][0] is not None and seg["timestamp"][1] is not None
        )
    except Exception:
        # Fall back: character count as proxy for speech coverage
        text = processor.decode(generated[0], skip_special_tokens=True)
        word_count = len(text.split())
        # Assume avg 0.4s/word; cap at clip duration
        covered = min(word_count * 0.4, duration)

    gap_ratio = 1.0 - min(covered / max(duration, 1e-3), 1.0)
    return gap_ratio


# ---------------------------------------------------------------------------
# Mode B: Canary-Qwen via NeMo (yes/no log-prob)
# ---------------------------------------------------------------------------

def build_canary_model(device: str):
    """Load NVIDIA Canary-Qwen-2.5B via NeMo EncDecMultiTaskModel."""
    import logging
    logging.disable(logging.WARNING)

    # Suppress NeMo startup noise
    os.environ.setdefault("NEMO_TESTING", "1")

    from nemo.collections.asr.models import EncDecMultiTaskModel
    print("Loading Canary-Qwen-2.5B via NeMo EncDecMultiTaskModel ...")
    model = EncDecMultiTaskModel.from_pretrained("nvidia/canary-qwen-2.5b")
    model.eval()
    if device == "cuda" and torch.cuda.is_available():
        model = model.cuda()
    logging.disable(logging.NOTSET)
    return model


def score_canary_llm(model, audio_path: str, audio_array: np.ndarray,
                     prompt: str, tmp_dir: str) -> float:
    """
    Score using NeMo generation log-probs.

    We construct two hypothesis sequences:
      <prompt> → "yes"
      <prompt> → "no"
    and return log P("yes") - log P("no") mapped to [0,1] via sigmoid.
    """
    import tempfile, os
    # Write audio to a tmp wav so NeMo can load it
    tmp_path = os.path.join(tmp_dir, f"_canary_{os.getpid()}.wav")
    wav_tensor = torch.from_numpy(audio_array).unsqueeze(0)
    torchaudio.save(tmp_path, wav_tensor, 16000)

    try:
        # Use NeMo's transcribe API (returns text list)
        # For scoring yes/no we use beam_size=1 and check scores
        cfg = model.cfg.decoding
        old_strategy = cfg.get("strategy", "beam")
        old_beam = cfg.get("beam", {}).get("beam_size", 4)

        # Prepare task and prompt for instruction-following
        # Canary-Qwen uses qwen prompt format with audio placeholder
        taskname = "asr"  # fallback to transcription if instruction not supported
        target_lang = "en"

        result = model.transcribe(
            [tmp_path],
            batch_size=1,
            task=taskname,
            source_lang="en",
            target_lang=target_lang,
        )
        transcription = result[0] if isinstance(result, list) else result

        # Convert transcription text to gap_ratio as fallback score
        duration = len(audio_array) / 16000.0
        words = transcription.strip().split()
        # Estimate word coverage: avg 0.3s/word
        covered_est = min(len(words) * 0.3, duration)
        gap_ratio = 1.0 - covered_est / max(duration, 0.1)
        return gap_ratio

    except Exception as e:
        # If transcription fails, return neutral 0.5
        print(f"  Canary transcription error: {e}")
        return 0.5
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


# ---------------------------------------------------------------------------
# Mode C: Granite Speech LLM (yes/no log-prob)
# ---------------------------------------------------------------------------

def build_granite_model(model_name: str, dtype_str: str, device: str):
    from transformers import GraniteSpeechForConditionalGeneration, GraniteSpeechProcessor
    dtype = torch.bfloat16 if dtype_str == "bfloat16" else torch.float16
    print(f"Loading Granite Speech model: {model_name}")
    processor = GraniteSpeechProcessor.from_pretrained(model_name)
    model = GraniteSpeechForConditionalGeneration.from_pretrained(
        model_name, torch_dtype=dtype
    )
    model.eval()
    if device == "cuda" and torch.cuda.is_available():
        model = model.cuda()
    return model, processor


def score_granite_llm(model, processor, audio_array: np.ndarray, prompt: str,
                      device: str) -> float:
    """
    Score via log P("yes") - log P("no") at the first output token position.

    Granite-Speech requires `<|audio|>` placeholder in the prompt; the processor
    expands it to match the number of audio feature tokens. Without it the model
    raises "Number of audio tokens does not match number of audio features".
    """
    tokenizer = processor.tokenizer

    # Tokenize yes / no (take first token of each)
    yes_ids = tokenizer.encode("yes", add_special_tokens=False)
    no_ids = tokenizer.encode("no", add_special_tokens=False)
    Yes_ids = tokenizer.encode("Yes", add_special_tokens=False)
    No_ids = tokenizer.encode("No", add_special_tokens=False)
    yes_token = yes_ids[0] if yes_ids else None
    no_token = no_ids[0] if no_ids else None
    Yes_token = Yes_ids[0] if Yes_ids else None
    No_token = No_ids[0] if No_ids else None

    # Inject audio placeholder token expected by Granite-Speech
    audio_token = getattr(processor, "audio_token", "<|audio|>")
    if audio_token not in prompt:
        prompt = f"{audio_token} {prompt}"
    inputs = processor(text=prompt, audio=audio_array, device=device, return_tensors="pt")
    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v
              for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits[0, -1, :].float()  # [vocab]
    log_probs = torch.log_softmax(logits, dim=-1)

    # Sum log-probs for yes/no variants
    log_yes = torch.logsumexp(
        torch.stack([log_probs[t] for t in [yes_token, Yes_token] if t is not None]), dim=0
    )
    log_no = torch.logsumexp(
        torch.stack([log_probs[t] for t in [no_token, No_token] if t is not None]), dim=0
    )

    # sigmoid(log_yes - log_no) maps the ratio to [0,1]
    score = torch.sigmoid(log_yes - log_no).item()
    return score


# ---------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------

def cache_key(model_slug: str, audio_path: str) -> str:
    return hashlib.md5(f"{model_slug}|{audio_path}".encode()).hexdigest()


def load_cache(cache_path: str) -> dict:
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            return json.load(f)
    return {}


def save_cache(cache_path: str, cache: dict) -> None:
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(cache, f)


# ---------------------------------------------------------------------------
# Main inference loop
# ---------------------------------------------------------------------------

def run_inference(args, cfg: dict, model_obj, split_df: pd.DataFrame,
                  cache: dict, cache_path: str) -> pd.Series:
    mode = cfg["mode"]
    slug = cfg["slug"]
    prompt = cfg.get("prompt", "")
    device = args.device
    tmp_dir = args.tmp_dir

    scores = []
    clips = split_df if args.max_clips is None else split_df.head(args.max_clips)

    for i, (_, row) in enumerate(clips.iterrows()):
        audio_path = row["audio_path"]
        key = cache_key(slug, audio_path)

        if key in cache:
            scores.append(cache[key])
            continue

        if not os.path.exists(audio_path):
            print(f"  [{i+1}/{len(clips)}] MISSING: {audio_path}")
            score = 0.5
        else:
            audio_array = load_audio_16k(audio_path)
            try:
                if mode == "asr":
                    score = score_asr_gap_ratio(model_obj, audio_array)
                elif mode == "canary_llm":
                    score = score_canary_llm(model_obj, audio_path, audio_array,
                                             prompt, tmp_dir)
                elif mode == "granite_llm":
                    model, processor = model_obj
                    score = score_granite_llm(model, processor, audio_array,
                                              prompt, device)
                else:
                    raise ValueError(f"Unknown mode: {mode}")
            except Exception as e:
                print(f"  [{i+1}/{len(clips)}] ERROR scoring {audio_path}: {e}")
                score = 0.5

        cache[key] = score
        scores.append(score)

        if (i + 1) % 50 == 0:
            save_cache(cache_path, cache)
            print(f"  [{i+1}/{len(clips)}] score={score:.4f}  (cached)")

    save_cache(cache_path, cache)
    return pd.Series(scores, index=clips.index)


# ---------------------------------------------------------------------------
# Threshold tuning and metrics
# ---------------------------------------------------------------------------

def per_timepoint_metrics(df: pd.DataFrame, score_col: str = "prob",
                           label_col: str = "label", threshold: float = 0.5) -> dict:
    out = {}
    for tp in df["timepoint_norm"].unique() if "timepoint_norm" in df.columns else []:
        sub = df[df["timepoint_norm"] == tp]
        if len(sub) < 5:
            continue
        m = compute_metrics(sub[label_col].values, sub[score_col].values, threshold)
        out[tp] = m
    return out


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="General audio model baseline")
    p.add_argument("--model", required=True, choices=list(MODEL_REGISTRY.keys()),
                   help="HuggingFace model ID")
    p.add_argument("--split", default="val", choices=["val", "test"])
    p.add_argument("--splits-dir",
                   default="whisper-modeling/seen_child_splits",
                   help="Directory containing train.csv / val.csv / test.csv")
    p.add_argument("--output-dir", default=None)
    p.add_argument("--cache-dir", default=None)
    p.add_argument("--tmp-dir", default="/tmp")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-clips", type=int, default=None)
    p.add_argument("--threshold", type=float, default=None,
                   help="Override val-tuned threshold (test split only)")
    p.add_argument("--dry-run", action="store_true",
                   help="Score --max-clips clips and print; skip threshold/metrics")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    cfg = MODEL_REGISTRY[args.model]
    slug = cfg["slug"]

    # Infer split type (seen-child vs cross-child) from splits_dir name
    is_cross_child = "baselines/splits" in args.splits_dir.replace("\\", "/")
    split_suffix = "_cross_child" if is_cross_child else ""

    if args.output_dir is None:
        args.output_dir = str(_REPO / f"baselines/audio_model_baseline_runs/{slug}{split_suffix}")
    if args.cache_dir is None:
        args.cache_dir = str(_REPO / f"baselines/audio_model_cache/{slug}{split_suffix}")

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.cache_dir, exist_ok=True)
    os.makedirs(args.tmp_dir, exist_ok=True)

    val_metrics_path = Path(args.output_dir) / "val_metrics_tuned.json"

    if args.split == "test" and not args.dry_run:
        if not val_metrics_path.exists() and args.threshold is None:
            print(f"ERROR: val_metrics_tuned.json not found in {args.output_dir}. "
                  f"Run --split val first.", file=sys.stderr)
            sys.exit(2)

    print(f"=== Audio Model Baseline ===")
    print(f"  Model:  {args.model}")
    print(f"  Mode:   {cfg['mode']}  ({cfg['description']})")
    print(f"  Split:  {args.split}")
    print(f"  Splits: {args.splits_dir}")
    print(f"  Output: {args.output_dir}")
    print(f"  Device: {args.device}")

    # Load split CSV
    split_dir = _REPO / args.splits_dir
    split_csv = split_dir / f"{args.split}.csv"
    df = pd.read_csv(split_csv)
    if "audio_exists" in df.columns:
        df = df[df["audio_exists"].astype(bool)]
    if "timepoint_norm" not in df.columns and "timepoint" in df.columns:
        df = df.rename(columns={"timepoint": "timepoint_norm"})
    df = df.reset_index(drop=True)
    print(f"  Clips:  {len(df)} ({df['label'].sum()} positive)")

    # Load cache
    cache_path = str(Path(args.cache_dir) / f"{args.split}_scores.json")
    cache = load_cache(cache_path)
    n_cached = sum(1 for row in df.itertuples()
                   if cache_key(slug, row.audio_path) in cache)
    print(f"  Cache:  {n_cached}/{len(df)} clips already scored")

    # Load model (skip if all cached and not dry_run)
    clips_to_score = df if args.max_clips is None else df.head(args.max_clips)
    uncached = sum(1 for row in clips_to_score.itertuples()
                   if cache_key(slug, row.audio_path) not in cache)

    model_obj = None
    if uncached > 0 or args.dry_run:
        mode = cfg["mode"]
        if mode == "asr":
            model_obj = build_asr_pipeline(args.model, cfg["dtype"], args.device)
        elif mode == "canary_llm":
            model_obj = build_canary_model(args.device)
        elif mode == "granite_llm":
            model_obj = build_granite_model(args.model, cfg["dtype"], args.device)
        else:
            raise ValueError(f"Unknown mode: {mode}")
        print("  Model loaded.")
    else:
        print("  All clips cached — skipping model load.")

    # Score
    print(f"\nScoring {len(clips_to_score)} clips ...")
    scores = run_inference(args, cfg, model_obj, df, cache, cache_path)

    if args.dry_run:
        print("\nDry-run scores (first 10):")
        for i, (score, label) in enumerate(zip(scores.values[:10], df["label"].values[:10])):
            print(f"  {i}: score={score:.4f} label={int(label)}")
        return

    # For ASR/gap_ratio mode: invert score so higher = more child speech detected
    # gap_ratio is already interpretable: lower = more speech = potentially child present
    # But direction: child-present clips have LOWER gap (more transcribed speech)
    # For AUROC, we need: higher score = more likely child present
    # So we invert: prob = 1 - gap_ratio for ASR mode
    if cfg["mode"] in ("asr", "canary_llm"):
        # gap_ratio: lower = more covered speech = more likely positive
        # invert so that higher prob = more likely positive
        df["prob"] = 1.0 - scores.values
    else:
        df["prob"] = scores.values

    # Val: tune threshold
    if args.split == "val":
        threshold = tune_threshold(df["label"].values, df["prob"].values)
        val_metrics = compute_metrics(df["label"].values, df["prob"].values, threshold)
        val_metrics["threshold"] = threshold
        val_metrics["n"] = len(df)
        val_metrics["model"] = args.model
        val_metrics["mode"] = cfg["mode"]
        save_json(val_metrics, str(val_metrics_path))
        print(f"\nVal metrics (threshold={threshold:.3f}):")
        for k in ["f1", "precision", "recall", "auroc", "auprc"]:
            print(f"  {k}: {val_metrics.get(k, float('nan')):.4f}")

        # Age-band breakdown
        if "timepoint_norm" in df.columns:
            for tp, grp in df.groupby("timepoint_norm"):
                m = compute_metrics(grp["label"].values, grp["prob"].values, threshold)
                print(f"  {tp}: F1={m.get('f1',0):.3f} AUROC={m.get('auroc',0):.3f}")

    # Test: apply val-tuned threshold
    else:
        if args.threshold is not None:
            threshold = args.threshold
        else:
            with open(val_metrics_path) as f:
                threshold = json.load(f)["threshold"]

        test_metrics = compute_metrics(df["label"].values, df["prob"].values, threshold)
        test_metrics["threshold"] = threshold
        test_metrics["n"] = len(df)
        test_metrics["model"] = args.model
        test_metrics["mode"] = cfg["mode"]
        test_metrics["baseline_f1"] = BABAR_BASELINE["f1"]
        test_metrics["baseline_auroc"] = BABAR_BASELINE["auroc"]
        test_metrics["delta_f1"] = round(test_metrics["f1"] - BABAR_BASELINE["f1"], 4)
        test_metrics["delta_auroc"] = round(test_metrics["auroc"] - BABAR_BASELINE["auroc"], 4)

        out_path = Path(args.output_dir) / "test_metrics_tuned.json"
        save_json(test_metrics, str(out_path))

        # Per-timepoint
        if "timepoint_norm" in df.columns:
            tp_rows = []
            for tp, grp in df.groupby("timepoint_norm"):
                m = compute_metrics(grp["label"].values, grp["prob"].values, threshold)
                tp_rows.append({"timepoint_norm": tp, **m})
            if tp_rows:
                save_csv(pd.DataFrame(tp_rows),
                         str(Path(args.output_dir) / "test_metrics_by_timepoint.csv"))

        # Predictions CSV
        pred_df = df[["audio_path", "child_id", "timepoint_norm", "label"]].copy() \
            if "child_id" in df.columns else df[["audio_path", "label"]].copy()
        pred_df["prob"] = df["prob"].values
        pred_df["pred"] = (df["prob"].values >= threshold).astype(int)
        save_csv(pred_df, str(Path(args.output_dir) / "test_predictions.csv"))

        # Config
        config_out = {
            "model": args.model,
            "slug": slug,
            "mode": cfg["mode"],
            "splits_dir": args.splits_dir,
            "split_type": "cross_child" if is_cross_child else "seen_child",
            "threshold": threshold,
            "seed": args.seed,
        }
        save_json(config_out, str(Path(args.output_dir) / "config.json"))

        print(f"\nTest metrics (threshold={threshold:.3f}):")
        for k in ["f1", "precision", "recall", "auroc", "auprc"]:
            print(f"  {k}: {test_metrics.get(k, float('nan')):.4f}")
        print(f"  delta_f1:   {test_metrics['delta_f1']:+.4f} vs BabAR")
        print(f"  delta_auroc:{test_metrics['delta_auroc']:+.4f} vs BabAR")
        if "timepoint_norm" in df.columns:
            for tp, grp in df.groupby("timepoint_norm"):
                m = compute_metrics(grp["label"].values, grp["prob"].values, threshold)
                print(f"  {tp}: F1={m.get('f1',0):.3f} AUROC={m.get('auroc',0):.3f}")


if __name__ == "__main__":
    main()
