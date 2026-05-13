"""
Zero-shot (and optional few-shot) child vocalization detection using Qwen2.5-Omni-7B.

Default model: Qwen/Qwen2.5-Omni-7B (Apache-2.0). The thinker (text-output)
component is loaded; the talker (speech-synthesis) component is skipped to
save GPU memory — only yes/no logits are needed.

Usage:
    python baselines/audio_llm_baseline.py --split val [OPTIONS]
    python baselines/audio_llm_baseline.py --split test [OPTIONS]  # val must complete first
    python baselines/audio_llm_baseline.py --split val --max-clips 10 --dry-run
"""

import argparse
import hashlib
import json
import os
import re
import sys
import tempfile
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchaudio

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

PROMPT_TEMPLATES = {
    "zero_shot_v1": (
        "Listen to this audio clip from a naturalistic social interaction recording. "
        "Is there a child vocalizing in this clip? "
        "Answer only: yes or no."
    ),
    # Target-child framing — only meaningful with few-shot demos from the same
    # child, where positive demos anchor what the target child sounds like and
    # negative demos anchor "any speech that is not the target child."
    "target_child_v1": (
        "All audio clips in this conversation are from a single target child's "
        "recording. The earlier clips with answers are demonstrations of when "
        "that target child is or is not vocalizing. "
        "Is the target child vocalizing in this clip? "
        "Answer only: yes or no."
    ),
}

# Qwen2.5-Omni's chat template requires a system prompt — the model card warns
# that omitting it can break audio understanding. Keep it minimal so it does
# not bias yes/no logits.
SYSTEM_PROMPT = (
    "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, "
    "capable of perceiving auditory and visual inputs, as well as generating "
    "text and speech."
)

BABAR_BASELINES = {"f1": 0.874, "auroc": 0.820, "auprc": 0.918}

# ---------------------------------------------------------------------------
# T002 — Cache helpers
# ---------------------------------------------------------------------------

def _cache_path(audio_path: str, model_slug: str, cache_dir: str) -> str:
    stem = Path(audio_path).stem
    md5 = hashlib.md5(audio_path.encode()).hexdigest()[:12]
    return str(Path(cache_dir) / f"{stem}__{md5}.json")


def _prompt_hash(prompt_text: str) -> str:
    """8-char SHA1 of the rendered prompt text. Used to invalidate stale caches
    when the prompt template changes."""
    return hashlib.sha1(prompt_text.encode()).hexdigest()[:8]


_PROMPT_HASH_WARNED = False


def _load_cache(path: str, expected_prompt_hash: str | None = None):
    """Load a cache entry. If expected_prompt_hash is given, only return the
    entry when the cached `prompt_text_hash` matches. Legacy entries (no hash
    field) are accepted with a one-time warning per run for backwards
    compatibility — pass `--invalidate-legacy-cache` to enforce strictly."""
    global _PROMPT_HASH_WARNED
    try:
        with open(path) as f:
            entry = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None
    if expected_prompt_hash is None:
        return entry
    cached_hash = entry.get("prompt_text_hash")
    if cached_hash is None:
        if not _PROMPT_HASH_WARNED:
            print(f"  [warn] cache entries lack 'prompt_text_hash'; accepting "
                  f"legacy caches but cannot verify prompt match. New entries "
                  f"will record hash={expected_prompt_hash}.", flush=True)
            _PROMPT_HASH_WARNED = True
        return entry
    if cached_hash != expected_prompt_hash:
        # Stale cache — different prompt. Treat as miss; will be overwritten.
        return None
    return entry


def _save_cache(path: str, entry: dict):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(entry, f)
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# T003 — Audio loading
# ---------------------------------------------------------------------------

def _load_audio(audio_path: str):
    try:
        wav, sr = torchaudio.load(audio_path)
    except Exception as e:
        warnings.warn(f"Cannot load {audio_path}: {e}")
        return None, 0

    # Mono
    if wav.shape[0] > 1:
        wav = wav.mean(0, keepdim=True)

    # Resample to 16 kHz
    target_sr = 16_000
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
        sr = target_sr

    # Truncate at 30 s
    max_samples = 30 * sr
    if wav.shape[1] > max_samples:
        wav = wav[:, :max_samples]

    return wav.squeeze(0).numpy().astype(np.float32), sr


# ---------------------------------------------------------------------------
# T004 — Model loading
# ---------------------------------------------------------------------------

def _resolve_model_class(model_class_name, model_name: str):
    """Map a string class name (or None) to a transformers model class.

    spec-022 US3: support Qwen3-Omni (30B MoE) alongside Qwen2.5-Omni (7B).
    Class auto-detected from the model HF name when --model-class is omitted;
    new Qwen-Omni variants override via the flag.
    """
    import transformers

    if model_class_name:
        cls = getattr(transformers, model_class_name, None)
        if cls is None:
            raise SystemExit(
                f"--model-class={model_class_name!r} not in transformers "
                f"({transformers.__version__}). Likely options: "
                "Qwen2_5OmniThinkerForConditionalGeneration, "
                "Qwen3OmniMoeForConditionalGeneration, "
                "AutoModelForCausalLM."
            )
        return cls

    lname = model_name.lower()
    if "qwen2.5-omni" in lname or "qwen2_5_omni" in lname:
        return transformers.Qwen2_5OmniThinkerForConditionalGeneration
    if "qwen3-omni" in lname or "qwen3.5-omni" in lname:
        # Thinker variant matches the Qwen2.5-Omni thinker pattern (text-output);
        # the non-Thinker class is the full audio-in/speech-out multimodal model
        # whose forward() does not accept input_ids the way the thinker does.
        for cand in ("Qwen3OmniMoeThinkerForConditionalGeneration",
                     "Qwen3OmniThinkerForConditionalGeneration",
                     "Qwen3_5OmniThinkerForConditionalGeneration",
                     "Qwen3OmniMoeForConditionalGeneration",
                     "Qwen3_5OmniForConditionalGeneration"):
            cls = getattr(transformers, cand, None)
            if cls is not None:
                return cls
        raise SystemExit(
            f"No Qwen3-Omni class found in transformers {transformers.__version__}. "
            "Upgrade transformers or pass --model-class explicitly."
        )
    return transformers.AutoModelForCausalLM


def _load_model(model_name: str, dtype: str, quantize_4bit: bool, device: str,
                model_class_name: str | None = None):
    """Load an Omni audio LLM thinker (text-output) component.

    Qwen2.5-Omni: thinker = LLM + audio/vision encoders; talker (speech synthesis)
    skipped to save ~10 GB GPU memory.
    Qwen3-Omni / Qwen3.5-Omni: equivalent thinker auto-detected or via --model-class.
    """
    from transformers import AutoProcessor

    ModelCls = _resolve_model_class(model_class_name, model_name)
    torch_dtype = torch.bfloat16 if dtype == "bfloat16" else torch.float32

    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

    model_kwargs = dict(
        device_map="auto",
        dtype=torch_dtype,
        trust_remote_code=True,
    )

    if quantize_4bit:
        from transformers import BitsAndBytesConfig
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch_dtype,
            bnb_4bit_quant_type="nf4",
        )
        model_kwargs.pop("dtype", None)

    model = ModelCls.from_pretrained(model_name, **model_kwargs)
    model.eval()
    return processor, model


# ---------------------------------------------------------------------------
# T005 — Per-clip inference
# ---------------------------------------------------------------------------

def _build_prompt_text(prompt_template: str) -> str:
    if prompt_template not in PROMPT_TEMPLATES:
        raise ValueError(f"Unknown prompt template: {prompt_template!r}. "
                         f"Available: {list(PROMPT_TEMPLATES)}")
    return PROMPT_TEMPLATES[prompt_template]


def _get_yes_no_token_ids(tokenizer):
    """Return all token IDs for yes/no variants (case-insensitive)."""
    yes_ids = [tokenizer.encode(w, add_special_tokens=False)[-1]
               for w in ("yes", "Yes", "YES")]
    no_ids  = [tokenizer.encode(w, add_special_tokens=False)[-1]
               for w in ("no",  "No",  "NO")]
    return yes_ids, no_ids


def _infer_clip(processor, model, waveform_np, sr: int, prompt_text: str,
                device: str, few_shot_examples=None):
    # Qwen2.5-Omni requires a system prompt for audio understanding.
    conversation = [{
        "role": "system",
        "content": [{"type": "text", "text": SYSTEM_PROMPT}],
    }]

    # Few-shot preamble turns
    if few_shot_examples:
        for ex_wav, ex_label in few_shot_examples:
            conversation.append({
                "role": "user",
                "content": [
                    {"type": "audio", "audio": ex_wav},
                    {"type": "text", "text": prompt_text},
                ],
            })
            conversation.append({
                "role": "assistant",
                "content": [{"type": "text",
                             "text": "yes" if ex_label == 1 else "no"}],
            })

    # Query turn
    conversation.append({
        "role": "user",
        "content": [
            {"type": "audio", "audio": waveform_np},
            {"type": "text", "text": prompt_text},
        ],
    })

    text_input = processor.apply_chat_template(
        conversation, add_generation_prompt=True, tokenize=False
    )

    # Collect all audio arrays from the conversation
    audios = []
    for turn in conversation:
        if isinstance(turn["content"], list):
            for part in turn["content"]:
                if part.get("type") == "audio":
                    audios.append(part["audio"])

    inputs = processor(
        text=text_input,
        audio=audios,
        sampling_rate=sr,
        return_tensors="pt",
        padding=True,
    )

    # Move inputs to model device; cast floating-point feature tensors to the
    # model's dtype. spec-022 US3 / Qwen3-Omni: the processor emits float32
    # input_features but the model is loaded in bfloat16, so audio_tower's
    # conv2d would crash with "Input type (float) and bias type (BFloat16)
    # should be the same". Cast feature tensors (but NOT integer tensors like
    # input_ids / attention_mask) to model dtype.
    model_dtype = getattr(model, "dtype", None)
    if model_dtype is None:
        try:
            model_dtype = next(model.parameters()).dtype
        except StopIteration:
            model_dtype = torch.float32
    inputs_cast = {}
    for k, v in inputs.items():
        if hasattr(v, "to"):
            v = v.to(model.device)
            if v.is_floating_point() and v.dtype != model_dtype:
                v = v.to(model_dtype)
        inputs_cast[k] = v
    inputs = inputs_cast

    tok = processor.tokenizer
    yes_ids, no_ids = _get_yes_no_token_ids(tok)

    # Forward pass — extract last-token logits for next-token prediction.
    # Using model(**inputs) rather than generate() to get unmasked vocabulary
    # logits (generate() applies constrained decoding that masks most tokens).
    with torch.no_grad():
        out = model(**inputs)

    last_logits = out.logits[0, -1, :].float()  # (vocab_size,)
    log_probs = torch.log_softmax(last_logits, dim=-1)

    # Sum log-probs across all capitalization variants of yes / no
    yes_lp = float(torch.logsumexp(
        torch.stack([log_probs[i] for i in yes_ids]), dim=0
    ))
    no_lp = float(torch.logsumexp(
        torch.stack([log_probs[i] for i in no_ids]), dim=0
    ))

    prob = float(torch.softmax(torch.tensor([yes_lp, no_lp]), dim=0)[0])

    # Derive response_raw from logit argmax (avoids extra generate() call)
    response_raw = "yes" if yes_lp > no_lp else "no"
    parse_status = "parsed"

    return {
        "prob": prob,
        "response_raw": response_raw,
        "parse_status": parse_status,
        "logit_yes": yes_lp,
        "logit_no": no_lp,
    }


# ---------------------------------------------------------------------------
# T008 — Metrics helpers
# ---------------------------------------------------------------------------

def _compute_metrics(y_true, y_score, threshold: float) -> dict:
    from sklearn.metrics import (
        f1_score, precision_score, recall_score,
        roc_auc_score, average_precision_score,
    )
    y_pred = (np.array(y_score) >= threshold).astype(int)
    return {
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "auroc": float(roc_auc_score(y_true, y_score)),
        "auprc": float(average_precision_score(y_true, y_score)),
    }


def _tune_threshold(y_true, y_score) -> float:
    from sklearn.metrics import f1_score
    best_thr, best_f1 = 0.5, -1.0
    for thr in np.linspace(0.05, 0.95, 19):
        y_pred = (np.array(y_score) >= thr).astype(int)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1, best_thr = f1, float(thr)
    return best_thr


# ---------------------------------------------------------------------------
# T013 — Few-shot example selection
# ---------------------------------------------------------------------------

def _find_few_shot_examples(audio_path: str, train_csv_path: str,
                             n_shot: int, seed: int, universal: bool = False):
    rng = np.random.default_rng(seed)
    try:
        train_df = pd.read_csv(train_csv_path)
    except FileNotFoundError:
        warnings.warn(f"train CSV not found: {train_csv_path}; skipping few-shot.")
        return []

    if universal:
        # Universal-shots mode (e.g. synthetic demos): use all rows of the CSV
        # without per-query child filtering.
        child_rows = train_df
        child_id = "<universal>"
    else:
        match = re.search(r"sub-([A-Za-z0-9]+)", audio_path)
        if not match:
            warnings.warn(f"Could not parse child_id from {audio_path!r}; skipping few-shot.")
            return []
        child_id = match.group(0)  # e.g. "sub-A1H3H9Y3T1"

        # Match child — try child_id column or parse from audio_path column
        if "child_id" in train_df.columns:
            child_rows = train_df[train_df["child_id"] == child_id]
        elif "audio_path" in train_df.columns:
            child_rows = train_df[train_df["audio_path"].str.contains(child_id, na=False)]
        else:
            warnings.warn("train CSV has no child_id or audio_path column; skipping few-shot.")
            return []

    # Exclude the query clip
    if "audio_path" in child_rows.columns:
        child_rows = child_rows[child_rows["audio_path"] != audio_path]

    label_col = "label" if "label" in child_rows.columns else (
        "child_vocalizing" if "child_vocalizing" in child_rows.columns else None
    )
    if label_col is None:
        warnings.warn("Cannot find label column in train CSV; skipping few-shot.")
        return []

    pos = child_rows[child_rows[label_col] == 1]["audio_path"].tolist()
    neg = child_rows[child_rows[label_col] == 0]["audio_path"].tolist()

    n_each = n_shot // 2
    if len(pos) < n_each or len(neg) < n_each:
        warnings.warn(
            f"Child {child_id} has {len(pos)} pos / {len(neg)} neg training clips; "
            f"need {n_each} each for {n_shot}-shot. Falling back to 0-shot."
        )
        return []

    chosen_pos = rng.choice(pos, size=n_each, replace=False).tolist()
    chosen_neg = rng.choice(neg, size=n_each, replace=False).tolist()
    examples = [(p, 1) for p in chosen_pos] + [(n, 0) for n in chosen_neg]

    # Verify files exist
    examples = [(p, l) for p, l in examples if Path(p).exists()]
    if len(examples) < n_shot:
        warnings.warn(f"Some few-shot audio files missing for {child_id}; falling back to 0-shot.")
        return []

    return examples


# ---------------------------------------------------------------------------
# T006 — CLI and main
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(description="Audio LLM zero-shot child vocalization baseline")
    p.add_argument("--split", default="val", choices=["val", "test", "test_all"])
    p.add_argument("--split-csv", default=None,
                   help="Override split CSV path. Defaults: val/test -> seen_child_splits/{split}.csv; "
                        "test_all -> all_children_splits/test_all.csv (spec 022 US3 universal coverage)")
    p.add_argument("--train-csv", default="whisper-modeling/seen_child_splits/train.csv")
    p.add_argument("--model", default="Qwen/Qwen2.5-Omni-7B")
    p.add_argument("--model-class", default=None,
                   help="transformers class name. Auto-detected from --model when omitted. "
                        "Set explicitly for Qwen3-Omni: e.g., Qwen3OmniMoeForConditionalGeneration.")
    p.add_argument("--model-slug", default="qwen25_omni_7b")
    p.add_argument("--output-dir", default=None,
                   help="Results folder (default: baselines/audio_llm_baseline_runs/{model_slug})")
    p.add_argument("--cache-dir", default=None,
                   help="Per-clip JSON cache (default: baselines/audio_llm_cache/{model_slug})")
    p.add_argument("--prompt-template", default="zero_shot_v1")
    p.add_argument("--n-shot", type=int, default=0)
    p.add_argument("--universal-shots", action="store_true",
                   help="If set, --train-csv is treated as a universal demo pool "
                        "(no per-query child_id filter). Used for synthetic shots.")
    p.add_argument("--threshold", type=float, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"])
    p.add_argument("--quantize-4bit", action="store_true")
    p.add_argument("--max-clips", type=int, default=None)
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def main():
    args = _parse_args()

    # Resolve defaults
    if args.split_csv is None:
        if args.split == "test_all":
            args.split_csv = "whisper-modeling/all_children_splits/test_all.csv"
        else:
            args.split_csv = f"whisper-modeling/seen_child_splits/{args.split}.csv"
    if args.output_dir is None:
        args.output_dir = f"baselines/audio_llm_baseline_runs/{args.model_slug}"
    if args.cache_dir is None:
        args.cache_dir = f"baselines/audio_llm_cache/{args.model_slug}"

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    Path(args.cache_dir).mkdir(parents=True, exist_ok=True)

    # Guard: test / test_all require val to have already run (threshold tuning)
    val_metrics_path = out_dir / "val_metrics_tuned.json"
    if args.split in {"test", "test_all"} and not val_metrics_path.exists():
        print(f"ERROR: val_metrics_tuned.json not found at {val_metrics_path}. "
              f"Run --split val first.", file=sys.stderr)
        sys.exit(2)

    # Load split CSV
    split_df = pd.read_csv(args.split_csv)
    if args.max_clips is not None:
        split_df = split_df.head(args.max_clips)

    prompt_text = _build_prompt_text(args.prompt_template)

    # Dry-run: print 3 example prompts and exit
    if args.dry_run:
        print(f"=== Dry run: prompt template '{args.prompt_template}' ===\n")
        for i, (_, row) in enumerate(split_df.head(3).iterrows()):
            clip_id = row.get("clip_id", row.get("audio_path", f"clip_{i}"))
            print(f"[{i+1}] clip_id={clip_id}")
            print(f"     prompt: {prompt_text}\n")
        sys.exit(0)

    # Load model
    print(f"Loading model {args.model!r} ...")
    processor, model = _load_model(args.model, args.dtype, args.quantize_4bit, args.device,
                                   model_class_name=args.model_class)
    print("Model loaded.\n")

    label_col = "label" if "label" in split_df.columns else "child_vocalizing"
    audio_col = "audio_path"

    rows = []
    n_cached = 0
    n_error = 0

    for idx, row in split_df.iterrows():
        clip_id = str(row.get("clip_id", row.get(audio_col, f"clip_{idx}")))
        audio_path = str(row[audio_col])
        label = int(row[label_col]) if label_col in row and not pd.isna(row[label_col]) else -1
        child_id = str(row.get("child_id", ""))
        timepoint_norm = str(row.get("timepoint_norm", ""))

        if (idx % 50 == 0) or idx == split_df.index[0]:
            print(f"  [{idx - split_df.index[0] + 1}/{len(split_df)}] {clip_id} ...")

        # Try cache first (validates prompt_text_hash to detect stale caches
        # after prompt template edits — see fix #3 in CHANGELOG.md)
        cp = _cache_path(audio_path, args.model_slug, args.cache_dir)
        cached = _load_cache(cp, expected_prompt_hash=_prompt_hash(prompt_text))

        if cached is not None:
            n_cached += 1
            result = cached
        else:
            # Load audio
            waveform, sr = _load_audio(audio_path)
            if waveform is None:
                n_error += 1
                rows.append({
                    "clip_id": clip_id, "child_id": child_id,
                    "timepoint_norm": timepoint_norm, "audio_path": audio_path,
                    "label": label, "prob": float("nan"), "predicted": float("nan"),
                    "model_name": args.model, "prompt_template": args.prompt_template,
                    "n_shot": args.n_shot, "response_raw": "",
                    "parse_status": "error", "logit_yes": float("nan"), "logit_no": float("nan"),
                })
                continue

            # Few-shot examples
            few_shot_wavs = None
            if args.n_shot > 0:
                ex_paths = _find_few_shot_examples(
                    audio_path, args.train_csv, args.n_shot, args.seed,
                    universal=args.universal_shots,
                )
                if ex_paths:
                    few_shot_wavs = []
                    for ex_path, ex_label in ex_paths:
                        ex_wav, ex_sr = _load_audio(ex_path)
                        if ex_wav is not None:
                            few_shot_wavs.append((ex_wav, ex_label))

            result = _infer_clip(
                processor, model, waveform, sr, prompt_text,
                args.device, few_shot_examples=few_shot_wavs
            )
            result["clip_id"] = clip_id
            result["audio_path"] = audio_path
            result["model_name"] = args.model
            result["prompt_template"] = args.prompt_template
            result["prompt_text_hash"] = _prompt_hash(prompt_text)
            result["timestamp"] = datetime.now(timezone.utc).isoformat()
            _save_cache(cp, result)

        rows.append({
            "clip_id": clip_id,
            "child_id": child_id,
            "timepoint_norm": timepoint_norm,
            "audio_path": audio_path,
            "label": label,
            "prob": result["prob"],
            "predicted": float("nan"),  # filled after threshold
            "model_name": result.get("model_name", args.model),
            "prompt_template": args.prompt_template,
            "n_shot": args.n_shot,
            "response_raw": result.get("response_raw", ""),
            "parse_status": result.get("parse_status", "error"),
            "logit_yes": result.get("logit_yes", float("nan")),
            "logit_no": result.get("logit_no", float("nan")),
        })

    pred_df = pd.DataFrame(rows)
    valid_mask = pred_df["prob"].notna() & (pred_df["label"] >= 0)
    y_true = pred_df.loc[valid_mask, "label"].values
    y_score = pred_df.loc[valid_mask, "prob"].values

    # T008 — Threshold and metrics
    if args.split == "val":
        threshold = args.threshold if args.threshold is not None else _tune_threshold(y_true, y_score)
        metrics = _compute_metrics(y_true, y_score, threshold)
        metrics["threshold"] = threshold
        metrics["val_f1"] = metrics["f1"]
        metrics["n_positive"] = int(y_true.sum())
        metrics["n_negative"] = int((y_true == 0).sum())
        metrics["delta_f1_vs_babar"] = round(metrics["f1"] - BABAR_BASELINES["f1"], 4)
        metrics["delta_auroc_vs_babar"] = round(metrics["auroc"] - BABAR_BASELINES["auroc"], 4)
        metrics["delta_auprc_vs_babar"] = round(metrics["auprc"] - BABAR_BASELINES["auprc"], 4)

        with open(out_dir / "val_metrics_tuned.json", "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"\nVal metrics (thr={threshold:.2f}): F1={metrics['f1']:.3f} "
              f"AUROC={metrics['auroc']:.3f} AUPRC={metrics['auprc']:.3f}")

    else:  # test
        with open(val_metrics_path) as f:
            val_metrics = json.load(f)
        threshold = val_metrics["threshold"]
        val_f1 = val_metrics.get("val_f1", val_metrics.get("f1"))
        metrics = _compute_metrics(y_true, y_score, threshold)
        metrics["threshold"] = threshold
        metrics["val_f1"] = val_f1
        metrics["n_positive"] = int(y_true.sum())
        metrics["n_negative"] = int((y_true == 0).sum())
        metrics["delta_f1_vs_babar"] = round(metrics["f1"] - BABAR_BASELINES["f1"], 4)
        metrics["delta_auroc_vs_babar"] = round(metrics["auroc"] - BABAR_BASELINES["auroc"], 4)
        metrics["delta_auprc_vs_babar"] = round(metrics["auprc"] - BABAR_BASELINES["auprc"], 4)

        with open(out_dir / "test_metrics_tuned.json", "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"\nTest metrics (thr={threshold:.2f}): F1={metrics['f1']:.3f} "
              f"AUROC={metrics['auroc']:.3f} AUPRC={metrics['auprc']:.3f}")

        # Per-timepoint breakdown
        tp_rows = []
        for tp, grp in pred_df.groupby("timepoint_norm"):
            vmask = grp["prob"].notna() & (grp["label"] >= 0)
            if vmask.sum() < 2:
                continue
            tp_m = _compute_metrics(grp.loc[vmask, "label"].values,
                                    grp.loc[vmask, "prob"].values, threshold)
            tp_m["timepoint_norm"] = tp
            tp_m["n_clips"] = int(vmask.sum())
            tp_rows.append(tp_m)
        overall = {**metrics, "timepoint_norm": "overall", "n_clips": int(valid_mask.sum())}
        tp_rows.append(overall)
        pd.DataFrame(tp_rows).to_csv(out_dir / "test_metrics_by_timepoint.csv", index=False)

    # Apply threshold to predictions
    pred_df["predicted"] = (pred_df["prob"] >= threshold).astype("Int64")
    pred_df.loc[pred_df["prob"].isna(), "predicted"] = pd.NA
    pred_df.to_csv(out_dir / f"{args.split}_predictions.csv", index=False)
    print(f"Predictions written: {out_dir / f'{args.split}_predictions.csv'}")

    # T009 — Degenerate detection + config.json
    valid_probs = [r for r in pred_df["prob"].tolist() if not (isinstance(r, float) and np.isnan(r))]
    pred_variance = float(np.var(valid_probs)) if valid_probs else 0.0
    degenerate = pred_variance < 0.01
    if degenerate:
        print(
            f"\n[WARNING] Degenerate predictions detected (variance={pred_variance:.4f}). "
            "Check prompt format.\n"
        )

    n_yes = int(pred_df["predicted"].fillna(0).sum())
    n_total = len(pred_df)
    frac_yes = n_yes / n_total if n_total > 0 else 0.0

    if args.split == "test":
        config = {
            "model_name": args.model,
            "model_slug": args.model_slug,
            "prompt_template": args.prompt_template,
            "prompt_text": prompt_text,
            "n_shot": args.n_shot,
            "threshold": threshold,
            "val_f1": float(val_f1),
            "seed": args.seed,
            "split": "seen_child_splits",
            "n_clips_total": n_total,
            "n_clips_cached": n_cached,
            "n_clips_error": n_error,
            "prediction_variance": round(pred_variance, 6),
            "degenerate_flag": degenerate,
            "frac_yes": round(frac_yes, 4),
            "frac_no": round(1.0 - frac_yes, 4),
        }
        with open(out_dir / "config.json", "w") as f:
            json.dump(config, f, indent=2)
        print(f"Config written: {out_dir / 'config.json'}")

    print(f"\nDone. Results in {out_dir}/")


if __name__ == "__main__":
    main()
