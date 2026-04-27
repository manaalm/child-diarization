"""Extract structured child-detection features from video frames using a local VLM.

Drop-in open-source replacement for extract_gpt4o_features.py.  Uses a local
HuggingFace vision-language model (default: LLaVA-1.5-7B) instead of the
OpenAI API — no API key, no cost, runs on one GPU.

Output schema is identical to gpt4o_features.csv so build_av_feature_table.py
can consume it via --gpt4o-features-csv without any changes.

Supported --model values:
  llava-1.5-7b   → llava-hf/llava-1.5-7b-hf        (~14 GB VRAM)
  llava-1.5-13b  → llava-hf/llava-1.5-13b-hf       (~26 GB VRAM)
  qwen2-vl-7b    → Qwen/Qwen2-VL-7B-Instruct        (~14 GB VRAM)
  qwen2-vl-2b    → Qwen/Qwen2-VL-2B-Instruct        (~4 GB VRAM)
  <hf-repo-id>   → any HF repo with a vision-language model

Caching: raw VLM responses saved per frame to
    av_fusion/llava_cache/<model_slug>/{clip_id}_{frame_idx}.json
Re-running skips clips already in the output CSV.

Usage:
    python av_fusion/scripts/extract_llava_features.py \\
        --metadata-csv whisper-modeling/seen_child_splits/master_with_split.csv \\
        --output       av_fusion/av_results/manual_only/gpt4o_features.csv \\
        [--model       llava-1.5-7b] \\
        [--sample-rate 2] \\
        [--batch-size  4] \\
        [--device      cuda]
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import get_repo_root

_REPO = get_repo_root()

_MODEL_ALIASES: Dict[str, str] = {
    "llava-1.5-7b":  "llava-hf/llava-1.5-7b-hf",
    "llava-1.5-13b": "llava-hf/llava-1.5-13b-hf",
    "qwen2-vl-7b":   "Qwen/Qwen2-VL-7B-Instruct",
    "qwen2-vl-2b":   "Qwen/Qwen2-VL-2B-Instruct",
}

_PROMPT = (
    "You are analyzing a video frame from a naturalistic home recording of a young child.\n"
    "Respond ONLY with a JSON object using this exact schema:\n"
    "{\n"
    '  "child_visible": "yes" | "no" | "uncertain",\n'
    '  "child_vocalizing": "yes" | "no" | "uncertain",\n'
    '  "n_children_visible": 0 | 1 | 2 | 3,\n'
    '  "visual_quality": "good" | "medium" | "poor",\n'
    '  "notes": "brief free-text description (max 50 words)"\n'
    "}\n"
    'Focus on the youngest person visible. "vocalizing" means mouth open, facing camera, '
    'or appearing to speak. "poor" quality means dark, blurry, or no recognizable scene content.'
)


# ---------------------------------------------------------------------------
# Frame helpers (shared with extract_gpt4o_features.py)
# ---------------------------------------------------------------------------

def _sample_frames(video_path: str, n_frames: int) -> List[np.ndarray]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return []
    indices = np.linspace(0, max(0, total - 1), n_frames, dtype=int)
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
    cap.release()
    return frames


def _bgr_to_pil(frame: np.ndarray):
    from PIL import Image
    return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def _load_cache(path: str) -> Optional[Dict]:
    if os.path.exists(path):
        try:
            with open(path) as f:
                return json.load(f)
        except Exception:
            pass
    return None


def _save_cache(path: str, data: Dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f)


# ---------------------------------------------------------------------------
# VLM backends
# ---------------------------------------------------------------------------

def _resolve_model_id(model_arg: str) -> str:
    return _MODEL_ALIASES.get(model_arg, model_arg)


def _load_llava(model_id: str, device: str):
    """Load LLaVA-1.5 via transformers."""
    import torch
    from transformers import AutoProcessor, LlavaForConditionalGeneration

    dtype = torch.float16 if "cuda" in device else torch.float32
    print(f"  Loading LLaVA model: {model_id} (dtype={dtype})")
    processor = AutoProcessor.from_pretrained(model_id)
    model = LlavaForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=dtype, device_map="auto" if device == "cuda" else device,
        low_cpu_mem_usage=True,
    )
    model.eval()
    return model, processor, "llava"


def _load_qwen2vl(model_id: str, device: str):
    """Load Qwen2-VL via transformers."""
    import torch
    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

    dtype = torch.bfloat16 if "cuda" in device else torch.float32
    print(f"  Loading Qwen2-VL model: {model_id} (dtype={dtype})")
    processor = AutoProcessor.from_pretrained(model_id)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=dtype, device_map="auto" if device == "cuda" else device,
        low_cpu_mem_usage=True,
    )
    model.eval()
    return model, processor, "qwen2vl"


def _load_model(model_id: str, device: str):
    """Auto-detect model family and load."""
    mid_lower = model_id.lower()
    if "qwen2" in mid_lower and "vl" in mid_lower:
        return _load_qwen2vl(model_id, device)
    else:
        return _load_llava(model_id, device)


# ---------------------------------------------------------------------------
# Per-frame inference
# ---------------------------------------------------------------------------

def _query_frame_llava(model, processor, pil_image, model_id: str) -> Dict[str, Any]:
    import torch

    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": _PROMPT},
            ],
        }
    ]
    prompt_text = processor.apply_chat_template(conversation, add_generation_prompt=True)
    inputs = processor(
        images=pil_image, text=prompt_text, return_tensors="pt"
    ).to(model.device, model.dtype if hasattr(model, "dtype") else torch.float16)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs, max_new_tokens=256, do_sample=False, temperature=1.0,
        )
    # Decode only the new tokens
    new_tokens = output_ids[0, inputs["input_ids"].shape[1]:]
    raw_text = processor.decode(new_tokens, skip_special_tokens=True).strip()
    return _parse_vlm_response(raw_text, model_id)


def _query_frame_qwen2vl(model, processor, pil_image, model_id: str) -> Dict[str, Any]:
    import torch

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": pil_image},
                {"type": "text", "text": _PROMPT},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[pil_image], return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=256, do_sample=False)
    new_tokens = output_ids[0, inputs["input_ids"].shape[1]:]
    raw_text = processor.batch_decode([new_tokens], skip_special_tokens=True)[0].strip()
    return _parse_vlm_response(raw_text, model_id)


def _query_frame(model, processor, family: str, pil_image, model_id: str) -> Dict[str, Any]:
    try:
        if family == "qwen2vl":
            return _query_frame_qwen2vl(model, processor, pil_image, model_id)
        else:
            return _query_frame_llava(model, processor, pil_image, model_id)
    except Exception as e:
        return {"_raw_text": str(e), "_api_error": True}


def _parse_vlm_response(raw_text: str, model_id: str) -> Dict[str, Any]:
    """Extract JSON from VLM response text."""
    # Try direct JSON parse
    try:
        parsed = json.loads(raw_text)
        parsed["_raw_text"] = raw_text
        parsed["_api_error"] = False
        return parsed
    except json.JSONDecodeError:
        pass

    # Try extracting JSON block from markdown fences or surrounding text
    json_match = re.search(r"\{[^{}]*\}", raw_text, re.DOTALL)
    if json_match:
        try:
            parsed = json.loads(json_match.group())
            parsed["_raw_text"] = raw_text
            parsed["_api_error"] = False
            return parsed
        except json.JSONDecodeError:
            pass

    return {"_raw_text": raw_text, "_api_error": False, "_parse_error": True}


# ---------------------------------------------------------------------------
# Aggregation (identical logic to extract_gpt4o_features.py)
# ---------------------------------------------------------------------------

def _aggregate_frames(frame_results: List[Dict], model_id: str) -> Dict[str, Any]:
    n_total = len(frame_results)
    n_api_error = sum(1 for r in frame_results if r.get("_api_error", False))
    good = [r for r in frame_results
            if not r.get("_api_error", False) and not r.get("_parse_error", False)]
    n_sampled = len(good)

    if n_sampled == 0:
        return {
            "child_visible_gpt4o": float("nan"),
            "child_vocalizing_gpt4o": float("nan"),
            "n_children_visible_mean": float("nan"),
            "visual_quality_gpt4o": float("nan"),
            "gpt4o_reasoning": "; ".join(r.get("_raw_text", "") for r in frame_results),
            "n_frames_sampled": 0,
            "n_frames_api_error": n_api_error,
            "model_used": model_id,
            "cost_usd_estimate": 0.0,
        }

    def _yn(val) -> Optional[float]:
        if isinstance(val, str):
            v = val.lower().strip()
            if v == "yes": return 1.0
            if v == "no": return 0.0
        return None

    def _safe_mean(vals, default=float("nan")):
        valid = [v for v in vals if v is not None and not (isinstance(v, float) and np.isnan(v))]
        return float(np.mean(valid)) if valid else default

    quality_map = {"good": 1.0, "medium": 0.5, "poor": 0.0}
    return {
        "child_visible_gpt4o": _safe_mean([_yn(r.get("child_visible")) for r in good]),
        "child_vocalizing_gpt4o": _safe_mean([_yn(r.get("child_vocalizing")) for r in good]),
        "n_children_visible_mean": _safe_mean(
            [r.get("n_children_visible", 0) for r in good
             if isinstance(r.get("n_children_visible"), (int, float))]
        ),
        "visual_quality_gpt4o": _safe_mean(
            [quality_map.get(str(r.get("visual_quality", "")).lower(), float("nan")) for r in good]
        ),
        "gpt4o_reasoning": " | ".join(
            str(r.get("notes", "")).strip() for r in good if r.get("notes")
        ),
        "n_frames_sampled": n_sampled,
        "n_frames_api_error": n_api_error,
        "model_used": model_id,
        "cost_usd_estimate": 0.0,
    }


def _nan_row(clip_id: str, model_id: str) -> Dict[str, Any]:
    return {
        "clip_id": clip_id,
        "child_visible_gpt4o": float("nan"),
        "child_vocalizing_gpt4o": float("nan"),
        "n_children_visible_mean": float("nan"),
        "visual_quality_gpt4o": float("nan"),
        "gpt4o_reasoning": "",
        "n_frames_sampled": 0,
        "n_frames_api_error": 0,
        "model_used": model_id,
        "cost_usd_estimate": 0.0,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract child-detection visual features using a local VLM (LLaVA / Qwen2-VL)."
    )
    parser.add_argument("--metadata-csv", required=True)
    parser.add_argument("--output", required=True,
                        help="Output CSV path (same schema as gpt4o_features.csv)")
    parser.add_argument("--model", default="llava-1.5-7b",
                        help="Model alias or HF repo ID (default: llava-1.5-7b). "
                             "Aliases: llava-1.5-7b, llava-1.5-13b, qwen2-vl-7b, qwen2-vl-2b")
    parser.add_argument("--sample-rate", type=int, default=2,
                        help="Frames to sample per clip (default: 2)")
    parser.add_argument("--device", default="cuda",
                        help="PyTorch device (default: cuda)")
    parser.add_argument("--cache-dir", default=None,
                        help="Per-frame JSON cache dir (default: av_fusion/llava_cache/<model>)")
    parser.add_argument("--max-clips", type=int, default=None)
    args = parser.parse_args()

    model_id = _resolve_model_id(args.model)
    model_slug = model_id.replace("/", "--")

    metadata_csv = args.metadata_csv if os.path.isabs(args.metadata_csv) else os.path.join(_REPO, args.metadata_csv)
    out_path = args.output if os.path.isabs(args.output) else os.path.join(_REPO, args.output)
    cache_dir = args.cache_dir or os.path.join(_REPO, "av_fusion", "llava_cache", model_slug)
    if not os.path.isabs(cache_dir):
        cache_dir = os.path.join(_REPO, cache_dir)
    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    if not os.path.exists(metadata_csv):
        print(f"ERROR: metadata CSV not found: {metadata_csv}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(metadata_csv, low_memory=False)
    video_col = next((c for c in ("video_path", "BidsProcessed", "BidsRaw") if c in df.columns), None)

    def _get_clip_id(row):
        return str(row["clip_id"]) if "clip_id" in row.index else str(row.name)

    def _get_video(row) -> Optional[str]:
        if video_col is None: return None
        val = row.get(video_col, None)
        if pd.isna(val) or not str(val).strip(): return None
        p = str(val)
        return p if os.path.exists(p) else None

    # Resume from existing output
    done_clips: set = set()
    existing_rows: List[Dict] = []
    if os.path.exists(out_path):
        existing_df = pd.read_csv(out_path, low_memory=False)
        if "clip_id" in existing_df.columns:
            done_clips = set(existing_df["clip_id"].astype(str))
            existing_rows = existing_df.to_dict("records")
            print(f"Resuming: {len(done_clips)} clips already done")

    video_clips, audio_only_clips = [], []
    for _, row in df.iterrows():
        cid = _get_clip_id(row)
        if cid in done_clips:
            continue
        vp = _get_video(row)
        if vp:
            video_clips.append((cid, vp))
        else:
            audio_only_clips.append(cid)

    if args.max_clips is not None:
        video_clips = video_clips[:args.max_clips]

    print(f"Model: {model_id}")
    print(f"Clips: {len(video_clips)} video + {len(audio_only_clips)} audio-only "
          f"+ {len(done_clips)} already done")
    print(f"Frames per clip: {args.sample_rate}")

    model, processor, family = _load_model(model_id, args.device)
    print("Model loaded.")

    rows = list(existing_rows)
    for cid in audio_only_clips:
        rows.append(_nan_row(cid, model_id))

    for i, (cid, vp) in enumerate(video_clips):
        print(f"  [{i + 1}/{len(video_clips)}] clip {cid}", flush=True)

        frames = _sample_frames(vp, args.sample_rate)
        if not frames:
            print(f"    WARNING: could not read frames from {vp}", file=sys.stderr)
            rows.append(_nan_row(cid, model_id))
            continue

        frame_results = []
        for fi, frame in enumerate(frames):
            cache_path = os.path.join(cache_dir, f"{cid}_{fi}.json")
            cached = _load_cache(cache_path)
            if cached is not None:
                frame_results.append(cached)
                continue

            pil_img = _bgr_to_pil(frame)
            result = _query_frame(model, processor, family, pil_img, model_id)
            _save_cache(cache_path, result)
            frame_results.append(result)

        agg = _aggregate_frames(frame_results, model_id)
        agg["clip_id"] = cid
        rows.append(agg)

        if (i + 1) % 20 == 0 or (i + 1) == len(video_clips):
            pd.DataFrame(rows).to_csv(out_path, index=False)
            print(f"    Checkpoint saved ({i + 1}/{len(video_clips)})")

    pd.DataFrame(rows).to_csv(out_path, index=False)
    n_success = sum(1 for r in rows if r.get("n_frames_sampled", 0) > 0)
    print(f"\nFeatures written to: {out_path}")
    print(f"  Total clips: {len(rows)}  |  With features: {n_success}")


if __name__ == "__main__":
    main()
