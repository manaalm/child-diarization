"""Extract structured child-detection features from video frames using GPT-4o vision API.

Samples N frames per clip, encodes them as base64 JPEG, and queries the OpenAI
vision API for structured JSON output per frame. Aggregates frame-level results
into per-clip GPT4oFeatureRow records.

Features per clip (GPT4oFeatureRow schema):
  child_visible_gpt4o       — fraction of sampled frames where child was visible
  child_vocalizing_gpt4o    — fraction of frames where child appeared to be vocalizing
  n_children_visible_mean   — mean count of children detected across frames
  visual_quality_gpt4o      — mean quality score (good=1.0 / medium=0.5 / poor=0.0)
  gpt4o_reasoning           — concatenated free-text notes from all frames
  n_frames_sampled          — frames successfully queried
  n_frames_api_error        — frames that returned API errors
  model_used                — GPT model name used
  cost_usd_estimate         — estimated API cost for this clip

Caching: raw API responses are saved per frame to
    av_fusion/gpt4o_cache/{clip_id}_{frame_idx}.json
Re-running skips clips already present in the output CSV.

Usage:
    export OPENAI_API_KEY=<your key>
    python av_fusion/scripts/extract_gpt4o_features.py \\
        --metadata-csv whisper-modeling/seen_child_splits/master_with_split.csv \\
        --output       av_fusion/av_results/run1/gpt4o_features.csv \\
        [--model       gpt-4o-mini] \\
        [--sample-rate 2] \\
        [--max-clips   N] \\
        [--dry-run]
"""

import argparse
import base64
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import get_repo_root

_REPO = get_repo_root()

# Cost estimates (USD per 1M tokens, as of gpt-4o-mini pricing)
_COST_PER_M_INPUT = {"gpt-4o-mini": 0.15, "gpt-4o": 2.50}
_TOKENS_PER_FRAME = 1000  # approximate for a JPEG frame

_SYSTEM_PROMPT = """You are analyzing a video frame from a naturalistic home recording of a young child.
Respond ONLY with a JSON object using this exact schema:
{
  "child_visible": "yes" | "no" | "uncertain",
  "child_vocalizing": "yes" | "no" | "uncertain",
  "n_children_visible": 0 | 1 | 2 | 3,
  "visual_quality": "good" | "medium" | "poor",
  "notes": "brief free-text description (max 50 words)"
}
Focus on the youngest person visible. "vocalizing" means mouth open, facing camera, or audibly speaking.
"poor" quality means dark, blurry, or no recognizable scene content."""


def _sample_frames(video_path: str, n_frames: int) -> List[np.ndarray]:
    """Sample n_frames evenly-spaced frames from a video file."""
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


def _encode_frame(frame: np.ndarray, quality: int = 85) -> str:
    """Encode a BGR frame to base64 JPEG string."""
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return base64.b64encode(buf.tobytes()).decode("utf-8")


def _load_frame_cache(cache_path: str) -> Optional[Dict[str, Any]]:
    if os.path.exists(cache_path):
        try:
            with open(cache_path) as f:
                return json.load(f)
        except Exception:
            pass
    return None


def _save_frame_cache(cache_path: str, data: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(data, f)


def _query_frame(client, model: str, b64_image: str, max_tokens: int = 256) -> Dict[str, Any]:
    """Query GPT-4o vision API with exponential backoff. Returns parsed dict or error dict."""
    for attempt in range(5):
        try:
            response = client.chat.completions.create(
                model=model,
                response_format={"type": "json_object"},
                max_tokens=max_tokens,
                temperature=0.0,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{b64_image}", "detail": "low"},
                            }
                        ],
                    },
                ],
            )
            raw_text = response.choices[0].message.content
            try:
                parsed = json.loads(raw_text)
                parsed["_raw_text"] = raw_text
                parsed["_api_error"] = False
                return parsed
            except json.JSONDecodeError:
                return {"_raw_text": raw_text, "_api_error": False, "_parse_error": True}
        except Exception as e:
            err_str = str(e).lower()
            if "rate" in err_str or "429" in err_str or "quota" in err_str:
                wait = 2 ** attempt
                print(f"    Rate limit hit, retrying in {wait}s (attempt {attempt + 1}/5)...", file=sys.stderr)
                time.sleep(wait)
                continue
            return {"_raw_text": str(e), "_api_error": True}
    return {"_raw_text": "Max retries exceeded", "_api_error": True}


def _aggregate_frames(frame_results: List[Dict[str, Any]], model: str) -> Dict[str, Any]:
    """Aggregate frame-level results into GPT4oFeatureRow."""
    n_total = len(frame_results)
    n_api_error = sum(1 for r in frame_results if r.get("_api_error", False))
    good_results = [r for r in frame_results if not r.get("_api_error", False) and not r.get("_parse_error", False)]
    n_sampled = len(good_results)

    if n_sampled == 0:
        return {
            "child_visible_gpt4o": float("nan"),
            "child_vocalizing_gpt4o": float("nan"),
            "n_children_visible_mean": float("nan"),
            "visual_quality_gpt4o": float("nan"),
            "gpt4o_reasoning": "; ".join(r.get("_raw_text", "") for r in frame_results),
            "n_frames_sampled": 0,
            "n_frames_api_error": n_api_error,
            "model_used": model,
            "cost_usd_estimate": 0.0,
        }

    def _yn(val) -> Optional[float]:
        if isinstance(val, str):
            v = val.lower().strip()
            if v == "yes":
                return 1.0
            if v == "no":
                return 0.0
        return None  # uncertain or missing

    visible_vals = [_yn(r.get("child_visible")) for r in good_results]
    vocal_vals = [_yn(r.get("child_vocalizing")) for r in good_results]
    n_children_vals = [r.get("n_children_visible", 0) for r in good_results
                       if isinstance(r.get("n_children_visible"), (int, float))]
    quality_map = {"good": 1.0, "medium": 0.5, "poor": 0.0}
    quality_vals = [quality_map.get(str(r.get("visual_quality", "")).lower(), float("nan"))
                    for r in good_results]
    notes_list = [str(r.get("notes", "")).strip() for r in good_results if r.get("notes")]

    def _safe_mean(vals, default=float("nan")):
        valid = [v for v in vals if v is not None and not (isinstance(v, float) and np.isnan(v))]
        return float(np.mean(valid)) if valid else default

    cost = n_total * _TOKENS_PER_FRAME / 1_000_000 * _COST_PER_M_INPUT.get(model, 0.15)

    return {
        "child_visible_gpt4o": _safe_mean(visible_vals),
        "child_vocalizing_gpt4o": _safe_mean(vocal_vals),
        "n_children_visible_mean": _safe_mean(n_children_vals),
        "visual_quality_gpt4o": _safe_mean(quality_vals),
        "gpt4o_reasoning": " | ".join(notes_list) if notes_list else "",
        "n_frames_sampled": n_sampled,
        "n_frames_api_error": n_api_error,
        "model_used": model,
        "cost_usd_estimate": round(cost, 6),
    }


def _nan_row(clip_id: str, model: str) -> Dict[str, Any]:
    return {
        "clip_id": clip_id,
        "child_visible_gpt4o": float("nan"),
        "child_vocalizing_gpt4o": float("nan"),
        "n_children_visible_mean": float("nan"),
        "visual_quality_gpt4o": float("nan"),
        "gpt4o_reasoning": "",
        "n_frames_sampled": 0,
        "n_frames_api_error": 0,
        "model_used": model,
        "cost_usd_estimate": 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract GPT-4o vision features from video frames."
    )
    parser.add_argument("--metadata-csv", required=True,
                        help="CSV with columns clip_id, video_path (audio-only clips are skipped)")
    parser.add_argument("--output", required=True,
                        help="Output path for gpt4o_features.csv")
    parser.add_argument("--model", default="gpt-4o-mini",
                        choices=["gpt-4o-mini", "gpt-4o"],
                        help="GPT model to use (default: gpt-4o-mini)")
    parser.add_argument("--sample-rate", type=int, default=2,
                        help="Frames to sample per clip (default: 2)")
    parser.add_argument("--cache-dir", default=None,
                        help="Directory for per-frame JSON cache (default: av_fusion/gpt4o_cache/)")
    parser.add_argument("--max-clips", type=int, default=None,
                        help="Cap on number of video clips to process (for cost control)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print cost estimate without making API calls")
    args = parser.parse_args()

    # Resolve paths
    metadata_csv = args.metadata_csv if os.path.isabs(args.metadata_csv) else os.path.join(_REPO, args.metadata_csv)
    out_path = args.output if os.path.isabs(args.output) else os.path.join(_REPO, args.output)
    cache_dir = args.cache_dir or os.path.join(_REPO, "av_fusion", "gpt4o_cache")
    if not os.path.isabs(cache_dir):
        cache_dir = os.path.join(_REPO, cache_dir)
    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    if not os.path.exists(metadata_csv):
        print(f"ERROR: metadata CSV not found: {metadata_csv}", file=sys.stderr)
        sys.exit(1)

    # Check API key (not needed for dry-run)
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key and not args.dry_run:
        print("ERROR: OPENAI_API_KEY environment variable not set.", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(metadata_csv, low_memory=False)

    # Determine video path column
    video_col = None
    for candidate in ("video_path", "BidsProcessed", "BidsRaw"):
        if candidate in df.columns:
            video_col = candidate
            break

    # Build list of clips with valid video
    def _get_clip_id(row):
        if "clip_id" in row.index:
            return str(row["clip_id"])
        return str(row.name)

    def _get_video(row) -> Optional[str]:
        if video_col is None:
            return None
        val = row.get(video_col, None)
        if pd.isna(val) or not str(val).strip():
            return None
        p = str(val)
        return p if os.path.exists(p) else None

    # Load already-completed clips if output exists
    done_clips = set()
    existing_rows = []
    if os.path.exists(out_path):
        existing_df = pd.read_csv(out_path, low_memory=False)
        if "clip_id" in existing_df.columns:
            done_clips = set(existing_df["clip_id"].astype(str))
            existing_rows = existing_df.to_dict("records")
            print(f"Resuming: {len(done_clips)} clips already in output")

    # Identify clips to process
    video_clips = []
    audio_only_clips = []
    for _, row in df.iterrows():
        cid = _get_clip_id(row)
        if cid in done_clips:
            continue
        vp = _get_video(row)
        if vp:
            video_clips.append((cid, vp))
        else:
            audio_only_clips.append(cid)

    # Apply max-clips cap
    if args.max_clips is not None:
        video_clips = video_clips[: args.max_clips]

    # Count already-cached frames
    cached_frames = sum(
        1 for cid, _ in video_clips
        for fi in range(args.sample_rate)
        if os.path.exists(os.path.join(cache_dir, f"{cid}_{fi}.json"))
    )
    total_frames = len(video_clips) * args.sample_rate
    new_frames = total_frames - cached_frames
    cost_rate = _COST_PER_M_INPUT.get(args.model, 0.15)
    est_cost = new_frames * _TOKENS_PER_FRAME / 1_000_000 * cost_rate

    print(f"Clips: {len(video_clips)} video + {len(audio_only_clips)} audio-only "
          f"+ {len(done_clips)} already done")
    print(f"Frames to query: {len(video_clips)} × {args.sample_rate} = {total_frames} "
          f"(already cached: {cached_frames})")
    print(f"Estimated cost ({args.model}): ${est_cost:.4f}")

    if args.dry_run:
        print("Dry run complete — no API calls made.")
        return

    # Confirm
    if new_frames > 0:
        ans = input("Proceed? [y/N] ").strip().lower()
        if ans not in ("y", "yes"):
            print("Aborted.")
            return

    try:
        import openai
        client = openai.OpenAI(api_key=api_key)
    except ImportError:
        print("ERROR: openai package not installed. Run: pip install openai", file=sys.stderr)
        sys.exit(1)

    rows = list(existing_rows)

    # Write NaN rows for audio-only clips
    for cid in audio_only_clips:
        rows.append(_nan_row(cid, args.model))

    # Process video clips
    for i, (cid, vp) in enumerate(video_clips):
        print(f"  [{i + 1}/{len(video_clips)}] {cid}", flush=True)

        frames = _sample_frames(vp, args.sample_rate)
        if not frames:
            print(f"    WARNING: could not read frames from {vp}", file=sys.stderr)
            rows.append(_nan_row(cid, args.model))
            continue

        frame_results = []
        for fi, frame in enumerate(frames):
            cache_path = os.path.join(cache_dir, f"{cid}_{fi}.json")
            cached = _load_frame_cache(cache_path)
            if cached is not None:
                frame_results.append(cached)
                continue

            b64 = _encode_frame(frame)
            result = _query_frame(client, args.model, b64)
            _save_frame_cache(cache_path, result)
            frame_results.append(result)

        agg = _aggregate_frames(frame_results, args.model)
        agg["clip_id"] = cid
        rows.append(agg)

        # Incremental write
        if (i + 1) % 10 == 0 or (i + 1) == len(video_clips):
            pd.DataFrame(rows).to_csv(out_path, index=False)

    pd.DataFrame(rows).to_csv(out_path, index=False)
    n_success = sum(1 for r in rows if r.get("n_frames_sampled", 0) > 0)
    total_cost = sum(r.get("cost_usd_estimate", 0.0) for r in rows)
    print(f"\nGPT-4o features written to: {out_path}")
    print(f"  Total clips: {len(rows)}  |  With features: {n_success}  |  Estimated cost: ${total_cost:.4f}")


if __name__ == "__main__":
    main()
