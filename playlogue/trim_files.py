#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import re
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd


def run(cmd: List[str]) -> None:
    subprocess.run(cmd, check=True)


def ffprobe_duration_s(path: Path) -> float:
    out = subprocess.check_output(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        text=True,
    ).strip()
    return float(out)


def build_mp3_index(audio_dir: Path) -> List[Tuple[str, Path]]:
    """
    Returns list of (stem_lower, path) for all mp3s under audio_dir (recursive).
    """
    items: List[Tuple[str, Path]] = []
    for dirpath, _, filenames in os.walk(audio_dir):
        for fn in filenames:
            p = Path(dirpath) / fn
            if p.suffix.lower() != ".mp3":
                continue
            items.append((p.stem.lower(), p))
    # Sort longer stems first so we prefer more specific matches if there are substrings.
    items.sort(key=lambda x: len(x[0]), reverse=True)
    return items


def resolve_source(id_str: str, mp3_index: List[Tuple[str, Path]]) -> Path:
    """
    Find the mp3 whose stem is a substring of id_str (case-insensitive).
    Prefer longest stem (index is pre-sorted).
    """
    hay = id_str.lower()
    for stem, p in mp3_index:
        if stem and stem in hay:
            return p
    raise FileNotFoundError(f"No mp3 filename stem found inside id={id_str}")


def safe_name(s: str, max_len: int = 220) -> str:
    # keep ID mostly intact but filesystem-safe
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", s)
    return s[:max_len]


def trim_to_wav(src_mp3: Path, dst_wav: Path, start_ms: int, end_ms: int) -> None:
    dst_wav.parent.mkdir(parents=True, exist_ok=True)
    if dst_wav.exists() and dst_wav.stat().st_size > 0:
        return

    start_s = start_ms / 1000.0
    if end_ms == -1:
        end_s = ffprobe_duration_s(src_mp3)
    else:
        end_s = end_ms / 1000.0

    dur_s = end_s - start_s
    if dur_s <= 0:
        raise ValueError(f"Non-positive duration: start={start_ms} end={end_ms} for {src_mp3.name}")

    # Decode mp3, trim, resample to 16kHz mono WAV
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(src_mp3),
        "-ss",
        f"{start_s:.3f}",
        "-t",
        f"{dur_s:.3f}",
        "-ac",
        "1",
        "-ar",
        "16000",
        str(dst_wav),
    ]
    run(cmd)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".", help="Project root containing huggingface/ and audio/")
    ap.add_argument("--timings", default="huggingface/clip_timings.csv", help="Path to clip_timings.csv (relative to root)")
    ap.add_argument("--audio_dir", default="audio", help="Directory containing source mp3s (relative to root)")
    ap.add_argument("--out_dir", default="trimmed", help="Output directory for trimmed wavs (relative to root)")
    ap.add_argument("--limit", type=int, default=None, help="Limit number of clips (debug)")
    args = ap.parse_args()

    root = Path(args.root)
    timings_path = root / args.timings
    audio_dir = root / args.audio_dir
    out_dir = root / args.out_dir

    if not timings_path.exists():
        raise SystemExit(f"Missing timings CSV: {timings_path}")
    if not audio_dir.exists():
        raise SystemExit(f"Missing audio dir: {audio_dir}")

    df = pd.read_csv(timings_path)
    required = {"id", "start_time", "end_time"}
    if not required.issubset(df.columns):
        raise SystemExit(f"Expected columns {sorted(required)}; got {list(df.columns)}")

    if args.limit is not None:
        df = df.head(args.limit)

    mp3_index = build_mp3_index(audio_dir)
    if not mp3_index:
        raise SystemExit(f"No mp3 files found under {audio_dir}")

    failures = 0
    for i, row in df.iterrows():
        clip_id = str(row["id"])
        start_ms = int(row["start_time"])
        end_ms = int(row["end_time"])

        try:
            src = resolve_source(clip_id, mp3_index)
            dst = out_dir / f"{safe_name(clip_id)}.wav"
            trim_to_wav(src, dst, start_ms, end_ms)
        except Exception as e:
            failures += 1
            print(f"[FAIL] row={i} id={clip_id}: {e}")

    print(f"Done. Wrote trimmed wavs to {out_dir} | failures={failures}/{len(df)}")


if __name__ == "__main__":
    main()