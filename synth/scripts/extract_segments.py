#!/usr/bin/env python3
"""
Extract audio segments from source recordings and save as 16 kHz mono WAV files.

Reads a segment manifest CSV produced by build_segment_manifest.py,
extracts each segment where usable_for_training=True, and saves to:
    {output_dir}/{speaker_role}/{segment_id}.wav

Updates the audio_path column in the manifest CSV in-place.
Idempotent: already-extracted files are skipped unless --force is given.

Usage:
    python synth/scripts/extract_segments.py \\
      --manifest  synth_results/manifests/segment_manifest.csv \\
      --output-dir data/segments/ \\
      --sample-rate 16000
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from synth.audio_utils import resample_to_16k

_ROLE_DIR_MAP = {
    "target_child": "child",
    "non_target_child": "child",
    "unknown_child": "child",
    "adult": "adult",
    "background": "background",
}


def _extract_segment(
    audio_path: str,
    start_sec: float,
    end_sec: float,
    target_sr: int,
) -> np.ndarray:
    """Load a sub-segment from an audio file, resample to target_sr."""
    info = sf.info(audio_path)
    src_sr = info.samplerate
    start_frame = int(start_sec * src_sr)
    n_frames = max(1, int((end_sec - start_sec) * src_sr))

    wav, _ = sf.read(
        audio_path,
        start=start_frame,
        frames=n_frames,
        dtype="float32",
        always_2d=False,
    )
    if wav.ndim > 1:
        wav = wav.mean(axis=1)

    return resample_to_16k(wav, src_sr)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract segments from source audio and save 16 kHz WAV files."
    )
    parser.add_argument(
        "--manifest",
        required=True,
        help="Path to segment manifest CSV (produced by build_segment_manifest.py).",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Base output directory; child/ and adult/ sub-dirs are created.",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=16000,
        help="Target sample rate (default 16000).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="Re-extract even if output WAV already exists.",
    )
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        print(f"ERROR: manifest not found: {manifest_path}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(manifest_path, low_memory=False)
    # Coerce usable_for_training
    df["usable_for_training"] = df["usable_for_training"].map(
        lambda v: (
            v if isinstance(v, bool)
            else str(v).strip().lower() not in ("false", "0", "no", "")
        )
    ).astype(bool)

    usable = df[df["usable_for_training"]].copy()
    print(f"Extracting {len(usable)} usable segments from {len(df)} total.")

    out_root = Path(args.output_dir)
    n_extracted = 0
    n_skipped = 0
    n_errors = 0

    audio_path_updates: dict = {}

    for _, row in usable.iterrows():
        seg_id = str(row["segment_id"])
        speaker_role = str(row["speaker_role"])
        subdir = _ROLE_DIR_MAP.get(speaker_role, "other")

        out_dir = out_root / subdir
        out_dir.mkdir(parents=True, exist_ok=True)
        out_wav = out_dir / f"{seg_id}.wav"

        if out_wav.exists() and not args.force:
            audio_path_updates[seg_id] = str(out_wav.resolve())
            n_skipped += 1
            continue

        src_path = str(row["audio_path"])
        start_sec = float(row["start_time_sec"])
        end_sec = float(row["end_time_sec"])

        if not Path(src_path).exists():
            print(f"  [SKIP] Source not found: {src_path}", file=sys.stderr)
            n_errors += 1
            continue

        try:
            wav = _extract_segment(src_path, start_sec, end_sec, args.sample_rate)
            sf.write(str(out_wav), wav, args.sample_rate, subtype="PCM_16")
            audio_path_updates[seg_id] = str(out_wav.resolve())
            n_extracted += 1

            if n_extracted % 500 == 0:
                print(f"  Extracted {n_extracted} segments …")

        except Exception as e:
            print(f"  [ERROR] {seg_id}: {e}", file=sys.stderr)
            n_errors += 1

    # Update audio_path column in-place
    if audio_path_updates:
        seg_to_path = pd.Series(audio_path_updates, name="audio_path")
        df = df.set_index("segment_id")
        df.update(seg_to_path)
        df = df.reset_index()
        df.to_csv(manifest_path, index=False)
        print(f"Updated {len(audio_path_updates)} audio_path entries in {manifest_path}")

    print(
        f"\nDone: extracted={n_extracted}, "
        f"skipped={n_skipped}, errors={n_errors}"
    )


if __name__ == "__main__":
    main()
