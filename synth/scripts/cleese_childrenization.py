#!/usr/bin/env python3
"""CLEESE-style phase-vocoder childrenization (Yiwere et al., IEEE Access 2023).

Unlike the WORLD pipeline (`world_childrenization.py`), CLEESE alters
*only* pitch and duration via a phase vocoder, leaving the spectral
envelope intact. This produces a "younger sounding" voice that retains
adult formant structure -- useful as a comparison condition.

Implementation: ``librosa.effects.pitch_shift`` (phase-vocoder pitch
shift) and ``librosa.effects.time_stretch``. Per-segment pitch shift in
semitones is sampled to lift adult F0 toward the child target band, with
the shift amount jittered for diversity.

Usage
-----
::

    python synth/scripts/cleese_childrenization.py \
        --segment-manifest synth_results/manifests/segment_manifest_v2.csv \
        --output-dir       data/segments/cleese_childrenized/ \
        --output-manifest  synth_results/manifests/cleese_childrenized_manifest.csv \
        --source-datasets  librispeech,providence_adults \
        --max-segments     20000 --seed 42
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


DEFAULT_F0_TARGET_HZ = 250.0
DEFAULT_PITCH_SHIFT_SEMITONES_RANGE = (6.0, 12.0)
DEFAULT_DURATION_STRETCH_RANGE = (0.92, 1.10)


def _estimate_mean_f0(y: np.ndarray, sr: int) -> float:
    try:
        import librosa
        f0, _, _ = librosa.pyin(
            y, fmin=60.0, fmax=400.0, sr=sr, fill_na=np.nan
        )
        valid = f0[~np.isnan(f0)]
        if valid.size == 0:
            return 150.0
        return float(np.median(valid))
    except Exception:
        return 150.0


def childrenize_cleese(
    y: np.ndarray,
    sr: int,
    pitch_shift_semitones: float,
    duration_stretch: float = 1.0,
) -> np.ndarray:
    """Phase-vocoder childrenization."""
    import librosa

    if y.size < int(0.05 * sr):
        return y
    y_in = y.astype(np.float32)
    # Pitch shift first; this raises every voiced frame's F0 by N semitones
    # while preserving formants (phase vocoder).
    try:
        y_pitched = librosa.effects.pitch_shift(
            y=y_in, sr=sr, n_steps=float(pitch_shift_semitones)
        )
    except Exception:
        y_pitched = y_in
    # Then optional duration stretch.
    if not math.isclose(duration_stretch, 1.0, abs_tol=1e-3):
        try:
            y_stretched = librosa.effects.time_stretch(
                y=y_pitched, rate=1.0 / duration_stretch
            )
        except Exception:
            y_stretched = y_pitched
    else:
        y_stretched = y_pitched

    peak_orig = float(np.max(np.abs(y))) or 1e-6
    peak_new = float(np.max(np.abs(y_stretched))) or 1e-6
    return (y_stretched * (peak_orig / peak_new)).astype(np.float32)


def filter_manifest(
    manifest_csv: Path,
    source_datasets: List[str],
    max_segments: Optional[int],
    seed: int,
) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with manifest_csv.open() as f:
        reader = csv.DictReader(f)
        for r in reader:
            ds = (r.get("source_dataset") or "").lower()
            if ds in source_datasets:
                rows.append(r)
    rng = np.random.default_rng(seed)
    if max_segments is not None and len(rows) > max_segments:
        idx = rng.choice(len(rows), max_segments, replace=False)
        rows = [rows[i] for i in idx]
    return rows


def stem_for(row: Dict[str, str]) -> str:
    raw = row.get("segment_id") or row.get("audio_path", "")
    h = hashlib.md5(raw.encode()).hexdigest()[:10]
    return f"{(row.get('source_dataset') or 'src')}_{h}"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--segment-manifest", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--output-manifest", type=Path, required=True)
    p.add_argument(
        "--source-datasets",
        type=str,
        default="librispeech,providence_adults,playlogue_adults",
    )
    p.add_argument("--max-segments", type=int, default=20000)
    p.add_argument("--target-f0-hz", type=float, default=DEFAULT_F0_TARGET_HZ)
    p.add_argument(
        "--pitch-shift-semitones-range",
        type=str,
        default=f"{DEFAULT_PITCH_SHIFT_SEMITONES_RANGE[0]},"
        f"{DEFAULT_PITCH_SHIFT_SEMITONES_RANGE[1]}",
    )
    p.add_argument(
        "--duration-stretch-range",
        type=str,
        default=f"{DEFAULT_DURATION_STRETCH_RANGE[0]},"
        f"{DEFAULT_DURATION_STRETCH_RANGE[1]}",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--sample-rate", type=int, default=16000)
    p.add_argument("--shard-id", type=int, default=0)
    p.add_argument("--n-shards", type=int, default=1)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)

    sources = [s.strip().lower() for s in args.source_datasets.split(",")
               if s.strip()]
    ps_lo, ps_hi = (float(x) for x in args.pitch_shift_semitones_range.split(","))
    ds_lo, ds_hi = (float(x) for x in args.duration_stretch_range.split(","))

    rows = filter_manifest(args.segment_manifest, sources, args.max_segments,
                           args.seed)
    if args.n_shards > 1:
        rows = [r for i, r in enumerate(rows) if i % args.n_shards == args.shard_id]
    print(f"CLEESE-childrenizing {len(rows)} segments "
          f"(shard {args.shard_id}/{args.n_shards}) ...")

    rng = np.random.default_rng(args.seed + args.shard_id * 7919)

    import librosa
    import soundfile as sf

    out_rows: List[Dict[str, str]] = []
    for i, r in enumerate(rows):
        try:
            audio_path = r["audio_path"]
            start = float(r.get("start_time_sec") or 0.0)
            end = float(r.get("end_time_sec") or 0.0)
            duration = end - start if end > start else None
            y, sr = librosa.load(audio_path, sr=args.sample_rate, mono=True,
                                 offset=start, duration=duration)
            if y.size < int(0.1 * sr):
                continue

            mean_f0 = _estimate_mean_f0(y, sr)
            # Per-source pitch shift in semitones: target / measured -> base
            # shift, plus random jitter inside the configured range.
            base_shift = 12.0 * math.log2(
                max(args.target_f0_hz, 80.0) / max(mean_f0, 70.0)
            )
            jitter = float(rng.uniform(-1.5, 1.5))
            shift = max(min(base_shift + jitter, ps_hi), ps_lo)
            stretch = float(rng.uniform(ds_lo, ds_hi))

            y_new = childrenize_cleese(
                y, sr,
                pitch_shift_semitones=shift,
                duration_stretch=stretch,
            )
            out_path = args.output_dir / f"{stem_for(r)}.wav"
            sf.write(str(out_path), y_new.astype(np.float32), sr)

            new_row = dict(r)
            new_row["source_dataset"] = (
                f"cleese_childrenized_{(r.get('source_dataset') or 'src').lower()}"
            )
            new_row["audio_path"] = str(out_path)
            new_row["start_time_sec"] = "0.0"
            new_row["end_time_sec"] = f"{(y_new.size / sr):.4f}"
            new_row["duration_sec"] = f"{(y_new.size / sr):.4f}"
            new_row["speaker_role"] = "target_child"
            new_row["age_band"] = "14_18_months"
            new_row["age_months"] = "16"
            new_row["transcript"] = ""
            new_row["phonetic_transcript"] = ""
            new_row["vocalization_type"] = "cleese_childrenized"
            new_row.setdefault("childrenization_pitch_shift_semitones", str(shift))
            new_row.setdefault("childrenization_duration_stretch", str(stretch))
            out_rows.append(new_row)
        except Exception as e:
            print(f"  [skip] {r.get('segment_id', '?')}: {e}")
            continue

        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{len(rows)} ...")

    fieldnames = list({k for row in out_rows for k in row.keys()})
    preferred = [
        "segment_id", "source_dataset", "source_recording_id", "speaker_id",
        "speaker_role", "age_months", "age_band", "start_time_sec",
        "end_time_sec", "duration_sec", "audio_path", "sample_rate",
        "transcript", "phonetic_transcript", "vocalization_type",
        "quality_score", "split", "usable_for_training",
        "childrenization_pitch_shift_semitones",
        "childrenization_duration_stretch",
    ]
    ordered = [c for c in preferred if c in fieldnames]
    extra = [c for c in fieldnames if c not in ordered]
    fieldnames = ordered + extra

    write_path = args.output_manifest
    if args.n_shards > 1:
        write_path = args.output_manifest.with_suffix(
            f".shard{args.shard_id:03d}.csv"
        )
    with write_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in out_rows:
            writer.writerow(row)
    print(f"Wrote {len(out_rows)} CLEESE-childrenized rows -> {write_path}")


if __name__ == "__main__":
    main()
