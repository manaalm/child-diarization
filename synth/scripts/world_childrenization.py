#!/usr/bin/env python3
"""WORLD-vocoder childrenization (Zhao et al., Interspeech 2023).

Transforms adult speech into child-like audio by:

1. Decomposing each waveform into F0, spectral envelope (SP), and
   aperiodicity (AP) using the WORLD vocoder (``pyworld``).
2. Lifting F0 toward a target child median (default 250 Hz) by a
   per-segment multiplicative factor sampled from a configurable range.
3. Warping the spectral envelope along the frequency axis by a factor in
   ``[1.2, 1.4]`` (Zhao 2023's reported childrenization band) to compress
   formants toward shorter vocal-tract values.
4. Optionally stretching duration to mimic slower vowel timings in
   younger children.
5. Re-synthesizing audio.

Inputs are a manifest of adult segments (CSV with at least ``audio_path``
columns and the same shape as ``segment_manifest_v2.csv``). Outputs are
extracted childrenized WAV files plus a manifest entry rewrite with
``source_dataset = world_childrenized_<orig>``.

Usage
-----
::

    python synth/scripts/world_childrenization.py \
        --segment-manifest synth_results/manifests/segment_manifest_v2.csv \
        --output-dir       data/segments/world_childrenized/ \
        --output-manifest  synth_results/manifests/world_childrenized_manifest.csv \
        --source-datasets  librispeech,providence_adults \
        --max-segments     20000 --seed 42
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# WORLD-based childrenization core
# ---------------------------------------------------------------------------

DEFAULT_F0_TARGET_HZ = 250.0  # Zhao 2023 childrenization target
DEFAULT_SPECTRAL_WARP_RANGE = (1.2, 1.4)
DEFAULT_DURATION_STRETCH_RANGE = (1.0, 1.15)


def warp_spectral_envelope(sp: np.ndarray, warp_factor: float) -> np.ndarray:
    """Frequency-axis warping of the spectral envelope.

    Higher ``warp_factor`` => formants compressed to lower frequencies, but
    Zhao 2023 reports moving formants *upward* via factor > 1 (shorter VTL).
    Implementation: re-sample each frame's envelope along the frequency
    axis by the given factor and clip / pad with the highest valid bin.
    """
    n_frames, n_bins = sp.shape
    out = np.zeros_like(sp)
    src_idx = np.arange(n_bins) / float(warp_factor)
    src_idx = np.clip(src_idx, 0, n_bins - 1)
    floor_idx = np.floor(src_idx).astype(np.int32)
    ceil_idx = np.minimum(floor_idx + 1, n_bins - 1)
    frac = src_idx - floor_idx
    for f in range(n_frames):
        out[f] = sp[f, floor_idx] * (1 - frac) + sp[f, ceil_idx] * frac
    return out


def childrenize_world(
    y: np.ndarray,
    sr: int,
    pitch_factor: float,
    spectral_warp: float,
    duration_stretch: float = 1.0,
) -> np.ndarray:
    """Apply WORLD-based childrenization to one mono waveform."""
    import pyworld as pw

    y64 = y.astype(np.float64)
    if y64.size < int(0.05 * sr):
        return y  # too short for WORLD; pass through

    # Use DIO + StoneMask for F0 (less prone to halving than HARVEST on
    # short segments) and CheapTrick + D4C for spectrum/aperiodicity.
    frame_period_ms = 5.0
    f0_raw, t = pw.dio(y64, sr, frame_period=frame_period_ms)
    f0 = pw.stonemask(y64, f0_raw, t, sr)
    sp = pw.cheaptrick(y64, f0, t, sr)
    ap = pw.d4c(y64, f0, t, sr)

    # Apply pitch factor on voiced frames (f0 > 0); leave unvoiced (0) alone.
    f0_new = np.where(f0 > 0, f0 * pitch_factor, 0.0)

    # Warp spectral envelope.
    sp_new = warp_spectral_envelope(sp.astype(np.float64), spectral_warp)

    # Synthesize.
    y_new = pw.synthesize(f0_new, sp_new, ap.astype(np.float64),
                          sr, frame_period=frame_period_ms)

    # Optional duration stretch via librosa (independent of WORLD synth).
    if not math.isclose(duration_stretch, 1.0, abs_tol=1e-3):
        try:
            import librosa
            y_new = librosa.effects.time_stretch(
                y_new.astype(np.float32), rate=1.0 / duration_stretch
            )
        except Exception:
            pass

    # Match peak loudness to original to avoid SNR drift downstream.
    peak_orig = float(np.max(np.abs(y))) or 1e-6
    peak_new = float(np.max(np.abs(y_new))) or 1e-6
    y_new = (y_new * (peak_orig / peak_new)).astype(np.float32)
    return y_new


# ---------------------------------------------------------------------------
# Manifest IO
# ---------------------------------------------------------------------------

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
    raw = (
        row.get("segment_id")
        or row.get("audio_path", "")
    )
    h = hashlib.md5(raw.encode()).hexdigest()[:10]
    return f"{(row.get('source_dataset') or 'src')}_{h}"


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--segment-manifest", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--output-manifest", type=Path, required=True)
    p.add_argument(
        "--source-datasets",
        type=str,
        default="librispeech,providence_adults,playlogue_adults",
        help="Comma-separated source_dataset values to childrenize.",
    )
    p.add_argument("--max-segments", type=int, default=20000)
    p.add_argument("--target-f0-hz", type=float, default=DEFAULT_F0_TARGET_HZ)
    p.add_argument(
        "--spectral-warp-range",
        type=str,
        default=f"{DEFAULT_SPECTRAL_WARP_RANGE[0]},{DEFAULT_SPECTRAL_WARP_RANGE[1]}",
    )
    p.add_argument(
        "--duration-stretch-range",
        type=str,
        default=f"{DEFAULT_DURATION_STRETCH_RANGE[0]},{DEFAULT_DURATION_STRETCH_RANGE[1]}",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--sample-rate", type=int, default=16000)
    p.add_argument("--shard-id", type=int, default=0,
                   help="Optional shard index for SLURM array.")
    p.add_argument("--n-shards", type=int, default=1)
    return p.parse_args()


def _per_segment_pitch_factor(
    rng: np.random.Generator, target_f0: float, mean_f0_estimate: float
) -> float:
    # Multiplicative jitter around target / measured for diversity.
    base = target_f0 / max(mean_f0_estimate, 80.0)
    jitter = float(rng.uniform(0.92, 1.10))
    return float(base * jitter)


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


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)

    sources = [s.strip().lower() for s in args.source_datasets.split(",") if s.strip()]
    sw_lo, sw_hi = (float(x) for x in args.spectral_warp_range.split(","))
    ds_lo, ds_hi = (float(x) for x in args.duration_stretch_range.split(","))

    rows = filter_manifest(args.segment_manifest, sources, args.max_segments,
                           args.seed)
    if args.n_shards > 1:
        rows = [r for i, r in enumerate(rows) if i % args.n_shards == args.shard_id]
    print(f"Childrenizing {len(rows)} segments "
          f"(shard {args.shard_id}/{args.n_shards}) ...")

    rng = np.random.default_rng(args.seed + args.shard_id * 7919)

    import soundfile as sf
    import librosa

    out_rows: List[Dict[str, str]] = []
    for i, r in enumerate(rows):
        try:
            audio_path = r["audio_path"]
            start = float(r.get("start_time_sec") or 0.0)
            end = float(r.get("end_time_sec") or 0.0)
            duration = end - start if end > start else None
            y, sr = librosa.load(audio_path, sr=args.sample_rate, mono=True,
                                 offset=start,
                                 duration=duration)
            if y.size < int(0.1 * sr):
                continue

            mean_f0 = _estimate_mean_f0(y, sr)
            pitch_factor = _per_segment_pitch_factor(
                rng, args.target_f0_hz, mean_f0
            )
            spectral_warp = float(rng.uniform(sw_lo, sw_hi))
            duration_stretch = float(rng.uniform(ds_lo, ds_hi))

            y_new = childrenize_world(
                y, sr, pitch_factor=pitch_factor,
                spectral_warp=spectral_warp,
                duration_stretch=duration_stretch,
            )
            out_path = args.output_dir / f"{stem_for(r)}.wav"
            sf.write(str(out_path), y_new.astype(np.float32), sr)

            new_row = dict(r)
            new_row["source_dataset"] = (
                f"world_childrenized_{(r.get('source_dataset') or 'src').lower()}"
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
            new_row["vocalization_type"] = "world_childrenized"
            new_row.setdefault("childrenization_pitch_factor", str(pitch_factor))
            new_row.setdefault("childrenization_spectral_warp", str(spectral_warp))
            new_row.setdefault("childrenization_duration_stretch", str(duration_stretch))
            out_rows.append(new_row)
        except Exception as e:
            print(f"  [skip] {r.get('segment_id', '?')}: {e}")
            continue

        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{len(rows)} ...")

    fieldnames = list({k for row in out_rows for k in row.keys()})
    # Stable column order matching segment_manifest_v2.csv where possible.
    preferred = [
        "segment_id", "source_dataset", "source_recording_id", "speaker_id",
        "speaker_role", "age_months", "age_band", "start_time_sec",
        "end_time_sec", "duration_sec", "audio_path", "sample_rate",
        "transcript", "phonetic_transcript", "vocalization_type",
        "quality_score", "split", "usable_for_training",
        "childrenization_pitch_factor", "childrenization_spectral_warp",
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
    print(f"Wrote {len(out_rows)} childrenized rows -> {write_path}")


if __name__ == "__main__":
    main()
