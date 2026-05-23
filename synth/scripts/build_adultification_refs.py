#!/usr/bin/env python3
"""Build reference CSVs for the adultification evaluation battery.

Reads the existing segment manifest (v2 by default) and produces three
CSVs (path, role, age_band) suitable for ``adultification_eval.py``:

* real_child_<band>.csv: real Providence + TinyVox child segments in band
* real_adult.csv: real LibriSpeech / Providence-adult segments
* synth_eval_<band>.csv: synth segments tagged TARGET_CHILD in band

Usage
-----
::

    python synth/scripts/build_adultification_refs.py \
        --segment-manifest synth_results/manifests/segment_manifest_v2.csv \
        --synth-rttm-dir synth_results/synthetic_scenes_v3_perturb/rttm \
        --synth-wav-dir synth_results/synthetic_scenes_v3_perturb/wav \
        --output-dir synth_results/manifests/adultification_refs/ \
        --age-band 14_18 --max-per-set 600
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


CHILD_DATASETS = {"providence", "tinyvox", "playlogue"}
ADULT_DATASETS = {"providence_adults", "librispeech", "playlogue_adults"}


def _parse_band(band: str) -> Tuple[int, int]:
    lo, hi = band.split("_")
    return int(lo), int(hi)


def _row_in_band(row: Dict[str, str], band: Tuple[int, int]) -> bool:
    age = row.get("age_band") or row.get("age_months") or ""
    age = age.strip().lower()
    # Accept both "14_18_months", "14_18m", or numeric months.
    m = re.match(r"(\d+)_(\d+)", age)
    if m:
        a_lo, a_hi = int(m.group(1)), int(m.group(2))
        return not (a_hi < band[0] or a_lo > band[1])
    if age.isdigit():
        a = int(age)
        return band[0] <= a <= band[1]
    # Fall through: accept rows that don't have an age field.
    return False


def filter_real(
    manifest_csv: Path,
    band: Tuple[int, int],
    role_datasets: set,
    max_count: int,
    seed: int = 42,
) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with manifest_csv.open() as f:
        reader = csv.DictReader(f)
        for r in reader:
            ds = (r.get("source_dataset") or "").lower()
            if ds not in role_datasets:
                continue
            # If we are pulling adults, skip the band check (adults are
            # age-invariant for the purposes of this battery).
            if role_datasets is ADULT_DATASETS or _row_in_band(r, band):
                rows.append(r)
    rng = np.random.default_rng(seed)
    if len(rows) > max_count:
        idx = rng.choice(len(rows), max_count, replace=False)
        rows = [rows[i] for i in idx]
    return rows


def write_csv(rows: List[Dict[str, str]], path: Path, role: str) -> int:
    """Write a CSV with columns (path, role, age_band, start_sec, end_sec).

    Audio is loaded with the start/end offsets at featurization time so we
    don't need pre-extracted standalone WAVs. ``start_sec``/``end_sec`` may
    be empty for already-extracted single-segment WAVs.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path", "role", "age_band", "start_sec", "end_sec"])
        for r in rows:
            audio_path = (
                r.get("audio_path")
                or r.get("extracted_path")
                or r.get("path")
                or r.get("wav")
                or ""
            )
            if not audio_path:
                continue
            age = r.get("age_band") or r.get("age_months") or ""
            start = r.get("start_time_sec") or ""
            end = r.get("end_time_sec") or ""
            w.writerow([audio_path, role, age, start, end])
            n += 1
    return n


def collect_synth_eval(
    synth_wav_dir: Path,
    synth_rttm_dir: Path,
    band_label: str,
    max_count: int,
    seed: int = 42,
) -> List[Tuple[str, str]]:
    """Extract per-scene TARGET_CHILD segments from synth RTTMs.

    Returns list of (segment_wav_path, age_band). The caller must produce
    the segment WAVs with ``--extract-synth-segments`` if not already
    present; for the lightweight version we point at the full scene WAV
    and let ``adultification_eval.py`` operate on the entire scene
    (acceptable when synth scene was generated with positive==1).
    """
    rng = np.random.default_rng(seed)
    wavs = sorted(synth_wav_dir.glob("*.wav"))
    if not wavs:
        return []
    if len(wavs) > max_count:
        idx = rng.choice(len(wavs), max_count, replace=False)
        wavs = [wavs[i] for i in idx]

    out: List[Tuple[str, str]] = []
    for wav in wavs:
        rttm = synth_rttm_dir / f"{wav.stem}.rttm"
        if not rttm.exists():
            continue
        # Quick check: does RTTM contain TARGET_CHILD?
        has_target = False
        with rttm.open() as f:
            for line in f:
                if "TARGET_CHILD" in line:
                    has_target = True
                    break
        if not has_target:
            continue
        out.append((str(wav), band_label))
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--segment-manifest", type=Path, required=True)
    p.add_argument("--synth-wav-dir", type=Path, default=None)
    p.add_argument("--synth-rttm-dir", type=Path, default=None)
    p.add_argument("--age-band", type=str, default="14_18")
    p.add_argument("--max-per-set", type=int, default=600)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    band = _parse_band(args.age_band)

    # Real-child CSV (band-restricted)
    child_rows = filter_real(args.segment_manifest, band,
                             CHILD_DATASETS, args.max_per_set, args.seed)
    n_child = write_csv(child_rows, args.output_dir /
                        f"real_child_{args.age_band}.csv", role="child")
    print(f"real_child: wrote {n_child} rows")

    # Real-adult CSV (age-invariant)
    adult_rows = filter_real(args.segment_manifest, band,
                             ADULT_DATASETS, args.max_per_set, args.seed)
    n_adult = write_csv(adult_rows, args.output_dir / "real_adult.csv",
                        role="adult")
    print(f"real_adult: wrote {n_adult} rows")

    # Synth-eval CSV: emit (scene_wav, child, band, child_start, child_end)
    # rows by reading TARGET_CHILD spans from each RTTM. This means we
    # featurize only the actual child portions of the synth scene rather
    # than the whole 30 s mixture.
    if args.synth_wav_dir is not None and args.synth_rttm_dir is not None:
        synth_path = args.output_dir / f"synth_eval_{args.age_band}.csv"
        synth_path.parent.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(args.seed)
        rttms = sorted(args.synth_rttm_dir.glob("*.rttm"))
        if len(rttms) > args.max_per_set * 4:
            idx = rng.choice(len(rttms), args.max_per_set * 4, replace=False)
            rttms = [rttms[i] for i in idx]
        n_rows = 0
        with synth_path.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["path", "role", "age_band", "start_sec", "end_sec"])
            for rttm in rttms:
                wav = args.synth_wav_dir / f"{rttm.stem}.wav"
                if not wav.exists():
                    continue
                with rttm.open() as fh:
                    for line in fh:
                        parts = line.split()
                        if len(parts) < 8 or parts[0] != "SPEAKER":
                            continue
                        if parts[7] != "TARGET_CHILD":
                            continue
                        try:
                            s = float(parts[3])
                            d = float(parts[4])
                        except ValueError:
                            continue
                        if d < 0.2:
                            continue
                        w.writerow([str(wav), "child", args.age_band,
                                    s, s + d])
                        n_rows += 1
                        if n_rows >= args.max_per_set:
                            break
                if n_rows >= args.max_per_set:
                    break
        print(f"synth_eval: wrote {n_rows} rows -> {synth_path}")
    else:
        print("synth_eval: skipped (no --synth-wav-dir/rttm-dir)")


if __name__ == "__main__":
    main()
