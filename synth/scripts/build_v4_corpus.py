#!/usr/bin/env python3
"""Build v4 unified segment manifest combining v2 sources plus childrenized
augmentations (WORLD, CLEESE, cross-lingual VC).

Reads ``segment_manifest_v2.csv`` plus optional childrenization manifests
and concatenates into a single v4 manifest with a unified column layout.
The v4 manifest can then drive the existing scene generator
(``generate_scenes.py``) with the new ``v4_perturb_14_18mo.yaml`` config
that wires up empirical turn-taking distributions.

Usage
-----
::

    python synth/scripts/build_v4_corpus.py \
        --base-manifest synth_results/manifests/segment_manifest_v2.csv \
        --add-manifest  synth_results/manifests/world_childrenized_manifest.csv \
        --add-manifest  synth_results/manifests/cleese_childrenized_manifest.csv \
        --add-manifest  synth_results/manifests/cross_lingual_vc_manifest.csv \
        --output        synth_results/manifests/segment_manifest_v4.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, Iterable, List, Set


PREFERRED_COLUMNS = [
    "segment_id", "source_dataset", "source_recording_id", "speaker_id",
    "speaker_role", "age_months", "age_band", "start_time_sec",
    "end_time_sec", "duration_sec", "audio_path", "sample_rate",
    "transcript", "phonetic_transcript", "vocalization_type",
    "quality_score", "split", "usable_for_training",
    "childrenization_pitch_factor", "childrenization_spectral_warp",
    "childrenization_duration_stretch",
    "childrenization_pitch_shift_semitones",
    "vc_source_language", "vc_target_child_id", "vc_source_path",
]


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        print(f"WARNING: skipping missing manifest {path}")
        return []
    with path.open() as f:
        return list(csv.DictReader(f))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-manifest", type=Path, required=True)
    p.add_argument(
        "--add-manifest",
        type=Path,
        action="append",
        default=[],
        help="Childrenization manifest to append. Repeat for multiple sources.",
    )
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--include-shards",
        action="store_true",
        help="Auto-include matching .shardNNN.csv files for each --add-manifest.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, str]] = []
    rows.extend(read_csv_rows(args.base_manifest))
    print(f"v4 build: base={args.base_manifest} -> {len(rows)} rows so far")

    for add in args.add_manifest:
        n_before = len(rows)
        if args.include_shards:
            shards = sorted(add.parent.glob(add.stem + ".shard*.csv"))
            for sh in shards:
                rows.extend(read_csv_rows(sh))
        rows.extend(read_csv_rows(add))
        print(f"v4 build: add={add} -> {len(rows) - n_before} new rows")

    if not rows:
        raise SystemExit("No rows to write -- aborting.")

    fieldnames: List[str] = list(PREFERRED_COLUMNS)
    extra: Set[str] = set()
    for r in rows:
        for k in r.keys():
            if k not in fieldnames:
                extra.add(k)
    fieldnames += sorted(extra)

    with args.output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    print(f"v4 build: wrote {len(rows)} rows -> {args.output}")


if __name__ == "__main__":
    main()
