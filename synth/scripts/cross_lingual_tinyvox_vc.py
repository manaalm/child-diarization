"""Cross-lingual TinyVox child-to-child voice conversion (Zhang et al., 2024).

Reads non-English TinyVox child segments from the on-disk audio directory
(``data/tinyvox/audio/phon_*_*.wav``), assigns each one to an English
target child (from the seen-child training split), and uses kNN-VC
(``bshall/knn-vc``) to convert the source vocalization to the target
child's voice. Output WAVs are written under
``data/segments/cross_lingual_vc/<target_child>/`` and a manifest CSV
is emitted in the same shape as ``segment_manifest_v2.csv``.

This implements the Zhang 2024 finding that cross-lingual child-to-
child VC is the most useful augmentation regime for children's ASR:
non-English vocalizations preserve infant acoustic idiosyncrasies that
adult-source childrenization (`world_/cleese_childrenization.py`) does
not, while VC matches the target child's speaker profile so they
function as plausible additional positive examples.

Usage
-----
::

    python synth/scripts/cross_lingual_tinyvox_vc.py \
        --tinyvox-audio-dir data/tinyvox/audio \
        --children-csv      whisper-modeling/seen_child_splits/train.csv \
        --output-dir        data/segments/cross_lingual_vc/ \
        --output-manifest   synth_results/manifests/cross_lingual_vc_manifest.csv \
        --n-per-target      10 \
        --max-targets       50 \
        --device            cuda
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

NON_ENGLISH_PREFIXES = (
    "Biling", "Clinical", "French", "German", "Romance", "Spanish",
)
ENGLISH_PREFIXES = ("Eng-NA",)


def list_tinyvox_wavs(
    audio_dir: Path, languages: List[str]
) -> List[Path]:
    """Return TinyVox WAV files whose ``phon_<lang>_*`` prefix matches."""
    out: List[Path] = []
    if not audio_dir.exists():
        return out
    rx = re.compile(r"^phon_([^_]+)_")
    for p in sorted(audio_dir.glob("phon_*_*.wav")):
        m = rx.match(p.name)
        if not m:
            continue
        lang = m.group(1)
        if lang in languages:
            out.append(p)
    return out


def child_reference_clips(
    children_csv: Path, child_id: str, max_dur_sec: float = 30.0
) -> List[str]:
    import pandas as pd
    import soundfile as sf

    df = pd.read_csv(children_csv)
    rows = df[(df["child_id"] == child_id) & (df["label"] == 1)]
    paths: List[str] = []
    cum = 0.0
    for _, row in rows.iterrows():
        ap = row.get("audio_path")
        if not isinstance(ap, str) or not os.path.exists(ap):
            continue
        try:
            paths.append(ap)
            cum += sf.info(ap).duration
            if cum >= max_dur_sec:
                break
        except Exception:
            continue
    return paths


def sorted_target_children(children_csv: Path, max_targets: int) -> List[str]:
    """Pick training children with the most positive clips first."""
    import pandas as pd

    df = pd.read_csv(children_csv)
    pos_counts = (
        df[df["label"] == 1].groupby("child_id").size().sort_values(ascending=False)
    )
    return list(pos_counts.index)[:max_targets]


def md5short(s: str) -> str:
    return hashlib.md5(s.encode()).hexdigest()[:10]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tinyvox-audio-dir", type=Path, required=True)
    p.add_argument("--children-csv", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--output-manifest", type=Path, required=True)
    p.add_argument(
        "--source-languages",
        type=str,
        default=",".join(NON_ENGLISH_PREFIXES),
        help="Comma-separated TinyVox language prefixes to use as VC sources.",
    )
    p.add_argument("--n-per-target", type=int, default=10)
    p.add_argument("--max-targets", type=int, default=50)
    p.add_argument("--ref-max-dur-sec", type=float, default=30.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--shard-id", type=int, default=0)
    p.add_argument("--n-shards", type=int, default=1)
    return p.parse_args()


def load_knn_vc(device: str):
    """Load bshall/knn-vc via torch.hub. Returns the model object."""
    import torch
    return torch.hub.load(
        "bshall/knn-vc", "knn_vc",
        prematched=True, trust_repo=True, pretrained=True, device=device,
    )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)

    import torch
    import soundfile as sf

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    rng = random.Random(args.seed + args.shard_id * 7919)

    languages = [s.strip() for s in args.source_languages.split(",") if s.strip()]
    sources = list_tinyvox_wavs(args.tinyvox_audio_dir, languages)
    print(f"[xlingual] {len(sources)} non-English source WAVs ({languages})")

    targets = sorted_target_children(args.children_csv, args.max_targets)
    if args.n_shards > 1:
        targets = [t for i, t in enumerate(targets)
                   if i % args.n_shards == args.shard_id]
    print(f"[xlingual] {len(targets)} target children (shard {args.shard_id}/"
          f"{args.n_shards}): {targets[:5]}...")

    print(f"[xlingual] device={device} n_per_target={args.n_per_target}")
    t0 = time.time()
    knn_vc = load_knn_vc(device)
    print(f"[xlingual] knn-vc loaded in {time.time()-t0:.1f}s")

    out_rows: List[Dict[str, str]] = []
    n_done = 0
    n_failed = 0

    for ti, target_id in enumerate(targets):
        ref_paths = child_reference_clips(
            args.children_csv, target_id, max_dur_sec=args.ref_max_dur_sec
        )
        if not ref_paths:
            print(f"[xlingual] target={target_id}: no reference audio, skipping")
            continue
        try:
            matching_set = knn_vc.get_matching_set(ref_paths)
        except Exception as e:
            print(f"[xlingual] target={target_id}: REF-BUILD-FAILED ({e})")
            n_failed += 1
            continue

        # Sample N source segments for this target.
        rng.shuffle(sources)
        chosen = sources[: args.n_per_target]
        target_dir = args.output_dir / target_id
        target_dir.mkdir(parents=True, exist_ok=True)

        for src_path in chosen:
            try:
                query_seq = knn_vc.get_features(str(src_path))
                wav_out = knn_vc.match(query_seq, matching_set, topk=4)
                if isinstance(wav_out, torch.Tensor):
                    y_out = wav_out.detach().cpu().numpy().astype(np.float32)
                else:
                    y_out = np.asarray(wav_out, dtype=np.float32)
                if y_out.ndim > 1:
                    y_out = y_out[0]
                out_name = f"{target_id}_{md5short(src_path.name)}.wav"
                out_path = target_dir / out_name
                sf.write(str(out_path), y_out, 16000)
                src_lang = src_path.name.split("_")[1]
                out_rows.append({
                    "segment_id": (
                        f"xlingvc_{target_id}_{md5short(src_path.name)}"
                    ),
                    "source_dataset": f"cross_lingual_vc_tinyvox_{src_lang}",
                    "source_recording_id": src_path.stem,
                    "speaker_id": target_id,
                    "speaker_role": "target_child",
                    "age_months": "",
                    "age_band": "14_18_months",
                    "start_time_sec": "0.0",
                    "end_time_sec": f"{(y_out.size / 16000):.4f}",
                    "duration_sec": f"{(y_out.size / 16000):.4f}",
                    "audio_path": str(out_path),
                    "sample_rate": "16000",
                    "transcript": "",
                    "phonetic_transcript": "",
                    "vocalization_type": "cross_lingual_vc",
                    "quality_score": "1.0",
                    "split": "train",
                    "usable_for_training": "True",
                    "vc_source_language": src_lang,
                    "vc_target_child_id": target_id,
                    "vc_source_path": str(src_path),
                })
                n_done += 1
            except Exception as e:
                n_failed += 1
                continue

        if (ti + 1) % 5 == 0:
            print(f"[xlingual] {ti+1}/{len(targets)} targets done | "
                  f"{n_done} converts | {n_failed} fails")

    fieldnames = [
        "segment_id", "source_dataset", "source_recording_id", "speaker_id",
        "speaker_role", "age_months", "age_band", "start_time_sec",
        "end_time_sec", "duration_sec", "audio_path", "sample_rate",
        "transcript", "phonetic_transcript", "vocalization_type",
        "quality_score", "split", "usable_for_training",
        "vc_source_language", "vc_target_child_id", "vc_source_path",
    ]
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
    print(f"\n[xlingual] {n_done} segments converted, {n_failed} failed.")
    print(f"[xlingual] manifest -> {write_path}")


if __name__ == "__main__":
    main()
