"""Build the TinyVox + Providence speaker-pair manifest for spec-021 US4 (T070).

Per R4.1: ~1500 child speakers from TinyVox + Providence, AAM-Softmax-style
class-balanced sampling. The manifest is *speaker-keyed*; pair sampling happens
at training time via class-balanced batch sampler over speaker_id.

Output: models/ecapa_child_finetune/speaker_pair_manifest.csv with columns:
    audio_path, speaker_id, source, duration_sec
"""
from __future__ import annotations
import csv
import os
import re
from pathlib import Path

import pandas as pd
import soundfile as sf

REPO = Path(__file__).resolve().parents[2]
TINYVOX = REPO / "data/tinyvox/audio"
PROVIDENCE_MANIFEST = REPO / "providence/manifest.csv"
PROVIDENCE_RTTM_DIR = REPO / "providence/rttm"
PROVIDENCE_AUDIO_DIR = REPO / "providence/audio"
OUT = REPO / "models/ecapa_child_finetune/speaker_pair_manifest.csv"

# TinyVox file naming has two variants:
#   phon_{Lang}_{Corpus}_{Speaker}_{Age}_{Start}_{End}.wav  (most)
#   phon_{Lang}_{Corpus}_{SubCorpus}_{Speaker}_{Age}_{Start}_{End}.wav  (Romance_Portuguese)
# General rule: drop "phon_" prefix and ".wav" suffix; the last 3 underscore-
# separated tokens are (age, start_ms, end_ms); everything before is the
# speaker key.

MIN_UTTERANCES_PER_SPEAKER = 10    # Need enough samples per speaker for contrastive
MIN_DURATION_SEC = 0.5             # Toss tiny clips
MAX_DURATION_SEC = 15.0            # Cap to bound training memory


def parse_tinyvox(filename: str) -> tuple[str, str] | None:
    """Return (speaker_id, source_tag) for a TinyVox file, or None on miss."""
    if not filename.startswith("phon_") or not filename.endswith(".wav"):
        return None
    stem = filename[len("phon_"):-len(".wav")]
    parts = stem.split("_")
    if len(parts) < 4:
        return None  # need at least Lang_Speaker_Age_X
    speaker_parts = parts[:-3]      # drop age, start, end
    speaker_id = "tinyvox_" + "_".join(speaker_parts)
    source_tag = f"tinyvox_{speaker_parts[0]}"
    return speaker_id, source_tag


def safe_duration(path: str) -> float | None:
    try:
        info = sf.info(path)
        return info.frames / info.samplerate
    except Exception:
        return None


def collect_tinyvox() -> list[dict]:
    rows = []
    for p in TINYVOX.glob("*.wav"):
        parsed = parse_tinyvox(p.name)
        if parsed is None:
            continue
        speaker_id, source_tag = parsed
        rows.append({
            "audio_path": str(p),
            "speaker_id": speaker_id,
            "source": source_tag,
            "duration_sec": None,
        })
    return rows


def collect_providence() -> list[dict]:
    rows = []
    if not PROVIDENCE_MANIFEST.exists():
        return rows
    df = pd.read_csv(PROVIDENCE_MANIFEST)
    # Treat each Providence file as a speaker turn from the target child.
    # Actual segment-level speaker labels need RTTM parsing; here we use
    # the audio file as the speaker key + recording-id index.
    for _, r in df.iterrows():
        cid = r.get("child_id")
        if pd.isna(cid):
            continue
        # Each session contributes a speaker-id; multi-session combines per child.
        rows.append({
            "audio_path": str(r["path"]),
            "speaker_id": f"providence_{cid}",
            "source": "providence",
            "duration_sec": None,
        })
    return rows


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    print(f"Collecting TinyVox files from {TINYVOX}...")
    tv = collect_tinyvox()
    print(f"  TinyVox rows: {len(tv)}")
    pv = collect_providence()
    print(f"  Providence rows: {len(pv)}")

    rows = tv + pv
    print(f"Total raw rows: {len(rows)}")

    # Filter: only collect duration when speakers pass the >=MIN_UTTERANCES bar
    # to avoid sf.info on 65k files for nothing.
    df = pd.DataFrame(rows)
    counts = df["speaker_id"].value_counts()
    keep_speakers = counts[counts >= MIN_UTTERANCES_PER_SPEAKER].index
    df = df[df["speaker_id"].isin(keep_speakers)].copy()
    print(f"After >={MIN_UTTERANCES_PER_SPEAKER}-utterance filter: {len(df)} rows, "
          f"{df['speaker_id'].nunique()} speakers")

    # Sample down to a manageable count per speaker (cap at 200 utterances
    # per speaker to keep total ~50-100k for AAM-Softmax fine-tune).
    df = df.groupby("speaker_id", group_keys=False).apply(
        lambda g: g.sample(n=min(len(g), 200), random_state=42)
    )
    print(f"After per-speaker cap (200): {len(df)} rows, {df['speaker_id'].nunique()} speakers")

    # Write a slim manifest first; durations omitted to keep this CPU-cheap.
    # The trainer can compute durations on first epoch via SoundFile.info.
    df_out = df[["audio_path", "speaker_id", "source", "duration_sec"]].copy()
    df_out.to_csv(OUT, index=False)

    print(f"Wrote {OUT}")
    print(f"  rows: {len(df_out)}")
    print(f"  speakers: {df_out['speaker_id'].nunique()}")
    print(f"  source breakdown: {df_out['source'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
