"""
Extract clean child speech segments from labeled datasets for synthesis training.

Reads manifest.csv for each labeled dataset, loads ground-truth RTTMs, extracts
KCHI/CHI segments (excluding overlap), resamples to 16kHz mono, and writes WAVs to
synthesis/data/{age_group}/. Logs skipped segments to synthesis/data/extraction_log.csv.

Usage:
    python synthesis/scripts/extract_segments.py --age-group 12_16m
    python synthesis/scripts/extract_segments.py --age-group 34_38m
    python synthesis/scripts/extract_segments.py --age-group all
"""

import argparse
import csv
import os
import sys
from pathlib import Path

import pandas as pd
import torchaudio

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
MANIFEST_DIRS = [
    REPO_ROOT / "playlogue",
    REPO_ROOT / "providence",
    REPO_ROOT / "seedlings",
]
OUTPUT_BASE = REPO_ROOT / "synthesis" / "data"
LOG_PATH = OUTPUT_BASE / "extraction_log.csv"

CHILD_LABELS = {"KCHI", "CHI", "chi", "child", "c"}
OVERLAP_LABELS = {"OVL", "overlap", "OVERLAP"}

MIN_DURATION_SEC = 0.10  # 100ms minimum for synthesis training
SAMPLE_RATE = 16000


def parse_rttm(rttm_path: str):
    segments = []
    if not os.path.exists(rttm_path):
        return segments
    with open(rttm_path) as f:
        for line in f:
            line = line.strip()
            if not line or not line.startswith("SPEAKER"):
                continue
            parts = line.split()
            if len(parts) < 8:
                continue
            start = float(parts[3])
            dur = float(parts[4])
            label = parts[7]
            segments.append({"start": start, "end": start + dur, "label": label})
    return segments


def has_overlap(start: float, end: float, segments) -> bool:
    for s in segments:
        if s["label"] in OVERLAP_LABELS:
            if s["start"] < end and s["end"] > start:
                return True
    return False


def load_audio(path: str):
    wav, sr = torchaudio.load(path)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != SAMPLE_RATE:
        wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE)
    return wav.squeeze(0)


def extract_for_manifest(manifest_path: Path, age_group_filter: str, out_dirs: dict, log_rows: list):
    df = pd.read_csv(manifest_path)
    required_cols = {"path", "age_group", "has_rttm", "rttm_path", "recording_id"}
    missing = required_cols - set(df.columns)
    if missing:
        print(f"  [SKIP] {manifest_path.parent.name}: missing columns {missing}")
        return 0

    if age_group_filter != "all":
        df = df[df["age_group"] == age_group_filter]

    n_extracted = 0
    for _, row in df.iterrows():
        if not row.get("has_rttm", False):
            continue
        rttm_path = str(row["rttm_path"])
        audio_path = str(row["path"])
        recording_id = str(row["recording_id"])
        age_group = str(row["age_group"])

        if age_group not in out_dirs:
            continue
        if not os.path.exists(audio_path):
            log_rows.append({"recording_id": recording_id, "reason": "audio_missing",
                             "start": "", "end": "", "label": ""})
            continue

        segments = parse_rttm(rttm_path)
        if not segments:
            continue

        try:
            wav = load_audio(audio_path)
        except Exception as e:
            log_rows.append({"recording_id": recording_id, "reason": f"load_error:{e}",
                             "start": "", "end": "", "label": ""})
            continue

        total_dur = wav.numel() / SAMPLE_RATE
        for seg in segments:
            label = seg["label"]
            if label not in CHILD_LABELS:
                continue
            start, end = seg["start"], seg["end"]
            dur = end - start
            if dur < MIN_DURATION_SEC:
                log_rows.append({"recording_id": recording_id, "reason": "too_short",
                                 "start": start, "end": end, "label": label})
                continue
            if has_overlap(start, end, segments):
                log_rows.append({"recording_id": recording_id, "reason": "overlap",
                                 "start": start, "end": end, "label": label})
                continue

            s_idx = max(0, int(round(start * SAMPLE_RATE)))
            e_idx = min(wav.numel(), int(round(end * SAMPLE_RATE)))
            if e_idx <= s_idx:
                continue

            clip = wav[s_idx:e_idx]
            out_name = f"{recording_id}_{start:.3f}.wav"
            out_path = out_dirs[age_group] / out_name
            torchaudio.save(str(out_path), clip.unsqueeze(0), SAMPLE_RATE)
            n_extracted += 1

    return n_extracted


def main():
    parser = argparse.ArgumentParser(description="Extract child speech segments for synthesis.")
    parser.add_argument("--age-group", default="all",
                        choices=["all", "12_16m", "34_38m"])
    parser.add_argument("--output-dir", default="",
                        help="Override synthesis/data/ output directory.")
    parser.add_argument("--min-duration", type=float, default=MIN_DURATION_SEC)
    args = parser.parse_args()

    out_base = Path(args.output_dir) if args.output_dir else OUTPUT_BASE
    age_groups = ["12_16m", "34_38m"] if args.age_group == "all" else [args.age_group]
    out_dirs = {}
    for ag in age_groups:
        d = out_base / ag
        d.mkdir(parents=True, exist_ok=True)
        out_dirs[ag] = d

    log_rows = []
    total = 0

    for manifest_dir in MANIFEST_DIRS:
        manifest_path = manifest_dir / "manifest.csv"
        if not manifest_path.exists():
            print(f"[SKIP] No manifest at {manifest_path}")
            continue
        print(f"Processing {manifest_path.parent.name}...")
        n = extract_for_manifest(manifest_path, args.age_group, out_dirs, log_rows)
        print(f"  Extracted {n} segments")
        total += n

    # Write extraction log
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["recording_id", "reason", "start", "end", "label"])
        writer.writeheader()
        writer.writerows(log_rows)

    print(f"\nTotal extracted: {total} segments")
    print(f"Skipped/logged: {len(log_rows)} (see {LOG_PATH})")

    if total == 0:
        print("ERROR: No segments extracted. Check manifests and RTTM paths.", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
