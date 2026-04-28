"""Extract hard-negative clips from Playlogue and Providence RTTM files.

Finds 30-second windows where CHI is silent but non-silence speech is present
(at least `min_activity_sec` of non-CHI RTTM segments). Outputs a CSV with
columns compatible with the seen_child_splits format so MIL training can append
these rows directly.

Output columns:
    audio_path, start_sec, end_sec, label (=0), child_id, timepoint_norm, source

Usage:
    python mil/scripts/extract_hard_negatives.py \\
        --output synth_results/manifests/hard_negatives_manifest.csv \\
        [--window-sec 30] [--stride-sec 15] [--min-activity-sec 3]
        [--max-per-file 20] [--seed 42]
"""

import argparse
import glob
import os
import random
import sys

import pandas as pd

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PLAYLOGUE_RTTM_DIR  = os.path.join(_REPO, "playlogue", "rttm")
PLAYLOGUE_AUDIO_DIR = os.path.join(_REPO, "playlogue", "trimmed")
PROVIDENCE_RTTM_DIR  = os.path.join(_REPO, "providence", "rttm")
PROVIDENCE_AUDIO_DIR = os.path.join(_REPO, "providence", "audio")

CHI_LABELS = {"CHI"}  # speaker labels that count as target-child speech


def parse_rttm(path: str) -> list[dict]:
    """Return list of {start, end, speaker} dicts from an RTTM file."""
    segments = []
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 9 or not parts[0].startswith("SPEAKER"):
                continue
            try:
                start = float(parts[3])
                dur   = float(parts[4])
                spk   = parts[7]
            except ValueError:
                continue
            segments.append({"start": start, "end": start + dur, "speaker": spk})
    return segments


def chi_activity_in_window(segments: list[dict], win_start: float, win_end: float) -> float:
    """Return total seconds of CHI speech overlapping [win_start, win_end]."""
    total = 0.0
    for s in segments:
        if s["speaker"] not in CHI_LABELS:
            continue
        overlap_start = max(s["start"], win_start)
        overlap_end   = min(s["end"], win_end)
        if overlap_end > overlap_start:
            total += overlap_end - overlap_start
    return total


def non_silence_in_window(segments: list[dict], win_start: float, win_end: float) -> float:
    """Return total seconds of any speech (CHI or not) in [win_start, win_end]."""
    total = 0.0
    for s in segments:
        overlap_start = max(s["start"], win_start)
        overlap_end   = min(s["end"], win_end)
        if overlap_end > overlap_start:
            total += overlap_end - overlap_start
    return total


def file_duration_sec(path: str) -> float:
    """Return audio file duration in seconds using soundfile (no torch needed)."""
    import soundfile as sf
    info = sf.info(path)
    return info.duration


def find_audio(stem: str, audio_dir: str) -> str | None:
    for ext in [".wav", ".mp3", ".flac"]:
        p = os.path.join(audio_dir, stem + ext)
        if os.path.exists(p):
            return p
    return None


def extract_negatives_from_file(
    rttm_path: str,
    audio_path: str,
    source: str,
    child_id: str,
    window_sec: float,
    stride_sec: float,
    min_activity_sec: float,
    max_per_file: int,
    rng: random.Random,
) -> list[dict]:
    segments = parse_rttm(rttm_path)
    if not segments:
        return []

    try:
        file_dur = file_duration_sec(audio_path)
    except Exception:
        return []

    candidates = []
    win_start = 0.0
    while win_start + window_sec <= file_dur:
        win_end = win_start + window_sec
        chi_sec = chi_activity_in_window(segments, win_start, win_end)
        if chi_sec == 0.0:
            activity = non_silence_in_window(segments, win_start, win_end)
            if activity >= min_activity_sec:
                candidates.append({"start_sec": win_start, "end_sec": win_end})
        win_start += stride_sec

    if not candidates:
        return []

    # Cap per file and shuffle so we don't always take the first N
    rng.shuffle(candidates)
    selected = candidates[:max_per_file]

    rows = []
    for c in selected:
        rows.append({
            "audio_path":     audio_path,
            "start_sec":      c["start_sec"],
            "end_sec":        c["end_sec"],
            "label":          0,
            "child_id":       child_id,
            "timepoint_norm": "unknown",
            "source":         source,
        })
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=os.path.join(
        _REPO, "synth_results", "manifests", "hard_negatives_manifest.csv"))
    parser.add_argument("--window-sec",      type=float, default=30.0)
    parser.add_argument("--stride-sec",      type=float, default=15.0,
                        help="Stride between candidate windows (use <window to allow overlap)")
    parser.add_argument("--min-activity-sec", type=float, default=3.0,
                        help="Min non-CHI speech in window to be considered hard (not silence)")
    parser.add_argument("--max-per-file",    type=int,   default=20,
                        help="Max windows sampled per audio file")
    parser.add_argument("--seed",            type=int,   default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    rows = []

    # ── Playlogue ────────────────────────────────────────────────────────
    play_rttms = sorted(glob.glob(os.path.join(PLAYLOGUE_RTTM_DIR, "*.rttm")))
    print(f"Playlogue: {len(play_rttms)} RTTM files")
    play_matched = 0
    for rttm_path in play_rttms:
        stem = os.path.basename(rttm_path).replace(".rttm", "")
        audio = find_audio(stem, PLAYLOGUE_AUDIO_DIR)
        if audio is None:
            continue
        play_matched += 1
        child_id = stem.split("_")[0]  # e.g. "cameron" from "cameron_aae_B3_26_47_PE_S22"
        found = extract_negatives_from_file(
            rttm_path, audio, "playlogue", child_id,
            args.window_sec, args.stride_sec, args.min_activity_sec,
            args.max_per_file, rng,
        )
        rows.extend(found)
    print(f"  Matched audio: {play_matched} | Windows extracted: {sum(1 for r in rows if r['source']=='playlogue')}")

    # ── Providence ───────────────────────────────────────────────────────
    prov_rttms = sorted(glob.glob(os.path.join(PROVIDENCE_RTTM_DIR, "*.rttm")))
    print(f"Providence: {len(prov_rttms)} RTTM files")
    prov_matched = 0; prov_windows = 0
    for rttm_path in prov_rttms:
        stem = os.path.basename(rttm_path).replace(".rttm", "")
        audio = find_audio(stem, PROVIDENCE_AUDIO_DIR)
        if audio is None:
            continue
        prov_matched += 1
        # stem like "alex_010427" → child_id = "alex"
        parts = stem.split("_")
        child_id = parts[0] if len(parts) > 1 and not parts[0].isdigit() else stem
        found = extract_negatives_from_file(
            rttm_path, audio, "providence", child_id,
            args.window_sec, args.stride_sec, args.min_activity_sec,
            args.max_per_file, rng,
        )
        rows.extend(found)
        prov_windows += len(found)
    print(f"  Matched audio: {prov_matched} | Windows extracted: {prov_windows}")

    if not rows:
        print("No windows found — check RTTM/audio paths.")
        sys.exit(1)

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"\nTotal hard negatives: {len(df)}")
    print(f"  playlogue: {(df['source']=='playlogue').sum()}")
    print(f"  providence: {(df['source']=='providence').sum()}")
    print(f"Saved → {args.output}")


if __name__ == "__main__":
    main()
