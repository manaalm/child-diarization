"""
Summarize extracted child speech segments per age group.

Reports per-age-group counts (n segments, total hours, mean/std duration).
Exits 1 if any age group has < 500 segments.

Usage:
    python synthesis/scripts/count_segments.py
    python synthesis/scripts/count_segments.py --segments-dir synthesis/data/
"""

import argparse
import os
import sys
from pathlib import Path

import torchaudio

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_SEGMENTS_DIR = REPO_ROOT / "synthesis" / "data"
MIN_SEGMENTS = 500
SAMPLE_RATE = 16000


def get_duration(wav_path: str) -> float:
    info = torchaudio.info(wav_path)
    return info.num_frames / info.sample_rate


def summarize_age_group(age_dir: Path) -> dict:
    wav_files = list(age_dir.glob("*.wav"))
    if not wav_files:
        return {"n_segments": 0, "total_hours": 0.0, "mean_dur_sec": 0.0, "std_dur_sec": 0.0}

    durations = []
    for f in wav_files:
        try:
            durations.append(get_duration(str(f)))
        except Exception:
            continue

    if not durations:
        return {"n_segments": 0, "total_hours": 0.0, "mean_dur_sec": 0.0, "std_dur_sec": 0.0}

    import statistics
    total = sum(durations)
    mean = total / len(durations)
    std = statistics.stdev(durations) if len(durations) > 1 else 0.0
    return {
        "n_segments": len(durations),
        "total_hours": round(total / 3600, 4),
        "mean_dur_sec": round(mean, 3),
        "std_dur_sec": round(std, 3),
    }


def main():
    parser = argparse.ArgumentParser(description="Count synthesis segments per age group.")
    parser.add_argument("--segments-dir", default="",
                        help="Directory containing 12_16m/ and 34_38m/ subdirectories.")
    parser.add_argument("--min-segments", type=int, default=MIN_SEGMENTS)
    args = parser.parse_args()

    base = Path(args.segments_dir) if args.segments_dir else DEFAULT_SEGMENTS_DIR
    age_groups = ["12_16m", "34_38m"]
    any_fail = False

    print(f"{'Age Group':<12} {'N Segments':>10} {'Total Hours':>12} {'Mean Dur':>10} {'Std Dur':>10}")
    print("-" * 60)
    for ag in age_groups:
        ag_dir = base / ag
        if not ag_dir.exists():
            print(f"{ag:<12} {'MISSING DIR':>10}")
            any_fail = True
            continue

        stats = summarize_age_group(ag_dir)
        status = "✓" if stats["n_segments"] >= args.min_segments else "✗ FAIL"
        print(f"{ag:<12} {stats['n_segments']:>10}  {stats['total_hours']:>10.4f}h  "
              f"{stats['mean_dur_sec']:>8.3f}s  {stats['std_dur_sec']:>8.3f}s  {status}")

        if stats["n_segments"] < args.min_segments:
            print(f"  ERROR: {ag} has {stats['n_segments']} segments, need ≥ {args.min_segments}",
                  file=sys.stderr)
            any_fail = True

    if any_fail:
        sys.exit(1)

    print("\nAll age groups meet minimum segment threshold.")


if __name__ == "__main__":
    main()
