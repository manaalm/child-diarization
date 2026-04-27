"""
Prepare per-dataset age-annotated manifests for the thesis pipeline.

Outputs manifest.csv to {dataset}/manifest.csv matching AudioRecording schema.

Usage:
    python scripts/prepare_age_manifests.py --dataset playlogue
    python scripts/prepare_age_manifests.py --dataset providence
    python scripts/prepare_age_manifests.py --dataset seedlings
"""

import argparse
import os
import re
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
SPLITS_DIR = REPO_ROOT / "whisper-modeling" / "seen_child_splits"
ANNOT_CSV = Path("/orcd/scratch/bcs/001/sensein/sails/BIDS_data/anotated_processed.csv")

TIMEPOINT_TO_AGE = {
    "14_month": "12_16m",
    "36_month": "34_38m",
}

MANIFEST_COLS = [
    "recording_id", "path", "dataset_name", "child_id", "age_group",
    "session_id", "duration_secs", "split", "has_rttm", "rttm_path",
]


def age_months_to_group(months: float) -> str:
    if 12 <= months <= 16:
        return "12_16m"
    if 34 <= months <= 38:
        return "34_38m"
    if months < 12 or months > 38:
        return "other"
    return "other"


def parse_chat_age(cha_path: Path) -> float | None:
    """Return CHI age in months from a CHAT file, or None if not found."""
    pat = re.compile(r"@ID:\s+\S+\|[^|]+\|CHI\|(\d+);(\d+)\.(\d+)\|")
    try:
        with open(cha_path, encoding="utf-8", errors="ignore") as f:
            for line in f:
                m = pat.search(line)
                if m:
                    years = int(m.group(1))
                    months = int(m.group(2))
                    days = int(m.group(3))
                    return years * 12 + months + days / 30.0
    except OSError:
        pass
    return None


def get_audio_duration(path: str) -> float:
    """Return duration in seconds using soundfile, falling back to 0."""
    try:
        import soundfile as sf
        info = sf.info(path)
        return info.duration
    except Exception:
        pass
    try:
        import torchaudio
        info = torchaudio.info(path)
        return info.num_frames / info.sample_rate
    except Exception:
        return 0.0


def prepare_playlogue() -> pd.DataFrame:
    """Build manifest from BIDS seen_child_splits data."""
    master = SPLITS_DIR / "master_with_split.csv"
    if not master.exists():
        # Fall back to concatenating train/val/test
        dfs = []
        for split in ("train", "val", "test"):
            p = SPLITS_DIR / f"{split}.csv"
            if p.exists():
                df = pd.read_csv(p)
                dfs.append(df)
        if not dfs:
            sys.exit(f"ERROR: No split CSVs found in {SPLITS_DIR}")
        df = pd.concat(dfs, ignore_index=True)
    else:
        df = pd.read_csv(master)

    rows = []
    for _, row in df.iterrows():
        audio_path = str(row["audio_path"])
        child_id = str(row["child_id"])
        timepoint = str(row.get("timepoint_norm", row.get("timepoint", "")))
        age_group = TIMEPOINT_TO_AGE.get(timepoint, "other")
        split = str(row.get("split", "unknown"))

        # Derive session_id from audio filename stem (drop _audio suffix)
        stem = Path(audio_path).stem
        session_id = stem.removesuffix("_audio") if stem.endswith("_audio") else stem

        recording_id = f"playlogue_{child_id}_{session_id}"

        rows.append({
            "recording_id": recording_id,
            "path": audio_path,
            "dataset_name": "playlogue",
            "child_id": child_id,
            "age_group": age_group,
            "session_id": session_id,
            "duration_secs": 0.0,
            "split": split,
            "has_rttm": False,
            "rttm_path": "",
        })

    return pd.DataFrame(rows, columns=MANIFEST_COLS)


def prepare_providence() -> pd.DataFrame:
    """Build manifest from Providence CHAT + audio/RTTM directories."""
    prov_dir = REPO_ROOT / "providence"
    audio_dir = prov_dir / "audio"
    rttm_dir = prov_dir / "rttm"
    cha_dir = prov_dir / "cha"

    if not audio_dir.exists():
        sys.exit(f"ERROR: Providence audio dir not found: {audio_dir}")

    # Build RTTM lookup: stem → path
    rttm_lookup: dict[str, str] = {}
    if rttm_dir.exists():
        for rttm in rttm_dir.glob("*.rttm"):
            rttm_lookup[rttm.stem.lower()] = str(rttm)

    # Build CHAT age lookup: stem → age_months
    chat_ages: dict[str, float] = {}
    if cha_dir.exists():
        for cha in cha_dir.glob("*.cha"):
            age = parse_chat_age(cha)
            if age is not None:
                chat_ages[cha.stem.lower()] = age

    rows = []
    audio_exts = {".wav", ".mp3", ".flac"}
    for audio_file in sorted(audio_dir.iterdir()):
        if audio_file.suffix.lower() not in audio_exts:
            continue

        stem = audio_file.stem
        stem_lower = stem.lower()

        # Derive child_id from filename prefix (up to first underscore or digit run)
        # e.g., "alex_010427" → child "alex", session "010427"
        m = re.match(r"^([a-zA-Z]+)_(.+)$", stem)
        if m:
            child_id = m.group(1).lower()
            session_id = m.group(2)
        else:
            child_id = stem_lower
            session_id = stem_lower

        # Age from CHAT
        age_months = chat_ages.get(stem_lower)
        if age_months is None:
            # Try just the child name prefix
            for cha_stem, age in chat_ages.items():
                if cha_stem.startswith(child_id + "_"):
                    pass  # Use first matching cha file per session
            age_group = "unknown"
        else:
            age_group = age_months_to_group(age_months)

        rttm_path = rttm_lookup.get(stem_lower, "")
        has_rttm = bool(rttm_path)

        recording_id = f"providence_{child_id}_{session_id}"

        rows.append({
            "recording_id": recording_id,
            "path": str(audio_file.resolve()),
            "dataset_name": "providence",
            "child_id": child_id,
            "age_group": age_group,
            "session_id": session_id,
            "duration_secs": 0.0,
            "split": "N/A",
            "has_rttm": has_rttm,
            "rttm_path": rttm_path,
        })

    return pd.DataFrame(rows, columns=MANIFEST_COLS)


def prepare_seedlings() -> pd.DataFrame:
    """Build manifest from Seedlings Databrary export."""
    seed_dir = REPO_ROOT / "seedlings"
    audio_dir = seed_dir / "audio"

    if not audio_dir.exists():
        sys.exit(
            f"ERROR: Seedlings audio dir not found: {audio_dir}\n"
            "Run seedlings_import.py first to download audio via Databrary."
        )

    # Try to load exported metadata
    meta_candidates = [
        seed_dir / "seedlings_metadata.csv",
        seed_dir / "manifest_source.csv",
    ]
    meta_df = None
    for candidate in meta_candidates:
        if candidate.exists():
            meta_df = pd.read_csv(candidate)
            break

    rttm_dir = seed_dir / "rttm"
    rttm_lookup: dict[str, str] = {}
    if rttm_dir.exists():
        for rttm in rttm_dir.glob("*.rttm"):
            rttm_lookup[rttm.stem.lower()] = str(rttm)

    rows = []
    audio_exts = {".wav", ".mp3", ".flac"}
    for audio_file in sorted(audio_dir.iterdir()):
        if audio_file.suffix.lower() not in audio_exts:
            continue

        stem = audio_file.stem
        stem_lower = stem.lower()

        age_group = "unknown"
        child_id = stem_lower
        session_id = stem_lower

        if meta_df is not None:
            # Try to match on filename
            matches = meta_df[meta_df.get("filename", meta_df.get("audio_file", pd.Series())).astype(str).str.contains(stem, case=False, na=False)]
            if not matches.empty:
                row = matches.iloc[0]
                age_months_col = next((c for c in row.index if "age" in c.lower() and "month" in c.lower()), None)
                if age_months_col:
                    try:
                        age_group = age_months_to_group(float(row[age_months_col]))
                    except (ValueError, TypeError):
                        pass
                for cid_col in ("child_id", "subject_id", "participant_id"):
                    if cid_col in row.index and pd.notna(row[cid_col]):
                        child_id = str(row[cid_col])
                        break

        rttm_path = rttm_lookup.get(stem_lower, "")
        has_rttm = bool(rttm_path)
        recording_id = f"seedlings_{child_id}_{session_id}"

        rows.append({
            "recording_id": recording_id,
            "path": str(audio_file.resolve()),
            "dataset_name": "seedlings",
            "child_id": child_id,
            "age_group": age_group,
            "session_id": session_id,
            "duration_secs": 0.0,
            "split": "N/A",
            "has_rttm": has_rttm,
            "rttm_path": rttm_path,
        })

    return pd.DataFrame(rows, columns=MANIFEST_COLS)


def main():
    parser = argparse.ArgumentParser(description="Prepare age-annotated manifests")
    parser.add_argument("--dataset", required=True,
                        choices=["playlogue", "providence", "seedlings"],
                        help="Dataset to prepare manifest for")
    parser.add_argument("--compute-duration", action="store_true",
                        help="Load audio files to compute duration_secs (slow)")
    args = parser.parse_args()

    if args.dataset == "playlogue":
        df = prepare_playlogue()
        out_path = REPO_ROOT / "playlogue" / "manifest.csv"
    elif args.dataset == "providence":
        df = prepare_providence()
        out_path = REPO_ROOT / "providence" / "manifest.csv"
    else:
        df = prepare_seedlings()
        out_path = REPO_ROOT / "seedlings" / "manifest.csv"

    if args.compute_duration:
        print(f"Computing durations for {len(df)} files...")
        df["duration_secs"] = df["path"].apply(get_audio_duration)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df)} records → {out_path}")

    # Summary
    print(f"\nAge group distribution:")
    print(df["age_group"].value_counts().to_string())
    print(f"\nSplit distribution:")
    print(df["split"].value_counts().to_string())
    print(f"\nhas_rttm: {df['has_rttm'].sum()} / {len(df)}")


if __name__ == "__main__":
    main()
