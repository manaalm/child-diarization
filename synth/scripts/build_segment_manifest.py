#!/usr/bin/env python3
"""
Build a segment manifest CSV from Providence RTTMs, TinyVox, and (optionally)
LibriSpeech.

The YYMMDD session IDs in Providence RTTM filenames and TinyVox filenames
encode the child's age at recording time (YY years, MM months, DD days), not a
calendar date.  This is used to infer the age_band for each segment.

Test-set speakers (from --exclude-speakers-csv) are marked
usable_for_training=false to prevent data leakage.

TinyVox filename format:
    phon_{lang_family}_{corpus}_{speaker}_{session}_{start_ms}_{end_ms}.wav
  e.g.: phon_Eng-NA_Providence_Alex_010427_00268035_00272151.wav
Only Eng-NA files are included by default (--tinyvox-lang-filter).

Usage:
    python synth/scripts/build_segment_manifest.py \\
      --providence-dir        providence/ \\
      --providence-rttm-dir   providence/rttm/ \\
      --tinyvox-dir           data/tinyvox/ \\
      --librispeech-dir       /path/to/LibriSpeech/train-clean-100/ \\
      --exclude-speakers-csv  whisper-modeling/seen_child_splits/test.csv \\
      --output                synth_results/manifests/segment_manifest.csv \\
      --min-duration-sec      0.3 \\
      --quality-threshold     0.4
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Age bands defined by the spec (inclusive lower, exclusive upper).
_AGE_BANDS = {
    "14_18_months": (14.0, 18.99),
    "34_38_months": (34.0, 38.99),
}
_AGE_BAND_OTHER = "other"


def _parse_age_months(session_id: str) -> float:
    """Parse a YYMMDD session ID string to age in months.

    Providence session IDs encode the child's age at recording:
    YY = years, MM = months, DD = days (zero-padded).
    Optional trailing 'a'/'b' suffixes are stripped.
    """
    s = re.sub(r"[ab]$", "", str(session_id).strip())
    if len(s) == 6 and s.isdigit():
        yy = int(s[0:2])
        mm = int(s[2:4])
        dd = int(s[4:6])
        return yy * 12.0 + mm + dd / 30.44
    return float("nan")


def _age_to_band(age_months: float) -> str:
    if np.isnan(age_months):
        return _AGE_BAND_OTHER
    for band, (lo, hi) in _AGE_BANDS.items():
        if lo <= age_months <= hi:
            return band
    return _AGE_BAND_OTHER


def _quality_score(
    audio_path: str, start_sec: float, dur_sec: float
) -> float:
    """Compute a simple quality proxy for a segment.

    Uses: duration score (0.4–1.0 range), RMS energy, and silence ratio.
    Falls back to duration-only if audio cannot be loaded.
    """
    # Duration score: 1.0 for ≥ 1s, scaled linearly below
    dur_score = min(1.0, dur_sec / 1.0)

    path = Path(audio_path)
    if not path.exists():
        return dur_score

    try:
        info = sf.info(str(path))
        sr = info.samplerate
        start_frame = int(start_sec * sr)
        n_frames = int(dur_sec * sr)
        wav, _ = sf.read(
            str(path),
            start=start_frame,
            frames=n_frames,
            dtype="float32",
            always_2d=False,
        )
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        if len(wav) == 0:
            return dur_score

        rms = float(np.sqrt(np.mean(wav ** 2)))
        rms_score = min(1.0, rms / 0.05)   # normalise: 0.05 RMS = 1.0

        silence_threshold = 0.01
        silence_ratio = float(np.mean(np.abs(wav) < silence_threshold))
        silence_score = 1.0 - silence_ratio

        return 0.4 * dur_score + 0.4 * rms_score + 0.2 * silence_score

    except Exception:
        return dur_score


_ADULT_LABELS = frozenset({
    "MOT", "FAT", "ADU", "AD1", "AD2", "OPE", "OP1",
    "BRO", "SIS", "UNC", "GRA", "GRN", "GR1", "GR2",
    "FRI", "VIS", "VI1",
})


def _parse_providence_rttm(
    rttm_path: str, recording_id: str, include_adults: bool = False
) -> tuple:
    """Parse a Providence RTTM and return (child_segs, adult_segs) lists.

    Parameters
    ----------
    rttm_path : str
        Path to the RTTM file.
    recording_id : str
        Recording identifier.
    include_adults : bool
        If True, also return adult-speaker segments.

    Returns
    -------
    tuple of (child_segments, adult_segments)
        Each element is a list of dicts with
        ``start_time_sec``, ``end_time_sec``, ``duration_sec``,
        and (for adults) ``speaker_label``.
    """
    child_segs = []
    adult_segs = []
    try:
        with open(rttm_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 9:
                    continue
                if parts[0] != "SPEAKER":
                    continue
                label = parts[7]
                start = float(parts[3])
                dur = float(parts[4])
                if dur <= 0:
                    continue
                seg = {
                    "start_time_sec": start,
                    "end_time_sec": start + dur,
                    "duration_sec": dur,
                }
                if label == "CHI":
                    child_segs.append(seg)
                elif include_adults and label in _ADULT_LABELS:
                    seg["speaker_label"] = label
                    adult_segs.append(seg)
    except Exception as e:
        print(f"  [WARN] Could not parse RTTM {rttm_path}: {e}", file=sys.stderr)

    return child_segs, adult_segs


def _load_exclude_speakers(exclude_csv: str) -> set:
    """Return the set of child_id values from the exclude CSV."""
    if not exclude_csv or not Path(exclude_csv).exists():
        return set()
    df = pd.read_csv(exclude_csv, low_memory=False)
    if "child_id" not in df.columns:
        print(f"  [WARN] --exclude-speakers-csv has no 'child_id' column; "
              "no speakers excluded.", file=sys.stderr)
        return set()
    return set(df["child_id"].dropna().astype(str))


def _scan_librispeech(librispeech_dir: str, min_dur: float) -> list:
    """Yield adult segment rows from a LibriSpeech directory.

    Expects the standard LibriSpeech layout:
        {speaker_id}/{chapter_id}/{speaker_id}-{chapter_id}-{utt_id}.flac
    """
    rows = []
    lib_root = Path(librispeech_dir)
    if not lib_root.exists():
        print(f"  [WARN] LibriSpeech dir not found: {lib_root}", file=sys.stderr)
        return rows

    flac_files = list(lib_root.rglob("*.flac"))
    print(f"  Scanning {len(flac_files)} LibriSpeech .flac files …")

    for flac_path in flac_files:
        try:
            info = sf.info(str(flac_path))
            dur = info.duration
        except Exception:
            continue

        if dur < min_dur:
            continue

        # Extract speaker_id from directory structure
        parts = flac_path.parts
        # Pattern: .../{speaker_id}/{chapter_id}/{file}.flac
        speaker_id = parts[-3] if len(parts) >= 3 else str(flac_path.stem)
        recording_id = str(flac_path.stem)
        seg_id = f"librispeech_{recording_id}"

        rows.append(
            {
                "segment_id": seg_id,
                "source_dataset": "librispeech",
                "source_recording_id": recording_id,
                "speaker_id": speaker_id,
                "speaker_role": "adult",
                "age_months": None,
                "age_band": "adult",
                "start_time_sec": 0.0,
                "end_time_sec": dur,
                "duration_sec": dur,
                "audio_path": str(flac_path.resolve()),
                "sample_rate": int(info.samplerate),
                "transcript": "",
                "phonetic_transcript": "",
                "vocalization_type": "speech",
                "quality_score": min(1.0, dur / 2.0),
                "split": "train",
                "usable_for_training": True,
            }
        )

    return rows


def _scan_playlogue(
    playlogue_audio_dir: str,
    playlogue_rttm_dir: str,
    min_dur: float,
    exclude_speakers: set,
) -> list:
    """Scan Playlogue RTTM + audio, return CHI + ADULT segment rows.

    Playlogue RTTM labels: CHI / ADULT / OVL.  We map CHI→target_child,
    ADULT→adult, skip OVL (overlap regions, not single-speaker segments).

    Audio filenames may differ from RTTM stems in case (e.g.
    ``cameron_AAE_...mp3`` vs ``cameron_aae_...rttm``); we match
    case-insensitively.

    Recording IDs of the form ``<child>_<corpus>_<...>`` give the speaker_id
    as the first ``_``-delimited token (cameron, ew, gleason, vh, ...).
    """
    rows = []
    rttm_root = Path(playlogue_rttm_dir)
    audio_root = Path(playlogue_audio_dir)

    if not rttm_root.exists():
        print(f"  [WARN] Playlogue RTTM dir not found: {rttm_root}", file=sys.stderr)
        return rows
    if not audio_root.exists():
        print(f"  [WARN] Playlogue audio dir not found: {audio_root}", file=sys.stderr)
        return rows

    rttm_files = list(rttm_root.glob("*.rttm"))
    print(f"  Scanning {len(rttm_files)} Playlogue .rttm files …")

    # Build lowercase-stem → audio path index
    audio_index = {}
    for ext in ("*.mp3", "*.wav", "*.flac"):
        for af in audio_root.glob(ext):
            audio_index[af.stem.lower()] = af

    n_chi = n_adult = n_skip_no_audio = n_excluded = 0

    for rttm_path in rttm_files:
        recording_id = rttm_path.stem
        speaker_prefix = recording_id.split("_")[0] if "_" in recording_id else recording_id

        audio_path = audio_index.get(recording_id.lower())
        if audio_path is None:
            n_skip_no_audio += 1
            continue

        is_excluded = speaker_prefix in exclude_speakers
        split = "test" if is_excluded else "train"
        if is_excluded:
            n_excluded += 1
        usable = not is_excluded

        try:
            with open(rttm_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 9 or parts[0] != "SPEAKER":
                        continue
                    label = parts[7]
                    start = float(parts[3])
                    dur = float(parts[4])
                    if dur < min_dur:
                        continue

                    if label == "CHI":
                        role = "target_child"
                        ds = "playlogue"
                        voc = "speech"
                        speaker_id = speaker_prefix
                        n_chi += 1
                    elif label == "ADULT":
                        role = "adult"
                        ds = "playlogue_adults"
                        voc = "speech"
                        # Disambiguate adult speakers by recording (true ID is unknown
                        # in the simplified Playlogue label set).
                        speaker_id = f"{speaker_prefix}_ADULT_{recording_id}"
                        n_adult += 1
                    else:
                        # Skip OVL and any other labels
                        continue

                    start_ms = int(start * 1000)
                    end_ms = int((start + dur) * 1000)
                    # Emit one row per target age band so the segment is selectable
                    # under either default config (14-18 mo or 34-38 mo). Playlogue
                    # CHILDES recordings span a wider age range than Providence; we
                    # don't have reliable per-recording ages, so we make these
                    # segments available across both bands rather than dropping them.
                    if role == "target_child":
                        emitted_bands = ["14_18_months", "34_38_months"]
                    else:
                        emitted_bands = ["adult"]
                    for band in emitted_bands:
                        band_suffix = f"_{band}" if role == "target_child" else ""
                        seg_id = (
                            f"playlogue_{label}_{recording_id}"
                            f"_{start_ms}_{end_ms}{band_suffix}"
                        )
                        rows.append(
                            {
                                "segment_id": seg_id,
                                "source_dataset": ds,
                                "source_recording_id": recording_id,
                                "speaker_id": speaker_id,
                                "speaker_role": role,
                                "age_months": None,
                                "age_band": band,
                                "start_time_sec": start,
                                "end_time_sec": start + dur,
                                "duration_sec": dur,
                                "audio_path": str(audio_path.resolve()),
                                "sample_rate": 16000,
                                "transcript": "",
                                "phonetic_transcript": "",
                                "vocalization_type": voc,
                                "quality_score": min(1.0, dur / 1.0),
                                "split": split,
                                "usable_for_training": usable,
                            }
                        )
        except Exception as e:
            print(f"  [WARN] Could not parse Playlogue RTTM {rttm_path}: {e}", file=sys.stderr)

    print(
        f"  Playlogue: {n_chi} CHI + {n_adult} ADULT segments "
        f"({n_skip_no_audio} RTTMs skipped — no matching audio; "
        f"{n_excluded} marked test/excluded by --exclude-speakers-csv)."
    )
    return rows


def _scan_tinyvox(
    tinyvox_dir: str,
    min_dur: float,
    exclude_speakers: set,
    language_filter: str = "Eng-NA",
) -> list:
    """Scan TinyVox pre-segmented phoneme WAVs and return segment rows.

    Filename format:
        phon_{lang_family}_{corpus}_{speaker}_{session}_{start_ms}_{end_ms}.wav
    where session is YYMMDD (same encoding as Providence).

    Each WAV is already a segment; start_time_sec is always 0.0 and the full
    file duration is the segment duration.
    """
    rows = []
    tv_root = Path(tinyvox_dir) / "audio"
    if not tv_root.exists():
        print(f"  [WARN] TinyVox audio dir not found: {tv_root}", file=sys.stderr)
        return rows

    wav_files = list(tv_root.glob("phon_*.wav"))
    print(f"  Scanning {len(wav_files)} TinyVox .wav files …")

    skipped_lang = 0
    skipped_dur = 0
    n_excluded = 0
    n_included = 0

    for wav_path in wav_files:
        stem = wav_path.stem
        parts = stem.split("_")
        # Minimum: phon, lang, corpus, speaker, session, start_ms, end_ms = 7 parts
        if len(parts) < 7:
            continue

        try:
            end_ms = int(parts[-1])
            start_ms = int(parts[-2])
            session_id = parts[-3]
            speaker = parts[-4]
            # corpus = parts[-5] — kept for seg_id construction
            corpus = parts[-5]
            # Language family occupies parts[1:-5]; join back in case it contains "_"
            lang_family = "_".join(parts[1:-5])
        except (ValueError, IndexError):
            continue

        if language_filter and lang_family != language_filter:
            skipped_lang += 1
            continue

        dur = (end_ms - start_ms) / 1000.0
        if dur < min_dur:
            skipped_dur += 1
            continue

        age_months = _parse_age_months(session_id)
        age_band = _age_to_band(age_months)

        is_excluded = speaker in exclude_speakers
        split = "test" if is_excluded else "train"
        usable = not is_excluded
        if is_excluded:
            n_excluded += 1

        seg_id = f"tinyvox_{corpus}_{speaker}_{session_id}_{start_ms}_{end_ms}"

        rows.append(
            {
                "segment_id": seg_id,
                "source_dataset": "tinyvox",
                "source_recording_id": f"{corpus}_{speaker}_{session_id}",
                "speaker_id": speaker,
                "speaker_role": "target_child",
                "age_months": round(age_months, 2)
                if not np.isnan(age_months)
                else None,
                "age_band": age_band,
                "start_time_sec": 0.0,
                "end_time_sec": dur,
                "duration_sec": dur,
                "audio_path": str(wav_path.resolve()),
                "sample_rate": 16000,
                "transcript": "",
                "phonetic_transcript": "",
                "vocalization_type": "speech",
                "quality_score": min(1.0, dur / 1.0),
                "split": split,
                "usable_for_training": usable,
            }
        )
        n_included += 1

    print(
        f"  TinyVox: {n_included} segments included "
        f"({skipped_lang} skipped by lang filter '{language_filter}', "
        f"{skipped_dur} too short, "
        f"{n_excluded} marked test/excluded)."
    )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build segment manifest from Providence RTTMs, TinyVox, and LibriSpeech."
    )
    parser.add_argument(
        "--providence-dir",
        default=None,
        help="Root Providence directory containing audio/ and manifest.csv.",
    )
    parser.add_argument(
        "--providence-rttm-dir",
        default=None,
        help="Directory containing Providence RTTM files.",
    )
    parser.add_argument(
        "--tinyvox-dir",
        default=None,
        help="Root TinyVox directory (expects audio/ subdirectory with phon_*.wav files).",
    )
    parser.add_argument(
        "--tinyvox-lang-filter",
        default="Eng-NA",
        help="Language family prefix to include from TinyVox (default: Eng-NA). "
             "Pass '' to include all languages.",
    )
    parser.add_argument(
        "--librispeech-dir",
        default=None,
        help="Path to LibriSpeech train-clean-100 directory.",
    )
    parser.add_argument(
        "--playlogue-dir",
        default=None,
        help="Path to Playlogue audio dir (e.g. playlogue/audio/).",
    )
    parser.add_argument(
        "--playlogue-rttm-dir",
        default=None,
        help="Path to Playlogue RTTM dir (e.g. playlogue/rttm/).",
    )
    parser.add_argument(
        "--exclude-speakers-csv",
        required=True,
        help="CSV with child_id column; matching speakers become usable_for_training=false. "
             "REQUIRED to prevent test-child speech from leaking into training segments. "
             "Pass whisper-modeling/seen_child_splits/test.csv for the default project setup.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output CSV path.",
    )
    parser.add_argument(
        "--min-duration-sec",
        type=float,
        default=0.3,
        help="Minimum segment duration to include (default 0.3).",
    )
    parser.add_argument(
        "--quality-threshold",
        type=float,
        default=0.4,
        help="Quality score threshold; below this → usable_for_training=false.",
    )
    parser.add_argument(
        "--skip-quality",
        action="store_true",
        help="Skip audio-based quality scoring (use duration-only). "
             "Reduces runtime from hours to minutes for large corpora.",
    )
    args = parser.parse_args()

    rows: list = []
    exclude_speakers = _load_exclude_speakers(args.exclude_speakers_csv)
    if exclude_speakers:
        print(f"Excluding {len(exclude_speakers)} speakers from test CSV.")

    # ---- Providence ----
    if args.providence_dir:
        prov_root = Path(args.providence_dir)
        prov_manifest_path = prov_root / "manifest.csv"

        if not prov_manifest_path.exists():
            print(f"  [WARN] Providence manifest not found: {prov_manifest_path}",
                  file=sys.stderr)
        else:
            prov_manifest = pd.read_csv(prov_manifest_path)
            has_rttm = prov_manifest[prov_manifest.get("has_rttm", False) == True] \
                       if "has_rttm" in prov_manifest.columns \
                       else prov_manifest[prov_manifest["has_rttm"].astype(str) == "True"]
            print(f"Providence: {len(has_rttm)} recordings with RTTM.")

            n_child_segs = 0
            for _, rec in has_rttm.iterrows():
                session_id = str(rec.get("session_id", "")).strip()
                child_id = str(rec.get("child_id", "")).strip()
                audio_path = str(rec.get("path", "")).strip()
                rttm_path = str(rec.get("rttm_path", "")).strip()

                if not rttm_path or not Path(rttm_path).exists():
                    # Try to find RTTM from rttm_dir
                    if args.providence_rttm_dir:
                        rttm_candidates = list(
                            Path(args.providence_rttm_dir).glob(
                                f"*{session_id}*.rttm"
                            )
                        )
                        if rttm_candidates:
                            rttm_path = str(rttm_candidates[0])

                if not rttm_path or not Path(rttm_path).exists():
                    continue

                age_months = _parse_age_months(session_id)
                age_band = _age_to_band(age_months)

                # Determine split and usability
                is_excluded = child_id in exclude_speakers
                split = "test" if is_excluded else "train"
                usable = not is_excluded

                chi_segs, adult_segs = _parse_providence_rttm(
                    rttm_path, session_id, include_adults=True
                )

                for seg in chi_segs:
                    dur = seg["duration_sec"]
                    if dur < args.min_duration_sec:
                        continue

                    # Encode as YYMMDDSSS (start in milliseconds)
                    start_ms = int(seg["start_time_sec"] * 1000)
                    end_ms = int(seg["end_time_sec"] * 1000)
                    seg_id = (
                        f"providence_{child_id}_{session_id}_{start_ms}_{end_ms}"
                    )

                    if args.skip_quality:
                        qscore = min(1.0, dur / 1.0)
                    else:
                        qscore = _quality_score(audio_path, seg["start_time_sec"], dur)
                    seg_usable = usable and (qscore >= args.quality_threshold)

                    rows.append(
                        {
                            "segment_id": seg_id,
                            "source_dataset": "providence",
                            "source_recording_id": session_id,
                            "speaker_id": child_id,
                            "speaker_role": "target_child",
                            "age_months": round(age_months, 2)
                            if not np.isnan(age_months)
                            else None,
                            "age_band": age_band,
                            "start_time_sec": seg["start_time_sec"],
                            "end_time_sec": seg["end_time_sec"],
                            "duration_sec": dur,
                            "audio_path": audio_path,
                            "sample_rate": 16000,
                            "transcript": "",
                            "phonetic_transcript": "",
                            "vocalization_type": "babble",
                            "quality_score": round(qscore, 4),
                            "split": split,
                            "usable_for_training": seg_usable,
                        }
                    )
                    n_child_segs += 1

                # Add adult segments (MOT/FAT/etc) as training-usable adult pool
                n_adult_segs_rec = 0
                for seg in adult_segs:
                    dur = seg["duration_sec"]
                    if dur < args.min_duration_sec:
                        continue
                    start_ms = int(seg["start_time_sec"] * 1000)
                    end_ms = int(seg["end_time_sec"] * 1000)
                    spk_label = seg.get("speaker_label", "ADU")
                    seg_id = (
                        f"prov_adult_{child_id}_{session_id}_{spk_label}_{start_ms}_{end_ms}"
                    )
                    rows.append(
                        {
                            "segment_id": seg_id,
                            "source_dataset": "providence_adults",
                            "source_recording_id": session_id,
                            "speaker_id": f"{child_id}_{spk_label}",
                            "speaker_role": "adult",
                            "age_months": None,
                            "age_band": "adult",
                            "start_time_sec": seg["start_time_sec"],
                            "end_time_sec": seg["end_time_sec"],
                            "duration_sec": dur,
                            "audio_path": audio_path,
                            "sample_rate": 16000,
                            "transcript": "",
                            "phonetic_transcript": "",
                            "vocalization_type": "speech",
                            "quality_score": min(1.0, dur / 1.0),
                            "split": split,
                            "usable_for_training": True,
                        }
                    )
                    n_adult_segs_rec += 1

            print(f"  Extracted {n_child_segs} CHI segments from Providence.")
            n_adult_total = sum(1 for r in rows if r.get("speaker_role") == "adult")
            print(f"  Extracted {n_adult_total} adult segments from Providence.")

    # ---- TinyVox ----
    if args.tinyvox_dir:
        tv_rows = _scan_tinyvox(
            args.tinyvox_dir,
            args.min_duration_sec,
            exclude_speakers,
            language_filter=args.tinyvox_lang_filter,
        )
        print(f"TinyVox: {len(tv_rows)} child segments.")
        rows.extend(tv_rows)

    # ---- LibriSpeech ----
    if args.librispeech_dir:
        lib_rows = _scan_librispeech(args.librispeech_dir, args.min_duration_sec)
        print(f"LibriSpeech: {len(lib_rows)} adult segments.")
        rows.extend(lib_rows)

    # ---- Playlogue ----
    if args.playlogue_dir and args.playlogue_rttm_dir:
        pl_rows = _scan_playlogue(
            args.playlogue_dir,
            args.playlogue_rttm_dir,
            args.min_duration_sec,
            exclude_speakers,
        )
        print(f"Playlogue: {len(pl_rows)} segments (CHI + ADULT).")
        rows.extend(pl_rows)

    if not rows:
        print("No segments extracted.  Check input paths.", file=sys.stderr)
        sys.exit(1)

    df = pd.DataFrame(rows)

    # ---- Split integrity summary ----
    train_speakers = set(df.loc[df["split"] == "train", "speaker_id"].dropna())
    test_speakers = set(df.loc[df["split"] == "test", "speaker_id"].dropna())
    overlap = train_speakers & test_speakers
    print(
        f"\nSplit integrity:"
        f"  {len(train_speakers)} train speakers, "
        f"{len(test_speakers)} test speakers, "
        f"{len(overlap)} in both (should be 0)."
    )
    if overlap:
        print(f"  [WARN] Overlap: {sorted(overlap)[:5]} …", file=sys.stderr)

    # ---- Age band summary ----
    print("\nAge band distribution:")
    print(df["age_band"].value_counts().to_string())

    print(f"\nDataset summary:")
    print(df.groupby(["source_dataset", "speaker_role", "split"])
          .size()
          .to_string())

    # ---- Write output ----
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"\nWrote {len(df)} rows to {out_path}")


if __name__ == "__main__":
    main()
