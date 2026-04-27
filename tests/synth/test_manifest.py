"""
Unit tests for synth/manifest.py and synth/scripts/build_segment_manifest.py.

Tests:
  1. load_manifest raises ValueError when a speaker_id appears in both
     split=train and split=test rows.
  2. filter_usable returns only usable_for_training=true rows.
  3. build_segment_manifest.py with --exclude-speakers-csv marks matching
     Providence speakers as usable_for_training=false.
"""

import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import soundfile as sf

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from synth.manifest import filter_usable, load_manifest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_valid_manifest_df(**overrides) -> pd.DataFrame:
    """Return a minimal valid manifest DataFrame."""
    rows = [
        dict(
            segment_id="prov_child0_010427_1000_2000",
            source_dataset="providence",
            source_recording_id="010427",
            speaker_id="child_0",
            speaker_role="target_child",
            age_months=16.9,
            age_band="14_18_months",
            start_time_sec=1.0,
            end_time_sec=2.0,
            duration_sec=1.0,
            audio_path="/fake/child_0.wav",
            sample_rate=16000,
            transcript="",
            phonetic_transcript="",
            vocalization_type="babble",
            quality_score=0.8,
            split="train",
            usable_for_training=True,
        ),
        dict(
            segment_id="lib_adult0_0_3000",
            source_dataset="librispeech",
            source_recording_id="adult0",
            speaker_id="adult_0",
            speaker_role="adult",
            age_months=None,
            age_band="adult",
            start_time_sec=0.0,
            end_time_sec=3.0,
            duration_sec=3.0,
            audio_path="/fake/adult_0.flac",
            sample_rate=16000,
            transcript="hello",
            phonetic_transcript="",
            vocalization_type="speech",
            quality_score=0.9,
            split="train",
            usable_for_training=True,
        ),
    ]
    df = pd.DataFrame(rows)
    for key, val in overrides.items():
        df[key] = val
    return df


# ---------------------------------------------------------------------------
# Test 1: load_manifest raises on train/test speaker overlap
# ---------------------------------------------------------------------------

def test_load_manifest_raises_on_train_test_speaker_overlap(tmp_path):
    """load_manifest must raise ValueError if a speaker_id appears in both splits."""
    df = _make_valid_manifest_df()
    # Add a test row for the same speaker_id as a train row
    leak_row = df.iloc[0].to_dict()
    leak_row["split"] = "test"
    leak_row["usable_for_training"] = False
    leak_row["segment_id"] = "leak_seg"
    df = pd.concat([df, pd.DataFrame([leak_row])], ignore_index=True)

    csv_path = tmp_path / "manifest.csv"
    df.to_csv(csv_path, index=False)

    with pytest.raises(ValueError, match="Split integrity violation"):
        load_manifest(str(csv_path))


def test_load_manifest_valid(tmp_path):
    """load_manifest succeeds on a clean manifest."""
    df = _make_valid_manifest_df()
    csv_path = tmp_path / "manifest.csv"
    df.to_csv(csv_path, index=False)
    loaded = load_manifest(str(csv_path))
    assert len(loaded) == len(df)
    assert loaded["usable_for_training"].dtype == bool


# ---------------------------------------------------------------------------
# Test 2: filter_usable returns only usable rows
# ---------------------------------------------------------------------------

def test_filter_usable_returns_only_usable_rows():
    """filter_usable must return only rows where usable_for_training=True."""
    df = _make_valid_manifest_df()
    # Add an unusable row
    bad_row = df.iloc[0].to_dict()
    bad_row["usable_for_training"] = False
    bad_row["segment_id"] = "bad_seg"
    bad_row["speaker_id"] = "child_bad"
    df = pd.concat([df, pd.DataFrame([bad_row])], ignore_index=True)

    result = filter_usable(df)
    assert all(result["usable_for_training"].astype(bool))
    assert "bad_seg" not in result["segment_id"].values


def test_filter_usable_by_age_band():
    """filter_usable with age_band filter returns only matching rows."""
    df = _make_valid_manifest_df()
    # Add a 34_38 row
    other = df.iloc[0].to_dict()
    other["age_band"] = "34_38_months"
    other["segment_id"] = "child_34_38"
    other["speaker_id"] = "child_old"
    df = pd.concat([df, pd.DataFrame([other])], ignore_index=True)

    result = filter_usable(df, age_band="14_18_months")
    assert all(result["age_band"] == "14_18_months")


# ---------------------------------------------------------------------------
# Test 3: build_segment_manifest.py with --exclude-speakers-csv
# ---------------------------------------------------------------------------

def _make_mock_rttm(rttm_path: Path, session_id: str, chi_segs: list) -> None:
    """Write a minimal RTTM file with CHI segments."""
    with open(rttm_path, "w") as f:
        for start, dur in chi_segs:
            f.write(
                f"SPEAKER {session_id} 1 {start:.3f} {dur:.3f}"
                " <NA> <NA> CHI <NA> <NA>\n"
            )


def _make_mock_providence_manifest(
    prov_dir: Path, rttm_dir: Path, child_id: str, session_id: str
) -> None:
    """Create minimal Providence manifest.csv and one RTTM file."""
    rttm_path = rttm_dir / f"{child_id}_{session_id}.rttm"
    _make_mock_rttm(rttm_path, session_id, [(1.0, 1.5), (5.0, 2.0)])

    manifest_rows = [
        {
            "recording_id": f"providence_{child_id}_{session_id}",
            "path": "",
            "dataset_name": "providence",
            "child_id": child_id,
            "age_group": "other",
            "session_id": session_id,
            "duration_secs": 60.0,
            "split": "N/A",
            "has_rttm": True,
            "rttm_path": str(rttm_path),
        }
    ]
    pd.DataFrame(manifest_rows).to_csv(prov_dir / "manifest.csv", index=False)


def test_build_segment_manifest_excludes_test_speakers(tmp_path):
    """build_segment_manifest.py marks excluded speakers as usable_for_training=false."""
    prov_dir = tmp_path / "providence"
    rttm_dir = prov_dir / "rttm"
    rttm_dir.mkdir(parents=True)

    # Child that should be excluded (in test set)
    excluded_child_id = "test_child"
    session_id = "011000"  # 1 yr 10 mos = 22 months (outside standard bands, 'other')

    _make_mock_providence_manifest(prov_dir, rttm_dir, excluded_child_id, session_id)

    # Create the exclude CSV
    exclude_csv = tmp_path / "test_split.csv"
    pd.DataFrame({"child_id": [excluded_child_id]}).to_csv(exclude_csv, index=False)

    output_csv = tmp_path / "manifest.csv"

    result = subprocess.run(
        [
            sys.executable,
            str(_REPO_ROOT / "synth" / "scripts" / "build_segment_manifest.py"),
            "--providence-dir", str(prov_dir),
            "--providence-rttm-dir", str(rttm_dir),
            "--exclude-speakers-csv", str(exclude_csv),
            "--output", str(output_csv),
            "--min-duration-sec", "0.3",
            "--quality-threshold", "0.0",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"Script failed:\n{result.stderr}"
    assert output_csv.exists(), "Output CSV not created"

    df = pd.read_csv(output_csv)
    excluded_rows = df[df["speaker_id"] == excluded_child_id]
    assert len(excluded_rows) > 0, "No rows for excluded speaker found"
    assert not excluded_rows["usable_for_training"].any(), (
        "Excluded speaker should have usable_for_training=false for all rows"
    )
    excluded_splits = excluded_rows["split"].unique()
    assert all(s == "test" for s in excluded_splits), (
        f"Excluded speaker should have split='test', got {excluded_splits}"
    )
