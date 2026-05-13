"""Smoke tests for whisper-modeling/bids_timepoint.py (spec 022 US1)."""

import os
import sys

import pytest

REPO_ROOT = "/orcd/scratch/orcd/008/manaal/child-adult-diarization"
sys.path.insert(0, os.path.join(REPO_ROOT, "whisper-modeling"))

from bids_timepoint import (  # noqa: E402
    SES_TO_TIMEPOINT,
    bids_session_to_timepoint,
    derive_timepoint_with_provenance,
    parse_session_id,
)


def test_ses_to_timepoint_constants():
    assert SES_TO_TIMEPOINT == {"ses-01": "14_month", "ses-02": "36_month"}


@pytest.mark.parametrize("path,expected", [
    ("/.../sub-A1H3H9Y3T1/ses-01/beh/sub-A1H3H9Y3T1_ses-01_task-x_run-01_audio.wav", "ses-01"),
    ("/.../sub-D9N0U7M9X3/ses-02/beh/sub-D9N0U7M9X3_ses-02_task-foo_run-02_audio.wav", "ses-02"),
    ("/.../sub-A1H3H9Y3T1_ses-99_task-y_audio.wav", "ses-99"),
    ("/some/random/path/no/bids/structure.wav", None),
    ("", None),
])
def test_parse_session_id(path, expected):
    assert parse_session_id(path) == expected


@pytest.mark.parametrize("path,expected_tp", [
    ("/.../sub-X/ses-01/beh/sub-X_ses-01_audio.wav", "14_month"),
    ("/.../sub-X/ses-02/beh/sub-X_ses-02_audio.wav", "36_month"),
    ("/.../sub-X_ses-99_audio.wav", None),
    ("not-a-bids-path", None),
])
def test_bids_session_to_timepoint(path, expected_tp):
    assert bids_session_to_timepoint(path) == expected_tp


def test_derive_provenance_agree():
    """BIDS=14_month, spreadsheet=14_month -> agree, keep-bids."""
    r = derive_timepoint_with_provenance("/sub-X/ses-01/sub-X_ses-01.wav", "14_month")
    assert r["bids_timepoint"] == "14_month"
    assert r["spreadsheet_timepoint"] == "14_month"
    assert r["agree"] is True
    assert r["decision"] == "keep-bids"


def test_derive_provenance_stale_spreadsheet():
    """BIDS=36_month, spreadsheet=14_month -> disagree, spreadsheet stale, keep-bids."""
    r = derive_timepoint_with_provenance("/sub-X/ses-02/sub-X_ses-02.wav", "14_month")
    assert r["bids_timepoint"] == "36_month"
    assert r["agree"] is False
    assert r["rationale_if_disagree"] == "spreadsheet-stale"
    assert r["decision"] == "keep-bids"


def test_derive_provenance_spreadsheet_missing():
    """BIDS=14_month, spreadsheet=None -> keep-bids (recovery case)."""
    r = derive_timepoint_with_provenance("/sub-X/ses-01/sub-X_ses-01.wav", None)
    assert r["bids_timepoint"] == "14_month"
    assert r["spreadsheet_timepoint"] == "unknown"
    assert r["rationale_if_disagree"] == "spreadsheet-missing"
    assert r["decision"] == "keep-bids"


def test_derive_provenance_bids_missing():
    """BIDS=None, spreadsheet=14_month -> keep-spreadsheet fallback."""
    r = derive_timepoint_with_provenance("/random/no/bids.wav", "14_month")
    assert r["bids_timepoint"] == "unknown"
    assert r["decision"] == "keep-spreadsheet"


def test_derive_provenance_both_unknown():
    r = derive_timepoint_with_provenance("/random/no/bids.wav", None)
    assert r["decision"] == "drop-row"
    assert r["rationale_if_disagree"] == "both-unknown"
