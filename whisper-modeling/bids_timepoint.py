"""BIDS-session-derived timepoint mapping (spec 022 US1).

The SAILS BIDS dataset at /orcd/scratch/bcs/001/sensein/sails/BIDS_data/final_bids-dataset/
uses sub-<ID>/ses-{01,02}/beh/ as the canonical layout. By convention:
    ses-01 -> 14-month visit  (annotated as "14_month" in the spreadsheet)
    ses-02 -> 36-month visit  (annotated as "36_month" in the spreadsheet)

Any other session id is treated as unknown; callers fall back to the spreadsheet's
own timepoint column if BIDS parsing fails and record the disagreement in the
bids_correction_provenance.json artefact.
"""

import os
import re
from typing import Optional

import pandas as pd


SES_TO_TIMEPOINT = {
    "ses-01": "14_month",
    "ses-02": "36_month",
}

_SES_RE = re.compile(r"_ses-(\d{2})_")
_SES_DIR_RE = re.compile(r"/ses-(\d{2})/")


def parse_session_id(path: str) -> Optional[str]:
    """Extract ses-NN from a BIDS audio/video path. Returns None if not parseable."""
    if not path or pd.isna(path):
        return None
    s = str(path)
    m = _SES_RE.search(s) or _SES_DIR_RE.search(s)
    if not m:
        return None
    return f"ses-{m.group(1)}"


def bids_session_to_timepoint(audio_path: str) -> Optional[str]:
    """Map a BIDS audio path to a normalised timepoint string.

    Returns one of {"14_month", "36_month"} or None for non-standard / unparseable
    paths. Caller decides whether to drop the row or fall back to the spreadsheet.
    """
    ses = parse_session_id(audio_path)
    if ses is None:
        return None
    return SES_TO_TIMEPOINT.get(ses)


def derive_timepoint_with_provenance(
    audio_path: str,
    spreadsheet_timepoint: Optional[str],
    spreadsheet_age_years: Optional[float] = None,
) -> dict:
    """Resolve the canonical timepoint for one row plus provenance metadata.

    Returns a dict with keys:
        bids_session_id, bids_timepoint, spreadsheet_timepoint,
        agree (bool), rationale_if_disagree (str), decision (str)

    decision is one of {"keep-bids", "keep-spreadsheet", "drop-row"}. keep-bids is
    the default whenever BIDS parsing yields a valid timepoint; the spreadsheet
    fallback is reserved for non-standard ses-* ids.
    """
    ses = parse_session_id(audio_path)
    bids_tp = SES_TO_TIMEPOINT.get(ses) if ses else None
    sheet_tp = spreadsheet_timepoint if spreadsheet_timepoint in {"14_month", "36_month"} else None

    agree = (bids_tp is not None) and (bids_tp == sheet_tp)

    if bids_tp is None and sheet_tp is None:
        decision = "drop-row"
        rationale = "both-unknown"
    elif bids_tp is None:
        decision = "keep-spreadsheet"
        rationale = "non-standard-session-id" if ses is not None else "bids-missing"
    elif sheet_tp is None:
        decision = "keep-bids"
        rationale = "spreadsheet-missing"
    elif bids_tp != sheet_tp:
        decision = "keep-bids"
        rationale = "spreadsheet-stale"
    else:
        decision = "keep-bids"
        rationale = ""

    return {
        "bids_session_id": ses or "unknown",
        "bids_timepoint": bids_tp or "unknown",
        "spreadsheet_timepoint": sheet_tp or "unknown",
        "agree": agree,
        "rationale_if_disagree": rationale,
        "decision": decision,
    }


def load_participants_tsv(bids_root: str) -> pd.DataFrame:
    """Load participants.tsv if present. The SAILS dataset's file only has
    participant_id and group; age is not encoded there. Caller is responsible
    for cross-validating timepoints against the spreadsheet Age column."""
    path = os.path.join(bids_root, "participants.tsv")
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path, sep="\t")
