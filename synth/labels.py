"""
Label generation utilities for synthetic scenes.

Produces RTTM files, clip-label rows, and scene metadata JSON files
matching the contracts defined in:
  - specs/008-synthetic-child-scenes/contracts/rttm-output.md
  - specs/008-synthetic-child-scenes/contracts/clip-labels.md
  - specs/008-synthetic-child-scenes/contracts/scene-metadata.md
"""

from __future__ import annotations

import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any

# Valid speaker label values (per rttm-output.md)
VALID_SPEAKER_LABELS = {
    "TARGET_CHILD",
    "ADULT_0",
    "ADULT_1",
    "OTHER_CHILD_0",
    "BACKGROUND_SPEECH",
}

# Frame grid resolution for overlap computation (seconds)
_FRAME_GRID_SEC = 0.010


def write_rttm(tracks: List[Dict[str, Any]], scene_id: str, path: str) -> None:
    """Write speaker segments to an RTTM file.

    Parameters
    ----------
    tracks : list of dict
        Each dict must contain:
          - ``speaker_label`` (str): one of VALID_SPEAKER_LABELS
          - ``start_sec`` (float): segment start time in seconds
          - ``end_sec`` (float): segment end time in seconds
    scene_id : str
        Scene identifier written in the ``<file_id>`` field of each RTTM line.
    path : str
        Output file path.  Parent directory is created if it does not exist.

    Notes
    -----
    Lines with ``dur <= 0`` are silently skipped.
    """
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w") as f:
        for track in tracks:
            start = float(track["start_sec"])
            end = float(track["end_sec"])
            dur = end - start
            if dur <= 0:
                continue
            label = track["speaker_label"]
            f.write(
                f"SPEAKER {scene_id} 1 {start:.3f} {dur:.3f} <NA> <NA> {label} <NA> <NA>\n"
            )


def _compute_overlap_duration(tracks: List[Dict[str, Any]]) -> float:
    """Compute total overlap duration across all tracks using a 10 ms frame grid.

    Parameters
    ----------
    tracks : list of dict
        Each dict has ``start_sec`` and ``end_sec`` keys.

    Returns
    -------
    float
        Total duration (seconds) during which ≥ 2 tracks are active simultaneously.
    """
    if not tracks:
        return 0.0

    # Build grid up to the maximum end time
    max_end = max(float(t["end_sec"]) for t in tracks)
    n_frames = int(np.ceil(max_end / _FRAME_GRID_SEC)) + 1
    frame_count = np.zeros(n_frames, dtype=np.int32)

    for track in tracks:
        start = float(track["start_sec"])
        end = float(track["end_sec"])
        if end <= start:
            continue
        i_start = int(np.floor(start / _FRAME_GRID_SEC))
        i_end = int(np.ceil(end / _FRAME_GRID_SEC))
        i_start = max(0, i_start)
        i_end = min(n_frames, i_end)
        frame_count[i_start:i_end] += 1

    overlap_frames = np.sum(frame_count >= 2)
    return float(overlap_frames) * _FRAME_GRID_SEC


def write_clip_labels_row(scene_meta: dict) -> dict:
    """Build a clip-label row dict from a scene metadata dict.

    Computes per-speaker durations and overlap duration from the ``tracks``
    list inside ``scene_meta``, then packages all fields required by
    ``contracts/clip-labels.md``.

    Parameters
    ----------
    scene_meta : dict
        Scene metadata dict.  Must contain:
          - ``synthetic_scene_id`` (str)
          - ``audio_path`` (str)
          - ``rttm_path`` (str)
          - ``tracks`` (list of dicts with speaker_label, start_sec, end_sec)
          - ``snr_db`` (float or None)
          - ``noise_type`` (str)
          - ``rir_type`` (str)
          - ``age_band`` (str)
          - ``scene_type`` (str)
          - ``generation_config_name`` (str)
          - ``generation_config_hash`` (str)

    Returns
    -------
    dict
        Row matching the clip-labels CSV schema.
    """
    tracks = scene_meta.get("tracks", [])

    target_child_dur = sum(
        float(t["end_sec"]) - float(t["start_sec"])
        for t in tracks
        if t["speaker_label"] == "TARGET_CHILD"
        and float(t["end_sec"]) > float(t["start_sec"])
    )

    adult_dur = sum(
        float(t["end_sec"]) - float(t["start_sec"])
        for t in tracks
        if t["speaker_label"] in {"ADULT_0", "ADULT_1"}
        and float(t["end_sec"]) > float(t["start_sec"])
    )

    other_child_dur = sum(
        float(t["end_sec"]) - float(t["start_sec"])
        for t in tracks
        if t["speaker_label"] == "OTHER_CHILD_0"
        and float(t["end_sec"]) > float(t["start_sec"])
    )

    overlap_dur = _compute_overlap_duration(tracks)

    target_child_vocalized = 1 if target_child_dur > 0 else 0

    return {
        "synthetic_scene_id": scene_meta["synthetic_scene_id"],
        "audio_path": scene_meta["audio_path"],
        "rttm_path": scene_meta["rttm_path"],
        "target_child_vocalized": target_child_vocalized,
        "target_child_duration_sec": target_child_dur,
        "adult_duration_sec": adult_dur,
        "other_child_duration_sec": other_child_dur,
        "overlap_duration_sec": overlap_dur,
        "snr_db": scene_meta.get("snr_db"),
        "noise_type": scene_meta.get("noise_type", ""),
        "rir_type": scene_meta.get("rir_type", ""),
        "age_band": scene_meta["age_band"],
        "scene_type": scene_meta["scene_type"],
        "generation_config_name": scene_meta["generation_config_name"],
        "generation_config_hash": scene_meta["generation_config_hash"],
    }


def write_scene_metadata(scene_meta: dict, path: str) -> None:
    """Write scene metadata to a JSON file.

    Parameters
    ----------
    scene_meta : dict
        Scene metadata dict (matching ``contracts/scene-metadata.md``).
    path : str
        Output file path.  Parent directory is created if it does not exist.
    """
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w") as f:
        json.dump(scene_meta, f, indent=2, default=str)
