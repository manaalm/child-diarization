"""
Integration smoke-test for the synthetic scene generation pipeline.

Generates 10 scenes from default_14_18mo.yaml using a tiny mock manifest
(5 child + 3 adult 1-second random WAVs) and verifies:
  a) All 10 WAV files exist and have duration >= 29.9 s
  b) Each RTTM file has >= 1 SPEAKER line (for non-noise_only scenes)
  c) target_child_vocalized in clip manifest matches TARGET_CHILD presence in RTTM
  d) Re-running with the same seed produces bitwise-identical WAV files
"""

import numpy as np
import pandas as pd
import pytest
import soundfile as sf
from pathlib import Path

import sys
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import yaml

from synth.manifest import load_manifest
from synth.scene_generator import SceneComposer
from synth.labels import write_clip_labels_row


_CONFIG_PATH = _REPO_ROOT / "synth" / "configs" / "default_14_18mo.yaml"
_N_SCENES = 10
_GLOBAL_SEED = 7


def _make_mock_manifest(tmp_path: Path) -> pd.DataFrame:
    """Create a tiny manifest with 5 child + 3 adult 1-second WAV files."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(99)
    segments = []

    for i in range(5):
        wav = rng.uniform(-0.1, 0.1, 16000).astype(np.float32)
        audio_path = tmp_path / f"child_{i}.wav"
        sf.write(str(audio_path), wav, 16000)
        segments.append({
            "segment_id": f"test_child_{i}_0_1000",
            "source_dataset": "providence",
            "source_recording_id": f"child_rec_{i}",
            "speaker_id": f"child_{i}",
            "speaker_role": "target_child",
            "age_months": 15.0,
            "age_band": "14_18_months",
            "start_time_sec": 0.0,
            "end_time_sec": 1.0,
            "duration_sec": 1.0,
            "audio_path": str(audio_path),
            "sample_rate": 16000,
            "transcript": "",
            "phonetic_transcript": "",
            "vocalization_type": "babble",
            "quality_score": 0.8,
            "split": "train",
            "usable_for_training": True,
        })

    for i in range(3):
        wav = rng.uniform(-0.1, 0.1, 16000).astype(np.float32)
        audio_path = tmp_path / f"adult_{i}.wav"
        sf.write(str(audio_path), wav, 16000)
        segments.append({
            "segment_id": f"test_adult_{i}_0_1000",
            "source_dataset": "librispeech",
            "source_recording_id": f"adult_rec_{i}",
            "speaker_id": f"adult_{i}",
            "speaker_role": "adult",
            "age_months": None,
            "age_band": "adult",
            "start_time_sec": 0.0,
            "end_time_sec": 1.0,
            "duration_sec": 1.0,
            "audio_path": str(audio_path),
            "sample_rate": 16000,
            "transcript": "test",
            "phonetic_transcript": "",
            "vocalization_type": "speech",
            "quality_score": 0.9,
            "split": "train",
            "usable_for_training": True,
        })

    return pd.DataFrame(segments)


def _generate_scenes(
    output_dir: Path, manifest_df: pd.DataFrame, config: dict, seed: int
) -> list:
    """Run SceneComposer for _N_SCENES scenes; return clip-label rows."""
    composer = SceneComposer(config, manifest_df)
    rows = []
    for i in range(_N_SCENES):
        scene_id = f"test_{seed}_{i:06d}"
        per_rng = np.random.default_rng(seed + i)
        meta = composer.compose(scene_id, per_rng)
        meta["random_seed"] = seed + i
        composer.write(meta, str(output_dir))
        rows.append(write_clip_labels_row(meta))
    return rows


@pytest.fixture(scope="module")
def generated_scenes(tmp_path_factory):
    """Generate 10 scenes once for the whole module."""
    tmp = tmp_path_factory.mktemp("scenes")
    manifest_df = _make_mock_manifest(tmp)
    config = yaml.safe_load(_CONFIG_PATH.read_text())
    clip_rows = _generate_scenes(tmp, manifest_df, config, seed=_GLOBAL_SEED)
    return tmp, clip_rows


def test_wav_files_exist_and_correct_duration(generated_scenes):
    """(a) All WAV files exist and have duration >= 29.9 s."""
    output_dir, clip_rows = generated_scenes
    for row in clip_rows:
        wav_path = Path(row["audio_path"])
        assert wav_path.exists(), f"WAV missing: {wav_path}"
        info = sf.info(str(wav_path))
        assert info.duration >= 29.9, (
            f"WAV too short: {wav_path} duration={info.duration:.3f}s"
        )


def test_rttm_files_have_speaker_lines(generated_scenes):
    """(b) Non-noise scenes have >= 1 SPEAKER line in their RTTM."""
    output_dir, clip_rows = generated_scenes
    for row in clip_rows:
        rttm_path = Path(row["rttm_path"])
        assert rttm_path.exists(), f"RTTM missing: {rttm_path}"
        if row["scene_type"] == "noise_only_negative":
            continue  # noise-only scenes legitimately have empty RTTMs
        lines = [l for l in rttm_path.read_text().splitlines() if l.startswith("SPEAKER")]
        assert len(lines) >= 1, (
            f"Expected >= 1 SPEAKER line for scene_type={row['scene_type']} "
            f"in {rttm_path}"
        )


def test_rttm_label_consistency(generated_scenes):
    """(c) target_child_vocalized matches TARGET_CHILD presence in RTTM."""
    output_dir, clip_rows = generated_scenes
    for row in clip_rows:
        rttm_text = Path(row["rttm_path"]).read_text()
        has_child_in_rttm = "TARGET_CHILD" in rttm_text
        expected = bool(row["target_child_vocalized"])
        assert has_child_in_rttm == expected, (
            f"Inconsistency for {row['synthetic_scene_id']}: "
            f"target_child_vocalized={expected} but "
            f"TARGET_CHILD_in_RTTM={has_child_in_rttm}"
        )


def test_deterministic_reproduction(tmp_path):
    """(d) Re-running with the same seed produces bitwise-identical WAVs."""
    manifest_df = _make_mock_manifest(tmp_path / "segs")
    config = yaml.safe_load(_CONFIG_PATH.read_text())

    dir_a = tmp_path / "run_a"
    dir_b = tmp_path / "run_b"

    _generate_scenes(dir_a, manifest_df, config, seed=_GLOBAL_SEED)
    _generate_scenes(dir_b, manifest_df, config, seed=_GLOBAL_SEED)

    for i in range(_N_SCENES):
        scene_id = f"test_{_GLOBAL_SEED}_{i:06d}"
        wav_a = (dir_a / "wav" / f"{scene_id}.wav").read_bytes()
        wav_b = (dir_b / "wav" / f"{scene_id}.wav").read_bytes()
        assert wav_a == wav_b, (
            f"Scene {scene_id}: WAV bytes differ between two runs with same seed"
        )
