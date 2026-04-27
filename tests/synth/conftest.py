import numpy as np
import pandas as pd
import pytest
import tempfile
import soundfile as sf
from pathlib import Path


@pytest.fixture
def tiny_manifest(tmp_path):
    """Minimal in-memory segment manifest for tests: 3 child + 2 adult segments."""
    segments = []
    rng = np.random.default_rng(42)

    # 3 child segments
    for i in range(3):
        wav = rng.uniform(-0.1, 0.1, 16000).astype(np.float32)  # 1s at 16kHz
        audio_path = tmp_path / f"child_{i}.wav"
        sf.write(str(audio_path), wav, 16000)
        segments.append({
            "segment_id": f"providence_child_{i}_0_1000",
            "source_dataset": "providence",
            "source_recording_id": f"child_{i}",
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

    # 2 adult segments
    for i in range(2):
        wav = rng.uniform(-0.1, 0.1, 48000).astype(np.float32)  # 3s
        audio_path = tmp_path / f"adult_{i}.wav"
        sf.write(str(audio_path), wav, 16000)
        segments.append({
            "segment_id": f"librispeech_adult_{i}_0_3000",
            "source_dataset": "librispeech",
            "source_recording_id": f"adult_{i}",
            "speaker_id": f"adult_{i}",
            "speaker_role": "adult",
            "age_months": None,
            "age_band": "adult",
            "start_time_sec": 0.0,
            "end_time_sec": 3.0,
            "duration_sec": 3.0,
            "audio_path": str(audio_path),
            "sample_rate": 16000,
            "transcript": "hello world",
            "phonetic_transcript": "",
            "vocalization_type": "speech",
            "quality_score": 0.9,
            "split": "train",
            "usable_for_training": True,
        })

    return pd.DataFrame(segments)
