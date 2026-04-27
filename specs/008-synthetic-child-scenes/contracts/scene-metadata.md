# Contract: Scene Metadata JSON

**File**: `synth_results/synthetic_scenes/json/{scene_id}.json`
**Produced by**: `synth/labels.py`

---

## Schema

```json
{
  "synthetic_scene_id": "string",
  "duration_sec": "float",
  "sample_rate": 16000,
  "target_age_band": "14_18_months | 34_38_months",
  "scene_type": "positive | adult_only_negative | ...",
  "target_child_present": "bool",
  "target_child_vocalized": "bool",
  "target_child_duration_sec": "float",
  "adult_present": "bool",
  "adult_duration_sec": "float",
  "non_target_child_present": "bool",
  "other_child_duration_sec": "float",
  "overlap_present": "bool",
  "max_overlap_speakers": "int",
  "mean_snr_db": "float | null",
  "rir_id": "string | null",
  "noise_id": "string | null",
  "generation_config_name": "string",
  "generation_config_hash": "string",
  "random_seed": "int",
  "source_segments": [
    {
      "speaker_label": "TARGET_CHILD | ADULT_0 | ...",
      "segment_id": "string",
      "source_dataset": "string",
      "start_sec": "float",
      "end_sec": "float",
      "gain_db": "float",
      "rir_id": "string | null"
    }
  ],
  "speakers": ["TARGET_CHILD", "ADULT_0"]
}
```

## Constraints

- `source_segments` list must be non-empty.
- All `start_sec` and `end_sec` values within `[0, duration_sec]`.
- `generation_config_hash` must match MD5 of the config YAML used.

## Example

```json
{
  "synthetic_scene_id": "default_14_18mo_42_000001",
  "duration_sec": 30.0,
  "sample_rate": 16000,
  "target_age_band": "14_18_months",
  "scene_type": "positive",
  "target_child_present": true,
  "target_child_vocalized": true,
  "target_child_duration_sec": 2.34,
  "adult_present": true,
  "adult_duration_sec": 8.75,
  "non_target_child_present": false,
  "other_child_duration_sec": 0.0,
  "overlap_present": true,
  "max_overlap_speakers": 2,
  "mean_snr_db": 12.3,
  "rir_id": "rirs_noises_room_004",
  "noise_id": "musan_noise_clip_182",
  "generation_config_name": "default_14_18mo",
  "generation_config_hash": "a3f2c1d8e5b7",
  "random_seed": 42,
  "source_segments": [
    {
      "speaker_label": "TARGET_CHILD",
      "segment_id": "providence_naima_1_1500_1820",
      "source_dataset": "providence",
      "start_sec": 2.34,
      "end_sec": 4.68,
      "gain_db": -3.2,
      "rir_id": "rirs_noises_room_004"
    }
  ],
  "speakers": ["TARGET_CHILD", "ADULT_0"]
}
```
