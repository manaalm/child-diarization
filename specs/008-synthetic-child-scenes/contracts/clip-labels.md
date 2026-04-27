# Contract: Clip-Level Labels CSV

**File**: `synth_results/manifests/synthetic_manifest.csv`
**Produced by**: `synth/scripts/generate_scenes.py`
**Consumed by**: `synth/scripts/generate_training_sets.py`, `synth/scripts/evaluate_synthetic_augmentation.py`

---

## Schema

| Column | Type | Required | Description |
|--------|------|----------|-------------|
| `synthetic_scene_id` | string | yes | unique scene identifier |
| `audio_path` | string | yes | absolute path to scene WAV file |
| `rttm_path` | string | yes | absolute path to scene RTTM file |
| `target_child_vocalized` | int | yes | 0 or 1; 1 if TARGET_CHILD RTTM segment exists |
| `target_child_duration_sec` | float | yes | total TARGET_CHILD RTTM duration; 0.0 for negatives |
| `adult_duration_sec` | float | yes | total ADULT_0 + ADULT_1 RTTM duration |
| `other_child_duration_sec` | float | yes | total OTHER_CHILD_0 RTTM duration; 0.0 if absent |
| `overlap_duration_sec` | float | yes | total duration of overlapping speaker intervals |
| `snr_db` | float | yes | mean SNR of child signal relative to background |
| `noise_type` | string | no | noise source name; empty if no noise applied |
| `rir_type` | string | no | RIR source name; empty if no RIR applied |
| `age_band` | string | yes | `14_18_months` or `34_38_months` |
| `scene_type` | string | yes | see scene_type enum in data-model.md |
| `generation_config_name` | string | yes | config YAML name (without extension) |
| `generation_config_hash` | string | yes | MD5 of the config YAML content |

## Integrity Constraints

- `target_child_vocalized = 1` ↔ `target_child_duration_sec > 0`
- `overlap_duration_sec ≥ 0`
- `snr_db` is present for all scenes with noise (NaN only if no background sources applied)

## Example Row

```csv
synthetic_scene_id,audio_path,rttm_path,target_child_vocalized,target_child_duration_sec,adult_duration_sec,other_child_duration_sec,overlap_duration_sec,snr_db,noise_type,rir_type,age_band,scene_type,generation_config_name,generation_config_hash
default_14_18mo_42_000001,/synth_results/synthetic_scenes/wav/default_14_18mo_42_000001.wav,/synth_results/synthetic_scenes/rttm/default_14_18mo_42_000001.rttm,1,2.34,8.75,0.0,0.48,12.3,musan_noise,rirs_noises,14_18_months,positive,default_14_18mo,a3f2c1d8e5b7
```
