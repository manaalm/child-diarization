# Contract: Segment Manifest CSV

**File**: `synth_results/manifests/segment_manifest.csv`
**Produced by**: `synth/scripts/build_segment_manifest.py`
**Consumed by**: `synth/scripts/generate_scenes.py`, `synth/manifest.py`

---

## Schema

| Column | Type | Required | Values / Constraints |
|--------|------|----------|----------------------|
| `segment_id` | string | yes | globally unique; `{dataset}_{recording_id}_{start_ms}_{end_ms}` |
| `source_dataset` | string | yes | `providence`, `tinyvox`, `librispeech`, `musan_speech`, `external` |
| `source_recording_id` | string | yes | recording identifier within the source corpus |
| `speaker_id` | string | yes | speaker identity within the corpus |
| `speaker_role` | string | yes | `target_child`, `non_target_child`, `adult`, `unknown_child`, `background` |
| `age_months` | float | no | null if unknown |
| `age_band` | string | yes | `14_18_months`, `34_38_months`, `older_child`, `adult`, `unknown` |
| `start_time_sec` | float | yes | ≥ 0.0 |
| `end_time_sec` | float | yes | > start_time_sec |
| `duration_sec` | float | yes | = end_time_sec − start_time_sec; ≥ 0.3 to be usable |
| `audio_path` | string | yes | absolute path to extracted 16 kHz mono WAV |
| `sample_rate` | int | yes | 16000 |
| `transcript` | string | no | empty string if unavailable |
| `phonetic_transcript` | string | no | empty string if unavailable |
| `vocalization_type` | string | no | `speech`, `babble`, `laugh`, `cry`, `squeal`, `grunt`, `proto_word`, `canonical_babble`, `noncanonical_vocalization`, `unknown` |
| `quality_score` | float | no | 0.0–1.0; null if not computed |
| `split` | string | yes | `train`, `val`, `test`, `external` |
| `usable_for_training` | bool | yes | `true` / `false`; false if split=test or quality below threshold |

## Integrity Constraints

1. No `speaker_id` appears in both a `train` row and a `test` row.
2. All rows with `split = test` have `usable_for_training = false`.
3. All rows with `split = test` and `source_dataset = providence` match a `child_id` from the real test split.
4. `duration_sec` = `end_time_sec` − `start_time_sec` (validated on load).

## Example Row

```csv
segment_id,source_dataset,source_recording_id,speaker_id,speaker_role,age_months,age_band,start_time_sec,end_time_sec,duration_sec,audio_path,sample_rate,transcript,phonetic_transcript,vocalization_type,quality_score,split,usable_for_training
providence_naima_1_1500_1820,providence,naima_1,naima,target_child,14.5,14_18_months,15.0,18.2,3.2,/data/segments/child/providence_naima_1_1500_1820.wav,16000,,,babble,0.72,train,true
```
