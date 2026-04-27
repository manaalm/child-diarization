# Data Model: Synthetic Child-Adult Scene Generator

**Date**: 2026-04-24 | **Plan**: [plan.md](plan.md)

---

## Entities

### 1. Segment

A contiguous vocalization or speech unit from a single speaker, extracted from a source corpus recording. The atomic unit consumed by the scene generator.

**Fields**:

| Field | Type | Required | Constraints |
|-------|------|----------|-------------|
| `segment_id` | string | yes | globally unique; format `{dataset}_{recording_id}_{start_ms}_{end_ms}` |
| `source_dataset` | string | yes | one of `{providence, tinyvox, librispeech, musan_speech, external}` |
| `source_recording_id` | string | yes | identifier within the source corpus |
| `speaker_id` | string | yes | speaker identity within the source corpus |
| `speaker_role` | enum | yes | `{target_child, non_target_child, adult, unknown_child, background}` |
| `age_months` | float | no | null if unknown |
| `age_band` | enum | yes | `{14_18_months, 34_38_months, older_child, adult, unknown}` |
| `start_time_sec` | float | yes | ≥ 0 |
| `end_time_sec` | float | yes | > start_time_sec |
| `duration_sec` | float | yes | end_time_sec − start_time_sec; must be ≥ 0.3 s to be usable |
| `audio_path` | string | yes | absolute or relative path to extracted WAV file |
| `sample_rate` | int | yes | must equal output_sample_rate (16000) after extraction |
| `transcript` | string | no | orthographic transcript if available |
| `phonetic_transcript` | string | no | IPA or DISC-encoded phonetic transcript |
| `vocalization_type` | enum | no | `{speech, babble, laugh, cry, squeal, grunt, proto_word, canonical_babble, noncanonical_vocalization, unknown}` |
| `quality_score` | float | no | 0.0–1.0; composite proxy or corpus-provided; null if not computed |
| `split` | enum | yes | `{train, val, test, external}` |
| `usable_for_training` | bool | yes | false if split=test, quality_score < threshold, or duration < min |

**Validation rules**:
- `duration_sec` = `end_time_sec` − `start_time_sec` (checked at manifest build time)
- `split = test` → `usable_for_training = false` (enforced; assertion raised if violated)
- A speaker_id appearing in the real test split must have all segments set to `usable_for_training = false` (enforced by `--exclude-speakers-csv` in `build_segment_manifest.py`)

**Relationships**: Many Segments → one SourceRecording. Many Segments → one Scene (via SceneTrack).

---

### 2. Scene

A synthetic multi-speaker audio clip of fixed duration, produced by the scene generator from one or more Segments.

**Fields**:

| Field | Type | Required | Constraints |
|-------|------|----------|-------------|
| `synthetic_scene_id` | string | yes | format `{config_name}_{seed}_{index:06d}` |
| `duration_sec` | float | yes | matches `scene_duration_sec` in config (default 30.0) |
| `sample_rate` | int | yes | 16000 |
| `target_age_band` | enum | yes | `{14_18_months, 34_38_months}` |
| `scene_type` | enum | yes | `{positive, adult_only_negative, background_speech_negative, silence_noise_negative, hard_overlap_positive, hard_overlap_negative, short_vocalization_positive, low_snr_positive}` |
| `target_child_present` | bool | yes | true if any TARGET_CHILD RTTM segment exists |
| `target_child_vocalized` | bool | yes | same as `target_child_present` for binary clip label |
| `target_child_duration_sec` | float | yes | sum of TARGET_CHILD RTTM durations; 0.0 for negatives |
| `adult_present` | bool | yes | |
| `adult_duration_sec` | float | yes | |
| `non_target_child_present` | bool | yes | |
| `overlap_present` | bool | yes | true if any two speaker RTTM intervals overlap |
| `max_overlap_speakers` | int | yes | max simultaneous speakers at any frame |
| `mean_snr_db` | float | yes | mean SNR of child relative to background across scene |
| `rir_id` | string | no | null if no RIR applied |
| `noise_id` | string | no | null if no noise applied |
| `generation_config_hash` | string | yes | MD5 of the YAML config used |
| `random_seed` | int | yes | per-scene seed (global_seed + scene_index) |

**Outputs per scene**:
- `{synthetic_scene_id}.wav` — mixed audio
- `{synthetic_scene_id}.rttm` — speaker segments (RTTM format)
- `{synthetic_scene_id}.json` — scene metadata (all fields above)
- `{synthetic_scene_id}_segments.csv` — per-track placement timeline

**Relationships**: One Scene → one SceneConfig. One Scene → many SceneTracks (one per speaker placement).

---

### 3. SceneTrack

A single placed segment on the scene timeline: one source Segment at a specific offset with optional augmentation applied.

**Fields**:

| Field | Type | Required |
|-------|------|----------|
| `scene_id` | string | yes |
| `segment_id` | string | yes |
| `speaker_label` | string | yes | one of `{TARGET_CHILD, ADULT_0, ADULT_1, OTHER_CHILD_0, BACKGROUND_SPEECH}` |
| `start_sec` | float | yes | offset within the scene |
| `end_sec` | float | yes |
| `source_start_sec` | float | yes | start within the source audio file |
| `source_end_sec` | float | yes |
| `gain_db` | float | yes | applied gain (positive = louder) |
| `rir_id` | string | no | RIR file applied to this track, if any |
| `pitch_shift_semitones` | float | no | 0.0 if not applied |
| `time_stretch_factor` | float | no | 1.0 if not applied |

---

### 4. SceneConfig

A YAML document that fully specifies how scenes are generated. Identified by its MD5 hash for reproducibility. See `contracts/scene-config.md` for the full YAML schema.

**Key fields** (for data model purposes):
- `scene_duration_sec`, `n_scenes`, `target_age_band`, `random_seed`
- `sampling.*` — scene type probabilities and turn-taking parameters
- `mixing.*` — SNR range, RIR probability, noise probability
- `sources.*` — which segment libraries to draw from

**Validation**: The config hash is stored in every Scene's `generation_config_hash`. Regenerating scenes with the same hash must produce identical outputs.

---

### 5. SegmentManifest

The flat CSV inventory of all available source segments. Input to all scene generation runs.

**Schema**: See `contracts/segment-manifest.md`.

**Key integrity rules**:
- No `speaker_id` appears in both a train and test row.
- All `audio_path` values point to existing files (checked at generation time, not at manifest-build time).
- `duration_sec` values are pre-computed and stored; they are re-verified on first load.

---

### 6. TrainingManifest

A CSV that lists audio files and labels for one downstream training run at a specific synthetic-to-real ratio.

**Schema**: See `contracts/training-manifest.md`.

**Key integrity rules**:
- `split` column contains only `train` values (validation and test clips are not included).
- Synthetic rows have `is_synthetic = true`; real rows have `is_synthetic = false`.
- The real rows in every ratio manifest are identical (same real training clips, same labels).
- Synthetic rows come from the committed synthetic scene pool only.

---

## State Transitions

### Segment lifecycle

```
RAW_CORPUS → [extract_segments.py] → EXTRACTED_WAV
EXTRACTED_WAV → [build_segment_manifest.py] → MANIFEST_ROW
MANIFEST_ROW (usable_for_training=true) → [generate_scenes.py] → SCENE_TRACK
MANIFEST_ROW (usable_for_training=false) → EXCLUDED
```

### Scene lifecycle

```
PENDING → [generate_scenes.py] → GENERATED (wav + rttm + json + csv written)
GENERATED → [generate_training_sets.py] → IN_TRAINING_MANIFEST
IN_TRAINING_MANIFEST → [train_with_synthetic.py] → USED_FOR_TRAINING
USED_FOR_TRAINING → [evaluate_synthetic_augmentation.py] → EVALUATED
```

---

## Key Relationships

```
SegmentManifest 1──* Segment
SceneConfig     1──* Scene
Scene           1──* SceneTrack
SceneTrack      *──1 Segment
TrainingManifest 1──* TrainingManifestRow
TrainingManifestRow *──1 Scene (synthetic) or real audio clip
```
