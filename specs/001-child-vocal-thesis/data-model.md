# Data Model: Child Vocalization Extraction & Synthesis Thesis

**Phase 1 Output** | **Date**: 2026-04-17 | **Feature**: 001-child-vocal-thesis

---

## Entity: AudioRecording

Represents one source audio file with associated metadata. May or may not have RTTM
ground truth depending on dataset origin.

| Field | Type | Description |
|-------|------|-------------|
| recording_id | string | Unique identifier (`{dataset}_{child_id}_{session_id}`) |
| path | string | Absolute path to WAV file (16kHz mono) |
| dataset_name | enum | `playlogue`, `providence`, `seedlings`, `tinyvox`, `core` |
| child_id | string | Child identifier within dataset |
| age_group | enum | `12_16m`, `34_38m`, `other`, `unknown` |
| session_id | string | Session/recording identifier within dataset |
| duration_secs | float | Total recording duration |
| split | enum | `train`, `val`, `test` — from `seen_child_splits/` |
| has_rttm | bool | Whether ground truth RTTM exists for this recording |
| rttm_path | string? | Path to ground truth RTTM file (null if `has_rttm=false`) |

**Validation rules**:
- `path` must resolve to a readable file with sample rate ≥ 16kHz.
- If `has_rttm=true`, `rttm_path` must be non-null and the file must exist.
- `age_group` must not be `unknown` for any recording used in age-stratified evaluation.
- No recording may appear in more than one split.

**State transitions**: None (immutable once registered in a dataset manifest).

---

## Entity: ChildVocalizationSegment

A timestamped audio segment attributed to a child speaker, either from ground truth
RTTM or from a model's predicted RTTM.

| Field | Type | Description |
|-------|------|-------------|
| segment_id | string | Auto-generated UUID |
| recording_id | string | FK → AudioRecording |
| start_time | float | Onset in seconds |
| end_time | float | Offset in seconds |
| duration | float | `end_time - start_time` in seconds |
| source | enum | `ground_truth`, `usc_sail`, `pyannote`, `babar` |
| age_group | enum | Inherited from parent AudioRecording |
| confidence | float? | Model confidence score (null for ground_truth) |

**Validation rules**:
- `start_time` < `end_time`; `duration` > 0.
- `duration` ≥ 0.05s (minimum 50ms — post-processing floor per architecture notes).
- `start_time` ≥ 0 and `end_time` ≤ parent `duration_secs`.

---

## Entity: SpeakerPrototype

An ECAPA-TDNN enrollment embedding representing either a specific target child or an
age-group aggregate, used for cosine similarity scoring.

| Field | Type | Description |
|-------|------|-------------|
| prototype_id | string | `{scope}_{id}` e.g. `child_C001` or `age_12_16m` |
| scope | enum | `child` (per-child enrollment), `age_group` (aggregate prototype) |
| child_id | string? | Child identifier (null if scope=age_group) |
| age_group | enum | Age group this prototype represents |
| embedding_path | string | Path to `.pt` file containing the embedding tensor |
| n_segments_used | int | Number of vocalization segments averaged |
| mean_duration_used | float | Mean segment duration in averaging pool |
| split | enum | Split from which segments were drawn (always `train`) |

**Validation rules**:
- `scope=child` requires `child_id` to be non-null.
- `n_segments_used` ≥ 1; prototypes with < 3 segments should be flagged as low-quality.
- `embedding_path` must exist and load as a 1D float tensor of dim 192 (ECAPA-TDNN default).

---

## Entity: SyntheticSpeechSample

A generated audio clip produced by the synthesis model, tagged by age group and used
either for quality evaluation or augmentation.

| Field | Type | Description |
|-------|------|-------------|
| sample_id | string | Auto-generated UUID |
| age_group | enum | `12_16m` or `34_38m` — the conditioning age group |
| model_name | string | Synthesis model identifier (`vits_34m_v1`, `vae_12m_v1`, etc.) |
| path | string | Absolute path to generated WAV file |
| seed | int | Random seed used for generation (reproducibility) |
| duration_secs | float | Duration of generated clip |
| mcd_score | float? | MCD vs. nearest ground-truth reference (null until evaluated) |
| speaker_similarity | float? | ECAPA cosine similarity to age-group prototype |
| age_classifier_pred | enum? | Predicted age group from age classifier |
| split_usage | enum | `eval_only` (quality eval only) or `augmentation` (added to training) |

**Validation rules**:
- `age_group` must match the conditioning input used during generation.
- `seed` must be recorded for every sample (reproducibility requirement).
- `mcd_score` and `speaker_similarity` must be populated before any sample is used in
  augmentation or quality reporting.

---

## Entity: ExperimentResult

A versioned collection of evaluation outputs for one model variant and data condition.
Must be committed to version control under a canonical results folder.

| Field | Type | Description |
|-------|------|-------------|
| experiment_id | string | `{diarizer}_{condition}_{split}_{age_group}` |
| diarizer | enum | `usc_sail`, `pyannote`, `babar` |
| condition | enum | `baseline`, `age_stratified`, `augmented`, `proxy` |
| split | enum | `val`, `test` |
| age_group | enum | `all`, `12_16m`, `34_38m` |
| threshold | float | Decision threshold (tuned on val) |
| f1 | float | F1 score at threshold |
| precision | float | Precision at threshold |
| recall | float | Recall at threshold |
| auroc | float | Area under ROC curve |
| auprc | float | Area under precision-recall curve |
| config_path | string | Path to committed config file used to produce this result |
| result_dir | string | Path to canonical results folder |
| created_at | string | ISO 8601 timestamp |

**Validation rules**:
- All metric fields (f1, precision, recall, auroc, auprc) must be in [0, 1].
- `config_path` must exist and be version-controlled at time of result reporting.
- `threshold` must have been selected on val, not on test data.
- For `condition=augmented`, the corresponding `condition=baseline` ExperimentResult
  with the same diarizer/split/age_group must exist before the augmented result is
  reported (enables direct comparison).

---

## Dataset Manifests (File Conventions)

Each dataset uses a manifest CSV at `{dataset}/manifest.csv` with columns matching
the AudioRecording entity. The Playlogue manifest is derived from
`BIDS_data/anotated_processed.csv`; Providence from CHAT transcript metadata;
Seedlings from Databrary API via `seedlings_import.py`.

The synthesis sample registry is a JSON file at
`synthesis/generated/{model_name}/registry.jsonl` — one JSON object per line,
matching the SyntheticSpeechSample entity schema.

---

## Split Conventions (from Architecture)

| Split set | Location | Paradigm | Used for |
|-----------|----------|----------|----------|
| Seen-child | `whisper-modeling/seen_child_splits/` | Within-child 60/20/20 | All enrollment + augmentation experiments |
| Cross-child | `baselines/splits/` | Disjoint children 97/21/21 | Baseline encoder models only |

All new experiments in this spec use the **seen-child split** exclusively (as per
existing project design for enrollment-based work).

---

## Entity: VideoRecording

Represents a SAILS BIDS video file paired with an AudioRecording. Only exists for SAILS data (Providence and Playlogue are audio-only).

| Field | Type | Description |
|-------|------|-------------|
| video_id | string | Same as parent `recording_id` |
| video_path | string | Absolute path to `.mp4` file (BIDS processed) |
| audio_path | string | Corresponding `_audio.wav` path (FK → AudioRecording.path) |
| face_cache_path | string? | Path to JSON face-track cache (null until face detection runs) |

**Naming rule**: `video_path = audio_path.replace("_audio.wav", "_desc-processed_beh.mp4")`

**Validation rules**:
- `video_path` must exist before any ASD frontend can run on this recording.
- If `video_path` does not exist, the ASD frontend must raise `FileNotFoundError` with a message indicating this is audio-only data (Providence/Playlogue).

---

## Entity: FaceTrack

A detected face trajectory across video frames for one speaker candidate.

| Field | Type | Description |
|-------|------|-------------|
| track_id | string | `{video_id}_track_{N}` |
| video_id | string | FK → VideoRecording |
| frame_boxes | list[dict] | List of `{"frame_idx": int, "bbox": [x1,y1,x2,y2], "score": float}` |
| is_child_candidate | bool | True if heuristically identified as child (smallest face) |
| n_frames | int | Number of frames this track spans |

**Validation rules**:
- `frame_boxes` must be non-empty; each bbox must have 4 coordinates.
- Exactly one track per video should have `is_child_candidate=True` (for TalkNet-ASD without enrollment); zero is acceptable if no face is detected.

---

## Entity: ASDPrediction

Per-frame active speaker detection output from a video ASD model for one face track.

| Field | Type | Description |
|-------|------|-------------|
| prediction_id | string | Auto-generated UUID |
| video_id | string | FK → VideoRecording |
| track_id | string | FK → FaceTrack |
| model_name | enum | `talknet_asd`, `ts_talknet`, `loconet` |
| frame_scores | list[float] | Per-frame speaking probability (0–1) |
| threshold | float | Decision threshold applied to convert to binary |
| child_segments | list[dict] | Aggregated `{"start": float, "end": float}` segments after thresholding + post-processing |

**Validation rules**:
- `len(frame_scores) == n_frames` of the associated FaceTrack.
- `threshold` must be in (0, 1).
- `child_segments` are derived deterministically from `frame_scores` + `threshold`; must be reproducible.
