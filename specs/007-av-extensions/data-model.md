# Data Model: AV Extended Experiments — 007-av-extensions

**Date**: 2026-04-24  
**Feature**: Extends `006-av-child-vocalization` data model

---

## New Entities

### CascadeStageRecord
Per-clip record of which cascade stage made the final decision and what scores each stage produced.

| Field | Type | Description |
|---|---|---|
| clip_id | str | Primary key (same as MasterFeatureTable) |
| vad_speech_detected | bool | True if BabAR/VTC RTTM has any speech segment |
| vad_child_dur_sec | float | Total KCHI segment duration from RTTM |
| child_id_score | float | ECAPA cosine similarity score (from enrollment) |
| av_fusion_prob | float | Output of GatedAVModel if reached; else NaN |
| cascade_stage | int | 1=resolved at VAD, 2=resolved at child ID, 3=AV fusion |
| final_prob | float | The probability used for the final prediction |
| vad_threshold | float | Val-tuned threshold for VAD stage |
| child_id_threshold | float | Val-tuned threshold for child ID early-exit |

**Validation**:
- `cascade_stage` in {1, 2, 3}
- `final_prob` in [0.0, 1.0]
- If `cascade_stage == 1` and `vad_speech_detected == False`, `final_prob == 0.0`

**Output file**: `av_fusion/av_results/{run_name}/cascade_stage_breakdown.csv`

---

### SmoothedPredictionRecord
Per-clip record with raw and smoothed probabilities, linked to a session ordering.

| Field | Type | Description |
|---|---|---|
| clip_id | str | Primary key |
| child_id | str | Target child ID |
| session_id | str | Recording session identifier (derived from audio path or timepoint) |
| clip_position | int | Ordinal position of this clip within the session (0-indexed) |
| prob_raw | float | Raw model probability before smoothing |
| prob_smoothed | float | Smoothed probability after applying temporal filter |
| smoothing_method | str | One of: gaussian, majority_vote, moving_average |
| smoothing_param | float | Bandwidth (gaussian), window size (majority/moving avg) |
| label | int | Ground truth (0/1) |
| split | str | train, val, or test |

**Validation**:
- `prob_raw`, `prob_smoothed` both in [0.0, 1.0]
- `clip_position` is unique within (child_id, session_id, split)
- `smoothing_method` must be one of the three allowed values

**Output file**: `av_fusion/av_results/{run_name}/predictions_{split}_smoothed.csv`

---

### GPT4oFeatureRow
Per-clip structured output from GPT-4o vision queries over sampled frames.

| Field | Type | Description |
|---|---|---|
| clip_id | str | Primary key |
| child_visible_gpt4o | float | Fraction of sampled frames where GPT-4o said "yes" to child visible |
| child_vocalizing_gpt4o | float | Fraction of sampled frames where GPT-4o said "yes" to vocalizing |
| n_children_visible_mean | float | Mean count of children detected across sampled frames |
| visual_quality_gpt4o | float | Mean quality score (good=1.0, medium=0.5, poor=0.0) across frames |
| gpt4o_reasoning | str | Concatenated free-text notes from all sampled frames |
| n_frames_sampled | int | Number of frames actually queried (may be < target if video is short) |
| n_frames_api_error | int | Number of frames that returned an API error (excluded from aggregation) |
| model_used | str | The GPT model used (e.g., "gpt-4o-mini") |
| cost_usd_estimate | float | Estimated API cost in USD for this clip |

**Validation**:
- `child_visible_gpt4o`, `child_vocalizing_gpt4o` in [0.0, 1.0]
- `n_frames_sampled >= 1` for clips with valid video
- `child_visible_gpt4o = NaN` if `n_frames_sampled == 0` (audio-only or video unavailable)

**Output file**: `av_fusion/av_results/{run_name}/gpt4o_features.csv`  
**Cache**: `av_fusion/gpt4o_cache/{clip_id}_{frame_idx}.json` (raw API responses)

---

### ASDFeatureRow (extended)
Extends the existing ASD feature row from 006 to support multiple ASD model backends.

| Field | Type | Description |
|---|---|---|
| clip_id | str | Primary key |
| asd_model | str | One of: talknet, loconet, light_asd |
| max_asd_score_any_face | float | Max ASD score across all detected face tracks |
| mean_asd_score_any_face | float | Mean ASD score across all detected face tracks |
| max_asd_score_target_candidate | float | ASD score for the most child-likely face track |
| mean_asd_score_target_candidate | float | Mean ASD score for the most child-likely face track |
| fraction_frames_active_speaker | float | Fraction of frames with any active speaker detected |
| n_active_speaker_tracks | int | Number of distinct face tracks classified as active |
| asd_confidence_summary | float | Mean confidence of ASD predictions across all frames |

**Validation**:
- All score fields in [0.0, 1.0]
- `asd_model` must be one of the three allowed values
- One row per (clip_id, asd_model) combination

**Output files**: `av_fusion/av_results/{run_name}/asd_features_{model}.csv` (one per model)

---

### Ego4DExperimentRecord
Summary record of an Ego4D-based ASD adaptation experiment.

| Field | Type | Description |
|---|---|---|
| experiment_id | str | Unique identifier (e.g., "ego4d_loconet_zeroshot") |
| asd_model | str | Base ASD model (talknet, loconet, etc.) |
| adaptation_type | str | One of: zero_shot, fine_tuned, ego4d_pretrained |
| ego4d_subset | str | Which Ego4D subset used (e.g., "avd_train_50h") |
| val_auroc_home_video | float | AUROC on child home video val set (child vocalization labels) |
| baseline_auroc | float | AUROC of the same model without Ego4D adaptation |
| delta_auroc | float | val_auroc_home_video - baseline_auroc |
| notes | str | Free-text notes on experiment conditions |

**Output file**: `av_fusion/av_results/{run_name}/ego4d_experiment_results.csv`

---

## Relationships

```text
MasterFeatureTable (from 006)
  ↓ adds columns
  ├── child_visible_gpt4o         (from GPT4oFeatureRow)
  ├── child_vocalizing_gpt4o      (from GPT4oFeatureRow)
  ├── asd_{model}_max_score       (from ASDFeatureRow, one column per model)
  └── cascade_final_prob          (from CascadeStageRecord, optional join)

CascadeStageRecord
  → references MasterFeatureTable.clip_id
  → uses child_id_score from ECAPA enrollment (existing artifact)
  → uses vad_child_dur_sec from RTTM cache (existing artifact)

SmoothedPredictionRecord
  → references predictions_{split}.csv from any trained model (audio_only, gated_av, cascaded_av)
  → adds session_id, clip_position, prob_smoothed columns

ASDFeatureRow
  → references face_track_cache/{clip_id}.json (shared with TalkNet, existing)
  → one row per (clip_id, asd_model)
```

---

## File Layout (extended from 006)

```text
av_fusion/
├── av_results/{run_name}/
│   ├── gpt4o_features.csv              ← GPT4oFeatureRow (all clips)
│   ├── asd_features_loconet.csv        ← ASDFeatureRow, model=loconet
│   ├── asd_features_light_asd.csv      ← ASDFeatureRow, model=light_asd
│   ├── cascade_stage_breakdown.csv     ← CascadeStageRecord (test set)
│   ├── predictions_test_smoothed.csv   ← SmoothedPredictionRecord
│   ├── ego4d_experiment_results.csv    ← Ego4DExperimentRecord (if run)
│   └── 1kd_integration_report.json    ← compatibility check output
├── gpt4o_cache/
│   └── {clip_id}_{frame_idx}.json     ← raw GPT-4o API responses
└── configs/
    └── av_extensions.yaml             ← config for 007 experiments
```
