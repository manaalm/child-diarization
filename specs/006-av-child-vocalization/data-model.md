# Data Model: Audio-Visual Target-Child Vocalization Detection

**Feature**: 006-av-child-vocalization  
**Date**: 2026-04-24

---

## Entities

### 1. ClipRecord

The primary unit of data. One row per clip in the dataset.

**Source**: `whisper-modeling/seen_child_splits/{train,val,test}.csv` (already exists)

| Field | Type | Description |
|-------|------|-------------|
| `clip_id` | str | Unique clip identifier (row index from original BIDS CSV, stringified) |
| `child_id` | str | Target child ID (e.g., `A1H3H9Y3T1`); used for group-wise split enforcement |
| `timepoint_norm` | str | Age band string: `"14_month"` or `"36_month"` |
| `age_band` | str | Normalized: `"14_18_months"` or `"34_38_months"` (remapped from `timepoint_norm`) |
| `split` | str | `"train"`, `"val"`, or `"test"` |
| `audio_path` | str | Absolute path to `.wav` file |
| `video_path` | str or None | Absolute path to `.mp4` file (from `BidsProcessed`; None if missing) |
| `label` | int | 0 = child did not vocalize; 1 = child vocalized (`label` column in split CSV) |
| `duration_sec` | float or None | Clip duration in seconds (derived from `Vid_duration` if parseable) |
| `existing_audio_score` | float or None | Audio-only probability from best existing baseline |

**Validation rules**:
- `child_id` must appear in exactly one split across train/val/test.
- `label` must be 0 or 1 (no NaN for training rows).
- `video_path` may be None; all downstream code must handle None gracefully.

---

### 2. ManualVisualAnnotation

Per-clip human-scored fields already present in the split CSV. Used directly as visual features in the MVP.

**Source**: `whisper-modeling/seen_child_splits/*.csv` (columns already present)

| Field | Type | Range | Description |
|-------|------|-------|-------------|
| `Video_Quality_Child_Face_Visibility` | float | 1–10 | Human score: how visible is the child's face |
| `Video_Quality_Child_Body_Visibility` | float | 1–10 | Human score: how visible is the child's body |
| `Video_Quality_Child_Hand_Visibility` | float | 1–10 | Human score: how visible are the child's hands |
| `Video_Quality_Lighting` | float | 1–10 | Human score: lighting quality |
| `Video_Quality_Resolution` | float | 1–10 | Human score: video resolution/clarity |
| `Video_Quality_Motion` | float | 1–10 | Human score: motion blur / camera shake |
| `Child_of_interest_clear` | str | "yes"/"no"/NaN | Is the target child clearly identifiable in frame? |
| `n_adults` | int | ≥ 0 | Number of adults present (`#_adults` column) |
| `n_children` | int | ≥ 0 | Number of children present (`#_children` column) |
| `Body_Parts_Visible` | str | categorical | Which body parts are visible (e.g., "upper", "whole hand") |
| `Angle_of_Body` | str | categorical | Body angle (e.g., "front", "variable") |

**Derived fields** (computed during feature table assembly):
| Field | Type | Formula |
|-------|------|---------|
| `child_of_interest_clear_binary` | int | 1 if `Child_of_interest_clear == "yes"`, else 0 |
| `manual_face_visibility_norm` | float | `Video_Quality_Child_Face_Visibility / 10` |
| `manual_quality_norm` | float | `(Video_Quality_Lighting + Video_Quality_Resolution) / 20` |
| `n_people_total` | int | `n_adults + n_children` |
| `multi_person_clip` | int | 1 if `n_people_total > 1`, else 0 |

---

### 3. AutomaticVisualFeatures

Per-clip features extracted by running face detection and tracking on video frames.

**Source**: `av_fusion/scripts/extract_visual_features.py` → `av_fusion/av_results/{run}/visual_features.csv`

| Field | Type | Description |
|-------|------|-------------|
| `clip_id` | str | Join key to ClipRecord |
| `n_faces_detected_mean` | float | Mean face count per frame (over sampled frames) |
| `n_faces_detected_max` | int | Max faces detected in any single frame |
| `n_face_tracks` | int | Number of distinct face tracks across clip |
| `max_face_track_duration_sec` | float | Duration of the longest face track |
| `max_face_track_fraction_clip` | float | Longest track duration / clip duration |
| `mean_face_detection_confidence` | float | Mean detection confidence across all detections |
| `max_face_detection_confidence` | float | Max detection confidence |
| `mean_face_box_area_fraction` | float | Mean face bounding box area / frame area |
| `max_face_box_area_fraction` | float | Max face bounding box area / frame area |
| `min_face_box_area_fraction` | float | Min face bounding box area / frame area (child proxy) |
| `face_center_motion_std` | float | Std dev of face centroid displacement across frames (motion proxy) |
| `visual_quality_score` | float | Automated blur/brightness estimate (Laplacian variance + brightness) |
| `child_visible_score` | float | Score for child-sized face track present (smallest-face heuristic) |
| `off_camera_likely_score` | float | 1 - max_face_track_fraction_clip when no face detected for majority of clip |
| `visual_eligibility_score` | float | Composite eligibility score (see research.md Decision 5 formula) |

**Validation rules**:
- All numeric fields must be in [0, 1] or [0, ∞) as documented; NaN is allowed for clips where video is missing or unreadable.
- `clip_id` must be unique.
- If video_path is None, all fields are NaN except `off_camera_likely_score = 1.0`.

---

### 4. ASDFeatures (Optional)

Per-clip active-speaker detection scores from TalkNet-ASD.

**Source**: `av_fusion/scripts/extract_asd_features.py` → `av_fusion/av_results/{run}/asd_features.csv`

| Field | Type | Description |
|-------|------|-------------|
| `clip_id` | str | Join key |
| `max_asd_score_any_face` | float | Max TalkNet score across all face tracks |
| `mean_asd_score_any_face` | float | Mean TalkNet score across all face tracks |
| `max_asd_score_smallest_face` | float | Max TalkNet score for the smallest-face (child candidate) track |
| `mean_asd_score_smallest_face` | float | Mean TalkNet score for the child candidate track |
| `fraction_frames_any_active` | float | Fraction of frames where any face is classified as speaking |
| `fraction_frames_child_active` | float | Fraction of frames where child candidate face is speaking |
| `n_active_speaker_tracks` | int | Number of distinct face tracks ever classified as active speaker |

---

### 5. MasterFeatureRow

The fully merged per-clip feature vector used as model input.

**Source**: `av_fusion/scripts/build_av_feature_table.py` → `av_fusion/av_results/{run}/av_master_features.csv`

Contains all fields from ClipRecord + ManualVisualAnnotation derived fields + AutomaticVisualFeatures + ASDFeatures (when available) + `visual_eligible` binary flag.

**Key derived fields**:
| Field | Type | Description |
|-------|------|-------------|
| `visual_eligible` | int | Binary gate (0/1) derived from `visual_eligibility_score` vs. val-tuned threshold |
| `visual_eligibility_threshold` | float | Stored alongside results; the val-tuned cutoff |
| `audio_only_prediction` | float | Audio-only model probability (filled after training) |
| `av_prediction` | float | Always-fuse AV model probability (filled after evaluation) |
| `gated_av_prediction` | float | Gated AV model probability (filled after evaluation) |

---

### 6. FusionModel

A trained classifier saved to disk.

**Source**: `av_fusion/scripts/train_av_fusion.py` → `av_fusion/av_results/{run}/models/*.pkl`

| Artifact | Description |
|----------|-------------|
| `audio_only.pkl` | Trained on audio features only |
| `video_only.pkl` | Trained on visual/manual annotation features only |
| `always_fuse_av.pkl` | Trained on all features; applied to every clip |
| `gated_av.pkl` | Same as always_fuse_av but gating applied at inference |
| `visual_eligibility_threshold.json` | `{"threshold": 0.XX, "val_balanced_acc": 0.XX}` |
| `config.json` | Full experiment config including feature columns used, model HPs, seed |

---

### 7. EvaluationReport

A collection of output files produced by the evaluation script.

**Source**: `av_fusion/scripts/evaluate_av_fusion.py` → `av_fusion/av_results/{run}/`

| File | Format | Content |
|------|--------|---------|
| `metrics_overall.json` | JSON | AUROC, AUPRC, F1, precision, recall, balanced_accuracy per model class |
| `metrics_by_age_band.csv` | CSV | Same metrics × {14_18_months, 34_38_months} × model class |
| `metrics_by_visual_eligibility.csv` | CSV | Same metrics × {eligible, ineligible} × model class |
| `metrics_by_failure_mode.csv` | CSV | Counts and metrics by error-mode category |
| `predictions_test.csv` | CSV | One row per test clip: clip_id, child_id, age_band, visual_eligible, label, + probability/prediction for each model class |
| `error_analysis_examples.csv` | CSV | Annotated error examples with error_mode, key features, predictions |

---

## Entity Relationships

```
ClipRecord ──── has ────> ManualVisualAnnotation   (1:1, from split CSV)
ClipRecord ──── has ────> AutomaticVisualFeatures  (1:1, from extract_visual_features.py)
ClipRecord ──── has ────> ASDFeatures              (1:1, optional)
ClipRecord + ManualVisualAnnotation + AutomaticVisualFeatures + ASDFeatures
    ──────────────────────> MasterFeatureRow        (1:1, assembled by build_av_feature_table.py)
MasterFeatureRow ──── trains ──> FusionModel       (N:4 — N clips train 4 models)
FusionModel + MasterFeatureRow[test] ──> EvaluationReport
```

---

## Split Integrity Invariant

For any `child_id` C and any pair of splits S1 ≠ S2:
```
MasterFeatureRow.where(child_id == C and split == S1).count > 0
→ MasterFeatureRow.where(child_id == C and split == S2).count == 0
```
This invariant MUST be verified by `build_av_feature_table.py` at build time and asserted by `train_av_fusion.py` at training time.
