# Feature Specification: Audio-Visual Target-Child Vocalization Detection

**Feature Branch**: `006-av-child-vocalization`  
**Created**: 2026-04-24  
**Status**: Draft  
**Project**: MIT EECS MEng Thesis — Audio-Visual Target-Child Vocalization Detection in Naturalistic Home Recordings

---

## User Scenarios & Testing *(mandatory)*

### User Story 1 — Visual Feature Extraction Pipeline (Priority: P1)

A researcher runs a script against a metadata CSV and obtains, for every clip, a structured set of visual features: how many faces were detected, whether a child-sized face was present, what fraction of the clip had a trackable face, visual quality indicators, and a composite visual eligibility score.

**Why this priority**: Visual features are the prerequisite for everything downstream. Without them, no fusion or gating is possible. This story delivers the first standalone artifact — a `visual_features.csv` — that has value even before any model is trained.

**Independent Test**: Run `extract_visual_features.py` on a metadata CSV; confirm `visual_features.csv` exists with one row per clip containing at minimum `clip_id`, `n_faces_detected_mean`, `n_face_tracks`, `visual_eligibility_score`, `child_visible_score`, `off_camera_likely_score`.

**Acceptance Scenarios**:

1. **Given** a metadata CSV with video paths, **When** `extract_visual_features.py` runs, **Then** every clip in the CSV has a corresponding row in `visual_features.csv` (no silently dropped clips).
2. **Given** a clip where no faces are detected, **When** visual features are extracted, **Then** face-related columns are 0 or NaN and `visual_eligibility_score` is low (≤ 0.2).
3. **Given** a clip where a small child face is clearly visible for most of the clip, **When** visual features are extracted, **Then** `child_visible_score` is high (≥ 0.7) and `visual_eligibility_score` is high (≥ 0.6).
4. **Given** a clip that is too dark or blurry, **When** visual features are extracted, **Then** `visual_quality_score` is low and `visual_eligibility_score` is penalized accordingly.

---

### User Story 2 — Audio-Visual Feature Table Assembly (Priority: P1)

A researcher merges clip metadata, existing audio-only baseline scores, visual features, and (optionally) ASD features into a single master feature table. The table covers all clips in all splits and is the input to every model in the pipeline.

**Why this priority**: Feature table assembly is a prerequisite for model training. It must correctly preserve group-wise child splits, handle missing video or missing ASD features gracefully, and include all required label and split columns.

**Independent Test**: Run `build_av_feature_table.py`; confirm `av_master_features.csv` exists, contains all expected columns, has no cross-child split leakage (all rows for a given child_id appear in exactly one of train/val/test), and no clip from the metadata CSV is silently dropped.

**Acceptance Scenarios**:

1. **Given** metadata CSV, visual features CSV, and audio baseline scores CSV, **When** `build_av_feature_table.py` runs, **Then** `av_master_features.csv` has all clips, all required columns, and the `split` column correctly assigns every clip by child ID.
2. **Given** a clip with missing video (video_path is None or file missing), **When** the feature table is assembled, **Then** the clip still appears with visual feature columns set to sentinel values (NaN or 0) and `visual_eligible` set to False.
3. **Given** ASD features CSV is not provided, **When** the feature table is assembled, **Then** ASD columns are omitted or set to NaN without crashing; all other features are populated correctly.
4. **Given** a child who has clips in the training set, **When** the feature table is assembled, **Then** none of that child's clips appear in the val or test sets.

---

### User Story 3 — Fusion Model Training (Priority: P2)

A researcher trains four model classes (audio-only, video-only, always-fuse AV, gated AV) on the training split, tunes classification thresholds on the validation split, and saves trained models and thresholds for later evaluation.

**Why this priority**: Fusion training is the core experimental contribution. The four models must be comparable, trained without using test data for any decision, and saved in a reproducible way.

**Independent Test**: Run `train_av_fusion.py`; confirm four model files exist under `outputs/models/`; load each model and confirm it can produce scalar probability predictions for a held-out batch.

**Acceptance Scenarios**:

1. **Given** `av_master_features.csv` with train and val splits, **When** `train_av_fusion.py` runs, **Then** four model artifacts are saved: `audio_only.pkl`, `video_only.pkl`, `always_fuse_av.pkl`, `gated_av.pkl`.
2. **Given** a run of `train_av_fusion.py`, **When** examined for data leakage, **Then** no validation or test clips are used during training; threshold tuning uses only the val split.
3. **Given** a class-imbalanced label distribution, **When** models are trained, **Then** class weighting or threshold calibration is applied so the audio-only model is not trivially biased toward the majority class.
4. **Given** the gated AV model at inference time, **When** a clip has `visual_eligible = False`, **Then** the model uses the audio-only probability; when `visual_eligible = True`, it uses the AV classifier probability.

---

### User Story 4 — Evaluation and Stratified Reporting (Priority: P2)

A researcher runs evaluation on the held-out test split and obtains a full metric report: overall AUROC/AUPRC/F1, metrics by age band (14–18 mo, 34–38 mo), metrics by visual eligibility, metrics for off-camera clips, and a confusion matrix. All results are written to structured output files suitable for thesis inclusion.

**Why this priority**: Stratified evaluation is the thesis contribution. An overall test AUROC alone is insufficient — the value of the system lies in understanding when and why AV helps or hurts.

**Independent Test**: Run `evaluate_av_fusion.py`; confirm `metrics_overall.json`, `metrics_by_age_band.csv`, `metrics_by_visual_eligibility.csv`, and `predictions_test.csv` all exist and contain the expected strata.

**Acceptance Scenarios**:

1. **Given** trained models and test split, **When** `evaluate_av_fusion.py` runs, **Then** `metrics_overall.json` contains AUROC, AUPRC, F1, precision, recall, balanced accuracy for all four model classes.
2. **Given** test clips labeled with `age_band`, **When** stratified metrics are computed, **Then** `metrics_by_age_band.csv` has separate rows for `14_18_months` and `34_38_months` for each model class.
3. **Given** test clips with `visual_eligible` flags, **When** stratified metrics are computed, **Then** `metrics_by_visual_eligibility.csv` has separate rows for eligible vs. ineligible clips for each model class.
4. **Given** a test run where AV does not improve on audio-only overall, **When** results are written, **Then** the output files clearly reflect this result without error or suppression; the system supports presenting null results.
5. **Given** `predictions_test.csv`, **When** examined, **Then** each row has `clip_id`, predicted probability, predicted label, ground truth label, and model identifier.

---

### User Story 5 — Error Analysis (Priority: P3)

A researcher runs an error analysis script that identifies the specific failure modes: clips where audio-only fails but AV succeeds, clips where AV fails but audio-only succeeds, clips where bad video introduces false positives, and clips where an off-camera or occluded child causes false negatives. The output is a structured table and summary suitable for thesis discussion.

**Why this priority**: Error analysis is required for thesis claims about conditional AV benefit. It transforms evaluation numbers into interpretable findings about when and why the system fails.

**Independent Test**: Run `error_analysis_av.py`; confirm `error_analysis_examples.csv` exists with columns for error type, clip_id, model predictions, ground truth, and key visual/audio features; confirm it contains rows in at least 3 of the 4 error-mode categories.

**Acceptance Scenarios**:

1. **Given** test predictions and feature table, **When** `error_analysis_av.py` runs, **Then** each clip is assigned to zero or more error-mode categories: AV-helped, AV-hurt, off-camera-miss, multi-face-confusion.
2. **Given** clips classified as AV-helped, **When** their features are examined, **Then** they tend to have high `visual_eligible` scores and audio features indicating audio-only uncertainty.
3. **Given** clips classified as AV-hurt (AV false positives), **When** their features are examined, **Then** they tend to have low `visual_quality_score` or `off_camera_likely_score` near 1.

---

### User Story 6 — Optional ASD Feature Extraction (Priority: P4)

A researcher optionally runs `extract_asd_features.py` to compute active-speaker detection scores for each clip: whether any face was detected as speaking, and whether the most child-likely face candidate was active. These features are then merged into the master feature table via `build_av_feature_table.py`.

**Why this priority**: ASD features can improve fusion quality if reliable, but they are computationally expensive and require an ASD model checkpoint. This story is stretch-goal — the system must work without it.

**Independent Test**: Run `extract_asd_features.py` on a subset; confirm `asd_features.csv` exists with ASD columns per clip; re-run `build_av_feature_table.py` with ASD CSV; confirm ASD columns appear in master features.

**Acceptance Scenarios**:

1. **Given** an ASD model checkpoint is available, **When** `extract_asd_features.py` runs, **Then** `asd_features.csv` contains `max_asd_score_any_face`, `fraction_frames_active_speaker_detected`, and related columns for every clip.
2. **Given** a clip where a child face is visually active and speaking, **When** ASD features are extracted, **Then** `max_asd_score_target_candidate` is high (≥ 0.6).
3. **Given** ASD extraction is not run, **When** `build_av_feature_table.py` runs without `--asd-features` argument, **Then** the pipeline completes normally with ASD columns as NaN and a warning logged.

---

### Edge Cases

- What happens when a clip's video file is missing or corrupted? → Clip still appears in feature table with visual features as NaN and `visual_eligible = False`.
- What happens when a clip contains no detectable faces at all? → Face columns are 0, `visual_eligibility_score` is 0, `off_camera_likely_score` is 1 by default.
- What happens when a clip contains multiple visible people of similar size? → `n_face_tracks` > 1; `target_child_candidate_visible` reflects ambiguity; system does not crash.
- What happens when the audio-only baseline score CSV is missing for some clips? → Those clips are excluded from AV fusion training with a warning; they are still included with NaN audio score in the master table.
- What happens when a stratum (e.g., 34–38 month visually eligible clips) has too few test samples for reliable metrics? → Metrics are still computed but annotated with `n_clips` count; user is responsible for interpreting small-n strata.
- What happens when visual_eligibility threshold tuning produces a threshold that marks all clips ineligible? → The gated AV model degrades to audio-only for all clips; this is a valid (and informative) result.
- What happens when the val set AV model outperforms audio-only but the test set does not? → Both results are reported without suppression; the discrepancy is recorded.

---

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The system MUST extract a structured visual feature row for every clip with a valid video path, including face detection statistics, face track statistics, visual quality, child visibility, off-camera likelihood, and a composite visual eligibility score.
- **FR-002**: The system MUST compute a `visual_eligibility_score` per clip and expose a `visual_eligible` binary flag derived from a threshold tuned only on the validation split.
- **FR-003**: The system MUST assemble a master feature table that merges clip metadata, labels, group-wise train/val/test split, audio-only baseline scores, and visual features into a single tabular artifact per clip.
- **FR-004**: The system MUST preserve group-wise child splits throughout: no clip from a given child_id may appear in more than one of train, val, test.
- **FR-005**: The system MUST train and produce separate saved models for: audio-only, video-only, always-fuse AV, and gated AV classifiers.
- **FR-006**: The system MUST tune classification thresholds using only the validation split and never expose the test split to any training or tuning decision.
- **FR-007**: The gated AV model MUST use audio-only prediction for clips where `visual_eligible = False` and AV prediction for clips where `visual_eligible = True`.
- **FR-008**: The evaluation script MUST report AUROC, AUPRC, F1, precision, recall, and balanced accuracy for each model class on the overall test set and on the following strata: 14–18 month, 34–38 month, visually eligible, visually ineligible, off-camera likely.
- **FR-009**: The error analysis script MUST categorize test clips into: AV-helped (audio FP/FN corrected by video), AV-hurt (video introduces new errors), off-camera miss, multi-face ambiguity; and produce a structured table of these examples.
- **FR-010**: The system MUST support a null or subset-limited result without suppression: if AV does not improve global performance, all metrics must be reported faithfully.
- **FR-011**: The system MUST handle missing video gracefully: clips with no video file produce sentinel visual features and are treated as `visual_eligible = False`.
- **FR-012**: ASD feature extraction MUST be optional: the pipeline must complete without ASD features; ASD columns are treated as NaN when the ASD script has not been run.
- **FR-013**: Visual eligibility threshold MUST be tuned only on validation data; the same threshold must be applied unchanged to the test set.
- **FR-014**: All trained models, thresholds, and feature tables MUST be saved to disk so evaluation can be re-run from saved artifacts without re-training.

### Key Entities

- **Clip**: The unit of prediction. Has a clip_id, child_id, age_band, split, audio path, video path, label (0/1 target child vocalized), optional duration, optional existing audio score.
- **VisualFeatureRow**: One row per clip. Contains all face detection, tracking, quality, visibility, eligibility, and (optionally) ASD features for that clip.
- **MasterFeatureTable**: Merged table of Clip metadata + VisualFeatureRow + audio baseline score + labels + split. Input to all models.
- **FusionModel**: One of {audio_only, video_only, always_fuse_av, gated_av}. Trained on the train split, threshold-tuned on the val split, evaluated on the test split. Serialized as a pkl file.
- **VisualEligibilityThreshold**: A single scalar tuned on the val split that converts `visual_eligibility_score` to the binary `visual_eligible` flag.
- **EvaluationReport**: Collection of metrics JSON/CSV files produced by evaluating FusionModels on the test split, stratified by age band, visual eligibility, off-camera status, and failure mode.

---

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Every clip in the metadata CSV has a corresponding row in `visual_features.csv` — zero clips are silently dropped.
- **SC-002**: `av_master_features.csv` contains no cross-child split leakage — verified by checking that all rows sharing a child_id have the same value in the `split` column.
- **SC-003**: All four model classes (audio-only, video-only, always-fuse AV, gated AV) produce non-NaN AUROC on the test set overall and on each of the two age bands.
- **SC-004**: Stratified metrics are reported for at least five strata beyond overall: 14–18 month, 34–38 month, visually eligible, visually ineligible, and off-camera likely.
- **SC-005**: Error analysis produces at least one populated row in each of the four error-mode categories (AV-helped, AV-hurt, off-camera miss, multi-face confusion), or explicitly documents which categories have zero examples and why.
- **SC-006**: The gated AV model's test performance on visually eligible clips is reported separately and can be compared to audio-only on the same eligible subset — enabling the core thesis claim to be evaluated.
- **SC-007**: The full evaluation pipeline (from saved model artifacts) can be re-run reproducibly and produces the same metrics on successive runs given the same test split.
- **SC-008**: The system produces at least one plot or structured table per model class showing the precision-recall tradeoff and the ROC curve, formatted for thesis inclusion.
- **SC-009**: If audio-visual fusion shows no improvement on the overall test set, the system clearly reports this and the stratified breakdown is sufficient to explain whether any subgroup benefits.
- **SC-010**: Manual annotation subset creation (target child visible, face visible, off-camera, visual quality) is supported for at least a small diagnostic set; the annotation schema is documented.

---

## Assumptions

- Video files exist for SAILS BIDS data clips (`.mp4`); Providence and Playlogue are audio-only and will have `video_path = None`, treated as visually ineligible by default.
- The existing metadata CSV (`whisper-modeling/seen_child_splits/`) is the authoritative source for clip_id, child_id, timepoint, audio_path, and labels; video_path can be derived or joined from a separate video manifest.
- Audio-only baseline scores from existing pipeline runs (e.g., BabAR enrollment probabilities, WavLM direct classifier scores) are already computed and available as CSVs; this system does not re-run audio-only training.
- The labeled training set is approximately 1,500 clips total across both age bands and all splits; models must be small enough to train reliably at this scale.
- Face detection is expected to work on infant/toddler faces but with lower reliability than adult face benchmarks; the system accounts for this by making eligibility gating a required component rather than an afterthought.
- The target child identity is not assumed to be known at visual feature extraction time; child identification relies on face-size heuristics (smallest face track) rather than enrollment-based face recognition.
- AV foundation models (AV-HuBERT, VideoMAE) may or may not be tractable given GPU budget; the minimum viable path uses face detection + tracking + visual quality features only.
- For audio-only baseline, the best-performing audio model from existing pipeline evaluation (likely BabAR or WavLM-based) is used as the audio feature/score input to fusion; this choice does not need to be re-optimized here.
- Evaluation outputs are intended for thesis inclusion and do not need to meet production latency or deployment requirements.
- If a video clip has faces but none can be confidently assigned to the target child (e.g., three adult faces, no small face), `target_child_candidate_visible` is 0 and `visual_eligibility_score` is low; this is a valid and informative outcome.
