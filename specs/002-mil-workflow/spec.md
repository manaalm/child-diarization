# Feature Specification: Multiple Instance Learning Workflow

**Feature Branch**: `002-mil-workflow`
**Created**: 2026-04-23
**Status**: Draft
**Input**: User description: "can you add a multiple instance learning workflow"

## User Scenarios & Testing *(mandatory)*

### User Story 1 — MIL Model Training (Priority: P1)

A researcher trains a weakly-supervised child presence detector that learns directly
from clip-level labels (child present / absent) without any frame-level annotations.
The researcher provides a split of annotated audio clips, and the system produces a
trained model that can score new clips for child presence.

**Why this priority**: This is the core novel contribution. The existing diarization
pipelines require either pre-trained speaker models or fine-tuned frame classifiers.
MIL removes that dependency — a researcher with only clip-level labels can train a
competitive child presence detector from scratch.

**Independent Test**: Run the training script on the seen-child train/val split;
confirm a checkpoint and config file are written; confirm val-set F1 is reported at
the end of training.

**Acceptance Scenarios**:

1. **Given** the seen-child train split with clip-level binary labels, **When** the
   researcher runs the MIL training script, **Then** the model trains to completion,
   val-set metrics are printed per epoch, and the best checkpoint is saved with a
   config file.
2. **Given** a prior checkpoint, **When** the researcher reruns training with the same
   config and seed, **Then** the val-set metrics match within floating-point tolerance
   (reproducibility).
3. **Given** a clip whose audio file is missing or corrupt, **When** that clip is
   encountered during training, **Then** the system logs a warning and skips it rather
   than crashing.

---

### User Story 2 — Comparative Evaluation Against Baselines (Priority: P2)

A researcher evaluates the trained MIL model on the held-out test split and compares
the resulting F1, AUROC, and AUPRC values against all existing diarization-based
baselines (USC-SAIL, BabAR, VTC, Pyannote, VBx) using the same evaluation harness.

**Why this priority**: Thesis impact depends on a credible, apples-to-apples comparison.
The MIL results must appear in the same result tables as the other systems.

**Independent Test**: Run the MIL evaluation script on the test split; confirm
`enroll_test_metrics.json` (or equivalent) exists with F1/AUROC/AUPRC fields matching
the format already used by existing result folders.

**Acceptance Scenarios**:

1. **Given** a trained MIL checkpoint, **When** the researcher runs evaluation on the
   test split, **Then** the system produces clip-level probability scores and binary
   predictions for every test clip.
2. **Given** the test predictions, **When** metrics are computed, **Then** F1,
   precision, recall, AUROC, and AUPRC are written to a JSON file in the canonical
   result folder format (mirroring existing `*_ecapa_enrollment_runs/` structure).
3. **Given** the MIL result folder, **When**
   `evaluation/aggregate_thesis_tables.py` is run, **Then** MIL metrics appear
   in the comparative baseline table without manual edits.

---

### User Story 3 — Age-Stratified MIL Analysis (Priority: P3)

A researcher obtains separate MIL performance metrics for the 12-16 month and 34-38
month age cohorts, matching the age-stratified evaluation already run for other
diarizers.

**Why this priority**: The thesis has a dedicated age-stratified chapter (US2 in the
existing plan). MIL results must feed into that chapter alongside existing diarizer
results.

**Independent Test**: Run age-stratified MIL evaluation for both age groups; confirm
per-cohort result subdirectories exist with the same metric files produced for other
diarizers by `unified_age_stratified.py`.

**Acceptance Scenarios**:

1. **Given** a trained MIL model and the age-annotated manifests, **When** the
   researcher runs age-stratified evaluation for `12_16m`, **Then** a result folder
   for that cohort is produced with F1/AUROC/AUPRC.
2. **Given** both age-group results, **When** metrics are compared, **Then** the
   difference on at least one metric is ≥ 0.05 (SC-002 threshold, same as other
   diarizers).

---

### Edge Cases

- What happens when a test clip has no audio above the silence threshold? The system
  should assign a near-zero child presence score rather than error.
- What happens when the training split has severe class imbalance (e.g., a child's
  recordings are all child-absent)? Training should proceed; any resulting prototype
  collapse should be logged.
- What happens when a clip is shorter than one audio window? The clip should be
  treated as a single instance (no windowing error).
- What happens when the MIL model is run on a Providence or Playlogue clip (no video)?
  Since MIL is audio-only, it should work identically to SAILS clips.

---

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST train a clip-level binary classifier from clip-level child
  presence labels without requiring frame-level or segment-level annotations.
- **FR-002**: System MUST partition each audio clip into overlapping fixed-length
  windows and extract a feature vector per window using one of the pre-trained audio
  encoders already present in the project (same encoders used by the existing baseline
  models).
- **FR-003**: System MUST aggregate window-level feature vectors into a single
  clip-level prediction via a learned attention mechanism that assigns higher weight
  to windows judged more informative for child presence detection.
- **FR-004**: System MUST train and evaluate using the `seen_child_splits/` train/val/test
  partition (identical to all other diarization-based systems) with seed=42 for
  reproducibility.
- **FR-005**: System MUST output continuous child presence probability scores and
  threshold-tuned binary predictions for every clip in the val and test splits.
- **FR-006**: System MUST report F1, precision, recall, AUROC, and AUPRC on val and
  test splits, plus per-timepoint (14-month, 36-month) breakdowns.
- **FR-007**: System MUST save a best checkpoint (selected by val-set F1 or AUROC)
  and a `config.json` copy alongside all result files.
- **FR-008**: Result files MUST mirror the folder structure and field names of existing
  result directories so that `evaluation/aggregate_thesis_tables.py` can include MIL
  results without code changes.
- **FR-009**: System MUST support running with each of the audio feature backbones
  already used in the project (at minimum the two that produced the strongest existing
  baseline results) so that backbone choice can be reported as a controlled variable.
- **FR-010**: Age-stratified evaluation MUST be producible by filtering the test split
  on age group, using the same age manifests as the other diarizers.

### Key Entities

- **Bag**: One audio clip; carries a binary label (child present = 1, absent = 0).
- **Instance**: One fixed-length audio window extracted from a bag; carries no label.
- **Instance Feature**: A dense vector representation of one instance, produced by a
  pre-trained audio encoder.
- **Attention Weight**: A scalar learned per instance indicating its relevance to the
  child presence prediction for its bag.
- **MIL Prediction**: A bag-level probability score ∈ [0, 1] produced by the
  attention-weighted aggregation of instance features passed through a classification
  head.
- **MIL Result Folder**: Directory containing `config.json`, `val_metrics_tuned.json`,
  `test_metrics_tuned.json`, `test_predictions.csv`,
  `test_metrics_by_timepoint.csv`, mirroring existing baseline result folders.

---

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: The best MIL variant achieves F1 ≥ 0.850 on the seen-child test split
  (matching or exceeding the lowest existing baseline), demonstrating competitive
  performance without diarization.
- **SC-002**: MIL results are directly comparable to existing baselines — same split,
  same metric definitions, same output JSON format — verified by successful ingestion
  into `aggregate_thesis_tables.py` without code changes.
- **SC-003**: Age-stratified MIL metrics are produced for both 12-16 month and 34-38
  month cohorts, with the inter-cohort performance difference reported for at least
  F1 and AUROC.
- **SC-004**: MIL model runs are reproducible: rerunning with the same config and
  seed produces test metrics that agree to ≥ 3 decimal places.
- **SC-005**: At least two backbone variants are evaluated, and the per-backbone
  results are reported in a single comparison table so the thesis can characterize
  backbone sensitivity.

---

## Assumptions

- The existing seen-child split CSV files and age manifests are already committed and
  stable — MIL training will use them as-is.
- MIL is audio-only; it does not require video or face tracks (runs on all three
  datasets: SAILS, Providence, Playlogue).
- The fixed-length window size and stride (e.g., 1 s windows, 0.5 s stride) are
  treated as hyperparameters tunable on the val set; the chosen values will be
  documented in `config.json`.
- Bag-level labels are the clip-level `label` column in the existing split CSVs
  (0 = child absent, 1 = child present) — no new annotation is needed.
- MIL training runs on a single GPU node (same SLURM resources as existing training
  jobs) and completes within 12 hours.
- MIL is positioned as a complementary baseline to the diarization-based systems,
  not a replacement — both approaches are reported in the thesis.
- The two backbone variants to compare are determined by the existing baseline
  results (best-performing encoder configurations); this choice is documented in
  the plan.
