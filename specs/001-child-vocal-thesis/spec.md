# Feature Specification: Child Vocalization Extraction & Synthesis Thesis

**Feature Branch**: `001-child-vocal-thesis`
**Created**: 2026-04-17
**Status**: Draft
**Input**: User description: "Research for thesis focused on supervised extraction of child
vocalizations from noisy child-adult home videos (core dataset, no RTTM files)..."

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Cross-Dataset Vocalization Detection (Priority: P1)

A thesis researcher trains child vocalization detection models on annotated datasets
(Providence, Playlogue, Seedlings, optionally TinyVox) and applies them to an unlabeled
core home video dataset. The system predicts child vocalization segments, evaluates
quantitatively on held-out labeled data, and produces inference outputs for the core dataset.

**Why this priority**: This is the central thesis contribution — demonstrating that models
trained on labeled recordings generalize to real, noisy, unannotated home video data.
It directly extends the existing baseline work (basic encoders + diarization models) to
a harder, more realistic setting.

**Independent Test**: Train on Providence/Playlogue/Seedlings labeled split; evaluate on
held-out test split (F1/AUROC/AUPRC reported per dataset and in aggregate); run inference
on core dataset and produce RTTM output files.

**Acceptance Scenarios**:

1. **Given** a test-split audio recording from Providence/Playlogue/Seedlings with ground
   truth RTTM, **When** the detection pipeline runs, **Then** child vocalization segments
   are predicted with F1 ≥ 0.875 vs. the ground truth.
2. **Given** an unlabeled home video from the core dataset (12-16 months or 34-38 months
   session), **When** inference runs, **Then** an RTTM file is produced with timestamped
   child vocalization segments and no crash occurs even if no child speech is detected.
3. **Given** the same experiment config with a fixed random seed, **When** the pipeline
   is run on two separate cluster jobs, **Then** prediction files and evaluation metrics
   are byte-identical.

---

### User Story 2 - Age-Stratified Analysis (Priority: P2)

A researcher runs separate training and evaluation for two developmental age cohorts —
12-16 months and 34-38 months — and compares detection performance across them to
motivate age-specific modeling as a thesis contribution.

**Why this priority**: Vocal characteristics differ dramatically between 12-16 month
infants (babbling, non-linguistic vocalizations) and 34-38 month toddlers (early words,
more structured speech). Demonstrating this gap quantitatively and addressing it is a
novel thesis angle beyond the existing work.

**Independent Test**: Run age-stratified evaluation on the labeled test splits, producing
separate F1/AUROC/AUPRC per age group. Compare age-specific vs. age-agnostic models.

**Acceptance Scenarios**:

1. **Given** Providence/Playlogue/Seedlings recordings annotated with child age, **When**
   age-stratified evaluation runs, **Then** separate metrics are produced for 12-16 month
   and 34-38 month cohorts.
2. **Given** models trained on age-specific subsets vs. the full combined training set,
   **When** evaluated on the held-out test split, **Then** a performance comparison table
   is produced documenting whether age conditioning improves or degrades results.
3. **Given** results for both age groups, **When** statistical analysis runs, **Then** a
   documented comparison (e.g., effect size or confidence intervals) characterizes
   whether differences are meaningful.

---

### User Story 3 - Child Speech Synthesis System (Priority: P3)

A researcher builds a child speech synthesis model trained on labeled segments from
Providence/Playlogue/Seedlings and evaluates it as a standalone thesis contribution:
generating age-conditioned child speech samples and measuring quality via objective
metrics (MCD, speaker similarity, age-group discriminability).

**Why this priority**: A child speech synthesis system is independently novel — existing
TTS systems do not target infant/toddler speech. Demonstrating age-conditioned generation
with measurable quality establishes a self-contained contribution before augmentation
experiments.

**Independent Test**: Train synthesis model on child speech segments; generate samples
for both age groups; compute MCD vs. ground-truth held-out segments and cosine speaker
similarity; verify samples are distinguishable by age group via an age classifier.

**Acceptance Scenarios**:

1. **Given** labeled child speech segments from Providence/Playlogue/Seedlings grouped
   by age (12-16 months, 34-38 months), **When** the synthesis model trains, **Then**
   it generates audio files for both age groups stored in a versioned output directory.
2. **Given** generated samples vs. held-out ground-truth child speech, **When** objective
   quality metrics are computed, **Then** MCD and speaker similarity scores are reported
   and compared against a reconstruction baseline (ground truth vs. vocoded ground truth).
3. **Given** generated samples from each age group, **When** an age-group classifier
   (trained on real speech) scores them, **Then** age-group accuracy on synthetic samples
   is reported, documenting whether age conditioning is effective.
4. **Given** the synthesis config and fixed seed, **When** the pipeline runs twice, **Then**
   the same synthetic samples are produced (deterministic generation).

---

### User Story 3b - Synthesis Augmentation for Detection (Priority: P3)

A researcher augments detection model training data with synthetic child speech from
User Story 3 and evaluates whether this improves vocalization detection F1/AUROC —
particularly for the data-scarce 12-16 month cohort — closing the loop between
synthesis and detection.

**Why this priority**: Augmentation directly answers "does synthesis help?" — a critical
thesis question. The win/loss/neutral result is valuable regardless of direction, and
it provides a concrete experimental connection between the synthesis and detection chapters.

**Independent Test**: Train detection model on (original + synthetic) data; compare
F1/AUROC/AUPRC on the same held-out test split as the non-augmented baseline; repeat
for each age group separately.

**Acceptance Scenarios**:

1. **Given** synthetic samples from User Story 3 and original training data, **When**
   a detection model trains on the combined set, **Then** F1/AUROC/AUPRC on the labeled
   test split is documented and compared against the original-data-only baseline.
2. **Given** augmented vs. non-augmented detection models, **When** results are
   stratified by age group, **Then** separate augmentation deltas are reported for
   12-16 month and 34-38 month cohorts.
3. **Given** a negative augmentation result (synthesis does not improve detection),
   **Then** an error analysis identifies why (e.g., domain mismatch, distribution shift)
   and the finding is documented as a thesis contribution.

---

### User Story 4 - Unified Evaluation Framework (Priority: P4)

A researcher has a single cohesive evaluation framework that supports all experiment
types (detection/diarization, age stratification, synthesis quality, synthesis
augmentation, core dataset proxy analysis), produces all thesis-ready metric tables
from committed output files, and ensures zero manual transcription between experimental
results and thesis content.

**Why this priority**: Framework coherence distinguishes a thesis from a collection of
disconnected scripts. It also ensures the thesis satisfies the reproducibility and
thesis-synchronization requirements of the project constitution.

**Independent Test**: Running the end-to-end evaluation script from committed configs
produces all thesis metric tables (by dataset, by age group, with/without synthesis
augmentation) as committed CSV/JSON files.

**Acceptance Scenarios**:

1. **Given** completed experiments for all three research directions, **When** the unified
   evaluation script runs, **Then** all results are saved to versioned output files under
   canonical folders and are reproducible from committed configs.
2. **Given** the output files, **When** thesis tables are assembled, **Then** every number
   traces to a specific committed CSV/JSON row without manual transcription.

---

### Edge Cases

- What happens when a core dataset recording contains no audible child speech? The system
  MUST produce an empty or near-empty RTTM without crashing.
- How are very short infant vocalizations (< 100ms, common at 12-16 months) handled?
  Post-processing minimum duration thresholds may need adjustment for infant data.
- What if TinyVox audio has different recording conditions (microphone type, room acoustics)
  from the training datasets? Domain mismatch must be assessed and documented.
- How are child-adult overlap segments handled during synthesis training? Clear policy
  needed (exclude overlapping segments vs. treat as child-only).
- What happens if a child in the core dataset has no analog in the enrollment training set
  (zero-shot enrollment scenario)?

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The system MUST support training vocalization detection models on any
  combination of Providence, Playlogue, Seedlings, and TinyVox datasets with RTTM labels.
- **FR-002**: The system MUST apply trained detection models to unlabeled recordings
  (core home video dataset) and produce RTTM files with child vocalization timestamps.
- **FR-003**: The system MUST evaluate detection performance (F1, Precision, Recall,
  AUROC, AUPRC) separately for 12-16 month and 34-38 month age groups on held-out
  labeled test splits — aggregate-only reporting is insufficient.
- **FR-004**: The system MUST include ablation studies comparing: (a) encoder architecture
  (Whisper vs WavLM), (b) diarization frontend (USC-SAIL vs Pyannote vs BabAR),
  (c) with and without enrollment-based personalization, (d) with and without
  age-stratified training.
- **FR-005**: The system MUST include a child speech synthesis module that: (a) trains on
  labeled child speech segments from annotated datasets, (b) generates samples conditioned
  on age group (12-16 months, 34-38 months), and (c) is evaluated as a standalone
  contribution via objective quality metrics (MCD, speaker similarity, age-group
  discriminability) independent of any augmentation use.
- **FR-006**: The system MUST compare detection performance on (original training data
  only) vs. (original + synthetic augmentation) for each age group separately and report
  the delta in F1/AUROC/AUPRC; negative results MUST be analyzed and documented.
- **FR-007**: All experiment configurations MUST be version-controlled and committed
  alongside result files; experiments MUST be reproducible from configs with fixed seeds.
- **FR-008**: The system MUST produce per-child error rate analysis (false positive and
  false negative characterization) for any method proposed as a thesis contribution.
- **FR-009**: The core home video dataset MUST be treated as a demonstration/application
  domain — no binding quantitative claims about detection performance are made on it.
  All quantitative evaluation MUST use held-out labeled test splits from Providence,
  Playlogue, and Seedlings. As supplementary qualitative analysis, the system MUST
  compute proxy metrics on core dataset recordings (enrollment cosine similarity scores,
  detection confidence distributions, cross-frontend agreement between USC-SAIL/Pyannote/
  BabAR) to characterize cross-dataset generalization qualitatively. Proxy metric results
  MUST be clearly labeled as supplementary and not used to support primary thesis claims.
- **FR-010**: All datasets used MUST have documented provenance: source, access method
  or license, age metadata availability, and preprocessing applied (resampling,
  segmentation, RTTM format).

### Key Entities

- **AudioRecording**: A source audio file with metadata (dataset name, child ID, age
  group label, session ID, duration). May or may not have associated RTTM ground truth.
- **ChildVocalizationSegment**: A timestamped audio segment attributed to a child
  speaker, with confidence score, predicted age group, and source flag (ground truth
  or model prediction).
- **SpeakerPrototype**: An ECAPA-TDNN enrollment embedding representing a target child,
  built from training-split vocalization segments and used for enrollment-based scoring.
- **SyntheticSpeechSample**: A generated audio clip targeting child speech for a given
  age group, with associated objective quality metrics (MCD, speaker similarity).
- **ExperimentResult**: A collection of metric files (JSON + prediction CSV) tied to
  a specific config, data split, model variant, and random seed. Must be committed to
  version control under a canonical result folder.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Detection models trained on labeled datasets achieve aggregate F1 ≥ 0.875
  on the held-out test split, matching or exceeding the current best BabAR enrollment
  baseline (F1 = 0.874).
- **SC-002**: Age-stratified evaluation produces distinct metric profiles for 12-16 month
  vs. 34-38 month cohorts, with at least one metric (F1 or AUROC) differing by ≥ 0.05,
  motivating age-specific modeling as a thesis contribution.
- **SC-003**: The synthesis system produces age-conditioned samples with MCD ≤ 8 dB
  vs. held-out ground-truth child speech (reconstruction baseline), and generated
  samples achieve ≥ 70% age-group accuracy when scored by a classifier trained on
  real speech — establishing synthesis as a standalone thesis contribution.
- **SC-003b**: Augmentation experiments are completed and reported for both age groups:
  the augmented model's F1/AUROC delta vs. baseline is documented (positive or negative),
  with error analysis explaining the result — ensuring a complete, honest thesis chapter
  regardless of outcome.
- **SC-004**: Proxy metrics computed on core home video recordings demonstrate qualitative
  cross-dataset generalization: enrollment cosine similarity scores show meaningful
  variation across recordings (not uniformly near-zero or near-one), and at least two
  diarization frontends produce consistent detection patterns (≥ 0.70 inter-frontend
  agreement on child-present vs. child-absent segment classification) on the same
  recordings. Results are presented as qualitative evidence only, not primary claims.
- **SC-005**: All primary experiments (detection comparison, age stratification, synthesis
  augmentation, ablations, error analysis) are fully reproducible from committed configs
  within a single cluster re-run (< 24 hours wall time total).
- **SC-006**: All thesis metric tables and figures are generated from committed output
  CSV/JSON files with zero manual transcription, verifiable by automated diff.

## Assumptions

- Providence, Playlogue, and Seedlings contain sufficient labeled child speech data
  (collectively ≥ 500 child vocalization segments per age group) to train both detection
  and synthesis models.
- Age metadata (approximate month range) is available or derivable for all recordings
  in Providence, Playlogue, and Seedlings to enable age-stratified splits.
- The core home video dataset covers two session types: 12-16 months and 34-38 months
  (one or more recordings per session type).
- TinyVox, if incorporated, can be integrated with minimal preprocessing (resampling to
  16kHz mono, RTTM format alignment).
- The thesis evaluates vocalization detection for a specific **target child** per
  recording (enrollment-based paradigm), consistent with the existing project design.
- Synthesis quality is evaluated via objective metrics (e.g., MCD, speaker similarity
  cosine score) rather than large-scale perceptual studies, which are outside thesis scope.
- The existing USC-SAIL, Pyannote, and BabAR implementations serve as primary baselines;
  this spec extends the research rather than replacing existing infrastructure.
- `uv` is used for all Python environment management; subsystem environments remain
  separate (whisper-modeling, BabAR, Pyannote) per project constitution.
