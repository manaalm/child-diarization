# Feature Specification: MIL Extensions — Aggregation Ablations, Transformer MIL, and Weak Diarization

**Feature Branch**: `005-mil-extensions`  
**Created**: 2026-04-24  
**Status**: Draft  

## User Scenarios & Testing *(mandatory)*

### User Story 1 — Gated Attention Ablation (Priority: P1)

A researcher running the segment-instance MIL sweep wants to compare plain softmax attention (ABMIL) against gated attention (Ilse et al. 2018, ICML) on each of the four diarizer frontends. The gated variant adds a sigmoid gate to the attention network, providing a learned filter that suppresses irrelevant segments. At ~1,500 training samples, gated attention is expected to be more stable than plain softmax attention. Both variants share the same backbone and RTTM-based bag construction; the only difference is the attention head.

**Why this priority**: Cheapest addition with the highest expected return. Extends the existing 16-cell sweep by 4 cells (one per frontend), requires no new data or infrastructure, and directly tests the Ilse et al. small-data stability claim. The comparison is needed to decide which attention variant to use in all downstream experiments.

**Independent Test**: Can be fully tested by adding `gated_attention` to the aggregators list in the sweep config and verifying that per-config result files and `all_configs.json` are written correctly with distinct AUROC values for `attention` vs. `gated_attention` rows.

**Acceptance Scenarios**:

1. **Given** the existing sweep config lists four aggregators, **When** `gated_attention` is added and the sweep runs, **Then** 4 new result directories appear under `mil/mil_results/seg_mil/` and `all_configs.json` gains 4 rows tagged `aggregator: gated_attention`.
2. **Given** the sweep completes, **When** AUROC values are compared between `attention` and `gated_attention` for the same frontend, **Then** the difference is measurable and directionally consistent across ≥3 of 4 frontends.
3. **Given** a previously-completed config's result directory already exists, **When** the sweep is re-run, **Then** that config is skipped without overwriting.

---

### User Story 2 — Age-Band Aggregation Ablation (Priority: P2)

A researcher wants to know whether different aggregation functions (max, gated attention, noisy-OR, top-k) perform differently on 14-month clips versus 36-month clips. The hypothesis is that sparse-vocalization clips (14-month) favor max/noisy-OR pooling while dense-vocalization clips (36-month) favor attention/top-k. Confirming this would be a developmental finding: the optimal MIL pooling strategy depends on how much the child is vocalizing, which is a function of age.

**Why this priority**: The hypothesis is specific, testable with existing data, and would be publishable as a standalone result. Requires running inference on age-stratified subsets using already-trained models — no retraining needed.

**Independent Test**: Can be tested by slicing the test set by timepoint label and computing per-aggregator metrics on each slice. Passes if per-age-band metric files are written and the aggregator ranking differs between 14-month and 36-month subsets in the predicted direction for ≥2 of 4 frontends.

**Acceptance Scenarios**:

1. **Given** trained MIL models (all frontends × all aggregators), **When** inference runs on 14-month test clips only, **Then** per-aggregator AUROC values are written to a results file covering only that age band.
2. **Given** the same models, **When** inference runs on 36-month test clips only, **Then** per-aggregator AUROC values for the 36-month slice are written separately.
3. **Given** both slices are computed, **When** aggregator rankings are compared, **Then** the ranking difference (max/noisy-OR relatively stronger at 14 mo) appears in at least 2 of 4 frontends.
4. **Given** noisy-OR and top-k are not yet implemented, **When** they are added to the sweep, **Then** they produce valid predictions on both age bands without NaN loss or empty-bag crashes.

---

### User Story 3 — Transformer MIL (Priority: P3)

A researcher wants segment instances to attend to each other before aggregation, capturing turn-taking structure (a segment is more likely target-child if it follows a parental utterance). This requires a small transformer encoder applied to the bag of segment embeddings, followed by a pooling step. Because transformers overfit at ~1,500 training samples, dropout, weight decay, and early stopping must be configured. This experiment runs only after simpler aggregators are benchmarked.

**Why this priority**: Potentially captures relational structure that order-agnostic pooling cannot, but carries higher risk of overfitting. Value depends on whether attention/gated-attention results (US1) show a ceiling that simpler aggregators cannot break.

**Independent Test**: Can be tested by adding a `transformer` aggregator to the sweep config and verifying that test AUROC is reportable for all 4 frontends and that training loss decreases without NaN.

**Acceptance Scenarios**:

1. **Given** `transformer` is added to the sweep config, **When** the sweep runs, **Then** transformer MIL trains to convergence (loss decreasing) on all 4 frontends without NaN.
2. **Given** the run completes, **When** test AUROC for `transformer` is compared to `gated_attention` on the best frontend, **Then** the comparison is available in `all_configs.json`.
3. **Given** overfitting risk, **When** training runs, **Then** dropout and weight-decay values are logged in `config.json` and early stopping activates when validation loss does not improve for `patience` epochs.

---

### User Story 4 — Weakly-Supervised Frame-Level Prediction (Priority: P4)

A researcher wants to evaluate whether MIL attention weights — trained only on clip-level labels — correlate with true target-child regions as indicated by ground-truth RTTM files. If per-segment attention scores align with ground-truth child-speech timing, this constitutes a target-child diarizer trained only on clip labels, extending Zhu et al. 2021 to the 36-month age band they did not cover.

**Why this priority**: A novel contribution with no additional training cost. Requires only an evaluation script that maps per-segment attention weights to frame-level predictions and compares against RTTM ground truth. Depends on trained attention-variant models from US1.

**Independent Test**: Can be tested by running an evaluation script that reads saved attention weight CSVs and ground-truth RTTMs, then reports frame-level precision/recall/F1 or Pearson correlation between attention score and ground-truth child-speech fraction per segment.

**Acceptance Scenarios**:

1. **Given** trained models with attention-variant aggregators, **When** the weak-diarization evaluator runs on the test set, **Then** per-segment attention weights are compared to ground-truth RTTM child-speech coverage and a correlation metric is reported.
2. **Given** the evaluation runs, **When** results are inspected, **Then** the script uses only saved attention weight CSVs — no retraining, no recomputation of embeddings.
3. **Given** evaluation completes, **When** metrics are reported, **Then** results are stratified by age band (14 mo vs. 36 mo) to match the Zhu et al. 2021 comparison.

---

### User Story 5 — TinyVox-Pretrained MIL Scorer (Priority: P5)

A researcher wants to pretrain the per-segment instance scorer on TinyVox child-speech data before fine-tuning with MIL on clip-label data, providing a better initialization than adult-speech foundation model weights. This experiment is only run if tier-1 and tier-2 results plateau and TinyVox data is available and preprocessed.

**Why this priority**: Higher engineering cost with harder-to-isolate contribution. Potentially meaningful if adult-pretrained encoders show a consistent gap vs. child-targeted pretraining. Explicitly gated on TinyVox availability and tier-1/2 AUROC ceiling.

**Independent Test**: Can be tested by comparing MIL AUROC on the best frontend/aggregator combination using a TinyVox-pretrained encoder vs. the standard adult-pretrained encoder. Passes if the pretrained scorer achieves measurably different AUROC on ≥2 of 4 frontends.

**Acceptance Scenarios**:

1. **Given** TinyVox data is downloaded and preprocessed, **When** a pretraining job runs, **Then** a pretrained checkpoint is produced and logged with training loss.
2. **Given** the checkpoint exists, **When** the MIL sweep runs with the TinyVox-pretrained encoder, **Then** results are written alongside adult-encoder results for direct comparison.
3. **Given** both runs complete, **When** AUROCs are compared, **Then** the delta (positive or negative) is reported per frontend in `all_configs.json`.

---

### User Story 6 — End-to-End MIL with Learned Instance Proposers (Priority: P6)

A researcher wants to replace fixed diarizer frontends with a learned segmenter trained jointly with the MIL classifier, potentially recovering from diarization errors that limit fixed-frontend performance. This is only pursued if one frontend is a clear bottleneck in tier-1 results.

**Why this priority**: Highest engineering cost. Only warranted if tier-1 data shows a specific frontend is the binding constraint rather than the MIL classifier itself. Explicitly out of scope unless that condition is met.

**Independent Test**: Can be tested by running the end-to-end model on the same train/val/test split and confirming test AUROC exceeds the best fixed-frontend result on at least one age band.

**Acceptance Scenarios**:

1. **Given** tier-1 results show a clear frontend bottleneck, **When** the end-to-end model trains, **Then** it produces non-degenerate segment proposals and valid clip predictions.
2. **Given** end-to-end training completes, **When** test AUROC is compared to the best fixed-frontend run, **Then** the comparison is logged in a results file parallel to `all_configs.json`.

---

### Edge Cases

- What happens when a clip has no segments from a given frontend (empty bag)? All aggregators must handle empty bags without crashing, returning a configurable default score.
- What happens when top-k aggregation receives a bag smaller than k? The implementation must clamp k to the actual bag size.
- What happens when transformer MIL receives a bag with exactly 1 instance? Self-attention over a single token must not produce NaN.
- What happens when the attention weight CSV is missing for a clip during weak-diarization evaluation? The evaluator skips the clip with a warning rather than crashing.
- What happens if TinyVox pretraining diverges? The main MIL sweep must still run with the standard encoder; the failure is logged but does not block other experiments.
- What happens when age-band subsets are very small (fewer than 20 test clips)? Results include the sample size count alongside metrics so the reader can assess reliability.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The system MUST support `gated_attention` as a named aggregator in the sweep config, producing per-config result directories and rows in `all_configs.json`.
- **FR-002**: The system MUST support `noisy_or` and `top_k` as additional aggregators; `top_k` accepts a configurable k parameter.
- **FR-003**: The system MUST support a `transformer` aggregator with configurable depth, dropout, and weight-decay; these hyperparameters MUST be logged in `config.json` for each run.
- **FR-004**: The system MUST produce per-age-band metric files (14-month and 36-month slices) for every frontend × aggregator combination, using already-trained model weights without retraining.
- **FR-005**: The system MUST provide a weak-diarization evaluation script that reads saved attention weight CSVs and ground-truth RTTMs and outputs frame-level metrics stratified by age band.
- **FR-006**: The sweep MUST remain resume-safe: any config whose result directory already contains `test_metrics.json` is skipped on re-run without overwriting.
- **FR-007**: All aggregators MUST handle empty bags without crashing, returning a configurable default prediction score.
- **FR-008**: The `all_configs.json` summary MUST be updated after every successfully completed config, not only at sweep end.
- **FR-009**: Experiments gated on prior results (Transformer MIL, TinyVox scorer, end-to-end) MUST be documented as conditional in both the SLURM scripts and the sweep config.

### Key Entities

- **Aggregator**: A pooling function applied over a variable-length bag of segment embeddings to produce a clip-level score. Variants across this feature: gated_attention, noisy_or, top_k, transformer.
- **Age Band**: A developmental stratum of the dataset; values are `14_month` and `36_month`. Used to slice test-set inference without retraining.
- **Attention Weight CSV**: Per-config output file storing (clip_id, segment_index, start, end, weight) rows; input to the weak-diarization evaluator.
- **Weak Diarization Score**: Frame-level or segment-level target-child prediction derived from MIL attention weights, without frame-level training supervision.
- **TinyVox Pretrained Encoder**: A per-segment encoder initialized from child-speech pretraining rather than adult-speech foundation model weights.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: The gated attention ablation (US1) produces a measurable AUROC difference vs. plain attention on ≥3 of 4 frontends within the same per-config compute budget as the baseline sweep.
- **SC-002**: The age-band ablation (US2) reports per-aggregator AUROC on both 14-month and 36-month subsets for all trained models, with aggregator rankings compared across age bands.
- **SC-003**: Transformer MIL (US3) trains to convergence without NaN loss on all 4 frontends, and its AUROC is available in `all_configs.json` for comparison against simpler aggregators.
- **SC-004**: The weak-diarization evaluation (US4) reports a Pearson correlation or frame-level F1 between MIL attention weights and ground-truth RTTM child-speech regions, stratified by age band.
- **SC-005**: All new aggregators (gated_attention, noisy_or, top_k, transformer) produce non-degenerate predictions on ≥95% of test clips.
- **SC-006**: Adding new aggregators does not invalidate or overwrite result directories from the existing 16-config baseline sweep.

## Assumptions

- The existing 16-config sweep (`mil/mil_results/seg_mil/`) has completed successfully and its results are committed; new experiments extend rather than replace this baseline.
- The `seen_child_splits/` metadata includes timepoint labels (`14_month`, `36_month`) enabling age-band stratification of inference without re-splitting.
- Ground-truth RTTMs for test-set clips are available in at least one of the four existing RTTM cache directories for weak-diarization evaluation.
- All new aggregators share the existing WavLM-base+ segment embedding cache (`mil/seg_embedding_cache/`); no re-embedding is required.
- Transformer MIL uses positional encoding based on segment order (sorted by start time), since temporal position within the clip is semantically meaningful.
- TinyVox experiments (US5) are explicitly gated on TinyVox data being downloaded, extracted, and validated (non-empty WAV files with usable annotations).
- End-to-end MIL (US6) is out of scope unless tier-1/2 results identify a specific frontend as the binding performance constraint.
