# Feature Specification: Metadata-Conditioned Routing and Ensemble Extensions

**Feature Branch**: `012-metadata-routing-ensemble`  
**Created**: 2026-04-28  
**Status**: Draft

## Overview

Four targeted improvements to the current ensemble that exploit BIDS metadata and error stratification findings. Listed in priority order (highest expected gain / lowest cost first).

---

## User Scenarios & Testing *(mandatory)*

### User Story 1 — Metadata-Augmented Stacker (Priority: P1)

A researcher wants to know whether adding BIDS scene metadata (n_adults, n_children, task, interaction, location, age band) as features alongside the 11 system scores significantly improves stacking over the current metadata-free LR stack (F1=0.897, AUROC=0.870).

**Why this priority**: Lowest implementation cost, builds directly on existing `ensemble_runs/` infrastructure, and the hypothesis is well-motivated: current stacker treats all clips identically even though per-stratum performance varies dramatically across systems.

**Independent Test**: Can be tested end-to-end on val+test splits with no new models or GPU. Pass when test AUROC or F1 improves over `all_available` LR baseline.

**Acceptance Scenarios**:

1. **Given** per-system `test_predictions.csv` files and the seen-child split metadata CSV, **When** the metadata-augmented stacker is trained on val and evaluated on test, **Then** `test_metrics_tuned.json` is written to `ensemble_runs/metadata_stack/` with valid F1/AUROC/AUPRC values
2. **Given** a completed run, **When** feature importances are read, **Then** at least one metadata feature (e.g., `n_children`, `n_adults`) appears among the top-10 importances, confirming metadata contributed signal
3. **Given** a completed run, **When** test F1 is compared against `all_available` LR stack (0.897), **Then** the result is reported as delta (positive or negative); the run is considered a success regardless of direction as long as the comparison is documented

---

### User Story 2 — Metadata-Conditioned Router (Priority: P1)

A researcher wants a rule-based or learned router that selects which system's score to trust per clip based on BIDS metadata, rather than averaging across all systems. The motivating findings: BabAR F1=0.00 on task=unknown clips; Whisper-MIL F1=0.75 when n_adults≥2; BabAR best on n_children=1.

**Why this priority**: Directly exploits the clearest stratified findings; expected 1–3pp F1 gain over mean ensemble (0.893). No new training data or GPU required.

**Independent Test**: Router assigns each test clip to exactly one system (or small ensemble) based on its metadata; overall test metrics are computed and compared to best_audio_mil mean baseline.

**Acceptance Scenarios**:

1. **Given** metadata features for each test clip and per-system prediction scores, **When** the router is applied using the rule set derived from stratified analysis, **Then** each clip receives a single scalar prediction from its routed system
2. **Given** a completed router run, **When** test F1 is compared against mean ensemble (0.893), **Then** the delta is documented; the router is considered beneficial if delta_F1 > 0 or delta_AUROC > 0
3. **Given** the learned variant (2-layer router trained on val), **When** evaluated on test, **Then** results are reported alongside the rule-based variant so both strategies can be compared

---

### User Story 3 — Multi-Child False-Positive Suppressor (Priority: P2)

A researcher wants to reduce false positives in clips where multiple children are present (n_children≥2), the largest concentration of persistent FPs across all systems. A clip-level classifier trained only on n_children≥2 clips distinguishes target-child-vocalizing from sibling-vocalizing.

**Why this priority**: Error analysis shows persistent FPs concentrate in this stratum; a targeted fix avoids degrading single-child performance. Expected 5–10pp F1 improvement in the multi-child stratum, ~1pp overall.

**Independent Test**: Evaluate separately on the n_children≥2 subset of the test split before and after applying the suppressor.

**Acceptance Scenarios**:

1. **Given** multi-child clips in the training set and their ground-truth labels, **When** the suppressor is trained on this subset, **Then** the model is saved and applied only to multi-child test clips without touching single-child clips
2. **Given** a completed run, **When** metrics are computed on n_children≥2 test clips before and after applying the suppressor, **Then** FP rate decreases and the change is documented
3. **Given** the suppressor is applied, **When** overall (all-clip) metrics are compared against the non-suppressed baseline, **Then** no regression larger than 0.5pp F1 is introduced on single-child or overall test performance

---

### User Story 4 — Short-Vocalization Specialized Head (Priority: P3)

A researcher wants to recover some of the 44 hard false-negative errors caused by vocalizations shorter than 0.5s, which are structurally underrepresented by the existing 1s+ embedding windows. A specialized detector with finer temporal resolution (≤500ms windows) is trained on short-vocalization clips and merged with the main pipeline output at clip level.

**Why this priority**: Addresses a characterized, quantified failure mode (44/81 persistent errors), but introduces the most implementation risk (may add new FPs) and requires the most careful evaluation. Expected ~1–2pp overall if successful.

**Independent Test**: Evaluate separately on clips that contain any vocalization segment shorter than 0.5s (identified from RTTM ground truth) before and after applying the specialized head.

**Acceptance Scenarios**:

1. **Given** clips annotated as containing short vocalizations (<0.5s) and fine-grained frame-level features, **When** the specialized head is trained, **Then** it produces a clip-level score that is merged with the main pipeline score via val-tuned interpolation weight
2. **Given** a completed run, **When** the short-vocalization FN rate (on the 44 hard clips) is compared before and after, **Then** at least 5 of the 44 previously-missed clips are correctly predicted
3. **Given** the merged pipeline, **When** FP rate is measured on clips with no short vocalizations, **Then** FP rate does not increase by more than 2pp over baseline

---

### Edge Cases

- What if a clip has missing/null metadata (e.g., `n_adults` is NaN)? Router and stacker must fall back to the best unconditional system (best_audio_mil mean) for clips with incomplete metadata.
- What if there are no n_children≥2 clips in the val split for training the suppressor? Use train split instead, with held-out val clips for threshold tuning only.
- What if the metadata-augmented stacker overfits val metadata? Report val vs. test delta to detect this; use L2 regularization by default.
- What if the short-vocalization head hurts overall metrics? Apply the head only when the router/stacker also predicts positive (gated application); this bounds the FP increase.

---

## Requirements *(mandatory)*

### Functional Requirements

**Sub-feature B (Metadata-Augmented Stacker — P1):**
- **FR-001**: Stacker MUST load per-system `test_predictions.csv` files for all available systems (≥11 systems) and join them with BIDS metadata from the seen-child split CSV on `audio_path`
- **FR-002**: Metadata features MUST include at minimum: `n_adults`, `n_children`, `timepoint_norm` (age band), `Interaction_with_child`, `Location`, and an indicator for `task=unknown` (derived from `Activity` column)
- **FR-003**: Stacker MUST train LR and GBM variants on the val split, evaluate on test, and write `test_metrics_tuned.json` and `val_metrics_tuned.json` to `ensemble_runs/metadata_stack/`
- **FR-004**: Feature importances MUST be saved alongside metrics so the contribution of metadata vs. system scores can be inspected

**Sub-feature A (Metadata-Conditioned Router — P1):**
- **FR-005**: Rule-based router MUST implement at least the three rules identified by stratified analysis: (task=unknown → Whisper-MIL), (n_adults≥2 → Whisper-MIL), (n_children=1 → BabAR)
- **FR-006**: Learned router MUST train a small classifier on val-split metadata features (no system scores as input) to predict which system achieves lowest per-clip error, then route accordingly
- **FR-007**: Both rule-based and learned router variants MUST write results to `ensemble_runs/metadata_router_{rule,learned}/` with the same output format as existing ensemble runs

**Sub-feature C (Multi-Child FP Suppressor — P2):**
- **FR-008**: Suppressor MUST train exclusively on clips where `n_children≥2`, using the same WavLM embeddings already available from the MIL pipeline
- **FR-009**: At inference, suppressor MUST be applied ONLY to clips where `n_children≥2`; all other clips pass through unchanged
- **FR-010**: Suppressor MUST write stratum-specific metrics (`n_children≥2` subset F1/AUROC before and after) alongside overall metrics

**Sub-feature D (Short-Vocalization Head — P3):**
- **FR-011**: Short-vocalization head MUST use window sizes ≤500ms (compared to 2s baseline) over the same WavLM backbone
- **FR-012**: "Short vocalization" training clips MUST be identified from ground-truth RTTM files (any vocalization segment <0.5s in the positive clips)
- **FR-013**: Merged pipeline score MUST be a val-tuned weighted combination of the main pipeline and the specialized head, NOT a hard replacement
- **FR-014**: FP rate on clips with NO short vocalizations MUST be evaluated separately to bound harm

### Key Entities

- **System score**: Per-clip probability from one of 11 diarization/detection systems; from `{system}_ecapa_enrollment_runs/enroll_test_predictions.csv` or equivalent
- **BIDS metadata**: Per-clip annotations from `whisper-modeling/seen_child_splits/master_with_split.csv` including `n_adults`, `n_children`, `Activity`, `Interaction_with_child`, `Location`, `timepoint_norm`
- **Router**: A function mapping (metadata) → system_choice (or weight vector over systems)
- **Stacker**: A classifier mapping (system_scores, metadata) → clip_label_probability
- **Stratum**: A subset of clips defined by a single metadata condition (e.g., n_children≥2)

---

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001 (Stacker)**: Metadata-augmented stacker achieves a positive delta on at least one of F1 or AUROC vs. `all_available` LR stack (F1=0.897, AUROC=0.870); or the null result is documented with feature importances confirming metadata contributed no signal
- **SC-002 (Router)**: At least one router variant (rule-based or learned) achieves test F1 ≥ best_audio_mil mean ensemble (0.893)
- **SC-003 (Multi-child suppressor)**: F1 on n_children≥2 test subset improves by ≥3pp over the best single-system baseline in that stratum, with no overall F1 regression >0.5pp
- **SC-004 (Short-voc head)**: At least 5 of the 44 persistent short-vocalization FN clips are recovered, with FP rate on non-short-voc clips increasing by <2pp
- **SC-005 (All sub-features)**: All sub-feature results can be reproduced from a single script call per sub-feature with no manual steps beyond loading pre-existing system predictions

---

## Assumptions

- Per-system `test_predictions.csv` files exist and are accessible for all 11 systems (BabAR, VTC, VTC-KCHI, VBx, USC-SAIL, Pyannote, Sortformer, EEND-EDA, WavLM-MIL, Whisper-MIL, Audio-LLM); these are the inputs to stacking/routing
- BIDS metadata columns `n_adults`, `n_children`, `Activity`, `Interaction_with_child`, `Location` are present in `whisper-modeling/seen_child_splits/master_with_split.csv` (they are — confirmed from column list)
- Ground-truth RTTM files are available for all seen-child test clips (from `whisper-modeling/usc_sail_rttm_cache/` or `pyannote/` RTTM folders) to identify short-vocalization clips for sub-feature D
- WavLM embeddings for seen-child clips are already cached (from MIL runs) and can be reused for the multi-child suppressor without re-running the backbone
- Sub-features A and B run on CPU in minutes; sub-features C and D may require short GPU time (~1–2h) for the specialized heads
- All four sub-features share the same seen-child test split (441 clips) for final evaluation
- Sub-features are implemented sequentially; A/B are prerequisites to confirm whether metadata helps before investing in C/D
