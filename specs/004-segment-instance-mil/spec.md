# Feature Specification: Segment-Instance MIL with Attention Aggregation

**Feature Branch**: `004-segment-instance-mil`
**Created**: 2026-04-23
**Status**: Draft

## Overview

Treat each diarizer-proposed speech segment as a bag instance for Multiple Instance Learning (MIL). Rather than aggregating frame-level features (as the current Whisper attention-pooling baseline does), pool encoder features over each diarizer segment, then train a MIL aggregator over the resulting ~5–30 per-segment embeddings per clip. Run the same MIL head on top of four diarization frontends (USC-SAIL, Pyannote, BabAR-VTC, VBx) and four aggregation strategies (mean, max, attention, gated-attention), producing a 16-cell comparison matrix. Results are evaluated on the seen-child split under the same metrics as the ECAPA enrollment baseline.

---

## User Scenarios & Testing *(mandatory)*

### User Story 1 — Run the 16-cell experiment matrix (Priority: P1)

A researcher wants to know how much of ECAPA enrollment's performance comes from the diarizer quality versus the scoring head. They run the segment-instance MIL pipeline with all four frontends and four aggregation strategies in a single batch job and receive a comparison table alongside the existing ECAPA baseline.

**Why this priority**: This is the primary research deliverable. All other stories depend on these results existing.

**Independent Test**: Submit a single training job; receive a results JSON and CSV with 16 rows (one per configuration), each containing F1, AUROC, and AUPRC on the test split. Delivers a complete comparison table even before interpretability tooling is built.

**Acceptance Scenarios**:

1. **Given** the seen-child split and cached RTTM outputs for all four frontends, **When** the training script is run with `--all-configs`, **Then** all 16 configurations complete training and evaluation, producing a `mil_results/all_configs.json` with one entry per configuration.
2. **Given** a clip where the diarizer found zero segments, **When** the MIL model scores that clip, **Then** it predicts "child absent" (negative) without crashing or emitting NaN.
3. **Given** a configuration with the attention or gated-attention aggregator, **When** training completes, **Then** per-segment attention weights are saved alongside predictions in the results CSV.

---

### User Story 2 — Inspect segment attention weights for a clip (Priority: P2)

A researcher wants to understand why the model made a particular prediction. They look up the attention weights for a specific clip and identify which diarizer-proposed segments contributed most to the "child present" score.

**Why this priority**: Interpretability is a key selling point of this design over ECAPA enrollment; it feeds directly into the existing error-analysis infrastructure.

**Independent Test**: Given a clip ID, query the saved predictions CSV and read the attention weight column; values sum to 1.0 across segments and the highest-weight segment corresponds to a plausible child vocalization time window.

**Acceptance Scenarios**:

1. **Given** a completed attention-MIL run, **When** the results CSV is read for a specific clip, **Then** each row for that clip includes segment start/end times, the attention weight assigned to that segment, and the per-segment cosine similarity to the child prototype (for comparison).
2. **Given** a clip with a correct positive prediction, **When** attention weights are examined, **Then** the highest-weight segment overlaps with ground-truth child speech (verified against the GT RTTM) in at least 60% of correctly-predicted positive clips in the test set.

---

### User Story 3 — Compare MIL to ECAPA enrollment under a unified table (Priority: P3)

A researcher wants to include MIL results in the thesis comparison table that already contains ECAPA enrollment results for all frontends. They point the existing table-generation script at the MIL results folder and get a merged table.

**Why this priority**: Publication-readiness; the value of the 16-cell matrix is only realized when it sits side-by-side with the ECAPA baseline in the same table format.

**Independent Test**: Running the thesis table script with MIL results produces a table row per (frontend, aggregator) pair with the same columns as the ECAPA rows, with no manual reformatting.

**Acceptance Scenarios**:

1. **Given** `mil_results/all_configs.json` and the existing ECAPA enrollment run folders, **When** the table generation script is run, **Then** it produces a combined Markdown/LaTeX table with both ECAPA and MIL results side by side.
2. **Given** the same seen-child test split, **When** MIL and ECAPA enrollment metrics are compared, **Then** the best MIL configuration achieves AUROC within 0.05 of the best ECAPA configuration (confirming the approach is competitive).

---

### Edge Cases

- A clip where the diarizer found zero segments (empty bag): model must predict "child absent" deterministically without NaN or crash.
- A segment shorter than one encoder frame (< 20 ms): treat as a zero-vector embedding or skip; never crash.
- A frontend whose RTTM cache is incomplete (some clips missing): skip missing clips with a warning; do not fail the whole run.
- Clips from 14-month vs. 36-month timepoints may have very different segment counts; the aggregator must handle both extremes (1 segment and 30+ segments) without bias.

---

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The pipeline MUST accept any of the four target frontends (USC-SAIL, Pyannote, BabAR-VTC, VBx) as the instance proposer, loading their cached RTTM outputs without re-running inference.
- **FR-002**: For each diarizer segment, the pipeline MUST pool the pre-trained encoder features (Whisper-small or WavLM-base+) over the segment's time span using either mean or attentive pooling, producing a fixed-size embedding vector.
- **FR-003**: The pipeline MUST support four aggregation strategies over the per-segment embedding bag: mean pooling, max pooling, standard attention MIL (ABMIL), and gated attention MIL (GABMIL).
- **FR-004**: Each of the 16 configurations (4 frontends × 4 aggregators) MUST be trainable and evaluable independently, with results written to a shared results directory.
- **FR-005**: The pipeline MUST cache per-segment encoder embeddings keyed on (audio path, segment start, segment end) to avoid recomputing them across configurations that share a frontend.
- **FR-006**: Clips with zero diarizer segments MUST be assigned a "child absent" prediction with confidence 0.0; they MUST NOT be excluded from evaluation.
- **FR-007**: For attention and gated-attention configurations, the pipeline MUST save per-segment attention weights alongside each clip's prediction, including segment timestamps.
- **FR-008**: The pipeline MUST evaluate each configuration on the seen-child val and test splits and record F1, precision, recall, AUROC, and AUPRC, with threshold tuned on the val split — identical protocol to ECAPA enrollment.
- **FR-009**: A combined summary file MUST be produced listing all 16 configurations' test-split metrics in a single JSON and CSV, sortable by AUROC.
- **FR-010**: The pipeline MUST integrate with the existing `thesis_tables` configuration so MIL results appear in the same generated comparison table as ECAPA enrollment results.

### Key Entities

- **SegmentBag**: The set of diarizer-proposed segments for one clip, each represented by a pooled encoder embedding. Bags are variable-length (0 to ~30 segments).
- **MILConfiguration**: A (frontend_name, aggregator_type, encoder_name, pool_method) tuple that fully identifies one experimental cell.
- **MILResult**: Per-clip predictions and per-segment attention weights produced by one MILConfiguration; stored in a CSV alongside the standard enrollment prediction columns.
- **AggregatorHead**: A trainable module that maps a variable-length bag of embeddings to a clip-level score; one of {mean, max, attention, gated-attention}.

---

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: All 16 configurations complete training and evaluation without error; `mil_results/all_configs.json` contains exactly 16 entries with non-NaN F1, AUROC, and AUPRC values.
- **SC-002**: At least one MIL configuration achieves test-split AUROC ≥ 0.80, matching the strongest ECAPA enrollment result (BabAR/VTC at 0.820).
- **SC-003**: The full 16-configuration sweep completes within a single 24-hour SLURM job on the available GPU cluster (assuming segment embedding caches are pre-populated).
- **SC-004**: For clips where the attention aggregator assigns its highest weight to a segment, that segment overlaps ground-truth child speech in ≥ 60% of correctly-predicted positive test clips, demonstrating meaningful interpretability.
- **SC-005**: The combined thesis table (ECAPA + MIL) generates without manual editing in under 60 seconds; every MIL row uses the same column schema as every ECAPA row.

---

## Assumptions

- RTTM outputs for all four target frontends (USC-SAIL, Pyannote, BabAR-VTC, VBx) already exist or can be generated before this feature runs; the MIL pipeline reads from cached RTTMs, it does not re-run diarization.
- The pre-trained encoder backbone (Whisper-small or WavLM-base+) is used frozen; no backbone fine-tuning is performed in this feature.
- Training uses the `seen_child_splits/` (1311 train / 431 val / 441 test, within-child) — the same split as ECAPA enrollment — so results are directly comparable.
- Segment embeddings are computed once per (frontend, encoder) pair and cached on disk; subsequent runs over different aggregators reuse the cache.
- The MIL head is lightweight (< 5M parameters) and trains in minutes per configuration on available GPU hardware.
- Only speech segments labeled by the diarizer (any speaker label) are used as instances; silence is not a bag instance.
- The video-only frontends (TalkNet-ASD, TS-TalkNet) are out of scope for this feature because they require video files and produce sparse segments incompatible with audio-only encoder pooling.
- An "empty bag" (diarizer found no segments in a clip) is handled by outputting score 0.0 (predict child absent); no imputation or fallback inference is used.
