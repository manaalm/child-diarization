# Feature Specification: Synthetic Child-Adult Scene Generator

**Feature Branch**: `008-synthetic-child-scenes`
**Created**: 2026-04-24
**Status**: Draft

## Overview

A configurable synthetic audio data generator that composes multi-speaker parent-child scenes from real child vocalization and adult speech segments. The generator outputs scene audio files with exact RTTM labels and clip-level vocalization labels, enabling controlled augmentation experiments for child-adult speaker diarization and target-child vocalization detection.

The system targets known failure modes in the existing pipeline: adult-trained embeddings that miss toddler speech, short vocalizations below VAD threshold, false positives from sibling/TV/background speech, and overlapping speech conditions. The minimum viable system does not require neural TTS.

---

## User Scenarios & Testing *(mandatory)*

### User Story 1 — Generate Labeled Synthetic Training Scenes (Priority: P1)

A researcher needs synthetic parent-child audio scenes with ground-truth labels to augment the ~1,500 available real clips. They provide a YAML scene config and segment manifest CSV, run the scene generator, and receive a directory of WAV files with paired RTTM annotations and a clip-level CSV indicating whether the target child vocalized.

**Why this priority**: This is the core deliverable. Every downstream experiment depends on having labeled synthetic scenes. Without it nothing else is testable.

**Independent Test**: Can be tested end-to-end with a small config (50 scenes, 30 s each) using only publicly available child segments from Providence and adult segments from LibriSpeech. Delivers value immediately as a training augmentation source.

**Acceptance Scenarios**:

1. **Given** a valid scene config YAML and a segment manifest CSV, **When** the generator runs, **Then** each requested scene produces a WAV file, an RTTM file with standardized labels (`TARGET_CHILD`, `ADULT_0`, etc.), a per-scene metadata JSON, and a segment timeline CSV — with no missing files.
2. **Given** a positive scene config (`target_child_probability > 0`), **When** the scene is generated, **Then** the RTTM contains at least one `TARGET_CHILD` segment, the clip-level label CSV records `target_child_vocalized = 1`, and the total `TARGET_CHILD` RTTM duration matches `target_child_duration_sec` in the CSV.
3. **Given** a negative adult-only scene type, **When** the scene is generated, **Then** the RTTM contains no `TARGET_CHILD` segments and the clip-level label CSV records `target_child_vocalized = 0`.
4. **Given** the same config and `random_seed`, **When** the generator runs twice, **Then** both runs produce bitwise-identical WAV files and identical labels, confirming reproducibility.
5. **Given** a scene config with `overlap_probability = 1.0`, **When** scenes are generated, **Then** all scenes include at least one time interval where two speaker RTTM segments overlap.

---

### User Story 2 — Build and Filter a Segment Manifest (Priority: P1)

A researcher needs a clean inventory of usable child and adult audio segments from multiple source corpora before generation can begin. They run a manifest-building script against local corpus directories and receive a filtered CSV that records per-segment metadata, quality scores, and split assignments.

**Why this priority**: The manifest is the input contract for the generator. All generation experiments depend on it.

**Independent Test**: Can be tested with a single source corpus (e.g., Providence child segments only). Delivers value as a standalone audit tool even before any scenes are generated.

**Acceptance Scenarios**:

1. **Given** a local directory of annotated child recordings, **When** the manifest builder runs, **Then** it produces a CSV with all required columns (`segment_id`, `speaker_role`, `age_band`, `duration_sec`, `audio_path`, `split`, `usable_for_training`, etc.) and no rows for segments shorter than the configured minimum duration.
2. **Given** segments drawn from multiple source datasets, **When** the manifest is built, **Then** each row records its `source_dataset` and `split`, and no speaker appears in more than one split (no speaker leakage).
3. **Given** a manifest with `usable_for_training = false` rows, **When** the generator loads the manifest, **Then** it never selects those rows as source segments for scenes.

---

### User Story 3 — Run Synthetic-to-Real Ratio Experiments (Priority: P2)

A researcher wants to measure whether adding synthetic data improves held-out real validation performance, and at what ratio gains plateau or reverse. They generate training manifests at six ratios (0×, 0.5×, 1×, 2×, 5×, 10×), train models at each ratio, and compare AUROC/AUPRC on the held-out real test set.

**Why this priority**: This is the primary experimental result. It answers the central research question. Depends on P1 deliverables being complete.

**Independent Test**: Can be tested by running the real-only baseline (0× ratio) and one augmented ratio (1×) and comparing evaluation metrics. Delivers a partial result even before the full sweep is complete.

**Acceptance Scenarios**:

1. **Given** a real training set and a synthetic scene pool, **When** training manifests are generated at each requested ratio, **Then** each manifest is a valid CSV with `audio_path`, `rttm_path`, `target_child_vocalized`, `age_band`, and `split` columns, and the synthetic-to-real ratio matches the requested value to within 5%.
2. **Given** training manifests at all six ratios, **When** models are trained and evaluated, **Then** the evaluation script reports AUROC, AUPRC, F1, precision, recall, and balanced accuracy on the held-out real test set for each ratio.
3. **Given** the ratio sweep results, **When** the analysis script runs, **Then** it produces a `metrics_by_synthetic_ratio.csv` and a `synthetic_ratio_vs_auprc.png` figure showing how each metric changes as the ratio increases.

---

### User Story 4 — Generate Targeted Hard-Negative and Stress Scenes (Priority: P2)

A researcher wants to specifically address false-positive failures (model triggers on adult/sibling/TV speech) and short-vocalization false-negatives. They generate dedicated negative scene sets (adult-only, background-speech, overlap-without-child) and short-vocalization positive sets, then retrain and compare error rates on these failure-mode subsets.

**Why this priority**: Addresses the most clinically relevant failure modes. Dependent on core generator (P1).

**Independent Test**: Can be tested by generating 100 adult-only negative scenes and verifying zero false `TARGET_CHILD` RTTM entries, independent of model training.

**Acceptance Scenarios**:

1. **Given** a `hard_negatives.yaml` config, **When** the generator runs, **Then** all produced scenes have `target_child_vocalized = 0` and RTTM files contain only `ADULT_0`, `ADULT_1`, `OTHER_CHILD_0`, or `BACKGROUND_SPEECH` labels.
2. **Given** a short-vocalization stress config with `short_child_vocalization_probability = 1.0` and `max_child_segment_dur_sec = 0.5`, **When** scenes are generated, **Then** all `TARGET_CHILD` RTTM segments are shorter than the configured threshold.
3. **Given** a low-SNR stress config with `snr_db_max = 5`, **When** scenes are generated, **Then** all scene metadata JSON files record `mean_snr_db ≤ 5`.

---

### User Story 5 — Analyze Synthetic vs. Real Data Distribution (Priority: P3)

A researcher needs to verify that synthetic scenes are acoustically comparable to real clips before drawing conclusions from augmentation experiments. They run a quality analysis script and receive distribution comparison plots and embedding-space visualizations.

**Why this priority**: Required for academic defensibility of results. Needed before final thesis write-up but not blocking earlier experiments.

**Independent Test**: Can be run as a standalone descriptive analysis without any model training.

**Acceptance Scenarios**:

1. **Given** synthetic and real training manifests, **When** the quality analysis script runs, **Then** it produces side-by-side histograms for duration, loudness, SNR, and child/adult ratio distributions.
2. **Given** embeddings extracted from real and synthetic segments using a frozen audio encoder, **When** embedding distance analysis runs, **Then** it reports mean cosine distance between real and synthetic child segments and produces a UMAP/t-SNE plot with real vs. synthetic coloring.
3. **Given** synthetic child segments generated with pitch/formant perturbation, **When** adultification checks run, **Then** the script reports the F0 distribution of perturbed segments relative to real child segments and flags any batch where median F0 falls below the adult reference threshold.

---

### Edge Cases

- What happens when the segment manifest has no usable child segments for the requested `age_band`? The generator should raise a descriptive error before writing any output files, not produce empty-child scenes silently labeled as positive.
- What happens when a requested scene duration exceeds the total available child speech duration? The generator should wrap-around/re-sample segments rather than truncate the scene or raise an error.
- What happens when `overlap_probability > 0` but only one speaker has segments? The generator should degrade gracefully to sequential (no overlap) rather than crash.
- What happens when the same real segment appears in both the synthetic training pool and the real test set? The split assignment in the manifest must prevent this; the manifest builder must assert no test-set speaker appears in synthetic training segments.
- What happens when a source audio file is missing from disk at generation time? The generator should skip and log the missing file, not abort the entire batch.
- What happens when the SNR range produces a mixture that clips? The generator must apply peak normalization after mixing to prevent clipping, logging when normalization rescaled the output.

---

## Requirements *(mandatory)*

### Functional Requirements

**Manifest building**

- **FR-001**: The manifest builder MUST produce a CSV with all required columns defined in the data contract (segment_id, source_dataset, source_recording_id, speaker_id, speaker_role, age_months, age_band, start_time_sec, end_time_sec, duration_sec, audio_path, sample_rate, transcript, phonetic_transcript, vocalization_type, quality_score, split, usable_for_training).
- **FR-002**: The manifest builder MUST assign each segment to exactly one split (train, val, test, or external) with no speaker appearing in more than one of train/val/test.
- **FR-003**: The manifest builder MUST filter out segments flagged `usable_for_training = false` before exposing them to the generator.
- **FR-004**: The manifest builder MUST document each source dataset with name, access method, license tier, age range, annotation type, and suitability flags.

**Scene generation**

- **FR-005**: The generator MUST accept a scene configuration YAML and a segment manifest CSV as its two primary inputs.
- **FR-006**: For each requested scene, the generator MUST produce four output files: a WAV file, an RTTM file, a metadata JSON, and a segment timeline CSV.
- **FR-007**: The generator MUST support all eight scene types: positive target-child, adult-only negative, background-speech negative, silence/noise negative, hard overlap positive, hard overlap negative, short-vocalization positive, and low-SNR positive.
- **FR-008**: RTTM output MUST use standardized speaker labels: `TARGET_CHILD`, `ADULT_0`, `ADULT_1`, `OTHER_CHILD_0`, `BACKGROUND_SPEECH`. Non-speech noise MUST NOT be assigned an RTTM speaker label.
- **FR-009**: The clip-level labels CSV MUST contain all required columns defined in the data contract.
- **FR-010**: The generator MUST be deterministically reproducible: identical config and random seed MUST produce identical output files.
- **FR-011**: The generator MUST resample all source segments to the configured output sample rate (default 16 kHz) and apply peak normalization after final mixing to prevent clipping.
- **FR-012**: The generator MUST apply configurable cross-fades at segment boundaries.
- **FR-013**: The generator MUST support optional RIR convolution per speaker using configurable RIR libraries.
- **FR-014**: The generator MUST support additive background noise mixing at a sampled SNR drawn from the configured distribution.
- **FR-015**: The turn-taking simulator MUST sample inter-turn pauses, speaker sequence transitions, and overlap durations from configurable distributions, with separate parameters per age band.
- **FR-016**: The generator MUST NOT require neural TTS for the minimum viable system.

**Training manifests and evaluation**

- **FR-017**: The training-set generator MUST produce mixture manifests at all six configured synthetic-to-real ratios (0×, 0.5×, 1×, 2×, 5×, 10×).
- **FR-018**: Each training manifest MUST include audio_path, rttm_path, target_child_vocalized, age_band, and split.
- **FR-019**: The evaluation script MUST report AUROC, AUPRC, F1, precision, recall, balanced accuracy, and a confusion matrix on held-out real test data only.
- **FR-020**: The evaluation script MUST report metrics separately for each age band (14–18 months, 34–38 months) and for each synthetic-to-real ratio.
- **FR-021**: The error analysis script MUST categorize errors into: real-only false positives fixed, real-only false negatives fixed, new false positives introduced, new false negatives introduced, short-vocalization errors, overlap errors, adult/sibling/background false positives, errors by age band.

**Quality analysis**

- **FR-022**: The quality analysis script MUST compare duration, loudness, SNR, overlap, pause, and child/adult ratio distributions between synthetic and real data.
- **FR-023**: The quality analysis script MUST compute embedding-space distance between real and synthetic child segments using a frozen audio encoder and produce a UMAP or t-SNE visualization.
- **FR-024**: For any experiment using pitch/formant perturbation, the quality analysis script MUST report whether perturbed segments fall within the F0 range of real child segments for the target age band.

### Key Entities

- **Segment**: A single contiguous vocalization or speech unit from one speaker, drawn from a source corpus. Attributes: source dataset, speaker identity, role, age band, duration, vocalization type, quality score, split assignment.
- **Scene**: A synthetic multi-speaker audio clip of fixed duration, composed by placing segments on a shared timeline with optional overlap, noise, and RIR. Each scene has a unique ID, exact RTTM labels, and a clip-level binary vocalization label.
- **Scene Config**: A YAML document specifying all parameters governing scene generation: source libraries, scene duration, speaker probabilities, overlap/pause distributions, SNR range, RIR probability, noise probability, random seed.
- **Segment Manifest**: A CSV inventory of all available source segments across all corpora, with split assignments and usability flags.
- **Training Manifest**: A CSV listing audio paths and labels for a specific synthetic-to-real ratio training run, used as input to downstream model training.

---

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: The generator produces at least 5,000 synthetic scenes in under 4 hours on a single CPU node, with no more than 1% of scenes failing due to processing errors.
- **SC-002**: Every generated scene has a deterministic correspondence between its RTTM labels and its audio content: independent frame-level verification of a random 100-scene sample confirms zero label–audio mismatches.
- **SC-003**: The manifest builder runs end-to-end on at least two publicly accessible child corpora (e.g., Providence + TinyVox) and produces a manifest covering ≥500 unique child speakers or speaker-equivalent segments with split integrity verified (zero test-set speakers in synthetic training pool).
- **SC-004**: The training-set generator produces all six ratio manifests; models trained at each ratio can be evaluated and results recorded for all metrics in under 48 hours of total computation on available hardware.
- **SC-005**: The evaluation report clearly shows whether real + synthetic performance is higher, lower, or equivalent to real-only across AUROC and AUPRC on the held-out real test set, with results broken out by age band.
- **SC-006**: The synthetic-ratio sweep plot shows the AUROC trend across all six ratios, making it visually apparent whether performance plateaus, peaks, or degrades beyond a certain ratio.
- **SC-007**: The quality analysis script produces at least one comparison figure (distribution or embedding visualization) per scene attribute category, usable directly in the thesis without manual reformatting.
- **SC-008**: The system supports a defensible negative result: if no ratio improves real-test AUROC over the baseline, the error analysis script identifies which specific failure modes were and were not addressed by synthetic augmentation.

---

## Assumptions

- The researcher has already downloaded or has access to Providence, TinyVox, and LibriSpeech locally; the generator does not handle corpus downloading.
- The existing `whisper-modeling/seen_child_splits/` seen-child split (109 children, 2,183 clips) serves as the real training/val/test base; synthetic data augments only the train split.
- Speaker identity in Providence child segments can be inferred from existing RTTM annotations with sufficient reliability for split assignment; no additional diarization pass is required for Providence.
- "Usable" child segments are defined as having duration ≥ 0.3 s, quality score ≥ a configurable threshold (default 0.5 if quality scores exist), and not flagged as non-speech artifacts.
- Adult segments from LibriSpeech are sufficiently clean and phonetically diverse for use as caregiver speech surrogates; no age-specific filtering beyond gender-neutral selection is needed for the MVP.
- MUSAN and RIRS_NOISES are available or downloadable under permissive terms; the generator config documents their locations.
- The downstream classifiers (WavLM/Whisper + linear head, BabAR enrollment) can be re-run with augmented training data using existing training scripts with only the training manifest CSV swapped.
- The thesis timeline does not require implementing child TTS or voice conversion for the minimum viable system; these are stretch goals.
- A negative result — finding that synthetic augmentation does not improve real-test performance — is explicitly a valid and publishable thesis outcome, requiring no further augmentation of the claims.
- The `child-vocalizations` conda environment is the deployment target; no new isolated environments are needed for the minimum viable system beyond what is already available.
