# Feature Specification: Audio LLM Zero-Shot Baseline

**Feature Branch**: `010-audio-llm-baseline`
**Created**: 2026-04-27
**Status**: Draft
**Input**: User description: "Use Qwen2.5-Omni or a similar audio LLM as a zero-shot or few-shot vocalization detector, evaluated as a baseline."

## User Scenarios & Testing *(mandatory)*

### User Story 1 — Zero-Shot Evaluation on Seen-Child Test Split (Priority: P1)

A researcher wants to know how well an off-the-shelf audio language model performs on the child vocalization detection task without any task-specific fine-tuning. They run the audio LLM over the held-out test split, prompt it to answer "yes or no" to whether a child is vocalizing, and compare the resulting F1/AUROC/AUPRC against all existing diarizer baselines (BabAR, USC-SAIL, VTC, WavLM-MIL, etc.).

**Why this priority**: This is a single afternoon of inference with no training; the outcome is thesis-critical regardless of direction. A strong audio LLM baseline means "our task-specific approach beats foundation models by N points." A weak baseline confirms the foundation-model degradation finding at the LLM level and justifies the domain-specific pipeline.

**Independent Test**: Run the inference script on the test split; confirm a predictions CSV is produced with `clip_id`, `prob`, and `label` columns; confirm F1/AUROC/AUPRC are printed and saved to a metrics JSON. Comparing this JSON against `babar_ecapa_enrollment_runs/enroll_test_metrics.json` fully demonstrates the baseline's relative value.

**Acceptance Scenarios**:

1. **Given** the test split CSV (2183 clips), **When** the inference script runs, **Then** every clip receives a predicted probability in [0, 1] and a binary prediction based on a tuned threshold.
2. **Given** audio clips from both 12–16 month and 34–38 month cohorts, **When** the model is queried, **Then** per-timepoint metrics are produced (matching the structure of all other diarizer result folders).
3. **Given** a clip whose audio file is missing or corrupted, **When** the model is queried, **Then** the clip is skipped with a logged warning and NaN prediction rather than crashing the run.
4. **Given** the run has already been completed, **When** the script is run again, **Then** already-completed clips are loaded from cache and the run completes without re-querying the model.

---

### User Story 2 — Few-Shot Comparison (Priority: P2)

A researcher wants to test whether providing 1–3 labeled example clips in the prompt (few-shot) improves detection accuracy over zero-shot, without any gradient updates. Example clips (one positive child-vocalizing, one negative) are drawn from the training split of the same child, matching the enrollment paradigm.

**Why this priority**: Few-shot in-context learning is the natural extension of zero-shot and requires minimal additional code. It also mirrors the ECAPA enrollment paradigm (using a training-split reference per child), making the comparison theoretically interesting.

**Independent Test**: Run the few-shot variant on the validation split; confirm per-child few-shot metrics exist in the output folder; confirm the few-shot predictions file has a `n_examples` column indicating how many examples were used per clip.

**Acceptance Scenarios**:

1. **Given** a test clip for a known child, **When** few-shot mode is used, **Then** the prompt includes at least one positive and one negative training-split clip for that child before the query clip.
2. **Given** a child with fewer than 2 training clips, **When** few-shot mode is used, **Then** the system falls back to zero-shot for that child with a logged note.

---

### User Story 3 — Thesis Comparison Table Update (Priority: P3)

The audio LLM baseline results are added to the unified comparison table in CLAUDE.md and `evaluation/thesis_tables/`, alongside all other diarizer results, so the thesis chapter on baselines is complete.

**Why this priority**: Packaging the result into the existing evaluation framework is low effort but required for the thesis chapter to be self-contained.

**Independent Test**: Run `evaluation/aggregate_thesis_tables.py`; confirm the audio LLM row appears in the baseline comparison table alongside BabAR, USC-SAIL, VTC, WavLM-MIL, and Whisper-MIL.

**Acceptance Scenarios**:

1. **Given** the inference and evaluation scripts have been run, **When** `aggregate_thesis_tables.py` is run, **Then** F1, AUROC, and AUPRC for the audio LLM baseline appear in the table without manual transcription.

---

### Edge Cases

- What happens when the model returns a non-binary response (e.g., "I cannot determine" or a long explanation instead of yes/no)? → Parse heuristically; if unparseable, assign 0.5 probability and log.
- What happens when a clip is silence or very short (< 1 second)? → Pass as-is; log audio duration alongside the prediction.
- What happens when the model times out or the inference process crashes mid-run? → Cache completed clips; resume from last checkpoint on restart.
- What if the model consistently returns the same answer for all clips (degenerate behavior)? → Report variance of predictions in the metrics JSON; flag if variance < 0.01.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The system MUST run a zero-shot audio LLM with a natural-language prompt asking whether a child is vocalizing in each clip, producing a binary prediction and a continuous confidence score.
- **FR-002**: The system MUST evaluate against the held-out test split (seen-child split, 2183 clips) and produce F1, Precision, Recall, AUROC, and AUPRC with val-tuned threshold, matching the output schema of all existing diarizer result folders.
- **FR-003**: The system MUST produce per-timepoint (14_month, 36_month) metrics alongside overall metrics, using the same cohort definitions as all other diarizers.
- **FR-004**: The system MUST cache inference results per clip so that interrupted runs can resume without re-querying the model.
- **FR-005**: The system MUST produce a predictions CSV with at minimum `clip_id`, `child_id`, `timepoint_norm`, `label`, `prob`, and `predicted` columns — matching the schema of existing `enroll_test_predictions.csv` files — so that all downstream analysis scripts work unchanged.
- **FR-006**: The system MUST fall back gracefully (NaN prediction, logged warning) when a clip's audio file is missing or unreadable, without crashing the full run.
- **FR-007**: The system MUST support few-shot mode where 1–3 reference clips from the same child's training split are prepended to the query prompt; few-shot mode is optional and the system defaults to zero-shot.
- **FR-008**: The system MUST report prediction variance across the test set and flag degenerate outputs (all-yes or all-no) in the metrics JSON.
- **FR-009**: The system MUST store the model name, prompt template, and shot count in a `config.json` alongside the results, consistent with all other result folder conventions.

### Key Entities

- **AudioLLMPrediction**: Per-clip record — `clip_id`, `child_id`, `timepoint_norm`, `audio_path`, `label` (ground truth), `prob` (raw confidence in [0,1]), `predicted` (binary, threshold-tuned on val), `model_name`, `prompt_template`, `n_shot`, `response_raw`, `parse_status` (parsed / fallback / error).
- **AudioLLMConfig**: Experiment config — `model_name`, `prompt_template`, `n_shot`, `threshold` (val-tuned), `val_f1`, `test_f1`, `test_auroc`, `test_auprc`, `seed`.
- **AudioLLMMetrics**: Summary metrics JSON — overall F1/AUROC/AUPRC, per-timepoint breakdown, prediction variance, degenerate-output flag, comparison delta vs. BabAR enrollment baseline.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: A complete predictions CSV and metrics JSON are produced for the full 2183-clip test split without manual intervention.
- **SC-002**: The audio LLM baseline result is produced in a single GPU job of ≤ 12 hours walltime (for a 7B-parameter model on one A100-class GPU).
- **SC-003**: The metrics JSON reports F1, AUROC, and AUPRC; the comparison delta vs. the best audio-only baseline (BabAR: F1=0.874, AUPRC=0.918) is computed and stored, enabling the thesis claim to be written with a specific number.
- **SC-004**: Interrupted runs can be resumed within 5 minutes of restarting the inference script, with no re-queried clips.
- **SC-005**: Per-timepoint metrics are available for both the 14-month and 36-month cohorts, enabling the age-stratified degradation pattern to be characterized if it exists.

## Assumptions

- The evaluation uses the **seen-child test split** (`whisper-modeling/seen_child_splits/test.csv`, 2183 clips) with val-set threshold tuning on the val split, consistent with all other diarizers. Cross-child splits are out of scope for this baseline.
- The model runs on the existing SLURM cluster (GPU nodes in `ou_bcs_normal` / `pi_satra` partitions); the 7B parameter model fits in 24 GB VRAM with appropriate quantization.
- The primary prompt template is: *"Listen to this audio clip. Is there a child vocalizing (speaking, babbling, or crying)? Answer only 'yes' or 'no'."* Alternative prompt phrasings are out of scope for the MVP but can be stored as config variants.
- Soft confidence scores are derived from the model's log-probability of the "yes" token relative to the "no" token; if the model does not expose token-level probabilities, the binary output is used with a fixed 0.5 fallback probability.
- Few-shot reference clips are drawn from the **training split only** (no val or test contamination); the same child-ID matching logic as TS-TalkNet and ECAPA enrollment is used.
- Results are stored in a new folder `audio_llm_baseline_runs/{model_slug}/` following the same layout as `babar_ecapa_enrollment_runs/`.
- The audio LLM is used **inference-only** — no fine-tuning, LoRA, or gradient updates of any kind.
- If the primary model (Qwen2.5-Omni-7B) is unavailable on the cluster, a functionally equivalent open audio LLM (e.g., Qwen2-Audio-7B, SALMONN-7B, or audio-capable Llama variant) is used as a drop-in replacement.
