# Feature Specification: Baseline Encoders on Seen-Child Splits

**Feature Branch**: `011-baseline-seen-child`  
**Created**: 2026-04-28  
**Status**: Draft  
**Input**: Run baseline encoders (Whisper/WavLM/Fused × mean/attn/stats pooling) on the seen-child within-child evaluation splits instead of the cross-child splits, so baseline results are directly comparable to enrollment-based diarizer metrics.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Run All Encoder Variants on Seen-Child Split (Priority: P1)

A researcher training the child vocalization detection system needs baseline encoder results on the same 109-child, within-child evaluation split used by all other diarizers (BabAR, VTC, Pyannote, MIL, etc.), so that comparisons in the thesis are apples-to-apples.

**Why this priority**: All thesis comparisons reference the seen-child split. Running baselines only on cross-child splits makes them incomparable to the 10+ other systems.

**Independent Test**: Can be tested by running `python baselines/baseline_encoders.py --seen-child --all-experiments` and verifying `baseline_results_seen_child/{whisper_mean,whisper_attn,wavlm_mean,wavlm_attn}/test_metrics_tuned.json` exist with valid F1/AUROC/AUPRC values.

**Acceptance Scenarios**:

1. **Given** the seen-child splits exist at `whisper-modeling/seen_child_splits/`, **When** `python baselines/baseline_encoders.py --seen-child --all-experiments` is run, **Then** all 13 experiment variants complete and write `test_metrics_tuned.json` to `baselines/baseline_results_seen_child/{variant}/`
2. **Given** a completed run, **When** results are compared against the cross-child baseline results, **Then** seen-child F1 values are ≥ cross-child values (within-child personalization should help)
3. **Given** a completed run, **When** `test_metrics_by_timepoint.csv` is read, **Then** it contains rows for both `14_month` and `36_month` cohorts with non-null metrics

---

### User Story 2 - Document Hard Negative MIL Training (Priority: P1)

A researcher reviewing the project needs to find documentation on the hard-negative MIL training extension (extracting silent-CHI, non-silent windows from Playlogue/Providence RTTMs as additional negative training examples for the frame-window MIL).

**Why this priority**: The implementation is complete and submitted (job 12770452) but CLAUDE.md and results_summary.md do not yet describe the approach or configuration.

**Independent Test**: Can be tested by reading `CLAUDE.md` and verifying the hard-negative MIL section describes `extract_hard_negatives.py`, the two new configs, and the SLURM job.

**Acceptance Scenarios**:

1. **Given** CLAUDE.md is read, **When** searching for "hard negative", **Then** a section exists describing the extraction script, YAML configs, and expected output location
2. **Given** `specs/009-synth-rir-noise/tasks.md` is read, **When** looking at MIL tasks, **Then** the hard-negative MIL task is listed with its SLURM job ID

---

### Edge Cases

- What if some seen-child audio files have moved or been deleted? `audio_exists` filtering in `load_seen_child_split()` handles this.
- What if a variant's `save_path` directory doesn't exist? `run_experiment` calls `os.makedirs(exp_dir)` but the `save_path` directory (which is the same `exp_dir`) must also exist — checked via `os.makedirs(os.path.dirname(save_path), exist_ok=True)` (already handled by `run_experiment`).
- What if the job runs out of time before all 13 experiments complete? Each experiment saves its best checkpoint independently; completed experiments can be identified by the presence of `test_metrics_tuned.json`.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: `baseline_encoders.py` MUST accept a `--seen-child` CLI flag that switches split loading to `whisper-modeling/seen_child_splits/{train,val,test}.csv`
- **FR-002**: When `--seen-child` is set, results MUST be written to `baselines/baseline_results_seen_child/` (separate from cross-child results)
- **FR-003**: The `--all-experiments` flag MUST run all 13 experiment variants (baselines + layer_weighted + lw_stats + fused + unfrozen + no_new_params phases)
- **FR-004**: The seen-child split loader MUST filter to `audio_exists == True` rows and compute `timepoint_feature` from `timepoint_norm`
- **FR-005**: A SLURM script at `baselines/slurm/run_baseline_seen_child.sh` MUST run the full seen-child experiment sweep in a single GPU job
- **FR-006**: CLAUDE.md MUST document the `--seen-child` flag, the new SLURM script, and the hard-negative MIL extension

### Key Entities

- **Seen-child split**: 2183 clips from 109 children; same children appear in train/val/test (within-child); stratified 60/20/20 by (child, timepoint)
- **Cross-child split**: 2377 clips from 139 children; disjoint children across train/val/test (at `baselines/splits/`)
- **Experiment variant**: One (model_type, pooling, use_layer_weights, unfreeze_last_n_layers, per_timepoint_threshold) configuration

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: All 13 experiment variants complete without error and produce `test_metrics_tuned.json` files
- **SC-002**: Seen-child whisper_attn F1 ≥ cross-child whisper_attn F1 (0.83, from prior runs) — within-child should help
- **SC-003**: Results are written to a separate directory (`baseline_results_seen_child/`) so cross-child results are not overwritten
- **SC-004**: CLAUDE.md update passes a grep for "seen-child" and "hard negative" within the baseline and MIL sections

## Assumptions

- Seen-child splits are already generated and present at `whisper-modeling/seen_child_splits/` (they are)
- All audio files referenced in the seen-child splits still exist on disk (checked by `audio_exists` column)
- The GPU node has enough memory for `fused_attn` (batch_size=1, ~40GB); matches existing SLURM configuration
- Cross-child baseline results in `baselines/baseline_results/` should be preserved as-is
