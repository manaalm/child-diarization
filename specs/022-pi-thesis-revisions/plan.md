# Implementation Plan: PI Thesis Revisions — Methodology, Baselines, Encoder Refactor

**Branch**: `021-post-thesis-future-work` (spec dir is `022-pi-thesis-revisions`; branch and dir are intentionally decoupled per spec) | **Date**: 2026-05-12 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/orcd/scratch/orcd/008/manaal/child-adult-diarization/specs/022-pi-thesis-revisions/spec.md`

## Summary

Five-slice revision plan responding to PI feedback. Slice ordering: methodology corrections first (BIDS-derived timepoints US1, imbalance-aware metrics + group-stratified k-fold US2), then audio-scene-analysis baseline expansion (Qwen 3.5-Omni + YAMNet + AST, with a new universal-coverage split US3), then encoder code relocation + figure + fusion docs (US4), then chapter restructure to move per-timepoint breakdowns to a dedicated posthoc subsection (US5). US1/US2 are CPU-only and produce regeneratable artefacts from cached predictions; US3 is the only GPU-heavy slice. Implementation reuses the existing `unified.py` enrollment scoring conventions, the `mil/mil_utils.compute_metrics` API (extended in place), and the existing `whisper-modeling/seen_child_splits/` split layout (extended with a new `all_children_splits/` sibling for the universal-coverage variant).

## Technical Context

**Language/Version**: Python 3.11 in the `child-vocalizations` conda env (per CLAUDE.md). Python 3.10 for `joint_asr_diar` env is not needed for this spec.
**Primary Dependencies**: pandas, numpy, scikit-learn 1.7.2 (already provides `StratifiedGroupKFold`, `balanced_accuracy_score`, `f1_score(average='weighted')`); transformers ≥4.45 with `TRANSFORMERS_OFFLINE=1`/`HF_HUB_OFFLINE=1` env vars for ≥4.57 (per CLAUDE.md gotcha); torchaudio for waveform loading; matplotlib for the US4 encoder-pipeline figure; HuggingFace `transformers` for AST (`MIT/ast-finetuned-audioset-10-10-0.4593`) and Qwen 3.5-Omni; TFHub `tensorflow_hub` + `tensorflow` for YAMNet (in a sibling env to avoid TF↔PyTorch ABI conflicts).
**Storage**: filesystem-only. Result CSVs and JSONs under canonical dirs per CLAUDE.md; new artefacts under `evaluation/` (US2), `baselines/scene_analysis_runs/` (US3), `baselines/audio_llm_baseline_runs/qwen35_omni_7b/` (US3), `whisper-modeling/all_children_splits/` (US3), `docs/per_model_training_data.csv` (US4), and `specs/022-pi-thesis-revisions/{bids_vs_spreadsheet_diff.csv, changelog.md}` (US1).
**Testing**: smoke pytest at `tests/spec022/` for the four new scripts (BIDS-timepoint deriver, balanced-metrics recomputer, group-stratified k-fold splitter, scene-analysis baseline). For end-to-end verification, regenerate `master_with_split.csv` from BIDS, diff against the committed version, confirm row counts and timepoint distributions match the published `split_summary.json` modulo the BIDS-vs-spreadsheet correction. For sklearn pieces, pin one `test_predictions.csv` (e.g., `mil/mil_results/whisper_mil/test_predictions.csv`) and assert that the recomputed `{f1_macro, balanced_accuracy, auroc, auprc}` match the existing `test_metrics_tuned.json` for the metrics that overlap.
**Target Platform**: Linux (ORCD scratch). CPU is sufficient for US1 (BIDS parse < 5 min), US2 (balanced-metrics rerun < 10 min over ~30 systems; k-fold audit < 5 min), US4 refactor, US5 chapter restructure. GPU SLURM jobs needed for US2 group-stratified k-fold retraining (5 folds × ~1h training × top-6 systems ≈ 30 GPU-hours), US2 LOOCV subset (capped at ≤3 systems × 109 folds × ~15 min = ~80 GPU-hours), US3 Qwen 3.5-Omni inference (~4h per split × 2 splits = ~8 GPU-hours), US3 AST inference (~1h per split × 2 splits), US3 YAMNet inference (CPU acceptable but TF env required).
**Project Type**: research codebase (mixed Python + shell scripts + result artefacts + thesis chapter). No web/mobile/library boundaries.
**Performance Goals**: BIDS parse < 5 min for ~109 children; balanced-metrics rerun < 10 min CPU across all systems; group-stratified k-fold retraining within the existing per-system fold-training budget already used for within-child k-fold (~1h GPU per fold per system); zero-shot YAMNet/AST evaluation < 1h per split each; Qwen 3.5-Omni evaluation matched to Qwen 2.5 budget (~4h per split).
**Constraints**:
- No new git branch — work lands on `021-post-thesis-future-work` per user direction.
- `TRANSFORMERS_OFFLINE=1` and `HF_HUB_OFFLINE=1` set in every SLURM script that loads a HF model (CLAUDE.md gotcha).
- `unset HF_TOKEN HUGGINGFACE_HUB_TOKEN HF_HUB_TOKEN` at top of any SLURM script that loads a public model (CLAUDE.md gotcha — Qwen 3.5 and AST both apply).
- For Qwen 3.5-Omni: `pip install --no-deps torchvision==<matched>` may be required (Qwen 2.5 carryover gotcha).
- YAMNet TFHub install requires a sibling env if it conflicts with the `child-vocalizations` env's torch pin (2.8.0+cu128). Plan: standalone `yamnet-eval` venv with `tensorflow` + `tensorflow-hub` + `soundfile`; bridged via subprocess from `baselines/scene_analysis_baseline.py`.
- File-deletion discipline (Constitution v1.1.0): do not delete the existing within-child k-fold result dirs (`*_kfold3_f{0,1,2}/`) when introducing group-stratified k-fold. Old dirs are preserved; new dirs land at `*_groupstrat5_f{0..4}/`.
- Inode quota on scratch (1M files) — none of the new artefacts add meaningful inode pressure (≤1000 new files total).
**Scale/Scope**: 5 user stories, 22 functional requirements, ~30 systems with existing `test_predictions.csv` to recompute metrics for, ~109 children, ~2183 clips in current seen-child split, expected 3000-4000 clips in new universal-coverage split.

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Principle | Status | Evidence |
|---|---|---|
| I. Reproducibility & Environment | PASS | All new scripts will commit `config.json` alongside results. Qwen 3.5 cache invalidation (`rm -rf baselines/audio_llm_cache/qwen35_omni_7b/`) is explicit in spec FR-012 acceptance scenario. Seeds fixed at 42. YAMNet's sibling env is documented in this plan. |
| II. Data Integrity & Split Discipline | PASS | New `all_children_splits/test_all.csv` is explicitly labelled zero-shot-eval-only in plan and quickstart; never used for training. Existing seen-child and cross-child splits not modified except for the BIDS-timepoint correction to `timepoint_norm` (preserves seed-42 stratification; FR-003). Trivial-floor F1 and balanced-acc reported alongside (FR-006) so reviewers see imbalance honestly. |
| III. Baseline-First Development | PASS | Spec adds new baselines (Qwen 3.5-Omni, YAMNet, AST) rather than new complex methods. Each new baseline is single-config zero-shot — no ablations to omit. Existing baselines remain in canonical dirs. |
| IV. Rigorous Evaluation & Appropriate Metrics | PASS (and extends) | Spec extends the principle's "F1, Precision, Recall, AUROC, AUPRC" to add balanced accuracy and class-weighted F1 (FR-006/007); this is additive, not a removal. Frame-level vs enrollment separation is preserved. Per-timepoint breakdown moves to a dedicated posthoc section but is NOT removed (FR-021); the principle's per-timepoint reporting requirement is satisfied via the posthoc subsection. Threshold tuning remains on val. **Recommend a future PATCH or MINOR constitution amendment** (v1.1.1 or v1.2.0) to formally add balanced accuracy and class-weighted F1 to the Primary metrics list; flagged in plan, not blocking. |
| V. Mandatory Ablations & Error Analysis | PASS | Zero-shot baselines don't have ablations to omit (single-config). Group-stratified k-fold is itself a methodological ablation (within-child vs group-stratified). LOOCV is a sensitivity check. Per-child error rates already produced by existing per-system error-analysis scripts; new baselines integrate with the existing harness. |
| VI. Thesis Synchronization | PASS | FR-005 explicitly requires updating `CLAUDE.md` per-timepoint blocks and recording the diff in a changelog inside this spec dir. Every new metric file lands under a canonical results dir. |
| VII. Documentation & Honest Reporting | PASS | FR-016 requires AudioSet class-to-score mapping documented in baseline README. FR-019 requires fusion-of-encoders prose elaboration. FR-020 requires per-model training-data CSV. CLAUDE.md updates required at FR-005. |
| File deletion discipline (Dev Std) | PASS | No deletions planned. Within-child k-fold dirs preserved + relabelled; encoder relocation uses `git mv` (preserves history, not delete-then-recreate). |

**Gate result**: PASS. No violations, no Complexity Tracking entries needed.

## Project Structure

### Documentation (this feature)

```text
specs/022-pi-thesis-revisions/
├── plan.md                       # This file
├── spec.md                       # Feature specification (already created)
├── research.md                   # Phase 0 output (this command)
├── data-model.md                 # Phase 1 output (this command)
├── quickstart.md                 # Phase 1 output (this command)
├── contracts/                    # Phase 1 output (this command)
│   ├── balanced_metrics_summary.schema.md
│   ├── group_stratified_kfold_summary.schema.md
│   ├── bids_vs_spreadsheet_diff.schema.md
│   ├── all_children_split.schema.md
│   ├── per_model_training_data.schema.md
│   └── cli_contracts.md          # CLI contracts for the four new scripts
├── checklists/
│   └── requirements.md           # Spec quality checklist (already created)
├── bids_vs_spreadsheet_diff.csv  # US1 artefact (produced during US1 implementation)
├── changelog.md                  # US1 post-correction change log (produced during US1)
└── tasks.md                      # Phase 2 output (NOT created by /speckit.plan)
```

### Source Code (repository root)

This is a research codebase; the layout is set by CLAUDE.md and not by this plan. The plan introduces a few additions and one relocation; the relocation is the only structural change.

```text
/orcd/scratch/orcd/008/manaal/child-adult-diarization/

# US1 — BIDS-derived timepoint correction
whisper-modeling/
├── make_seen_child_split.py                       # MODIFY: switch timepoint source from spreadsheet to BIDS
├── bids_timepoint.py                              # NEW: BIDS session → timepoint mapping module
├── seen_child_splits/
│   ├── master_with_split.csv                      # REGEN: corrected timepoint_norm
│   ├── train.csv / val.csv / test.csv             # REGEN: corrected timepoint_norm
│   ├── split_summary.json                         # REGEN: includes BIDS-correction provenance
│   └── bids_correction_provenance.json            # NEW: per-child BIDS-vs-spreadsheet diff summary
└── all_children_splits/                           # NEW (US3 dependency)
    └── test_all.csv                               # NEW: universal-coverage zero-shot eval

# US2 — Imbalance-aware metrics + group-stratified k-fold
mil/
└── mil_utils.py                                   # MODIFY: extend compute_metrics() with f1_weighted, balanced_accuracy
evaluation/
├── balanced_metrics.py                            # NEW: recompute extended metric set from cached predictions
├── balanced_metrics_summary.csv                   # NEW: one row per system, extended metric set
├── kfold_audit.md                                 # NEW: audit of *_kfold3_f{0,1,2}/ splitting behaviour
├── group_stratified_kfold.py                      # NEW: split + train + eval loop with StratifiedGroupKFold
├── group_stratified_kfold_summary.csv             # NEW: mean ± std AUROC + balanced_accuracy per top-band system
├── loocv_subset.py                                # NEW: LOOCV runner for ≤3 top-band systems
├── loocv_subset_summary.csv                       # NEW: per-child held-out AUROC
└── slurm/
    └── run_group_stratified_kfold.sh              # NEW: GPU SLURM for group-stratified k-fold per system
mil/mil_results/
├── *_kfold3_f{0,1,2}/                             # PRESERVE: legacy within-child k-fold dirs (relabelled in CLAUDE.md)
└── *_groupstrat5_f{0..4}/                         # NEW: group-stratified k-fold result dirs

# US3 — Audio-scene-analysis baseline expansion
baselines/
├── audio_llm_baseline.py                          # MODIFY: register qwen35_omni_7b model slug
├── scene_analysis_baseline.py                     # NEW: --model {yamnet,ast} runner
├── audio_llm_baseline_runs/qwen35_omni_7b/        # NEW result dir
│   ├── val_metrics_tuned.json / test_metrics_tuned.json
│   ├── val_predictions.csv / test_predictions.csv
│   └── README.md                                  # documents prompt template, cache invalidation
├── scene_analysis_runs/
│   ├── yamnet/                                    # NEW result dir
│   │   ├── val_metrics_tuned.json / test_metrics_tuned.json
│   │   ├── val_predictions.csv / test_predictions.csv
│   │   └── README.md                              # AudioSet class-to-score mapping (FR-016)
│   └── ast/                                       # NEW result dir
│       ├── val_metrics_tuned.json / test_metrics_tuned.json
│       ├── val_predictions.csv / test_predictions.csv
│       └── README.md                              # AudioSet class-to-score mapping (FR-016)
└── slurm/
    ├── run_audio_llm_baseline.sh                  # MODIFY: add Qwen 3.5 dispatch arm
    └── run_scene_analysis_baseline.sh             # NEW: SLURM dispatch for YAMNet+AST

# US4 — Encoder section restructure
encoders/                                          # NEW top-level module (relocation target)
├── __init__.py                                    # NEW
├── baseline_encoders.py                           # MOVED from baselines/ (git mv)
├── run_fused_attn_unfreeze2_backbone_swap.py      # MOVED from baselines/
├── run_fused_attn_unfreeze2_kfold.py              # MOVED from baselines/
└── README.md                                      # NEW: maps old → new import paths
baselines/
├── baseline_encoders.py                           # NEW shim: re-exports from encoders/ (one-cycle compat)
├── run_fused_attn_unfreeze2_backbone_swap.py      # NEW shim: re-exports from encoders/
└── run_fused_attn_unfreeze2_kfold.py              # NEW shim: re-exports from encoders/
docs/
├── per_model_training_data.csv                    # NEW: one row per evaluated system
├── per_model_training_data.py                     # NEW: introspects saved configs to produce the CSV
└── figures/
    └── encoder_pipeline.{png,pdf}                 # NEW: thesis figure (input → encoder → pooling → linear head)

# US5 — Per-timepoint posthoc analysis
# Thesis chapter restructure — not in this repo; affects the megadoc / chapter LaTeX.
# Documented in this plan's quickstart for chapter-author handoff.
```

**Structure Decision**: This is a research codebase with conventions set by CLAUDE.md. The plan does not introduce a new top-level layout; it adds `encoders/` (relocation target), `evaluation/` artefacts (already-conventional dir), and result subdirs under existing canonical roots. The relocation preserves `git mv` history per Constitution VI (Thesis Synchronization → no silent overwrites).

## Complexity Tracking

No Constitution violations identified. Table omitted.
