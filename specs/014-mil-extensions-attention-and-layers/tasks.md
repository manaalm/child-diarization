---
description: "Task list for spec-014 — MIL Extensions: Weighted-Layer-Sum, Child-Adapted Backbone, ACMIL"
---

# Tasks: MIL Extensions — Weighted-Layer-Sum, Child-Adapted Backbone, ACMIL

**Input**: Design documents from `/specs/014-mil-extensions-attention-and-layers/`
**Prerequisites**: plan.md, spec.md, research.md, data-model.md, quickstart.md

## Status — 2026-04-29 (verified at artifact level, not per-task)

All spec-014 user stories shipped. Per-task `[ ]` boxes below are intentionally **not bulk-ticked** because individual verification of 62 micro-steps was not performed in this session; instead, completeness is confirmed via the following artifact audit:

**Confirmed via filesystem (`mil/mil_results/`):**
- US1 layer-sum: `wavlm_mil_layersum/`, `whisper_mil_layersum/`, `hubert_large_mil_layersum/` — all 3 have `test_metrics_tuned.json`, `config.json`, `layer_weights.json` ✅
- US2 child-adapted: `wavlm_mil_child_adapted/` — has `test_metrics_tuned.json`, `config.json` ✅ (NEGATIVE result: AUROC=0.500 random collapse)
- US3 ACMIL: `wavlm_mil_acmil/`, `whisper_mil_acmil/` — both have `test_metrics_tuned.json`, `config.json`, `branch_weights_test.json`, `branch_attention_test.csv` ✅; plus 6 branch-aggregation retrain dirs (`{wavlm,whisper}_mil_acmil_{max,gated,topk}/`) added 2026-04-29
- US4 TS-MIL: `wavlm_mil_tsmil_concat/`, `wavlm_mil_tsmil_film/`, `whisper_mil_tsmil_concat/` — all 3 have artifacts ✅. Cross-child arm permanently skipped (methodological — see `mil/spec014_jobs.json` `attempt: 99` + `skipped_reason`)
- US5/US6 seg-MIL extensions: 16 new `mil/mil_results/seg_mil/{frontend}_{aggregator}/` cells (4 frontends × 4 new aggregators: `dsmil`, `auto_pool`, `exp_softmax_pool`, `gmap`) ✅

**Confirmed via tracker (`mil/spec014_jobs.json`):** 19 job entries with COMPLETED states, test_f1/auroc/auprc, and notes; orchestrator `mil/slurm/run_spec014.sh` and tracker `mil/scripts/track_spec014.py` recorded in CLAUDE.md "Recent Changes".

**Code changes confirmed by artifact existence (cannot have written `layer_weights.json` without T004–T008; cannot have written `branch_*` files without T022–T026; cannot have run cross-child without `mil_train.py` cross-child path; etc.).**

**Result narrative recorded in `CLAUDE.md` "Spec-014 MIL Extensions completed" entry**: Whisper-MIL TS-MIL concat is the only positive frame-window result (+0.016 AUROC over Whisper-MIL); HuBERT-large layersum is a useful new model variant (+0.042 AUROC vs WavLM-MIL); ACMIL US3 originally NEGATIVE but the branch-aggregation extension (`whisper_mil_acmil_max`) is the best new variant overall (F1=0.891, AUROC=0.842, AUPRC=0.936, +0.091 AUROC vs mean).

**What remains genuinely incomplete:** the per-task `[ ]` boxes themselves are stale documentation. They are NOT load-bearing — all numerical results live in `spec014_jobs.json` + `CLAUDE.md` + the result dirs. Future bookkeeping pass can convert the `[ ]` boxes individually by re-reading each task and matching to the audit above; deferring to keep this session honest.

---

**Tests**: Not explicitly requested. The acceptance scenarios in spec.md and the regression-check protocol in quickstart.md (Step 0) supply the test discipline; no separate unit/contract test tasks are added beyond the mandated reproducibility regression check.

**Organization**: Tasks are grouped by user story so each story can be implemented and SLURM-submitted independently.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies on other in-flight tasks)
- **[Story]**: User story label — `[US1]` weighted-layer-sum, `[US2]` child-adapted backbone, `[US3]` ACMIL
- File paths are absolute or relative to repo root `/orcd/scratch/orcd/008/manaal/child-adult-diarization/`

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Workspace prep. No new Python env or dependencies required (all three US use the existing `child-vocalizations` conda env).

- [ ] T001 Confirm git branch is `014-mil-extensions-attention-and-layers` (create from `main` if not yet present) — `git switch -c 014-mil-extensions-attention-and-layers main` once outstanding work is committed
- [ ] T002 Verify required artifacts exist via `ls`-style checks: `mil/mil_results/wavlm_mil/test_metrics_tuned.json`, `mil/mil_results/whisper_mil/test_metrics_tuned.json`, `mil/configs/wavlm_mil_child_adapted.yaml`, `synth_results/child_wavlm_checkpoint/step_50000/config.json`. Document any missing artifact in `specs/014-mil-extensions-attention-and-layers/quickstart.md` Troubleshooting section
- [ ] T003 [P] Record committed baseline numbers (WavLM-MIL F1 0.882 / AUROC 0.771; Whisper-MIL F1 0.886 / AUROC 0.853; cross-child Whisper 0.876) into `specs/014-mil-extensions-attention-and-layers/research.md` R5 as the regression-check tolerance reference

---

## Phase 2: Foundational (Blocking Prerequisites for US1 and US3)

**Purpose**: Refactor `mil/mil_model.py`, `mil/mil_train.py`, `mil/mil_evaluate.py` to support the new config keys (`layer_aggregation`, `head`) with backward-compatible defaults BEFORE any US1/US3 implementation begins. US2 does not depend on this phase.

**⚠️ CRITICAL**: The regression-check at the end of Phase 2 (T010) MUST pass before US1/US3 code lands. If `wavlm_mil` baseline diverges by more than the tolerance, fix the regression first.

- [ ] T004 Refactor `BackboneExtractor.__init__` in `mil/mil_model.py` to accept `layer_aggregation: str = "last"` and `layer_aggregation_skip_first: bool = True`; when `weighted_sum`, register `self.layer_weights = nn.Parameter(torch.zeros(num_hidden_layers))` per data-model.md §1
- [ ] T005 Modify `BackboneExtractor.forward` in `mil/mil_model.py` so that when `self.layer_aggregation == "weighted_sum"`, replace the single `out.hidden_states[self.layer]` read at lines 65 and 68 with `(softmax(layer_weights) @ stacked_hidden_states[1:])`; preserve existing `last`-layer behavior when key absent
- [ ] T006 Add a `head: str = "gated_abmil"` dispatch in `build_mil_model` in `mil/mil_model.py` (data-model.md §3); leave `GatedABMILHead` as the default factory result so existing configs are unaffected
- [ ] T007 Modify `mil/mil_train.py` to handle both 2-tuple `(logit, attn)` and 4-tuple `(logit, attn, branch_attn, div_loss)` return values from `MILModel.forward`; add `loss_div` column to `training_history.csv` (zero for non-ACMIL runs); add `loss_bce` column for symmetry
- [ ] T008 Modify `mil/mil_train.py` end-of-training hook to write `{run_dir}/layer_weights.json` when `cfg.layer_aggregation == "weighted_sum"` per FR-004
- [ ] T009 Modify `mil/mil_evaluate.py` to (a) preserve all existing outputs unchanged when `head == "gated_abmil"`, (b) write `branch_weights.json` and per-clip `branch_attention.csv` when `head == "acmil"` per data-model.md §6
- [ ] T010 Run regression check from quickstart.md Step 0: `sbatch mil/slurm/train_mil.sh mil/configs/wavlm_mil.yaml`, then diff `mil/mil_results/wavlm_mil/test_metrics_tuned.json` against committed baseline; PASS criterion is `|delta_AUROC| <= 0.005 AND |delta_F1| <= 0.01`. If FAIL, do not proceed past Phase 2

**Checkpoint**: Foundational refactor committed and regression-verified. US1 and US3 can now proceed in parallel; US2 was already independent.

---

## Phase 3: User Story 2 — Child-Adapted WavLM Wired Into MIL (Priority: P1)

**Goal**: Train+evaluate the existing `mil/configs/wavlm_mil_child_adapted.yaml` end-to-end and integrate the result row into `results_summary.md` and `CLAUDE.md`.

**Independent Test**: After T014 completes, `mil/mil_results/wavlm_mil_child_adapted/test_metrics_tuned.json` exists with valid F1/AUROC/AUPRC, and per-timepoint metrics CSV exists. The result row in `results_summary.md` cites the deltas vs. off-the-shelf `wavlm_mil` baseline (AUROC 0.771).

**Note**: Listed as Phase 3 (before US1/US3) because it requires no Phase 2 code changes — the existing config + existing SLURM scripts are sufficient. Run it first to get a low-risk early data point.

- [ ] T011 [US2] Add a pre-flight assertion to `mil/slurm/train_mil.sh` that exits with code 2 and prints "ERROR: pretrain not finished; submit synth/slurm/run_wavlm_pretrain.sh first" if `synth_results/child_wavlm_checkpoint/step_50000/config.json` is missing per FR-007
- [ ] T012 [US2] Submit training: `sbatch mil/slurm/train_mil.sh mil/configs/wavlm_mil_child_adapted.yaml`; record SLURM job ID for the CLAUDE.md entry
- [ ] T013 [US2] After T012 completes, submit evaluation: `sbatch mil/slurm/eval_mil.sh` (which generalizes via glob over `mil/mil_results/*/best_checkpoint.pt`); confirm all standard MIL output files appear in `mil/mil_results/wavlm_mil_child_adapted/`
- [ ] T014 [US2] Compare `mil/mil_results/wavlm_mil_child_adapted/test_metrics_tuned.json` and `test_metrics_by_timepoint.csv` against the off-the-shelf `wavlm_mil` baseline; record delta_F1, delta_AUROC, delta_AUPRC overall and per-timepoint (14_month vs 36_month) per US2 acceptance #3

**Checkpoint**: US2 result is in hand. If positive (delta_AUROC > 0), this becomes a candidate for combination with US1 layer-sum (FR-010, conditional T024).

---

## Phase 4: User Story 1 — Weighted-Layer-Sum (Priority: P1)

**Goal**: Train three layer-sum variants (WavLM, Whisper, HuBERT-Large) on the seen-child split, dump `layer_weights.json` per run, integrate result rows into `results_summary.md`.

**Independent Test**: Each of `mil/mil_results/wavlm_mil_layersum/`, `whisper_mil_layersum/`, `hubert_large_mil_layersum/` contains `test_metrics_tuned.json` AND `layer_weights.json` showing non-trivial layer selection (not a one-hot at the last layer per US1 acceptance #1 sanity check).

- [ ] T015 [P] [US1] Create `mil/configs/wavlm_mil_layersum.yaml` by copying `mil/configs/wavlm_mil.yaml` and adding `run_name: wavlm_mil_layersum`, `layer_aggregation: weighted_sum`, `layer_aggregation_skip_first: true` (per data-model.md §5)
- [ ] T016 [P] [US1] Create `mil/configs/whisper_mil_layersum.yaml` by copying `mil/configs/whisper_mil.yaml` and adding the same three keys; bump `run_name` to `whisper_mil_layersum`
- [ ] T017 [P] [US1] Create `mil/configs/hubert_large_mil_layersum.yaml` by copying `mil/configs/hubert_large_mil.yaml` and adding the same three keys; bump `run_name` to `hubert_large_mil_layersum`
- [ ] T018 [P] [US1] Submit training jobs: `sbatch mil/slurm/train_mil.sh mil/configs/wavlm_mil_layersum.yaml`, `sbatch mil/slurm/train_mil.sh mil/configs/whisper_mil_layersum.yaml`, `sbatch mil/slurm/train_mil.sh mil/configs/hubert_large_mil_layersum.yaml` (HuBERT job uses 36 h walltime per Technical Context); record all three SLURM job IDs
- [ ] T019 [US1] After all three T018 jobs complete, submit evaluation: `sbatch mil/slurm/eval_mil.sh`; verify each run dir contains `test_metrics_tuned.json`, `test_predictions.csv`, `test_metrics_by_timepoint.csv`, `layer_weights.json`
- [ ] T020 [US1] For each of the three layer-sum runs, sanity-check `layer_weights.json`: load JSON, assert top weight is NOT a one-hot at the final layer; record top-5 layer indices per backbone per quickstart.md Step 1 inspection block
- [ ] T021 [US1] Append three result rows (one per backbone) to `results_summary.md` showing F1/Precision/Recall/AUROC/AUPRC and `delta_AUROC` vs the corresponding non-layersum baseline; include a one-paragraph note citing which transformer layer dominated per backbone

**Checkpoint**: US1 result is in hand for all three backbones; Phase 6 (FR-010, conditional combined run) can fire if any are positive.

---

## Phase 5: User Story 3 — ACMIL Head Drop-In (Priority: P2)

**Goal**: Implement `ACMILHead` with MBA + STKIM + cosine diversity loss; train WavLM and Whisper variants; verify branch diversity via per-branch weak-diarization alignment.

**Independent Test**: `mil/mil_results/wavlm_mil_acmil/` and `whisper_mil_acmil/` contain `test_metrics_tuned.json`, `branch_weights.json`, and `branch_attention.csv`. Per-branch alignment check (US3 acceptance #4) shows non-collapsed branches (alignment differs across branches by ≥ 0.02 AUROC, OR alternatively all-branches-mean alignment exceeds the gated-ABMIL baseline).

**Depends on**: Phase 2 (foundational refactor) must be complete and regression-verified (T010 PASS).

- [ ] T022 [US3] Implement `ACMILHead(nn.Module)` class in `mil/mil_model.py` per data-model.md §2: parameters `n_branches`, `stkim_p`, `stkim_k_frac`, `stkim_k_cap`, `mba_diversity_weight`, `dropout`; forward returns `(logit, attn, branch_attn, div_loss)`; STKIM applied only when `self.training` is True per FR-013
- [ ] T023 [US3] Wire `ACMILHead` into `build_mil_model` in `mil/mil_model.py` so that `head: acmil` selects it (per FR-015 and data-model.md §3); reuse defaults from data-model.md §5 (`acmil_n_branches: 5`, `acmil_stkim_p: 0.5`, `acmil_stkim_k_frac: 0.1`, `acmil_stkim_k_cap: 10`, `acmil_mba_diversity_weight: 0.1`)
- [ ] T024 [P] [US3] Create `mil/configs/wavlm_mil_acmil.yaml` per data-model.md §5 (inherits from `wavlm_mil.yaml`, adds the six `acmil_*` keys plus `head: acmil` and `run_name: wavlm_mil_acmil`)
- [ ] T025 [P] [US3] Create `mil/configs/whisper_mil_acmil.yaml` analogously from `whisper_mil.yaml`
- [ ] T026 [US3] Extend `mil/eval_weak_diarization.py` (or add a small wrapper script `mil/eval_weak_diarization_branches.py`) to read `branch_attention.csv` and emit per-branch Pearson/Spearman/AUROC alignment vs RTTM ground truth; preserve existing single-branch behavior for non-ACMIL runs (per FR-017 and research.md R6)
- [ ] T027 [P] [US3] Submit training jobs: `sbatch mil/slurm/train_mil.sh mil/configs/wavlm_mil_acmil.yaml` (36 h walltime), `sbatch mil/slurm/train_mil.sh mil/configs/whisper_mil_acmil.yaml` (36 h); record job IDs
- [ ] T028 [US3] After T027 jobs complete, submit evaluation: `sbatch mil/slurm/eval_mil.sh`; verify each run dir contains `test_metrics_tuned.json`, `branch_weights.json`, `branch_attention.csv`
- [ ] T029 [US3] Run per-branch weak-diarization alignment via the script extended in T026 against `whisper-modeling/usc_sail_rttm_cache` and the test split CSV; output `mil/mil_results/{wavlm_mil_acmil,whisper_mil_acmil}/branch_alignment.csv`
- [ ] T030 [US3] Inspect `branch_alignment.csv`: confirm branches are not collapsed (alignment AUROC across branches has stdev ≥ 0.02). If collapsed, increase `acmil_mba_diversity_weight` to 0.5 and re-train one variant for an ablation comparison
- [ ] T031 [US3] Append result rows to `results_summary.md` for both ACMIL configs showing F1/P/R/AUROC/AUPRC, delta_AUROC vs the corresponding gated-ABMIL baseline, and per-branch alignment summary stats; include a note on whether MBA diversity bit (US3 acceptance #1)

**Checkpoint**: US3 result is in hand for WavLM and Whisper; per-branch attention alignment numbers are recorded.

---

## Phase 6: User Story 4 — TS-MIL: Target-Speaker Conditional MIL Head (Priority: P2)

**Goal**: Build prototype cache + add `TSMILHead` (concat and FiLM) + train+evaluate WavLM and Whisper variants on seen-child and cross-child splits.

**Independent Test**: `mil/prototypes/babar_vtc.npz` exists with one entry per (child, timepoint) covered by the train split; `mil/mil_results/wavlm_mil_tsmil_concat/test_metrics_tuned.json` exists; `missing_prototypes.json` documents any dropped clips.

**Depends on**: Phase 2 (foundational refactor for the 4-tuple training loop is reused; prototype-loading is added as an additional kwarg path).

- [ ] T037 [US4] Implement `mil/scripts/build_prototype_cache.py` reusing `pyannote/unified.py:559` `build_child_prototypes` logic; output `mil/prototypes/{frontend}.npz` (key=`{child_id}__{timepoint_norm}`, value=192-d L2-normalized float32) and `mil/prototypes/{frontend}_stats.csv`. CLI: `--frontend babar_vtc --train-csv whisper-modeling/seen_child_splits/train.csv --output mil/prototypes/babar_vtc.npz`
- [ ] T038 [US4] Run the cache builder once per split paradigm: seen-child (`whisper-modeling/seen_child_splits/train.csv`) and cross-child (`baselines/splits/train.csv` or equivalent) producing `mil/prototypes/babar_vtc.npz` and `mil/prototypes/babar_vtc_cross_child.npz`
- [ ] T039 [US4] Implement `TSMILHead` class in `mil/mil_model.py` with `mode: "concat" | "film"`, supporting both prototype injection flavors; forward signature `forward(h, prototype) -> (logit, attn)`
- [ ] T040 [US4] Wire `head: tsmil` dispatch into `build_mil_model` in `mil/mil_model.py`; reuse defaults from data-model.md §10
- [ ] T041 [US4] Modify `mil/mil_train.py` to load `prototype_cache` if set in config, merge per-clip prototype tensors into the training records, and pass them via a new `prototype` kwarg to `MILModel.forward`; drop clips with missing prototypes and write `missing_prototypes.json` per FR-021
- [ ] T042 [US4] Modify `MILModel.forward` in `mil/mil_model.py` to optionally accept a `prototype` kwarg and pass it to the head when the head is TSMIL (no-op for `gated_abmil`/`acmil`)
- [ ] T043 [P] [US4] Create `mil/configs/wavlm_mil_tsmil_concat.yaml` (data-model.md §10)
- [ ] T044 [P] [US4] Create `mil/configs/wavlm_mil_tsmil_film.yaml` (FiLM ablation)
- [ ] T045 [P] [US4] Create `mil/configs/whisper_mil_tsmil_concat.yaml`
- [ ] T046 [P] [US4] Create `mil/configs/wavlm_mil_tsmil_concat_cross_child.yaml` (points to `baselines/splits/` and `mil/prototypes/babar_vtc_cross_child.npz`)
- [ ] T047 [P] [US4] Submit training jobs for the four TS-MIL configs via `mil/slurm/train_mil.sh`
- [ ] T048 [US4] After training jobs complete, run `mil/slurm/eval_mil.sh`; verify each run dir contains the standard MIL output schema; record delta vs baselines

**Checkpoint**: TS-MIL results in hand for both flavors and both splits.

---

## Phase 7: User Story 5 — DSMIL Dual-Stream Aggregator (Priority: P2)

**Goal**: Add `DSMILAgg` to `mil/seg_model.py:build_aggregator()`; rerun the segment-MIL sweep with `dsmil` added to the aggregators list.

**Independent Test**: `mil/mil_results/seg_mil/{frontend}_dsmil/test_metrics_tuned.json` exists for all four frontends; `all_configs.json` gains four `dsmil` rows; per-stream logits saved in `test_predictions.csv`.

- [ ] T049 [US5] Implement `DSMILAgg` class in `mil/seg_model.py` per data-model.md §9 (max stream + cosine-distance attention stream + averaged BCE loss)
- [ ] T050 [US5] Register `dsmil` in `build_aggregator()` in `mil/seg_model.py:246`
- [ ] T051 [US5] Modify `mil/seg_train.py` to detect when the aggregator is `DSMILAgg` (returns 3-tuple `(logit_max, logit_attn, attn)`) and apply averaged BCE loss; final per-clip score = mean of the two sigmoid outputs; save both raw logits as new columns in `test_predictions.csv`
- [ ] T052 [US5] Add `dsmil` to the `aggregators` list in `mil/configs/seg_mil_sweep.yaml`
- [ ] T053 [US5] Submit `sbatch mil/slurm/seg_mil_sweep.sh` (resume-safe; only new (frontend × dsmil) cells will run)
- [ ] T054 [US5] After sweep completes, verify four new rows appear in `mil/mil_results/seg_mil/all_configs.json`; record delta_AUROC vs the best gated-attention baseline (babar_vtc 0.808) per frontend

**Checkpoint**: DSMIL results in hand for all four frontends.

---

## Phase 8: User Story 6 — Adaptive Pooling Operators (Priority: P2)

**Goal**: Add `AutoPoolAgg`, `ExpSoftmaxPoolAgg`, `GMAPAgg` to `mil/seg_model.py:build_aggregator()`; rerun the segment-MIL sweep.

**Independent Test**: 12 new run dirs (3 aggregators × 4 frontends) under `mil/mil_results/seg_mil/`; `all_configs.json` gains 12 new rows; AutoPool runs log final `alpha` to `config.json`; GMAP runs save per-head attention to `head_attention.csv`.

- [ ] T055 [P] [US6] Implement `AutoPoolAgg` class in `mil/seg_model.py` (scalar `alpha` initialized 0.0; pool over instance scores)
- [ ] T056 [P] [US6] Implement `ExpSoftmaxPoolAgg` class in `mil/seg_model.py` (no learnable param beyond score head; clamp logits ±10)
- [ ] T057 [P] [US6] Implement `GMAPAgg` class in `mil/seg_model.py` (n_heads=4 default; per-head sigmoid gate)
- [ ] T058 [US6] Register `auto_pool`, `exp_softmax_pool`, `gmap` in `build_aggregator()` factory
- [ ] T059 [US6] Modify `mil/seg_train.py` to (a) log final `alpha` to `config.json` for AutoPool runs, (b) save per-head attention CSV for GMAP runs
- [ ] T060 [US6] Add `auto_pool`, `exp_softmax_pool`, `gmap` to `aggregators` list in `mil/configs/seg_mil_sweep.yaml`
- [ ] T061 [US6] Submit `sbatch mil/slurm/seg_mil_sweep.sh` (resume-safe; runs the 12 new cells)
- [ ] T062 [US6] After sweep completes, verify 12 new rows in `all_configs.json`; record per-aggregator-per-frontend delta_AUROC vs baselines; compare adaptive operators against fixed (mean/max/attention) on each frontend

**Checkpoint**: Adaptive pooling results in hand. Final segment-MIL sweep covers 11 aggregators × 4 frontends = 44 cells.

---

## Phase 9: Polish & Cross-Cutting Concerns

**Purpose**: Cross-child evaluation for any positive seen-child results, conditional combined run (FR-010), and the documentation/synchronization sweep mandated by Constitution VI/VII.

- [ ] T063 [P] If US1 layer-sum delta_AUROC > 0 on any backbone AND US2 child-adapted delta_AUROC > 0, create `mil/configs/wavlm_mil_child_adapted_layersum.yaml` per data-model.md §5 and submit `sbatch mil/slurm/train_mil.sh mil/configs/wavlm_mil_child_adapted_layersum.yaml`; eval and record delta vs both `wavlm_mil_child_adapted` and `wavlm_mil_layersum` baselines (FR-010)
- [ ] T064 [P] For each US run that improved seen-child AUROC, repeat on cross-child split. Verify Whisper-MIL cross-child baseline (0.876) does not regress
- [ ] T065 [P] Append a Recent Changes entry to `CLAUDE.md` for each US that completed (US1–US6, plus conditional combined runs), mirroring the format of existing entries (e.g., "TinyVox MIL augmentation negative result" line). Include spec ID, date, SLURM job IDs, key delta numbers, and root-cause one-liner. Apply the file-deletion-discipline rule: do not delete prior entries
- [ ] T066 [P] Update `results_summary.md` master table with all new rows from US1–US6 grouped under a "spec-014 MIL Extensions" subsection; include both seen-child and (where run) cross-child columns; explicitly note any result that did NOT beat its baseline so the negative finding is preserved per Constitution VII
- [ ] T067 Run quickstart.md "Definition of Done" checklist top-to-bottom and confirm every item passes; mark spec-014 status "Complete" in `specs/014-mil-extensions-attention-and-layers/spec.md` header

---

## Dependencies & Execution Order

### Phase Dependencies

- **Phase 1 (Setup)**: No dependencies. Run T001–T003 first.
- **Phase 2 (Foundational)**: Depends on Phase 1. T004–T009 are sequential edits to `mil/mil_model.py` / `mil/mil_train.py` / `mil/mil_evaluate.py`; T010 (regression check) gates progression to Phase 4 / Phase 5.
- **Phase 3 (US2)**: Depends only on Phase 1 (no code changes). Can run **in parallel with Phase 2** since US2 only uses existing config/scripts.
- **Phase 4 (US1)**: Depends on Phase 2 completion (T010 PASS). Within Phase 4, T015–T017 are parallelizable; T018 SLURM submissions are parallel; T019 waits on all three; T020–T021 are sequential.
- **Phase 5 (US3)**: Depends on Phase 2 completion (T010 PASS). T022–T023 are sequential code edits; T024–T025 parallel config creation; T027 parallel SLURM submissions; T028–T031 sequential.
- **Phase 6 (Polish)**: Depends on Phase 3, 4, 5 completion. T032–T035 are largely parallel; T036 runs last.

### User Story Dependencies

- **US2 (P1)**: Independent of all other US — runs first because it's lowest-risk.
- **US1 (P1)**: Depends only on Phase 2 foundational refactor; independent of US2 and US3.
- **US3 (P2)**: Depends only on Phase 2 foundational refactor; independent of US1 and US2.

### Within Each User Story

- Config files before SLURM submission
- Training before evaluation
- Evaluation before post-hoc analysis (layer-weight inspection, per-branch alignment)
- Per-story `results_summary.md` row written only after the corresponding test metrics are committed

### Parallel Opportunities

- Phase 3 (US2 T012 train job) can launch in parallel with Phase 2 (T004–T009 code edits) since US2 does not touch the modified code.
- Phase 4 config creation (T015, T016, T017) is fully parallel — three different files.
- Phase 4 SLURM submissions (T018) — three jobs in parallel as ORCD allows.
- Phase 5 config creation (T024, T025) is parallel.
- Phase 5 SLURM submissions (T027) — two jobs in parallel.
- Phase 6 documentation tasks (T034, T035) are independent of T032/T033 SLURM jobs and can be drafted while jobs run.

---

## Implementation Strategy

### MVP Scope (Phase 3 + first half of Phase 6)

The cheapest path to a publishable result row is **US2 (child-adapted WavLM) only**: it requires no code changes, just running an already-configured SLURM job and writing one paragraph in `results_summary.md` and `CLAUDE.md`. Estimated calendar time: 1–2 days from job submission to documented result.

### Incremental Delivery Order

1. **Day 0**: T001–T003 (setup) and T011–T012 (US2 pre-flight + SLURM submit).
2. **Days 0–1 (parallel)**: T004–T009 foundational refactor (Phase 2), then T010 regression check.
3. **Day 1**: US2 training completes (T013), evaluate, write `results_summary.md` row (T014). MVP shipped.
4. **Days 1–2**: T015–T018 (Phase 4 US1 configs + submit) and T022–T027 (Phase 5 US3 implementation + configs + submit) launched together.
5. **Days 2–4**: SLURM jobs run; T019–T021 (US1 eval + write-up) and T028–T031 (US3 eval + branch analysis + write-up).
6. **Days 4–5**: Phase 6 polish — conditional combined run T032 if applicable, cross-child T033 for positives, CLAUDE.md/results_summary.md sweep T034–T036.

### Failure-Mode Branch Points

- **Phase 2 regression check (T010) fails** → stop; debug `BackboneExtractor` / `build_mil_model` changes until baseline reproduces. Do not proceed to Phase 4 or 5.
- **US2 result is negative** → still ship the row; the negative finding is publishable per CLAUDE.md "TinyVox negative result" precedent. Skip the conditional combined run T032.
- **US1 layer-weights collapse to one-hot at last layer** (T020) → bug in T004/T005; verify gradients flow through `layer_weights` and that `softmax` is computed every forward.
- **US3 ACMIL branches collapse** (T030) → tune `acmil_mba_diversity_weight` upward (0.5 or 1.0) and re-run one variant as an ablation, document in `results_summary.md`.
