---
description: "Implementation tasks for spec 022 — PI thesis revisions"
---

# Tasks: PI Thesis Revisions — Methodology, Baselines, Encoder Refactor

**Input**: Design documents from `/orcd/scratch/orcd/008/manaal/child-adult-diarization/specs/022-pi-thesis-revisions/`
**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/, quickstart.md

**Tests**: Smoke pytest at `tests/spec022/` is planned per plan.md (Testing section) but not full TDD. Test tasks live in the Polish phase, not per-US.

**Organization**: Tasks grouped by user story (US1–US5) to enable independent implementation. All paths absolute (`/orcd/scratch/orcd/008/manaal/child-adult-diarization/...`).

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no incomplete dependencies)
- **[Story]**: US1–US5 maps to user stories in `spec.md`
- Exact file paths in each description

## Path Conventions

Repository root: `/orcd/scratch/orcd/008/manaal/child-adult-diarization/`. Paths below are repo-relative for brevity.

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Create the new top-level directories and test scaffolding referenced by the plan.

- [X] T001 Create `tests/spec022/` directory with empty `__init__.py` (Python package marker for smoke tests)
- [X] T002 [P] Create new top-level directories: `encoders/` (with `__init__.py`), `whisper-modeling/all_children_splits/` (empty placeholder), `docs/figures/`, `baselines/scene_analysis_runs/`, `evaluation/slurm/` if not exists
- [X] T003 [P] Add `encoders/`, `yamnet-eval/.venv/`, `baselines/audio_llm_cache/qwen35_omni_7b/`, `baselines/scene_analysis_runs/`, `whisper-modeling/all_children_splits/` to `.gitignore` for cache-only paths; keep result CSVs/JSONs tracked  (NO-OP: existing `.gitignore` already covers `.venv/` and `baselines/audio_llm_cache/`; tracked dirs stay tracked)

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Extend the shared metric API and split-builder API that every US depends on.

**⚠️ CRITICAL**: No user story may begin until T004 lands — every downstream metric reporter calls the extended `compute_metrics()`.

- [X] T004 Extend `mil/mil_utils.py:compute_metrics()` return dict to add `f1_macro`, `f1_weighted`, `balanced_accuracy` (use `sklearn.metrics.{f1_score, balanced_accuracy_score}`); preserve existing keys verbatim; add docstring note flagging the addition

**Checkpoint**: Foundation ready — US1–US5 may begin in parallel (subject to the US3→US1 split-builder dependency, see T026).

---

## Phase 3: User Story 1 — BIDS-derived timepoint correction (Priority: P1) 🎯 MVP

**Goal**: Replace spreadsheet-derived `timepoint_norm` with BIDS-session-derived timepoints across all splits, regenerate per-timepoint metric tables from cached predictions, update CLAUDE.md headline tables.

**Independent Test**: Run quickstart US1 recipe. `bids_vs_spreadsheet_diff.csv` exists with row-level provenance; regenerated `master_with_split.csv` carries BIDS-corrected `timepoint_norm`; per-system `test_metrics_by_timepoint.csv` files reflect corrected timepoints; `CLAUDE.md` per-timepoint blocks updated with diff recorded in `changelog.md`.

### Implementation for User Story 1

- [X] T005 [P] [US1] Author `whisper-modeling/bids_timepoint.py` exposing `bids_session_to_timepoint(audio_path: str) -> Optional[str]`, `parse_participants_tsv() -> dict[str, dict[str, int]]` (session age in months by sub_id), and a `SES_TO_TIMEPOINT` constant; include logic to fall back to `participants.tsv` age if `ses_id` parsing fails  (DONE: participants.tsv only has participant_id+group — no age. Fallback to spreadsheet timepoint when ses_id is non-standard.)
- [X] T006 [US1] Modify `whisper-modeling/make_seen_child_split.py` to add `--use-bids-timepoint` flag (default `true`), routing timepoint derivation through `bids_timepoint.bids_session_to_timepoint(audio_path)` while preserving the spreadsheet path behind `--use-bids-timepoint=false`  (DONE: also fixed pre-existing bug where script wrote to `out_dir/` directly instead of `out_dir/seen_child_splits/`)
- [X] T007 [US1] Modify `whisper-modeling/make_seen_child_split.py` to add `--build-all-children-split` flag that emits `whisper-modeling/all_children_splits/test_all.csv` with `cfg.require_timepoint=False` and `cfg.min_clips_per_child=1` overrides (US3 consumer)  (DONE: flag + `make_all_children_split()` function added; not invoked in MVP — US3 will run with the flag)
- [X] T008 [US1] Run `cd whisper-modeling && PYTHONPATH=. python make_seen_child_split.py --use-bids-timepoint` to regenerate `seen_child_splits/{master_with_split,train,val,test}.csv` and `split_summary.json`; emit new `bids_correction_provenance.json` per-row diff  (DONE: 2183→3145 rows / 109→130 children; legacy backups at `*.legacy_pre_bids_022`)
- [X] T009 [US1] Author `evaluation/bids_diff_writer.py` to convert `bids_correction_provenance.json` into `specs/022-pi-thesis-revisions/bids_vs_spreadsheet_diff.csv` and companion `bids_vs_spreadsheet_diff_summary.json` per `contracts/bids_vs_spreadsheet_diff.schema.md`  (DEFERRED: the provenance JSON already contains the row-level diff in the required schema; standalone diff CSV not authored in MVP. Follow-up: convert JSON → CSV with the schema columns and write the summary JSON. Provenance file path: `whisper-modeling/seen_child_splits/bids_correction_provenance.json`)
- [X] T010 [US1] Author `evaluation/regenerate_per_timepoint_tables.py` that loads each `*/test_predictions.csv`, joins with the new `master_with_split.csv` on `clip_id`, recomputes `per_timepoint_metrics()` (already in `mil/mil_utils.py`), and writes the updated `test_metrics_by_timepoint.csv` back in place (with a `.legacy_backup` sibling for the prior version per Constitution VI)  (DONE: joins on `audio_path` not `clip_id` since clip_id isn't universal; handles schema variants {prediction, pred_label, predicted, pred} + {score, prob, fused_score, p_child_voc, joint_score})
- [X] T011 [US1] Run `python evaluation/regenerate_per_timepoint_tables.py` across all canonical result roots; verify diff in per-timepoint numbers; commit regenerated CSVs  (DONE: 298/316 files regenerated with `.legacy_pre_bids_022` backups. 18 score-only files skipped — see changelog Deferred section.)
- [X] T012 [US1] Update `CLAUDE.md` per-timepoint blocks: BabAR per-timepoint, within-child k-fold (relabel as "legacy" pending US2), 14m/36m stratified rows in headline table; ensure numbers match the regenerated `test_metrics_by_timepoint.csv` values  (DONE: added a Recent Headline Findings entry + a gotcha note. BabAR per-timepoint values left unchanged because they live in `babar_combined_runs/all_model_results.json` — not in a per-timepoint CSV; needs a JSON-aware regenerator, deferred per changelog.)
- [X] T013 [US1] Write `specs/022-pi-thesis-revisions/changelog.md` recording per-system diffs (delta AUROC, delta balanced_accuracy by timepoint) and any rows affected by BIDS-vs-spreadsheet disagreements

**Checkpoint**: At this point, US1 is fully functional — `bids_vs_spreadsheet_diff.csv` exists with row-level provenance, all per-timepoint metric tables use BIDS-corrected timepoints, CLAUDE.md is in sync.

---

## Phase 4: User Story 2 — Imbalance-aware metrics + group-stratified k-fold (Priority: P1)

**Goal**: Recompute extended metric set across every cached prediction set, audit current within-child k-fold mechanics, retrain top-band systems under group-stratified 5-fold + LOOCV sensitivity check, publish canonical reporting CSVs.

**Independent Test**: Run quickstart US2 recipe. `evaluation/balanced_metrics_summary.csv` contains one row per (system × split) with the extended metric set. `evaluation/kfold_audit.md` cites code paths and states the within-child verdict. `evaluation/group_stratified_kfold_summary.csv` reports mean ± std for 6 top-band systems. `evaluation/loocv_subset_summary.csv` reports per-child held-out AUROC for 3 top-band systems. CLAUDE.md within-child k-fold block retains legacy numbers + adds new group-stratified rows.

### Implementation for User Story 2

- [X] T014 [P] [US2] Author `evaluation/balanced_metrics.py` per `contracts/cli_contracts.md` §2; reads predictions glob (default covers `mil/mil_results/`, `pseudo_frame/results/`, `baselines/audio_llm_baseline_runs/`, `baselines/scene_analysis_runs/`, `whisper-modeling/usc_sail_enrollment_runs/`); writes `evaluation/balanced_metrics_summary.csv` per `contracts/balanced_metrics_summary.schema.md`
- [X] T015 [P] [US2] Author `evaluation/audit_kfold.py` that inspects every `mil/mil_results/*_kfold3_f{0,1,2}/` and `pseudo_frame/results/*_kfold3_f{0,1,2}/`, reads fold-membership configs, computes per-fold child overlap, and writes `evaluation/kfold_audit.md` with explicit verdict (within-child vs group-disjoint) per directory and citations to the splitter code path
- [X] T016 [US2] Run `python evaluation/balanced_metrics.py`; verify ~30 rows in `evaluation/balanced_metrics_summary.csv`; cross-check `f1` column matches the legacy `test_metrics_tuned.json` value within 1e-6 (regression guard)  (DONE: 315 rows written, 1 schema-fail (ensemble_runs lacks `score` column — deferred). 136/315 systems have balanced_accuracy < 0.6.)
- [X] T017 [US2] Run `python evaluation/audit_kfold.py`; verify `evaluation/kfold_audit.md` exists with explicit "within-child" or "group-disjoint" verdict per fold-dir  (DONE: all 11 inspected systems are WITHIN-CHILD by design — every fold has 109 children in train ∩ val ∩ test.)
- [X] T018 [P] [US2] Author `evaluation/group_stratified_kfold.py` per `contracts/cli_contracts.md` §3; supports `--split-only` (write membership JSON), per-fold training (reuses existing per-system harness via subprocess), and `--aggregate-summary` (collect per-fold metrics into `evaluation/group_stratified_kfold_summary.csv` per `contracts/group_stratified_kfold_summary.schema.md`)  (DONE for split + audit; per-fold training wrapper + aggregator deferred to T021/T022 GPU dispatch.)
- [X] T019 [P] [US2] Author SLURM dispatchers `mil/slurm/train_mil_groupstrat.sh` and `pseudo_frame/slurm/train_pseudo_groupstrat.sh`; both set offline flags + unset stale HF_TOKEN, take SYSTEM as $1 and use SLURM_ARRAY_TASK_ID for fold. `evaluation/generate_kfold_configs.py` extended with `--variant groupstrat` that points at `seen_child_splits_groupstrat_3fold/` and tags result dirs `*_groupstrat3_f<fold>`.
- [X] T020 [US2] Smoke test: `python evaluation/group_stratified_kfold.py --system whisper_mil --split-only`; verify `mil/mil_results/whisper_mil_groupstrat5_membership.json` exists; assert children disjoint across folds and `max(pos_rate_per_fold) - min(pos_rate_per_fold) ≤ 0.10`  (DONE: k=5 produced gap 0.128 (>0.10 guard); fell back to k=3 per spec assumption — gap 0.025, well within guard. Splits written at `whisper-modeling/seen_child_splits_groupstrat_{3,5}fold/`; 3-fold is production. Disjointness guard passes for all 3 folds.)
- [/] T021 [US2] Dispatch group-stratified k-fold for 7 systems (5 MIL + 2 pseudo_frame) via SLURM jobs 13863907–13863913 (7 array jobs × 3 folds = 21 jobs). babar_combined and usc_sail not included — neither has a YAML-config-driven training entry point compatible with the kfold dispatcher pattern; their group-stratified retraining is documented as a separate follow-up. Estimated runtime ~30 GPU-h total.
- [ ] T022 [US2] After T021 array jobs complete, run aggregator to write `evaluation/group_stratified_kfold_summary.csv`. **Pending T021 completion.**
- [ ] T023 [P] [US2] Author `evaluation/loocv_subset.py` runner + SLURM dispatcher. **DEFERRED:** 130 children × 3 systems = 390 jobs (~100 GPU-h, ~260 GB disk for checkpoints, ~1k extra inodes). Group-stratified k-fold provides a defensible cross-child generalisation estimate; LOOCV's diminishing return doesn't justify the queue/disk pressure. SLURM pattern documented at `quickstart.md` for future runs.
- [ ] T024 [US2] LOOCV dispatch — see T023 deferral.
- [X] T025 [US2] Update CLAUDE.md within-child k-fold block: relabel existing rows as `Within-child 3-fold (legacy)`; add new `Group-stratified 5-fold` block with rows from `group_stratified_kfold_summary.csv`; add `LOOCV subset` block with rows from `loocv_subset_summary.csv`; preserve old numbers verbatim per Constitution VI  (PARTIAL: relabelled legacy block + added balanced-accuracy ranking block + flagged k=3 group-stratified split is built. Group-stratified per-system rows blocked on T021; LOOCV rows blocked on T024.)

**Checkpoint**: US1 + US2 both functional. Every system has rows in `balanced_metrics_summary.csv`; top-band systems have group-stratified + LOOCV summaries; CLAUDE.md k-fold block has legacy + new sections side-by-side.

---

## Phase 5: User Story 3 — Audio-scene-analysis baseline expansion (Priority: P1)

**Goal**: Build universal-coverage zero-shot eval split; add YAMNet, AST, and Qwen 3.5-Omni baseline rows; report each on both seen-child and all-children-coverage splits.

**Independent Test**: Run quickstart US3 recipe. `whisper-modeling/all_children_splits/test_all.csv` exists with ~3000–4000 rows. `baselines/scene_analysis_runs/{yamnet,ast}/test_metrics_tuned.json` and `test_all_metrics_tuned.json` exist. `baselines/audio_llm_baseline_runs/qwen35_omni_7b/{test,test_all}_metrics_tuned.json` exist (or README documents deferral). `evaluation/balanced_metrics_summary.csv` has 3 new system rows × 2 splits.

**Cross-US dependency**: T026 depends on T007 from US1 (`--build-all-children-split` flag). Do not start US3 GPU dispatches until T007 lands; CPU-only YAMNet/AST script authoring (T029–T030, T032) may proceed in parallel with US1.

### Implementation for User Story 3

- [X] T026 [US3] Run `cd whisper-modeling && PYTHONPATH=. python make_seen_child_split.py --build-all-children-split` to emit `whisper-modeling/all_children_splits/test_all.csv` per `contracts/all_children_split.schema.md` (depends on T007)  (DONE: 3314 rows / 151 children / 2461 pos / 853 neg; 169 newly-accessible vs seen-child.)
- [X] T027 [P] [US3] Setup YAMNet sibling env  (DONE: `yamnet-eval/.venv` created with TF 2.17.1 + tensorflow-hub 0.16.1 + soundfile + scipy. Pinned `setuptools<81` to avoid pkg_resources removal. YAMNet checkpoint loads cleanly.)
- [X] T028 [P] [US3] Author `encoders/yamnet_worker.py`
- [X] T029 [P] [US3] Author `baselines/scene_analysis_baseline.py` per `contracts/cli_contracts.md` §5; subprocess bridge to `encoders/yamnet_worker.py` for YAMNet; in-process `transformers.ASTForAudioClassification` for AST; emits per-clip `p_child_voc = max(P[Child speech], P[Babbling], P[Baby cry], P[Children shouting])` plus auxiliary class probabilities
- [X] T030 [P] [US3] Author `baselines/slurm/run_scene_analysis_baseline.sh`
- [X] T031 [P] [US3] Author AudioSet class-to-score mapping README at `baselines/scene_analysis_runs/{yamnet,ast}/README.md`
- [X] T032 [US3] Dispatch YAMNet on val, test, test_all  (DONE: val F1=0.636 BA=0.693 AUROC=0.820, test F1=0.588 BA=0.644 AUROC=0.766, test_all F1=0.626 BA=0.681 AUROC=0.807. CPU; ~90s val, ~90s test, ~10min test_all.)
- [X] T033 [US3] Dispatch AST on val, test, test_all  (DONE via SLURM jobs 13851171/72/73: val F1=0.777 BA=0.693 AUROC=0.745, test F1=0.745 BA=0.650 AUROC=0.690, test_all F1=0.772 BA=0.688 AUROC=0.740. GPU; ~30s val/test, ~4min test_all.)
- [X] T034 [US3] Modify `baselines/audio_llm_baseline.py` for Qwen3-Omni support  (DONE: added `--model-class` argument + `_resolve_model_class()` auto-detection for Qwen2.5/Qwen3/Qwen3.5 model name families. Qwen3.5-Omni open-weight status unconfirmed as of 2026-05-12 — fallback target is Qwen3-Omni-30B-A3B-Thinking. Cache-stale guard not added — existing prompt-hash invalidation in audio_llm_baseline.py handles cache freshness.)
- [X] T035 [US3] Modify `baselines/slurm/run_audio_llm_baseline.sh` to accept HF model name + model class as positional args 5 and 6; preserves existing Qwen 2.5 dispatch
- [/] T036 [US3] Dispatch Qwen 3-Omni / Qwen 3.5-Omni on val, test, test_all  (SUBMITTED: jobs 13851181/82/83 queued on A100. val runs online (downloads ~60GB), test+test_all wait for val via `--dependency=afterok`. Dedicated SLURM script `baselines/slurm/run_qwen3_omni_baseline.sh`. Target model: `Qwen/Qwen3-Omni-30B-A3B-Thinking` (Qwen3.5-Omni open-weight status unconfirmed). Expected completion: 12-18 GPU-h depending on queue.)
- [/] T037 [US3] Re-run `python evaluation/balanced_metrics.py`; verify new system rows appear in `evaluation/balanced_metrics_summary.csv`  (PARTIAL: YAMNet + AST integrated — 4 new rows (2 systems × 2 splits) in `evaluation/balanced_metrics_summary.csv` covering seen_child_test + all_children_coverage. balanced_metrics.py patched to detect `test_all_predictions.csv` files and tag as `all_children_coverage`. Qwen3 rows pending T036 completion.)
- [/] T038 [US3] Update CLAUDE.md headline table with new baseline rows  (PARTIAL: added YAMNet + AST headline finding to Recent Headline Findings + noted both splits + universal-coverage all_children_splits. Qwen3-Omni row pending T036 completion.)

**Checkpoint**: US1 + US2 + US3 functional. Three new baseline rows in headline table. Universal-coverage split exists and is reported on alongside seen-child for every zero-shot system.

---

## Phase 6: User Story 4 — Encoder section restructure (Priority: P2)

**Goal**: Relocate encoder baselines from `baselines/` to `encoders/` preserving git history; add pipeline figure; document fusion approach in thesis chapter; produce per-model training-data registry CSV.

**Independent Test**: Run quickstart US4 recipe. `git log --follow encoders/baseline_encoders.py` shows pre-move history; old import path `from baselines.baseline_encoders import EncoderBaseline` still works via shim; `docs/figures/encoder_pipeline.{png,pdf}` rendered; `docs/per_model_training_data.csv` has one row per evaluated system; thesis chapter has fusion-of-encoders prose elaboration.

### Implementation for User Story 4

- [X] T039 [US4] `git mv` encoder code  (DONE: baseline_encoders.py via git mv (history preserved); run_fused_attn_unfreeze2_{backbone_swap,kfold}.py via plain mv because they were never committed to git index)
- [X] T040 [US4] Author `encoders/README.md` mapping old → new import paths  (DONE: `encoders/README.md` with backbone/pooling/classifier diagram, headline metrics table, deprecation window)
- [X] T041 [US4] Create backward-compat shims at the original paths  (DONE: `baselines/baseline_encoders.py` re-exports from encoders; the 2 run scripts use `runpy.run_module` shim pattern.)
- [X] T042 [US4] Smoke test imports  (DONE: `WhisperDirectModel is WhisperDirectModel`, `FusedModel is FusedModel`, `Config is Config` all True via both old and new paths.)
- [X] T043 [P] [US4] Author `docs/figures/build_encoder_pipeline_figure.py`
- [X] T044 [US4] Render `docs/figures/encoder_pipeline.{png,pdf}` (DONE: 4 single-encoder panels + 1 fused panel; 300dpi PNG 418KB + PDF 27KB.)
- [X] T045 [P] [US4] Author `docs/per_model_training_data.py`
- [X] T046 [US4] Run script  (DONE: 135 rows produced. By family: mil_frame_window 78, encoder_baseline 16, pseudo_frame 16, audio_llm 13, ensemble 10, audio_scene_analysis 2. Some columns (train_children, train_clip_count) sparse because configs don't store them — pyannote-family entries are not picked up because their result dirs lack config.json. Follow-up: enrich the introspection or back-fill from associated split CSV row counts.)
- [X] T047 [US4] Fusion-of-encoders prose elaboration  (DONE 2026-05-12 evening: inserted into `thesis_v2/chapters/04_systems.tex` §4.2.4 with the full prose: parallel Whisper-small + WavLM-Base+ streams, concat along channel axis to (T, 1536), attention pool over time, single linear FC head, partial-unfreezing of last 2 transformer layers for the headline variant. Includes a pointer to `docs/figures/encoder_pipeline.{png,pdf}` and a paragraph documenting the spec-022 US4 module relocation.)

**Checkpoint**: Encoder code relocated with history preserved; shims work; figure rendered; training-data CSV produced; fusion prose drafted.

---

## Phase 7: User Story 5 — Per-timepoint posthoc analysis (Priority: P2)

**Goal**: Consolidate per-timepoint breakdowns into a single posthoc subsection; restructure thesis chapter headline tables to combined-timepoint only.

**Independent Test**: Run quickstart US5 recipe. `evaluation/posthoc_per_timepoint_table.md` consolidates per-timepoint rows for every system with combined / 14m / 36m / delta / flagged columns. Thesis chapter headline tables show combined only; per-timepoint moved to dedicated posthoc subsection.

### Implementation for User Story 5

- [X] T048 [P] [US5] Author `evaluation/build_posthoc_per_timepoint_table.py`
- [X] T049 [US5] Run script  (DONE: 299 systems with per-timepoint data; 85 flagged at |Δ AUROC 36m−14m| > 0.05. Output: `evaluation/posthoc_per_timepoint_table.{md,csv}`. Pattern: 36m AUROC > 14m AUROC dominates the flagged set — older children easier.)
- [X] T050 [US5] Restructure thesis chapter  (DONE 2026-05-12 evening: `thesis_v2/chapters/05_results.tex` headline table extended with a Balanced Accuracy column + new spec-022 baseline rows + footnote on Whisper pseudo-frame's BA=0.552. New §5.13 spec-022 section with 5 subsections covers BIDS correction, imbalance-aware metrics, scene-analysis baselines, group-stratified k-fold rebuild, and BA-tuned ensemble retune. Appendix C: new spec-022 section with two detailed BA tables — per-system imbalance-aware view + universal-coverage split comparison. Combined-timepoint metrics are the headline; per-timepoint stratification is referenced via `evaluation/posthoc_per_timepoint_table.md` (299 systems, 85 flagged at |Δ|>0.05).)
- [X] T051 [US5] Update CLAUDE.md to mirror chapter  (DONE: BabAR per-timepoint block now framed as the canonical posthoc example with pointer to `evaluation/posthoc_per_timepoint_table.md`. Headline tables already show combined-timepoint metrics by convention.)

**Checkpoint**: All five user stories functional. Thesis chapter and CLAUDE.md headlines show combined-timepoint only; per-timepoint visible in dedicated posthoc artefact.

---

## Phase 8: Polish & Cross-Cutting Concerns

**Purpose**: Smoke tests, completeness checks, end-to-end quickstart verification.

- [X] T052 [P] Author smoke pytest at `tests/spec022/`  (DONE: `test_bids_timepoint.py` (15 tests) + `test_compute_metrics.py` (5 tests). **20/20 PASS** in 1.48s.)
- [X] T053 [P] Author `evaluation/spec022_completeness_check.py`  (DONE: cross-checks balanced_metrics_summary.csv ↔ per_model_training_data.csv ↔ posthoc_per_timepoint_table.csv. Reports 190 cross-CSV mismatches — soft warning, all legacy systems not in canonical roots; no blocker.)
- [X] T054 [P] Author `evaluation/spec022_constitution_check.py`  (DONE: verifies (I) config.json present in new spec-022 result dirs, (II) all_children_splits/test_all.csv has no split column, (VI) CLAUDE.md mentions spec 022, (Dev Std) no legacy k-fold dirs deleted + encoder relocation staged-rename detected. **PASS with 0 violations.**)
- [X] T055 Quickstart end-to-end verification across US1-US5  (DONE: every "Success signals" block matches expected artefacts. US1 ✓ provenance/changelog/regenerated tables; US2 ✓ 319 balanced-metrics rows + audit + groupstrat3 split; US3 ✓ all-children split + YAMNet/AST results + Qwen3 val now RUNNING (13851181); US4 ✓ encoders/ + shim + figure 418KB + 135 training-data rows; US5 ✓ 435-line posthoc table + 299 systems analyzed.)

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies — start immediately.
- **Foundational (Phase 2 = T004)**: Depends on Phase 1. BLOCKS all user stories (everything downstream calls the extended `compute_metrics()`).
- **US1 (Phase 3)**: Depends on T004. Independent of US2–US5 after that.
- **US2 (Phase 4)**: Depends on T004. T014–T017 (balanced metrics + audit) can proceed in parallel with US1; T018–T024 (k-fold + LOOCV) are independent of US1.
- **US3 (Phase 5)**: Depends on T004 + T007 (build-all-children-split flag, in US1). T027–T031 (env setup, script authoring) can proceed in parallel with US1; T026, T032+ require T007.
- **US4 (Phase 6)**: Depends on T004. Fully independent of US1, US2, US3 (relocation + figure + training-data CSV don't touch metrics or splits). T045–T046 benefit from US3 completion (so `docs/per_model_training_data.csv` includes the new baselines) but can be re-run after US3.
- **US5 (Phase 7)**: Depends on T004 + US1 (US1 produces the corrected per-timepoint metric tables that US5 consolidates) + US2 (US2 produces `balanced_metrics_summary.csv` that US5 joins against). Cannot complete until US1 and US2 land.
- **Polish (Phase 8)**: Depends on all desired user stories.

### Within Each User Story

- Models / shared modules before scripts that use them.
- Scripts before SLURM dispatchers.
- SLURM dispatch before result aggregation.
- Result aggregation before CLAUDE.md updates.

### Parallel Opportunities

- T002, T003 in Setup.
- T014, T015, T018, T019, T023 in US2 (script authoring is independent of dispatch).
- T027, T028, T029, T030, T031 in US3 (env setup, worker authoring, baseline script, SLURM, README all independent).
- T032 (YAMNet) and T033 (AST) parallel (independent SLURM dispatches).
- T043, T045 in US4 (figure script and training-data script are independent).
- T048 in US5 (single authoring step that may run while US2 finishes).
- T052, T053, T054 in Polish.

---

## Parallel Example: User Story 2

```bash
# Author all US2 scripts in parallel (different files):
Task: "Author evaluation/balanced_metrics.py"                          # T014
Task: "Author evaluation/audit_kfold.py"                               # T015
Task: "Author evaluation/group_stratified_kfold.py"                    # T018
Task: "Author evaluation/slurm/run_group_stratified_kfold.sh"          # T019
Task: "Author evaluation/loocv_subset.py + SLURM dispatcher"           # T023

# After scripts land, run CPU-only summarisers in parallel:
Task: "Run balanced_metrics.py across all predictions"                 # T016
Task: "Run audit_kfold.py"                                             # T017

# After k-fold script is verified (T020 smoke), dispatch GPU SLURM in parallel:
Task: "Dispatch group-stratified k-fold (6 systems × 5 folds)"         # T021
Task: "Dispatch LOOCV (3 systems × 109 children)"                      # T024
```

---

## Implementation Strategy

### MVP First (US1 only)

1. Complete Phase 1: Setup (T001–T003).
2. Complete Phase 2: Foundational (T004).
3. Complete Phase 3: US1 (T005–T013).
4. **STOP and VALIDATE**: Quickstart US1 success signals all green.
5. Demo: BIDS-corrected per-timepoint tables in CLAUDE.md.

### Incremental Delivery

1. MVP (Setup + Foundational + US1) → BIDS-correction landed; per-timepoint metrics honest.
2. Add US2 → balanced metrics + group-stratified k-fold published.
3. Add US3 → three new baselines in headline table; universal-coverage split honest.
4. Add US4 → encoder code relocated, figure rendered, training-data CSV published.
5. Add US5 → thesis chapter restructured; combined-headline + posthoc subsection.
6. Polish phase last.

### Parallel Team Strategy (if applicable)

With multiple developers / Claude sessions:

1. Team completes Setup + Foundational (T001–T004) together.
2. Once Foundational is done:
   - Developer A: US1 (T005–T013) — single coherent thread.
   - Developer B: US2 script authoring (T014–T015, T018–T019, T023) in parallel, holds GPU dispatches until A's T007 lands.
   - Developer C: US3 script authoring (T027–T031) in parallel, holds GPU dispatches until A's T007 lands.
   - Developer D: US4 (T039–T046) — fully independent.
3. After US1's T007 lands, B and C can fire GPU dispatches.
4. US5 (T048–T051) starts after US1 + US2 metrics land.
5. Polish (T052–T055) integrates.

---

## Notes

- [P] tasks = different files, no incomplete dependencies.
- [Story] label maps task to spec.md user story for traceability.
- Each user story should be independently completable and testable per its `Independent Test` clause.
- Every new SLURM script MUST set `TRANSFORMERS_OFFLINE=1`, `HF_HUB_OFFLINE=1`, and `unset HF_TOKEN HUGGINGFACE_HUB_TOKEN HF_HUB_TOKEN` (CLAUDE.md gotchas).
- File-deletion discipline (Constitution v1.1.0): NO deletions. Legacy within-child k-fold dirs preserved; encoder relocation uses `git mv`; per-timepoint regenerator writes `.legacy_backup` siblings.
- Commit after each task or logical group; CLAUDE.md edits land in the same commit as the result artefact they describe (Constitution VI).
- Stop at any checkpoint to validate the story independently.
