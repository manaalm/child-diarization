# Feature Specification: PI Thesis Revisions — Methodology, Baselines, Encoder Refactor

**Feature Branch**: `021-post-thesis-future-work` (spec dir-only; no new branch)
**Created**: 2026-05-12
**Status**: Draft
**Input**: User description (PI suggestions): swap Qwen 2.5→3.5; add YAMNet and AST as audio-scene-analysis child baselines; move encoders out of `baselines/`; add a steps figure for encoders; elaborate on encoder fusion; run zero-shot baselines across every child (ignore missing-timepoint filter); pull session timepoints from BIDS directory structure rather than the spreadsheet and propagate corrections; ensure every metric handles class imbalance (class-weighted F1 and/or balanced accuracy) and report balanced accuracy everywhere; document the training-data set used by each model (encoders, MIL, all variants); investigate what the existing k-fold evaluation is actually splitting, and replace within-child k-fold with group-stratified k-fold (with leave-one-child-out as a sensitivity check); treat per-timepoint analyses as posthoc.

This spec packages the PI feedback from 2026-05-12 as five independent user-story slices. Slices are ordered so that methodology corrections (US1, US2) land first because they cascade into every downstream rerun; baseline expansion (US3) lands next because new rows are net-additive to the headline table; encoder restructure and per-timepoint posthoc are presentation-quality deliverables with low blast radius.

## User Scenarios & Testing *(mandatory)*

### User Story 1 — BIDS-derived timepoint correction (Priority: P1)

A reader of the thesis chapter wants to trust that every per-timepoint number (14-month vs. 36-month) was computed against the canonical age assignment recorded in the BIDS dataset itself, not against a separately-maintained spreadsheet that may have drifted. Today, `make_seen_child_split.py` reads `anotated_processed.csv` for `timepoint_norm`. The PI's directive is to derive timepoints from the BIDS session directory structure (`/orcd/scratch/bcs/001/sensein/sails/BIDS_data/sub-XXX/ses-YYY/`) and to update every downstream artefact — splits CSVs, per-timepoint metric tables, multi-child suppressor metadata, age-stratified MIL evals — to use the corrected mapping.

**Why this priority**: Every per-timepoint result in the headline tables and every cohort-routing decision in spec-012 depends on the timepoint label. If BIDS sessions and the spreadsheet disagree for any meaningful number of children, every downstream rerun must use the corrected mapping; doing this after re-running new baselines would force a second round of reruns. Cheapest first.

**Independent Test**: A diff between the spreadsheet-derived timepoint column and the BIDS-derived timepoint column for every child in the dataset is produced, the disagreements are enumerated, and a corrected `master_with_split.csv` is regenerated. Existing per-timepoint metric tables can then be regenerated from cached predictions without re-running any model.

**Acceptance Scenarios**:

1. **Given** the BIDS session directory structure has been parsed for every child in `whisper-modeling/seen_child_splits/master_with_split.csv`, **When** the BIDS-derived timepoint is compared row-by-row against the spreadsheet `timepoint_norm`, **Then** a `bids_vs_spreadsheet_diff.csv` is written listing every (child_id, clip_id) where the two disagree, the disagreement is summarised with row counts per (BIDS-timepoint, spreadsheet-timepoint) cell, and a short rationale per disagreement is recorded if the BIDS session ID is non-standard.
2. **Given** the BIDS-derived timepoint mapping is canonical, **When** the splits are regenerated, **Then** `master_with_split.csv`, `train.csv`, `val.csv`, `test.csv` under `whisper-modeling/seen_child_splits/` are rewritten with the corrected `timepoint_norm` column, the seed-42 stratification is preserved, and the per-system per-timepoint metric tables (e.g., `mil/mil_results/*/test_metrics_by_timepoint.csv`) are regenerated from the existing cached predictions.
3. **Given** the regeneration produced new per-timepoint metric tables, **When** the deltas vs. the old tables exceed bootstrap noise on any system, **Then** the affected rows in `CLAUDE.md`'s headline tables, in the within-child 3-fold AUROC list, and in the BabAR per-timepoint block are updated and the diff is recorded in a short changelog note inside this spec directory.

---

### User Story 2 — Imbalance-aware metrics + group-stratified k-fold (Priority: P1)

A reader of the thesis chapter wants every reported metric to be honest under the 76%-positive test imbalance, and wants the k-fold cross-validation to actually measure cross-child generalisation rather than within-child fold variance. Today, `compute_metrics()` returns F1 plus AUROC plus AUPRC; the trivial-predict-all baseline already scores F1=0.864, which means many "negative" rows in the headline tables sit at the floor. The current 3-fold evaluation splits *clips* across folds for the same child population, not *children*. The PI's directive: add balanced accuracy everywhere, add class-weighted F1 alongside macro F1, document what the existing k-fold is actually splitting, and replace it with group-stratified k-fold (with leave-one-child-out as a sensitivity check on a subset).

**Why this priority**: Imbalance correction and proper cross-child folds are methodological table-stakes; reviewers will demand both. Doing this before US3 (new baselines) means the new baselines are reported with the corrected metrics from day one, avoiding a second-pass rerun.

**Independent Test**: A new `evaluation/balanced_metrics.py` recomputes balanced accuracy + class-weighted F1 + macro F1 + the existing F1/AUROC/AUPRC from every existing per-system `test_predictions.csv`, producing a single `evaluation/balanced_metrics_summary.csv` that fully replaces the headline table. A new `evaluation/group_stratified_kfold.py` produces a `evaluation/group_stratified_kfold_summary.csv` with mean ± std per system, alongside a documentation note explaining the contrast with the current within-child k-fold.

**Acceptance Scenarios**:

1. **Given** every existing per-system `test_predictions.csv` is on disk, **When** `evaluation/balanced_metrics.py` is run, **Then** `evaluation/balanced_metrics_summary.csv` contains one row per system with columns `{f1_macro, f1_weighted, balanced_accuracy, auroc, auprc, trivial_floor_f1, trivial_floor_balanced_acc}` and every numeric column is computed from the same prediction file used in the existing headline tables.
2. **Given** the existing `*_kfold3_f{0,1,2}/` directories have been inspected, **When** an audit doc `evaluation/kfold_audit.md` is written, **Then** it states exactly which children appear in train vs. test in each fold, confirms or refutes that the current 3-fold is within-child, and quotes the relevant code paths.
3. **Given** group-stratified k-fold has been implemented (default k=5, children disjoint per fold, stratification preserving the positive-rate within bootstrap noise), **When** the top-band systems (Whisper pseudo-frame, Whisper-medium-MIL, Whisper-MIL, Whisper-MIL TS-MIL concat, BabAR/VTC-KCHI, USC-SAIL) are re-evaluated under the new scheme, **Then** `evaluation/group_stratified_kfold_summary.csv` reports mean ± std AUROC and balanced accuracy per system and the result is added to a new sub-section of `CLAUDE.md`'s within-child k-fold block (the old within-child numbers are kept and labelled).
4. **Given** the leave-one-child-out (LOOCV) sensitivity check is added on a cost-controlled subset, **When** LOOCV is run on at most three top-band systems (e.g., Whisper pseudo-frame, Whisper-medium-MIL, BabAR), **Then** per-child held-out AUROC distributions are written to `evaluation/loocv_subset_summary.csv` and any system whose group-stratified k-fold AUROC mean differs from its LOOCV mean by more than 0.03 is flagged.

---

### User Story 3 — Audio-scene-analysis baseline expansion (Priority: P1)

A reader of the thesis chapter wants the comparison band to include current-generation audio-scene-analysis baselines (YAMNet, AST) in addition to the now-canonical Qwen-Omni audio LLM, and wants the audio LLM rerun on the latest model (Qwen 3.5-Omni). The PI also wants every zero-shot baseline reported across every child in the dataset — not just children who survived the ≥5-clip-per-timepoint filter — to make the zero-shot coverage claim honest.

**Why this priority**: Each new baseline is an additive row in the headline table; none requires touching existing trained models. Qwen 3.5 is a single-line model swap plus cache invalidation. YAMNet and AST are off-the-shelf and produce a child-speech probability per clip with no training. Universal-coverage zero-shot is a re-evaluation pass on existing models against a broader split.

**Independent Test**: Three new rows appear in `evaluation/balanced_metrics_summary.csv`: (1) Qwen 3.5-Omni 0-shot, (2) YAMNet child-speech probability, (3) AST child-speech probability. Each row reports both the seen-child-split numbers (for direct comparison to trained systems) and the all-children-coverage numbers (the new universal-coverage split).

**Acceptance Scenarios**:

1. **Given** Qwen 3.5-Omni is available on HuggingFace and reachable from the `child-vocalizations` env, **When** `baselines/audio_llm_baseline.py` is invoked with the new model slug after the Qwen 2.5 cache is invalidated (`rm -rf baselines/audio_llm_cache/qwen35_omni_7b/`), **Then** `baselines/audio_llm_baseline_runs/qwen35_omni_7b/{val,test}_metrics_tuned.json` and `..._predictions.csv` are produced and a row appears in `evaluation/balanced_metrics_summary.csv`.
2. **Given** YAMNet (TFHub) and AST (HF) checkpoints are available, **When** `baselines/scene_analysis_baseline.py --model {yamnet|ast}` is run, **Then** the child-speech-class probability (AudioSet "Child speech" or nearest semantic equivalent) is computed per clip, the val threshold is tuned, and `baselines/scene_analysis_runs/{yamnet,ast}/test_metrics_tuned.json` is produced.
3. **Given** the universal-coverage all-children split has been built (every SAILS clip with usable annotations, no timepoint-balance filter), **When** Qwen 3.5-Omni, YAMNet, and AST are evaluated against it, **Then** `evaluation/balanced_metrics_summary.csv` has both seen-child and all-children rows for each zero-shot system, and the gap between the two is summarised in a short note (do zero-shot baselines preserve their seen-child ranking on the broader population?).
4. **Given** the AudioSet class taxonomy does not have a one-to-one "child vocalising" label, **When** the class-to-score mapping for YAMNet/AST is documented, **Then** the chosen aggregation (e.g., `max(P[child speech], P[babbling], P[crying])` if relevant) is recorded in the baseline's README with citations to the AudioSet ontology entries used.

---

### User Story 4 — Encoder section restructure (Priority: P2)

A reader of the encoder section in the thesis chapter wants (a) encoder baselines to live in their own top-level module rather than buried in `baselines/`, (b) a pipeline figure that explicitly shows the encoder steps (waveform → encoder → pooling → linear head → score), (c) explicit prose elaborating how the fused-encoder variant combines the Whisper and WavLM streams, and (d) a per-model table documenting exactly what data each variant was trained on (which split, which children, which clip counts, whether synthetic was included).

**Why this priority**: Presentation quality. No metric changes. Risk-free relative to US1/US2/US3, so it can land in parallel with the methodology reruns.

**Independent Test**: The thesis chapter renders with the new figure embedded; the encoder section's prose explicitly answers "how are the two streams fused?" without forcing a reader to consult code; the per-model training-data table appears in the chapter and is reproducible from a script that introspects the saved configs in `baselines/audio_llm_baseline_runs/`, `mil/mil_results/`, `pseudo_frame/results/`, etc.

**Acceptance Scenarios**:

1. **Given** `baselines/` currently contains both audio-LLM scripts and encoder baselines, **When** the encoder code is moved to `encoders/` with backward-compatible import shims preserved for one release cycle, **Then** all existing run scripts continue to work without flag changes, and `git mv` history is preserved for blame.
2. **Given** the encoder pipeline figure has been authored, **When** it is embedded in the thesis chapter, **Then** it shows the four canonical steps (input waveform → frozen encoder → pooling → linear classifier) for both the mean-pool and attention-pool variants, plus a fifth panel for the fused-encoder variant.
3. **Given** the fusion-of-encoders prose elaboration has been written, **When** a reader unfamiliar with the codebase reads it, **Then** they can describe in their own words: which encoders are fused, at what stage (early/late), with what weighting, and how the fusion-head linear classifier is parameterised.
4. **Given** the per-model training-data summary script has been written, **When** it is run against the live result dirs, **Then** `docs/per_model_training_data.csv` contains one row per evaluated system with columns `{system_name, train_split, train_children, train_clip_count, includes_synthetic, synth_corpus_version, eval_split}`.

---

### User Story 5 — Per-timepoint posthoc analysis (Priority: P2)

A reader of the thesis chapter wants per-timepoint stratification (14-month vs. 36-month) treated as a posthoc analysis appendix rather than as a primary headline reporting axis. The PI's directive is that the headline metrics should be combined across timepoints, with the per-timepoint breakdown relegated to a single dedicated posthoc section so the reader's eye lands on the system-level comparison first.

**Why this priority**: Presentation choice that affects how reviewers read the chapter. Independent of any model rerun.

**Independent Test**: The thesis chapter is restructured so that headline tables show only combined-timepoint metrics, and a new appendix-style section assembles the per-timepoint breakdown for every system into one consolidated table.

**Acceptance Scenarios**:

1. **Given** the per-timepoint metric tables have been regenerated under US1 with corrected timepoints, **When** the thesis chapter's results section is restructured, **Then** the headline comparison table contains only combined-timepoint metrics and a separate `### Posthoc: per-timepoint stratification` subsection contains the per-timepoint breakdown for every system in a single combined table.
2. **Given** the posthoc section is in place, **When** the reader looks for cohort-specific patterns, **Then** the section explicitly calls out any system whose 14-month vs. 36-month delta exceeds 0.05 AUROC, with a short interpretation.

---

### Edge Cases

- Children whose BIDS sessions and spreadsheet timepoints disagree but who currently appear in the seen-child split: kept with corrected timepoint by default; ambiguous cases annotated in `bids_vs_spreadsheet_diff.csv`.
- Children present in BIDS but missing from the spreadsheet (or vice versa): asymmetry documented; the all-children-coverage split for zero-shot baselines (US3) includes every annotated child regardless of spreadsheet presence.
- Children whose BIDS structure has only one session: included in the all-children split (US3); excluded from the seen-child split per existing convention.
- Group-stratified k-fold (US2) with very small per-fold positive counts: minimum-positive-count guard per fold; fall back to k=3 if k=5 violates the guard.
- YAMNet/AST AudioSet ontology lacks a precise "child vocalising" label: aggregation rule documented (US3 AS-4) and flagged as a methodological caveat in the chapter.
- Qwen 3.5-Omni `AutoProcessor` may inherit the Qwen 2.5 torchvision dependency footgun: if so, the same `pip install --no-deps torchvision==<matched>` workaround applies (per CLAUDE.md gotcha) and is documented in the new baseline's README.
- LOOCV (US2 AS-4) cost: capped to at most three top-band systems on the seen-child split (109 folds × 3 systems is tractable; full 109 × 20 systems is not).
- Existing `CLAUDE.md` headline tables under the within-child k-fold paragraph: old within-child numbers are not deleted; relabelled as `Within-child 3-fold (legacy)` and a new `Group-stratified 5-fold` block is added below for direct comparison.

## Requirements *(mandatory)*

### Functional Requirements

**BIDS timepoint correction (US1)**

- **FR-001**: System MUST derive each (child_id, clip_id)'s timepoint from the BIDS session directory under `/orcd/scratch/bcs/001/sensein/sails/BIDS_data/sub-XXX/ses-YYY/` rather than from `anotated_processed.csv`.
- **FR-002**: System MUST produce `specs/022-pi-thesis-revisions/bids_vs_spreadsheet_diff.csv` enumerating every disagreement (child_id, clip_id, bids_timepoint, spreadsheet_timepoint, rationale_if_known).
- **FR-003**: System MUST regenerate `whisper-modeling/seen_child_splits/{master_with_split,train,val,test}.csv` with corrected `timepoint_norm`, preserving the seed-42 stratification.
- **FR-004**: System MUST regenerate every existing `*/test_metrics_by_timepoint.csv` from cached predictions (no model rerun required for this requirement).
- **FR-005**: System MUST update `CLAUDE.md` per-timepoint blocks (BabAR per-timepoint, within-child k-fold, 14m/36m stratified rows) with corrected numbers and record the diff in a short changelog inside this spec directory.

**Imbalance-aware metrics + group-stratified k-fold (US2)**

- **FR-006**: System MUST compute, for every existing per-system `test_predictions.csv`, the metric set `{f1_macro, f1_weighted, balanced_accuracy, auroc, auprc}` and write them to `evaluation/balanced_metrics_summary.csv`.
- **FR-007**: System MUST add `balanced_accuracy` and `f1_weighted` to `mil/mil_utils.py:compute_metrics()`'s return dict so all future evals report them by default.
- **FR-008**: System MUST produce an audit doc `evaluation/kfold_audit.md` that explicitly describes how `*_kfold3_f{0,1,2}/` splits children and clips, with code-path citations.
- **FR-009**: System MUST implement `evaluation/group_stratified_kfold.py` with default k=5, children disjoint per fold, positive-rate stratification within bootstrap noise, and minimum-positive-count guard per fold.
- **FR-010**: System MUST re-evaluate the top-band systems (Whisper pseudo-frame, Whisper-medium-MIL, Whisper-MIL, Whisper-MIL TS-MIL concat, BabAR/VTC-KCHI, USC-SAIL) under the group-stratified k-fold scheme and produce `evaluation/group_stratified_kfold_summary.csv` with `{mean, std, per_fold_values}` for AUROC and balanced accuracy.
- **FR-011**: System MUST implement a leave-one-child-out (LOOCV) sensitivity script `evaluation/loocv_subset.py` runnable on at most three top-band systems, producing `evaluation/loocv_subset_summary.csv` with per-child held-out AUROC.

**Audio-scene-analysis baseline expansion (US3)**

- **FR-012**: System MUST add a Qwen 3.5-Omni model slug to `baselines/audio_llm_baseline.py` and produce `baselines/audio_llm_baseline_runs/qwen35_omni_7b/{val,test}_metrics_tuned.json` on both the seen-child split and the universal-coverage split.
- **FR-013**: System MUST add `baselines/scene_analysis_baseline.py` supporting `--model {yamnet,ast}`; each model produces a child-speech probability per clip, val threshold is tuned, and `baselines/scene_analysis_runs/{yamnet,ast}/test_metrics_tuned.json` is written for both splits.
- **FR-014**: System MUST build a universal-coverage eval split that includes every SAILS clip with usable annotations regardless of timepoint balance, written to `whisper-modeling/all_children_splits/test_all.csv`.
- **FR-015**: System MUST run every zero-shot baseline (audio LLM, YAMNet, AST) on the universal-coverage split and report both seen-child and all-children rows in `evaluation/balanced_metrics_summary.csv`.
- **FR-016**: System MUST document the AudioSet class-to-score mapping used for YAMNet/AST in the corresponding baseline README.

**Encoder section restructure (US4)**

- **FR-017**: System MUST relocate encoder baseline code from `baselines/` to a new `encoders/` top-level module, preserving `git mv` history and providing one-cycle backward-compatible import shims.
- **FR-018**: System MUST produce a thesis-ready encoder pipeline figure (input waveform → frozen encoder → pooling → linear classifier; one panel per variant including fused).
- **FR-019**: System MUST author a fusion-of-encoders prose elaboration in the thesis chapter describing which encoders are fused, fusion stage, weighting, and head parameterisation.
- **FR-020**: System MUST produce `docs/per_model_training_data.csv` listing `{system_name, train_split, train_children, train_clip_count, includes_synthetic, synth_corpus_version, eval_split}` for every evaluated system, generated by a reproducible script that introspects saved configs.

**Per-timepoint posthoc analysis (US5)**

- **FR-021**: System MUST restructure the thesis chapter results section so that headline tables show only combined-timepoint metrics and per-timepoint breakdowns appear in a dedicated `### Posthoc: per-timepoint stratification` subsection.
- **FR-022**: System MUST flag in the posthoc section every system whose 14-month vs. 36-month AUROC delta exceeds 0.05, with a short interpretation.

### Key Entities

- **BIDS session**: a `sub-XXX/ses-YYY/` directory in the SAILS BIDS dataset. Authoritative source of (child_id, session_id) and, by extension, age band / timepoint.
- **Timepoint mapping**: a function from (child_id, clip_id) → {14_month, 36_month, unknown} derived from BIDS session IDs; replaces the spreadsheet `timepoint_norm` column.
- **Split**: a CSV of (child_id, clip_id, label) rows. Existing: `seen_child_splits/` (within-child, ≥5 clips/timepoint filter), `baselines/splits/` (cross-child, disjoint). New: `all_children_splits/test_all.csv` (universal coverage for zero-shot).
- **K-fold scheme**: today's within-child 3-fold (clips of the same children split across folds) is being replaced by group-stratified k-fold (children disjoint per fold, positive-rate balanced) with LOOCV as a subset sensitivity check.
- **Metric set**: today's `{f1, auroc, auprc}` is being extended to `{f1_macro, f1_weighted, balanced_accuracy, auroc, auprc}` with the trivial-floor baselines reported alongside.
- **Baseline system**: a named (system_id, predictions_csv, metrics_json) triple. New entries: `qwen35_omni_7b`, `yamnet`, `ast`. Restructured: encoder baselines moved from `baselines/` to `encoders/`.
- **Encoder pipeline**: waveform → frozen encoder (Whisper / WavLM / Fused) → pooling (mean / attention) → linear classifier; visualised in the new figure.
- **Posthoc analysis section**: a dedicated chapter subsection consolidating per-timepoint breakdowns for every system into one table.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Every per-timepoint metric in the thesis chapter and `CLAUDE.md` headline tables is computed from BIDS-derived timepoints; the BIDS-vs-spreadsheet disagreement CSV is published with row-level provenance.
- **SC-002**: Every reported headline metric set includes balanced accuracy and class-weighted F1 alongside macro F1, AUROC, and AUPRC; the trivial-floor baselines for balanced accuracy are reported in the same table.
- **SC-003**: The group-stratified 5-fold AUROC and balanced-accuracy mean ± std are published for the top-band systems; the LOOCV-subset sensitivity check is published for at least three top-band systems; any system whose group-stratified mean differs from its LOOCV mean by more than 0.03 is flagged.
- **SC-004**: At least three new rows (Qwen 3.5-Omni 0-shot, YAMNet, AST) appear in `evaluation/balanced_metrics_summary.csv`, each with both seen-child and all-children-coverage variants.
- **SC-005**: The thesis chapter's encoder section renders with the new pipeline figure embedded, the fusion-of-encoders prose elaboration in place, and a reader unfamiliar with the codebase can describe the fusion approach without reading code.
- **SC-006**: `docs/per_model_training_data.csv` exists with one row per evaluated system, reproducible from a single script.
- **SC-007**: The thesis chapter's headline results table shows only combined-timepoint metrics; per-timepoint breakdowns live in a single dedicated posthoc subsection.

## Assumptions

- BIDS session IDs encode visit number in a parseable convention (e.g., `ses-01`/`ses-02` or `ses-14mo`/`ses-36mo`); the parsing rule will be confirmed by reading the BIDS dataset README during US1 implementation and recorded in `bids_vs_spreadsheet_diff.csv`'s rationale column.
- YAMNet/AST exist as off-the-shelf checkpoints loadable inside the `child-vocalizations` env (or a sibling env if dependency conflicts arise); a child-speech class probability can be aggregated from one or more AudioSet ontology entries.
- Qwen 3.5-Omni is or will be available on HuggingFace at the time of implementation; if not, US3 is partially shippable with YAMNet + AST + Qwen 2.5 carryover and the Qwen 3.5 swap deferred.
- "Group-stratified k-fold" defaults to k=5 with children disjoint per fold and positive-rate balanced within bootstrap noise; if positive-rate balance is infeasible at k=5, the spec allows falling back to k=3 with the same disjoint-children guarantee.
- "Leave-one-child-out" sensitivity check is cost-controlled to at most three top-band systems on the seen-child split; full LOOCV across all systems is out of scope.
- "Move encoders out of baselines" is a `git mv` plus shim refactor that preserves existing run scripts unchanged for one release cycle; no semantic behaviour changes are introduced.
- "Per-timepoint posthoc" means headline tables show combined-timepoint metrics only and per-timepoint breakdowns are consolidated into a dedicated posthoc subsection; per-timepoint metrics are not removed from the codebase or from `test_metrics_by_timepoint.csv` files.
- Existing within-child k-fold numbers in `CLAUDE.md` are preserved and relabelled as `Within-child 3-fold (legacy)`; the new group-stratified numbers are published alongside, not in place of.
- Baseline expansion (US3) reuses the existing `unified.py` enrollment-style scoring or its zero-shot equivalent; no new evaluation harness is introduced.
- The spec is implementable on the current `021-post-thesis-future-work` branch without creating a new git branch; commits land on `021-post-thesis-future-work` until the user decides to switch.

## Dependencies

- US3 depends on US1 only to the extent that the all-children-coverage split (FR-014) should also use BIDS-derived timepoints if it reports per-timepoint breakdowns; US3 can otherwise be implemented in parallel with US1.
- US2 depends on US1 in that the group-stratified k-fold (FR-009) should use BIDS-derived timepoints if stratification is conditioned on timepoint; US2 can otherwise be implemented in parallel.
- US4 (encoder restructure) and US5 (per-timepoint posthoc) are independent of US1/US2/US3 and can land in any order.
- All US slices share `evaluation/balanced_metrics_summary.csv` as the single canonical reporting artefact; updates from any slice land in that file.
