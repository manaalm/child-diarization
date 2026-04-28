# Tasks: Metadata-Conditioned Routing and Ensemble Extensions

**Input**: Design documents from `specs/012-metadata-routing-ensemble/`
**Prerequisites**: plan.md ✅, spec.md ✅, research.md ✅, data-model.md ✅

**Organization**: Tasks grouped by user story (US1=Stacker, US2=Router, US3=MultiChild Suppressor, US4=ShortVoc Head). US1 and US2 are CPU-only and can run back-to-back in minutes. US3 and US4 require GPU SLURM jobs.

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Shared data loading utilities used by all sub-features

- [X] T001 Create `evaluation/metadata_router.py` with shared helpers: `load_system_scores()` loads all 10 system CSVs and joins on `audio_path`, normalising "score"→"prob" for MIL systems; `load_metadata()` reads seen-child `master_with_split.csv` and parses `#_adults`/`#_children` to ints with NaN→0/1 fallback; `load_split_labels()` returns val/test DataFrames with label; write `save_results(out_dir, metrics, predictions, config)` that saves `test_metrics_tuned.json` (with `baseline_f1`, `baseline_auroc`, `delta_f1`, `delta_auroc` fields), `val_metrics_tuned.json`, `test_predictions.csv`, and `config.json` per the data-model contract in `data-model.md`
- [X] T002 Add `compute_metrics_full(y_true, y_prob, threshold)` and `tune_threshold(y_true, y_prob)` to `evaluation/metadata_router.py` (can import from `mil/mil_utils.py`); add `assert_no_test_leakage(split_col)` that raises if any threshold-tuning call receives test-split rows; add baseline constants `BASELINE_F1=0.893`, `BASELINE_AUROC=0.878`

---

## Phase 2: Foundational (Blocking Prerequisite)

**Purpose**: Verify all 10 system predictions are loadable and joinable before writing any sub-feature code

- [X] T003 Add a `--verify` CLI mode to `evaluation/metadata_router.py`: load all 10 system CSVs, join on `audio_path`, report row counts per split (expect val=431, test=441), flag any system with <441 test clips, print metadata column presence check — confirm Context/`#_adults`/`#_children`/`Interaction_with_child`/`Location` all present; run and confirm output is clean before proceeding

**Checkpoint**: Foundation ready when `python evaluation/metadata_router.py --verify` exits 0

---

## Phase 3: User Story 1 — Metadata-Augmented Stacker (Priority: P1) 🎯 MVP

**Goal**: Train LR and GBM stackers on val using 10 system scores + 7 metadata features; evaluate on test; save feature importances showing whether metadata added signal over scores alone.

**Independent Test**: `python evaluation/metadata_router.py --mode stack` exits 0 and writes `ensemble_runs/metadata_stack/test_metrics_tuned.json` with valid F1/AUROC values and `feature_importances.json` with non-null entries for both model variants.

### Implementation for User Story 1

- [X] T004 [US1] In `evaluation/metadata_router.py`, implement `build_feature_matrix(scores_df, meta_df, split)` that returns a DataFrame with columns: `babar_prob`, `vtc_prob`, `vtc_kchi_prob`, `vbx_prob`, `usc_sail_prob`, `pyannote_prob`, `eend_eda_prob`, `sortformer_prob`, `wavlm_mil_prob`, `whisper_mil_prob`, `n_adults_int`, `n_children_int`, `n_adults_ge2`, `n_children_ge2`, `context_unknown`, `has_interaction`, `timepoint_is_36m`; audio_llm rows imputed with 0.5 if missing; assert no NaN in output
- [X] T005 [US1] Implement `run_metadata_stack(val_df, test_df, feature_cols, out_dir, seed=42)` in `evaluation/metadata_router.py`: train `LogisticRegression(C=1.0, max_iter=500, random_state=seed)` and `HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, random_state=seed)` on val features+labels; tune threshold on val per-model; evaluate on test; pick best model by val F1; save `feature_importances.json` with LR coefficients and GBM feature_importances_ as dicts keyed by feature name; call `save_results()` to write outputs to `ensemble_runs/metadata_stack/`
- [X] T006 [US1] Wire `--mode stack` CLI arg in `evaluation/metadata_router.py` `main()` that calls `run_metadata_stack()`; create `ensemble_runs/metadata_stack/` directory; run the stacker and confirm `test_metrics_tuned.json` and `feature_importances.json` are written with correct schema

**Checkpoint**: `ensemble_runs/metadata_stack/test_metrics_tuned.json` exists with non-null F1/AUROC and `delta_f1` vs. baseline

---

## Phase 4: User Story 2 — Metadata-Conditioned Router (Priority: P1)

**Goal**: Two router variants (rule-based and learned) that select per-clip which system score to use based on BIDS metadata; compare both against best_audio_mil mean baseline.

**Independent Test**: `python evaluation/metadata_router.py --mode router` exits 0 and writes `ensemble_runs/metadata_router_rule/test_metrics_tuned.json` and `ensemble_runs/metadata_router_learned/test_metrics_tuned.json`, each with `routed_system` breakdown counts confirming all 5 rule branches are exercised.

### Implementation for User Story 2

- [X] T007 [US2] Implement `apply_rule_router(clip_row, scores_row)` in `evaluation/metadata_router.py` using the priority-ordered rules from `research.md`: (1) context contains "unknown" → sortformer_prob; (2) n_adults_int ≥ 2 → mean(wavlm_mil_prob, eend_eda_prob); (3) n_children_int ≥ 2 → whisper_mil_prob; (4) n_children_int == 1 → vtc_prob; (5) default → mean(babar_prob, vtc_prob, wavlm_mil_prob, vbx_prob); return (score, rule_name) tuple
- [X] T008 [US2] Implement `run_rule_router(val_df, test_df, out_dir)` in `evaluation/metadata_router.py`: apply `apply_rule_router` to each row; tune threshold on val; evaluate on test; include `routed_system` value_counts in config.json so rule coverage can be inspected; call `save_results()` writing to `ensemble_runs/metadata_router_rule/`
- [X] T009 [US2] Implement `run_learned_router(val_df, test_df, out_dir, seed=42)` in `evaluation/metadata_router.py`: train a `LogisticRegression` on val metadata features ONLY (no system scores as input — predicts which single system minimises error per clip using val ground truth); at inference, use the predicted system's probability; tune threshold on val; evaluate on test; write to `ensemble_runs/metadata_router_learned/`; note: if learned router degenerates to always picking one system, document this in config.json as `degenerate: true`
- [X] T010 [US2] Wire `--mode router` CLI arg that calls both `run_rule_router()` and `run_learned_router()`; create output directories; run and confirm both `test_metrics_tuned.json` files written; print side-by-side comparison table of rule-router, learned-router, and best_audio_mil mean baseline

**Checkpoint**: Both router result directories exist; rule breakdown shows all 5 cases used

---

## Phase 5: User Story 3 — Multi-Child FP Suppressor (Priority: P2)

**Goal**: Train a lightweight clip-level classifier on `n_children≥2` train clips using WavLM mean-pool embeddings; apply only to multi-child test clips; document per-stratum improvement and any regression on single-child clips.

**Independent Test**: `python evaluation/multi_child_suppressor.py` exits 0 and writes `mil/mil_results/multi_child_suppressor/test_metrics_multi_child_only.json` (FP count before vs. after) and `test_metrics_single_child_only.json` (guard against regression).

### Implementation for User Story 3

- [X] T011 [US3] Create `evaluation/multi_child_suppressor.py`; implement `embed_clip(audio_path, model, device)` that loads a clip, runs frozen WavLM-Base+ through `mil.mil_model.BackboneExtractor`, and returns mean-pooled embedding (D=768); implement `build_embedding_cache(df, model, device, cache_path)` that caches embeddings to `mil/mil_results/multi_child_suppressor/emb_cache.npy` (keyed by audio_path hash) so re-runs are fast
- [X] T012 [US3] Implement `train_suppressor(train_df, meta_df, model, device, seed=42)` in `evaluation/multi_child_suppressor.py`: filter train_df to `n_children_int ≥ 2`; embed all clips (or load cache); train `LogisticRegression(C=0.1, max_iter=500)` on embeddings; tune alpha (merge weight) on val `n_children≥2` subset; return `(clf, alpha, val_metrics_stratum)`
- [X] T013 [US3] Implement `apply_suppressor(test_df, meta_df, clf, alpha, main_scores, device)` in `evaluation/multi_child_suppressor.py`: for `n_children≥2` clips, compute `final_score = alpha * main_score + (1-alpha) * suppressor_score`; for `n_children<2` clips, pass through main_score unchanged; compute overall metrics + `test_metrics_multi_child_only.json` + `test_metrics_single_child_only.json`; main_score = best_audio_mil mean from `ensemble_runs/test_predictions.csv`
- [X] T014 [US3] Wire `main()` in `evaluation/multi_child_suppressor.py` with `mkdir -p mil/mil_results/multi_child_suppressor`; call train then apply; call `save_results()` writing all metric files and `test_predictions.csv` to `mil/mil_results/multi_child_suppressor/`; add `--dry-run` flag that prints n_children≥2 stratum size and exits before training
- [X] T015 [US3] Create `evaluation/slurm/run_multi_child_suppressor.sh`: SLURM job (partition `ou_bcs_normal,pi_satra`, 1 GPU, 16GB RAM, 1h, logs to `logs/evaluation/suppressor_%j.out`); activates `child-vocalizations` conda env; sets `HF_HOME`; runs `python evaluation/multi_child_suppressor.py` from repo root

**Checkpoint**: `mil/mil_results/multi_child_suppressor/test_metrics_multi_child_only.json` exists with `before_f1` and `after_f1` entries

---

## Phase 6: User Story 4 — Short-Vocalization Specialized Head (Priority: P3)

**Goal**: Train a fine-grained WavLM head with 500ms/250ms-hop windows on clips containing <0.5s CHI vocalizations; merge with main pipeline via val-tuned beta; evaluate recovery rate on the 44 hard FN clips.

**Independent Test**: `python evaluation/short_voc_head.py` exits 0 and writes `mil/mil_results/short_voc_head/test_metrics_short_voc_clips.json` with `n_recovered` (clips correct after merge that were wrong before) ≥ 0.

### Implementation for User Story 4

- [X] T016 [US4] Create `evaluation/short_voc_head.py`; implement `identify_short_voc_clips(split_df, rttm_dir, threshold_sec=0.5)` that reads ground-truth RTTMs from `whisper-modeling/usc_sail_rttm_cache/` (or `pyannote/vtc_rttm_cache/` as fallback), extracts CHI segment durations per clip, and flags clips where any CHI segment < threshold_sec; return a boolean mask aligned to split_df; log fraction of positive clips that are short-voc
- [X] T017 [US4] Implement `ShortVocHead` class in `evaluation/short_voc_head.py`: frozen WavLM-Base+ backbone (reuse `mil.mil_model.BackboneExtractor`); extract frame-level features at native 20ms resolution; apply 1D-CNN with kernel_size=25 (=500ms) and stride=12 (=240ms) over feature sequence; global max-pool → linear(768→1); train with BCEWithLogitsLoss; freeze backbone, train head only; `window_ms=500`, `hop_ms=250` documented in config
- [X] T018 [US4] Implement `train_short_head(train_df, short_voc_mask_train, model, device, seed=42, epochs=15, patience=5)` in `evaluation/short_voc_head.py`: use all train positive clips (not just short-voc) as positives; use hard negatives from `synth_results/manifests/hard_negatives_manifest.csv` as negatives (cap=344 to match original count); early stopping on val short-voc F1; save checkpoint to `mil/mil_results/short_voc_head/best_checkpoint.pt`
- [X] T019 [US4] Implement `merge_and_evaluate(test_df, main_scores, head_scores, val_df, val_main_scores, val_head_scores, out_dir)` in `evaluation/short_voc_head.py`: tune beta on val (sweep 0.0–1.0 in 0.05 steps, maximise val F1); `final_score = beta * main_score + (1-beta) * head_score`; compute `test_metrics_short_voc_clips.json` (clips where short_voc_mask_test=True) with keys: `before_f1`, `after_f1`, `n_recovered` (previously-wrong clips now correct), `n_hurt` (previously-correct clips now wrong); compute `test_metrics_non_short_voc_clips.json` for FP-harm guard; call `save_results()` for overall metrics
- [X] T020 [US4] Create `evaluation/slurm/run_short_voc_head.sh`: SLURM job (partition `ou_bcs_normal,pi_satra`, 1 GPU, 32GB RAM, 4h, logs to `logs/evaluation/short_voc_%j.out`); activates `child-vocalizations` conda env; sets `HF_HOME`; runs `python evaluation/short_voc_head.py` from repo root

**Checkpoint**: `mil/mil_results/short_voc_head/test_metrics_short_voc_clips.json` exists with `n_recovered` ≥ 0 and `n_hurt` documented

---

## Phase 7: Polish & Cross-Cutting Concerns

- [X] T021 [P] Update `CLAUDE.md`: add `evaluation/metadata_router.py` (`--mode stack`, `--mode router`, `--verify`), `evaluation/multi_child_suppressor.py`, and `evaluation/short_voc_head.py` to the Key Commands section; add result paths `ensemble_runs/metadata_stack/`, `ensemble_runs/metadata_router_{rule,learned}/`, `mil/mil_results/{multi_child_suppressor,short_voc_head}/` to the Results Storage section; add `evaluation/slurm/` SLURM scripts
- [X] T022 [P] Update `results_summary.md` with sub-feature A/B results table (router and stacker vs. best_audio_mil mean baseline) immediately after T010 completes; update with C/D results after SLURM jobs finish; include `delta_f1` and `delta_auroc` columns
- [X] T023 Add `mkdir -p logs/evaluation` guard to both SLURM scripts and confirm partition/mem matches other working SLURM scripts in the repo (cross-check against `mil/slurm/train_mil_hardneg.sh`)
- [ ] T024 Check short_voc_head job (12774459) once complete: verify `mil/mil_results/short_voc_head/test_metrics_short_voc_clips.json` exists with `n_recovered`/`n_hurt` entries; add results to §11b and §12 of `results_summary.md`; add finding to §13; if failed, diagnose and resubmit
- [ ] T025 Check baseline_seen_child job (12770942) once complete: collect all 13 seen-child encoder results from `baselines/baseline_results_seen_child/`; add a new §4b table to `results_summary.md` comparing cross-child vs. seen-child encoder performance; update §12 summary table
- [ ] T026 **[CRITICAL — age-band prototype fix]** Fixed critical bug: `build_child_prototypes` was grouping only by `child_id`, pooling 14-month and 36-month voice recordings into one prototype. Fixed in `unified.py`, `usc_sail_run_enrollment.py`, `babar_three.py`, `babar_updated.py`. Resubmitted all enrollment jobs (BabAR 12775522, VTC+VBx 12775523, Pyannote 12775524, EEND-EDA 12775525, Sortformer 12775526, USC-SAIL 12775535, TalkNet-ASD 12775586). Once jobs complete: (a) ADD new "age-band prototype" rows alongside existing pooled-prototype rows in §5–§9 results tables in `results_summary.md` — do NOT overwrite; old pooled results are still valid and informative; (b) rerun `evaluation/metadata_router.py --mode stack` and `--mode router` since they consume enrollment-based prob columns; ADD new stacker/router rows to §11b; (c) add new rows to §12 summary table for age-band-specific variants; (d) rerun babar combined models (`python pyannote/babar_three.py`) and ADD results to §10; (e) add finding to §13 noting the delta between pooled vs. age-band-specific prototypes
- [ ] T027 Check synth_scene_gen job (12770080) once complete: verify `synth_results/synthetic_scenes/` has expected WAV+RTTM+JSON files and `synth_results/manifests/synthetic_manifest.csv` exists; add scene count and distribution summary to `results_summary.md` §7; if failed, diagnose and resubmit `synth/slurm/run_scene_generation.sh`

---

## Dependencies & Execution Order

### Phase Dependencies

- **Phase 1 (Setup)**: No dependencies — start immediately
- **Phase 2 (Foundation)**: Depends on Phase 1; BLOCKS all user stories
- **Phase 3 (US1 Stacker)**: Depends on Phase 2; CPU-only, ~1 min
- **Phase 4 (US2 Router)**: Depends on Phase 2; CPU-only, ~1 min; can run in parallel with Phase 3
- **Phase 5 (US3 Suppressor)**: Depends on Phase 2; requires GPU (~30 min SLURM); can start after Phase 2 regardless of US1/US2
- **Phase 6 (US4 ShortVoc)**: Depends on Phase 2; requires GPU (~2–4h SLURM); can start after Phase 2 regardless of other stories
- **Phase 7 (Polish)**: T021/T022 depend on respective story completion; T023 can run anytime

### Parallel Opportunities

```bash
# After Phase 2 completes, launch all four in parallel:
python evaluation/metadata_router.py --mode stack          # US1, ~1 min
python evaluation/metadata_router.py --mode router         # US2, ~1 min
sbatch evaluation/slurm/run_multi_child_suppressor.sh      # US3, ~30 min
sbatch evaluation/slurm/run_short_voc_head.sh              # US4, ~2-4h
```

---

## Implementation Strategy

### MVP First (User Story 1 + 2 Only)

1. Complete Phase 1 + 2 (T001–T003)
2. Complete Phase 3 (T004–T006) — stacker done
3. Complete Phase 4 (T007–T010) — router done
4. **STOP and VALIDATE**: compare both against best_audio_mil mean baseline
5. If stacker/router show improvement → proceed to US3/US4
6. If null result → document and stop; US3/US4 optional

### Key Implementation Notes

- All threshold tuning must use val split only; `assert_no_test_leakage()` called before every `tune_threshold()` invocation
- `evaluation/metadata_router.py` is the single file for US1 + US2; keep it under 400 lines
- US3 and US4 reuse `mil.mil_model.BackboneExtractor` — import from existing module, do not copy
- RTTM reading in US4 should use a try/except per file (some clips may not have cached RTTMs)
