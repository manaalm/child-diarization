# Tasks: Segment-Instance MIL with Attention Aggregation

**Input**: Design documents from `specs/004-segment-instance-mil/`
**Branch**: `004-segment-instance-mil`

## Format: `[ID] [P?] [Story?] Description`

- **[P]**: Can run in parallel (different files, no dependencies on incomplete tasks)
- **[Story]**: Which user story ([US1], [US2], [US3])
- Exact file paths included in all descriptions

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Create output directories and confirm prerequisites before any code is written.

- [x] T001 Create output directory stubs: `mil/mil_results/seg_mil/` and `mil/seg_embedding_cache/` (run `mkdir -p` and add `.gitkeep` files so directories are tracked)

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Segment embedding cache and dataset — both must exist before any aggregator or training task can run.

**⚠️ CRITICAL**: No user story work can begin until this phase is complete.

- [x] T002 Implement `SegmentEmbeddingCache` class in `mil/seg_embedding_cache.py` — disk-backed cache keyed on `MD5("{audio_path}|{start:.4f}|{end:.4f}")`, storing embeddings as `.npy` files under `mil/seg_embedding_cache/{frontend_name}/`; expose `get(audio_path, start, end) → np.ndarray | None` and `put(audio_path, start, end, embedding)` methods

- [x] T003 [P] Write `mil/configs/seg_mil_sweep.yaml` — list all 4 frontends (`usc_sail`, `pyannote`, `babar_vtc`, `vbx`) with their RTTM cache paths, 4 aggregators (`mean`, `max`, `attention`, `gated_attention`), encoder name (`wavlm-base-plus`), training hyperparameters (LR=1e-3, epochs=20, patience=5, batch_size=32), seed=42, output dir `mil/mil_results/seg_mil/`, and `min_seg_dur_sec` matching `pyannote/unified.py`

- [x] T004 Implement `SegmentBagDataset` in `mil/seg_dataset.py` — accepts a frontend name, RTTM cache directory, split DataFrame (`audio_path`, `child_id`, `timepoint_norm`, `label`), and a `SegmentEmbeddingCache`; loads RTTM for each clip, checks cache for each segment embedding (runs WavLM-base+ frozen forward pass + mean pool over segment frames on cache miss), zero-pads to a max-bag-size tensor `(K_max × D)` with a boolean mask `(K_max,)`; empty bags (K=0) return an all-zeros tensor with a zero mask; `__getitem__` returns `(bag_tensor, mask, label, metadata_dict)`

- [x] T005 Implement `precompute_embeddings(frontend, rttm_dir, df, cache, device)` standalone function in `mil/seg_dataset.py` — iterates all unique audio paths in `df`, loads the RTTM, and for each segment that is a cache miss runs the WavLM forward pass and writes the embedding to cache; prints progress every 100 clips; used by the `--precompute-only` flag in `seg_train.py`

**Checkpoint**: Embedding cache populates correctly for a single clip — `python mil/seg_dataset.py --smoke-test` runs without error.

---

## Phase 3: User Story 1 — Run the 16-Cell Experiment Matrix (Priority: P1) 🎯 MVP

**Goal**: Train and evaluate all 16 (frontend × aggregator) configurations, producing `mil/mil_results/seg_mil/all_configs.json` with test-split metrics for each cell.

**Independent Test**: Submit `mil/slurm/seg_mil_sweep.sh`; after the job completes, `mil/mil_results/seg_mil/all_configs.json` contains exactly 16 non-NaN entries. Delivers the complete comparison table before any interpretability tooling exists.

- [x] T006 [P] [US1] Implement `MeanAgg` and `MaxAgg` aggregator classes in `mil/seg_model.py` — each accepts `(bag: Tensor[K, D], mask: Tensor[K]) → (logit: Tensor[1], weights: None)`; `MeanAgg` computes masked mean; `MaxAgg` computes masked element-wise max (replace masked positions with -inf before max); both include a linear classification head `(D → 1)` for the final logit

- [x] T007 [P] [US1] Implement `AttnAgg` (standard ABMIL, Ilse et al. 2018) in `mil/seg_model.py` — attention score `a_k = softmax(w^T · tanh(V · h_k))` with learnable `V ∈ R^{attn_dim × D}` and `w ∈ R^{attn_dim}`; masked softmax (zero out masked instances before softmax); returns `(logit, attention_weights: Tensor[K])`; classification head on weighted sum

- [x] T008 [US1] Implement `GatedAttnAgg` in `mil/seg_model.py` — wraps the existing `GatedABMILHead` from `mil/mil_model.py`; exposes the same `(bag, mask) → (logit, weights)` interface; adds a linear head on the GatedABMIL output; import `GatedABMILHead` directly rather than re-implementing

- [x] T009 [US1] Implement `train_one_config(config, train_bags, val_bags, test_bags, device) → metrics_dict` in `mil/seg_train.py` — instantiates the correct aggregator head from `config["aggregator"]`, trains with `BCEWithLogitsLoss` and Adam, implements early stopping on val AUROC with `patience` from config, calls `mil_utils.compute_metrics()` and `mil_utils.tune_threshold()` for evaluation, returns val and test metrics dictionaries

- [x] T010 [US1] Implement 16-config sweep loop in `mil/seg_train.py` `main()` — reads `seg_mil_sweep.yaml`, iterates all (frontend, aggregator) pairs, skips configs whose `mil/mil_results/seg_mil/{frontend}_{aggregator}/test_metrics.json` already exists (resume support), calls `precompute_embeddings()` once per frontend before training its 4 aggregators

- [x] T011 [US1] Add `--precompute-only` CLI flag to `mil/seg_train.py` — when set, runs `precompute_embeddings()` for all 4 frontends and exits without training; add `--config` flag for YAML path; use `argparse`

- [x] T012 [US1] Implement per-config results writing in `mil/seg_train.py` — for each completed config write to `mil/mil_results/seg_mil/{frontend}_{aggregator}/`: `config.json` (full config dict), `val_predictions.csv` and `test_predictions.csv` (columns: `audio_path`, `child_id`, `timepoint_norm`, `label`, `prob`, `pred`, `n_instances`, `top_seg_start`, `top_seg_end`, `top_seg_weight`), `val_metrics.json` and `test_metrics.json` (F1, precision, recall, AUROC, AUPRC, threshold)

- [x] T013 [US1] Implement `write_all_configs_summary()` in `mil/seg_train.py` — after all 16 configs complete, reads each `test_metrics.json` and `val_metrics.json`, assembles a list of 16 `ConfigSummaryEntry` dicts (fields per data-model.md), writes to `mil/mil_results/seg_mil/all_configs.json`; call this after every completed config so the summary is updated incrementally

- [x] T014 [US1] Create `mil/slurm/seg_mil_sweep.sh` — SLURM script with `#SBATCH -t 24:00:00`, `--gres=gpu:1`, `-p ou_bcs_normal,pi_satra`, `--mem=40G`, `export PYTHONUNBUFFERED=1`, conda activate `child-vocalizations`, preflight check that all 4 RTTM cache dirs are non-empty, `cd mil && python seg_train.py --config configs/seg_mil_sweep.yaml`; log to `logs/seg_mil_%j.out`

**Checkpoint**: `python mil/seg_train.py --config mil/configs/seg_mil_sweep.yaml` completes for at least one (frontend, aggregator) pair; `mil/mil_results/seg_mil/all_configs.json` contains a valid entry for that pair.

---

## Phase 4: User Story 2 — Segment Attention Weights for Interpretability (Priority: P2)

**Goal**: Predictions CSVs include per-segment attention weight data (segment timestamps + weight), enabling researchers to identify which segments drove each prediction.

**Independent Test**: Read `mil/mil_results/seg_mil/vbx_gated_attention/test_predictions.csv`; confirm `top_seg_start`, `top_seg_end`, `top_seg_weight` columns are populated for positive-prediction clips; read `test_segment_weights.csv` and confirm one row per (clip, segment) with weights summing to 1.0 per clip.

- [x] T015 [US2] Add per-segment attention weight file output in `mil/seg_train.py` — for `attention` and `gated_attention` configs only, write `val_segment_weights.csv` and `test_segment_weights.csv` alongside the per-clip CSVs; columns: `audio_path`, `child_id`, `seg_start`, `seg_end`, `attention_weight`; one row per (clip, segment); skip silently for `mean` and `max` configs (which have no meaningful per-segment weights)

- [x] T016 [US2] Save per-timepoint metrics breakdown in `mil/seg_train.py` — after evaluation on val and test splits, call `mil_utils.per_timepoint_metrics()` and write output to `val_metrics_by_timepoint.json` and `test_metrics_by_timepoint.json` per config, mirroring the format used by `pyannote/` enrollment runs

- [x] T017 [US2] Verify `top_seg_start`, `top_seg_end`, `top_seg_weight` columns in `test_predictions.csv` are computed correctly in `mil/seg_train.py` — for attention/gated configs, extract the argmax attention weight and its corresponding segment timestamps; for mean/max configs, set these columns to `None`; add an assertion that weights sum to ~1.0 per clip (within floating-point tolerance)

**Checkpoint**: For a completed `vbx_gated_attention` config, `test_segment_weights.csv` exists, has >0 rows, and weights for each `audio_path` sum to 1.0.

---

## Phase 5: User Story 3 — Unified Thesis Table (Priority: P3)

**Goal**: `all_configs.json` is wired into `evaluation/configs/thesis_tables.yaml` so the thesis table generation script produces a side-by-side comparison of all 16 MIL configurations and the existing ECAPA enrollment baselines.

**Independent Test**: Run `python evaluation/build_thesis_tables.py`; output includes a `table_segment_mil` table with 16 rows, same column schema (`diarizer`, `aggregator`, `f1`, `precision`, `recall`, `auroc`, `auprc`) as the ECAPA enrollment table rows.

- [x] T018 [US3] Add `table_segment_mil` entry to `evaluation/configs/thesis_tables.yaml` — source file `mil/mil_results/seg_mil/all_configs.json`; `key_map` mapping JSON fields (`frontend`, `aggregator`, `test_f1`, `test_precision`, `test_recall`, `test_auroc`, `test_auprc`) to thesis table column names; one entry per configuration row

- [x] T019 [P] [US3] Update `CLAUDE.md` — add `mil/mil_results/seg_mil/` to the Results Storage table, add "Segment-instance MIL sweep" to Key Commands with the correct `sbatch` invocation, add `seg_*.py` module descriptions to the Architecture section under `mil/`

- [x] T020 [US3] Run `python evaluation/build_thesis_tables.py` (or equivalent table generation script) with the new `thesis_tables.yaml` entry pointing at a real `all_configs.json`; confirm the `table_segment_mil` table generates without errors and all 16 rows are populated

**Checkpoint**: Thesis table script runs end-to-end without KeyError or missing-file errors; MIL rows appear alongside ECAPA enrollment rows.

---

## Phase 6: Polish & Cross-Cutting Concerns

**Purpose**: Documentation, code quality, and quickstart validation.

- [x] T021 [P] Add module-level docstrings to `mil/seg_embedding_cache.py`, `mil/seg_dataset.py`, `mil/seg_model.py`, and `mil/seg_train.py` — each docstring must state: purpose, inputs, outputs, and any caching/side effects, per constitution Principle VII

- [x] T022 Validate `specs/004-segment-instance-mil/quickstart.md` against the actual implementation — run each command in the quickstart, update any paths or flag names that changed during implementation, confirm the troubleshooting table covers actual failure modes encountered

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies — start immediately
- **Foundational (Phase 2)**: Depends on Phase 1 — **BLOCKS all user stories**
  - T002 and T004/T005 are sequential (T004 imports T002)
  - T003 is independent (config file only)
- **US1 (Phase 3)**: Depends on Phase 2 completion
  - T006, T007 can run in parallel (different aggregator classes)
  - T008 depends on confirming GatedABMILHead interface (read mil/mil_model.py first)
  - T009 depends on T006/T007/T008 (needs all four aggregators)
  - T010/T011/T012/T013 depend on T009 (training loop must exist)
  - T014 (SLURM script) is independent within Phase 3
- **US2 (Phase 4)**: Depends on T012 (predictions CSV format must exist first)
- **US3 (Phase 5)**: Depends on T013 (`all_configs.json` schema must be finalized)
- **Polish (Phase 6)**: Depends on all user story phases being complete

### User Story Dependencies

- **US1 (P1)**: Starts after Phase 2. No dependency on US2 or US3. MVP scope.
- **US2 (P2)**: Starts after US1's T012 establishes the predictions CSV format. Adds columns/files to existing output.
- **US3 (P3)**: Starts after US1's T013 finalizes `all_configs.json` schema.

### Within Phase 3 (US1)

```
T006 [parallel] ─┐
T007 [parallel] ─┤→ T009 → T010 → T012 → T013
T008            ─┘          ↑
T003 (config)  ─────────────┘
T011 ─ independent (CLI wiring)
T014 ─ independent (SLURM script)
```

---

## Parallel Examples

### Phase 2 (Foundational)
```
# These can run in parallel:
Task T002: mil/seg_embedding_cache.py
Task T003: mil/configs/seg_mil_sweep.yaml
```

### Phase 3, Part A (Aggregator heads — all different files/classes)
```
# These can run in parallel:
Task T006: MeanAgg + MaxAgg in mil/seg_model.py
Task T007: AttnAgg in mil/seg_model.py  ← different class, same file; coordinate on file structure first
Task T014: mil/slurm/seg_mil_sweep.sh
```

### Phase 5 (US3)
```
# These can run in parallel:
Task T018: evaluation/configs/thesis_tables.yaml
Task T019: CLAUDE.md
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup (T001)
2. Complete Phase 2: Foundational (T002–T005)
3. Complete Phase 3: User Story 1 (T006–T014)
4. **STOP and VALIDATE**: Submit `mil/slurm/seg_mil_sweep.sh`; confirm `all_configs.json` has 16 entries
5. **MVP delivered**: Full 16-cell comparison table available for thesis

### Incremental Delivery

1. Setup + Foundational → Cache + dataset ready
2. **US1** → 16-cell matrix + `all_configs.json` (MVP)
3. **US2** → Attention-weight CSVs for interpretability analysis
4. **US3** → Thesis table auto-generation wired up
5. Polish → Docstrings, quickstart validation

---

## Notes

- No tests are specified in the feature spec — test tasks are omitted per the task generation rules
- T006 and T007 touch the same file (`seg_model.py`) — write the class skeleton for the file in T006, then add `AttnAgg` in T007 without conflict
- The `GatedABMILHead` in `mil/mil_model.py` already exists — T008 is a wrapper, not a reimplementation; read `mil/mil_model.py` before starting T008
- All results paths use the pattern `{frontend}_{aggregator}` (e.g., `vbx_gated_attention`) — keep consistent across all tasks
- Constitution requires seed=42 in every config — enforce in `seg_mil_sweep.yaml` (T003) and assert in training loop (T009)
