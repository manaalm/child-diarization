# Tasks: MIL Extensions — Aggregation Ablations, Transformer MIL, and Weak Diarization

**Input**: Design documents from `specs/005-mil-extensions/`
**Branch**: `005-mil-extensions`

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (US1–US4)

---

## Phase 1: Setup

**Purpose**: Verify prerequisites from feature 004 are in place before any new code is written.

- [x] T001 Verify baseline exists: check that `mil/mil_results/seg_mil/all_configs.json` has 16 entries and `mil/seg_embedding_cache/` is populated for all 4 frontends (usc_sail, pyannote, babar_vtc, vbx)

---

## Phase 2: Foundational (Blocking Prerequisite)

**Purpose**: Confirm the gated_attention aggregator from feature 004 already produces valid results — no re-implementation needed, just verification.

**⚠️ CRITICAL**: US1 comparison data must be confirmed present before US2–US4 can build on it.

- [x] T002 Verify `GatedAttnAgg` is registered in `build_aggregator()` in `mil/seg_model.py` and that `mil/mil_results/seg_mil/all_configs.json` contains rows for both `attention` and `gated_attention` aggregators for all 4 frontends (16 entries expected from feature 004)

**Checkpoint**: Foundation ready — US1 analysis confirmed, US2–US4 can proceed.

---

## Phase 3: User Story 1 — Gated Attention Ablation (Priority: P1) 🎯 MVP

**Goal**: Confirm the gated-vs-plain attention comparison is readable from existing `all_configs.json`. No new code required — `GatedAttnAgg` was implemented and run in feature 004.

**Independent Test**: Read `all_configs.json`, confirm AUROC values differ between `attention` and `gated_attention` rows for the same frontend.

- [x] T003 [US1] Inspect existing results and document the attention vs. gated_attention AUROC delta per frontend in `mil/mil_results/seg_mil/all_configs.json` — if any `gated_attention` rows are missing, add `gated_attention` to the aggregators list in `mil/configs/seg_mil_sweep.yaml` and re-run the sweep for only the missing configs

**Checkpoint**: US1 complete — gated vs. plain attention comparison is confirmed and readable.

---

## Phase 4: User Story 2 — Age-Band Aggregation Ablation (Priority: P2)

**Goal**: Implement noisy-OR and top-k aggregators, add age-band stratified inference to the training loop, and produce per-age-band metrics for all frontend × aggregator combinations.

**Independent Test**: After running the extended sweep, `mil/mil_results/seg_mil/all_configs.json` entries have `test_auroc_14month` and `test_auroc_36month` fields, and `test_metrics_by_timepoint.csv` exists in each config directory.

- [x] T004 [P] [US2] Implement `NoisyORAgg` class in `mil/seg_model.py` — per-instance linear head → sigmoid → log-space product over valid (non-masked) instances: `log_bag_complement = sum(log(1 - sigma(logit_k)))` for k in mask; `bag_logit = logaddexp(0, log_bag_complement)`; mask padding with neutral log(1.0)=0; returns `(logit, None)`
- [x] T005 [P] [US2] Implement `TopKAgg` class in `mil/seg_model.py` — score each instance with a separate linear `score_head`, mask padding with `-inf`, compute `k_actual = min(self.k, mask.sum().clamp(1))`, select top-k indices, mean-pool `bag[topk_idx]`, apply final `head` linear; default `k=3`; returns `(logit, None)`
- [x] T006 [US2] Register `NoisyORAgg` and `TopKAgg` in `build_aggregator()` in `mil/seg_model.py` under keys `"noisy_or"` and `"top_k"` respectively; `TopKAgg` reads `k` from `attn_dim` param (repurpose) or add a `k` kwarg to `build_aggregator()`
- [x] T007 [US2] Add `noisy_or` and `top_k` to the `aggregators` list in `mil/configs/seg_mil_sweep.yaml`; add a `top_k: 3` field under a new `aggregator_config:` block
- [x] T008 [US2] Add age-band stratified inference to `train_one_config()` in `mil/seg_train.py`: after computing `test_pred_df`, load `whisper-modeling/seen_child_splits/test.csv`, join on audio path to get `timepoint` column, compute per-band metrics for each of `["14_month", "36_month"]` using the val-tuned threshold, write `test_metrics_by_timepoint.csv` in the config output directory
- [x] T009 [US2] Extend `write_all_configs_summary()` in `mil/seg_train.py` to read `test_metrics_by_timepoint.csv` from each config directory and populate `test_auroc_14month`, `test_auroc_36month`, `test_f1_14month`, `test_f1_36month` fields in each `all_configs.json` entry (use `None` for configs without the file for backward compatibility)
- [x] T010 [US2] Run the extended sweep to generate noisy_or and top_k configs: `sbatch mil/slurm/seg_mil_sweep.sh` — resume-safe, existing 16 configs will be skipped
- [x] T011 [US2] Verify extended sweep output: confirm `all_configs.json` has ≥24 entries (16 baseline + 4 noisy_or + 4 top_k) with `test_auroc_14month` and `test_auroc_36month` populated; confirm `test_metrics_by_timepoint.csv` exists in all config directories

**Checkpoint**: US2 complete — age-band metrics available for all trained models, noisy-OR and top-k results visible.

---

## Phase 5: User Story 3 — Transformer MIL (Priority: P3)

**Goal**: Implement a small transformer aggregator (2-layer, 4-head, CLS token, learned PE) and add it to the sweep.

**Independent Test**: After running transformer configs, `all_configs.json` has 4 rows tagged `aggregator: transformer`, test AUROC values are non-NaN, and `config.json` in each transformer config directory includes `transformer_num_layers`, `transformer_num_heads`, `transformer_ffn_dim`, `transformer_dropout`.

- [x] T012 [P] [US3] Implement `TransformerAgg` class in `mil/seg_model.py`:
  - Prepend a learned CLS embedding (parameter `self.cls_token: nn.Parameter(torch.zeros(1, embed_dim))`)
  - Add learned positional embeddings for positions 0…K_max (parameter `self.pos_embed: nn.Embedding(K_max+1, embed_dim)`)
  - Sort bag by segment start time before PE (pass segment start times as extra input, or handle in forward via meta)
  - Stack `num_layers` of `nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=ffn_dim, dropout=dropout, batch_first=True, norm_first=True)` (pre-norm)
  - After encoder, take CLS output (position 0) → linear head → logit
  - For attention weights: average the last layer's self-attention from CLS row across heads, scatter to K_max positions; returns `(logit, weights[K_max])`
  - Constructor args: `embed_dim`, `num_layers=2`, `num_heads=4`, `ffn_dim=1536`, `dropout=0.3`, `k_max=64`
- [x] T013 [US3] Register `TransformerAgg` in `build_aggregator()` in `mil/seg_model.py` under key `"transformer"`; pass `num_layers`, `num_heads`, `ffn_dim`, `dropout` from a `transformer_config` dict in the call site
- [x] T014 [US3] Add `transformer_config` block to `mil/configs/seg_mil_sweep.yaml`:
  ```yaml
  transformer_config:
    num_layers: 2
    num_heads: 4
    ffn_dim: 1536
    dropout: 0.3
    weight_decay: 0.01
  ```
  and add `transformer` to the `aggregators` list
- [x] T015 [US3] Pass `transformer_config` from sweep config into `train_one_config()` and into `build_aggregator()` call in `mil/seg_train.py`; set `weight_decay` in the optimizer when aggregator is `transformer` (override the default optimizer weight_decay)
- [x] T016 [US3] Log transformer HPs in `config.json` output in `mil/seg_train.py`: when aggregator is `transformer`, add `transformer_num_layers`, `transformer_num_heads`, `transformer_ffn_dim`, `transformer_dropout`, `transformer_weight_decay` to the written config dict
- [x] T017 [US3] Extend wall time in `mil/slurm/seg_mil_sweep.sh` from 24h to 48h to accommodate the full 28-config sweep
- [x] T018 [US3] Run transformer configs: `sbatch mil/slurm/seg_mil_sweep.sh` — resume-safe, only transformer configs will run
- [x] T019 [US3] Verify transformer results: confirm `all_configs.json` has 28 entries, transformer rows have non-NaN AUROC, and `config.json` in each transformer directory contains the expected HP fields

**Checkpoint**: US3 complete — transformer MIL results available for comparison against simpler aggregators.

---

## Phase 6: User Story 4 — Weakly-Supervised Frame-Level Prediction (Priority: P4)

**Goal**: Implement `mil/eval_weak_diarization.py` to measure how well MIL attention weights correlate with ground-truth child speech from RTTM files.

**Independent Test**: Running `python mil/eval_weak_diarization.py --results-dir mil/mil_results/seg_mil --split-csv whisper-modeling/seen_child_splits/test.csv --rttm-cache whisper-modeling/usc_sail_rttm_cache --output mil/mil_results/seg_mil/weak_diarization_eval.csv` produces a CSV with Pearson, Spearman, and AUROC columns for all attention-variant configs.

- [x] T020 [US4] Create `mil/eval_weak_diarization.py` with CLI args `--results-dir`, `--split-csv`, `--rttm-cache`, `--output`; add a docstring explaining inputs, outputs, and the child-speaker label convention
- [x] T021 [US4] Implement RTTM child-fraction computation in `mil/eval_weak_diarization.py`: for a given `(audio_path, seg_start, seg_end)`, load the RTTM via the `{stem}__{md5(audio_path)}.rttm` naming convention, parse all SPEAKER lines, identify child-speaker lines by label containing any of `{"CHI", "KCHI", "CHILD"}` (case-insensitive), compute overlap between each child segment and `[seg_start, seg_end]`, return `child_overlap_duration / segment_duration`
- [x] T022 [US4] Implement correlation metrics in `mil/eval_weak_diarization.py`: for each `test_segment_weights.csv` found under `--results-dir`, load rows, compute GT child fraction per segment via T021, then per `(frontend, aggregator, timepoint)` group compute `scipy.stats.pearsonr`, `scipy.stats.spearmanr`, and AUROC (treating GT fraction ≥ 0.5 as binary positive, attention weight as ranking score); skip groups with <5 segments; write results to `--output` CSV with columns `frontend, aggregator, timepoint, pearson_r, pearson_pval, spearman_rho, spearman_pval, auroc_ranking, n_segments, n_clips`
- [x] T023 [US4] Run the weak diarization evaluation: `python mil/eval_weak_diarization.py --results-dir mil/mil_results/seg_mil --split-csv whisper-modeling/seen_child_splits/test.csv --rttm-cache whisper-modeling/usc_sail_rttm_cache --output mil/mil_results/seg_mil/weak_diarization_eval.csv`
- [x] T024 [US4] Verify `mil/mil_results/seg_mil/weak_diarization_eval.csv` has rows for all attention-variant configs (attention, gated_attention, transformer) × 2 age bands; confirm no unexpected NaN correlations; note if any frontend shows near-zero correlation (important negative result)

**Checkpoint**: US4 complete — weak diarization evaluation written and results available for thesis.

---

## Phase 7: Polish & Cross-Cutting Concerns

**Purpose**: Documentation, thesis table verification, and CLAUDE.md sync.

- [x] T025 [P] Update the `mil/` Architecture section in `CLAUDE.md` to document `NoisyORAgg`, `TopKAgg`, `TransformerAgg`, and `eval_weak_diarization.py`; add `mil/eval_weak_diarization.py` to Key Commands
- [x] T026 [P] Verify `evaluation/configs/thesis_tables.yaml` `table_segment_mil` entry still resolves correctly against the updated `all_configs.json` (which now has more entries and new age-band fields)
- [x] T027 Run `python evaluation/aggregate_thesis_tables.py --skip-missing` from repo root and confirm `evaluation/thesis_tables/table_segment_mil.csv` generates without errors and contains all 28 rows

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies — verify immediately
- **Foundational (Phase 2)**: Depends on Phase 1 verification — blocks all story work
- **US1 (Phase 3)**: Depends on Phase 2 — zero new code, just confirmation
- **US2 (Phase 4)**: Depends on US1 confirmation (T003); T004 and T005 can run in parallel
- **US3 (Phase 5)**: Independent of US2 — can start after Phase 2; T012 can run in parallel with US2 tasks
- **US4 (Phase 6)**: Depends on US1/US3 trained models having produced `test_segment_weights.csv`; T020–T022 can be written while sweep runs
- **Polish (Phase 7)**: Depends on US2, US3, US4 completion

### User Story Dependencies

- **US1 (P1)**: No new code — reads existing results. Gate for all other stories.
- **US2 (P2)**: T004/T005 (new model classes) parallel; T006 depends on T004+T005; T008/T009 (seg_train.py) independent of new aggregators
- **US3 (P3)**: T012 (TransformerAgg) can be written in parallel with US2 sweep running
- **US4 (P4)**: T020–T022 can be written while sweep runs; T023 waits for attention-variant configs to complete

### Parallel Opportunities

- T004 (NoisyORAgg) ‖ T005 (TopKAgg) — different classes in same file, write sequentially
- T008 (age-band inference) ‖ T012 (TransformerAgg) — entirely different concerns
- T020–T022 (eval script) can be written while T018 (transformer sweep) runs
- T025 (CLAUDE.md) ‖ T026 (thesis_tables.yaml check) — different files

---

## Parallel Example: US2

```bash
# These two classes can be drafted in parallel (same file, but independent):
Task: "Implement NoisyORAgg in mil/seg_model.py"       # T004
Task: "Implement TopKAgg in mil/seg_model.py"           # T005

# While the sweep runs (T010), write the eval script:
Task: "Create mil/eval_weak_diarization.py"             # T020
Task: "Implement RTTM child-fraction computation"       # T021
```

---

## Implementation Strategy

### MVP (US1 + US2 Only)

1. Phase 1: Verify prerequisites (T001)
2. Phase 2: Confirm gated_attention results (T002)
3. Phase 3: Document US1 comparison (T003)
4. Phase 4: Implement NoisyORAgg + TopKAgg + age-band inference → run sweep (T004–T011)
5. **STOP and VALIDATE**: Confirm age-band AUROC delta in the predicted direction
6. Write up developmental finding

### Full Scope (US1–US4)

1. MVP above → then US3 (TransformerAgg) → then US4 (eval_weak_diarization.py)
2. Each story adds an independently reportable result

### Out of Scope (US5/US6)

- TinyVox-pretrained encoder (US5): gated on TinyVox extraction and tier-1/2 plateau
- End-to-end learned proposers (US6): gated on frontend bottleneck analysis from US1–US3 results

---

## Notes

- Resume-safe: any sweep config with an existing `test_metrics.json` is skipped
- Transformer attention weights: use `need_weights=True` in `nn.MultiheadAttention` and average CLS row across heads
- NoisyORAgg uses `logaddexp(0, ...)` to avoid log(0) via log-space arithmetic
- `test_metrics_by_timepoint.csv` uses the same val-tuned threshold as `test_metrics.json` — do not re-tune on age-band subsets
- Weak diarization eval uses USC-SAIL RTTM cache by default (most complete child-speaker labels); fall back to VTC cache for any clips missing from USC-SAIL cache
