---
description: "Task list for Multiple Instance Learning Workflow"
---

# Tasks: Multiple Instance Learning Workflow

**Input**: Design documents from `specs/002-mil-workflow/`
**Prerequisites**: plan.md ✅, spec.md ✅, research.md ✅, data-model.md ✅,
contracts/script-interfaces.md ✅, quickstart.md ✅

**Tests**: No automated test suite (ML research project). Validation is
experimental — val-set performance, per-timepoint metrics, and thesis table
integration per Constitution Principles IV–V.

**Organization**: Tasks are grouped by user story (US1–US3) to enable
independent implementation and testing. See plan.md for file paths.

## Format: `[ID] [P?] [Story?] Description`

- **[P]**: Can run in parallel (different files or independent jobs)
- **[Story]**: US1 / US2 / US3
- File paths are relative to repo root unless otherwise noted

---

## Phase 1: Setup

**Purpose**: Create the `mil/` module skeleton, configs, and SLURM script.
All tasks are independent and can run in parallel.

- [X] T001 Create directory structure: `mil/`, `mil/configs/`, `mil/slurm/`,
  `mil/mil_results/`, `logs/mil/`
- [X] T002 [P] Create `mil/configs/wavlm_mil.yaml` — per contracts/script-interfaces.md:
  `variant_name: wavlm_mil`, `backbone: microsoft/wavlm-base-plus`,
  `backbone_layer: -1`, `window_sec: 2.0`, `stride_sec: 1.0`,
  `mil_hidden_dim: 256`, `mil_dropout: 0.25`, `lr: 1.0e-3`, `epochs: 20`,
  `patience: 5`, `batch_size: 8`, `pos_weight: null`, `seed: 42`,
  `split_dir: whisper-modeling/seen_child_splits`, `device: cuda`
- [X] T003 [P] Create `mil/configs/whisper_mil.yaml` — identical to T002 except
  `variant_name: whisper_mil` and `backbone: openai/whisper-small`
- [X] T004 Create `mil/slurm/train_mil.sh` — SLURM job script:
  headers `#SBATCH -c 4 -t 8:00:00 -p ou_bcs_normal,pi_satra --mem=40G --gres=gpu:1`
  with logs to `logs/mil/train_%j.out` / `.err`; activates `child-vocalizations`
  conda env; calls `python mil/mil_train.py --config $1`

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Shared utilities and dataset loader used by all three user stories.
T005 and T006 are independent and can run in parallel; both must complete before
any US1–US3 work begins.

**⚠️ CRITICAL**: US1 training (T008) cannot start until T006 is complete.

- [X] T005 [P] Implement `mil/mil_utils.py` — shared metric helpers:
  `compute_metrics(y_true, y_score) → dict(f1, precision, recall, auroc, auprc)`
  using sklearn; `tune_threshold(val_labels, val_scores, np.arange(0.05, 0.96, 0.05))
  → float` that maximises F1; `per_timepoint_metrics(df) → DataFrame` grouping by
  `timepoint_norm` with columns `[timepoint, f1, precision, recall, auroc, auprc, n]`
  matching unified.py schema; `save_json(d, path)` and `save_csv(df, path)` helpers

- [X] T006 [P] Implement `mil/mil_dataset.py` — `MILBagDataset(Dataset)`:
  `__init__(df, window_sec, stride_sec, sample_rate=16000)` where `df` is a
  seen_child_splits CSV DataFrame pre-filtered to `audio_exists==True`; `__getitem__`
  loads audio (torchaudio, resample to 16kHz mono), slices into windows of
  `window_sec` at `stride_sec` stride, returns
  `dict(windows=List[Tensor(1,T)], label=int, child_id=str, timepoint_norm=str,
  audio_path=str)`; clips shorter than `window_sec` are zero-padded to exactly one
  window; implement `mil_collate_fn(batch) → dict` that handles variable window counts
  across bags (return list-of-lists, not stacked tensors)

**Checkpoint**: Dataset and utilities ready — US1 implementation can begin.

---

## Phase 3: User Story 1 — MIL Model Training (Priority: P1) 🎯 MVP

**Goal**: Train ABMIL child presence detector from clip-level labels; produce a
committed checkpoint and val metrics for both WavLM and Whisper backbones.

**Independent Test**: `python mil/mil_train.py --config mil/configs/wavlm_mil.yaml`
completes without crash; `mil/mil_results/wavlm_mil/val_metrics_tuned.json` exists
with `f1 ≥ 0.800` on the val set; `training_history.csv` shows decreasing loss.

### Implementation for User Story 1

- [X] T007 [US1] Implement `mil/mil_model.py` — three classes:
  (1) `BackboneExtractor(nn.Module)`: loads WavLM-base+ via
  `WavLMModel.from_pretrained()` or Whisper-small via `WhisperModel.from_pretrained()`
  depending on `backbone_name`; all params frozen at init (`requires_grad=False`);
  `forward(waveform: Tensor (B,T)) → Tensor (B, T_frames, 768)` — for WavLM uses
  `output_hidden_states=True` and picks the final layer; for Whisper calls
  `encoder(input_features=mel_spectrogram).last_hidden_state`.
  (2) `GatedABMILHead(nn.Module)`: gated attention per Ilse et al. 2018;
  `__init__(in_dim=768, hidden_dim=256, dropout=0.25)`: Linear V (tanh branch),
  Linear U (sigmoid branch) both `(in_dim → hidden_dim)`; score vector
  `w = Linear(hidden_dim, 1, bias=False)`; classifier `head = Linear(in_dim, 1)`;
  `forward(h: Tensor (N, D)) → (logit: Tensor scalar, attn: Tensor (N,))`:
  `A = softmax(w.T * (tanh(V*h) ⊙ σ(U*h)))`, bag embedding `z = (A * h).sum(0)`,
  `logit = head(z).squeeze()`.
  (3) `MILModel(nn.Module)`: composes BackboneExtractor + GatedABMILHead;
  `forward(windows: List[Tensor]) → Tuple[logit, attn_weights]`: backbone encodes
  each window → mean-pool over frames → stack N instance embeddings → GatedABMILHead.
  Top-level `build_mil_model(cfg: dict) → MILModel` factory.

- [X] T008 [US1] Implement `mil/mil_train.py` — entry point:
  `parse_config(path) → dict` loads YAML, validates required keys; `setup_seed(seed)`
  seeds torch/numpy/random with `deterministic=True`; main `train(cfg)` function:
  (a) loads `{split_dir}/train.csv` and `{split_dir}/val.csv`, filters
  `audio_exists==True`; instantiates `MILBagDataset` for each; DataLoader with
  `mil_collate_fn`, `shuffle=True` for train.
  (b) `build_mil_model(cfg)` → model on device; optimizer = `Adam` over MIL head
  params only (backbone frozen); loss = `BCEWithLogitsLoss(pos_weight=cfg.pos_weight)`.
  (c) Per epoch: forward/backward on train bags; val forward → `val_scores`;
  `compute_metrics` → `val_f1`; save best checkpoint if `val_f1` improves; early stop
  after `patience` epochs without improvement.
  (d) After training: `tune_threshold(val_labels, val_scores)` → threshold; write
  `mil/mil_results/{variant_name}/config.json` (copy of cfg),
  `training_history.csv` (epoch, train_loss, val_loss, val_f1, val_auroc),
  `best_checkpoint.pt`, `val_metrics_tuned.json` (f1, precision, recall, auroc, auprc,
  threshold), `val_predictions.csv` (audio_path, child_id, timepoint_norm, label,
  score, prediction).
  CLI: `python mil/mil_train.py --config <yaml_path>`

- [X] T009 [US1] Submit wavlm_mil SLURM training job: job 12380891
  `sbatch mil/slurm/train_mil.sh mil/configs/wavlm_mil.yaml`; note job ID; monitor
  with `tail -f logs/mil/train_<jobid>.out`; wait for completion before running T012

- [X] T010 [P] [US1] Submit whisper_mil SLURM training job: job 12380892
  `sbatch mil/slurm/train_mil.sh mil/configs/whisper_mil.yaml`; runs in parallel
  with T009; wait for completion before T013

**Checkpoint**: Both checkpoints committed, val metrics logged — US1 complete and
independently demonstrable as a diarization-free trained model.

---

## Phase 4: User Story 2 — Comparative Evaluation Against Baselines (Priority: P2)

**Goal**: Produce test-set metrics for both MIL variants in the canonical result
folder format and verify they appear in thesis tables.

**Independent Test**: `python mil/mil_evaluate.py --checkpoint
mil/mil_results/wavlm_mil/best_checkpoint.pt --config
mil/mil_results/wavlm_mil/config.json` completes; `test_metrics_tuned.json` exists
with `f1 ≥ 0.850`; `evaluation/aggregate_thesis_tables.py` runs without error and
output CSVs include MIL rows.

**Dependency**: Requires T009 and T010 (checkpoints must exist before evaluation).

### Implementation for User Story 2

- [X] T011 [US2] Implement `mil/mil_evaluate.py` — loads `config.json` and
  `best_checkpoint.pt` from the result folder; loads `threshold` from
  `val_metrics_tuned.json`; loads `{split_dir}/test.csv` filtered to
  `audio_exists==True`; forward pass on all test bags → scores; `compute_metrics` at
  loaded threshold → `test_metrics_tuned.json`; `per_timepoint_metrics` → 
  `test_metrics_by_timepoint.csv`; also produces `val_metrics_by_timepoint.csv` if
  not already present; writes `test_predictions.csv`
  (columns: `audio_path, child_id, timepoint_norm, label, score, prediction`).
  CLI: `python mil/mil_evaluate.py --checkpoint <pt_path> --config <json_path>`

- [X] T012 [US2] Run mil_evaluate.py for wavlm_mil:
  `python mil/mil_evaluate.py --checkpoint mil/mil_results/wavlm_mil/best_checkpoint.pt
  --config mil/mil_results/wavlm_mil/config.json`; confirm
  `test_metrics_tuned.json` is written with all five metric fields.

- [X] T013 [P] [US2] Run mil_evaluate.py for whisper_mil:
  `python mil/mil_evaluate.py --checkpoint mil/mil_results/whisper_mil/best_checkpoint.pt
  --config mil/mil_results/whisper_mil/config.json`

- [X] T014 [US2] Update `evaluation/configs/thesis_tables.yaml` — add `mil_wavlm`
  and `mil_whisper` sections per contracts/script-interfaces.md; verify the file is
  valid YAML before proceeding

- [X] T015 [US2] Run `python evaluation/aggregate_thesis_tables.py`; confirm MIL rows
  appear in the comparative baseline table alongside USC-SAIL/BabAR/VTC/VBx; verify
  SC-001 (F1 ≥ 0.850 for both variants) and SC-002 (no code changes needed);
  commit thesis_tables/ CSVs

**Checkpoint**: Test metrics committed for both MIL variants; visible in thesis
comparison table — US2 independently demonstrable.

---

## Phase 5: User Story 3 — Age-Stratified MIL Analysis (Priority: P3)

**Goal**: Per-cohort (12-16 m, 34-38 m) metrics for both MIL variants, feeding
into the existing age-stratified thesis chapter.

**Independent Test**: Running `mil_age_stratified.py` for `wavlm_mil --age-group
12_16m` produces `mil/mil_results/wavlm_mil/age_stratified/12_16m/
test_metrics_tuned.json` with populated F1/AUROC; equivalent for 34_38m.

**Dependency**: Requires T009 and T010 (checkpoints) and the age manifests from
T011 in `001-child-vocal-thesis` (playlogue/manifest.csv, etc.).

### Implementation for User Story 3

- [X] T016 [US3] Implement `mil/mil_age_stratified.py` — loads config.json +
  best_checkpoint.pt + threshold from val_metrics_tuned.json; loads test split and
  inner-joins with `--manifest` CSV on `audio_path`; filters to rows where
  `age_group == --age-group` and file exists; forward pass → per-cohort scores →
  `compute_metrics` at loaded threshold → writes
  `mil/mil_results/{variant}/age_stratified/{age_group}/test_metrics_tuned.json`,
  `test_predictions.csv`, `test_metrics_by_timepoint.csv`.
  CLI: `python mil/mil_age_stratified.py --checkpoint <> --config <> --age-group
  <12_16m|34_38m> --manifest <path_to_manifest.csv>`

- [X] T017 [US3] Run age-stratified evaluation for wavlm_mil — both age groups:
  ```
  python mil/mil_age_stratified.py --checkpoint mil/mil_results/wavlm_mil/best_checkpoint.pt
    --config mil/mil_results/wavlm_mil/config.json --age-group 12_16m --manifest playlogue/manifest.csv
  python mil/mil_age_stratified.py --checkpoint mil/mil_results/wavlm_mil/best_checkpoint.pt
    --config mil/mil_results/wavlm_mil/config.json --age-group 34_38m --manifest playlogue/manifest.csv
  ```

- [X] T018 [P] [US3] Run age-stratified evaluation for whisper_mil — both age groups
  (same commands, replace `wavlm_mil` with `whisper_mil`)

- [X] T019 [US3] Verify SC-003: confirm all four age-stratified result folders exist
  (wavlm_mil × 2 age groups, whisper_mil × 2 age groups); compare 12_16m vs 34_38m
  metrics and document the inter-cohort difference in a comment in
  evaluation/configs/thesis_tables.yaml; commit all age-stratified result files

**Checkpoint**: Age-stratified metrics committed for both variants × both cohorts —
US3 independently demonstrable.

---

## Phase N: Polish & Cross-Cutting Concerns

- [X] T020 Update `CLAUDE.md` — add `mil/` to Architecture section: describe
  `mil_model.py` (BackboneExtractor + GatedABMILHead + MILModel), `mil_train.py`,
  `mil_evaluate.py`, `mil_age_stratified.py`; add `mil/mil_results/` to Results
  Storage section with key test metrics (fill in actual numbers after T015 completes);
  add MIL to the comparison table

- [ ] T021 [P] Commit all result artifacts per Constitution Principle VI: one commit
  per experiment type (`feat: add wavlm_mil results`, `feat: add whisper_mil results`);
  each commit includes `config.json`, `*_metrics_tuned.json`, `*_predictions.csv`,
  `*_by_timepoint.csv` under the correct `mil/mil_results/{variant}/` path

- [X] T022 [P] Run per-child error analysis: for each variant, compute per-child F1
  using `test_predictions.csv` grouped by `child_id`; write
  `mil/mil_results/{variant}/per_child_error_rates.csv` (columns: child_id, n_clips,
  f1, n_fp, n_fn); commit — required by Constitution Principle V

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies — start immediately
- **Foundational (Phase 2)**: Depends on Setup (T001–T004)
- **US1 (Phase 3)**: T007 and T008 depend on Foundational; T009/T010 depend on T007+T008
- **US2 (Phase 4)**: T011 implementation can begin while training runs (T009/T010) are
  in progress; T012/T013 evaluation runs require checkpoints (T009/T010 complete)
- **US3 (Phase 5)**: Requires T009/T010 (checkpoints) and T016 implementation; depends
  on age manifests from `001-child-vocal-thesis` task T011 (playlogue/manifest.csv)
- **Polish (Phase N)**: Depends on all evaluation runs (T012/T013) and age-stratified
  runs (T017/T018) complete

### User Story Dependencies

- **US1 (P1)**: Starts after Foundational — no dependency on US2/US3
- **US2 (P2)**: Starts after US1 training completes (checkpoints needed); T011
  implementation can overlap with training
- **US3 (P3)**: Starts after US1 training completes (checkpoints needed); T016
  implementation can overlap with training

### Critical Path

```
Phase 1 (Setup: T001–T004)
    ↓
Phase 2 (Foundational: T005, T006 in parallel)
    ↓
Phase 3 (US1: T007 → T008 → T009 ‖ T010)
    ↓
Phase 4 (US2: T011 → T012 ‖ T013 → T014 → T015)
    ‖
Phase 5 (US3: T016 → T017 ‖ T018 → T019)
    ↓
Phase N (Polish: T020, T021, T022 in parallel)
```

---

## Parallel Opportunities

```bash
# Phase 1 (all in parallel):
Task: "T002 Create wavlm_mil.yaml"
Task: "T003 Create whisper_mil.yaml"
Task: "T004 Create train_mil.sh"

# Phase 2 (in parallel):
Task: "T005 Implement mil_utils.py"
Task: "T006 Implement mil_dataset.py"

# Phase 3 training (submit both jobs, run concurrently on cluster):
Task: "T009 Submit wavlm_mil training job"
Task: "T010 Submit whisper_mil training job"

# While training jobs run, implement evaluation scripts:
Task: "T011 Implement mil_evaluate.py"
Task: "T016 Implement mil_age_stratified.py"

# Phase 4 evaluation (after checkpoints ready):
Task: "T012 Run mil_evaluate.py wavlm_mil"
Task: "T013 Run mil_evaluate.py whisper_mil"

# Phase 5 age-stratified (in parallel):
Task: "T017 wavlm_mil age-stratified"
Task: "T018 whisper_mil age-stratified"

# Polish (in parallel):
Task: "T021 Commit result artifacts"
Task: "T022 Per-child error analysis"
```

---

## Implementation Strategy

### MVP First (US1 Only)

1. Phase 1: Setup (T001–T004)
2. Phase 2: Foundational (T005, T006)
3. Phase 3: US1 — implement model + train script, submit training jobs (T007–T010)
4. **STOP and VALIDATE**: val metrics confirm F1 > 0.800, checkpoint saved
5. Thesis claim: "MIL model trained on clip-level labels without diarization" is
   fully supported

### Incremental Delivery

1. Setup + Foundational → dataset and utilities ready
2. US1 → trained checkpoints + val metrics (MVP)
3. US2 → test metrics + thesis table integration → comparable to baselines
4. US3 → age-stratified results → age-stratified chapter supported
5. Polish → per-child error rates, CLAUDE.md updated, artifacts committed

---

## Notes

- [P] tasks = different files or independent SLURM jobs, safe to run concurrently
- SLURM training jobs (T009, T010) may run 4–8 hours; implement T011 and T016
  while they run to avoid blocking
- All result folders MUST contain `config.json` per Constitution Principle VI
- Training MUST use `seed=42` per Constitution Principle I
- Never tune threshold on test set — val only (Constitution Principle II)
- `mil/mil_results/` artifacts are committed but `best_checkpoint.pt` files may be
  large — check with `du -sh mil/mil_results/` and consider adding to `.gitignore`
  if they exceed ~500MB; if excluded, document checkpoint location in CLAUDE.md
- Age-stratified evaluation (T017/T018) requires `playlogue/manifest.csv` from
  `001-child-vocal-thesis` task T011 — confirm it exists before running
