# Tasks: Audio LLM Zero-Shot Baseline

**Input**: Design documents from `specs/010-audio-llm-baseline/`
**Plan**: [plan.md](plan.md) | **Spec**: [spec.md](spec.md)
**Data model**: [data-model.md](data-model.md) | **Contracts**: [contracts/cli_inference.md](contracts/cli_inference.md)

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel with other [P] tasks in the same phase
- **[US#]**: User story this task belongs to
- No test tasks (not requested in spec; validation is experimental)

---

## Phase 1: Setup

**Purpose**: Directory structure, gitignore, and log folder for the new baseline.

- [x] T001 Add `baselines/audio_llm_cache/` to `.gitignore` (alongside existing `pyannote/video_face_cache/` entry); create `baselines/slurm/` directory if it does not exist; create `logs/baselines/` log directory

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Shared helper functions required by all three user stories. All live in `baselines/audio_llm_baseline.py`.

**⚠️ CRITICAL**: All user story implementation depends on these helpers being complete.

- [x] T002 Scaffold `baselines/audio_llm_baseline.py` with imports (transformers, torchaudio, torch, pandas, numpy, sklearn, hashlib, json, pathlib, argparse, re, warnings); implement cache helpers: `_cache_path(audio_path, model_slug, cache_dir) -> str` using `{stem}__{md5(audio_path)}.json` naming; `_load_cache(path) -> dict | None` (returns None if file absent); `_save_cache(path, entry: dict)` (atomic write via temp file + rename)
- [x] T003 Implement `_load_audio(audio_path: str) -> tuple[np.ndarray | None, int]` in `baselines/audio_llm_baseline.py`: load WAV with `torchaudio.load`, squeeze to mono float32, resample to 16 kHz via `torchaudio.functional.resample` if needed, truncate at 30 s; return `(waveform_np, 16000)`; on FileNotFoundError return `(None, 0)` and log warning
- [x] T004 Implement `_load_model(model_name, dtype, quantize_4bit, device)` in `baselines/audio_llm_baseline.py`: load `AutoProcessor.from_pretrained(model_name)` and `AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.bfloat16)`; if `quantize_4bit=True`, pass `BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type="nf4")` as `quantization_config`; call `model.eval()`; return `(processor, model)`
- [x] T005 Implement `_infer_clip(processor, model, waveform_np, sr, prompt_text, device, few_shot_examples=None)` in `baselines/audio_llm_baseline.py`: build conversation using Qwen2-Audio chat template — if `few_shot_examples` is non-empty prepend each `(audio, label)` pair as a user+assistant turn; add query audio as final user turn with `prompt_text`; call `model.generate(max_new_tokens=10, output_scores=True, return_dict_in_generate=True)`; extract `scores[0][0]` logit tensor; get yes/no token IDs via `processor.tokenizer.encode("yes")[-1]` and same for "no"; compute `prob = float(torch.softmax(torch.tensor([logit_yes, logit_no]), dim=0)[0])`; decode `response_raw` from generated tokens; set `parse_status="parsed"` if response_raw.strip().lower().startswith(("yes","no")), else "fallback" with `prob=0.5`; return dict matching `AudioLLMCacheEntry` schema from data-model.md

**Checkpoint**: All helper functions implemented — user story phases can begin.

---

## Phase 3: User Story 1 — Zero-Shot Evaluation on Seen-Child Test Split (Priority: P1) 🎯 MVP

**Goal**: Run Qwen2-Audio-7B-Instruct zero-shot over the full val + test split; produce `val_metrics_tuned.json` and `test_metrics_tuned.json` with all five metrics plus delta vs. BabAR; SLURM job script for cluster submission.

**Independent Test**: `python baselines/audio_llm_baseline.py --split val --max-clips 10 --dry-run` exits 0 and prints 3 example prompts; `python baselines/audio_llm_baseline.py --split val --max-clips 10` produces `val_predictions.csv` with 10 rows and `prob` values in [0.0, 1.0]; `--split test` exits with code 2 if `val_metrics_tuned.json` is absent.

- [x] T006 [US1] Implement argparse CLI in `baselines/audio_llm_baseline.py` per `contracts/cli_inference.md`: add all arguments — `--split` (val/test), `--split-csv` (default `whisper-modeling/seen_child_splits/{split}.csv`), `--train-csv`, `--model` (default `Qwen/Qwen2-Audio-7B-Instruct`), `--model-slug` (default `qwen2_audio_7b`), `--output-dir` (default `baselines/audio_llm_baseline_runs/{model_slug}`), `--cache-dir` (default `baselines/audio_llm_cache/{model_slug}`), `--prompt-template` (default `zero_shot_v1`), `--n-shot` (default `0`), `--threshold` (float, default `None`), `--seed` (default `42`), `--device` (default `cuda`), `--dtype` (default `bfloat16`), `--quantize-4bit` (flag), `--max-clips` (int, default `None`), `--dry-run` (flag); implement `--dry-run` path: print 3 example prompts built from the first 3 clips in split CSV and exit 0; implement test-before-val guard: if `--split test` and `val_metrics_tuned.json` absent, print error and exit 2
- [x] T007 [US1] Implement main zero-shot inference loop in `baselines/audio_llm_baseline.py` (`--n-shot 0` path): load split CSV with pandas; apply `--max-clips` cap; load model once via `_load_model()`; iterate rows — check cache first and load if hit; on cache miss call `_load_audio()` and `_infer_clip()`; save result to cache; append `AudioLLMPrediction` row (all fields from data-model.md); print progress every 50 clips; at end write `{split}_predictions.csv` to `--output-dir`; handle missing audio files (NaN row, parse_status=error) without crashing
- [x] T008 [US1] Implement threshold tuning and metrics in `baselines/audio_llm_baseline.py`: `_compute_metrics(y_true, y_score, threshold) -> dict` using sklearn `f1_score`, `precision_score`, `recall_score`, `roc_auc_score`, `average_precision_score`; `_tune_threshold(y_true, y_score) -> float` grid-search over `np.linspace(0.05, 0.95, 19)` maximizing F1; when `--split val`: tune threshold, compute all metrics, compute `delta_f1_vs_babar = f1 - 0.874`, `delta_auroc_vs_babar = auroc - 0.820`, `delta_auprc_vs_babar = auprc - 0.918`, write `val_metrics_tuned.json`; when `--split test`: load threshold from `val_metrics_tuned.json`, compute per-timepoint breakdown grouped by `timepoint_norm`, write `test_metrics_tuned.json` and `test_metrics_by_timepoint.csv`
- [x] T009 [US1] Implement degenerate detection and `config.json` in `baselines/audio_llm_baseline.py`: after all predictions collected, compute `prediction_variance = float(np.var([r["prob"] for r in rows if not np.isnan(r["prob"])]))` and `degenerate_flag = prediction_variance < 0.01`; compute `frac_yes = sum(r["predicted"] for r in rows) / len(rows)`; if `degenerate_flag`, print `\n[WARNING] Degenerate predictions detected (variance={prediction_variance:.4f}). Check prompt format.\n`; write `config.json` at end of test run with all `AudioLLMConfig` fields including n_clips_total, n_clips_cached, n_clips_error, prediction_variance, degenerate_flag, frac_yes, frac_no
- [x] T010 [P] [US1] Create `baselines/slurm/run_audio_llm_baseline.sh`: SLURM header `--job-name=audio_llm`, `--gres=gpu:1`, `-t 12:00:00`, `--mem=64G`, `-c 4`, `-p ou_bcs_normal,pi_satra`, `-o logs/baselines/audio_llm_%j.out`, `-e logs/baselines/audio_llm_%j.out`; source miniforge conda; activate `child-vocalizations`; `cd /orcd/scratch/orcd/008/manaal/child-adult-diarization`; accept `$1=split` (default `val`), `$2=model_slug` (default `qwen2_audio_7b`), `$3=n_shot` (default `0`); run `python baselines/audio_llm_baseline.py --split $1 --model-slug $2 --n-shot $3 --seed 42`; print `Start: $(date)` and `Done: $(date)`
- [x] T011 [US1] Val-set zero-shot inference complete (job 12643230, 2026-04-27): F1=0.859, AUROC=0.781, AUPRC=0.898, threshold=0.85; fixed 3 bugs in sequence: AutoModelForCausalLM→Qwen2AudioForConditionalGeneration, audios=→audio=, generate()→model(**inputs) forward pass + logsumexp; results in `baselines/audio_llm_baseline_runs/qwen2_audio_7b/val_predictions.csv` and `val_metrics_tuned.json`
- [x] T012 [US1] Test-set zero-shot inference complete (job 12643375, 2026-04-27): F1=0.871, AUROC=0.725, AUPRC=0.853, thr=0.85; 14_month F1=0.838, 36_month F1=0.904; delta_f1=-0.003, delta_auroc=-0.095, delta_auprc=-0.065 vs BabAR; results committed to git

**Checkpoint**: Zero-shot results committed — US1 independently demonstrable with a single metrics comparison command.

---

## Phase 4: User Story 2 — Few-Shot Comparison (Priority: P2)

**Goal**: 2-shot variant using same-child training clips as in-context examples; separate result folder `qwen2_audio_7b_2shot`; comparable metrics structure.

**Independent Test**: `python baselines/audio_llm_baseline.py --split val --n-shot 2 --model-slug qwen2_audio_7b_2shot --max-clips 10` produces `val_predictions.csv` with `n_shot=2` column; falls back to n_shot=0 for children with < 2 training clips and logs a warning.

- [x] T013 [US2] Implement `_find_few_shot_examples(audio_path, train_csv_path, n_shot, seed) -> list[tuple[str, int]]` in `baselines/audio_llm_baseline.py`: parse `sub-{ID}` from BIDS audio_path using `re.search(r"sub-([A-Za-z0-9]+)", audio_path)`; load train CSV; filter to same `child_id` and exclude query clip; sample `n_shot // 2` positive (label=1) and `n_shot // 2` negative (label=0) rows using `rng.choice` with fixed seed; verify each audio file exists on disk; return list of `(audio_path, int(label))` tuples; return `[]` with logged warning if fewer than `n_shot` examples available
- [x] T014 [US2] Wire few-shot path into main loop in `baselines/audio_llm_baseline.py`: when `--n-shot > 0`, call `_find_few_shot_examples()` for each clip; pass `few_shot_examples` to `_infer_clip()`; `_infer_clip()` extends the chat conversation by prepending each `(audio_waveform, label)` pair as `user: <audio> + question` then `assistant: yes/no` turns before the query; update `n_shot` field in predictions CSV and cache entry
- [ ] T015 [US2] Submit few-shot val + test inference and commit results: `sbatch baselines/slurm/run_audio_llm_baseline.sh val qwen2_audio_7b_2shot 2` then `sbatch ... test qwen2_audio_7b_2shot 2` after val completes; verify results in `baselines/audio_llm_baseline_runs/qwen2_audio_7b_2shot/`; commit all result files

**Checkpoint**: Zero-shot and 2-shot results both committed — compare `delta_f1_vs_babar` across both model slugs.

---

## Phase 5: User Story 3 — Thesis Comparison Table Update (Priority: P3)

**Goal**: Audio LLM rows appear in the unified baseline comparison table automatically — no manual transcription.

**Independent Test**: `python evaluation/build_master_table.py` completes without error and the output CSV contains at least one row with `diarizer` starting with `audio_llm_`; values are read from committed JSON files not hardcoded.

- [x] T016 [US3] Extend `evaluation/build_master_table.py`: add discovery loop — `glob.glob("baselines/audio_llm_baseline_runs/*/test_metrics_tuned.json")`; for each found file, read JSON and extract `f1`, `auroc`, `auprc`, `precision`, `recall`, `delta_f1_vs_babar`, `delta_auroc_vs_babar`, `delta_auprc_vs_babar`; derive `model_slug` from parent directory name; add row with `diarizer = f"audio_llm_{model_slug}"`; append to the master comparison DataFrame before writing the output CSV; handle missing fields gracefully (NaN) without crashing
- [x] T017 [P] [US3] Update `CLAUDE.md`: add "10. **Audio LLM Baseline** — Qwen2-Audio-7B-Instruct zero-shot child vocalization detection (`baselines/audio_llm_baseline.py`)" to Project Overview diarizer list; add audio LLM baseline key commands (val inference, test inference, few-shot variant) to Key Commands section; add `baselines/audio_llm_baseline_runs/{model_slug}/` to Results Storage section with sub-file layout; add audio LLM row (TBD until T012 completes) to the enrollment test metrics table; add gotcha: "Prompt cache invalidation — if prompt template changes, delete `baselines/audio_llm_cache/{model_slug}/` before rerunning"

**Checkpoint**: `aggregate_thesis_tables.py` includes audio LLM row — US3 independently verifiable.

---

## Phase N: Polish & Cross-Cutting Concerns

- [x] T018 [P] Update `evaluation/thesis_tables.yaml` (or equivalent config read by `aggregate_thesis_tables.py`) to include `audio_llm_qwen2_audio_7b` in the baseline comparison table section if that table config is file-driven (check whether `evaluation/configs/thesis_tables.yaml` lists individual result paths or auto-discovers them)
- [x] T019 Update `specs/001-child-vocal-thesis/tasks.md` Phase 8 to add a new task T097 tracking the audio LLM baseline enrollment run and results commit, matching the format of T089/T095

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies — start immediately
- **Foundational (Phase 2)**: Depends on T001; helpers must exist before any inference code
- **US1 (Phase 3)**: Depends on T002–T005; T010 (SLURM script) can be written in parallel with T006–T009; T011 depends on T006–T010; T012 depends on T011
- **US2 (Phase 4)**: Depends on T002–T009 (extends existing functions); T013–T014 can be written in parallel with T011/T012 if T006–T009 are complete; T015 depends on T013–T014
- **US3 (Phase 5)**: T016 depends on T012 (result files must exist); T017 depends on T012 (to add real metrics to table)
- **Polish (Phase N)**: Depends on all prior phases

### User Story Dependencies

- **US1 (P1)**: Starts after Foundational (T002–T005); no dependency on US2/US3
- **US2 (P2)**: Starts after US1 helpers exist (T002–T009); soft dependency on US1 completion for comparison; T015 should run after T012 for clean comparison
- **US3 (P3)**: Hard dependency on T012 (test results committed); otherwise independent

### Same-File Constraints (sequential)

- T002 → T003 → T004 → T005 → T006 → T007 → T008 → T009 (all in `baselines/audio_llm_baseline.py`; implement top-to-bottom in one pass)
- T013 → T014 (both extend same file; T014 depends on T013's function signature)

### Parallel Opportunities

- T010 (SLURM script) can be written in parallel with T007–T009 (different file)
- T016 + T017 (evaluation table + CLAUDE.md) can be done in parallel (different files)
- T018 + T019 (two different spec/config files) can be done in parallel
- T011 (val SLURM job) and T013–T014 (few-shot code) can proceed in parallel after T010

---

## Parallel Execution Examples

```bash
# US1: write SLURM script while finishing inference loop and metrics
Task: "T010 Write baselines/slurm/run_audio_llm_baseline.sh"
Task: "T007–T009 Implement inference loop + metrics in audio_llm_baseline.py"

# Polish: parallel doc updates
Task: "T016 Extend evaluation/build_master_table.py"
Task: "T017 Update CLAUDE.md"
Task: "T018 Update thesis_tables.yaml"
Task: "T019 Update specs/001-child-vocal-thesis/tasks.md"
```

---

## Implementation Strategy

### MVP First (US1 Only)

1. Phase 1: Setup (T001)
2. Phase 2: Foundational helpers (T002–T005)
3. Phase 3: US1 zero-shot inference + SLURM + runs (T006–T012)
4. **STOP and VALIDATE**: Check `test_metrics_tuned.json` for F1/AUROC/AUPRC; inspect `degenerate_flag`
5. This alone produces the thesis datapoint — "audio LLMs perform at X vs. BabAR Y"

### Incremental Delivery

1. Setup + Foundational → helpers ready
2. US1 (zero-shot) → thesis baseline row exists (MVP)
3. US2 (few-shot) → second baseline row, shows whether in-context examples help
4. US3 (table integration) → auto-generated thesis table complete
5. Polish → all tracking updated

---

## Notes

- All SLURM jobs write logs to `logs/baselines/audio_llm_{SLURM_JOB_ID}.out`
- US1 is independently completable in a single afternoon (no training required)
- If `degenerate_flag=true`, check: (a) model loaded correctly, (b) processor audio input format, (c) yes/no token IDs match tokenizer vocabulary
- Do not tune threshold on test set — T008 guards this with an exit-code-2 check
- US2 reference clips must come from `train.csv` only — same constraint as ECAPA enrollment
- Commit result files in `baselines/audio_llm_baseline_runs/` per Constitution §VI; cache files in `baselines/audio_llm_cache/` are gitignored
