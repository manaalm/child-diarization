# CLI Contracts — Spec 022 PI Thesis Revisions

CLI surface for the four new scripts introduced by this spec. Each contract documents the command, required and optional arguments, exit codes, and the canonical output artefacts.

---

## 1. `whisper-modeling/make_seen_child_split.py` (modified, US1)

**Command**:

```bash
cd whisper-modeling
PYTHONPATH=. python make_seen_child_split.py [--use-bids-timepoint] [--annotations-csv PATH] [--out-dir DIR] [--seed N]
```

**Required**: none (every arg has a default).

**Optional**:
- `--use-bids-timepoint` (bool, default `true` after US1 lands): switch the `timepoint_norm` source from `anotated_processed.csv` to `bids_session_to_timepoint(audio_path)`. The legacy spreadsheet path is preserved behind `--use-bids-timepoint=false` so old behaviour is reproducible.
- `--annotations-csv` (path): spreadsheet source. Default `/orcd/scratch/bcs/001/sensein/sails/BIDS_data/anotated_processed.csv`.
- `--out-dir` (path): output directory for split CSVs. Default `/home/manaal/orcd/scratch/child-adult-diarization/whisper-modeling/seen_child_splits/`.
- `--seed` (int): split seed. Default 42.

**Outputs**:
- `<out-dir>/master_with_split.csv` — full split with corrected `timepoint_norm`.
- `<out-dir>/{train,val,test}.csv` — per-split files.
- `<out-dir>/split_summary.json` — counts + provenance.
- `<out-dir>/bids_correction_provenance.json` — per-(child, clip) BIDS-vs-spreadsheet decision log.

**Exit codes**: 0 success; 1 BIDS dataset root missing; 2 spreadsheet missing; 3 row count regression > 5% vs prior split (safety guard).

**Side effects**: overwrites `master_with_split.csv` and split CSVs in `<out-dir>`. The prior versions MUST be committed before invocation (Constitution VI; spec quickstart enforces).

---

## 2. `evaluation/balanced_metrics.py` (new, US2)

**Command**:

```bash
python evaluation/balanced_metrics.py [--predictions-glob "PATTERN"] [--output PATH] [--threshold-source val|test]
```

**Required**: none.

**Optional**:
- `--predictions-glob` (string): glob pattern for predictions CSVs. Default `mil/mil_results/*/test_predictions.csv pseudo_frame/results/*/test_predictions.csv baselines/audio_llm_baseline_runs/*/test_predictions.csv baselines/scene_analysis_runs/*/test_predictions.csv whisper-modeling/usc_sail_enrollment_runs/*/test_predictions.csv`.
- `--output` (path): output CSV. Default `evaluation/balanced_metrics_summary.csv`.
- `--threshold-source` (enum {`val`, `test`}, default `val`): which split's tuned threshold to apply when computing F1/balanced-accuracy. `val` is the canonical setting (Constitution IV).

**Outputs**:
- `evaluation/balanced_metrics_summary.csv` — one row per system per split (see `contracts/balanced_metrics_summary.schema.md`).

**Exit codes**: 0 success; 1 no predictions matched glob; 2 a predictions CSV is malformed (missing required column).

---

## 3. `evaluation/group_stratified_kfold.py` (new, US2)

**Command**:

```bash
python evaluation/group_stratified_kfold.py --system SYSTEM_NAME [--k N] [--seed N] [--split-only]
```

**Required**:
- `--system` (string): system slug (e.g., `whisper_mil`, `whisper_pseudo_frame`, `babar_combined`). Maps to a config under `mil/configs/` or equivalent.

**Optional**:
- `--k` (int, default 5): number of folds.
- `--seed` (int, default 42).
- `--split-only` (flag): write the fold-membership JSON and exit (no training). Useful for the SLURM array dispatch.

**Outputs**:
- `mil/mil_results/<SYSTEM>_groupstrat<K>_f<I>/best_checkpoint.pt` per fold (training mode).
- `mil/mil_results/<SYSTEM>_groupstrat<K>_f<I>/test_metrics_tuned.json` per fold.
- `mil/mil_results/<SYSTEM>_groupstrat<K>_membership.json` — fold-membership descriptor (split-only mode).
- Aggregated row in `evaluation/group_stratified_kfold_summary.csv` (appended atomically).

**Exit codes**: 0 success; 1 unknown system slug; 2 fold-stratification guard violated (positive-rate gap > 0.05 with k=5; retry with k=3).

---

## 4. `evaluation/loocv_subset.py` (new, US2)

**Command**:

```bash
python evaluation/loocv_subset.py --system SYSTEM_NAME [--child CHILD_ID] [--max-children N]
```

**Required**:
- `--system` (string): one of the three top-band systems (`whisper_pseudo_frame`, `whisper_medium_mil`, `babar_combined`).

**Optional**:
- `--child` (string): single-child LOOCV (run one fold). Used by SLURM array dispatch.
- `--max-children` (int): cap the LOOCV to the first N children (for cost-controlled smoke runs).

**Outputs**:
- `mil/mil_results/<SYSTEM>_loocv/<CHILD_ID>/test_metrics_tuned.json` per held-out child.
- Aggregated rows in `evaluation/loocv_subset_summary.csv`.

**Exit codes**: 0 success; 1 system not in the LOOCV-approved subset; 2 cost guard triggered (estimated runtime > budget).

---

## 5. `baselines/scene_analysis_baseline.py` (new, US3)

**Command**:

```bash
python baselines/scene_analysis_baseline.py --model {yamnet|ast} --split {val|test|test_all} [--out-dir DIR]
```

**Required**:
- `--model` (enum {`yamnet`, `ast`}).
- `--split` (enum {`val`, `test`, `test_all`}): `val` and `test` use `whisper-modeling/seen_child_splits/`; `test_all` uses `whisper-modeling/all_children_splits/test_all.csv`.

**Optional**:
- `--out-dir` (path): override the default `baselines/scene_analysis_runs/<model>/`.

**Outputs**:
- `<out-dir>/{val,test,test_all}_predictions.csv` — columns `{clip_id, child_id, label, p_child_voc, p_child_speech, p_babbling, p_baby_cry, p_children_shouting, prediction, timepoint_norm}`.
- `<out-dir>/{val,test,test_all}_metrics_tuned.json` — extended metric set.
- `<out-dir>/README.md` — AudioSet class-to-score mapping (FR-016).

**Exit codes**: 0 success; 2 YAMNet sibling env missing (subprocess bridge failed); 3 AST checkpoint download failed (offline-mode + cache miss).

**Subprocess bridge (YAMNet only)**: shells out to `encoders/yamnet_worker.py` running in the `yamnet-eval/.venv` sibling env. The worker reads `(clip_id, audio_path)` from stdin (CSV), writes `(clip_id, p_class_0, p_class_1, ..., p_class_N)` to stdout (CSV). The parent aggregates per the class-mapping rule.

---

## 6. `baselines/audio_llm_baseline.py` (modified, US3)

**Command** (no signature change):

```bash
python baselines/audio_llm_baseline.py --split {val|test|test_all} --model qwen35_omni_7b [--prompt-template TEMPLATE]
```

**New model slug**: `qwen35_omni_7b` (in addition to existing `qwen25_omni_7b`, `qwen2_audio_7b`).

**Cache invalidation guard**: if `baselines/audio_llm_cache/qwen35_omni_7b/` exists and was last written before the current `audio_llm_baseline.py` mtime, the script EXITS with code 4 and prints `cache stale — rm -rf baselines/audio_llm_cache/qwen35_omni_7b/ and retry`.

**Outputs**: standard audio-LLM result layout under `baselines/audio_llm_baseline_runs/qwen35_omni_7b/<split>/`.

---

## 7. `docs/per_model_training_data.py` (new, US4)

**Command**:

```bash
python docs/per_model_training_data.py [--output PATH]
```

**Optional**:
- `--output` (path): default `docs/per_model_training_data.csv`.

**Behaviour**: walks every canonical result root, reads each `config.json`, and produces one row per system using the schema in `contracts/per_model_training_data.schema.md`.

**Outputs**:
- `docs/per_model_training_data.csv`.

**Exit codes**: 0 success; 1 a `config.json` is malformed (logs path, continues with rest).

---

## Invocation contract for SLURM scripts

Every new SLURM script (`evaluation/slurm/run_group_stratified_kfold.sh`, `baselines/slurm/run_scene_analysis_baseline.sh`) MUST:

1. Set `export TRANSFORMERS_OFFLINE=1` and `export HF_HUB_OFFLINE=1` (CLAUDE.md gotcha for transformers ≥4.57).
2. `unset HF_TOKEN HUGGINGFACE_HUB_TOKEN HF_HUB_TOKEN` (CLAUDE.md gotcha for public-model 401).
3. Activate the canonical conda env: `conda activate child-vocalizations` (or the YAMNet sibling env for YAMNet jobs).
4. Log the SLURM job ID into the result dir's `config.json` under `slurm_job_id`.
5. Write stdout/stderr to `logs/adult/<scriptname>_<jobid>.out`.
