# Contract: Audio LLM Inference CLI

**Script**: `baselines/audio_llm_baseline.py`
**Purpose**: Zero-shot (and optional few-shot) child vocalization detection using a pre-trained audio LLM; produces predictions matching the enrollment result schema.

---

## CLI Arguments

```
python baselines/audio_llm_baseline.py [OPTIONS]
```

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--split` | str | `val` | Which split to run: `val` or `test` |
| `--split-csv` | path | `whisper-modeling/seen_child_splits/{split}.csv` | Override split CSV path |
| `--train-csv` | path | `whisper-modeling/seen_child_splits/train.csv` | Training split (for few-shot reference lookup) |
| `--model` | str | `Qwen/Qwen2-Audio-7B-Instruct` | HuggingFace model ID |
| `--model-slug` | str | `qwen2_audio_7b` | Filesystem-safe name for output folder |
| `--output-dir` | path | `baselines/audio_llm_baseline_runs/{model_slug}` | Results folder |
| `--cache-dir` | path | `baselines/audio_llm_cache/{model_slug}` | Per-clip JSON cache folder |
| `--prompt-template` | str | `zero_shot_v1` | Prompt variant identifier |
| `--n-shot` | int | `0` | Number of in-context audio examples (0 = zero-shot) |
| `--threshold` | float | `None` | Binary threshold; if None, tune on val set |
| `--seed` | int | `42` | Random seed for few-shot example selection |
| `--device` | str | `cuda` | Device: `cuda` or `cpu` |
| `--dtype` | str | `bfloat16` | Model dtype: `bfloat16` or `float32` |
| `--quantize-4bit` | flag | off | Load in 4-bit NF4 quantization via bitsandbytes |
| `--max-clips` | int | `None` | Cap number of clips (for dry runs / smoke tests) |
| `--dry-run` | flag | off | Print first 3 prompts and exit without inference |

---

## Output Files

All output files are written to `--output-dir`. The folder is created if it does not exist.

### `{split}_predictions.csv`

One row per clip. Columns: `clip_id`, `child_id`, `timepoint_norm`, `audio_path`, `label`, `prob`, `predicted`, `model_name`, `prompt_template`, `n_shot`, `response_raw`, `parse_status`, `logit_yes`, `logit_no`.

### `{split}_metrics_tuned.json`

When `--split val`: tuned threshold is written here; `f1`, `precision`, `recall`, `auroc`, `auprc`, `threshold`, `val_f1`, `n_positive`, `n_negative`, `delta_*_vs_babar`.

When `--split test`: threshold is loaded from `val_metrics_tuned.json` (must already exist); same metric fields written.

### `test_metrics_by_timepoint.csv`

Written after `--split test`. Columns: `timepoint_norm`, `f1`, `precision`, `recall`, `auroc`, `auprc`, `n_clips`.

### `config.json`

Written once at end of run. Contains all config fields from `AudioLLMConfig` schema. Overwrites any previous config.json for the same model_slug.

---

## Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Success — all clips processed (including cached), outputs written |
| 1 | Missing dependency (model not downloadable, audio file not found for > 50% of clips) |
| 2 | `--split test` invoked before `--split val` (val_metrics_tuned.json missing) |
| 3 | Invalid argument combination |

---

## Behavioral Contracts

- **Resume-safe**: If `--cache-dir/{stem}__{md5}.json` exists, the clip is loaded from cache and skipped during model inference. The script MUST process all clips (including cached) when assembling the final CSV.
- **No test-set threshold tuning**: When `--split test`, threshold is ALWAYS loaded from the previously written `val_metrics_tuned.json`. The script exits with code 2 if that file does not exist.
- **Graceful degradation**: If an audio file is missing or unreadable, `prob=NaN`, `predicted=NaN`, `parse_status=error` is written for that clip. The run continues.
- **Degenerate detection**: After all clips are processed, compute `prediction_variance`. If < 0.01, set `degenerate_flag=true` in `config.json` and print a prominent `[WARNING] Degenerate predictions detected` message.
- **Idempotent**: Running the script twice with the same arguments produces identical outputs (given identical cache).
