# Data Model: Audio LLM Zero-Shot Baseline

**Feature**: 010-audio-llm-baseline
**Date**: 2026-04-27

---

## Entities

### AudioLLMPrediction (per-clip inference record)

Written per clip to `val_predictions.csv` and `test_predictions.csv`.

| Field | Type | Description |
|-------|------|-------------|
| `clip_id` | string | Unique clip identifier; matches existing split CSVs |
| `child_id` | string | Target child identifier (sub-{ID} BIDS format) |
| `timepoint_norm` | string | Cohort: `14_month` or `36_month` |
| `audio_path` | string | Absolute path to 16kHz mono WAV file |
| `label` | int | Ground truth: 1 = child vocalizing, 0 = silent/adult-only |
| `prob` | float | Predicted probability of child vocalizing in [0.0, 1.0] |
| `predicted` | int | Binary prediction at val-tuned threshold: 1 or 0 |
| `model_name` | string | HuggingFace model ID (e.g., `Qwen/Qwen2-Audio-7B-Instruct`) |
| `prompt_template` | string | Short identifier for prompt variant (e.g., `zero_shot_v1`) |
| `n_shot` | int | Number of in-context examples: 0 for zero-shot, 1–3 for few-shot |
| `response_raw` | string | Raw text output from the model before parsing |
| `parse_status` | string | `parsed` / `fallback` / `error` |
| `logit_yes` | float | Raw logit for "yes" token (NaN if logprobs unavailable) |
| `logit_no` | float | Raw logit for "no" token (NaN if logprobs unavailable) |

**Validation rules**:
- `prob` ∈ [0.0, 1.0]
- `predicted` ∈ {0, 1}
- `parse_status` ∈ {"parsed", "fallback", "error"}
- `n_shot` ≥ 0
- `label` ∈ {0, 1}

**State transitions**:
- Cache miss → model inference → `parse_status=parsed` (or `fallback` / `error`)
- Cache hit → load from JSON, write to predictions CSV without re-querying model

---

### AudioLLMConfig (per-run configuration)

Written to `config.json` in the result folder. Matches the schema of all other `config.json` files in the project.

| Field | Type | Description |
|-------|------|-------------|
| `model_name` | string | HuggingFace model ID |
| `model_slug` | string | Short filesystem-safe name (e.g., `qwen2_audio_7b`) |
| `prompt_template` | string | Zero-shot or few-shot prompt template identifier |
| `prompt_text` | string | Full prompt text used for zero-shot inference |
| `n_shot` | int | Number of in-context examples |
| `threshold` | float | Val-set-tuned binary threshold |
| `val_f1` | float | F1 at tuned threshold on val set |
| `seed` | int | Random seed (42) |
| `split` | string | Which split CSV was used (`seen_child_splits`) |
| `n_clips_total` | int | Total number of clips processed |
| `n_clips_cached` | int | Clips loaded from cache (not re-inferred) |
| `n_clips_error` | int | Clips that returned parse_status=error |
| `prediction_variance` | float | Variance of `prob` across all test clips |
| `degenerate_flag` | bool | True if prediction_variance < 0.01 |
| `frac_yes` | float | Fraction of clips predicted as yes (at threshold) |
| `frac_no` | float | Fraction of clips predicted as no (at threshold) |

---

### AudioLLMMetrics (per-run summary metrics)

Written to `val_metrics_tuned.json` and `test_metrics_tuned.json`. Matches the schema of all other `*_metrics_tuned.json` files.

| Field | Type | Description |
|-------|------|-------------|
| `f1` | float | F1 score at tuned threshold |
| `precision` | float | Precision at tuned threshold |
| `recall` | float | Recall at tuned threshold |
| `auroc` | float | Area under ROC curve |
| `auprc` | float | Area under precision-recall curve |
| `threshold` | float | Val-tuned threshold used |
| `val_f1` | float | F1 on val set at this threshold |
| `n_positive` | int | Number of positive (child vocalizing) clips |
| `n_negative` | int | Number of negative clips |
| `delta_f1_vs_babar` | float | F1 minus BabAR baseline (0.874) |
| `delta_auroc_vs_babar` | float | AUROC minus BabAR baseline (0.820) |
| `delta_auprc_vs_babar` | float | AUPRC minus BabAR baseline (0.918) |

---

### AudioLLMPerTimepointMetrics (stratified breakdown)

Written to `test_metrics_by_timepoint.csv`. Matches schema of all other `*_metrics_by_timepoint.csv` files.

| Column | Type | Description |
|--------|------|-------------|
| `timepoint_norm` | string | `14_month` or `36_month` or `overall` |
| `f1` | float | F1 at threshold |
| `precision` | float | Precision at threshold |
| `recall` | float | Recall at threshold |
| `auroc` | float | AUROC |
| `auprc` | float | AUPRC |
| `n_clips` | int | Number of clips in this stratum |

---

### AudioLLMCacheEntry (per-clip cache file)

Stored at `baselines/audio_llm_cache/{model_slug}/{stem}__{md5}.json`. Gitignored.

| Field | Type | Description |
|-------|------|-------------|
| `clip_id` | string | Clip identifier |
| `audio_path` | string | Source audio path |
| `prob` | float | Predicted probability |
| `response_raw` | string | Raw model text output |
| `parse_status` | string | `parsed` / `fallback` / `error` |
| `logit_yes` | float | Raw yes logit (or NaN) |
| `logit_no` | float | Raw no logit (or NaN) |
| `model_name` | string | Model used |
| `timestamp` | string | ISO 8601 inference timestamp |
