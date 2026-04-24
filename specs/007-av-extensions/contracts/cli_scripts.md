# CLI Script Contracts: AV Extended Experiments — 007-av-extensions

All scripts live in `av_fusion/scripts/` and follow the existing 006 conventions:
- Config via `av_fusion/configs/av_extensions.yaml`
- Outputs in `av_fusion/av_results/{run_name}/`
- Idempotent: re-running skips already-computed artifacts
- Seed: 42 (fixed in config)

---

## extract_asd_features.py (extended)

**Purpose**: Extract active-speaker detection scores from video clips. Extends 006 to support LocoNet and Light-ASD in addition to TalkNet.

```
python av_fusion/scripts/extract_asd_features.py \
    --metadata-csv  whisper-modeling/seen_child_splits/master_with_split.csv \
    --output        av_fusion/av_results/{run_name}/asd_features_{model}.csv \
    --model         {talknet|loconet|light_asd}           [default: talknet] \
    [--loconet-checkpoint  /path/to/loconet_best.ckpt]    [required if --model loconet] \
    [--light-asd-checkpoint /path/to/light_asd.pt]        [required if --model light_asd] \
    [--face-cache-dir  av_fusion/face_track_cache/]       [default: shared cache] \
    [--batch-size   16]                                    [default: 16] \
    [--device       cuda]
```

**Inputs**:
- `--metadata-csv`: CSV with columns `clip_id`, `child_id`, `video_path`, `split`
- `--model`: selects which ASD checkpoint to run; controls output filename suffix

**Outputs**:
- `av_fusion/av_results/{run_name}/asd_features_{model}.csv` — one row per clip, schema: ASDFeatureRow
- Reuses `av_fusion/face_track_cache/` (populated by 006 `extract_visual_features.py`)

**Error behavior**:
- Missing checkpoint → FileNotFoundError with setup instructions
- Missing video file → NaN row in output; warning logged; no crash
- Clips where face detection finds no faces → all ASD scores = 0.0

---

## extract_gpt4o_features.py

**Purpose**: Query GPT-4o (or gpt-4o-mini) vision API for structured child-detection output from sampled video frames.

```
python av_fusion/scripts/extract_gpt4o_features.py \
    --metadata-csv  whisper-modeling/seen_child_splits/master_with_split.csv \
    --output        av_fusion/av_results/{run_name}/gpt4o_features.csv \
    [--model        gpt-4o-mini]             [default: gpt-4o-mini] \
    [--sample-rate  2]                       [frames per clip, default: 2] \
    [--cache-dir    av_fusion/gpt4o_cache/]  [default: above path] \
    [--max-clips    N]                       [optional cost cap; omit = all clips] \
    [--dry-run]                              [print cost estimate, no API calls]
```

**Inputs**:
- `--metadata-csv`: CSV with `clip_id`, `video_path`; audio-only clips (no video_path) are skipped with NaN row
- `OPENAI_API_KEY` environment variable must be set

**Outputs**:
- `av_fusion/av_results/{run_name}/gpt4o_features.csv` — one row per clip, schema: GPT4oFeatureRow
- `av_fusion/gpt4o_cache/{clip_id}_{frame_idx}.json` — raw API responses (cache)

**Cost estimation** (printed before processing):
```
Estimated clips: 1311 train + 431 val + 441 test = 2183 total
Frames to query: 2183 × 2 = 4366 frames (already cached: 0)
Estimated tokens: 4366 × ~1000 = 4.4M tokens
Estimated cost (gpt-4o-mini): $0.66
Proceed? [y/N]
```

**Error behavior**:
- API rate limit → exponential backoff, retry up to 5 times
- Malformed JSON response → NaN for all structured fields; raw text saved to `gpt4o_reasoning`
- Missing video → NaN row; no API call made

---

## train_cascaded_pipeline.py

**Purpose**: Tune the cascade thresholds (VAD gate, child ID gate) on the validation set and save the threshold configuration. No new model training is required.

```
python av_fusion/scripts/train_cascaded_pipeline.py \
    --feature-dir   av_fusion/av_results/{run_name}/ \
    --output-dir    av_fusion/av_results/{run_name}/models/ \
    [--vad-feature  kchi_total_dur]       [column from MasterFeatureTable, default: kchi_total_dur] \
    [--child-id-feature  prob]            [enrollment score column, default: prob from BabAR] \
    [--seed         42]
```

**Inputs**:
- `av_{train,val,test}.csv` from the feature-dir (from 006 `build_av_feature_table.py`)
- Val set used for threshold tuning; train set for any weighting decisions

**Outputs**:
- `models/cascade_thresholds.json` — `{"vad_threshold": 0.X, "child_id_threshold": 0.Y, "val_f1": ..., "val_auroc": ...}`
- `cascade_val_stage_breakdown.csv` — per-clip stage assignments on val set

**Threshold tuning**: Grid search over `vad_threshold ∈ [0.0, 2.0]` (seconds of KCHI speech) and `child_id_threshold ∈ [0.1, 0.9]` maximizing val F1. Only val labels used; test set never touched.

---

## smooth_predictions.py

**Purpose**: Apply temporal smoothing to any model's raw probability sequence within recording sessions.

```
python av_fusion/scripts/smooth_predictions.py \
    --predictions   av_fusion/av_results/{run_name}/predictions_test.csv \
    --output        av_fusion/av_results/{run_name}/predictions_test_smoothed.csv \
    [--method       gaussian]              [gaussian | majority_vote | moving_average] \
    [--param        None]                  [bandwidth/window; if None, tune on val] \
    [--val-predictions  av_fusion/av_results/{run_name}/predictions_val.csv]  [for param tuning] \
    [--group-cols   child_id,timepoint_norm]  [columns defining session groups]
```

**Inputs**:
- Predictions CSV: must have `clip_id`, `prob`, `label`, `child_id`, `timepoint_norm` (or `session_id`)
- `--val-predictions`: required if `--param None` (parameter tuned on val)

**Outputs**:
- Smoothed predictions CSV with added columns: `prob_smoothed`, `smoothing_method`, `smoothing_param`
- Prints val F1 (raw) vs val F1 (smoothed) as diagnostic

**Session ordering**: Within each (child_id, timepoint_norm) group, clips are ordered by `clip_id` (lexicographic); if `clip_position` column exists it is used instead.

---

## evaluate_av_fusion.py (extended, from 006)

**Extension only**: 006's `evaluate_av_fusion.py` is called with `--model cascaded_av` to evaluate the cascade output. The script must accept a `cascade_stage_breakdown.csv` alongside the standard predictions CSV and report per-stage metrics.

```
python av_fusion/scripts/evaluate_av_fusion.py \
    --feature-dir av_fusion/av_results/{run_name}/ \
    --model-dir   av_fusion/av_results/{run_name}/models/ \
    --output-dir  av_fusion/av_results/{run_name}/ \
    [--cascade-breakdown  av_fusion/av_results/{run_name}/cascade_stage_breakdown.csv] \
    [--smoothed-predictions  av_fusion/av_results/{run_name}/predictions_test_smoothed.csv] \
    [--plot]
```

**Added outputs** (beyond 006):
- `metrics_cascade_by_stage.csv` — AUROC/F1 broken down by `cascade_stage` (1/2/3)
- `metrics_smoothed.csv` — metrics for the smoothed probability column vs. raw

---

## 1kd_integration.py

**Purpose**: Check compatibility of a candidate 1kd dataset directory with the existing clip schema and produce a JSON report.

```
python av_fusion/scripts/1kd_integration.py \
    --data-dir     /path/to/1kd/data/ \
    --output       av_fusion/av_results/{run_name}/1kd_integration_report.json \
    [--dry-run]    [check schema only, do not copy any files]
```

**Inputs**:
- `--data-dir`: directory containing 1kd audio/video files and an annotation CSV

**Outputs**:
- JSON report: `{"status": "compatible"|"incompatible"|"not_found", "n_clips": N, "missing_columns": [...], "age_range_overlap": [...], "notes": "..."}`
- If incompatible or not found: exits with code 0 (not a crash), writes report documenting the gap

---

## Config: av_extensions.yaml

```yaml
# av_fusion/configs/av_extensions.yaml
seed: 42

asd_models:
  loconet:
    checkpoint: ""  # set at runtime or in env
    batch_size: 16
    device: cuda
  light_asd:
    checkpoint: ""
    batch_size: 32
    device: cuda

gpt4o:
  model: gpt-4o-mini
  sample_rate: 2        # frames per clip
  max_tokens: 256
  temperature: 0.0
  cache_dir: av_fusion/gpt4o_cache/

cascade:
  vad_feature: kchi_total_dur
  child_id_feature: prob       # enrollment score column
  vad_threshold_grid: [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0, 1.5, 2.0]
  child_id_threshold_grid: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

temporal_smoothing:
  default_method: gaussian
  gaussian_bandwidth_grid: [0.5, 1.0, 1.5, 2.0, 3.0]
  majority_vote_window_grid: [3, 5, 7]
  moving_average_window_grid: [3, 5, 7]
  group_cols: [child_id, timepoint_norm]
```
