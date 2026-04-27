# CLI Contracts: Audio-Visual Fusion Pipeline

**Feature**: 006-av-child-vocalization  
**Date**: 2026-04-24

All scripts live under `av_fusion/scripts/`. All paths may be absolute or relative to repo root.

---

## Script 1: `extract_visual_features.py`

**Purpose**: Run face detection and tracking on all clips in a metadata CSV; write per-clip visual feature rows.

```
python av_fusion/scripts/extract_visual_features.py \
    --metadata-csv   whisper-modeling/seen_child_splits/master_with_split.csv \
    --output         av_fusion/av_results/{run}/visual_features.csv \
    --sample-fps     2 \
    [--detector      yunet|mediapipe]              # default: yunet
    [--face-cache-dir av_fusion/face_track_cache/] # cache per-clip detection results
    [--workers       4]                            # parallel workers
```

**Inputs**:
- `--metadata-csv`: CSV with at minimum `clip_id`, `video_path` (or `BidsProcessed`) columns.

**Outputs**:
- `visual_features.csv` with one row per clip. Schema: see `AutomaticVisualFeatures` in data-model.md.
- If `--face-cache-dir` is set, per-clip detection JSON files are cached there keyed by clip_id.

**Exit codes**:
- 0: success (all clips processed, including those with missing video)
- 1: input file missing or unreadable
- 2: output directory not writable

**Guarantees**:
- Every clip_id from the input CSV appears in the output (no silent drops).
- Clips with missing/unreadable video have NaN feature values and `off_camera_likely_score = 1.0`.
- Script is idempotent: re-running with an existing cache skips already-processed clips.

---

## Script 2: `extract_asd_features.py` (Optional)

**Purpose**: Run TalkNet-ASD on all clips and write per-clip ASD scores.

```
python av_fusion/scripts/extract_asd_features.py \
    --metadata-csv   whisper-modeling/seen_child_splits/master_with_split.csv \
    --output         av_fusion/av_results/{run}/asd_features.csv \
    [--asd-model     talknet_asd|light_asd]        # default: talknet_asd
    [--visual-features-csv av_fusion/av_results/{run}/visual_features.csv]
    [--workers       1]
```

**Inputs**:
- `--metadata-csv`: Same metadata CSV used by `extract_visual_features.py`.
- `--visual-features-csv` (optional): If provided, skips clips where `n_face_tracks == 0`.

**Outputs**:
- `asd_features.csv` with one row per clip. Schema: see `ASDFeatures` in data-model.md.

**Exit codes**:
- 0: success
- 1: ASD model checkpoint not found
- 2: video directory not accessible

**Guarantees**:
- Clips with no detected faces have all ASD scores = 0.0.
- Script is idempotent with caching.

---

## Script 3: `build_av_feature_table.py`

**Purpose**: Merge metadata, labels, split, audio baseline scores, manual annotations, visual features, and (optionally) ASD features into a single master feature table. Assert split integrity.

```
python av_fusion/scripts/build_av_feature_table.py \
    --metadata-csv        whisper-modeling/seen_child_splits/master_with_split.csv \
    --audio-scores-csv    <path-to-audio-baseline-predictions.csv> \
    --audio-score-col     enroll_proba \
    --output-dir          av_fusion/av_results/{run}/ \
    [--visual-features-csv  av_fusion/av_results/{run}/visual_features.csv] \
    [--asd-features-csv     av_fusion/av_results/{run}/asd_features.csv] \
    [--run-name             {run}]
```

**Inputs**:
- `--metadata-csv`: Master split CSV (contains manual annotations, labels, splits, video paths).
- `--audio-scores-csv`: CSV with `clip_id` and at least one probability column from an existing audio baseline.
- `--audio-score-col`: Column name for the audio probability score (default: `enroll_proba`).
- `--visual-features-csv` (optional): Output of `extract_visual_features.py`.
- `--asd-features-csv` (optional): Output of `extract_asd_features.py`.

**Outputs** (all in `--output-dir`):
- `av_master_features.csv`: Full merged table (all clips, all splits).
- `av_train.csv`, `av_val.csv`, `av_test.csv`: Per-split subsets.
- `feature_manifest.json`: List of feature column names used, with source (manual/automatic/asd/audio).
- `split_integrity_report.json`: Verification that no child_id spans multiple splits.

**Exit codes**:
- 0: success, split integrity verified
- 1: split integrity violation detected (ERROR: stops execution)
- 2: required input CSV missing

**Guarantees**:
- If `--visual-features-csv` is not provided, visual feature columns are NaN and `visual_eligible = 0` for all clips.
- If `--asd-features-csv` is not provided, ASD feature columns are NaN.
- Clips with missing audio scores are included with NaN audio score.

---

## Script 4: `train_av_fusion.py`

**Purpose**: Train all four fusion model classes on the train split; tune thresholds on the val split; save models and thresholds.

```
python av_fusion/scripts/train_av_fusion.py \
    --feature-dir    av_fusion/av_results/{run}/ \
    --output-dir     av_fusion/av_results/{run}/models/ \
    --config         av_fusion/configs/av_fusion.yaml \
    [--seed          42] \
    [--no-asd]                    # skip ASD features even if present
    [--models        audio_only,video_only,always_fuse,gated_av]  # default: all four
```

**Inputs**:
- `--feature-dir`: Directory containing `av_train.csv`, `av_val.csv` (from `build_av_feature_table.py`).
- `--config`: YAML with model hyperparameters per model class.

**Outputs** (all in `--output-dir`):
- `audio_only.pkl`
- `video_only.pkl`
- `always_fuse_av.pkl`
- `gated_av.pkl`
- `visual_eligibility_threshold.json`: `{"threshold": float, "val_balanced_acc": float}`
- `val_metrics.json`: Validation-set metrics for all four models post-threshold-tuning.
- `config.json`: Full experiment config (model HPs, feature columns, seed, split paths).

**Exit codes**:
- 0: success
- 1: feature CSV not found
- 2: no training examples after filtering

**Guarantees**:
- No test data is loaded or used during training or threshold tuning.
- `visual_eligibility_threshold` is tuned by maximizing balanced accuracy of the eligibility gate on val, not by optimizing vocalization F1.
- Each model pkl includes the fitted scaler/preprocessor for inference.

---

## Script 5: `evaluate_av_fusion.py`

**Purpose**: Evaluate all four trained models on the held-out test split; produce full stratified metrics.

```
python av_fusion/scripts/evaluate_av_fusion.py \
    --feature-dir    av_fusion/av_results/{run}/ \
    --model-dir      av_fusion/av_results/{run}/models/ \
    --output-dir     av_fusion/av_results/{run}/ \
    [--models        audio_only,video_only,always_fuse,gated_av]
    [--plot]                         # generate PNG figures
```

**Inputs**:
- `--feature-dir`: Directory containing `av_test.csv`.
- `--model-dir`: Directory with trained pkl files and `visual_eligibility_threshold.json`.

**Outputs** (all in `--output-dir`):
- `metrics_overall.json`
- `metrics_by_age_band.csv`
- `metrics_by_visual_eligibility.csv`
- `metrics_by_strata.csv` (off-camera likely, multi-person, low-quality, high-audio-uncertainty)
- `predictions_test.csv`
- `figures/pr_curve.png` (if `--plot`)
- `figures/roc_curve.png` (if `--plot`)
- `figures/stratified_bar_metrics.png` (if `--plot`)
- `figures/visual_eligibility_histogram.png` (if `--plot`)

**Exit codes**:
- 0: success
- 1: model pkl not found
- 2: test feature CSV not found

**Guarantees**:
- Test set is loaded only by this script; no thresholds are changed here.
- All four models are evaluated with their val-tuned thresholds.
- Strata with < 10 test clips are flagged with `n_clips` count for caution.

---

## Script 6: `error_analysis_av.py`

**Purpose**: Produce a structured error-mode breakdown comparing audio-only and AV predictions.

```
python av_fusion/scripts/error_analysis_av.py \
    --predictions-csv   av_fusion/av_results/{run}/predictions_test.csv \
    --feature-dir       av_fusion/av_results/{run}/ \
    --output-dir        av_fusion/av_results/{run}/ \
    [--n-examples       20]    # number of examples per error category
```

**Inputs**:
- `--predictions-csv`: Output of `evaluate_av_fusion.py`; has clip-level predictions for all models.

**Outputs**:
- `error_analysis_examples.csv`: Up to `n_examples` clips per error-mode category.
- `error_analysis_summary.json`: Counts and aggregate metrics per error mode.

**Error mode categories**:
- `av_helped_fp`: audio-only false positive, AV correctly negative
- `av_helped_fn`: audio-only false negative, AV correctly positive
- `av_hurt_fp`: audio-only correct, AV introduces false positive (high `off_camera_likely_score` expected)
- `av_hurt_fn`: audio-only correct, AV introduces false negative
- `off_camera_miss`: ground truth positive, no face detected, AV model wrong
- `multi_face_ambiguous`: multiple faces detected, neither model confident

**Exit codes**:
- 0: success (even if some categories have 0 examples)
- 1: predictions CSV not found
