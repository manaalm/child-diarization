# Quickstart: AV Extended Experiments — 007-av-extensions

**Date**: 2026-04-24  
**Prerequisites**: 006 pipeline complete — `av_fusion/av_results/{run_name}/av_{train,val,test}.csv` must exist; BabAR enrollment predictions in `babar_ecapa_enrollment_runs/enroll_{val,test}_predictions.csv`

---

## Story 1: Cascaded Detection Pipeline

Run the three-stage cascade on the existing feature table (no new feature extraction required):

```bash
# Step 1: Tune cascade thresholds on val set
python av_fusion/scripts/train_cascaded_pipeline.py \
    --feature-dir   av_fusion/av_results/manual_only/ \
    --output-dir    av_fusion/av_results/manual_only/models/ \
    --vad-feature   kchi_total_dur \
    --child-id-feature prob

# Expected output:
# av_fusion/av_results/manual_only/models/cascade_thresholds.json
# av_fusion/av_results/manual_only/cascade_val_stage_breakdown.csv

# Step 2: Evaluate cascade on test set (extends existing evaluate_av_fusion.py)
python av_fusion/scripts/evaluate_av_fusion.py \
    --feature-dir   av_fusion/av_results/manual_only/ \
    --model-dir     av_fusion/av_results/manual_only/models/ \
    --output-dir    av_fusion/av_results/manual_only/ \
    --cascade-breakdown av_fusion/av_results/manual_only/cascade_stage_breakdown.csv \
    --plot

# Key outputs to check:
# cascade_stage_breakdown.csv  → which stage decided each clip
# metrics_cascade_by_stage.csv → AUROC/F1 per stage
```

**Validation**: Confirm `cascade_thresholds.json` exists and `metrics_overall.json` contains `cascaded_av` row with non-NaN AUROC.

---

## Story 2: Temporal Smoothing

Apply Gaussian smoothing post-hoc to any model's predictions:

```bash
# Smooth test predictions (bandwidth auto-tuned on val)
python av_fusion/scripts/smooth_predictions.py \
    --predictions     av_fusion/av_results/manual_only/predictions_test.csv \
    --val-predictions av_fusion/av_results/manual_only/predictions_val.csv \
    --output          av_fusion/av_results/manual_only/predictions_test_smoothed.csv \
    --method          gaussian \
    --param           None \
    --group-cols      child_id,timepoint_norm

# Try majority vote instead:
python av_fusion/scripts/smooth_predictions.py \
    --predictions     av_fusion/av_results/manual_only/predictions_test.csv \
    --val-predictions av_fusion/av_results/manual_only/predictions_val.csv \
    --output          av_fusion/av_results/manual_only/predictions_test_smoothed_mv.csv \
    --method          majority_vote
```

**Validation**: Output CSV has `prob_smoothed` column; script prints raw vs. smoothed val F1.

---

## Story 3: GPT-4o Vision Features

First run dry-run to check estimated cost, then extract:

```bash
export OPENAI_API_KEY=<your key>

# Cost estimate (no API calls made)
python av_fusion/scripts/extract_gpt4o_features.py \
    --metadata-csv whisper-modeling/seen_child_splits/master_with_split.csv \
    --output       av_fusion/av_results/manual_only/gpt4o_features.csv \
    --model        gpt-4o-mini \
    --sample-rate  2 \
    --dry-run

# Full extraction (idempotent — re-running skips cached clips)
python av_fusion/scripts/extract_gpt4o_features.py \
    --metadata-csv whisper-modeling/seen_child_splits/master_with_split.csv \
    --output       av_fusion/av_results/manual_only/gpt4o_features.csv \
    --model        gpt-4o-mini \
    --sample-rate  2
```

**To test on a small subset first** (cap spending):
```bash
python av_fusion/scripts/extract_gpt4o_features.py \
    --metadata-csv whisper-modeling/seen_child_splits/master_with_split.csv \
    --output       av_fusion/av_results/manual_only/gpt4o_features.csv \
    --max-clips    50
```

**Validation**: `gpt4o_features.csv` has columns `child_visible_gpt4o`, `child_vocalizing_gpt4o`, `visual_quality_gpt4o`, `gpt4o_reasoning`; NaN rows for audio-only clips.

---

## Story 4: LocoNet and Light-ASD Frontends

```bash
# LocoNet (requires checkpoint download — see research.md Decision 1)
python av_fusion/scripts/extract_asd_features.py \
    --metadata-csv        whisper-modeling/seen_child_splits/master_with_split.csv \
    --output              av_fusion/av_results/manual_only/asd_features_loconet.csv \
    --model               loconet \
    --loconet-checkpoint  /path/to/loconet_best.ckpt \
    --device              cuda

# Light-ASD (checkpoint bundled with repo — see research.md Decision 2)
python av_fusion/scripts/extract_asd_features.py \
    --metadata-csv          whisper-modeling/seen_child_splits/master_with_split.csv \
    --output                av_fusion/av_results/manual_only/asd_features_light_asd.csv \
    --model                 light_asd \
    --light-asd-checkpoint  video/Light-ASD/weight/pretrain_AVA_CVPR22.pt \
    --device                cuda
```

**Checkpoint setup**:
- LocoNet: `huggingface-cli download Superxixixi/LoCoNet_ASD --local-dir video/LoCoNet_ASD/`
- Light-ASD: `git clone https://github.com/Junhua-Liao/Light-ASD video/Light-ASD`

**Validation**: Both CSVs have same schema as `asd_features_talknet.csv`; one row per clip.

---

## Story 5: Ego4D Reference Experiment (Optional)

```bash
# Register at ego4d-data.org and get CLI token
pip install ego4d
ego4d --output_directory /path/to/ego4d/ --datasets full_scale --benchmarks AV

# Zero-shot evaluate TalkNet on Ego4D AVD val subset (50 clips)
python av_fusion/scripts/extract_asd_features.py \
    --metadata-csv /path/to/ego4d/av_val_metadata.csv \
    --output       av_fusion/av_results/ego4d_eval/asd_features_talknet_ego4d.csv \
    --model        talknet

# Document results in ego4d_experiment_results.csv
```

**If Ego4D access is not available**: Document the access requirements and skip this story.

---

## Story 6: 1kd Dataset Compatibility Check

```bash
# Schema compatibility check (does not require actual data — exits gracefully if path missing)
python av_fusion/scripts/1kd_integration.py \
    --data-dir /path/to/1kd/data/ \
    --output   av_fusion/av_results/manual_only/1kd_integration_report.json \
    --dry-run

# Expected JSON output (even if data not found):
# {"status": "not_found", "n_clips": 0, "missing_columns": [], "notes": "..."}
```

---

## Full Pipeline (all stories, in order)

```bash
RUN=manual_only

# 1. Build feature table (006 step — must exist first)
# python av_fusion/scripts/build_av_feature_table.py ...

# 2. Train 006 fusion models (must exist first)
# python av_fusion/scripts/train_av_fusion.py ...

# 3. Extract GPT-4o features (Story 3)
python av_fusion/scripts/extract_gpt4o_features.py \
    --metadata-csv whisper-modeling/seen_child_splits/master_with_split.csv \
    --output av_fusion/av_results/${RUN}/gpt4o_features.csv

# 4. Extract LocoNet / Light-ASD features (Story 4)
python av_fusion/scripts/extract_asd_features.py --model loconet \
    --metadata-csv whisper-modeling/seen_child_splits/master_with_split.csv \
    --output av_fusion/av_results/${RUN}/asd_features_loconet.csv
python av_fusion/scripts/extract_asd_features.py --model light_asd \
    --metadata-csv whisper-modeling/seen_child_splits/master_with_split.csv \
    --output av_fusion/av_results/${RUN}/asd_features_light_asd.csv

# 5. Tune and evaluate cascade (Story 1)
python av_fusion/scripts/train_cascaded_pipeline.py \
    --feature-dir av_fusion/av_results/${RUN}/ \
    --output-dir  av_fusion/av_results/${RUN}/models/

# 6. Evaluate all models including cascade (extends 006 evaluation)
python av_fusion/scripts/evaluate_av_fusion.py \
    --feature-dir av_fusion/av_results/${RUN}/ \
    --model-dir   av_fusion/av_results/${RUN}/models/ \
    --output-dir  av_fusion/av_results/${RUN}/ \
    --cascade-breakdown av_fusion/av_results/${RUN}/cascade_stage_breakdown.csv \
    --plot

# 7. Temporal smoothing on best model predictions (Story 2)
python av_fusion/scripts/smooth_predictions.py \
    --predictions     av_fusion/av_results/${RUN}/predictions_test.csv \
    --val-predictions av_fusion/av_results/${RUN}/predictions_val.csv \
    --output          av_fusion/av_results/${RUN}/predictions_test_smoothed.csv

# 8. Check 1kd compatibility (Story 6)
python av_fusion/scripts/1kd_integration.py \
    --data-dir /path/to/1kd/ \
    --output   av_fusion/av_results/${RUN}/1kd_integration_report.json
```
