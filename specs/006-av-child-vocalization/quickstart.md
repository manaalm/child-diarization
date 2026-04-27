# Quickstart: Audio-Visual Fusion Pipeline

**Feature**: 006-av-child-vocalization  
**Date**: 2026-04-24

This guide shows how to run the full AV fusion pipeline end-to-end, from existing split CSVs to stratified evaluation results.

---

## Prerequisites

- Conda env `child-vocalizations` active (same as MIL sweep).
- BabAR enrollment results available at `babar_ecapa_enrollment_runs/` (provides audio baseline scores).
- SAILS BIDS preprocessed video files accessible at paths in `BidsProcessed` column of split CSV.
- Repo root is the working directory.

Install any new dependencies:
```bash
conda activate child-vocalizations
pip install opencv-python xgboost  # if not already present
```

---

## Scenario 1: Fast-Track MVP (Manual Annotations Only, No Video Processing)

This is the quickest path to a trained and evaluated AV fusion model. Uses only the manual BIDS annotations already in the split CSV — no face detection required.

```bash
RUN=manual_only
OUTDIR=av_fusion/av_results/${RUN}
mkdir -p ${OUTDIR}

# Step 1: Build master feature table (no visual features CSV → manual annotations only)
python av_fusion/scripts/build_av_feature_table.py \
    --metadata-csv whisper-modeling/seen_child_splits/master_with_split.csv \
    --audio-scores-csv babar_ecapa_enrollment_runs/test_predictions.csv \
    --audio-score-col enroll_proba \
    --output-dir ${OUTDIR} \
    --run-name ${RUN}

# Step 2: Train four model classes
python av_fusion/scripts/train_av_fusion.py \
    --feature-dir ${OUTDIR} \
    --output-dir ${OUTDIR}/models \
    --config av_fusion/configs/av_fusion.yaml \
    --seed 42

# Step 3: Evaluate on test split
python av_fusion/scripts/evaluate_av_fusion.py \
    --feature-dir ${OUTDIR} \
    --model-dir ${OUTDIR}/models \
    --output-dir ${OUTDIR} \
    --plot

# Step 4: Error analysis
python av_fusion/scripts/error_analysis_av.py \
    --predictions-csv ${OUTDIR}/predictions_test.csv \
    --feature-dir ${OUTDIR} \
    --output-dir ${OUTDIR}
```

**Expected outputs** (all under `av_fusion/av_results/manual_only/`):
- `av_master_features.csv`, `av_train.csv`, `av_val.csv`, `av_test.csv`
- `models/audio_only.pkl`, `models/video_only.pkl`, `models/always_fuse_av.pkl`, `models/gated_av.pkl`
- `models/val_metrics.json`, `models/config.json`
- `metrics_overall.json`, `metrics_by_age_band.csv`, `metrics_by_visual_eligibility.csv`
- `predictions_test.csv`, `error_analysis_examples.csv`
- `figures/pr_curve.png`, `figures/roc_curve.png`, `figures/stratified_bar_metrics.png`

---

## Scenario 2: Full Pipeline (Manual Annotations + Automatic Visual Features)

Adds face detection and tracking on video frames. Requires GPU (or runs slowly on CPU for large clip sets).

```bash
RUN=manual_plus_auto
OUTDIR=av_fusion/av_results/${RUN}
mkdir -p ${OUTDIR}

# Step 1: Extract automatic visual features from video frames
python av_fusion/scripts/extract_visual_features.py \
    --metadata-csv whisper-modeling/seen_child_splits/master_with_split.csv \
    --output ${OUTDIR}/visual_features.csv \
    --face-cache-dir av_fusion/face_track_cache \
    --sample-fps 2 \
    --detector yunet

# Steps 2–5: Same as Scenario 1 but pass --visual-features-csv
python av_fusion/scripts/build_av_feature_table.py \
    --metadata-csv whisper-modeling/seen_child_splits/master_with_split.csv \
    --audio-scores-csv babar_ecapa_enrollment_runs/test_predictions.csv \
    --audio-score-col enroll_proba \
    --visual-features-csv ${OUTDIR}/visual_features.csv \
    --output-dir ${OUTDIR} \
    --run-name ${RUN}

python av_fusion/scripts/train_av_fusion.py \
    --feature-dir ${OUTDIR} \
    --output-dir ${OUTDIR}/models \
    --config av_fusion/configs/av_fusion.yaml \
    --seed 42

python av_fusion/scripts/evaluate_av_fusion.py \
    --feature-dir ${OUTDIR} \
    --model-dir ${OUTDIR}/models \
    --output-dir ${OUTDIR} \
    --plot

python av_fusion/scripts/error_analysis_av.py \
    --predictions-csv ${OUTDIR}/predictions_test.csv \
    --feature-dir ${OUTDIR} \
    --output-dir ${OUTDIR}
```

---

## Scenario 3: Full Pipeline + ASD Features (Stretch)

Add TalkNet-ASD scores on top of Scenario 2. Requires `video/` env and TalkNet-ASD checkpoint.

```bash
RUN=full_with_asd
OUTDIR=av_fusion/av_results/${RUN}
mkdir -p ${OUTDIR}

# Step 1: Automatic visual features (same as Scenario 2)
python av_fusion/scripts/extract_visual_features.py \
    --metadata-csv whisper-modeling/seen_child_splits/master_with_split.csv \
    --output ${OUTDIR}/visual_features.csv \
    --face-cache-dir av_fusion/face_track_cache \
    --sample-fps 2

# Step 2: ASD features
python av_fusion/scripts/extract_asd_features.py \
    --metadata-csv whisper-modeling/seen_child_splits/master_with_split.csv \
    --output ${OUTDIR}/asd_features.csv \
    --visual-features-csv ${OUTDIR}/visual_features.csv

# Steps 3–6: Build table with all three feature sources
python av_fusion/scripts/build_av_feature_table.py \
    --metadata-csv whisper-modeling/seen_child_splits/master_with_split.csv \
    --audio-scores-csv babar_ecapa_enrollment_runs/test_predictions.csv \
    --audio-score-col enroll_proba \
    --visual-features-csv ${OUTDIR}/visual_features.csv \
    --asd-features-csv ${OUTDIR}/asd_features.csv \
    --output-dir ${OUTDIR} \
    --run-name ${RUN}

python av_fusion/scripts/train_av_fusion.py \
    --feature-dir ${OUTDIR} \
    --output-dir ${OUTDIR}/models \
    --config av_fusion/configs/av_fusion.yaml \
    --seed 42

python av_fusion/scripts/evaluate_av_fusion.py \
    --feature-dir ${OUTDIR} \
    --model-dir ${OUTDIR}/models \
    --output-dir ${OUTDIR} \
    --plot

python av_fusion/scripts/error_analysis_av.py \
    --predictions-csv ${OUTDIR}/predictions_test.csv \
    --feature-dir ${OUTDIR} \
    --output-dir ${OUTDIR}
```

---

## Scenario 4: SLURM Batch Run (Visual Feature Extraction for All Clips)

Visual feature extraction is embarrassingly parallel; run it as a SLURM job when the cluster is available.

```bash
sbatch av_fusion/slurm/run_av_pipeline.sh
# Logs: logs/av_fusion/av_pipeline_{jobid}.out
# After completion: visual_features.csv is populated; run Scenarios 1 steps 2+ on login node
```

---

## Comparing Results Across Scenarios

```python
import json, pandas as pd

results = {}
for run in ["manual_only", "manual_plus_auto", "full_with_asd"]:
    with open(f"av_fusion/av_results/{run}/metrics_overall.json") as f:
        results[run] = json.load(f)

# Print AUROC per model class per run
for run, metrics in results.items():
    print(f"\n{run}:")
    for model, m in metrics.items():
        print(f"  {model}: AUROC={m['auroc']:.3f}, AUPRC={m['auprc']:.3f}, F1={m['f1']:.3f}")
```

---

## Verifying Split Integrity

The `build_av_feature_table.py` script writes `split_integrity_report.json`. Check it:

```bash
python -c "
import json
with open('av_fusion/av_results/manual_only/split_integrity_report.json') as f:
    r = json.load(f)
print('Leakage detected:', r['leakage_detected'])
print('Children per split:', r['children_per_split'])
"
```

Expected output: `Leakage detected: false`

---

## Quick Sanity Check After Training

```bash
python -c "
import pickle, pandas as pd
m = pickle.load(open('av_fusion/av_results/manual_only/models/gated_av.pkl', 'rb'))
df = pd.read_csv('av_fusion/av_results/manual_only/av_test.csv').head(5)
# Should produce 5 probabilities without error
print(m.predict_proba(df)[:, 1])
"
```
