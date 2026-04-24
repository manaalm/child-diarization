#!/bin/bash
#SBATCH -t 1:00:00
#SBATCH -p ou_bcs_normal,pi_satra
#SBATCH --mem=16G
#SBATCH -c 4
#SBATCH -o logs/av_fusion/av_mvp_%j.out
#SBATCH -e logs/av_fusion/av_mvp_%j.err

# AV fusion MVP — manual BIDS annotations only, no video extraction required.
# Steps: build feature table → train → evaluate → error analysis

set -euo pipefail
export PYTHONUNBUFFERED=1
source /orcd/home/002/manaal/miniforge3/etc/profile.d/conda.sh
conda activate child-vocalizations

REPO=/orcd/scratch/orcd/008/manaal/child-adult-diarization
cd "$REPO"
mkdir -p logs/av_fusion

echo "=== Step 1: Build feature table ==="
python av_fusion/scripts/build_av_feature_table.py \
    --metadata-csv  whisper-modeling/seen_child_splits/master_with_split.csv \
    --audio-scores-val  babar_ecapa_enrollment_runs/enroll_val_predictions.csv \
    --audio-scores-test babar_ecapa_enrollment_runs/enroll_test_predictions.csv \
    --audio-score-col prob \
    --output-dir  av_fusion/av_results/manual_only/ \
    --run-name    manual_only

echo "=== Step 2: Train fusion models ==="
python av_fusion/scripts/train_av_fusion.py \
    --feature-dir av_fusion/av_results/manual_only/ \
    --output-dir  av_fusion/av_results/manual_only/models/ \
    --config      av_fusion/configs/av_fusion.yaml \
    --seed 42

echo "=== Step 3: Evaluate on test ==="
python av_fusion/scripts/evaluate_av_fusion.py \
    --feature-dir av_fusion/av_results/manual_only/ \
    --model-dir   av_fusion/av_results/manual_only/models/ \
    --output-dir  av_fusion/av_results/manual_only/ \
    --plot

echo "=== Step 4: Error analysis ==="
python av_fusion/scripts/error_analysis_av.py \
    --predictions-csv av_fusion/av_results/manual_only/predictions_test.csv \
    --feature-dir     av_fusion/av_results/manual_only/ \
    --output-dir      av_fusion/av_results/manual_only/

echo "Done. Results: av_fusion/av_results/manual_only/"
