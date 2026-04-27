#!/bin/bash
#SBATCH -J av_extensions
#SBATCH -t 1:00:00
#SBATCH -p ou_bcs_normal,pi_satra
#SBATCH --mem=16G
#SBATCH -c 4
#SBATCH -o ../logs/av_fusion/av_extensions_%j.out
#SBATCH -e ../logs/av_fusion/av_extensions_%j.err

# 007-av-extensions pipeline steps (CPU-only, no GPU needed):
#   Step 5: cascade threshold tuning on val
#   Step 6a: re-evaluate with cascade flag
#   Step 6b: generate val predictions (for smoothing)
#   Step 7: temporal smoothing (auto-tunes bandwidth on val)
#
# Prerequisites: run_av_fusion_mvp.sh must have completed first.
# Results in: av_fusion/av_results/manual_only/

set -euo pipefail
export PYTHONUNBUFFERED=1
source /orcd/home/002/manaal/miniforge3/etc/profile.d/conda.sh
conda activate child-vocalizations

REPO=/orcd/scratch/orcd/008/manaal/child-adult-diarization
cd "$REPO"
mkdir -p logs/av_fusion

FEAT=av_fusion/av_results/manual_only
MODELS=$FEAT/models

echo "=== Step 5: Cascade threshold tuning (val) ==="
python av_fusion/scripts/train_cascaded_pipeline.py \
    --feature-dir "$FEAT" \
    --output-dir  "$MODELS"

echo "=== Step 6a: Re-evaluate with cascade breakdown ==="
python av_fusion/scripts/evaluate_av_fusion.py \
    --feature-dir "$FEAT" \
    --model-dir   "$MODELS" \
    --output-dir  "$FEAT" \
    --cascade-breakdown "$FEAT/cascade_stage_breakdown.csv" \
    --eval-val \
    --plot

echo "=== Step 7: Temporal smoothing ==="
python av_fusion/scripts/smooth_predictions.py \
    --predictions     "$FEAT/predictions_test.csv" \
    --val-predictions "$FEAT/predictions_val.csv" \
    --output          "$FEAT/predictions_test_smoothed.csv" \
    --method gaussian

echo "=== Step 6b: Re-evaluate with smoothed predictions ==="
python av_fusion/scripts/evaluate_av_fusion.py \
    --feature-dir "$FEAT" \
    --model-dir   "$MODELS" \
    --output-dir  "$FEAT" \
    --cascade-breakdown "$FEAT/cascade_stage_breakdown.csv" \
    --smoothed-predictions "$FEAT/predictions_test_smoothed.csv"

echo "Done. Results: $FEAT"
