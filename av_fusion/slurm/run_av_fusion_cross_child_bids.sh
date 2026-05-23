#!/bin/bash
#SBATCH -t 2:00:00
#SBATCH -p ou_bcs_normal,pi_satra
#SBATCH --mem=24G
#SBATCH -c 4
#SBATCH -o logs/av_fusion/av_cross_child_bids_%j.out
#SBATCH -e logs/av_fusion/av_cross_child_bids_%j.err

# AV fusion (manual visibility) on the BIDS-corrected cross-child split.
# The default within-speaker pipeline uses BabAR enrollment as the audio
# pillar, which is structurally undefined cross-speaker (per-(child,
# timepoint) prototypes have no train support). For cross-child we
# substitute the BIDS-corrected BabAR role-only predictions produced in
# evaluation/cross_child_babar_role_only_bids/ this revision.

set -euo pipefail
export PYTHONUNBUFFERED=1
source /orcd/home/002/manaal/miniforge3/etc/profile.d/conda.sh
conda activate child-vocalizations

REPO=/orcd/scratch/orcd/008/manaal/child-adult-diarization
cd "$REPO"
mkdir -p logs/av_fusion

OUT=av_fusion/av_results/manual_only_cross_child_bids
mkdir -p "$OUT/models"

echo "=== AV fusion cross-child BIDS (job $SLURM_JOB_ID) ==="
echo "Start: $(date)"

echo "--- Step 1: Build feature table ---"
python av_fusion/scripts/build_av_feature_table.py \
    --metadata-csv  baselines/splits/master_with_split.csv \
    --audio-scores-val  evaluation/cross_child_babar_role_only_bids/val_predictions.csv \
    --audio-scores-test evaluation/cross_child_babar_role_only_bids/test_predictions.csv \
    --audio-score-col prob \
    --output-dir  "$OUT/" \
    --run-name    manual_only_cross_child_bids

echo "--- Step 2: Train fusion models ---"
python av_fusion/scripts/train_av_fusion.py \
    --feature-dir "$OUT/" \
    --output-dir  "$OUT/models/" \
    --config      av_fusion/configs/av_fusion.yaml \
    --seed 42

echo "--- Step 3: Evaluate on test ---"
python av_fusion/scripts/evaluate_av_fusion.py \
    --feature-dir "$OUT/" \
    --model-dir   "$OUT/models/" \
    --output-dir  "$OUT/" \
    --plot

echo "--- Step 4: Error analysis ---"
python av_fusion/scripts/error_analysis_av.py \
    --predictions-csv "$OUT/predictions_test.csv" \
    --feature-dir     "$OUT/" \
    --output-dir      "$OUT/"

echo "Done: $(date). Results: $OUT/"
