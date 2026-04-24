#!/bin/bash
#SBATCH -t 48:00:00
#SBATCH --gres=gpu:1
#SBATCH -p ou_bcs_normal,pi_satra
#SBATCH --mem=40G
#SBATCH -c 4
#SBATCH -o logs/av_fusion/av_pipeline_%j.out
#SBATCH -e logs/av_fusion/av_pipeline_%j.err

set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
export PYTHONUNBUFFERED=1
source /orcd/home/002/manaal/miniforge3/etc/profile.d/conda.sh
conda activate child-vocalizations

REPO=/home/manaal/orcd/scratch/child-adult-diarization
RUN=${RUN_NAME:-auto_visual}
OUTDIR="$REPO/av_fusion/av_results/${RUN}"

mkdir -p "$REPO/logs/av_fusion" "$OUTDIR"

echo "=== AV Pipeline: visual feature extraction ==="
echo "  Run name: ${RUN}"
echo "  Output dir: ${OUTDIR}"

cd "$REPO"

# Step 1: Extract automatic visual features from BIDS video files
echo "=== Step 1: Extract visual features ==="
python av_fusion/scripts/extract_visual_features.py \
    --metadata-csv whisper-modeling/seen_child_splits/master_with_split.csv \
    --output "${OUTDIR}/visual_features.csv" \
    --face-cache-dir av_fusion/face_track_cache \
    --sample-fps 2 \
    --detector yunet \
    --workers 4

echo "=== Step 1 complete: ${OUTDIR}/visual_features.csv ==="
echo "Run the remaining steps (build, train, evaluate) on the login node:"
echo "  RUN=${RUN} bash av_fusion/slurm/run_remaining_steps.sh"
