#!/bin/bash
#SBATCH -J babar_three_bids
#SBATCH -c 2
#SBATCH -t 6:00:00
#SBATCH -p ou_bcs_normal,pi_satra
#SBATCH --mem=40G
#SBATCH --requeue
#SBATCH --gres=gpu:1
#SBATCH -o logs/combined_bids_%j.out
#SBATCH -e logs/combined_bids_%j.err

# BabAR feature-fusion retrain on the BIDS-corrected splits (spec-022 US1).
# Outputs to babar_combined_runs_bids/ to preserve the legacy n=441 results.
# Re-extracts features (no --skip-extraction) because the cached CSVs in
# babar_combined_runs/ are at n=441 legacy.

set -euo pipefail

source /orcd/home/002/manaal/miniforge3/etc/profile.d/conda.sh
conda activate child-vocalizations
module load ffmpeg/5.1.4
export LD_LIBRARY_PATH="/orcd/software/community/001/spack/pkg/ffmpeg/5.1.4/6kcopsg/lib:${LD_LIBRARY_PATH:-}"

REPO=/orcd/scratch/orcd/008/manaal/child-adult-diarization
cd "$REPO/pyannote"
mkdir -p "$REPO/babar_combined_runs_bids" "$REPO/logs"

echo "=== BabAR feature-fusion BIDS retrain (job $SLURM_JOB_ID) ==="
python babar_three.py \
    --babar-output "$REPO/babar/babar_output" \
    --results-dir "$REPO/babar_combined_runs_bids"
echo "=== Done at $(date) ==="
