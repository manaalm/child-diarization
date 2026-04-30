#!/bin/bash
#SBATCH -c 4
#SBATCH -t 03:00:00
#SBATCH -p ou_bcs_normal,pi_satra
#SBATCH --mem=24G
#SBATCH --requeue
#SBATCH --gres=gpu:1
#SBATCH -o logs/pseudo_frame/per_track_%j.out
#SBATCH -e logs/pseudo_frame/per_track_%j.err

set -euo pipefail

source /orcd/home/002/manaal/miniforge3/etc/profile.d/conda.sh
conda activate child-vocalizations

REPO=/orcd/scratch/orcd/008/manaal/child-adult-diarization
cd "$REPO"
mkdir -p logs/pseudo_frame

echo "=== per-track speaker score (job $SLURM_JOB_ID) ==="
python pseudo_frame/per_track_speaker_score.py --device cuda
echo "=== Done ==="
