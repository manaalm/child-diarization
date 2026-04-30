#!/bin/bash
#SBATCH -c 4
#SBATCH -t 6:00:00
#SBATCH -p ou_bcs_normal,pi_satra
#SBATCH --mem=20G
#SBATCH --requeue
#SBATCH -o logs/pseudo_frame/extract_mouth_%j.out
#SBATCH -e logs/pseudo_frame/extract_mouth_%j.err

set -euo pipefail

source /orcd/home/002/manaal/miniforge3/etc/profile.d/conda.sh
conda activate child-vocalizations

REPO=/orcd/scratch/orcd/008/manaal/child-adult-diarization
cd "$REPO"
mkdir -p logs/pseudo_frame

echo "=== Extracting mouth-motion features (job $SLURM_JOB_ID) ==="
python pseudo_frame/extract_mouth_motion.py
echo "=== Done ==="
