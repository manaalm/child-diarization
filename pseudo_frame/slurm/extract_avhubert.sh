#!/bin/bash
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH -t 8:00:00
#SBATCH -p ou_bcs_normal,pi_satra
#SBATCH --mem=32G
#SBATCH --requeue
#SBATCH -o logs/pseudo_frame/extract_avhubert_%j.out
#SBATCH -e logs/pseudo_frame/extract_avhubert_%j.err

set -euo pipefail

source /orcd/home/002/manaal/miniforge3/etc/profile.d/conda.sh
conda activate avhubert

REPO=/orcd/scratch/orcd/008/manaal/child-adult-diarization
cd "$REPO"
mkdir -p logs/pseudo_frame

# transformers >=4.57 has_file() bug guard (carry-over from spec-016 lesson)
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

echo "=== Extracting AV-HuBERT-Large embeddings (job $SLURM_JOB_ID) ==="
python pseudo_frame/extract_avhubert_embeddings.py --all
echo "=== Done ==="
