#!/bin/bash
#SBATCH --job-name=parakeet_asr
#SBATCH --gres=gpu:1
#SBATCH -t 4:00:00
#SBATCH --mem=48G
#SBATCH -c 4
#SBATCH -p ou_bcs_normal,pi_satra
#SBATCH -o logs/baselines/parakeet_%j.out
#SBATCH -e logs/baselines/parakeet_%j.out

# Usage:
#   sbatch baselines/slurm/run_parakeet_baseline.sh val
#   sbatch baselines/slurm/run_parakeet_baseline.sh test

SPLIT=${1:-val}

echo "Start: $(date)"
echo "SPLIT=${SPLIT}"
echo "Node: $(hostname)  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo none)"

source /home/manaal/miniforge3/etc/profile.d/conda.sh
conda activate child-vocalizations

export HF_HOME=/orcd/scratch/orcd/008/manaal/.cache/huggingface

cd /orcd/scratch/orcd/008/manaal/child-adult-diarization
mkdir -p logs/baselines

python baselines/parakeet_baseline.py \
    --split "${SPLIT}" \
    --batch-size 32 \
    --seed 42

echo "Done: $(date)"
