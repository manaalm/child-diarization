#!/bin/bash
#SBATCH -J baseline_seen_child
#SBATCH -p ou_bcs_normal,pi_satra
#SBATCH -t 48:00:00
#SBATCH -c 4
#SBATCH --mem=40G
#SBATCH --requeue
#SBATCH --gres=gpu:1
#SBATCH -o logs/baselines/baseline_seen_child_%j.out
#SBATCH -e logs/baselines/baseline_seen_child_%j.err

set -euo pipefail

source /home/manaal/miniforge3/etc/profile.d/conda.sh
conda activate child-vocalizations

export HF_HOME=/orcd/scratch/orcd/008/manaal/.cache/huggingface

cd /orcd/scratch/orcd/008/manaal/child-adult-diarization

mkdir -p logs/baselines baselines/baseline_results_seen_child

echo "Start: $(date)"
python baselines/baseline_encoders.py --seen-child --all-experiments
echo "Done: $(date)"
