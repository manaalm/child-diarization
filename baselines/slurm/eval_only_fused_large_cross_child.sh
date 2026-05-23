#!/bin/bash
#SBATCH -J fused_large_evalonly
#SBATCH -p ou_bcs_normal,pi_satra,mit_normal
#SBATCH -t 1:00:00
#SBATCH -c 4
#SBATCH --mem=80G
#SBATCH --requeue
#SBATCH --gres=gpu:1
#SBATCH -o logs/baselines/fused_large_evalonly_%j.out
#SBATCH -e logs/baselines/fused_large_evalonly_%j.err

set -euo pipefail
source /home/manaal/miniforge3/etc/profile.d/conda.sh
conda activate child-vocalizations
unset HF_TOKEN HUGGINGFACE_HUB_TOKEN HF_HUB_TOKEN 2>/dev/null || true
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export HF_HOME=/orcd/scratch/orcd/008/manaal/.cache/huggingface
cd /orcd/scratch/orcd/008/manaal/child-adult-diarization
mkdir -p logs/baselines
echo "Start: $(date)"
python baselines/eval_only_fused_large_cross_child.py
echo "Done: $(date)"
