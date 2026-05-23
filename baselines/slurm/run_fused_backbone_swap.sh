#!/bin/bash
#SBATCH -J fused_swap
#SBATCH -p ou_bcs_normal,pi_satra
#SBATCH -t 12:00:00
#SBATCH -c 4
#SBATCH --mem=80G
#SBATCH --requeue
#SBATCH --gres=gpu:a100:1
#SBATCH -o logs/baselines/fused_swap_%A_%a.out
#SBATCH -e logs/baselines/fused_swap_%A_%a.err

# Backbone-swap sweep for fused_attn_unfreeze2.
# Submit as array — task index 0 = medium, 1 = large:
#   sbatch --array=0-1 baselines/slurm/run_fused_backbone_swap.sh

set -euo pipefail

source /home/manaal/miniforge3/etc/profile.d/conda.sh
conda activate child-vocalizations

export HF_HOME=/orcd/scratch/orcd/008/manaal/.cache/huggingface
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export PYTHONPATH=/orcd/scratch/orcd/008/manaal/child-adult-diarization

cd /orcd/scratch/orcd/008/manaal/child-adult-diarization
mkdir -p logs/baselines

case "${SLURM_ARRAY_TASK_ID:-0}" in
    0) BACKBONE=medium ;;
    1) BACKBONE=large ;;
    *) echo "Unknown SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}"; exit 2 ;;
esac

echo "=== fused_attn_unfreeze2 backbone-swap: ${BACKBONE} (job ${SLURM_JOB_ID}) ==="
echo "Start: $(date)"

python -u baselines/run_fused_attn_unfreeze2_backbone_swap.py --backbone "${BACKBONE}"

echo "Done: $(date)"
