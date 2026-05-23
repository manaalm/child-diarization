#!/bin/bash
#SBATCH -J fused_kfold
#SBATCH -p ou_bcs_normal,pi_satra
#SBATCH -t 12:00:00
#SBATCH -c 4
#SBATCH --mem=40G
#SBATCH --requeue
#SBATCH --gres=gpu:1
#SBATCH -o logs/baselines/fused_kfold_%A_%a.out
#SBATCH -e logs/baselines/fused_kfold_%A_%a.err

# 3-fold within-child run of fused_attn_unfreeze2.
# Submit as: sbatch --array=0-2 baselines/slurm/run_fused_attn_unfreeze2_kfold.sh

set -euo pipefail

source /home/manaal/miniforge3/etc/profile.d/conda.sh
conda activate child-vocalizations

export HF_HOME=/orcd/scratch/orcd/008/manaal/.cache/huggingface
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export PYTHONPATH=/orcd/scratch/orcd/008/manaal/child-adult-diarization

cd /orcd/scratch/orcd/008/manaal/child-adult-diarization
mkdir -p logs/baselines

FOLD=${SLURM_ARRAY_TASK_ID:-0}
echo "=== fused_attn_unfreeze2 fold=${FOLD} (job ${SLURM_JOB_ID} array ${SLURM_ARRAY_TASK_ID:-na}) ==="
echo "Start: $(date)"

python -u baselines/run_fused_attn_unfreeze2_kfold.py --fold "${FOLD}"

echo "Done: $(date)"
