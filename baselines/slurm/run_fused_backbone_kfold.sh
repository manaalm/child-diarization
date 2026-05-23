#!/bin/bash
#SBATCH -J fused_bb_kf
#SBATCH -p ou_bcs_normal,pi_satra
#SBATCH -t 12:00:00
#SBATCH -c 4
#SBATCH --mem=80G
#SBATCH --requeue
#SBATCH --gres=gpu:h100:1
#SBATCH -o logs/baselines/fused_bb_kf_%A_%a.out
#SBATCH -e logs/baselines/fused_bb_kf_%A_%a.err

# 3-fold within-child run of fused_attn_unfreeze2 with Whisper-medium and
# Whisper-large-v3 backbones. Submit as 6-task array:
#   sbatch --array=0-5 baselines/slurm/run_fused_backbone_kfold.sh
# Mapping (BACKBONE × FOLD):
#   0 = medium fold 0   3 = large  fold 0
#   1 = medium fold 1   4 = large  fold 1
#   2 = medium fold 2   5 = large  fold 2

set -euo pipefail

source /home/manaal/miniforge3/etc/profile.d/conda.sh
conda activate child-vocalizations

export HF_HOME=/orcd/scratch/orcd/008/manaal/.cache/huggingface
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
unset HF_TOKEN HUGGINGFACE_HUB_TOKEN HF_HUB_TOKEN
export PYTHONPATH=/orcd/scratch/orcd/008/manaal/child-adult-diarization

cd /orcd/scratch/orcd/008/manaal/child-adult-diarization
mkdir -p logs/baselines

TASK=${SLURM_ARRAY_TASK_ID:-0}
case "${TASK}" in
    0) BACKBONE=medium; FOLD=0 ;;
    1) BACKBONE=medium; FOLD=1 ;;
    2) BACKBONE=medium; FOLD=2 ;;
    3) BACKBONE=large;  FOLD=0 ;;
    4) BACKBONE=large;  FOLD=1 ;;
    5) BACKBONE=large;  FOLD=2 ;;
    *) echo "Unknown SLURM_ARRAY_TASK_ID=${TASK}"; exit 2 ;;
esac

echo "=== fused_attn_unfreeze2 backbone=${BACKBONE} fold=${FOLD} (job ${SLURM_JOB_ID} array task ${TASK}) ==="
echo "Start: $(date)"
python -u baselines/run_fused_attn_unfreeze2_kfold.py --backbone "${BACKBONE}" --fold "${FOLD}"
echo "Done: $(date)"
