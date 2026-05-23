#!/bin/bash
#SBATCH -J fused_groupstrat
#SBATCH -p ou_bcs_normal,pi_satra
#SBATCH -t 12:00:00
#SBATCH -c 4
#SBATCH --mem=80G
#SBATCH --requeue
#SBATCH --gres=gpu:a100:1
#SBATCH -o logs/baselines/fused_groupstrat_%A_%a.out
#SBATCH -e logs/baselines/fused_groupstrat_%A_%a.err

# Group-stratified 3-fold for fused_attn_unfreeze2 with 3 backbones.
# Array maps: 0-2 = small fold 0/1/2, 3-5 = medium fold 0/1/2, 6-8 = large fold 0/1/2.
#   sbatch --array=0-8 baselines/slurm/run_fused_backbone_groupstrat.sh

set -euo pipefail

source /home/manaal/miniforge3/etc/profile.d/conda.sh
conda activate child-vocalizations

export HF_HOME=/orcd/scratch/orcd/008/manaal/.cache/huggingface
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export PYTHONPATH=/orcd/scratch/orcd/008/manaal/child-adult-diarization

cd /orcd/scratch/orcd/008/manaal/child-adult-diarization
mkdir -p logs/baselines

IDX=${SLURM_ARRAY_TASK_ID:-0}
FOLD=$((IDX % 3))
BACKBONE_IDX=$((IDX / 3))
case "$BACKBONE_IDX" in
    0) BACKBONE=small ;;
    1) BACKBONE=medium ;;
    2) BACKBONE=large ;;
    *) echo "Unknown BACKBONE_IDX=$BACKBONE_IDX"; exit 2 ;;
esac

echo "=== fused_attn_unfreeze2 groupstrat: ${BACKBONE} fold ${FOLD} (job ${SLURM_JOB_ID} array ${SLURM_ARRAY_TASK_ID}) ==="
echo "Start: $(date)"
python -u baselines/run_fused_attn_unfreeze2_kfold.py --backbone "${BACKBONE}" --fold "${FOLD}" --paradigm groupstrat
echo "Done: $(date)"
