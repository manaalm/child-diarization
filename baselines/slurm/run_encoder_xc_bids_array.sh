#!/bin/bash
#SBATCH -J encoder_xc_array
#SBATCH -p ou_bcs_normal,pi_satra
#SBATCH -t 4:00:00
#SBATCH -c 4
#SBATCH --mem=40G
#SBATCH --requeue
#SBATCH --gres=gpu:1
#SBATCH --array=0-8
#SBATCH -o logs/baselines/encoder_xc_array_%A_%a.out
#SBATCH -e logs/baselines/encoder_xc_array_%A_%a.err

# Parallel SLURM array for the 9 remaining encoder variants on BIDS cross-child
# (after wavlm_mean lands sequentially). Each task trains a single variant.
# --skip-existing protects any variant that finished sequentially first.

set -euo pipefail

VARIANTS=(
    wavlm_attn
    whisper_attn_lw
    wavlm_attn_lw
    fused_attn
    whisper_attn_unfreeze2
    fused_attn_unfreeze2
    whisper_attn_ptt
    whisper_attn_aug
    whisper_attn_aug_ptt
)
V="${VARIANTS[$SLURM_ARRAY_TASK_ID]}"

source /home/manaal/miniforge3/etc/profile.d/conda.sh
conda activate child-vocalizations
unset HF_TOKEN HUGGINGFACE_HUB_TOKEN HF_HUB_TOKEN 2>/dev/null || true
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export HF_HOME=/orcd/scratch/orcd/008/manaal/.cache/huggingface

cd /orcd/scratch/orcd/008/manaal/child-adult-diarization
mkdir -p logs/baselines

echo "Array task $SLURM_ARRAY_TASK_ID: variant=$V  start=$(date)"
python baselines/baseline_encoders.py \
    --all-experiments \
    --experiments "$V" \
    --skip-existing \
    --results-root ./baselines/baseline_results_cross_child_bids
echo "Array task $SLURM_ARRAY_TASK_ID: variant=$V  end=$(date)"
