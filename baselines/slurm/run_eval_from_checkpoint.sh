#!/bin/bash
#SBATCH -J eval_from_ckpt
#SBATCH -p ou_bcs_normal,pi_satra
#SBATCH -t 0:45:00
#SBATCH -c 4
#SBATCH --mem=48G
#SBATCH --requeue
#SBATCH --gres=gpu:1
#SBATCH -o logs/baselines/eval_from_ckpt_%j.out
#SBATCH -e logs/baselines/eval_from_ckpt_%j.err

# Eval-only wrapper for encoders/eval_from_checkpoint.py. Pass the checkpoint
# path as the first positional argument:
#     sbatch baselines/slurm/run_eval_from_checkpoint.sh \
#         baseline_results_seen_child/fused_attn_unfreeze2/best_model.pt

set -euo pipefail

CKPT="${1:?usage: sbatch $0 <path/to/best_model.pt>}"

source /home/manaal/miniforge3/etc/profile.d/conda.sh
conda activate child-vocalizations
unset HF_TOKEN HUGGINGFACE_HUB_TOKEN HF_HUB_TOKEN 2>/dev/null || true
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export HF_HOME=/orcd/scratch/orcd/008/manaal/.cache/huggingface

cd /orcd/scratch/orcd/008/manaal/child-adult-diarization
mkdir -p logs/baselines

echo "Start: $(date)"
echo "Checkpoint: $CKPT"
python encoders/eval_from_checkpoint.py --ckpt "$CKPT"
echo "Done: $(date)"
