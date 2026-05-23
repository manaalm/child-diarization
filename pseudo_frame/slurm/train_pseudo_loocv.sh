#!/bin/bash
#SBATCH --job-name=pseudo_loocv
#SBATCH -c 4
#SBATCH -t 2:00:00
#SBATCH -p ou_bcs_normal,pi_satra
#SBATCH --mem=40G
#SBATCH --requeue
#SBATCH --gres=gpu:1
#SBATCH -o logs/pseudo_frame/loocv_%A_%a.out
#SBATCH -e logs/pseudo_frame/loocv_%A_%a.err

# LOOCV (130-fold) driver for pseudo-frame (spec-022 follow-up).
# Usage: sbatch --array=0-129%25 pseudo_frame/slurm/train_pseudo_loocv.sh <system>

set -euo pipefail

SYSTEM="${1:?Usage: sbatch --array=0-129%25 train_pseudo_loocv.sh <system>}"
FOLD="${SLURM_ARRAY_TASK_ID:-0}"

CONFIG="pseudo_frame/configs/loocv/${SYSTEM}_fold${FOLD}.yaml"
VARIANT="${SYSTEM}_loocv_f${FOLD}"

export PATH="$HOME/.local/bin:$PATH"
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
unset HF_TOKEN HUGGINGFACE_HUB_TOKEN HF_HUB_TOKEN || true
source /orcd/home/002/manaal/miniforge3/etc/profile.d/conda.sh
conda activate child-vocalizations

REPO=/orcd/scratch/orcd/008/manaal/child-adult-diarization
cd "$REPO"
mkdir -p logs/pseudo_frame

if [ ! -f "$CONFIG" ]; then
    echo "ERROR: config not found: $CONFIG" >&2
    echo "Run: python evaluation/generate_loocv_configs.py --systems $SYSTEM" >&2
    exit 2
fi

echo "=== pseudo-frame LOOCV: $SYSTEM fold $FOLD (job $SLURM_JOB_ID array $SLURM_ARRAY_TASK_ID) ==="
python pseudo_frame/pseudo_train.py --config "$CONFIG"

CKPT="pseudo_frame/results/${VARIANT}/best_checkpoint.pt"
if [ ! -f "$CKPT" ]; then
    echo "ERROR: post-train checkpoint missing: $CKPT" >&2
    exit 3
fi
echo "--- Evaluating $VARIANT on fold $FOLD test (1 held-out child) ---"
python pseudo_frame/pseudo_evaluate.py --checkpoint "$CKPT" --split test

echo "=== Done: $VARIANT ==="
