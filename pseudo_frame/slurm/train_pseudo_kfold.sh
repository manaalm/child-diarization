#!/bin/bash
#SBATCH -c 4
#SBATCH -t 8:00:00
#SBATCH -p ou_bcs_normal,pi_satra
#SBATCH --mem=40G
#SBATCH --requeue
#SBATCH --gres=gpu:1
#SBATCH -o logs/pseudo_frame/kfold_%A_%a.out
#SBATCH -e logs/pseudo_frame/kfold_%A_%a.err

# k-fold pseudo-frame training driver. Submit as an array job:
#   sbatch --array=0-2 pseudo_frame/slurm/train_pseudo_kfold.sh wavlm_pseudo_frame

set -euo pipefail

SYSTEM=${1:?"Usage: sbatch --array=0-K-1 train_pseudo_kfold.sh <system>"}
FOLD=${SLURM_ARRAY_TASK_ID:-0}
K=${KFOLD_K:-3}

CONFIG="pseudo_frame/configs/kfold_${K}fold/${SYSTEM}_fold${FOLD}.yaml"
VARIANT="${SYSTEM}_kfold${K}_f${FOLD}"

export PATH="$HOME/.local/bin:$PATH"
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
source /orcd/home/002/manaal/miniforge3/etc/profile.d/conda.sh
conda activate child-vocalizations

REPO=/orcd/scratch/orcd/008/manaal/child-adult-diarization
cd "$REPO"
mkdir -p logs/pseudo_frame

echo "=== pseudo-frame k-fold: $SYSTEM fold $FOLD (job $SLURM_JOB_ID array $SLURM_ARRAY_TASK_ID) ==="
echo "Config: $CONFIG"

if [ ! -f "$CONFIG" ]; then
    echo "ERROR: config not found: $CONFIG" >&2
    echo "Run: python evaluation/generate_kfold_configs.py --k $K" >&2
    exit 2
fi

# Train (assumes pseudo_labels already built via build_pseudo_labels.py)
python pseudo_frame/pseudo_train.py --config "$CONFIG"

# Evaluate on test of this fold
CKPT="pseudo_frame/results/${VARIANT}/best_checkpoint.pt"
if [ ! -f "$CKPT" ]; then
    echo "ERROR: post-train checkpoint missing: $CKPT" >&2
    exit 3
fi
echo "--- Evaluating $VARIANT on fold $FOLD test ---"
python pseudo_frame/pseudo_evaluate.py --checkpoint "$CKPT" --split test

echo "=== Done: $VARIANT ==="
