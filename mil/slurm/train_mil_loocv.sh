#!/bin/bash
#SBATCH --job-name=mil_loocv
#SBATCH -c 4
#SBATCH -t 4:00:00
#SBATCH -p ou_bcs_normal,pi_satra
#SBATCH --mem=40G
#SBATCH --requeue
#SBATCH --gres=gpu:1
#SBATCH -o logs/mil/loocv_%A_%a.out
#SBATCH -e logs/mil/loocv_%A_%a.err

# LOOCV (130-fold) driver for the MIL family (spec-022 follow-up).
# Usage: sbatch --array=0-129%25 mil/slurm/train_mil_loocv.sh <system_name>
#   e.g. sbatch --array=0-129%25 mil/slurm/train_mil_loocv.sh whisper_mil

set -euo pipefail

SYSTEM="${1:?Usage: sbatch --array=0-129%25 train_mil_loocv.sh <system_name>}"
FOLD="${SLURM_ARRAY_TASK_ID:-0}"

CONFIG="mil/configs/loocv/${SYSTEM}_fold${FOLD}.yaml"
VARIANT="${SYSTEM}_loocv_f${FOLD}"

export PATH="$HOME/.local/bin:$PATH"
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
unset HF_TOKEN HUGGINGFACE_HUB_TOKEN HF_HUB_TOKEN || true
source /orcd/home/002/manaal/miniforge3/etc/profile.d/conda.sh
conda activate child-vocalizations

REPO=/orcd/scratch/orcd/008/manaal/child-adult-diarization
cd "$REPO"
mkdir -p logs/mil

if [ ! -f "$CONFIG" ]; then
    echo "ERROR: config not found: $CONFIG" >&2
    echo "Run: python evaluation/generate_loocv_configs.py --systems $SYSTEM" >&2
    exit 2
fi

echo "=== MIL LOOCV: $SYSTEM fold $FOLD (job $SLURM_JOB_ID array $SLURM_ARRAY_TASK_ID) ==="
python mil/mil_train.py --config "$CONFIG"

CKPT="mil/mil_results/${VARIANT}/best_checkpoint.pt"
CFG_JSON="mil/mil_results/${VARIANT}/config.json"
if [ ! -f "$CKPT" ]; then
    echo "ERROR: post-train checkpoint missing: $CKPT" >&2
    exit 3
fi

echo "--- Evaluating $VARIANT on fold $FOLD test (1 held-out child) ---"
python mil/mil_evaluate.py --checkpoint "$CKPT" --config "$CFG_JSON"

echo "=== Done: $VARIANT ==="
