#!/bin/bash
#SBATCH -c 4
#SBATCH -t 8:00:00
#SBATCH -p ou_bcs_normal,pi_satra
#SBATCH --mem=40G
#SBATCH --requeue
#SBATCH --gres=gpu:1
#SBATCH -o logs/mil/kfold_%A_%a.out
#SBATCH -e logs/mil/kfold_%A_%a.err

# k-fold MIL training driver. Submit as an array job:
#   sbatch --array=0-2 mil/slurm/train_mil_kfold.sh wavlm_mil
#   sbatch --array=0-2 mil/slurm/train_mil_kfold.sh whisper_mil
#   sbatch --array=0-2 mil/slurm/train_mil_kfold.sh whisper_mil_tsmil_concat
#
# Reads SLURM_ARRAY_TASK_ID as the fold index. Configs must already be
# generated via:
#   python evaluation/generate_kfold_configs.py --k 3
#
# Trains, then runs mil_evaluate.py to produce test_predictions.csv keyed
# in mil/mil_results/<system>_kfold3_f<fold>/.

set -euo pipefail

SYSTEM=${1:?"Usage: sbatch --array=0-K-1 train_mil_kfold.sh <system>"}
FOLD=${SLURM_ARRAY_TASK_ID:-0}
K=${KFOLD_K:-3}

CONFIG="mil/configs/kfold_${K}fold/${SYSTEM}_fold${FOLD}.yaml"
VARIANT="${SYSTEM}_kfold${K}_f${FOLD}"

export PATH="$HOME/.local/bin:$PATH"
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
source /orcd/home/002/manaal/miniforge3/etc/profile.d/conda.sh
conda activate child-vocalizations

REPO=/orcd/scratch/orcd/008/manaal/child-adult-diarization
cd "$REPO"
mkdir -p logs/mil

echo "=== MIL k-fold training: $SYSTEM fold $FOLD (job $SLURM_JOB_ID array $SLURM_ARRAY_TASK_ID) ==="
echo "Config: $CONFIG"

if [ ! -f "$CONFIG" ]; then
    echo "ERROR: config not found: $CONFIG" >&2
    echo "Run: python evaluation/generate_kfold_configs.py --k $K" >&2
    exit 2
fi

# Train
python mil/mil_train.py --config "$CONFIG"

# Evaluate on test of this fold
CKPT="mil/mil_results/${VARIANT}/best_checkpoint.pt"
CFG_JSON="mil/mil_results/${VARIANT}/config.json"
if [ ! -f "$CKPT" ]; then
    echo "ERROR: post-train checkpoint missing: $CKPT" >&2
    exit 3
fi

echo "--- Evaluating $VARIANT on fold $FOLD test ---"
python mil/mil_evaluate.py --checkpoint "$CKPT" --config "$CFG_JSON"

echo "=== Done: $VARIANT ==="
