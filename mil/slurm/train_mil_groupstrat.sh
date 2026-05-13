#!/bin/bash
#SBATCH -c 4
#SBATCH -t 8:00:00
#SBATCH -p ou_bcs_normal,pi_satra
#SBATCH --mem=40G
#SBATCH --requeue
#SBATCH --gres=gpu:1
#SBATCH -o logs/mil/groupstrat_%A_%a.out
#SBATCH -e logs/mil/groupstrat_%A_%a.err

# spec-022 US2: group-stratified k-fold MIL training. Submit as array job:
#   sbatch --array=0-2 mil/slurm/train_mil_groupstrat.sh wavlm_mil
#   sbatch --array=0-2 mil/slurm/train_mil_groupstrat.sh whisper_mil
#   etc.
#
# Configs must already exist at mil/configs/groupstrat3/<system>_fold<k>.yaml
# (run evaluation/generate_kfold_configs.py --variant groupstrat first).
# Result dirs land at mil/mil_results/<system>_groupstrat3_f<fold>/.

set -euo pipefail

SYSTEM=${1:?"Usage: sbatch --array=0-K-1 train_mil_groupstrat.sh <system>"}
FOLD=${SLURM_ARRAY_TASK_ID:-0}
K=${KFOLD_K:-3}

CONFIG="mil/configs/groupstrat${K}/${SYSTEM}_fold${FOLD}.yaml"
VARIANT="${SYSTEM}_groupstrat${K}_f${FOLD}"

export PATH="$HOME/.local/bin:$PATH"
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
unset HF_TOKEN HUGGINGFACE_HUB_TOKEN HF_HUB_TOKEN || true
source /orcd/home/002/manaal/miniforge3/etc/profile.d/conda.sh
conda activate child-vocalizations

REPO=/orcd/scratch/orcd/008/manaal/child-adult-diarization
cd "$REPO"
mkdir -p logs/mil

echo "=== MIL group-stratified k-fold training: $SYSTEM fold $FOLD (job $SLURM_JOB_ID array $SLURM_ARRAY_TASK_ID) ==="
echo "Config: $CONFIG"

if [ ! -f "$CONFIG" ]; then
    echo "ERROR: config not found: $CONFIG" >&2
    echo "Run: python evaluation/generate_kfold_configs.py --k $K --variant groupstrat" >&2
    exit 2
fi

python mil/mil_train.py --config "$CONFIG"

CKPT="mil/mil_results/${VARIANT}/best_checkpoint.pt"
CFG_JSON="mil/mil_results/${VARIANT}/config.json"
if [ ! -f "$CKPT" ]; then
    echo "ERROR: post-train checkpoint missing: $CKPT" >&2
    exit 3
fi

echo "--- Evaluating $VARIANT on fold $FOLD test ---"
python mil/mil_evaluate.py --checkpoint "$CKPT" --config "$CFG_JSON"

echo "=== Done: $VARIANT ==="
