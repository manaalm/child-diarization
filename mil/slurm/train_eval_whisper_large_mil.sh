#!/bin/bash
#SBATCH -J wlarge_mil
#SBATCH -c 4
#SBATCH -t 12:00:00
#SBATCH -p ou_bcs_normal,pi_satra
#SBATCH --mem=80G
#SBATCH --requeue
#SBATCH --gres=gpu:h100:1
#SBATCH -o logs/mil/train_eval_whisper_large_%j.out
#SBATCH -e logs/mil/train_eval_whisper_large_%j.err

# Train + eval whisper-large-v3 MIL backbone variant.
# Usage: sbatch mil/slurm/train_eval_whisper_large_mil.sh

set -euo pipefail

CONFIG="mil/configs/backbone_size/whisper_large_mil.yaml"
VARIANT="whisper_large_mil"

export PATH="$HOME/.local/bin:$PATH"
export HF_HOME=/orcd/scratch/orcd/008/manaal/.cache/huggingface
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
unset HF_TOKEN HUGGINGFACE_HUB_TOKEN HF_HUB_TOKEN
source /orcd/home/002/manaal/miniforge3/etc/profile.d/conda.sh
conda activate child-vocalizations

REPO=/orcd/scratch/orcd/008/manaal/child-adult-diarization
cd "$REPO"
mkdir -p logs/mil

echo "=== MIL train+eval: $VARIANT (job $SLURM_JOB_ID) ==="
echo "Config: $CONFIG"
echo "Start: $(date)"

python mil/mil_train.py --config "$CONFIG"

CKPT="mil/mil_results/${VARIANT}/best_checkpoint.pt"
CFG_JSON="mil/mil_results/${VARIANT}/config.json"
if [ ! -f "$CKPT" ]; then
    echo "ERROR: post-train checkpoint missing: $CKPT" >&2
    exit 3
fi

echo "--- Evaluating $VARIANT on test ---"
python mil/mil_evaluate.py --checkpoint "$CKPT" --config "$CFG_JSON"

echo "Done: $(date)"
