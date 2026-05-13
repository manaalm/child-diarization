#!/bin/bash
#SBATCH --job-name=audio_llm
#SBATCH --gres=gpu:1
#SBATCH -t 12:00:00
#SBATCH --mem=64G
#SBATCH -c 4
#SBATCH -p ou_bcs_normal,pi_satra
#SBATCH -o logs/baselines/audio_llm_%j.out
#SBATCH -e logs/baselines/audio_llm_%j.out

SPLIT=${1:-val}
MODEL_SLUG=${2:-qwen25_omni_7b}
N_SHOT=${3:-0}
PROMPT_TEMPLATE=${4:-zero_shot_v1}
HF_MODEL=${5:-Qwen/Qwen2.5-Omni-7B}
MODEL_CLASS=${6:-}    # spec-022 US3: e.g. Qwen3OmniMoeForConditionalGeneration for Qwen3-Omni

echo "Start: $(date)"
echo "SPLIT=${SPLIT}  MODEL_SLUG=${MODEL_SLUG}  N_SHOT=${N_SHOT}  PROMPT_TEMPLATE=${PROMPT_TEMPLATE}"
echo "HF_MODEL=${HF_MODEL}  MODEL_CLASS=${MODEL_CLASS:-(auto)}"

source /home/manaal/miniforge3/etc/profile.d/conda.sh
conda activate child-vocalizations

# Cache model weights in scratch to avoid home quota pressure
export HF_HOME=/orcd/scratch/orcd/008/manaal/.cache/huggingface
# transformers >=4.57 has_file() network bug — force offline mode (HF cache populated)
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
# Expired HF_TOKEN from inherited env causes 401 even on offline-load paths
unset HF_TOKEN HUGGINGFACE_HUB_TOKEN HF_HUB_TOKEN || true

cd /orcd/scratch/orcd/008/manaal/child-adult-diarization

EXTRA_ARGS=()
if [ -n "${MODEL_CLASS}" ]; then
    EXTRA_ARGS+=(--model-class "${MODEL_CLASS}")
fi

python baselines/audio_llm_baseline.py \
    --split "${SPLIT}" \
    --model "${HF_MODEL}" \
    --model-slug "${MODEL_SLUG}" \
    --n-shot "${N_SHOT}" \
    --prompt-template "${PROMPT_TEMPLATE}" \
    --seed 42 \
    "${EXTRA_ARGS[@]}"

echo "Done: $(date)"
