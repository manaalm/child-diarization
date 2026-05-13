#!/bin/bash
#SBATCH --job-name=qwen3_omni
#SBATCH --gres=gpu:a100:1
#SBATCH -t 16:00:00
#SBATCH --mem=160G
#SBATCH -c 8
#SBATCH -p ou_bcs_normal,pi_satra
#SBATCH -o logs/baselines/qwen3_omni_%j.out
#SBATCH -e logs/baselines/qwen3_omni_%j.out

# spec-022 US3 — Qwen3-Omni-30B-A3B-Thinking dispatcher.
# Args:
#   $1 = split (val|test|test_all)
#   $2 = offline mode (1 default; pass 0 for the first job to download weights)
#
# Qwen3.5-Omni open-weight status unconfirmed as of 2026-05-12; this script
# targets the confirmed-open-weight Qwen3-Omni-30B-A3B-Thinking variant.
# Model is ~60GB total / ~6GB active per forward (MoE). Needs A100-80G or H100.

SPLIT=${1:-val}
OFFLINE=${2:-1}

HF_MODEL="Qwen/Qwen3-Omni-30B-A3B-Thinking"
MODEL_SLUG="qwen3_omni_30b_thinking"
# Class candidates auto-detected by audio_llm_baseline.py:_resolve_model_class

echo "Start: $(date)"
echo "SPLIT=${SPLIT}  OFFLINE=${OFFLINE}"
echo "HF_MODEL=${HF_MODEL}  MODEL_SLUG=${MODEL_SLUG}"

source /home/manaal/miniforge3/etc/profile.d/conda.sh
conda activate child-vocalizations

export HF_HOME=/orcd/scratch/orcd/008/manaal/.cache/huggingface
if [ "${OFFLINE}" = "1" ]; then
    export TRANSFORMERS_OFFLINE=1
    export HF_HUB_OFFLINE=1
    echo "  offline mode: TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1"
else
    unset TRANSFORMERS_OFFLINE HF_HUB_OFFLINE
    echo "  online mode (download): TRANSFORMERS_OFFLINE / HF_HUB_OFFLINE unset"
fi
unset HF_TOKEN HUGGINGFACE_HUB_TOKEN HF_HUB_TOKEN || true

cd /orcd/scratch/orcd/008/manaal/child-adult-diarization

python baselines/audio_llm_baseline.py \
    --split "${SPLIT}" \
    --model "${HF_MODEL}" \
    --model-slug "${MODEL_SLUG}" \
    --n-shot 0 \
    --prompt-template zero_shot_v1 \
    --seed 42

echo "Done: $(date)"
