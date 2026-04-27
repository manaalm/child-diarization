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
MODEL_SLUG=${2:-qwen2_audio_7b}
N_SHOT=${3:-0}

echo "Start: $(date)"
echo "SPLIT=${SPLIT}  MODEL_SLUG=${MODEL_SLUG}  N_SHOT=${N_SHOT}"

source /home/manaal/miniforge3/etc/profile.d/conda.sh
conda activate child-vocalizations

# Cache model weights in scratch to avoid home quota pressure
export HF_HOME=/orcd/scratch/orcd/008/manaal/.cache/huggingface

cd /orcd/scratch/orcd/008/manaal/child-adult-diarization

python baselines/audio_llm_baseline.py \
    --split "${SPLIT}" \
    --model-slug "${MODEL_SLUG}" \
    --n-shot "${N_SHOT}" \
    --seed 42

echo "Done: $(date)"
