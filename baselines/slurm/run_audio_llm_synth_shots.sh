#!/bin/bash
#SBATCH --job-name=audio_llm_synth
#SBATCH --gres=gpu:1
#SBATCH -t 12:00:00
#SBATCH --mem=64G
#SBATCH -c 4
#SBATCH -p ou_bcs_normal,pi_satra
#SBATCH -o logs/baselines/audio_llm_synth_%j.out
#SBATCH -e logs/baselines/audio_llm_synth_%j.out

# Audio LLM 2-shot variant with universal synthetic demos (1 positive + 1 adult-only-negative
# synth scene used for every test query, replacing per-child same-speaker demos).
SPLIT=${1:-val}
MODEL_SLUG=${2:-qwen2_audio_7b_synth_2shot}

echo "Start: $(date)"
echo "SPLIT=${SPLIT}  MODEL_SLUG=${MODEL_SLUG}"

source /home/manaal/miniforge3/etc/profile.d/conda.sh
conda activate child-vocalizations

export HF_HOME=/orcd/scratch/orcd/008/manaal/.cache/huggingface

cd /orcd/scratch/orcd/008/manaal/child-adult-diarization

python baselines/audio_llm_baseline.py \
    --split "${SPLIT}" \
    --model-slug "${MODEL_SLUG}" \
    --train-csv synth_results/manifests/synthetic_audio_llm_shots.csv \
    --universal-shots \
    --n-shot 2 \
    --seed 42

echo "Done: $(date)"
