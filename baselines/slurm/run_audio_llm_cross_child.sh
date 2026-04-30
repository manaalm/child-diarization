#!/bin/bash
#SBATCH --job-name=audio_llm_cross
#SBATCH --gres=gpu:1
#SBATCH -t 12:00:00
#SBATCH --mem=64G
#SBATCH -c 4
#SBATCH -p ou_bcs_normal,pi_satra
#SBATCH -o logs/baselines/audio_llm_cross_%j.out
#SBATCH -e logs/baselines/audio_llm_cross_%j.out

# Qwen2-Audio-7B-Instruct zero-shot baseline on the cross-child split (baselines/splits/).
# Runs val (threshold tuning) then test in one job.
# Results: baselines/audio_llm_baseline_runs/qwen2_audio_7b_cross_child/
# Cache:   baselines/audio_llm_cache/qwen2_audio_7b_cross_child/
#   (separate from the seen-child cache to avoid stale-threshold cross-contamination)

set -euo pipefail

echo "Start: $(date)"
echo "SPLIT=val+test  MODEL=qwen2_audio_7b_cross_child"

source /home/manaal/miniforge3/etc/profile.d/conda.sh
conda activate child-vocalizations

export HF_HOME=/orcd/scratch/orcd/008/manaal/.cache/huggingface

REPO=/orcd/scratch/orcd/008/manaal/child-adult-diarization
cd "$REPO"
mkdir -p logs/baselines

OUT_DIR=baselines/audio_llm_baseline_runs/qwen2_audio_7b_cross_child
CACHE_DIR=baselines/audio_llm_cache/qwen2_audio_7b_cross_child

echo "--- Audio LLM cross-child val ---"
python baselines/audio_llm_baseline.py \
    --split val \
    --split-csv baselines/splits/val.csv \
    --train-csv baselines/splits/train.csv \
    --model-slug qwen2_audio_7b_cross_child \
    --output-dir "$OUT_DIR" \
    --cache-dir  "$CACHE_DIR" \
    --seed 42

echo "--- Audio LLM cross-child test ---"
python baselines/audio_llm_baseline.py \
    --split test \
    --split-csv baselines/splits/test.csv \
    --train-csv baselines/splits/train.csv \
    --model-slug qwen2_audio_7b_cross_child \
    --output-dir "$OUT_DIR" \
    --cache-dir  "$CACHE_DIR" \
    --seed 42

echo "Done: $(date)"
