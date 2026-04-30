#!/bin/bash
#SBATCH --job-name=cohere_seen
#SBATCH --gres=gpu:1
#SBATCH -t 12:00:00
#SBATCH --mem=64G
#SBATCH -c 4
#SBATCH -p ou_bcs_normal,pi_satra
#SBATCH -o logs/baselines/cohere_seen_%j.out
#SBATCH -e logs/baselines/cohere_seen_%j.out

# CohereLabs/cohere-transcribe-03-2026 on the seen-child split.
# Runs val (threshold tuning) then test in one job.
# Results: baselines/audio_model_baseline_runs/cohere_transcribe/

set -euo pipefail

echo "Start: $(date)"
echo "Node: $(hostname)  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo none)"

source /home/manaal/miniforge3/etc/profile.d/conda.sh
conda activate child-vocalizations

export HF_HOME=/orcd/scratch/orcd/008/manaal/.cache/huggingface

REPO=/orcd/scratch/orcd/008/manaal/child-adult-diarization
cd "$REPO"
mkdir -p logs/baselines

MODEL="CohereLabs/cohere-transcribe-03-2026"
OUT_DIR=baselines/audio_model_baseline_runs/cohere_transcribe
CACHE_DIR=baselines/audio_model_cache/cohere_transcribe

echo "--- Cohere seen-child val ---"
python baselines/audio_model_baseline.py \
    --model "$MODEL" \
    --split val \
    --output-dir "$OUT_DIR" \
    --cache-dir  "$CACHE_DIR" \
    --seed 42

echo "--- Cohere seen-child test ---"
python baselines/audio_model_baseline.py \
    --model "$MODEL" \
    --split test \
    --output-dir "$OUT_DIR" \
    --cache-dir  "$CACHE_DIR" \
    --seed 42

echo "Done: $(date)"
