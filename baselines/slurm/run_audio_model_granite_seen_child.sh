#!/bin/bash
#SBATCH --job-name=granite_seen
#SBATCH --gres=gpu:1
#SBATCH -t 12:00:00
#SBATCH --mem=64G
#SBATCH -c 4
#SBATCH -p ou_bcs_normal,pi_satra
#SBATCH -o logs/baselines/granite_seen_%j.out
#SBATCH -e logs/baselines/granite_seen_%j.out

# ibm-granite/granite-4.0-1b-speech on the seen-child split.
# Runs val (threshold tuning) then test in one job.
# Results: baselines/audio_model_baseline_runs/granite_speech_1b/

set -euo pipefail

echo "Start: $(date)"
echo "Node: $(hostname)  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo none)"

source /home/manaal/miniforge3/etc/profile.d/conda.sh
conda activate child-vocalizations

export HF_HOME=/orcd/scratch/orcd/008/manaal/.cache/huggingface

REPO=/orcd/scratch/orcd/008/manaal/child-adult-diarization
cd "$REPO"
mkdir -p logs/baselines

MODEL="ibm-granite/granite-4.0-1b-speech"
OUT_DIR=baselines/audio_model_baseline_runs/granite_speech_1b
CACHE_DIR=baselines/audio_model_cache/granite_speech_1b

echo "--- Granite-Speech seen-child val ---"
python baselines/audio_model_baseline.py \
    --model "$MODEL" \
    --split val \
    --output-dir "$OUT_DIR" \
    --cache-dir  "$CACHE_DIR" \
    --seed 42

echo "--- Granite-Speech seen-child test ---"
python baselines/audio_model_baseline.py \
    --model "$MODEL" \
    --split test \
    --output-dir "$OUT_DIR" \
    --cache-dir  "$CACHE_DIR" \
    --seed 42

echo "Done: $(date)"
