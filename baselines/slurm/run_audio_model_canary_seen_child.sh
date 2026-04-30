#!/bin/bash
#SBATCH --job-name=canary_seen
#SBATCH --gres=gpu:1
#SBATCH -t 12:00:00
#SBATCH --mem=64G
#SBATCH -c 4
#SBATCH -p ou_bcs_normal,pi_satra
#SBATCH -o logs/baselines/canary_seen_%j.out
#SBATCH -e logs/baselines/canary_seen_%j.out

# nvidia/canary-qwen-2.5b on the seen-child split.
# Runs val (threshold tuning) then test in one job.
# Results: baselines/audio_model_baseline_runs/canary_qwen_2_5b/

set -euo pipefail

echo "Start: $(date)"
echo "Node: $(hostname)  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo none)"

source /home/manaal/miniforge3/etc/profile.d/conda.sh
conda activate child-vocalizations

export HF_HOME=/orcd/scratch/orcd/008/manaal/.cache/huggingface

REPO=/orcd/scratch/orcd/008/manaal/child-adult-diarization
cd "$REPO"
mkdir -p logs/baselines

MODEL="nvidia/canary-qwen-2.5b"
OUT_DIR=baselines/audio_model_baseline_runs/canary_qwen_2_5b
CACHE_DIR=baselines/audio_model_cache/canary_qwen_2_5b

echo "--- Canary-Qwen seen-child val ---"
python baselines/audio_model_baseline.py \
    --model "$MODEL" \
    --split val \
    --output-dir "$OUT_DIR" \
    --cache-dir  "$CACHE_DIR" \
    --seed 42

echo "--- Canary-Qwen seen-child test ---"
python baselines/audio_model_baseline.py \
    --model "$MODEL" \
    --split test \
    --output-dir "$OUT_DIR" \
    --cache-dir  "$CACHE_DIR" \
    --seed 42

echo "Done: $(date)"
