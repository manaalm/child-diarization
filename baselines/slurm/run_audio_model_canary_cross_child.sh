#!/bin/bash
#SBATCH --job-name=canary_cross
#SBATCH --gres=gpu:1
#SBATCH -t 12:00:00
#SBATCH --mem=64G
#SBATCH -c 4
#SBATCH -p ou_bcs_normal,pi_satra
#SBATCH -o logs/baselines/canary_cross_%j.out
#SBATCH -e logs/baselines/canary_cross_%j.out

# nvidia/canary-qwen-2.5b on the cross-child split.
# Runs val (threshold tuning) then test in one job.
# Results: baselines/audio_model_baseline_runs/canary_qwen_2_5b_cross_child/
# Cache:   baselines/audio_model_cache/canary_qwen_2_5b_cross_child/

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
OUT_DIR=baselines/audio_model_baseline_runs/canary_qwen_2_5b_cross_child
CACHE_DIR=baselines/audio_model_cache/canary_qwen_2_5b_cross_child

echo "--- Canary-Qwen cross-child val ---"
python baselines/audio_model_baseline.py \
    --model "$MODEL" \
    --split val \
    --splits-dir baselines/splits \
    --output-dir "$OUT_DIR" \
    --cache-dir  "$CACHE_DIR" \
    --seed 42

echo "--- Canary-Qwen cross-child test ---"
python baselines/audio_model_baseline.py \
    --model "$MODEL" \
    --split test \
    --splits-dir baselines/splits \
    --output-dir "$OUT_DIR" \
    --cache-dir  "$CACHE_DIR" \
    --seed 42

echo "Done: $(date)"
