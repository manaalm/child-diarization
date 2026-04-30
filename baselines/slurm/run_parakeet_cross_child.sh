#!/bin/bash
#SBATCH --job-name=parakeet_cross
#SBATCH --gres=gpu:1
#SBATCH -t 4:00:00
#SBATCH --mem=48G
#SBATCH -c 4
#SBATCH -p ou_bcs_normal,pi_satra
#SBATCH -o logs/baselines/parakeet_cross_%j.out
#SBATCH -e logs/baselines/parakeet_cross_%j.out

# Parakeet TDT gap-ratio baseline on the cross-child split (baselines/splits/).
# Runs val (threshold tuning) then test in one job.
# Results: baselines/parakeet_baseline_runs/parakeet_tdt_0.6b_v2_cross_child/

set -euo pipefail

echo "Start: $(date)"
echo "Node: $(hostname)  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo none)"

source /home/manaal/miniforge3/etc/profile.d/conda.sh
conda activate child-vocalizations

export HF_HOME=/orcd/scratch/orcd/008/manaal/.cache/huggingface
export CUDA_LAUNCH_BLOCKING=1

REPO=/orcd/scratch/orcd/008/manaal/child-adult-diarization
cd "$REPO"
mkdir -p logs/baselines

OUT_DIR=baselines/parakeet_baseline_runs/parakeet_tdt_0.6b_v2_cross_child

echo "--- Parakeet cross-child val ---"
python baselines/parakeet_baseline.py \
    --split val \
    --splits-dir baselines/splits \
    --output-dir "$OUT_DIR" \
    --batch-size 4 \
    --seed 42

echo "--- Parakeet cross-child test ---"
python baselines/parakeet_baseline.py \
    --split test \
    --splits-dir baselines/splits \
    --output-dir "$OUT_DIR" \
    --batch-size 4 \
    --seed 42

echo "Done: $(date)"
