#!/bin/bash
#SBATCH --job-name=raw_ecapa
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH -c 4
#SBATCH -t 8:00:00
#SBATCH -p ou_bcs_normal,pi_satra
#SBATCH -o logs/baselines/raw_ecapa_%j.out
#SBATCH -e logs/baselines/raw_ecapa_%j.out

echo "Start: $(date)"
echo "Node: $(hostname)"

source /home/manaal/miniforge3/etc/profile.d/conda.sh
conda activate child-vocalizations

export HF_HOME=/orcd/scratch/orcd/008/manaal/.cache/huggingface
export TORCH_HOME=/orcd/scratch/orcd/008/manaal/.cache/torch

cd /orcd/scratch/orcd/008/manaal/child-adult-diarization

# Seen-child split, all three aggregation modes
# (val then test for each; prototype rebuilt each time to keep runs independent)

echo "=== mean (val) ==="
python baselines/raw_ecapa_baseline.py --mode mean --split val --seed 42

echo "=== mean (test) ==="
python baselines/raw_ecapa_baseline.py --mode mean --split test --seed 42

echo "=== max (val) ==="
python baselines/raw_ecapa_baseline.py --mode max --split val --seed 42

echo "=== max (test) ==="
python baselines/raw_ecapa_baseline.py --mode max --split test --seed 42

echo "=== top3 (val) ==="
python baselines/raw_ecapa_baseline.py --mode top3 --split val --seed 42

echo "=== top3 (test) ==="
python baselines/raw_ecapa_baseline.py --mode top3 --split test --seed 42

echo "Done: $(date)"
