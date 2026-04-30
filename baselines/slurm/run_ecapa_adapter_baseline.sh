#!/bin/bash
#SBATCH --job-name=ecapa_adapter
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH -c 4
#SBATCH -t 6:00:00
#SBATCH -p ou_bcs_normal,pi_satra
#SBATCH -o logs/baselines/ecapa_adapter_%j.out
#SBATCH -e logs/baselines/ecapa_adapter_%j.out

# ECAPA Adapter Triplet Baseline (US7, spec-013, Tier 4)
# Fine-tunes 192->64->192 adapter on ECAPA with triplet loss on KCHI segments.
# ~1h: triplet pool build (~20min) + adapter training (30 epochs, ~10min)
#      + prototype rebuild (~20min) + val/test scoring (~20min each).

echo "Start: $(date)"
echo "Node: $(hostname)"

source /home/manaal/miniforge3/etc/profile.d/conda.sh
conda activate child-vocalizations

export HF_HOME=/orcd/scratch/orcd/008/manaal/.cache/huggingface
export TORCH_HOME=/orcd/scratch/orcd/008/manaal/.cache/torch

cd /orcd/scratch/orcd/008/manaal/child-adult-diarization
mkdir -p logs/baselines

echo "=== ECAPA Adapter (val) ==="
python baselines/ecapa_adapter_baseline.py \
    --split val \
    --device cuda \
    --epochs 30 \
    --n-triplets 1024 \
    --margin 0.3

echo "=== ECAPA Adapter (test) ==="
python baselines/ecapa_adapter_baseline.py \
    --split test \
    --device cuda \
    --epochs 30 \
    --n-triplets 1024 \
    --margin 0.3

echo "Done: $(date)"
