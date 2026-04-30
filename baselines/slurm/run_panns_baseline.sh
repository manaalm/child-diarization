#!/bin/bash
#SBATCH --job-name=panns_baseline
#SBATCH --mem=32G
#SBATCH -c 4
#SBATCH -t 8:00:00
#SBATCH -p ou_bcs_normal,pi_satra
#SBATCH -o logs/baselines/panns_%j.out
#SBATCH -e logs/baselines/panns_%j.out

# PANNS CNN14 runs on CPU (no GPU needed — embeddings are fast on CPU)
echo "Start: $(date)"
echo "Node: $(hostname)"

source /home/manaal/miniforge3/etc/profile.d/conda.sh
conda activate child-vocalizations

export HF_HOME=/orcd/scratch/orcd/008/manaal/.cache/huggingface
export TORCH_HOME=/orcd/scratch/orcd/008/manaal/.cache/torch

cd /orcd/scratch/orcd/008/manaal/child-adult-diarization

# Install panns_inference if not already present
pip install panns_inference --quiet

# ── Seen-child split (train LR head + eval) ──────────────────────────────
echo "=== PANNS CNN14 (seen-child val) ==="
python baselines/panns_baseline.py --split val --seed 42

echo "=== PANNS CNN14 (seen-child test) ==="
python baselines/panns_baseline.py --split test --seed 42

# ── Cross-child split (reuse seen-child LR head, no retraining) ──────────
echo "=== PANNS CNN14 (cross-child val) ==="
python baselines/panns_baseline.py --split val \
    --splits-dir baselines/splits \
    --output-dir baselines/panns_baseline_runs/cnn14_cross_child \
    --lr-weights baselines/panns_baseline_runs/cnn14/lr_weights.npz \
    --seed 42

echo "=== PANNS CNN14 (cross-child test) ==="
python baselines/panns_baseline.py --split test \
    --splits-dir baselines/splits \
    --output-dir baselines/panns_baseline_runs/cnn14_cross_child \
    --lr-weights baselines/panns_baseline_runs/cnn14/lr_weights.npz \
    --seed 42

echo "Done: $(date)"
