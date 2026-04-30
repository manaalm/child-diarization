#!/bin/bash
#SBATCH -c 4
#SBATCH -t 12:00:00
#SBATCH -p ou_bcs_normal,pi_satra
#SBATCH --mem=40G
#SBATCH --requeue
#SBATCH --gres=gpu:1
#SBATCH -o logs/pseudo_frame/train_%j.out
#SBATCH -e logs/pseudo_frame/train_%j.err

# Usage: sbatch pseudo_frame/slurm/train_pseudo.sh [config.yaml]

set -euo pipefail

CONFIG=${1:-pseudo_frame/configs/wavlm_pseudo.yaml}

export PATH="$HOME/.local/bin:$PATH"
source /orcd/home/002/manaal/miniforge3/etc/profile.d/conda.sh
conda activate child-vocalizations

REPO=/orcd/scratch/orcd/008/manaal/child-adult-diarization
cd "$REPO"
mkdir -p logs/pseudo_frame

echo "=== pseudo-frame training: $CONFIG (job $SLURM_JOB_ID) ==="

# Step 1: build pseudo-labels if missing
if [ ! -f pseudo_frame/pseudo_labels/index.csv ] || [ "$(wc -l < pseudo_frame/pseudo_labels/index.csv)" -lt 2000 ]; then
    echo "--- building pseudo-labels ---"
    python pseudo_frame/build_pseudo_labels.py
fi

# Step 2: train
echo "--- training ---"
python pseudo_frame/pseudo_train.py --config "$CONFIG"

# Step 3: evaluate on test
VARIANT=$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG'))['variant_name'])")
CKPT="pseudo_frame/results/$VARIANT/best_checkpoint.pt"
echo "--- evaluating ---"
python pseudo_frame/pseudo_evaluate.py --checkpoint "$CKPT" --split test

echo "=== Done ==="
