#!/bin/bash
#SBATCH -J encoder_xc_bids
#SBATCH -p ou_bcs_normal,pi_satra
#SBATCH -t 23:30:00
#SBATCH -c 4
#SBATCH --mem=40G
#SBATCH --requeue
#SBATCH --gres=gpu:1
#SBATCH -o logs/baselines/encoder_xc_bids_%j.out
#SBATCH -e logs/baselines/encoder_xc_bids_%j.err

# Cross-child BIDS-corrected encoder grid retrain.
# Uses baselines/splits/ (BIDS-corrected since 2026-05-12) as the cross-child
# train/val/test pool. Writes to a fresh results dir so the legacy
# baselines/baseline_results/ values stay preserved.

set -euo pipefail

source /home/manaal/miniforge3/etc/profile.d/conda.sh
conda activate child-vocalizations
unset HF_TOKEN HUGGINGFACE_HUB_TOKEN HF_HUB_TOKEN 2>/dev/null || true
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export HF_HOME=/orcd/scratch/orcd/008/manaal/.cache/huggingface

cd /orcd/scratch/orcd/008/manaal/child-adult-diarization
mkdir -p logs/baselines

echo "Start: $(date)"
python baselines/baseline_encoders.py \
    --all-experiments \
    --results-root ./baselines/baseline_results_cross_child_bids
echo "Done: $(date)"
