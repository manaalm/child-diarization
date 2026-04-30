#!/bin/bash
#SBATCH -c 4
#SBATCH -t 12:00:00
#SBATCH -p ou_bcs_normal,pi_satra
#SBATCH --mem=40G
#SBATCH --requeue
#SBATCH --gres=gpu:1
#SBATCH -o logs/pseudo_frame/distill_c1_%j.out
#SBATCH -e logs/pseudo_frame/distill_c1_%j.err

# Spec-016 follow-up #8: C1 self-distillation pseudo-label loop.
#   Step 1: distill C1 USC-SAIL synth-only frame classifier predictions
#           into pseudo_frame/pseudo_labels_c1/index.csv (~10 min on GPU)
#   Step 2: train WavLM-Base+ frame classifier on the C1-distilled labels
#   Step 3: evaluate on the test split (clip-level + frame-localization)
#
# Usage: sbatch pseudo_frame/slurm/distill_c1_train.sh

set -euo pipefail

export PATH="$HOME/.local/bin:$PATH"
source /orcd/home/002/manaal/miniforge3/etc/profile.d/conda.sh
conda activate child-vocalizations

REPO=/orcd/scratch/orcd/008/manaal/child-adult-diarization
cd "$REPO"
mkdir -p logs/pseudo_frame

CKPT=whisper-modeling/checkpoints/whisper_base_synth/epoch=17-val_loss=0.235.ckpt
CONFIG=pseudo_frame/configs/wavlm_pseudo_c1distill.yaml
INDEX=pseudo_frame/pseudo_labels_c1/index.csv

echo "=== C1 self-distillation (job $SLURM_JOB_ID) ==="

if [ ! -f "$CKPT" ]; then
    echo "ERROR: C1 checkpoint missing: $CKPT" >&2
    exit 2
fi

# Step 1: distill if not already done
if [ ! -f "$INDEX" ] || [ "$(wc -l < "$INDEX")" -lt 2000 ]; then
    echo "--- distilling C1 pseudo-labels ---"
    PYTHONPATH=. python pseudo_frame/distill_c1_pseudo_labels.py
else
    echo "--- C1 distill index already present ($INDEX): skipping ---"
fi

# Step 2: train
echo "--- training pseudo-frame on C1-distilled labels ---"
python pseudo_frame/pseudo_train.py --config "$CONFIG"

# Step 3: evaluate on test
VARIANT=$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG'))['variant_name'])")
PSEUDO_CKPT="pseudo_frame/results/$VARIANT/best_checkpoint.pt"
echo "--- evaluating ---"
python pseudo_frame/pseudo_evaluate.py --checkpoint "$PSEUDO_CKPT" --split test

echo "=== Done ==="
