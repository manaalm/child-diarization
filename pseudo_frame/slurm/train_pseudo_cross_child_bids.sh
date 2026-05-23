#!/bin/bash
#SBATCH -c 4
#SBATCH -t 12:00:00
#SBATCH -p ou_bcs_normal,pi_satra
#SBATCH --mem=40G
#SBATCH --requeue
#SBATCH --gres=gpu:1
#SBATCH -o logs/pseudo_frame/cross_child_bids_%j.out
#SBATCH -e logs/pseudo_frame/cross_child_bids_%j.err

# Cross-child BIDS-corrected pseudo-frame retrain.
# Trains WavLM-Base+ and Whisper-small pseudo-frame heads on the
# BIDS-corrected cross-child train pool (baselines/splits/, n=2128 train /
# 444 val / 742 test).

set -euo pipefail

export PATH="$HOME/.local/bin:$PATH"
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
source /orcd/home/002/manaal/miniforge3/etc/profile.d/conda.sh
conda activate child-vocalizations

REPO=/orcd/scratch/orcd/008/manaal/child-adult-diarization
cd "$REPO"
mkdir -p logs/pseudo_frame

echo "=== pseudo-frame cross-child BIDS train (job $SLURM_JOB_ID) ==="
echo "Start: $(date)"

if [ ! -f pseudo_frame/pseudo_labels/index.csv ] \
   || [ "$(wc -l < pseudo_frame/pseudo_labels/index.csv)" -lt 2000 ]; then
    echo "--- building pseudo-labels ---"
    python pseudo_frame/build_pseudo_labels.py
fi

for CONFIG in \
    pseudo_frame/configs/wavlm_pseudo_cross_child.yaml \
    pseudo_frame/configs/whisper_pseudo_cross_child.yaml; do
    echo "--- training $CONFIG ---"
    python pseudo_frame/pseudo_train.py --config "$CONFIG"
    VARIANT=$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG'))['variant_name'])")
    CKPT="pseudo_frame/results/$VARIANT/best_checkpoint.pt"
    echo "--- evaluating $VARIANT ---"
    python pseudo_frame/pseudo_evaluate.py --checkpoint "$CKPT" --split test
done

echo "=== Done. End: $(date) ==="
