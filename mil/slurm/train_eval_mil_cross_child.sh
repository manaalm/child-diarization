#!/bin/bash
#SBATCH --job-name=mil_cross_child
#SBATCH -c 4
#SBATCH -t 48:00:00
#SBATCH -p ou_bcs_normal,pi_satra
#SBATCH --mem=40G
#SBATCH --requeue
#SBATCH --gres=gpu:1
#SBATCH -o logs/mil/cross_child_%j.out
#SBATCH -e logs/mil/cross_child_%j.err

# Train WavLM-MIL and Whisper-MIL on the cross-child split (baselines/splits/),
# then evaluate both on the cross-child test set.
# Results: mil/mil_results/{wavlm_mil_cross_child,whisper_mil_cross_child}/

set -euo pipefail

export PATH="$HOME/.local/bin:$PATH"
source /orcd/home/002/manaal/miniforge3/etc/profile.d/conda.sh
conda activate child-vocalizations

REPO=/home/manaal/orcd/scratch/child-adult-diarization
cd "$REPO"
mkdir -p logs/mil

echo "=== Cross-child MIL train+eval (job $SLURM_JOB_ID) ==="
echo "Start: $(date)"

echo "--- Training wavlm_mil_cross_child ---"
python mil/mil_train.py --config mil/configs/wavlm_mil_cross_child.yaml

echo "--- Training whisper_mil_cross_child ---"
python mil/mil_train.py --config mil/configs/whisper_mil_cross_child.yaml

echo "--- Evaluating wavlm_mil_cross_child ---"
python mil/mil_evaluate.py \
    --checkpoint mil/mil_results/wavlm_mil_cross_child/best_checkpoint.pt \
    --config     mil/mil_results/wavlm_mil_cross_child/config.json

echo "--- Evaluating whisper_mil_cross_child ---"
python mil/mil_evaluate.py \
    --checkpoint mil/mil_results/whisper_mil_cross_child/best_checkpoint.pt \
    --config     mil/mil_results/whisper_mil_cross_child/config.json

echo "=== Done. End: $(date) ==="
