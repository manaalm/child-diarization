#!/bin/bash
#SBATCH --job-name=mil_eval
#SBATCH -c 4
#SBATCH -t 4:00:00
#SBATCH -p ou_bcs_normal,pi_satra
#SBATCH --mem=40G
#SBATCH --gres=gpu:1
#SBATCH -o logs/mil/eval_%j.out
#SBATCH -e logs/mil/eval_%j.err

# Evaluate both MIL checkpoints on the test split (T012 + T013 in 002-mil-workflow).
# Writes test_metrics_tuned.json, test_predictions.csv, test_metrics_by_timepoint.csv
# to each variant's result dir.
#
# Usage: sbatch mil/slurm/eval_mil.sh

set -euo pipefail

export PATH="$HOME/.local/bin:$PATH"
source /orcd/home/002/manaal/miniforge3/etc/profile.d/conda.sh
conda activate child-vocalizations

REPO=/home/manaal/orcd/scratch/child-adult-diarization
cd "$REPO"
mkdir -p logs/mil

echo "=== MIL evaluation (job ${SLURM_JOB_ID}) ==="
echo "Start time: $(date)"

for VARIANT in wavlm_mil whisper_mil; do
    CKPT="mil/mil_results/${VARIANT}/best_checkpoint.pt"
    CFG="mil/mil_results/${VARIANT}/config.json"
    if [[ ! -f "$CKPT" ]]; then
        echo "ERROR: checkpoint not found: $CKPT" >&2
        exit 1
    fi
    echo "--- Evaluating ${VARIANT} ---"
    python mil/mil_evaluate.py --checkpoint "$CKPT" --config "$CFG"
done

echo "Done. End time: $(date)"
