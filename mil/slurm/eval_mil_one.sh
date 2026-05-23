#!/bin/bash
#SBATCH --job-name=mil_eval_one
#SBATCH -c 4
#SBATCH -t 2:00:00
#SBATCH -p ou_bcs_normal,pi_satra
#SBATCH --mem=40G
#SBATCH --gres=gpu:1
#SBATCH -o logs/mil/eval_one_%j.out
#SBATCH -e logs/mil/eval_one_%j.err

# Evaluate a single MIL variant on the test split (spec-022 BIDS rerun follow-up).
# Usage: sbatch mil/slurm/eval_mil_one.sh <variant_name>
#   e.g.  sbatch mil/slurm/eval_mil_one.sh wavlm_mil

set -euo pipefail

VARIANT="${1:?Usage: sbatch eval_mil_one.sh <variant_name>}"

export PATH="$HOME/.local/bin:$PATH"
source /orcd/home/002/manaal/miniforge3/etc/profile.d/conda.sh
conda activate child-vocalizations

# transformers >= 4.57 has_file() network bug — keep offline
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

REPO=/home/manaal/orcd/scratch/child-adult-diarization
cd "$REPO"
mkdir -p logs/mil

CKPT="mil/mil_results/${VARIANT}/best_checkpoint.pt"
CFG="mil/mil_results/${VARIANT}/config.json"
if [[ ! -f "$CKPT" ]]; then
    echo "ERROR: checkpoint not found: $CKPT" >&2
    exit 1
fi
if [[ ! -f "$CFG" ]]; then
    echo "ERROR: config not found: $CFG" >&2
    exit 1
fi

echo "=== MIL evaluation: ${VARIANT} (job ${SLURM_JOB_ID}) ==="
echo "Start time: $(date)"
python mil/mil_evaluate.py --checkpoint "$CKPT" --config "$CFG"
echo "Done. End time: $(date)"
