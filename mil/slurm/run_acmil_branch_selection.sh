#!/bin/bash
#SBATCH -c 4
#SBATCH -t 01:00:00
#SBATCH -p ou_bcs_normal,pi_satra
#SBATCH --mem=20G
#SBATCH --gres=gpu:1
#SBATCH -o logs/mil/branch_selection_%j.out
#SBATCH -e logs/mil/branch_selection_%j.err

# Run no-retrain ACMIL branch-selection eval on one trained ACMIL checkpoint.
# Usage: sbatch mil/slurm/run_acmil_branch_selection.sh <results_dir>

set -euo pipefail

RESULTS_DIR=${1:?"Usage: sbatch run_acmil_branch_selection.sh <results_dir>"}

export PATH="$HOME/.local/bin:$PATH"
export PYTHONUNBUFFERED=1
source /orcd/home/002/manaal/miniforge3/etc/profile.d/conda.sh
conda activate child-vocalizations

REPO=/home/manaal/orcd/scratch/child-adult-diarization
cd "$REPO"

echo "=== branch_selection eval: $RESULTS_DIR (job $SLURM_JOB_ID) ==="
python mil/eval_acmil_branch_selection.py --results-dir "$RESULTS_DIR"
echo "=== Done at $(date) ==="
