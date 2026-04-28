#!/bin/bash
#SBATCH -t 2:00:00
#SBATCH -p ou_bcs_normal,pi_satra
#SBATCH --mem=16G
#SBATCH -c 2
#SBATCH -o logs/mil/weak_diar_%j.out
#SBATCH -e logs/mil/weak_diar_%j.err

set -euo pipefail
export PYTHONUNBUFFERED=1
source /orcd/home/002/manaal/miniforge3/etc/profile.d/conda.sh
conda activate child-vocalizations

REPO=/orcd/scratch/orcd/008/manaal/child-adult-diarization
cd "$REPO"

python mil/eval_weak_diarization.py \
    --results-dir mil/mil_results/seg_mil \
    --split-csv   whisper-modeling/seen_child_splits/test.csv \
    --rttm-cache  whisper-modeling/usc_sail_rttm_cache \
    --output      mil/mil_results/seg_mil/weak_diarization_eval.csv

echo "Done. Output: mil/mil_results/seg_mil/weak_diarization_eval.csv"
