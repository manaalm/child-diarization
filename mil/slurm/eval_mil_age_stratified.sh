#!/bin/bash
#SBATCH -J mil_age_stratified
#SBATCH -t 2:00:00
#SBATCH -p ou_bcs_normal,pi_satra
#SBATCH --mem=16G
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH -o logs/mil/age_stratified_%j.out
#SBATCH -e logs/mil/age_stratified_%j.err

set -euo pipefail
export PYTHONUNBUFFERED=1

source /orcd/home/002/manaal/miniforge3/etc/profile.d/conda.sh
conda activate child-vocalizations

REPO=/orcd/scratch/orcd/008/manaal/child-adult-diarization
cd "$REPO"
mkdir -p logs/mil

for VARIANT in wavlm_mil whisper_mil; do
  for AG in 12_16m 34_38m; do
    OUT="mil/mil_results/${VARIANT}/age_stratified/${AG}/test_metrics_tuned.json"
    if [[ -f "$OUT" ]]; then
      echo "SKIP ${VARIANT} ${AG} (already done)"
      continue
    fi
    echo "=== ${VARIANT} ${AG} ==="
    python mil/mil_age_stratified.py \
      --checkpoint "mil/mil_results/${VARIANT}/best_checkpoint.pt" \
      --config     "mil/mil_results/${VARIANT}/config.json" \
      --age-group  "$AG" \
      --manifest   playlogue/manifest.csv
  done
done

echo "All age-stratified evals done."
