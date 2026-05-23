#!/bin/bash
#SBATCH --job-name=ecapa_child
#SBATCH --output=logs/adult/ecapa_child_%j.out
#SBATCH --error=logs/adult/ecapa_child_%j.err
#SBATCH --time=05:00:00
#SBATCH --partition=pi_satra
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G

# spec-021 US4 T072: ECAPA-TDNN fine-tune on TinyVox + Providence child speech.

set -euo pipefail
cd /orcd/scratch/orcd/008/manaal/child-adult-diarization

source specs/021-post-thesis-future-work/scripts/slurm_env_template.sh
export PYTHONUNBUFFERED=1

REPO=$(pwd)
PAIRS=${REPO}/models/ecapa_child_finetune/speaker_pair_manifest.csv
OUT=${REPO}/models/ecapa_child_finetune

mkdir -p "${OUT}" logs/adult

echo "=== ECAPA-TDNN child fine-tune (job ${SLURM_JOB_ID:-local}) ==="
echo "  pairs: ${PAIRS}"
echo "  out:   ${OUT}"
nvidia-smi -L 2>&1 | head -2 || true

python pyannote/scripts/fit_ecapa_child.py \
  --pairs "${PAIRS}" \
  --out "${OUT}" \
  --epochs 10 \
  --lr 1e-4 \
  --batch 64 \
  --n-batches 200 \
  --margin 0.2 \
  --scale 30.0

echo "=== ECAPA fine-tune done ==="
ls -la "${OUT}"
