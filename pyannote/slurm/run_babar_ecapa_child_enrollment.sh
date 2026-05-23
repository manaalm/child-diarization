#!/bin/bash
#SBATCH --job-name=babar_ecapa_child
#SBATCH --output=logs/adult/babar_ecapa_child_%j.out
#SBATCH --error=logs/adult/babar_ecapa_child_%j.err
#SBATCH --time=02:00:00
#SBATCH --partition=pi_satra
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G

# spec-021 US4 T074: BabAR enrollment using the fine-tuned ECAPA from T073.
set -euo pipefail
cd /orcd/scratch/orcd/008/manaal/child-adult-diarization
source specs/021-post-thesis-future-work/scripts/slurm_env_template.sh
export PYTHONUNBUFFERED=1

REPO=$(pwd)
CKPT=${REPO}/models/ecapa_child_finetune/best.pt

mkdir -p logs/adult
echo "=== BabAR enrollment with fine-tuned ECAPA (job ${SLURM_JOB_ID:-local}) ==="
echo "  ckpt: ${CKPT}"
nvidia-smi -L 2>&1 | head -2 || true

cd "${REPO}/pyannote"
python unified.py --diarizer babar --ecapa-checkpoint "${CKPT}"

echo "=== done ==="
