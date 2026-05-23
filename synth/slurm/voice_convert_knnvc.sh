#!/bin/bash
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH -t 12:00:00
#SBATCH -p ou_bcs_normal,pi_satra
#SBATCH --mem=24G
#SBATCH --requeue
#SBATCH -o logs/synth/voice_convert_knnvc_%j.out
#SBATCH -e logs/synth/voice_convert_knnvc_%j.err

set -euo pipefail

source /orcd/home/002/manaal/miniforge3/etc/profile.d/conda.sh
conda activate knnvc

REPO=/orcd/scratch/orcd/008/manaal/child-adult-diarization
cd "$REPO"
mkdir -p logs/synth

export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

echo "=== KNN-VC bulk voice conversion (job $SLURM_JOB_ID) ==="
python synth/scripts/voice_convert_knnvc.py --n-per-child 10 --seed 42
echo "=== Done ==="
