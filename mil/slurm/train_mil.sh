#!/bin/bash
#SBATCH -c 4
#SBATCH -t 8:00:00
#SBATCH -p ou_bcs_normal,pi_satra
#SBATCH --mem=40G
#SBATCH --requeue
#SBATCH --gres=gpu:1
#SBATCH -o logs/mil/train_%j.out
#SBATCH -e logs/mil/train_%j.err

# Usage: sbatch mil/slurm/train_mil.sh mil/configs/wavlm_mil.yaml

set -euo pipefail

CONFIG=${1:?"Usage: sbatch train_mil.sh <config.yaml>"}

export PATH="$HOME/.local/bin:$PATH"
source /orcd/home/002/manaal/miniforge3/etc/profile.d/conda.sh
conda activate child-vocalizations

REPO=/home/manaal/orcd/scratch/child-adult-diarization
cd "$REPO"
mkdir -p logs/mil

echo "=== MIL training: $CONFIG (job $SLURM_JOB_ID) ==="
python mil/mil_train.py --config "$CONFIG"
echo "=== Done ==="
