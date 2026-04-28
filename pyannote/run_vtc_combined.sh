#!/bin/bash
#SBATCH -c 4
#SBATCH -t 8:00:00
#SBATCH -p ou_bcs_normal,pi_satra
#SBATCH --mem=40G
#SBATCH --gres=gpu:1
#SBATCH --requeue
#SBATCH -o ../logs/enrollment/vtc_combined_%j.out
#SBATCH -e ../logs/enrollment/vtc_combined_%j.err

set -euo pipefail
source /orcd/home/002/manaal/miniforge3/etc/profile.d/conda.sh
conda activate child-vocalizations

cd /home/manaal/orcd/scratch/child-adult-diarization/pyannote

mkdir -p ../logs/enrollment
mkdir -p ../vtc_combined_runs

echo "=== VTC Combined Features ==="
python vtc_combined.py --results-dir ../vtc_combined_runs/

echo "=== VTC Combined done — running ensemble ==="
python ensemble_combined.py --results-dir ../ensemble_runs/
