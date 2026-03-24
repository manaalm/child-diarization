#!/bin/bash

#SBATCH -c 1
#SBATCH -t 12:00:00
#SBATCH -p ou_bcs_normal,pi_satra
#SBATCH --mem=40G
#SBATCH --requeue
#SBATCH --gres=gpu:1
#SBATCH -o logs/pyannote_ecapa_%j.out
#SBATCH -e logs/pyannote_ecapa_%j.err

set -euo pipefail

source /orcd/home/002/manaal/miniforge3/etc/profile.d/conda.sh
conda activate child-vocalizations

mkdir -p logs


# python make_seen_child_split.py

python pyannote_ecapa.py