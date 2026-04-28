#!/bin/bash
#SBATCH -J mc_suppressor
#SBATCH -p ou_bcs_normal,pi_satra
#SBATCH -t 1:00:00
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH -o logs/evaluation/suppressor_%j.out
#SBATCH -e logs/evaluation/suppressor_%j.err

set -e
cd /orcd/scratch/orcd/008/manaal/child-adult-diarization
source /home/manaal/miniforge3/etc/profile.d/conda.sh
conda activate child-vocalizations

mkdir -p logs/evaluation mil/mil_results/multi_child_suppressor

export HF_HOME=/orcd/scratch/orcd/008/manaal/.cache/huggingface

python evaluation/multi_child_suppressor.py "$@"
