#!/bin/bash
#SBATCH -J short_voc_head
#SBATCH -p ou_bcs_normal,pi_satra
#SBATCH -t 4:00:00
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH -o logs/evaluation/short_voc_%j.out
#SBATCH -e logs/evaluation/short_voc_%j.err

set -e
cd /orcd/scratch/orcd/008/manaal/child-adult-diarization
source /home/manaal/miniforge3/etc/profile.d/conda.sh
conda activate child-vocalizations

mkdir -p logs/evaluation mil/mil_results/short_voc_head

export HF_HOME=/orcd/scratch/orcd/008/manaal/.cache/huggingface

python evaluation/short_voc_head.py "$@"
