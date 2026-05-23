#!/bin/bash
#SBATCH -J score_bark
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH --time=00:15:00
#SBATCH --mem=24G
#SBATCH -o logs/synth/score_bark_%j.out
#SBATCH -e logs/synth/score_bark_%j.err

set -euo pipefail
cd /home/manaal/orcd/scratch/child-adult-diarization
source /orcd/home/002/manaal/miniforge3/etc/profile.d/conda.sh
conda activate child-vocalizations

export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
unset HF_TOKEN HUGGINGFACE_HUB_TOKEN HF_HUB_TOKEN

python synth/synth_voice/score_bark_with_mil.py
