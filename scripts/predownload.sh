#!/bin/bash
#SBATCH --job-name=hf_download
#SBATCH -c 2
#SBATCH -t 0:30:00
#SBATCH -p ou_bcs_normal,pi_satra
#SBATCH --mem=8G
#SBATCH -o logs/predownload_%j.out
#SBATCH -e logs/predownload_%j.err

source /orcd/home/002/manaal/miniforge3/etc/profile.d/conda.sh
conda activate child-vocalizations
cd /orcd/scratch/orcd/008/manaal/child-adult-diarization
unset TRANSFORMERS_OFFLINE
unset HF_HUB_OFFLINE
python scripts/predownload_whisper.py
