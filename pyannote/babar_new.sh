#!/bin/bash
#SBATCH -c 2
#SBATCH -t 6:00:00
#SBATCH -p ou_bcs_normal,pi_satra
#SBATCH --mem=40G
#SBATCH --requeue
#SBATCH --gres=gpu:1
#SBATCH -o logs/combined_%j.out
#SBATCH -e logs/combined_%j.err
set -euo pipefail

source /orcd/home/002/manaal/miniforge3/etc/profile.d/conda.sh
conda activate child-vocalizations
module load ffmpeg/5.1.4
export LD_LIBRARY_PATH="/orcd/software/community/001/spack/pkg/ffmpeg/5.1.4/6kcopsg/lib:${LD_LIBRARY_PATH:-}"

# python babar_updated.py \
#     --babar-output /home/manaal/orcd/scratch/child-adult-diarization/babar/babar_output \
#     --results-dir /home/manaal/orcd/scratch/child-adult-diarization/babar_combined_runs

python babar_three.py \
    --babar-output /home/manaal/orcd/scratch/child-adult-diarization/babar/babar_output/three \
    --results-dir /home/manaal/orcd/scratch/child-adult-diarization/babar_combined_runs \
    --skip-extraction