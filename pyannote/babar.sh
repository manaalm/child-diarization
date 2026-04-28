#!/bin/bash
#SBATCH -c 1
#SBATCH -t 12:00:00
#SBATCH -p ou_bcs_normal,pi_satra
#SBATCH --mem=40G
#SBATCH --requeue
#SBATCH --gres=gpu:1
#SBATCH -o logs/babar_ecapa_%j.out
#SBATCH -e logs/babar_ecapa_%j.err
set -euo pipefail

source /orcd/home/002/manaal/miniforge3/etc/profile.d/conda.sh
conda activate child-vocalizations

module load ffmpeg/5.1.4
export LD_LIBRARY_PATH="/orcd/software/community/001/spack/pkg/ffmpeg/5.1.4/6kcopsg/lib:${LD_LIBRARY_PATH:-}"

python unified.py \
    --diarizer babar \
    --babar-batch-size 32