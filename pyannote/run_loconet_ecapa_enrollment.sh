#!/bin/bash
#SBATCH --job-name=loconet_ecapa
#SBATCH -c 8
#SBATCH -t 24:00:00
#SBATCH -p ou_bcs_normal,pi_satra
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH -o logs/enrollment/loconet_ecapa_%j.out
#SBATCH -e logs/enrollment/loconet_ecapa_%j.out

# LocoNet + ECAPA speaker-identity enrollment pipeline.
# Runs LocoNet per-track on every SAILS BIDS clip, then uses ECAPA cosine
# similarity against a train-split reference to identify the target child.
# Requires: video/LoCoNet_ASD/pytorch_model.bin, video/.venv active.
# Output: video_asd_ecapa_enrollment_runs/loconet_ecapa/

set -euo pipefail
source /orcd/home/002/manaal/miniforge3/etc/profile.d/conda.sh
conda activate child-vocalizations
module load ffmpeg/5.1.4
export LD_LIBRARY_PATH="/orcd/software/community/001/spack/pkg/ffmpeg/5.1.4/6kcopsg/lib:${LD_LIBRARY_PATH:-}"
export HF_TOKEN=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

cd /orcd/scratch/orcd/008/manaal/child-adult-diarization

mkdir -p logs/enrollment
echo "Start: $(date)"
python pyannote/unified.py --diarizer loconet_ecapa
echo "Done: $(date)"
