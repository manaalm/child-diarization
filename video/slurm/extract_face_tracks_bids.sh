#!/bin/bash
#SBATCH --job-name=face_tracks_bids
#SBATCH -c 4
#SBATCH -t 8:00:00
#SBATCH -p ou_bcs_normal,pi_satra
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --requeue
#SBATCH -o logs/video/extract_face_tracks_bids_%j.out
#SBATCH -e logs/video/extract_face_tracks_bids_%j.out

# Batch S3FD face-track extraction for the 963 BIDS clips missing per-frame
# face cache entries in av_fusion/face_track_cache/. Resume-safe (skips
# already-cached clips). Output is the canonical {md5(BidsProcessed)}.json
# format consumed by pseudo_frame/extract_mouth_motion.py and the
# speaker-informed AV pipeline.

set -euo pipefail

cd /orcd/scratch/orcd/008/manaal/child-adult-diarization
mkdir -p logs/video av_fusion/face_track_cache

module load ffmpeg/5.1.4 || true
export LD_LIBRARY_PATH="/orcd/software/community/001/spack/pkg/ffmpeg/5.1.4/6kcopsg/lib:${LD_LIBRARY_PATH:-}"

UV_PYTHON="video/.venv/bin/python"

echo "=== S3FD face-track extraction (BIDS expansion) job $SLURM_JOB_ID ==="
echo "Start: $(date)"
nvidia-smi | head -8 || true

$UV_PYTHON video/extract_face_tracks_bids.py --split all --device cuda

echo "=== Done $(date) ==="
