#!/bin/bash
#SBATCH --job-name=ts_talknet_enroll
#SBATCH -c 8
#SBATCH -t 12:00:00
#SBATCH -p ou_bcs_normal,pi_satra
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH -o logs/enrollment/ts_talknet_enroll_%j.out
#SBATCH -e logs/enrollment/ts_talknet_enroll_%j.out

# Run TS-TalkNet diarizer through unified.py (ECAPA enrollment pipeline).
# Requires: video/TS-TalkNet cloned, video/pretrain/ts_talknet.model checkpoint,
#           preprocessed mp4s at derivatives/preprocessed/*.._desc-processed_beh.mp4.
# Output: video_asd_ecapa_enrollment_runs/ts_talknet/

set -euo pipefail
source /orcd/home/002/manaal/miniforge3/etc/profile.d/conda.sh
conda activate child-vocalizations
module load ffmpeg/5.1.4
export LD_LIBRARY_PATH="/orcd/software/community/001/spack/pkg/ffmpeg/5.1.4/6kcopsg/lib:${LD_LIBRARY_PATH:-}"
export HF_TOKEN=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

cd /orcd/scratch/orcd/008/manaal/child-adult-diarization

mkdir -p logs/enrollment
echo "Start: $(date)"
python pyannote/unified.py --diarizer ts_talknet
echo "Done: $(date)"
