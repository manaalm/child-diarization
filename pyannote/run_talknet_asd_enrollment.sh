#!/bin/bash
#SBATCH --job-name=talknet_asd_enroll
#SBATCH -c 8
#SBATCH -t 12:00:00
#SBATCH -p ou_bcs_normal,pi_satra
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH -o logs/enrollment/talknet_asd_enroll_%j.out
#SBATCH -e logs/enrollment/talknet_asd_enroll_%j.out

# Run TalkNet diarizer through unified.py (ECAPA enrollment pipeline).
# Requires: video/TalkNet cloned, video/pretrain/talknet_asd.model checkpoint,
#           preprocessed mp4s at derivatives/preprocessed/*.._desc-processed_beh.mp4.
# Output: video_asd_ecapa_enrollment_runs/talknet_asd/

set -euo pipefail
source /orcd/home/002/manaal/miniforge3/etc/profile.d/conda.sh
conda activate child-vocalizations
module load ffmpeg/5.1.4
export LD_LIBRARY_PATH="/orcd/software/community/001/spack/pkg/ffmpeg/5.1.4/6kcopsg/lib:${LD_LIBRARY_PATH:-}"
: "${HF_TOKEN:?HF_TOKEN must be set in environment (was previously hardcoded; rotate at https://huggingface.co/settings/tokens and export before submit)}"

cd /orcd/scratch/orcd/008/manaal/child-adult-diarization

mkdir -p logs/enrollment
echo "Start: $(date)"
python pyannote/unified.py --diarizer talknet_asd
echo "Done: $(date)"
