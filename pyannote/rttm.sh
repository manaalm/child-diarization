#!/bin/bash
#SBATCH -c 1
#SBATCH -t 12:00:00
#SBATCH -p ou_bcs_normal,pi_satra
#SBATCH --mem=40G
#SBATCH --requeue
#SBATCH --gres=gpu:1
#SBATCH -o logs/rttm/file_%j.out
#SBATCH -e logs/rttm/file_%j.err
set -euo pipefail
source /orcd/home/002/manaal/miniforge3/etc/profile.d/conda.sh
conda activate child-vocalizations
module load ffmpeg/5.1.4
export LD_LIBRARY_PATH="/orcd/software/community/001/spack/pkg/ffmpeg/5.1.4/6kcopsg/lib:${LD_LIBRARY_PATH:-}"

# # Playlogue — USC-SAIL
# python unified_rttm.py \
#     --dataset playlogue \
#     --audio-dir /home/manaal/orcd/scratch/child-adult-diarization/playlogue/audio \
#     --rttm-dir  /home/manaal/orcd/scratch/child-adult-diarization/playlogue/rttm_norm \
#     --diarizer usc_sail

# # Playlogue — pyannote
# python unified_rttm.py \
#     --dataset playlogue \
#     --audio-dir /home/manaal/orcd/scratch/child-adult-diarization/playlogue/audio \
#     --rttm-dir  /home/manaal/orcd/scratch/child-adult-diarization/playlogue/rttm_norm \
#     --diarizer pyannote



# 


# Playlogue — BabAR
python unified_rttm.py \
    --dataset playlogue \
    --audio-dir /home/manaal/orcd/scratch/child-adult-diarization/playlogue/audio \
    --rttm-dir  /home/manaal/orcd/scratch/child-adult-diarization/playlogue/rttm_norm \
    --diarizer babar \
    --babar-dir /home/manaal/orcd/scratch/child-adult-diarization/BabAR/ \
    --babar-batch-size 1
    
# Providence — BabAR
python unified_rttm.py \
    --dataset providence \
    --audio-dir /home/manaal/orcd/scratch/child-adult-diarization/providence/audio \
    --rttm-dir  /home/manaal/orcd/scratch/child-adult-diarization/providence/rttm \
    --diarizer babar \
    --babar-dir /home/manaal/orcd/scratch/child-adult-diarization/BabAR/ \
    --babar-batch-size 1