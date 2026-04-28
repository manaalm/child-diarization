#!/bin/bash
#SBATCH -c 4
#SBATCH -t 8:00:00
#SBATCH -p pi_satra,ou_bcs_normal
#SBATCH --mem=40G
#SBATCH --requeue
#SBATCH --gres=gpu:1
#SBATCH -o logs/rttm/vbx_%j.out
#SBATCH -e logs/rttm/vbx_%j.err

# Run VBx speaker diarization on Playlogue and Providence.
#
# VBx uses pyannote/segmentation-3.0 (VAD) and pyannote/embedding (ECAPA).
# Models are loaded once and reused across all files in each dataset.
# Anonymous SPEAKER_XX labels are resolved to CHI/ADT via GT-overlap mapping.
#
# Prerequisites (run once before submitting):
#   cd /home/manaal/orcd/scratch/child-adult-diarization/VBx && uv sync
#   export HF_TOKEN=<your_token>   # needed for pyannote models

set -euo pipefail
export HF_TOKEN=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
source /orcd/home/002/manaal/miniforge3/etc/profile.d/conda.sh
conda activate child-vocalizations
module load ffmpeg/5.1.4
export LD_LIBRARY_PATH="/orcd/software/community/001/spack/pkg/ffmpeg/5.1.4/6kcopsg/lib:${LD_LIBRARY_PATH:-}"

AUDIO_PLAY=/home/manaal/orcd/scratch/child-adult-diarization/playlogue/audio
RTTM_PLAY=/home/manaal/orcd/scratch/child-adult-diarization/playlogue/rttm_norm
AUDIO_PROV=/home/manaal/orcd/scratch/child-adult-diarization/providence/audio
RTTM_PROV=/home/manaal/orcd/scratch/child-adult-diarization/providence/rttm

cd /home/manaal/orcd/scratch/child-adult-diarization/pyannote

# ── Playlogue — VBx ──────────────────────────────────────────────────────
python unified_rttm.py \
    --dataset playlogue \
    --audio-dir "$AUDIO_PLAY" \
    --rttm-dir  "$RTTM_PLAY" \
    --diarizer vbx

# ── Providence — VBx ─────────────────────────────────────────────────────
python unified_rttm.py \
    --dataset providence \
    --audio-dir "$AUDIO_PROV" \
    --rttm-dir  "$RTTM_PROV" \
    --diarizer vbx
