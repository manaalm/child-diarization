#!/bin/bash
#SBATCH -c 4
#SBATCH -t 12:00:00
#SBATCH -p pi_satra,ou_bcs_normal
#SBATCH --mem=40G
#SBATCH --requeue
#SBATCH --gres=gpu:1
#SBATCH -o logs/rttm/eend_eda_%j.out
#SBATCH -e logs/rttm/eend_eda_%j.err

# Frame-level RTTM accuracy for EEND-EDA on Playlogue + Providence.
# Mirrors the layout in rttm.sh / rttm_vbx.sh.
#
# Prereq: pip install espnet espnet_model_zoo soundfile (in child-vocalizations env)
# Anonymous spk1/spk2/... labels are resolved to CHI/ADT via GT-overlap mapping.

set -euo pipefail
source /orcd/home/002/manaal/miniforge3/etc/profile.d/conda.sh
conda activate child-vocalizations
module load ffmpeg/5.1.4
export LD_LIBRARY_PATH="/orcd/software/community/001/spack/pkg/ffmpeg/5.1.4/6kcopsg/lib:${LD_LIBRARY_PATH:-}"

AUDIO_PLAY=/home/manaal/orcd/scratch/child-adult-diarization/playlogue/audio
RTTM_PLAY=/home/manaal/orcd/scratch/child-adult-diarization/playlogue/rttm_norm
AUDIO_PROV=/home/manaal/orcd/scratch/child-adult-diarization/providence/audio
RTTM_PROV=/home/manaal/orcd/scratch/child-adult-diarization/providence/rttm

cd /home/manaal/orcd/scratch/child-adult-diarization/pyannote

# ── Playlogue — EEND-EDA ─────────────────────────────────────────────────
python unified_rttm.py \
    --dataset playlogue \
    --audio-dir "$AUDIO_PLAY" \
    --rttm-dir  "$RTTM_PLAY" \
    --diarizer eend_eda

# ── Providence — EEND-EDA ────────────────────────────────────────────────
python unified_rttm.py \
    --dataset providence \
    --audio-dir "$AUDIO_PROV" \
    --rttm-dir  "$RTTM_PROV" \
    --diarizer eend_eda
