#!/bin/bash
#SBATCH -c 4
#SBATCH -t 8:00:00
#SBATCH -p ou_bcs_normal,pi_satra
#SBATCH --mem=40G
#SBATCH --requeue
#SBATCH --gres=gpu:1
#SBATCH -o logs/rttm/vtc_%j.out
#SBATCH -e logs/rttm/vtc_%j.err

# Run VTC 2.0 (standalone, no BabAR phoneme step) on Playlogue and Providence.
#
# Two label variants are evaluated:
#   vtc      → child = KCHI + OCH  (key child + other child)
#   vtc_kchi → child = KCHI only   (key/target child only)
#
# Prerequisites (run once before submitting):
#   cd /home/manaal/orcd/scratch/child-adult-diarization/BabAR/VTC && uv sync

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

# ── Playlogue — vtc (KCHI + OCH as child) ─────────────────────────────────
python unified_rttm.py \
    --dataset playlogue \
    --audio-dir "$AUDIO_PLAY" \
    --rttm-dir  "$RTTM_PLAY" \
    --diarizer vtc

# ── Playlogue — vtc_kchi (KCHI only) ─────────────────────────────────────
python unified_rttm.py \
    --dataset playlogue \
    --audio-dir "$AUDIO_PLAY" \
    --rttm-dir  "$RTTM_PLAY" \
    --diarizer vtc_kchi

# ── Providence — vtc (KCHI + OCH as child) ───────────────────────────────
python unified_rttm.py \
    --dataset providence \
    --audio-dir "$AUDIO_PROV" \
    --rttm-dir  "$RTTM_PROV" \
    --diarizer vtc

# ── Providence — vtc_kchi (KCHI only) ────────────────────────────────────
python unified_rttm.py \
    --dataset providence \
    --audio-dir "$AUDIO_PROV" \
    --rttm-dir  "$RTTM_PROV" \
    --diarizer vtc_kchi
