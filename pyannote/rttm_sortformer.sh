#!/bin/bash
#SBATCH -c 4
#SBATCH -t 12:00:00
#SBATCH -p pi_satra,ou_bcs_normal
#SBATCH --mem=40G
#SBATCH --requeue
#SBATCH --gres=gpu:1
#SBATCH -o logs/rttm/sortformer_%j.out
#SBATCH -e logs/rttm/sortformer_%j.err

# Frame-level RTTM accuracy for Sortformer on Playlogue + Providence.
# Mirrors the layout in rttm.sh / rttm_vbx.sh.
#
# Prereq: pip install nemo_toolkit[asr] (in child-vocalizations env)
# Anonymous SPEAKER_XX labels are resolved to CHI/ADT via GT-overlap mapping.

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

# ── Playlogue — Sortformer ───────────────────────────────────────────────
python unified_rttm.py \
    --dataset playlogue \
    --audio-dir "$AUDIO_PLAY" \
    --rttm-dir  "$RTTM_PLAY" \
    --diarizer sortformer

# ── Providence — Sortformer ──────────────────────────────────────────────
python unified_rttm.py \
    --dataset providence \
    --audio-dir "$AUDIO_PROV" \
    --rttm-dir  "$RTTM_PROV" \
    --diarizer sortformer
