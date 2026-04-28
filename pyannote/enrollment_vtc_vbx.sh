#!/bin/bash
#SBATCH -c 4
#SBATCH -t 8:00:00
#SBATCH -p ou_bcs_normal,pi_satra
#SBATCH --mem=40G
#SBATCH --requeue
#SBATCH --gres=gpu:1
#SBATCH -o logs/enrollment/vtc_vbx_%j.out
#SBATCH -e logs/enrollment/vtc_vbx_%j.err

# Run VTC and VBx enrollment evaluation on SAILS (seen_child_splits).
#
# Prerequisites (run once before submitting):
#   cd /home/manaal/orcd/scratch/child-adult-diarization/BabAR/VTC && uv sync
#   cd /home/manaal/orcd/scratch/child-adult-diarization/VBx && uv sync
#   export HF_TOKEN=<your_token>   # needed for VBx (pyannote models)

set -euo pipefail
export HF_TOKEN=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
source /orcd/home/002/manaal/miniforge3/etc/profile.d/conda.sh
conda activate child-vocalizations
module load ffmpeg/5.1.4
export LD_LIBRARY_PATH="/orcd/software/community/001/spack/pkg/ffmpeg/5.1.4/6kcopsg/lib:${LD_LIBRARY_PATH:-}"

cd /home/manaal/orcd/scratch/child-adult-diarization/pyannote

mkdir -p logs/enrollment

# ── VTC (KCHI + OCH as child) ────────────────────────────────────────────────
python unified.py --diarizer vtc

# ── VTC (KCHI only) ──────────────────────────────────────────────────────────
python unified.py --diarizer vtc_kchi

# ── VBx ──────────────────────────────────────────────────────────────────────
python unified.py --diarizer vbx
