#!/bin/bash
#SBATCH --job-name=pyannote_enroll
#SBATCH -c 4
#SBATCH -t 8:00:00
#SBATCH -p ou_bcs_normal,pi_satra
#SBATCH --mem=40G
#SBATCH --gres=gpu:1
#SBATCH -o logs/enrollment/pyannote_enroll_%j.out
#SBATCH -e logs/enrollment/pyannote_enroll_%j.out

# Run pyannote diarizer through unified.py (ECAPA enrollment pipeline).
# Output: /orcd/scratch/orcd/008/manaal/child-adult-diarization/pyannote_ecapa_enrollment_runs/

set -euo pipefail
source /orcd/home/002/manaal/miniforge3/etc/profile.d/conda.sh
conda activate child-vocalizations
module load ffmpeg/5.1.4
export LD_LIBRARY_PATH="/orcd/software/community/001/spack/pkg/ffmpeg/5.1.4/6kcopsg/lib:${LD_LIBRARY_PATH:-}"
export HF_TOKEN=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

cd /orcd/scratch/orcd/008/manaal/child-adult-diarization/pyannote

mkdir -p logs/enrollment
echo "Start: $(date)"
python unified.py --diarizer pyannote
echo "Done: $(date)"
