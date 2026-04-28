#!/bin/bash
#SBATCH -c 4
#SBATCH -t 24:00:00
#SBATCH -p ou_bcs_normal,pi_satra
#SBATCH --mem=40G
#SBATCH --gres=gpu:1
#SBATCH --requeue
#SBATCH -o ../logs/enrollment/sortformer_%j.out
#SBATCH -e ../logs/enrollment/sortformer_%j.err

# Sortformer (NeMo) diarization + ECAPA enrollment pipeline.
#
# Setup (once, before submitting):
#   conda activate child-vocalizations
#   pip install nemo_toolkit[asr]
#   # The model (diar_sortformer_4spk-v1) downloads from NGC on first run.
#
# Results written to:
#   ../sortformer_ecapa_enrollment_runs/
#     config.json
#     child_prototype_stats.csv
#     enroll_{val,test}_predictions.csv
#     enroll_{val,test}_metrics.json
#     enroll_{val,test}_metrics_by_timepoint.csv
#     role_only_*

set -euo pipefail
source /orcd/home/002/manaal/miniforge3/etc/profile.d/conda.sh
conda activate child-vocalizations

cd /home/manaal/orcd/scratch/child-adult-diarization/pyannote

mkdir -p ../logs/enrollment
mkdir -p ../sortformer_ecapa_enrollment_runs
mkdir -p ../pyannote/sortformer_rttm_cache

echo "=== Sortformer enrollment ==="
python unified.py --diarizer sortformer

echo "=== Done ==="
