#!/bin/bash
#SBATCH -c 4
#SBATCH -t 24:00:00
#SBATCH -p ou_bcs_normal,pi_satra
#SBATCH --mem=40G
#SBATCH --gres=gpu:1
#SBATCH --requeue
#SBATCH -o ../logs/enrollment/eend_eda_%j.out
#SBATCH -e ../logs/enrollment/eend_eda_%j.err

# EEND-EDA diarization + ECAPA enrollment pipeline.
#
# Setup (once, before submitting):
#   conda activate child-vocalizations
#   pip install espnet espnet_model_zoo soundfile
#
# The default ESPnet model (eend_eda_model_tag in BaseConfig) will be
# downloaded from the ESPnet Model Zoo on first run.  To use a different
# model, pass --eend-eda-model-tag <tag_or_local_dir>.
#
# Results written to:
#   ../eend_eda_ecapa_enrollment_runs/
#     config.json
#     child_prototype_stats.csv
#     enroll_{val,test}_predictions.csv
#     enroll_{val,test}_metrics.json
#     enroll_{val,test}_metrics_by_timepoint.csv
#     role_only_*

set -euo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source /orcd/home/002/manaal/miniforge3/etc/profile.d/conda.sh
conda activate child-vocalizations

cd /home/manaal/orcd/scratch/child-adult-diarization/pyannote

mkdir -p ../logs/enrollment
mkdir -p ../eend_eda_ecapa_enrollment_runs
mkdir -p ../pyannote/eend_eda_rttm_cache

echo "=== EEND-EDA enrollment ==="
python unified.py --diarizer eend_eda

echo "=== Done ==="
