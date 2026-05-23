#!/bin/bash
#SBATCH -J baseline_resume
#SBATCH -p ou_bcs_normal,pi_satra
#SBATCH -t 23:30:00
#SBATCH -c 4
#SBATCH --mem=40G
#SBATCH --requeue
#SBATCH --gres=gpu:1
#SBATCH -o logs/baselines/baseline_resume_%j.out
#SBATCH -e logs/baselines/baseline_resume_%j.err

# Resume the encoder baseline sweep after a timeout.
# --skip-existing detects experiments whose test_metrics_tuned.json already
# exists and skips them, so only missing experiments are rerun.

set -euo pipefail

source /home/manaal/miniforge3/etc/profile.d/conda.sh
conda activate child-vocalizations
unset HF_TOKEN HUGGINGFACE_HUB_TOKEN HF_HUB_TOKEN 2>/dev/null || true
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export HF_HOME=/orcd/scratch/orcd/008/manaal/.cache/huggingface

cd /orcd/scratch/orcd/008/manaal/child-adult-diarization
mkdir -p logs/baselines baseline_results_seen_child

echo "Start: $(date)"
python baselines/baseline_encoders.py --seen-child --all-experiments --skip-existing
echo "Done: $(date)"
