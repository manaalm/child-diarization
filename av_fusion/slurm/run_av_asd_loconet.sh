#!/bin/bash
#SBATCH -J av_asd_loconet
#SBATCH -t 24:00:00
#SBATCH -p ou_bcs_normal,pi_satra
#SBATCH --mem=32G
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH -o ../logs/av_fusion/av_asd_loconet_%j.out
#SBATCH -e ../logs/av_fusion/av_asd_loconet_%j.err

# LocoNet ASD feature extraction for all clips with video.
#
# PREREQUISITES (one-time setup):
#   huggingface-cli download Superxixixi/LoCoNet_ASD --local-dir video/LoCoNet_ASD/
#   # Checkpoint: video/LoCoNet_ASD/pytorch_model.bin  (HuggingFace format)
#
# Results in: av_fusion/av_results/manual_only/asd_features_loconet.csv

set -euo pipefail
export PYTHONUNBUFFERED=1

REPO=/orcd/scratch/orcd/008/manaal/child-adult-diarization
cd "$REPO"
mkdir -p logs/av_fusion

# HuggingFace repo uses pytorch_model.bin; fall back to any .ckpt if present
LOCONET_CKPT=$(find video/LoCoNet_ASD -name "pytorch_model.bin" | head -1)
if [[ -z "$LOCONET_CKPT" ]]; then
    LOCONET_CKPT=$(find video/LoCoNet_ASD -name "*.ckpt" | head -1)
fi
if [[ -z "$LOCONET_CKPT" ]]; then
    echo "ERROR: No checkpoint found under video/LoCoNet_ASD/." >&2
    echo "Run: huggingface-cli download Superxixixi/LoCoNet_ASD --local-dir video/LoCoNet_ASD/" >&2
    exit 1
fi
echo "Using LocoNet checkpoint: $LOCONET_CKPT"

# Use the video/ Python 3.10 uv env (isolated from child-vocalizations)
VIDEO_PYTHON="$REPO/video/.venv/bin/python"

echo "=== LocoNet ASD feature extraction ==="
"$VIDEO_PYTHON" av_fusion/scripts/extract_asd_features.py \
    --metadata-csv  whisper-modeling/seen_child_splits/master_with_split.csv \
    --output        av_fusion/av_results/manual_only/asd_features_loconet.csv \
    --model         loconet \
    --loconet-checkpoint "$LOCONET_CKPT"

echo "=== Rebuild feature table with LocoNet ASD features ==="
source /orcd/home/002/manaal/miniforge3/etc/profile.d/conda.sh
conda activate child-vocalizations

python av_fusion/scripts/build_av_feature_table.py \
    --metadata-csv  whisper-modeling/seen_child_splits/master_with_split.csv \
    --audio-scores-val  babar_ecapa_enrollment_runs/enroll_val_predictions.csv \
    --audio-scores-test babar_ecapa_enrollment_runs/enroll_test_predictions.csv \
    --audio-score-col prob \
    --babar-rttm-dir babar/babar_output/rttm/ \
    --asd-features-csv-extra loconet:av_fusion/av_results/manual_only/asd_features_loconet.csv \
    --output-dir  av_fusion/av_results/manual_only/ \
    --run-name    manual_only

echo "Done. LocoNet ASD features: av_fusion/av_results/manual_only/asd_features_loconet.csv"
