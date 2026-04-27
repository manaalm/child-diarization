#!/bin/bash
#SBATCH -J av_asd_light_asd
#SBATCH -t 24:00:00
#SBATCH -p ou_bcs_normal,pi_satra
#SBATCH --mem=32G
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH -o ../logs/av_fusion/av_asd_light_asd_%j.out
#SBATCH -e ../logs/av_fusion/av_asd_light_asd_%j.err

# Light-ASD feature extraction for all clips with video.
#
# PREREQUISITES (one-time setup):
#   git clone https://github.com/Junhua-Liao/Light-ASD video/Light-ASD
#   # Checkpoint: video/Light-ASD/weight/pretrain_AVA_CVPR22.pt
#   # Download from: https://github.com/Junhua-Liao/Light-ASD/releases
#
# Results in: av_fusion/av_results/manual_only/asd_features_light_asd.csv

set -euo pipefail
export PYTHONUNBUFFERED=1

REPO=/orcd/scratch/orcd/008/manaal/child-adult-diarization
cd "$REPO"
mkdir -p logs/av_fusion

# Prefer AVA pretrain checkpoint; fall back to TalkSet
LIGHT_ASD_CKPT=$(find "$REPO/video/Light-ASD/weight" -name "pretrain_AVA_CVPR*.model" -o -name "pretrain_AVA_CVPR*.pt" 2>/dev/null | head -1)
if [[ -z "$LIGHT_ASD_CKPT" ]]; then
    LIGHT_ASD_CKPT=$(find "$REPO/video/Light-ASD/weight" -name "*.model" -o -name "*.pt" 2>/dev/null | head -1)
fi
if [[ -z "$LIGHT_ASD_CKPT" ]]; then
    echo "ERROR: No Light-ASD checkpoint found under video/Light-ASD/weight/" >&2
    echo "Clone the repo: git clone https://github.com/Junhua-Liao/Light-ASD video/Light-ASD" >&2
    exit 1
fi
echo "Using Light-ASD checkpoint: $LIGHT_ASD_CKPT"

# Use the video/ Python 3.10 uv env
VIDEO_PYTHON="$REPO/video/.venv/bin/python"

echo "=== Light-ASD feature extraction ==="
"$VIDEO_PYTHON" av_fusion/scripts/extract_asd_features.py \
    --metadata-csv  whisper-modeling/seen_child_splits/master_with_split.csv \
    --output        av_fusion/av_results/manual_only/asd_features_light_asd.csv \
    --model         light_asd \
    --light-asd-checkpoint "$LIGHT_ASD_CKPT"

echo "=== Rebuild feature table with Light-ASD features ==="
source /orcd/home/002/manaal/miniforge3/etc/profile.d/conda.sh
conda activate child-vocalizations

python av_fusion/scripts/build_av_feature_table.py \
    --metadata-csv  whisper-modeling/seen_child_splits/master_with_split.csv \
    --audio-scores-val  babar_ecapa_enrollment_runs/enroll_val_predictions.csv \
    --audio-scores-test babar_ecapa_enrollment_runs/enroll_test_predictions.csv \
    --audio-score-col prob \
    --babar-rttm-dir babar/babar_output/rttm/ \
    --asd-features-csv-extra light_asd:av_fusion/av_results/manual_only/asd_features_light_asd.csv \
    --output-dir  av_fusion/av_results/manual_only/ \
    --run-name    manual_only

echo "Done. Light-ASD features: av_fusion/av_results/manual_only/asd_features_light_asd.csv"
