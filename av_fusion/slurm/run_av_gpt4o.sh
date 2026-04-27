#!/bin/bash
#SBATCH -J av_gpt4o
#SBATCH -t 4:00:00
#SBATCH -p ou_bcs_normal,pi_satra
#SBATCH --mem=8G
#SBATCH -c 2
#SBATCH -o ../logs/av_fusion/av_gpt4o_%j.out
#SBATCH -e ../logs/av_fusion/av_gpt4o_%j.err

# GPT-4o-mini visual feature extraction for all 2183 clips (~$0.66, ~1-2h).
#
# PREREQUISITES:
#   export OPENAI_API_KEY=<your_key>   # must be set before sbatch
#   Video files must be accessible at paths in master_with_split.csv.
#
# Resumable: skips clips already in the output CSV.
# Results in: av_fusion/av_results/manual_only/gpt4o_features.csv

set -euo pipefail
export PYTHONUNBUFFERED=1
source /orcd/home/002/manaal/miniforge3/etc/profile.d/conda.sh
conda activate child-vocalizations

REPO=/orcd/scratch/orcd/008/manaal/child-adult-diarization
cd "$REPO"
mkdir -p logs/av_fusion

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
    echo "ERROR: OPENAI_API_KEY not set. Export it before submitting:" >&2
    echo "  export OPENAI_API_KEY=<key> && sbatch av_fusion/slurm/run_av_gpt4o.sh" >&2
    exit 1
fi

echo "=== GPT-4o feature extraction ==="
python av_fusion/scripts/extract_gpt4o_features.py \
    --metadata-csv whisper-modeling/seen_child_splits/master_with_split.csv \
    --output av_fusion/av_results/manual_only/gpt4o_features.csv

echo "=== Rebuild feature table with GPT-4o features ==="
python av_fusion/scripts/build_av_feature_table.py \
    --metadata-csv  whisper-modeling/seen_child_splits/master_with_split.csv \
    --audio-scores-val  babar_ecapa_enrollment_runs/enroll_val_predictions.csv \
    --audio-scores-test babar_ecapa_enrollment_runs/enroll_test_predictions.csv \
    --audio-score-col prob \
    --babar-rttm-dir babar/babar_output/rttm/ \
    --gpt4o-features-csv av_fusion/av_results/manual_only/gpt4o_features.csv \
    --output-dir  av_fusion/av_results/manual_only/ \
    --run-name    manual_only

echo "Done. GPT-4o features: av_fusion/av_results/manual_only/gpt4o_features.csv"
