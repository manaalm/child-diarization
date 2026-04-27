#!/bin/bash
#SBATCH -J av_llava
#SBATCH -t 12:00:00
#SBATCH -p ou_bcs_normal,pi_satra
#SBATCH --mem=40G
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH -o ../logs/av_fusion/av_llava_%j.out
#SBATCH -e ../logs/av_fusion/av_llava_%j.err

# LLaVA-based visual child-detection feature extraction (open-source GPT-4o replacement).
# Uses LLaVA-1.5-7B or Qwen2-VL-7B to produce the same output schema as gpt4o_features.csv.
#
# Model is downloaded from HuggingFace on first run (~14 GB for llava-1.5-7b).
# Subsequent runs use HF cache.
#
# Results in: av_fusion/av_results/manual_only/gpt4o_features.csv
# (same filename as GPT-4o output so downstream scripts work unchanged)

set -euo pipefail
export PYTHONUNBUFFERED=1
source /orcd/home/002/manaal/miniforge3/etc/profile.d/conda.sh
conda activate child-vocalizations

REPO=/orcd/scratch/orcd/008/manaal/child-adult-diarization
cd "$REPO"
mkdir -p logs/av_fusion

MODEL="${LLAVA_MODEL:-llava-1.5-7b}"   # override with: LLAVA_MODEL=qwen2-vl-7b sbatch ...
OUTPUT="av_fusion/av_results/manual_only/gpt4o_features.csv"

echo "=== LLaVA visual feature extraction (model: $MODEL) ==="
python av_fusion/scripts/extract_llava_features.py \
    --metadata-csv  whisper-modeling/seen_child_splits/master_with_split.csv \
    --output        "$OUTPUT" \
    --model         "$MODEL" \
    --sample-rate   2

echo "=== Rebuild feature table with LLaVA visual features ==="
python av_fusion/scripts/build_av_feature_table.py \
    --metadata-csv  whisper-modeling/seen_child_splits/master_with_split.csv \
    --audio-scores-val  babar_ecapa_enrollment_runs/enroll_val_predictions.csv \
    --audio-scores-test babar_ecapa_enrollment_runs/enroll_test_predictions.csv \
    --audio-score-col prob \
    --babar-rttm-dir babar/babar_output/rttm/ \
    --gpt4o-features-csv "$OUTPUT" \
    --output-dir  av_fusion/av_results/manual_only/ \
    --run-name    manual_only

echo "=== Re-run AV fusion + cascade + smoothing ==="
python av_fusion/scripts/train_av_fusion.py \
    --feature-dir av_fusion/av_results/manual_only/ \
    --output-dir  av_fusion/av_results/manual_only/models/ \
    --config      av_fusion/configs/av_fusion.yaml \
    --seed 42

python av_fusion/scripts/train_cascaded_pipeline.py \
    --feature-dir av_fusion/av_results/manual_only/ \
    --output-dir  av_fusion/av_results/manual_only/models/

python av_fusion/scripts/evaluate_av_fusion.py \
    --feature-dir av_fusion/av_results/manual_only/ \
    --model-dir   av_fusion/av_results/manual_only/models/ \
    --output-dir  av_fusion/av_results/manual_only/ \
    --cascade-breakdown av_fusion/av_results/manual_only/cascade_stage_breakdown.csv \
    --plot

python av_fusion/scripts/smooth_predictions.py \
    --predictions     av_fusion/av_results/manual_only/predictions_test.csv \
    --val-predictions av_fusion/av_results/manual_only/predictions_val.csv \
    --output          av_fusion/av_results/manual_only/predictions_test_smoothed.csv \
    --method gaussian

echo "Done. LLaVA features: $OUTPUT"
