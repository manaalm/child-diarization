#!/bin/bash
#SBATCH --job-name=adult_eval
#SBATCH --output=logs/adult/adultification_eval_%j.out
#SBATCH --error=logs/adult/adultification_eval_%j.out
#SBATCH --partition=ou_bcs_normal
#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
# CPU-only: F0/LPC/sklearn LR.

set -euo pipefail

REPO=/orcd/scratch/orcd/008/manaal/child-adult-diarization
ENV_PY=/orcd/home/002/manaal/miniforge3/envs/child-vocalizations/bin/python
cd "$REPO"

AGE_BAND="${1:-14_18}"
SYNTH_TAG="${2:-v3_perturb}"
MAX_PER_SET="${3:-600}"

REFS_DIR="synth_results/manifests/adultification_refs"
OUT_DIR="synth_results/adultification_eval/${SYNTH_TAG}_${AGE_BAND}mo"

# 1. Build reference manifests (one-time per (age_band, synth_tag) pair).
"$ENV_PY" synth/scripts/build_adultification_refs.py \
    --segment-manifest synth_results/manifests/segment_manifest_v2.csv \
    --synth-wav-dir   "synth_results/synthetic_scenes_${SYNTH_TAG}/wav" \
    --synth-rttm-dir  "synth_results/synthetic_scenes_${SYNTH_TAG}/rttm" \
    --age-band "$AGE_BAND" --max-per-set "$MAX_PER_SET" \
    --output-dir "$REFS_DIR"

# 2. Run the eval.
"$ENV_PY" synth/scripts/adultification_eval.py \
    --real-child-csv "$REFS_DIR/real_child_${AGE_BAND}.csv" \
    --real-adult-csv "$REFS_DIR/real_adult.csv" \
    --eval-csv       "$REFS_DIR/synth_eval_${AGE_BAND}.csv" \
    --output-dir     "$OUT_DIR" \
    --max-clips-per-set "$MAX_PER_SET"

echo "[done] adultification eval -> $OUT_DIR"
