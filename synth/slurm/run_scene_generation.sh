#!/bin/bash
#SBATCH --job-name=synth_scene_gen
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=24:00:00
#SBATCH -p ou_bcs_normal,pi_satra
#SBATCH --output=logs/synth/scene_gen_%j.out
#SBATCH --error=logs/synth/scene_gen_%j.out

# Usage:
#   sbatch synth/slurm/run_scene_generation.sh synth/configs/default_14_18mo.yaml [--rir-dir PATH] [--noise-dir PATH]
#
# Generates 5000 synthetic scenes (or as configured in the YAML).
# --rir-dir and --noise-dir override mixing.rir_dir / mixing.noise_dir in the YAML.
# If omitted, the generator produces clean-mix scenes (FR-005 fallback).
# Logs go to logs/synth/scene_gen_${SLURM_JOB_ID}.out

set -euo pipefail

CONFIG="${1:-synth/configs/default_14_18mo.yaml}"
shift || true  # consume first arg; remaining args forwarded to generate_scenes.py
EXTRA_ARGS=("$@")

# Validate config argument
if [[ ! -f "${CONFIG}" ]]; then
    echo "ERROR: Config file not found: ${CONFIG}" >&2
    exit 1
fi

# Activate conda environment
source /orcd/home/002/manaal/miniforge3/etc/profile.d/conda.sh
conda activate child-vocalizations

# Move to repo root (hardcoded — SLURM may run from /var/spool)
REPO_ROOT="/orcd/scratch/orcd/008/manaal/child-adult-diarization"
cd "${REPO_ROOT}"
echo "Working directory: $(pwd)"
echo "Config: ${CONFIG}"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Start time: $(date)"

# Create log directory
mkdir -p logs/synth

# Step 1: build segment manifest (fast, skip per-segment audio quality scoring)
if [[ ! -f synth_results/manifests/segment_manifest.csv ]]; then
    echo "=== Building segment manifest ==="
    python synth/scripts/build_segment_manifest.py \
        --providence-dir        providence/ \
        --providence-rttm-dir   providence/rttm/ \
        --tinyvox-dir           data/tinyvox/ \
        --exclude-speakers-csv  whisper-modeling/seen_child_splits/test.csv \
        --output                synth_results/manifests/segment_manifest.csv \
        --skip-quality
fi

# Step 2: extract segments to WAV
if [[ ! -d data/segments ]] || [[ -z "$(ls -A data/segments 2>/dev/null)" ]]; then
    echo "=== Extracting segments ==="
    python synth/scripts/extract_segments.py \
        --manifest   synth_results/manifests/segment_manifest.csv \
        --output-dir data/segments/
fi

# Step 3: generate scenes
echo "=== Generating scenes ==="
python synth/scripts/generate_scenes.py \
    --config  "${CONFIG}" \
    --manifest synth_results/manifests/segment_manifest.csv \
    --output-dir synth_results/synthetic_scenes/ \
    "${EXTRA_ARGS[@]}"

echo "Done. End time: $(date)"
