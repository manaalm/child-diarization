#!/bin/bash
#SBATCH --job-name=synth_scene_gen_v2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=24:00:00
#SBATCH -p ou_bcs_normal,pi_satra
#SBATCH --output=logs/synth/scene_gen_v2_%j.out
#SBATCH --error=logs/synth/scene_gen_v2_%j.out

# v2: Generates scenes from a manifest that includes LibriSpeech and Playlogue
# in addition to Providence + TinyVox. Writes scenes to a separate output dir
# so the original 5000-scene corpus is not overwritten.
#
# Usage:
#   sbatch synth/slurm/run_scene_generation_v2.sh \
#       [config.yaml] [n_scenes] [output_subdir]
#
# Defaults:
#   config       = synth/configs/default_14_18mo.yaml
#   n_scenes     = (whatever the YAML says — typically 5000)
#   output_subdir = synthetic_scenes_v2

set -euo pipefail

CONFIG="${1:-synth/configs/default_14_18mo.yaml}"
N_SCENES_OVERRIDE="${2:-}"
OUTPUT_SUBDIR="${3:-synthetic_scenes_v2}"

if [[ ! -f "${CONFIG}" ]]; then
    echo "ERROR: Config file not found: ${CONFIG}" >&2
    exit 1
fi

source /orcd/home/002/manaal/miniforge3/etc/profile.d/conda.sh
conda activate child-vocalizations

REPO_ROOT="/orcd/scratch/orcd/008/manaal/child-adult-diarization"
cd "${REPO_ROOT}"
echo "Working directory: $(pwd)"
echo "Config: ${CONFIG}"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Start time: $(date)"

mkdir -p logs/synth synth_results/manifests

MANIFEST=synth_results/manifests/segment_manifest_v2.csv

# ---- Step 1: Build v2 segment manifest (Providence + TinyVox + Playlogue + LibriSpeech) ----
if [[ ! -f "${MANIFEST}" ]]; then
    echo "=== Building v2 segment manifest ==="
    python synth/scripts/build_segment_manifest.py \
        --providence-dir        providence/ \
        --providence-rttm-dir   providence/rttm/ \
        --tinyvox-dir           data/tinyvox/ \
        --playlogue-dir         playlogue/audio/ \
        --playlogue-rttm-dir    playlogue/rttm/ \
        --librispeech-dir       data/LibriSpeech/LibriSpeech/train-clean-100/ \
        --exclude-speakers-csv  whisper-modeling/seen_child_splits/test.csv \
        --output                "${MANIFEST}" \
        --skip-quality
fi

# ---- Step 1b: Pre-extract every segment to WAV (avoids 8 sec/draw flac/MP3 decode) ----
# Without this, scene gen runs at ~2.5 scenes/min instead of ~40/min because
# every draw seeks into a long Providence MP3 or decodes a fresh LibriSpeech flac.
# extract_segments writes per-segment 16kHz WAVs to data/segments_v2/{role}/ and
# updates the manifest's audio_path column in place.
SEG_DIR=data/segments_v2
if [[ ! -d "${SEG_DIR}" ]] || [[ "$(find ${SEG_DIR} -name '*.wav' 2>/dev/null | wc -l)" -lt 100000 ]]; then
    echo "=== Pre-extracting segments to ${SEG_DIR} (one-time, ~30-60 min) ==="
    python synth/scripts/extract_segments.py \
        --manifest    "${MANIFEST}" \
        --output-dir  "${SEG_DIR}" \
        --sample-rate 16000
fi

# ---- Step 2: Generate scenes ----
EXTRA=""
if [[ -n "${N_SCENES_OVERRIDE}" ]]; then
    EXTRA="--n-scenes ${N_SCENES_OVERRIDE}"
fi

echo "=== Generating scenes (manifest=${MANIFEST}, output=${OUTPUT_SUBDIR}) ==="
python synth/scripts/generate_scenes.py \
    --config  "${CONFIG}" \
    --manifest "${MANIFEST}" \
    --output-dir "synth_results/${OUTPUT_SUBDIR}/" \
    ${EXTRA}

echo "Done. End time: $(date)"
