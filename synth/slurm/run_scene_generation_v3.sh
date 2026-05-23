#!/bin/bash
#SBATCH --job-name=synth_scene_gen_v3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=24:00:00
#SBATCH -p ou_bcs_normal,pi_satra
#SBATCH --output=logs/synth/scene_gen_v3_%j.out
#SBATCH --error=logs/synth/scene_gen_v3_%j.out

# v3: Reuses the v2 segment manifest (Providence + TinyVox + LibriSpeech +
# Playlogue) but generates scenes with per-segment VTLP + speed perturbation
# enabled. Output dir is separate so v2 corpus is not overwritten.
#
# Usage:
#   sbatch synth/slurm/run_scene_generation_v3.sh \
#       [config.yaml] [n_scenes] [output_subdir]
#
# Defaults:
#   config        = synth/configs/v3_perturb_14_18mo.yaml
#   n_scenes      = (whatever the YAML says — typically 5000)
#   output_subdir = synthetic_scenes_v3_perturb

set -euo pipefail

CONFIG="${1:-synth/configs/v3_perturb_14_18mo.yaml}"
N_SCENES_OVERRIDE="${2:-}"
OUTPUT_SUBDIR="${3:-synthetic_scenes_v3_perturb}"

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

MANIFEST_FULL=synth_results/manifests/segment_manifest_v2.csv
MANIFEST=synth_results/manifests/segment_manifest_v2_sampled.csv
SEG_DIR=data/segments_v2

if [[ ! -f "${MANIFEST_FULL}" ]]; then
    echo "ERROR: ${MANIFEST_FULL} missing — run synth/slurm/run_scene_generation_v2.sh first to build full segment manifest." >&2
    exit 2
fi

# Detect a stale post-extraction subsampled manifest whose audio_path entries
# no longer exist (e.g., segments_v2/ was purged by inode-quota cleanup).
# extract_segments.py mutates audio_path in-place to the extraction destination,
# so once destinations are deleted the manifest is unrecoverable without a fresh
# subsample from MANIFEST_FULL.
NEED_RESAMPLE=0
if [[ ! -f "${MANIFEST}" ]]; then
    NEED_RESAMPLE=1
else
    EXIST_COUNT=$(python -c "
import pandas as pd, os
df = pd.read_csv('${MANIFEST}', low_memory=False)
print((df['audio_path'].apply(os.path.isfile)).sum())
")
    if [[ "${EXIST_COUNT}" -lt 15000 ]]; then
        echo "Subsampled manifest has only ${EXIST_COUNT}/18000 valid paths — resampling fresh from full manifest."
        NEED_RESAMPLE=1
    fi
fi

if [[ "${NEED_RESAMPLE}" == "1" ]]; then
    echo "=== Subsampling manifest fresh (3k per source_dataset) ==="
    python -c "
import pandas as pd
df = pd.read_csv('${MANIFEST_FULL}', low_memory=False)
sampled = []
for ds, grp in df.groupby('source_dataset'):
    sampled.append(grp.sample(n=min(3000, len(grp)), random_state=42))
out = pd.concat(sampled, ignore_index=True)
out.to_csv('${MANIFEST}', index=False)
print(f'Subsampled: {len(out)} segments')
"
fi

# ---- Re-extract segments if the cache was cleaned (inode quota recipe) ----
# segments_v2 is gitignored and gets purged when scratch hits its 1M file cap.
# Re-extraction takes ~70 min CPU, regenerable from the manifest.
if [[ ! -d "${SEG_DIR}" ]] || [[ "$(find ${SEG_DIR} -name '*.wav' 2>/dev/null | wc -l)" -lt 15000 ]]; then
    echo "=== Pre-extracting subsampled segments to ${SEG_DIR} (one-time, ~70 min) ==="
    python synth/scripts/extract_segments.py \
        --manifest    "${MANIFEST}" \
        --output-dir  "${SEG_DIR}" \
        --sample-rate 16000
fi

# ---- Generate scenes ----
EXTRA=""
if [[ -n "${N_SCENES_OVERRIDE}" ]]; then
    EXTRA="--n-scenes ${N_SCENES_OVERRIDE}"
fi

echo "=== Generating v3 scenes with VTLP + speed perturbation ==="
echo "  manifest=${MANIFEST}"
echo "  output=${OUTPUT_SUBDIR}"
python synth/scripts/generate_scenes.py \
    --config  "${CONFIG}" \
    --manifest "${MANIFEST}" \
    --output-dir "synth_results/${OUTPUT_SUBDIR}/" \
    ${EXTRA}

echo "Done. End time: $(date)"
