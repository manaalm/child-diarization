#!/bin/bash
#SBATCH --job-name=synth_ratio_sweep
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:1
#SBATCH -p ou_bcs_normal,pi_satra
#SBATCH --output=logs/synth/ratio_sweep_%j.out
#SBATCH --error=logs/synth/ratio_sweep_%j.out

# Usage:
#   sbatch synth/slurm/run_ratio_sweep.sh synth/configs/default_14_18mo.yaml
#
# Runs Steps 4-6 from quickstart.md sequentially for all 6 ratios:
#   4. generate_training_sets.py (all ratios)
#   5. train_with_synthetic.py (calls BabAR enrollment per ratio)
#   6. evaluate_synthetic_augmentation.py
#
# IMPORTANT: Before running this sweep, you must re-generate synthetic scenes
# with acoustic augmentation (RIR + noise) via run_scene_generation.sh. The
# clean-mix scenes in synth_results/synthetic_scenes/ do not include room
# acoustics and will produce null results (identical metrics to baseline at all
# ratios). See spec-009 T020 for the correct invocation with --rir-dir / --noise-dir.
#
# GPU is needed for BabAR enrollment (ECAPA prototype computation).
# Logs go to logs/synth/ratio_sweep_${SLURM_JOB_ID}.out

set -euo pipefail

CONFIG="${1:-synth/configs/default_14_18mo.yaml}"

if [[ ! -f "${CONFIG}" ]]; then
    echo "ERROR: Config file not found: ${CONFIG}" >&2
    exit 1
fi

source /orcd/home/002/manaal/miniforge3/etc/profile.d/conda.sh
conda activate child-vocalizations

# ffmpeg shared libs required by torchcodec inside BabAR/.venv (Python 3.13).
# The module only sets PATH, not LD_LIBRARY_PATH, so set it explicitly.
export LD_LIBRARY_PATH="/orcd/software/community/001/spack/pkg/ffmpeg/5.1.4/6kcopsg/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

REPO_ROOT="/orcd/scratch/orcd/008/manaal/child-adult-diarization"
cd "${REPO_ROOT}"
echo "Working directory: $(pwd)"
echo "Config: ${CONFIG}"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Start time: $(date)"

mkdir -p logs/synth

# Extract config name for output directory
CONFIG_NAME=$(python -c "import yaml,sys; c=yaml.safe_load(open('${CONFIG}')); print(c['project']['name'])")

SYNTH_MANIFEST="synth_results/manifests/synthetic_manifest.csv"
MANIFEST_DIR="synth_results/manifests"
EXPERIMENT_DIR="synth_results/augmentation_experiments/${CONFIG_NAME}"

# Step 4: Generate training sets at all 6 ratios
echo "=== Step 4: generate_training_sets.py ==="
python synth/scripts/generate_training_sets.py \
    --real-train-csv        whisper-modeling/seen_child_splits/train.csv \
    --synthetic-manifest    "${SYNTH_MANIFEST}" \
    --ratios                0 0.5 1 2 5 10 \
    --output-dir            "${MANIFEST_DIR}" \
    --seed 42

# Step 5: Train (enrollment) at each ratio
echo "=== Step 5: train_with_synthetic.py ==="
python synth/scripts/train_with_synthetic.py \
    --manifest-dir  "${MANIFEST_DIR}" \
    --ratios        0 0.5 1 2 5 10 \
    --output-dir    "${EXPERIMENT_DIR}"

# Step 6: Evaluate on held-out test set
echo "=== Step 6: evaluate_synthetic_augmentation.py ==="
python synth/scripts/evaluate_synthetic_augmentation.py \
    --experiment-dir  "${EXPERIMENT_DIR}" \
    --test-csv        whisper-modeling/seen_child_splits/test.csv \
    --output-dir      "${EXPERIMENT_DIR}" \
    --plot

echo "Done. End time: $(date)"
