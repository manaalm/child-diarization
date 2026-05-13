#!/bin/bash
#SBATCH --job-name=scene_analysis
#SBATCH --gres=gpu:1
#SBATCH -t 04:00:00
#SBATCH --mem=32G
#SBATCH -c 4
#SBATCH -p ou_bcs_normal,pi_satra
#SBATCH -o logs/baselines/scene_analysis_%j.out
#SBATCH -e logs/baselines/scene_analysis_%j.out

# spec-022 US3 — Audio-scene-analysis baseline dispatcher.
# Usage: sbatch baselines/slurm/run_scene_analysis_baseline.sh <model> <split>
#   model: ast | yamnet
#   split: val | test | test_all
#
# YAMNet runs in the yamnet-eval sibling env (TensorFlow); AST runs in the
# project's child-vocalizations env (PyTorch).

MODEL=${1:-ast}
SPLIT=${2:-val}

echo "Start: $(date)"
echo "MODEL=${MODEL}  SPLIT=${SPLIT}"

source /home/manaal/miniforge3/etc/profile.d/conda.sh
conda activate child-vocalizations

# CLAUDE.md gotchas — apply across both AST (transformers >=4.57) and YAMNet
# (subprocess inherits env)
export HF_HOME=/orcd/scratch/orcd/008/manaal/.cache/huggingface
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
unset HF_TOKEN HUGGINGFACE_HUB_TOKEN HF_HUB_TOKEN || true

cd /orcd/scratch/orcd/008/manaal/child-adult-diarization

python baselines/scene_analysis_baseline.py --model "${MODEL}" --split "${SPLIT}"

echo "Done: $(date)"
