#!/bin/bash
#SBATCH -c 4
#SBATCH -t 4:00:00
#SBATCH -p ou_bcs_normal,pi_satra
#SBATCH --mem=32G
#SBATCH --requeue
#SBATCH --gres=gpu:1
#SBATCH -o logs/synth/voice_transfer_%j.out
#SBATCH -e logs/synth/voice_transfer_%j.err

# Spec-016 follow-up #1: per-child WavLM-feature voice-transfer synth augmentation.
# Single-script experiment: extract WavLM mean features for all clips + 5000 synth
# scenes, compute per-child prototypes, voice-transfer synth via mean shift in
# WavLM space, train LR with and without augmentation, compare.
#
# Usage: sbatch synth/slurm/voice_transfer_experiment.sh

set -euo pipefail

export PATH="$HOME/.local/bin:$PATH"
source /orcd/home/002/manaal/miniforge3/etc/profile.d/conda.sh
conda activate child-vocalizations

REPO=/orcd/scratch/orcd/008/manaal/child-adult-diarization
cd "$REPO"
mkdir -p logs/synth synth_results/voice_transfer_experiment

echo "=== voice-transfer C7 experiment (job $SLURM_JOB_ID) ==="

PYTHONPATH=. python synth/scripts/voice_transfer_experiment.py

echo "=== Done ==="
