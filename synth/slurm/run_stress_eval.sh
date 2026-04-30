#!/bin/bash
#SBATCH --job-name=stress_eval
#SBATCH -c 4
#SBATCH -t 4:00:00
#SBATCH -p ou_bcs_normal,pi_satra
#SBATCH --mem=40G
#SBATCH --gres=gpu:1
#SBATCH -o logs/synth/stress_eval_%j.out
#SBATCH -e logs/synth/stress_eval_%j.out

# Evaluate Whisper-MIL on stress-test synth scenes.
# Run after 3 stress scene-gen jobs (hard_negatives / overlap_stress / low_snr_stress) complete.
# Submit with dependency:
#   sbatch --dependency=afterok:<gen1>:<gen2>:<gen3> synth/slurm/run_stress_eval.sh

set -euo pipefail

source /orcd/home/002/manaal/miniforge3/etc/profile.d/conda.sh
conda activate child-vocalizations

REPO=/orcd/scratch/orcd/008/manaal/child-adult-diarization
cd "$REPO"
mkdir -p logs/synth synth_results/stress_test_results

echo "=== Stress-test eval (job ${SLURM_JOB_ID}) ==="
echo "Start time: $(date)"

python synth/scripts/evaluate_stress_configs.py \
    --checkpoint mil/mil_results/whisper_mil/best_checkpoint.pt \
    --config     mil/mil_results/whisper_mil/config.json \
    --scenes-dir synth_results/synthetic_scenes \
    --configs    hard_negatives overlap_stress low_snr_stress \
    --output-dir synth_results/stress_test_results

echo "=== Done at $(date) ==="
