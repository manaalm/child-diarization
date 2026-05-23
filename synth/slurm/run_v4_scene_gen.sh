#!/bin/bash
#SBATCH --job-name=v4_scene_gen
#SBATCH --output=logs/adult/v4_scene_gen_%j.out
#SBATCH --error=logs/adult/v4_scene_gen_%j.out
#SBATCH -p ou_bcs_normal,pi_satra
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --requeue
#
# Scene-generation-only step for the v4 corpus. Assumes
#   synth_results/manifests/empirical_turn_taking.json AND
#   synth_results/manifests/segment_manifest_v4.csv
# already exist (built by run_v4_pipeline.sh). Use this when the v4
# pipeline has been split or you want to re-run scene gen at a different
# n_scenes / sampling_mode.

set -euo pipefail

REPO=/orcd/scratch/orcd/008/manaal/child-adult-diarization
ENV_PY=/orcd/home/002/manaal/miniforge3/envs/child-vocalizations/bin/python
cd "$REPO"

mkdir -p synth_results/synthetic_scenes_v4

"$ENV_PY" synth/scripts/generate_scenes.py \
    --config   synth/configs/v4_14_18mo.yaml \
    --manifest synth_results/manifests/segment_manifest_v4.csv \
    --output-dir synth_results/synthetic_scenes_v4/

echo "[done] v4 scene gen"
