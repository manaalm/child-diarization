#!/bin/bash
#SBATCH --job-name=v4_pipeline
#SBATCH --output=logs/adult/v4_pipeline_%j.out
#SBATCH --error=logs/adult/v4_pipeline_%j.out
#SBATCH --partition=ou_bcs_normal
#SBATCH --time=8:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#
# End-to-end v4 corpus build:
#   (1) Fit empirical turn-taking distributions from Providence + Playlogue.
#   (2) Merge v2 + WORLD/CLEESE/cross-lingual VC manifests into v4.
#   (3) Generate the v4 synth scenes.
#
# Pre-requisites: WORLD / CLEESE / cross-lingual VC SLURM jobs must have
# completed first (see synth/slurm/run_world_childrenization.sh,
# run_cleese_childrenization.sh, run_cross_lingual_tinyvox_vc.sh).

set -euo pipefail

REPO=/orcd/scratch/orcd/008/manaal/child-adult-diarization
ENV_PY=/orcd/home/002/manaal/miniforge3/envs/child-vocalizations/bin/python
cd "$REPO"

# 1. Fit empirical turn-taking (idempotent; small CPU job ~1-2 min).
"$ENV_PY" synth/scripts/fit_empirical_turn_taking.py \
    --providence-rttm-dir providence/rttm \
    --playlogue-rttm-dir   playlogue/rttm \
    --playlogue-manifest   playlogue/manifest.csv \
    --age-bands 14_18 34_38 \
    --output synth_results/manifests/empirical_turn_taking.json \
    --write-config-stub synth/configs/empirical_turn_taking_stub.yaml

# 2. Merge manifests into v4.
"$ENV_PY" synth/scripts/build_v4_corpus.py \
    --base-manifest synth_results/manifests/segment_manifest_v2.csv \
    --add-manifest  synth_results/manifests/world_childrenized_manifest.csv \
    --add-manifest  synth_results/manifests/cleese_childrenized_manifest.csv \
    --add-manifest  synth_results/manifests/cross_lingual_vc_manifest.csv \
    --include-shards \
    --output        synth_results/manifests/segment_manifest_v4.csv

# 3. Generate v4 scenes.
mkdir -p synth_results/synthetic_scenes_v4
"$ENV_PY" synth/scripts/generate_scenes.py \
    --config   synth/configs/v4_14_18mo.yaml \
    --manifest synth_results/manifests/segment_manifest_v4.csv \
    --output-dir synth_results/synthetic_scenes_v4/

echo "[done] v4 pipeline"
