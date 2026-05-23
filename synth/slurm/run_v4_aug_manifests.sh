#!/bin/bash
#SBATCH --job-name=v4_aug_man
#SBATCH --output=logs/adult/v4_aug_manifests_%j.out
#SBATCH --error=logs/adult/v4_aug_manifests_%j.out
#SBATCH -p ou_bcs_normal,pi_satra
#SBATCH --time=1:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#
# Run AFTER v4 scene generation completes. Snapshots the canonical synth
# manifest as synthetic_manifest_v4.csv, builds the *_v4 augmentation
# manifests, and produces baselines/splits_synth_aug_v4/.

set -euo pipefail

REPO=/orcd/scratch/orcd/008/manaal/child-adult-diarization
ENV_PY=/orcd/home/002/manaal/miniforge3/envs/child-vocalizations/bin/python
cd "$REPO"

"$ENV_PY" synth/scripts/build_v4_aug_manifests.py
echo "[done] v4 aug manifests"
