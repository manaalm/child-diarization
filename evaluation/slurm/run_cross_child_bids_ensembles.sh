#!/bin/bash
#SBATCH -J xc_bids_ensembles
#SBATCH -p mit_normal,ou_bcs_normal,pi_satra
#SBATCH -t 1:00:00
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -o logs/ensemble/xc_bids_ensembles_%j.out
#SBATCH -e logs/ensemble/xc_bids_ensembles_%j.err

# Cross-speaker BIDS LR-stacker ensemble sweep.
# Runs the cross-child counterpart of ensemble_runs/advanced/ on BIDS-corrected
# val (n=444) + test (n=742). CPU-only.
#
# Auto-discovers available component systems at run time, so we can launch
# this immediately and have it pick up whatever cross-child BIDS prediction
# CSVs are present (zero-shot + role-only + AV + pseudo-frame already there;
# encoder + MIL added when their SLURM jobs finish).
#
# To make this fire automatically after the encoder + MIL retrains land,
# submit with:
#   sbatch --dependency=afterok:14191408:14194134 \
#       evaluation/slurm/run_cross_child_bids_ensembles.sh

set -euo pipefail

source /home/manaal/miniforge3/etc/profile.d/conda.sh
conda activate child-vocalizations

REPO=/orcd/scratch/orcd/008/manaal/child-adult-diarization
cd "$REPO"
mkdir -p logs/ensemble

echo "Start: $(date)"
python evaluation/cross_child_bids_advanced_ensembles.py
echo "Done: $(date)"
