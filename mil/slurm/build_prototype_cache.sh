#!/bin/bash
#SBATCH --job-name=proto_cache
#SBATCH -c 4
#SBATCH -t 2:00:00
#SBATCH -p ou_bcs_normal,pi_satra
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH -o logs/mil/proto_cache_%j.out
#SBATCH -e logs/mil/proto_cache_%j.err

# Build per-(child, timepoint) ECAPA prototype cache for TS-MIL training (spec-014 US4).
# Idempotent — exits 0 immediately if --output already exists.
#
# Usage: sbatch mil/slurm/build_prototype_cache.sh <frontend> <train_csv> <output_npz>
# Defaults: babar_vtc + seen-child train + mil/prototypes/babar_vtc.npz

set -euo pipefail

FRONTEND=${1:-babar_vtc}
TRAIN_CSV=${2:-whisper-modeling/seen_child_splits/train.csv}
OUTPUT=${3:-mil/prototypes/babar_vtc.npz}
EXTRA_ARGS=${4:-}

export PATH="$HOME/.local/bin:$PATH"
source /orcd/home/002/manaal/miniforge3/etc/profile.d/conda.sh
conda activate child-vocalizations

REPO=/home/manaal/orcd/scratch/child-adult-diarization
cd "$REPO"
mkdir -p logs/mil mil/prototypes

if [[ -f "$OUTPUT" ]]; then
    echo "Prototype cache already exists at $OUTPUT — skipping (idempotent)."
    exit 0
fi

echo "=== Building prototype cache (job $SLURM_JOB_ID) ==="
echo "  frontend:  $FRONTEND"
echo "  train csv: $TRAIN_CSV"
echo "  output:    $OUTPUT"

python mil/scripts/build_prototype_cache.py \
    --frontend "$FRONTEND" \
    --train-csv "$TRAIN_CSV" \
    --output "$OUTPUT" \
    $EXTRA_ARGS

echo "=== Done at $(date) ==="
