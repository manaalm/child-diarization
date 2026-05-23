#!/bin/bash
#SBATCH -c 4
#SBATCH -t 4:00:00
#SBATCH -p ou_bcs_normal,pi_satra
#SBATCH --mem=20G
#SBATCH --requeue
#SBATCH --gres=gpu:1
#SBATCH -o logs/enrollment/cross_child_%A_%j.out
#SBATCH -e logs/enrollment/cross_child_%A_%j.err

# Cross-child enrollment driver. Submit one job per diarizer:
#   sbatch pyannote/run_unified_cross_child.sh usc_sail
#   sbatch pyannote/run_unified_cross_child.sh pyannote
#   sbatch pyannote/run_unified_cross_child.sh babar
#   sbatch pyannote/run_unified_cross_child.sh vtc
#   sbatch pyannote/run_unified_cross_child.sh vtc_kchi
#   sbatch pyannote/run_unified_cross_child.sh vbx
#   sbatch pyannote/run_unified_cross_child.sh sortformer
#   sbatch pyannote/run_unified_cross_child.sh eend_eda
#
# Uses baselines/splits/ (97 train / 21 val / 21 test, disjoint children).
# RTTM cache is shared with seen-child runs because it is keyed on audio path.
# Total cost: ~10-30 min per diarizer (after RTTMs are already cached).
#
# Requires HF_TOKEN exported in environment for pyannote/vbx (rotated token).

set -euo pipefail

DIARIZER=${1:?"Usage: sbatch run_unified_cross_child.sh <diarizer>"}
SPLIT_DIR="baselines/splits"
OUT_DIR="${DIARIZER}_ecapa_enrollment_runs_cross_child"

if [[ "$DIARIZER" == "pyannote" || "$DIARIZER" == "vbx" ]]; then
    : "${HF_TOKEN:?HF_TOKEN must be set in environment for $DIARIZER}"
fi

source /orcd/home/002/manaal/miniforge3/etc/profile.d/conda.sh
conda activate child-vocalizations
module load ffmpeg/5.1.4 || true
export LD_LIBRARY_PATH="/orcd/software/community/001/spack/pkg/ffmpeg/5.1.4/6kcopsg/lib:${LD_LIBRARY_PATH:-}"

REPO=/orcd/scratch/orcd/008/manaal/child-adult-diarization
cd "$REPO"
mkdir -p logs/enrollment

if [ ! -d "$SPLIT_DIR" ]; then
    echo "ERROR: split dir not found: $SPLIT_DIR" >&2
    exit 2
fi

echo "=== unified cross-child: $DIARIZER (job $SLURM_JOB_ID) ==="
echo "Split: $SPLIT_DIR"
echo "Output: $OUT_DIR"

cd pyannote
python unified.py \
    --diarizer "$DIARIZER" \
    --train-csv "$REPO/$SPLIT_DIR/train.csv" \
    --val-csv "$REPO/$SPLIT_DIR/val.csv" \
    --test-csv "$REPO/$SPLIT_DIR/test.csv" \
    --output-dir "$REPO/$OUT_DIR"

echo "=== Done: $DIARIZER → $OUT_DIR ==="
