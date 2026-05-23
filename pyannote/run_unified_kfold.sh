#!/bin/bash
#SBATCH -c 4
#SBATCH -t 4:00:00
#SBATCH -p ou_bcs_normal,pi_satra
#SBATCH --mem=20G
#SBATCH --requeue
#SBATCH --gres=gpu:1
#SBATCH -o logs/enrollment/kfold_%A_%a.out
#SBATCH -e logs/enrollment/kfold_%A_%a.err

# k-fold enrollment driver. Submit as an array job:
#   sbatch --array=0-2 pyannote/run_unified_kfold.sh pyannote
#   sbatch --array=0-2 pyannote/run_unified_kfold.sh babar
#   sbatch --array=0-2 pyannote/run_unified_kfold.sh vtc
#   sbatch --array=0-2 pyannote/run_unified_kfold.sh vtc_kchi
#   sbatch --array=0-2 pyannote/run_unified_kfold.sh vbx
#   sbatch --array=0-2 pyannote/run_unified_kfold.sh sortformer
#   sbatch --array=0-2 pyannote/run_unified_kfold.sh eend_eda
#
# For pretrained diarizers, only the ECAPA child-prototype computation and
# threshold tuning differ across folds (the underlying RTTM cache is shared
# because audio paths are deterministic). Total cost: ~10 min per fold per
# diarizer (after RTTMs are cached from the original seen-child run).
#
# Requires HF_TOKEN exported in environment for pyannote/vbx (rotated token).

set -euo pipefail

DIARIZER=${1:?"Usage: sbatch --array=0-K-1 run_unified_kfold.sh <diarizer>"}
FOLD=${SLURM_ARRAY_TASK_ID:-0}
K=${KFOLD_K:-3}

SPLIT_DIR="whisper-modeling/seen_child_splits_kfold_${K}fold/fold_${FOLD}"
OUT_DIR="${DIARIZER}_ecapa_enrollment_runs_kfold${K}_f${FOLD}"

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
    echo "Run first: python whisper-modeling/make_kfold_seen_child_split.py --k $K" >&2
    exit 2
fi

echo "=== unified k-fold: $DIARIZER fold $FOLD (job $SLURM_JOB_ID array $SLURM_ARRAY_TASK_ID) ==="
echo "Split: $SPLIT_DIR"
echo "Output: $OUT_DIR"

cd pyannote
python unified.py \
    --diarizer "$DIARIZER" \
    --train-csv "$REPO/$SPLIT_DIR/train.csv" \
    --val-csv "$REPO/$SPLIT_DIR/val.csv" \
    --test-csv "$REPO/$SPLIT_DIR/test.csv" \
    --output-dir "$REPO/$OUT_DIR"

echo "=== Done: $DIARIZER fold $FOLD → $OUT_DIR ==="
