#!/bin/bash
#SBATCH -J whisper_attn_gs
#SBATCH -p ou_bcs_normal,pi_satra
#SBATCH -t 6:00:00
#SBATCH -c 4
#SBATCH --mem=40G
#SBATCH --requeue
#SBATCH --gres=gpu:1
#SBATCH -o logs/baselines/whisper_attn_gs_%A_%a.out
#SBATCH -e logs/baselines/whisper_attn_gs_%A_%a.err

# Group-stratified 3-fold for whisper_attn + whisper_attn_lw.
# Array: 0-2 = whisper_attn folds 0/1/2; 3-5 = whisper_attn_lw folds 0/1/2.
#   sbatch --array=0-5 baselines/slurm/run_whisper_attn_groupstrat.sh

set -euo pipefail

source /home/manaal/miniforge3/etc/profile.d/conda.sh
conda activate child-vocalizations
unset HF_TOKEN HUGGINGFACE_HUB_TOKEN HF_HUB_TOKEN 2>/dev/null || true
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export HF_HOME=/orcd/scratch/orcd/008/manaal/.cache/huggingface
export PYTHONPATH=/orcd/scratch/orcd/008/manaal/child-adult-diarization

cd /orcd/scratch/orcd/008/manaal/child-adult-diarization
mkdir -p logs/baselines

IDX=${SLURM_ARRAY_TASK_ID:-0}
FOLD=$((IDX % 3))
EXP_IDX=$((IDX / 3))
case "$EXP_IDX" in
    0) EXP=whisper_attn ;;
    1) EXP=whisper_attn_lw ;;
    *) echo "Unknown EXP_IDX=$EXP_IDX"; exit 2 ;;
esac

SUFFIX="_groupstrat3_f${FOLD}"
SPLIT_DIR="whisper-modeling/seen_child_splits_groupstrat_3fold/fold_${FOLD}"

echo "=== ${EXP} groupstrat3 fold ${FOLD} (job ${SLURM_JOB_ID} array ${IDX}) ==="
echo "Start: $(date)"
python baselines/baseline_encoders.py --seen-child --all-experiments \
    --experiments "$EXP" \
    --split-dir "$SPLIT_DIR" \
    --variant-suffix "$SUFFIX"
echo "Done: $(date)"
