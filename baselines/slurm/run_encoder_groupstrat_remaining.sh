#!/bin/bash
#SBATCH -J encoder_gs_rem
#SBATCH -p ou_bcs_normal,pi_satra
#SBATCH -t 6:00:00
#SBATCH -c 4
#SBATCH --mem=40G
#SBATCH --requeue
#SBATCH --gres=gpu:1
#SBATCH -o logs/baselines/encoder_gs_rem_%A_%a.out
#SBATCH -e logs/baselines/encoder_gs_rem_%A_%a.err

# Group-stratified 3-fold for the 8 remaining encoder variants (after whisper_attn,
# whisper_attn_lw, whisper_mean, wavlm_mean, wavlm_attn, fused_attn_unfreeze2 family
# are all separately covered).
#
# Array index → variant + fold:
#   idx // 3 = variant_idx, idx % 3 = fold
#   0..2   wavlm_attn_lw  f0/f1/f2
#   3..5   whisper_stats_lw f0/f1/f2
#   6..8   wavlm_stats_lw f0/f1/f2
#   9..11  fused_attn f0/f1/f2
#   12..14 whisper_attn_unfreeze2 f0/f1/f2
#   15..17 whisper_attn_ptt f0/f1/f2
#   18..20 whisper_attn_aug f0/f1/f2
#   21..23 whisper_attn_aug_ptt f0/f1/f2
#
# Submit: sbatch --array=0-23 baselines/slurm/run_encoder_groupstrat_remaining.sh

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
VAR_IDX=$((IDX / 3))

case "$VAR_IDX" in
    0) EXP=wavlm_attn_lw ;;
    1) EXP=whisper_stats_lw ;;
    2) EXP=wavlm_stats_lw ;;
    3) EXP=fused_attn ;;
    4) EXP=whisper_attn_unfreeze2 ;;
    5) EXP=whisper_attn_ptt ;;
    6) EXP=whisper_attn_aug ;;
    7) EXP=whisper_attn_aug_ptt ;;
    *) echo "Unknown VAR_IDX=$VAR_IDX"; exit 2 ;;
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
