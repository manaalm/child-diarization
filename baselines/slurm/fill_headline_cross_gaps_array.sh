#!/bin/bash
#SBATCH -J fill_xc_gaps
#SBATCH -p ou_bcs_normal,pi_satra
#SBATCH -t 6:00:00
#SBATCH -c 4
#SBATCH --mem=80G
#SBATCH --requeue
#SBATCH --gres=gpu:1
#SBATCH --array=0-7
#SBATCH -o logs/baselines/fill_xc_gaps_%A_%a.out
#SBATCH -e logs/baselines/fill_xc_gaps_%A_%a.err

# Fills the four remaining empty cells of tab:headline-cross:
#
#   idx 0: fused_attn_unfreeze2_whisper_medium  BIDS cross-child single split
#   idx 1: fused_attn_unfreeze2_whisper_large   BIDS cross-child single split
#   idx 2: whisper_mean  group-strat 3-fold fold 0
#   idx 3: whisper_mean  group-strat 3-fold fold 1
#   idx 4: whisper_mean  group-strat 3-fold fold 2
#   idx 5: wavlm_attn    group-strat 3-fold fold 0
#   idx 6: wavlm_attn    group-strat 3-fold fold 1
#   idx 7: wavlm_attn    group-strat 3-fold fold 2

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
echo "=== fill_xc_gaps array task $IDX  start=$(date) ==="

case "$IDX" in
    0)
        # fused-medium PU2 cross-child single split
        python -u encoders/run_fused_unfreeze2_backbone_swap_cross_child.py --backbone medium
        ;;
    1)
        # fused-large-v3 PU2 cross-child single split
        python -u encoders/run_fused_unfreeze2_backbone_swap_cross_child.py --backbone large
        ;;
    2|3|4)
        FOLD=$((IDX - 2))
        OUT="./baseline_results_seen_child"
        python -u baselines/baseline_encoders.py \
            --seen-child \
            --all-experiments \
            --experiments whisper_mean \
            --split-dir "whisper-modeling/seen_child_splits_groupstrat_3fold/fold_${FOLD}" \
            --variant-suffix "_groupstrat3_f${FOLD}" \
            --results-root "$OUT" \
            --skip-existing
        ;;
    5|6|7)
        FOLD=$((IDX - 5))
        OUT="./baseline_results_seen_child"
        python -u baselines/baseline_encoders.py \
            --seen-child \
            --all-experiments \
            --experiments wavlm_attn \
            --split-dir "whisper-modeling/seen_child_splits_groupstrat_3fold/fold_${FOLD}" \
            --variant-suffix "_groupstrat3_f${FOLD}" \
            --results-root "$OUT" \
            --skip-existing
        ;;
    *)
        echo "Unknown SLURM_ARRAY_TASK_ID=$IDX"
        exit 2
        ;;
esac

echo "=== task $IDX  end=$(date) ==="
