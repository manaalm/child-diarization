#!/bin/bash
#SBATCH -J encoder_xc_bids_nostats
#SBATCH -p ou_bcs_normal,pi_satra
#SBATCH -t 23:30:00
#SBATCH -c 4
#SBATCH --mem=40G
#SBATCH --requeue
#SBATCH --gres=gpu:1
#SBATCH -o logs/baselines/encoder_xc_bids_nostats_%j.out
#SBATCH -e logs/baselines/encoder_xc_bids_nostats_%j.err

# Cross-child BIDS encoder grid retrain, excluding the stats-pooling variants
# (whisper_stats_lw, wavlm_stats_lw) per directive. Uses --skip-existing so any
# variant that already wrote test_metrics_tuned.json (whisper_mean from the
# initial 14191409 run) is preserved without rerunning. whisper_attn was
# in-progress at cancel time without test_metrics_tuned.json yet, so it will
# re-train from scratch.

set -euo pipefail

source /home/manaal/miniforge3/etc/profile.d/conda.sh
conda activate child-vocalizations
unset HF_TOKEN HUGGINGFACE_HUB_TOKEN HF_HUB_TOKEN 2>/dev/null || true
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export HF_HOME=/orcd/scratch/orcd/008/manaal/.cache/huggingface

cd /orcd/scratch/orcd/008/manaal/child-adult-diarization
mkdir -p logs/baselines

KEEP="whisper_mean,whisper_attn,wavlm_mean,wavlm_attn,whisper_attn_lw,wavlm_attn_lw,fused_attn,whisper_attn_unfreeze2,fused_attn_unfreeze2,whisper_attn_ptt,whisper_attn_aug,whisper_attn_aug_ptt"

echo "Start: $(date)"
echo "Variants (no-stats): $KEEP"
# --all-experiments populates the full variant registry; --experiments then
# filters down to the 12 keepers (excluding whisper_stats_lw and
# wavlm_stats_lw per directive).
# ./splits/master_with_split.csv has been refreshed to baselines/splits/
# (BIDS-corrected, n=3314, 105/23/23 children) so the encoder script picks
# up BIDS data via its existing-master-file path.
python baselines/baseline_encoders.py \
    --all-experiments \
    --experiments "$KEEP" \
    --skip-existing \
    --results-root ./baselines/baseline_results_cross_child_bids
echo "Done: $(date)"
