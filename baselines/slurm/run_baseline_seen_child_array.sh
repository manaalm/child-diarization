#!/bin/bash
#SBATCH -J baseline_seen_arr
#SBATCH -p ou_bcs_normal,pi_satra
#SBATCH -t 6:00:00
#SBATCH -c 4
#SBATCH --mem=40G
#SBATCH --requeue
#SBATCH --gres=gpu:1
#SBATCH -o logs/baselines/baseline_seen_arr_%A_%a.out
#SBATCH -e logs/baselines/baseline_seen_arr_%A_%a.err

# Parallel replacement for run_baseline_seen_child_resume.sh.
# One array task per remaining seen-child variant — replaces a single 23h
# serial sweep with N independent jobs that fan out across available GPUs.
#
#   sbatch --array=0-6 baselines/slurm/run_baseline_seen_child_array.sh
#
# --skip-existing kept as a guard so a task is a no-op if test_metrics_tuned.json
# was already produced by another job (e.g. baseline_resume finishing wavlm_stats_lw
# before this array fully drains).

set -euo pipefail

source /home/manaal/miniforge3/etc/profile.d/conda.sh
conda activate child-vocalizations
unset HF_TOKEN HUGGINGFACE_HUB_TOKEN HF_HUB_TOKEN 2>/dev/null || true
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export HF_HOME=/orcd/scratch/orcd/008/manaal/.cache/huggingface

cd /orcd/scratch/orcd/008/manaal/child-adult-diarization
mkdir -p logs/baselines baseline_results_seen_child

IDX=${SLURM_ARRAY_TASK_ID:-0}
case "$IDX" in
    0) EXP=wavlm_stats_lw ;;
    1) EXP=fused_attn ;;
    2) EXP=fused_attn_unfreeze2 ;;
    3) EXP=whisper_attn_ptt ;;
    4) EXP=whisper_attn_unfreeze2 ;;
    5) EXP=whisper_attn_aug ;;
    6) EXP=whisper_attn_aug_ptt ;;
    *) echo "Unknown IDX=$IDX"; exit 2 ;;
esac

echo "=== ${EXP} seen-child (job ${SLURM_JOB_ID} array ${IDX}) ==="
echo "Start: $(date)"
python baselines/baseline_encoders.py --seen-child --all-experiments \
    --experiments "$EXP" --skip-existing
echo "Done: $(date)"
