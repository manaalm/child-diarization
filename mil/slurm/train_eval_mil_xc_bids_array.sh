#!/bin/bash
#SBATCH -J mil_xc_array
#SBATCH -p ou_bcs_normal,pi_satra
#SBATCH -t 8:00:00
#SBATCH -c 4
#SBATCH --mem=48G
#SBATCH --requeue
#SBATCH --gres=gpu:1
#SBATCH --array=0-2
#SBATCH -o logs/mil/mil_xc_array_%A_%a.out
#SBATCH -e logs/mil/mil_xc_array_%A_%a.err

# Parallel SLURM array for the 3 remaining MIL variants on BIDS cross-child
# (after wavlm_mil + whisper_mil land sequentially). Each task trains+evals
# a single variant.

set -euo pipefail

CONFIGS=(
    mil/configs/whisper_medium_mil_cross_child.yaml
    mil/configs/whisper_mil_acmil_max_cross_child.yaml
    mil/configs/whisper_mil_tsmil_concat_cross_child.yaml
)
CFG="${CONFIGS[$SLURM_ARRAY_TASK_ID]}"

source /orcd/home/002/manaal/miniforge3/etc/profile.d/conda.sh
conda activate child-vocalizations
unset HF_TOKEN HUGGINGFACE_HUB_TOKEN HF_HUB_TOKEN 2>/dev/null || true
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export HF_HOME=/orcd/scratch/orcd/008/manaal/.cache/huggingface

cd /orcd/scratch/orcd/008/manaal/child-adult-diarization
mkdir -p logs/mil

name=$(python -c "import yaml; print(yaml.safe_load(open('$CFG'))['variant_name'])")
out="mil/mil_results/${name}"

echo "Array task $SLURM_ARRAY_TASK_ID: variant=$name  start=$(date)"
if [ -f "${out}/test_metrics_tuned.json" ] && [ -f "${out}/.bids_retrain_done" ]; then
    echo "  $name: BIDS retrain already complete, skip"
else
    python mil/mil_train.py --config "$CFG"
    python mil/mil_evaluate.py \
        --checkpoint "${out}/best_checkpoint.pt" \
        --config     "${out}/config.json"
    touch "${out}/.bids_retrain_done"
fi
echo "Array task $SLURM_ARRAY_TASK_ID: variant=$name  end=$(date)"
