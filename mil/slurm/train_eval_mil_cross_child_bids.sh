#!/bin/bash
#SBATCH --job-name=mil_xc_bids
#SBATCH -c 4
#SBATCH -t 48:00:00
#SBATCH -p ou_bcs_normal,pi_satra
#SBATCH --mem=48G
#SBATCH --requeue
#SBATCH --gres=gpu:1
#SBATCH -o logs/mil/cross_child_bids_%j.out
#SBATCH -e logs/mil/cross_child_bids_%j.err

# BIDS-corrected cross-child MIL retrain queue (spec-022 thesis v3 update).
# Trains 5 MIL variants on the BIDS-corrected cross-child split
# (baselines/splits/, n=2128 train / 444 val / 742 test) then evaluates each.
# Each config points its split_dir at baselines/splits/, which has been
# BIDS-corrected since 2026-05-12.

set -euo pipefail

export PATH="$HOME/.local/bin:$PATH"
source /orcd/home/002/manaal/miniforge3/etc/profile.d/conda.sh
conda activate child-vocalizations
unset HF_TOKEN HUGGINGFACE_HUB_TOKEN HF_HUB_TOKEN 2>/dev/null || true
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export HF_HOME=/orcd/scratch/orcd/008/manaal/.cache/huggingface

REPO=/orcd/scratch/orcd/008/manaal/child-adult-diarization
cd "$REPO"
mkdir -p logs/mil

echo "=== Cross-child BIDS MIL retrain (job $SLURM_JOB_ID) ==="
echo "Start: $(date)"

CONFIGS=(
  mil/configs/wavlm_mil_cross_child.yaml
  mil/configs/whisper_mil_cross_child.yaml
  mil/configs/whisper_medium_mil_cross_child.yaml
  mil/configs/whisper_mil_acmil_max_cross_child.yaml
  mil/configs/whisper_mil_tsmil_concat_cross_child.yaml
)

for cfg in "${CONFIGS[@]}"; do
  name=$(python -c "import yaml; print(yaml.safe_load(open('$cfg'))['variant_name'])")
  out="mil/mil_results/${name}"
  # Skip if a fresh BIDS retrain already landed
  if [ -f "${out}/test_metrics_tuned.json" ] && [ -f "${out}/.bids_retrain_done" ]; then
    echo "--- $name: BIDS retrain already complete, skip ---"
    continue
  fi
  echo "--- Training $name ---"
  python mil/mil_train.py --config "$cfg"
  echo "--- Evaluating $name ---"
  python mil/mil_evaluate.py \
      --checkpoint "${out}/best_checkpoint.pt" \
      --config     "${out}/config.json"
  touch "${out}/.bids_retrain_done"
done

echo "=== Done. End: $(date) ==="
