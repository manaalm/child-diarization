#!/bin/bash
#SBATCH --job-name=mil_tinyvox
#SBATCH -c 4
#SBATCH -t 12:00:00
#SBATCH -p ou_bcs_normal,pi_satra
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH -o logs/mil/tinyvox_%j.out
#SBATCH -e logs/mil/tinyvox_%j.err

# TinyVox-augmented WavLM-MIL experiment.
# Step 1: build augmented train split (real train + 15k TinyVox Providence positives)
# Step 2: train WavLM-MIL with pad_to_sec=10.0 so short TinyVox clips produce
#         the same number of windows as full 10s real clips
# Step 3: evaluate on the standard (non-augmented) test split

set -euo pipefail

export PATH="$HOME/.local/bin:$PATH"
source /orcd/home/002/manaal/miniforge3/etc/profile.d/conda.sh
conda activate child-vocalizations

REPO=/home/manaal/orcd/scratch/child-adult-diarization
cd "$REPO"
mkdir -p logs/mil

echo "=== TinyVox MIL experiment (job ${SLURM_JOB_ID}) ==="
echo "Start: $(date)"

echo "--- Step 1: build augmented splits ---"
python mil/build_tinyvox_splits.py

echo "--- Step 2: train ---"
python mil/mil_train.py --config mil/configs/wavlm_mil_tinyvox.yaml

echo "--- Step 3: evaluate on test split ---"
CKPT="mil/mil_results/wavlm_mil_tinyvox/best_checkpoint.pt"
CFG="mil/mil_results/wavlm_mil_tinyvox/config.json"
python mil/mil_evaluate.py --checkpoint "$CKPT" --config "$CFG"

echo "=== Done: $(date) ==="
echo "Results in mil/mil_results/wavlm_mil_tinyvox/"
echo "Compare AUROC vs wavlm_mil baseline:"
python3 -c "
import json
base = json.load(open('mil/mil_results/wavlm_mil/test_metrics_tuned.json'))
aug  = json.load(open('mil/mil_results/wavlm_mil_tinyvox/test_metrics_tuned.json'))
for k in ('f1', 'auroc', 'auprc'):
    delta = aug[k] - base[k]
    sign = '+' if delta >= 0 else ''
    print(f'  {k}: baseline={base[k]:.4f}  tinyvox={aug[k]:.4f}  delta={sign}{delta:.4f}')
"
