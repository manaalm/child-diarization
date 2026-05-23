#!/bin/bash
#SBATCH --job-name=eval_backbone
#SBATCH -c 4
#SBATCH -t 2:00:00
#SBATCH -p ou_bcs_normal,pi_satra
#SBATCH --mem=40G
#SBATCH --gres=gpu:1
#SBATCH -o logs/mil/eval_backbone_%j.out
#SBATCH -e logs/mil/eval_backbone_%j.err

set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
source /orcd/home/002/manaal/miniforge3/etc/profile.d/conda.sh
conda activate child-vocalizations

cd /orcd/scratch/orcd/008/manaal/child-adult-diarization
mkdir -p logs/mil

echo "=== Eval backbone-size MIL variants (job $SLURM_JOB_ID) ==="
echo "Start: $(date)"

for d in mil/mil_results/whisper_tiny_mil mil/mil_results/whisper_base_mil mil/mil_results/whisper_medium_mil; do
    if [ ! -f "$d/test_metrics_tuned.json" ] && [ -f "$d/best_checkpoint.pt" ]; then
        echo "--- $(basename $d) ---"
        python mil/mil_evaluate.py --checkpoint "$d/best_checkpoint.pt" --config "$d/config.json"
    fi
done

echo "Done: $(date)"
