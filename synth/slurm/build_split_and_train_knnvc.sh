#!/bin/bash
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH -t 8:00:00
#SBATCH -p ou_bcs_normal,pi_satra
#SBATCH --mem=40G
#SBATCH --requeue
#SBATCH -o logs/synth/build_and_train_knnvc_%j.out
#SBATCH -e logs/synth/build_and_train_knnvc_%j.err

# T150 + T152: build augmented split + train+eval WavLM-MIL on it.
# Sequential: T150 is CPU-only (~5 s), T152 is the GPU MIL run (~6 h).
set -euo pipefail

source /orcd/home/002/manaal/miniforge3/etc/profile.d/conda.sh
conda activate child-vocalizations

REPO=/orcd/scratch/orcd/008/manaal/child-adult-diarization
cd "$REPO"
mkdir -p logs/synth logs/mil

export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

if [[ ! -f "synth_results/manifests/synthetic_voice_converted.csv" ]]; then
    echo "ERROR: T130 manifest synthetic_voice_converted.csv missing — KNN-VC bulk job not done."
    exit 2
fi

echo "=== T150: build_knnvc_synth_split (job $SLURM_JOB_ID) ==="
python synth/scripts/build_knnvc_synth_split.py

echo "=== T152: WavLM-MIL train + eval on augmented split ==="
bash mil/slurm/train_eval_spec014.sh mil/configs/wavlm_mil_knnvc.yaml

echo "=== Done at $(date) ==="
