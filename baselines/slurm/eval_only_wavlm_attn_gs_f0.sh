#!/bin/bash
#SBATCH -J wavlm_attn_f0_evalonly
#SBATCH -p ou_bcs_normal,pi_satra,mit_normal
#SBATCH -t 30:00
#SBATCH -c 4
#SBATCH --mem=24G
#SBATCH --requeue
#SBATCH --gres=gpu:1
#SBATCH -o logs/baselines/wavlm_attn_f0_evalonly_%j.out
#SBATCH -e logs/baselines/wavlm_attn_f0_evalonly_%j.err

set -euo pipefail
source /home/manaal/miniforge3/etc/profile.d/conda.sh
conda activate child-vocalizations
unset HF_TOKEN HUGGINGFACE_HUB_TOKEN HF_HUB_TOKEN 2>/dev/null || true
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export HF_HOME=/orcd/scratch/orcd/008/manaal/.cache/huggingface

cd /orcd/scratch/orcd/008/manaal/child-adult-diarization
mkdir -p logs/baselines

echo "Start: $(date)"
python baselines/eval_only_wavlm_attn_gs_f0.py
echo "Done: $(date)"
