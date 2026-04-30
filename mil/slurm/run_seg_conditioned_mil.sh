#!/bin/bash
#SBATCH --job-name=seg_cond_mil
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH -c 4
#SBATCH -t 12:00:00
#SBATCH -p ou_bcs_normal,pi_satra
#SBATCH -o logs/mil/seg_cond_mil_%j.out
#SBATCH -e logs/mil/seg_cond_mil_%j.out

# Conditioned Segment MIL (US6, spec-013, Tier 4)
# GatedABMIL over ECAPA segment embeddings conditioned on child prototype.
# Each instance: [seg_emb(192), proto(192), seg_emb-proto(192)] = 576-dim.
# ~2-4h: prototype building (~30min) + bag pre-computation (~1.5h) + training (~30min).

echo "Start: $(date)"
echo "Node: $(hostname)"

source /home/manaal/miniforge3/etc/profile.d/conda.sh
conda activate child-vocalizations

export HF_HOME=/orcd/scratch/orcd/008/manaal/.cache/huggingface
export TORCH_HOME=/orcd/scratch/orcd/008/manaal/.cache/torch

cd /orcd/scratch/orcd/008/manaal/child-adult-diarization
mkdir -p logs/mil

python mil/seg_conditioned_train.py \
    --device cuda \
    --lr 1e-3 \
    --epochs 20 \
    --patience 5 \
    --batch-size 8 \
    --dropout 0.25 \
    --output-dir mil/mil_results/seg_conditioned_mil

echo "Done: $(date)"
