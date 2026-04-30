#!/bin/bash
#SBATCH --job-name=hubert_w2v2_mil
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH -c 4
#SBATCH -t 48:00:00
#SBATCH -p ou_bcs_normal,pi_satra
#SBATCH -o logs/mil/hubert_w2v2_mil_%j.out
#SBATCH -e logs/mil/hubert_w2v2_mil_%j.out

# Trains HuBERT-large and wav2vec2-large MIL baselines (Tier 3, US4).
# Each model is 316M params / 1024-dim; batch_size=4, mil_hidden_dim=512.
# Estimated: ~12h each = ~24h total; may need full 48h if preempted.

echo "Start: $(date)"
echo "Node: $(hostname)"

source /home/manaal/miniforge3/etc/profile.d/conda.sh
conda activate child-vocalizations

export HF_HOME=/orcd/scratch/orcd/008/manaal/.cache/huggingface
export TORCH_HOME=/orcd/scratch/orcd/008/manaal/.cache/torch

cd /orcd/scratch/orcd/008/manaal/child-adult-diarization
mkdir -p logs/mil

# ── HuBERT-large ─────────────────────────────────────────────────────────
echo "=== Training HuBERT-large MIL ==="
python mil/mil_train.py --config mil/configs/hubert_large_mil.yaml

echo "=== Evaluating HuBERT-large MIL ==="
CKPT=$(ls mil/mil_results/hubert_large_mil/best_checkpoint.pt 2>/dev/null || echo "")
if [ -n "$CKPT" ]; then
    python mil/mil_evaluate.py \
        --checkpoint mil/mil_results/hubert_large_mil/best_checkpoint.pt \
        --config mil/mil_results/hubert_large_mil/config.json
else
    echo "WARNING: HuBERT checkpoint not found, skipping eval."
fi

# ── wav2vec2-large ────────────────────────────────────────────────────────
echo "=== Training wav2vec2-large MIL ==="
python mil/mil_train.py --config mil/configs/wav2vec2_large_mil.yaml

echo "=== Evaluating wav2vec2-large MIL ==="
CKPT=$(ls mil/mil_results/wav2vec2_large_mil/best_checkpoint.pt 2>/dev/null || echo "")
if [ -n "$CKPT" ]; then
    python mil/mil_evaluate.py \
        --checkpoint mil/mil_results/wav2vec2_large_mil/best_checkpoint.pt \
        --config mil/mil_results/wav2vec2_large_mil/config.json
else
    echo "WARNING: wav2vec2 checkpoint not found, skipping eval."
fi

echo "Done: $(date)"
