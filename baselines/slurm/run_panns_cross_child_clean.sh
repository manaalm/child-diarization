#!/bin/bash
#SBATCH --job-name=panns_cc_clean
#SBATCH --mem=32G
#SBATCH -c 4
#SBATCH -t 6:00:00
#SBATCH -p ou_bcs_normal,pi_satra
#SBATCH -o logs/baselines/panns_cc_clean_%j.out
#SBATCH -e logs/baselines/panns_cc_clean_%j.out

# Clean cross-child PANNS CNN14 — fix for the data-leakage issue documented
# in THESIS_MEGADOC.md §22b.5.
#
# Original cnn14_cross_child run reused the seen-child-trained LR head; 19 of
# 21 cross-child test children also appeared in seen-child train, so the LR
# was effectively trained on those test children's other clips. This script
# trains a fresh LR head on baselines/splits/train.csv (97 disjoint children),
# then scores baselines/splits/{val,test}.csv (21+21 disjoint test children).
#
# CNN14 runs on CPU. ~2-3 hours total (97 train + 21 val + 21 test children
# of clips at ~5-10 sec embedding extraction each).

echo "Start: $(date)"
echo "Node: $(hostname)"

source /home/manaal/miniforge3/etc/profile.d/conda.sh
conda activate child-vocalizations

export HF_HOME=/orcd/scratch/orcd/008/manaal/.cache/huggingface
export TORCH_HOME=/orcd/scratch/orcd/008/manaal/.cache/torch

cd /orcd/scratch/orcd/008/manaal/child-adult-diarization

pip install panns_inference --quiet

OUT=baselines/panns_baseline_runs/cnn14_cross_child_clean
mkdir -p "$OUT"

# ── Cross-child val (TRAINS LR head on baselines/splits/train.csv this time) ──
echo "=== PANNS CNN14 cross-child val (clean LR retrain) ==="
python baselines/panns_baseline.py --split val \
    --splits-dir baselines/splits \
    --output-dir "$OUT" \
    --seed 42

# ── Cross-child test (loads the LR head we just trained) ────────────────────
echo "=== PANNS CNN14 cross-child test (using freshly trained LR) ==="
python baselines/panns_baseline.py --split test \
    --splits-dir baselines/splits \
    --output-dir "$OUT" \
    --lr-weights "$OUT/lr_weights.npz" \
    --seed 42

echo "Done: $(date)"
