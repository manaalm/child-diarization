#!/bin/bash
#SBATCH --job-name=clap_baseline
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH -c 4
#SBATCH -t 6:00:00
#SBATCH -p ou_bcs_normal,pi_satra
#SBATCH -o logs/baselines/clap_%j.out
#SBATCH -e logs/baselines/clap_%j.out

echo "Start: $(date)"
echo "Node: $(hostname)"

source /home/manaal/miniforge3/etc/profile.d/conda.sh
conda activate child-vocalizations

export HF_HOME=/orcd/scratch/orcd/008/manaal/.cache/huggingface
export TORCH_HOME=/orcd/scratch/orcd/008/manaal/.cache/torch

cd /orcd/scratch/orcd/008/manaal/child-adult-diarization

# ── Seen-child split ──────────────────────────────────────────────────────
echo "=== CLAP (seen-child val) ==="
python baselines/clap_baseline.py --split val --seed 42

echo "=== CLAP (seen-child test) ==="
python baselines/clap_baseline.py --split test --seed 42

# ── Cross-child split ─────────────────────────────────────────────────────
echo "=== CLAP (cross-child val) ==="
python baselines/clap_baseline.py --split val \
    --splits-dir baselines/splits \
    --output-dir baselines/clap_baseline_runs/clap_htsat_fused_cross_child \
    --seed 42

echo "=== CLAP (cross-child test) ==="
python baselines/clap_baseline.py --split test \
    --splits-dir baselines/splits \
    --output-dir baselines/clap_baseline_runs/clap_htsat_fused_cross_child \
    --seed 42

echo "Done: $(date)"
