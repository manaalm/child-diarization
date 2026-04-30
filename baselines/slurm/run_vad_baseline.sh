#!/bin/bash
#SBATCH --job-name=vad_baseline
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH -t 4:00:00
#SBATCH -p ou_bcs_normal,pi_satra
#SBATCH -o logs/baselines/vad_%j.out
#SBATCH -e logs/baselines/vad_%j.out

echo "Start: $(date)"
echo "Node: $(hostname)"

source /home/manaal/miniforge3/etc/profile.d/conda.sh
conda activate child-vocalizations

export HF_HOME=/orcd/scratch/orcd/008/manaal/.cache/huggingface
export TORCH_HOME=/orcd/scratch/orcd/008/manaal/.cache/torch

cd /orcd/scratch/orcd/008/manaal/child-adult-diarization

# ── Seen-child split ──────────────────────────────────────────────────────
echo "=== Silero VAD (seen-child val) ==="
python baselines/vad_baseline.py --mode silero --split val --seed 42

echo "=== Silero VAD (seen-child test) ==="
python baselines/vad_baseline.py --mode silero --split test --seed 42

echo "=== Energy VAD (seen-child val) ==="
python baselines/vad_baseline.py --mode energy --split val --seed 42

echo "=== Energy VAD (seen-child test) ==="
python baselines/vad_baseline.py --mode energy --split test --seed 42

# ── Cross-child split ─────────────────────────────────────────────────────
echo "=== Silero VAD (cross-child val) ==="
python baselines/vad_baseline.py --mode silero --split val \
    --splits-dir baselines/splits \
    --output-dir baselines/vad_baseline_runs/silero_cross_child \
    --seed 42

echo "=== Silero VAD (cross-child test) ==="
python baselines/vad_baseline.py --mode silero --split test \
    --splits-dir baselines/splits \
    --output-dir baselines/vad_baseline_runs/silero_cross_child \
    --seed 42

echo "=== Energy VAD (cross-child val) ==="
python baselines/vad_baseline.py --mode energy --split val \
    --splits-dir baselines/splits \
    --output-dir baselines/vad_baseline_runs/energy_cross_child \
    --seed 42

echo "=== Energy VAD (cross-child test) ==="
python baselines/vad_baseline.py --mode energy --split test \
    --splits-dir baselines/splits \
    --output-dir baselines/vad_baseline_runs/energy_cross_child \
    --seed 42

echo "Done: $(date)"
