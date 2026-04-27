#!/bin/bash
#SBATCH --job-name=synth_train
#SBATCH --output=logs/synth_train_%j.out
#SBATCH --error=logs/synth_train_%j.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

# Usage:
#   sbatch synthesis/slurm/train_synthesis.sh \
#       --config synthesis/configs/vits_34m.yaml --age-group 34_38m
#   sbatch synthesis/slurm/train_synthesis.sh \
#       --config synthesis/configs/vae_12m.yaml  --age-group 12_16m

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

mkdir -p logs

# Activate synthesis uv environment
# uv creates .venv inside synthesis/ directory
SYNTH_DIR="$REPO_ROOT/synthesis"
if [ ! -d "$SYNTH_DIR/.venv" ]; then
    echo "Setting up synthesis uv environment..."
    cd "$SYNTH_DIR"
    uv sync
    cd "$REPO_ROOT"
fi

PYTHON="$SYNTH_DIR/.venv/bin/python"
if [ ! -f "$PYTHON" ]; then
    echo "ERROR: Python not found at $PYTHON" >&2
    exit 1
fi

echo "Using Python: $PYTHON"
echo "CUDA available: $($PYTHON -c 'import torch; print(torch.cuda.is_available())')"
echo "Args: $*"

"$PYTHON" "$REPO_ROOT/synthesis/train.py" "$@"
