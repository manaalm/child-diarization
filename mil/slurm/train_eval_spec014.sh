#!/bin/bash
#SBATCH -c 4
#SBATCH -t 24:00:00
#SBATCH -p ou_bcs_normal,pi_satra
#SBATCH --mem=40G
#SBATCH --requeue
#SBATCH --gres=gpu:1
#SBATCH -o logs/mil/spec014_%j.out
#SBATCH -e logs/mil/spec014_%j.err

# Train + evaluate one spec-014 MIL config in a single SLURM job.
# Idempotent: if test_metrics_tuned.json already exists in the result dir, exits 0.
#
# Usage: sbatch mil/slurm/train_eval_spec014.sh mil/configs/<config>.yaml

set -euo pipefail

CONFIG=${1:?"Usage: sbatch train_eval_spec014.sh <config.yaml>"}

export PATH="$HOME/.local/bin:$PATH"
export PYTHONUNBUFFERED=1
# transformers >=4.57 bug: has_file() does network roundtrip even for cached
# models and misinterprets responses. Force offline mode.
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
source /orcd/home/002/manaal/miniforge3/etc/profile.d/conda.sh
conda activate child-vocalizations

REPO=/home/manaal/orcd/scratch/child-adult-diarization
cd "$REPO"
mkdir -p logs/mil

# Extract variant_name from yaml
VARIANT=$(python -c "import yaml,sys; print(yaml.safe_load(open('$CONFIG'))['variant_name'])")
RESULT_DIR="mil/mil_results/$VARIANT"
TEST_METRICS="$RESULT_DIR/test_metrics_tuned.json"

echo "=== spec-014 train+eval: $VARIANT (job $SLURM_JOB_ID) ==="
echo "  config:     $CONFIG"
echo "  result dir: $RESULT_DIR"

if [[ -f "$TEST_METRICS" ]]; then
    echo "  Already complete — $TEST_METRICS exists; skipping."
    exit 0
fi

# --- Train ---
if [[ ! -f "$RESULT_DIR/best_checkpoint.pt" ]]; then
    echo "--- Training ---"
    python mil/mil_train.py --config "$CONFIG"
else
    echo "--- Checkpoint exists, skipping training ---"
fi

# --- Evaluate ---
echo "--- Evaluating on test split ---"
python mil/mil_evaluate.py \
    --checkpoint "$RESULT_DIR/best_checkpoint.pt" \
    --config "$RESULT_DIR/config.json"

# --- Per-branch diagnostics for ACMIL configs ---
HEAD=$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG')).get('head', 'gated_abmil'))")
if [[ "$HEAD" == "acmil" ]]; then
    echo "--- ACMIL branch diagnostics ---"
    python mil/eval_acmil_branches.py --results-dir "$RESULT_DIR" --split test || true
fi

echo "=== Done at $(date) ==="
