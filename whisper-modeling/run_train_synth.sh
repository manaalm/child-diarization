#!/bin/bash
#SBATCH -t 24:00:00
#SBATCH --gres=gpu:1
#SBATCH -p ou_bcs_normal,pi_satra
#SBATCH --mem=80G
#SBATCH -c 4
#SBATCH -o logs/whisper_modeling/train_synth_%j.out
#SBATCH -e logs/whisper_modeling/train_synth_%j.err

set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
export PYTHONUNBUFFERED=1
source /orcd/home/002/manaal/miniforge3/etc/profile.d/conda.sh
conda activate child-vocalizations

REPO=/orcd/scratch/orcd/008/manaal/child-adult-diarization
cd "$REPO/whisper-modeling"
mkdir -p "$REPO/logs/whisper_modeling"

# Verify data layout
TRAIN_N=$(ls "$REPO/synth_results/usc_sail_data/audios/train/" | wc -l)
VAL_N=$(ls "$REPO/synth_results/usc_sail_data/audios/val/" | wc -l)
echo "Train clips: $TRAIN_N  Val clips: $VAL_N"
if [[ "$TRAIN_N" -lt 4000 ]]; then
    echo "ERROR: train data missing" >&2
    exit 1
fi

echo "=== USC-SAIL synth training ==="
PYTHONPATH=. python scripts/main.py --debug f --config configs/config_synth.yaml
echo "=== Done $(date) ==="
