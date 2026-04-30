#!/bin/bash
#SBATCH -t 24:00:00
#SBATCH --gres=gpu:1
#SBATCH -p ou_bcs_normal,pi_satra
#SBATCH --mem=40G
#SBATCH -c 4
#SBATCH -o logs/mil/seg_mil_synth_%j.out
#SBATCH -e logs/mil/seg_mil_synth_%j.err

set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
export PYTHONUNBUFFERED=1
source /orcd/home/002/manaal/miniforge3/etc/profile.d/conda.sh
conda activate child-vocalizations

REPO=/home/manaal/orcd/scratch/child-adult-diarization
cd "$REPO"
mkdir -p logs/mil

CONFIG=${1:-mil/configs/seg_mil_synth.yaml}

echo "=== Seg-MIL synth-augmentation training ==="
echo "config: $CONFIG"

# Verify combined cache is populated
N=$(ls mil/seg_mil_combined_cache/*.rttm 2>/dev/null | wc -l)
echo "Combined RTTM cache: $N entries"
if [[ "$N" -lt 7000 ]]; then
    echo "ERROR: combined cache too small (expected ~7100)" >&2
    exit 1
fi

python mil/seg_train.py --config "$CONFIG"
echo "=== Done $(date) ==="
