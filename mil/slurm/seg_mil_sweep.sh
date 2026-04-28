#!/bin/bash
#SBATCH -t 48:00:00
#SBATCH --gres=gpu:1
#SBATCH -p ou_bcs_normal,pi_satra
#SBATCH --mem=40G
#SBATCH -c 4
#SBATCH -o logs/mil/seg_mil_%j.out
#SBATCH -e logs/mil/seg_mil_%j.err

set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
export PYTHONUNBUFFERED=1
source /orcd/home/002/manaal/miniforge3/etc/profile.d/conda.sh
conda activate child-vocalizations

REPO=/home/manaal/orcd/scratch/child-adult-diarization

# Preflight: verify all 4 RTTM cache dirs are non-empty
echo "=== Preflight checks ==="
for RTTM_DIR in \
    "$REPO/whisper-modeling/usc_sail_rttm_cache" \
    "$REPO/pyannote/pyannote_rttm_cache" \
    "$REPO/pyannote/vtc_rttm_cache" \
    "$REPO/pyannote/vbx_rttm_cache"; do
    if [[ ! -d "$RTTM_DIR" ]]; then
        echo "ERROR: RTTM cache dir missing: $RTTM_DIR" >&2
        exit 1
    fi
    N=$(ls "$RTTM_DIR"/*.rttm 2>/dev/null | wc -l)
    if [[ "$N" -eq 0 ]]; then
        echo "ERROR: RTTM cache dir is empty: $RTTM_DIR" >&2
        echo "Run python pyannote/unified.py --diarizer <name> first" >&2
        exit 1
    fi
    echo "  OK: $RTTM_DIR ($N files)"
done

mkdir -p "$REPO/logs/mil"

cd "$REPO"
echo "=== Segment-instance MIL sweep ==="
python mil/seg_train.py --config mil/configs/seg_mil_sweep.yaml
