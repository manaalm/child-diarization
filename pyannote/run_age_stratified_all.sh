#!/bin/bash
#SBATCH --job-name=age_strat_all
#SBATCH -c 8
#SBATCH -t 12:00:00
#SBATCH -p ou_bcs_normal,pi_satra
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH -o logs/enrollment/age_strat_all_%j.out
#SBATCH -e logs/enrollment/age_strat_all_%j.out

# Run unified_age_stratified.py for all 6 diarizers × 2 age groups.
# All RTTM caches are already populated so diarization is skipped; only
# ECAPA prototype construction and threshold tuning re-run per cohort.
# Output: pyannote/{diarizer}_age_stratified/{12_16m,34_38m}/

set -euo pipefail
source /orcd/home/002/manaal/miniforge3/etc/profile.d/conda.sh
conda activate child-vocalizations
module load ffmpeg/5.1.4
export LD_LIBRARY_PATH="/orcd/software/community/001/spack/pkg/ffmpeg/5.1.4/6kcopsg/lib:${LD_LIBRARY_PATH:-}"
export HF_TOKEN=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

cd /orcd/scratch/orcd/008/manaal/child-adult-diarization

mkdir -p logs/enrollment
echo "Start: $(date)"

for DIARIZER in usc_sail pyannote babar vtc vtc_kchi vbx; do
    for AGE_GROUP in 12_16m 34_38m; do
        OUT_DIR="pyannote/${DIARIZER}_age_stratified/${AGE_GROUP}"
        if [[ -f "${OUT_DIR}/test_metrics_tuned.json" ]]; then
            echo "[SKIP] ${DIARIZER}/${AGE_GROUP} already done"
            continue
        fi
        echo ""
        echo "=== ${DIARIZER} / ${AGE_GROUP} ==="
        python pyannote/unified_age_stratified.py \
            --diarizer "${DIARIZER}" \
            --age-group "${AGE_GROUP}" \
            --output-dir "${OUT_DIR}"
        echo "  done -> ${OUT_DIR}"
    done
done

echo ""
echo "All age-stratified runs complete. End: $(date)"
