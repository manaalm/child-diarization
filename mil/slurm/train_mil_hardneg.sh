#!/bin/bash
#SBATCH -J mil_hardneg
#SBATCH -p ou_bcs_normal,pi_satra
#SBATCH -t 48:00:00
#SBATCH -c 4
#SBATCH --mem=40G
#SBATCH --gres=gpu:1
#SBATCH -o logs/mil/hardneg_%j.out
#SBATCH -e logs/mil/hardneg_%j.err

set -e
cd /orcd/scratch/orcd/008/manaal/child-adult-diarization
source /home/manaal/miniforge3/etc/profile.d/conda.sh
conda activate child-vocalizations

mkdir -p logs/mil synth_results/manifests

# Step 1: extract hard negatives (CPU only, ~5 min)
if [ ! -f synth_results/manifests/hard_negatives_manifest.csv ]; then
    echo "=== Extracting hard negatives ==="
    python mil/scripts/extract_hard_negatives.py \
        --output synth_results/manifests/hard_negatives_manifest.csv \
        --window-sec 30 \
        --stride-sec 15 \
        --min-activity-sec 3 \
        --max-per-file 20 \
        --seed 42
else
    echo "=== Hard negatives manifest already exists, skipping extraction ==="
    wc -l synth_results/manifests/hard_negatives_manifest.csv
fi

# Step 2: train WavLM-MIL with hard negatives
echo "=== Training WavLM-MIL (hard negatives) ==="
python mil/mil_train.py --config mil/configs/wavlm_mil_hardneg.yaml

# Step 3: train Whisper-MIL with hard negatives
echo "=== Training Whisper-MIL (hard negatives) ==="
python mil/mil_train.py --config mil/configs/whisper_mil_hardneg.yaml

# Step 4: evaluate both
echo "=== Evaluating hard-neg MIL variants ==="
for variant in wavlm_mil_hardneg whisper_mil_hardneg; do
    ckpt="mil/mil_results/${variant}/best_checkpoint.pt"
    cfg_json="mil/mil_results/${variant}/config.json"
    if [ -f "$ckpt" ]; then
        python mil/mil_evaluate.py --checkpoint "$ckpt" --config "$cfg_json"
    fi
done

echo "Done. Results in mil/mil_results/{wavlm_mil_hardneg,whisper_mil_hardneg}/"
