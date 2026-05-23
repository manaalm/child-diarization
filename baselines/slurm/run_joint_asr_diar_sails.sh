#!/bin/bash
# Run USC-SAIL joint ASR+diar on the SAILS BIDS seen-child val+test split
# (n=872) and produce clip-level child-presence scores.
#
# Pipeline:
#   1. Build file list from val.csv + test.csv (BidsProcessed audio paths)
#   2. Run joint_asr_diar_batch.py on each clip → per-file RTTM
#   3. Run score_joint_asr_diar_sails.py to convert RTTMs → clip-level
#      scores, val-tune threshold, write {val,test}_metrics_tuned.json
#
# Compute: 872 clips × ~3 s/clip (most ≤30 s, single-pass, no chunking) ≈
# 45 min on 1 A100; budget 4 h with margin.

#SBATCH -c 4
#SBATCH -t 04:00:00
#SBATCH -p ou_bcs_normal,pi_satra
#SBATCH --mem=40G
#SBATCH --requeue
#SBATCH --gres=gpu:1
#SBATCH -o logs/joint_asr_diar/sails_%j.out
#SBATCH -e logs/joint_asr_diar/sails_%j.err

set -euo pipefail

REPO=/orcd/scratch/orcd/008/manaal/child-adult-diarization
SPLITS_DIR="${REPO}/whisper-modeling/seen_child_splits"
RESULTS_DIR="${REPO}/pyannote/eval_results/joint_asr_diar_sails"
OUTPUT_DIR="${REPO}/joint_asr_diar_sails_runs"
FILE_LIST="${REPO}/synth_results/manifests/joint_asr_diar_sails_filelist.txt"

source /orcd/home/002/manaal/miniforge3/etc/profile.d/conda.sh
conda activate joint_asr_diar
unset HF_TOKEN HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
export CUDA_LAUNCH_BLOCKING=1

mkdir -p "${REPO}/logs/joint_asr_diar" "${RESULTS_DIR}" "${OUTPUT_DIR}"
cd "${REPO}"

# Build val+test file list from BidsProcessed audio_path column
python -c "
import csv, os
splits = ['${SPLITS_DIR}/val.csv', '${SPLITS_DIR}/test.csv']
seen = set()
with open('${FILE_LIST}', 'w') as out:
    for path in splits:
        with open(path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                ap = row.get('audio_path', '').strip()
                ae = row.get('audio_exists', 'True').strip().lower()
                if not ap or ae != 'true' or ap in seen:
                    continue
                if not os.path.exists(ap):
                    continue
                seen.add(ap)
                out.write(ap + '\n')
print(f'wrote {len(seen)} unique SAILS clips to ${FILE_LIST}')
"

echo "=== Stage 1: batch inference (single-pass, no chunking — clips ≤30 s) ==="
python baselines/joint_asr_diar_batch.py \
    --file-list "${FILE_LIST}" \
    --wav-dir "${SPLITS_DIR}" \
    --results-dir "${RESULTS_DIR}" \
    --device cuda \
    --max-len 300 \
    --chunk-sec 30

echo "=== Stage 2: clip-level scoring + threshold tuning ==="
python baselines/score_joint_asr_diar_sails.py \
    --results-dir "${RESULTS_DIR}" \
    --val-csv "${SPLITS_DIR}/val.csv" \
    --test-csv "${SPLITS_DIR}/test.csv" \
    --output-dir "${OUTPUT_DIR}"

echo "DONE: joint ASR+diar SAILS run → ${OUTPUT_DIR}"
