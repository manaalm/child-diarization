#!/bin/bash
# Run joint ASR+diar on the SAILS seen-child *train* split (n=1311) so
# the unified.py enrollment pipeline can build per-(child, timepoint)
# ECAPA prototypes from joint_asr_diar's own CHI segments rather than
# borrowing them from BabAR. After this completes, run:
#   python pyannote/unified.py --diarizer joint_asr_diar
# to score val+test against per-clip prototypes.

#SBATCH -c 4
#SBATCH -t 04:00:00
#SBATCH -p ou_bcs_normal,pi_satra
#SBATCH --mem=40G
#SBATCH --requeue
#SBATCH --gres=gpu:1
#SBATCH -o logs/joint_asr_diar/sails_train_%j.out
#SBATCH -e logs/joint_asr_diar/sails_train_%j.err

set -euo pipefail

REPO=/orcd/scratch/orcd/008/manaal/child-adult-diarization
SPLITS_DIR="${REPO}/whisper-modeling/seen_child_splits"
RESULTS_DIR="${REPO}/pyannote/eval_results/joint_asr_diar_sails"
FILE_LIST="${REPO}/synth_results/manifests/joint_asr_diar_sails_train_filelist.txt"

source /orcd/home/002/manaal/miniforge3/etc/profile.d/conda.sh
conda activate joint_asr_diar
unset HF_TOKEN HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
export CUDA_LAUNCH_BLOCKING=1

mkdir -p "${REPO}/logs/joint_asr_diar"
cd "${REPO}"

# Build train file list (skip clips already in the val+test predictions cache)
python -c "
import csv, os
seen = set()
existing = '${RESULTS_DIR}/per_file_predictions'
if os.path.isdir(existing):
    for f in os.listdir(existing):
        if f.endswith('_pred.rttm'):
            seen.add(f[:-len('_pred.rttm')])
with open('${FILE_LIST}', 'w') as out:
    n = 0
    with open('${SPLITS_DIR}/train.csv') as f:
        reader = csv.DictReader(f)
        for row in reader:
            ap = row.get('audio_path', '').strip()
            ae = row.get('audio_exists', 'True').strip().lower()
            if not ap or ae != 'true' or not os.path.exists(ap):
                continue
            stem = os.path.splitext(os.path.basename(ap))[0]
            if stem in seen:
                continue
            out.write(ap + '\n')
            n += 1
print(f'wrote {n} train clips to ${FILE_LIST}')
"

echo '=== Joint ASR+diar inference on train clips ==='
python baselines/joint_asr_diar_batch.py \
    --file-list "${FILE_LIST}" \
    --wav-dir "${SPLITS_DIR}" \
    --results-dir "${RESULTS_DIR}" \
    --device cuda \
    --max-len 300 \
    --chunk-sec 30

echo "DONE: train clips processed; RTTMs in ${RESULTS_DIR}/per_file_predictions/"
