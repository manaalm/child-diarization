#!/bin/bash
# Run USC-SAIL joint ASR+diarization on a Providence sample (long-form, chunked
# at 30 s). Providence files are ~58 min each; 30 randomly-sampled files at
# ~2 sec/chunk ≈ 2 hours of GPU. Wall budget 6 h with margin.
#
# Sample: deterministic (seed 43), .wav files only (avoid alex_*.mp3 to side-
# step transcoding overhead), 30 files. Larger sweep would scale linearly.
#
# NOTE: model was TRAINED ON PLAYLOGUE → Providence is a clean OOD eval.

#SBATCH -c 4
#SBATCH -t 06:00:00
#SBATCH -p ou_bcs_normal,pi_satra
#SBATCH --mem=40G
#SBATCH --requeue
#SBATCH --gres=gpu:1
#SBATCH -o logs/joint_asr_diar/providence_%j.out
#SBATCH -e logs/joint_asr_diar/providence_%j.err

set -euo pipefail

REPO=/orcd/scratch/orcd/008/manaal/child-adult-diarization
PROVIDENCE_AUDIO="${REPO}/providence/audio"
PROVIDENCE_RTTM="${REPO}/providence/rttm"
RESULTS_DIR="${REPO}/pyannote/eval_results/joint_asr_diar_providence"
FILE_LIST="${REPO}/synth_results/manifests/joint_asr_diar_providence_pilot.txt"

source /orcd/home/002/manaal/miniforge3/etc/profile.d/conda.sh
conda activate joint_asr_diar
unset HF_TOKEN HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
export CUDA_LAUNCH_BLOCKING=1

mkdir -p "${REPO}/logs/joint_asr_diar" "${RESULTS_DIR}"
cd "${REPO}"

# Build deterministic 30-file pilot list (.wav only, sorted, seeded shuffle)
python -c "
import os, random
audio_dir = '${PROVIDENCE_AUDIO}'
rttm_dir = '${PROVIDENCE_RTTM}'
wavs = sorted(f for f in os.listdir(audio_dir) if f.endswith('.wav'))
# Keep only those with matching RTTMs
rttms = {os.path.splitext(f)[0] for f in os.listdir(rttm_dir) if f.endswith('.rttm')}
matched = [w for w in wavs if os.path.splitext(w)[0] in rttms]
print(f'matched {len(matched)} wavs with rttms (out of {len(wavs)})')
rng = random.Random(43)
sample = sorted(rng.sample(matched, min(30, len(matched))))
with open('${FILE_LIST}', 'w') as f:
    for s in sample:
        f.write(os.path.join(audio_dir, s) + '\n')
print(f'wrote {len(sample)} files to ${FILE_LIST}')
"

echo "=== Stage 1: chunked batch inference (30s chunks) ==="
python baselines/joint_asr_diar_batch.py \
    --file-list "${FILE_LIST}" \
    --wav-dir "${PROVIDENCE_AUDIO}" \
    --results-dir "${RESULTS_DIR}" \
    --device cuda \
    --max-len 300 \
    --chunk-sec 30

echo "=== Stage 2: scoring against Providence GT (CHI label) ==="
python baselines/score_joint_asr_diar.py \
    --results-dir "${RESULTS_DIR}" \
    --gt-dir "${PROVIDENCE_RTTM}" \
    --audio-dir "${PROVIDENCE_AUDIO}" \
    --gt-child-labels CHI

echo "DONE: joint ASR+diar Providence pilot → ${RESULTS_DIR}"
