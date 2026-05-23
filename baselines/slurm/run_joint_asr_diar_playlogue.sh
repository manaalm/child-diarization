#!/bin/bash
# =====================================================================
# DO NOT RUN. The released joint ASR+diarization checkpoint
# (AlexXu811/child-adult-joint-asr-diarization) was trained on
# Playlogue. Evaluating it here would be training-set evaluation,
# not a held-out result, and would not be comparable to the other
# systems' Playlogue numbers in the localization chapter table.
#
# This script is kept for completeness (it mirrors the Providence
# pilot launcher and demonstrates the case-insensitive .mp3↔.rttm
# matcher needed for Playlogue), but it should NEVER be sbatched.
# Use the Providence pilot (run_joint_asr_diar_providence.sh) for
# the held-out OOD eval; use the synth-holdout pilot for short-clip
# evaluation. See thesis_v2/chapters/06_localization_diagnostics.tex
# §sec:localization-joint "Important training-data caveat".
#
# Original SLURM launch on 2026-05-05 (job 13376487) was cancelled
# upon recognizing this issue.
# =====================================================================
exit 1

# Run USC-SAIL joint ASR+diarization on a Playlogue sample (long-form, chunked
# at 30 s). Mirrors run_joint_asr_diar_providence.sh.
#
# NOTE: Playlogue audio is .mp3 with uppercase basename component (AAE),
# while RTTMs are lowercase (aae) — matcher uses casefold for cross-platform
# robustness. Sample size = 30 deterministic (seed 43) wav/mp3 files.

#SBATCH -c 4
#SBATCH -t 06:00:00
#SBATCH -p ou_bcs_normal,pi_satra
#SBATCH --mem=40G
#SBATCH --requeue
#SBATCH --gres=gpu:1
#SBATCH -o logs/joint_asr_diar/playlogue_%j.out
#SBATCH -e logs/joint_asr_diar/playlogue_%j.err

set -euo pipefail

REPO=/orcd/scratch/orcd/008/manaal/child-adult-diarization
PLAYLOGUE_AUDIO="${REPO}/playlogue/audio"
PLAYLOGUE_RTTM="${REPO}/playlogue/rttm"
RESULTS_DIR="${REPO}/pyannote/eval_results/joint_asr_diar_playlogue"
FILE_LIST="${REPO}/synth_results/manifests/joint_asr_diar_playlogue_pilot.txt"

source /orcd/home/002/manaal/miniforge3/etc/profile.d/conda.sh
conda activate joint_asr_diar
unset HF_TOKEN HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
export CUDA_LAUNCH_BLOCKING=1

mkdir -p "${REPO}/logs/joint_asr_diar" "${RESULTS_DIR}"
cd "${REPO}"

# Build deterministic 30-file pilot list with case-insensitive basename matching
python -c "
import os, random
audio_dir = '${PLAYLOGUE_AUDIO}'
rttm_dir = '${PLAYLOGUE_RTTM}'
audio_files = sorted(f for f in os.listdir(audio_dir) if f.endswith(('.wav', '.mp3')))
rttm_basenames = {os.path.splitext(f)[0].casefold() for f in os.listdir(rttm_dir) if f.endswith('.rttm')}
matched = [w for w in audio_files if os.path.splitext(w)[0].casefold() in rttm_basenames]
print(f'matched {len(matched)} audio files with rttms (out of {len(audio_files)})')
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
    --wav-dir "${PLAYLOGUE_AUDIO}" \
    --results-dir "${RESULTS_DIR}" \
    --device cuda \
    --max-len 300 \
    --chunk-sec 30

echo "=== Stage 2: scoring against Playlogue GT (CHI label) ==="
python baselines/score_joint_asr_diar.py \
    --results-dir "${RESULTS_DIR}" \
    --gt-dir "${PLAYLOGUE_RTTM}" \
    --audio-dir "${PLAYLOGUE_AUDIO}" \
    --gt-child-labels CHI

echo "DONE: joint ASR+diar Playlogue pilot → ${RESULTS_DIR}"
