#!/bin/bash
# Run USC-SAIL joint ASR+diarization (AlexXu811/child-adult-joint-asr-diarization)
# on the synth holdout (200 v2 scenes) and the seen-child SAILS test split.
#
# Output layout:
#   pyannote/eval_results/joint_asr_diar_synth_holdout/
#     per_file_predictions/*.rttm
#     raw_predictions/*.txt
#     aggregate_metrics.json
#     per_file_metrics.csv
#
# Both evaluation/frame_localization_gt.py and evaluation/onset_tolerance_f1.py
# auto-discover the new directory if its name ends in `_synth_holdout`.
#
# NOTE: This model was TRAINED ON PLAYLOGUE → eval on Playlogue is circular
# (do not score against playlogue/rttm/). Synth holdout is independently
# generated; Providence is independent. SAILS BIDS clip-level eval is also
# clean for joint-model purposes.

#SBATCH -c 4
#SBATCH -t 04:00:00
#SBATCH -p ou_bcs_normal,pi_satra
#SBATCH --mem=40G
#SBATCH --requeue
#SBATCH --gres=gpu:1
#SBATCH -o logs/joint_asr_diar/synth_%j.out
#SBATCH -e logs/joint_asr_diar/synth_%j.err

set -euo pipefail

REPO=/orcd/scratch/orcd/008/manaal/child-adult-diarization
HOLDOUT_WAV="${REPO}/synth_results/synthetic_scenes_v2/holdout_eval_200/wav"
HOLDOUT_RTTM="${REPO}/synth_results/synthetic_scenes_v2/holdout_eval_200/rttm"
RESULTS_DIR="${REPO}/pyannote/eval_results/joint_asr_diar_synth_holdout"

source /orcd/home/002/manaal/miniforge3/etc/profile.d/conda.sh
conda activate joint_asr_diar
unset HF_TOKEN HF_HUB_OFFLINE TRANSFORMERS_OFFLINE

mkdir -p "${REPO}/logs/joint_asr_diar" "${RESULTS_DIR}"
cd "${REPO}"

echo "=== Stage 1: batch inference ==="
export CUDA_LAUNCH_BLOCKING=1
python baselines/joint_asr_diar_batch.py \
    --wav-dir "${HOLDOUT_WAV}" \
    --results-dir "${RESULTS_DIR}" \
    --device cuda \
    --max-len 300

echo "=== Stage 2: scoring against synth GT (TARGET_CHILD label) ==="
python baselines/score_joint_asr_diar_synth.py \
    --results-dir "${RESULTS_DIR}" \
    --gt-dir "${HOLDOUT_RTTM}" \
    --gt-child-label TARGET_CHILD

echo "DONE: joint ASR+diar → ${RESULTS_DIR}"
