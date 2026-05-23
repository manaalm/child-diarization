#!/bin/bash
# Run unified_rttm.py on the held-out v2 synth eval set (200 scenes,
# balanced 100 pos / 100 neg) for one diarizer.
#
# Usage:
#   sbatch pyannote/run_synth_holdout_eval.sh <diarizer>
#
# Where <diarizer> is one of:
#   usc_sail | pyannote | babar | vtc | vtc_kchi | vbx | eend_eda | sortformer
#
# Output: pyannote/eval_results/<diarizer>_synth_holdout/
#
# Holdout selection: synth/scripts/build_synth_holdout_eval.py
# GT label: TARGET_CHILD (passed via --child-labels)

#SBATCH -c 4
#SBATCH -t 06:00:00
#SBATCH -p ou_bcs_normal,pi_satra
#SBATCH --mem=40G
#SBATCH --requeue
#SBATCH --gres=gpu:1
#SBATCH -o logs/rttm/synth_holdout_%j.out
#SBATCH -e logs/rttm/synth_holdout_%j.err

set -euo pipefail

DIARIZER="${1:?usage: $0 <diarizer>}"
REPO=/orcd/scratch/orcd/008/manaal/child-adult-diarization
HOLDOUT_DIR="${REPO}/synth_results/synthetic_scenes_v2/holdout_eval_200"
RESULTS_DIR="${REPO}/pyannote/eval_results/${DIARIZER}_synth_holdout"

source /orcd/home/002/manaal/miniforge3/etc/profile.d/conda.sh
conda activate child-vocalizations
module load ffmpeg/5.1.4 2>/dev/null || true
export LD_LIBRARY_PATH="/orcd/software/community/001/spack/pkg/ffmpeg/5.1.4/6kcopsg/lib:${LD_LIBRARY_PATH:-}"
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

mkdir -p logs/rttm "${RESULTS_DIR}"
cd "${REPO}/pyannote"

EXTRA_ARGS=()
if [ "${DIARIZER}" = "usc_sail" ] || [ "${DIARIZER}" = "usc_sail_synth" ]; then
    # CHUNKING WORKAROUND for the "USC-SAIL window_size=30 + ckpt mismatch"
    # gotcha (CLAUDE.md). The 5k pretrained ckpt has positional emb
    # [500,512] (window_size=10), so forcing window_size=30 fails state-dict
    # load. The fix is to keep window_size=10 — process_wav_file's existing
    # sliding window already chunks the 30s synth scenes into 10s sub-windows
    # (0-10, 5-15, ..., 25-30 with stride=5), and the WhisperWrapper
    # forward() dynamically resizes positional embeddings per-input via
    # tmp_length = get_feat_extract_output_lengths(160000) = 500, matching
    # the ckpt's [500,512]. The "transformers>=4.57 mel-3000 enforcement"
    # cited in the prior comment is handled by the Whisper feature
    # extractor's default 30 s padding (same path Playlogue/Providence
    # USC-SAIL evals already use).
    EXTRA_ARGS+=(--usc-window-size 10 --usc-stride 5)
fi

python unified_rttm.py \
    --dataset playlogue \
    --child-labels TARGET_CHILD \
    --audio-dir "${HOLDOUT_DIR}/wav" \
    --rttm-dir  "${HOLDOUT_DIR}/rttm" \
    --diarizer "${DIARIZER}" \
    --results-dir "${RESULTS_DIR}" \
    "${EXTRA_ARGS[@]}"

echo "DONE: ${DIARIZER} → ${RESULTS_DIR}"
