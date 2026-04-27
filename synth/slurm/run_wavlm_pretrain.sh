#!/bin/bash
#SBATCH --job-name=wavlm_pretrain
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:1
#SBATCH -p ou_bcs_normal,pi_satra
#SBATCH --output=logs/synth/wavlm_pretrain_%j.out
#SBATCH --error=logs/synth/wavlm_pretrain_%j.out

# Continued masked-speech-unit pretraining of WavLM-Base+ on child speech.
# Outputs checkpoint to synth_results/child_wavlm_checkpoint/.
# If a checkpoint already exists, resumes from the most recent step_* directory.
#
# Usage:
#   sbatch synth/slurm/run_wavlm_pretrain.sh
#   sbatch synth/slurm/run_wavlm_pretrain.sh --max-steps 100000 --batch-size 16

set -euo pipefail

source /home/manaal/miniforge3/etc/profile.d/conda.sh
conda activate child-vocalizations

REPO_ROOT="/orcd/scratch/orcd/008/manaal/child-adult-diarization"
cd "${REPO_ROOT}"

mkdir -p logs/synth synth_results/child_wavlm_checkpoint

echo "Job ID:   ${SLURM_JOB_ID}"
echo "Node:     $(hostname)"
echo "GPU:      $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'none')"
echo "Start:    $(date)"

WAV_LIST="synth_results/child_wavs.txt"
OUTPUT_DIR="synth_results/child_wavlm_checkpoint"
EXTRA_ARGS="${@}"  # pass any extra CLI args (e.g. --max-steps, --batch-size)

# Auto-resume from most recent checkpoint if one exists
RESUME_ARG=""
LATEST=$(ls -d "${OUTPUT_DIR}"/step_* 2>/dev/null | sort -V | tail -1)
if [[ -n "${LATEST}" && -f "${LATEST}/trainer_state.pt" ]]; then
    echo "Resuming from checkpoint: ${LATEST}"
    RESUME_ARG="--resume-from-checkpoint ${LATEST}"
fi

python synth/scripts/pretrain_wavlm_child.py \
    --wav-list    "${WAV_LIST}" \
    --output-dir  "${OUTPUT_DIR}" \
    --max-steps   50000 \
    --batch-size  8 \
    --lr          1e-4 \
    --save-every  5000 \
    --log-every   100 \
    --seed        42 \
    ${RESUME_ARG} \
    ${EXTRA_ARGS}

echo "Done: $(date)"
