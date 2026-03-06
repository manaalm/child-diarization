#!/bin/bash

#SBATCH -c 1
#SBATCH -t 12:00:00
#SBATCH -p ou_bcs_normal,pi_satra
#SBATCH --mem=40G
#SBATCH --requeue
#SBATCH --gres=gpu:1
#SBATCH -o logs/adult/features_%A_%a.out
#SBATCH -e logs/adult/features_%A_%a.err
#SBATCH --array=0-155

set -euo pipefail

source /orcd/home/002/manaal/miniforge3/etc/profile.d/conda.sh
conda activate child-vocalizations

# Always run from the directory you submitted from
cd "$SLURM_SUBMIT_DIR"

# Adjust these if your folder names differ
WHISPER_DIR="$SLURM_SUBMIT_DIR/whisper-modeling"
WAV_LIST="$SLURM_SUBMIT_DIR/wavs.txt"
OUT_DIR="$SLURM_SUBMIT_DIR/playlogue/rttm"
MODEL_PATH="$WHISPER_DIR/whisper-base_rank8_pretrained_50k.pt"

mkdir -p "$OUT_DIR" logs/adult

# Get wav for this array task
WAV_FILE=$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" "$WAV_LIST")

if [[ "$WAV_FILE" != /* ]]; then
  WAV_FILE="$SLURM_SUBMIT_DIR/$WAV_FILE"
fi

echo "PWD: $(pwd)"
echo "WHISPER_DIR: $WHISPER_DIR"
echo "WAV_FILE: $WAV_FILE"
echo "MODEL_PATH: $MODEL_PATH"
echo "OUT_DIR: $OUT_DIR"

# Make the repo importable (models.whisper)
export PYTHONPATH="$WHISPER_DIR"

# Run from inside whisper-modeling so any relative file refs behave
cd "$WHISPER_DIR"

python scripts/infer_long_wav_files.py \
  --wav_file "$WAV_FILE" \
  --out_dir "$OUT_DIR" \
  --model_path "$MODEL_PATH" \
  --device cuda \
  --window_size 10 \
  --stride 5

echo "Done."