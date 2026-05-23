#!/bin/bash
#SBATCH -J bark_smoke
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --mem=32G
#SBATCH -o logs/synth/bark_smoke_%j.out
#SBATCH -e logs/synth/bark_smoke_%j.err

set -euo pipefail
cd /home/manaal/orcd/scratch/child-adult-diarization
source /orcd/home/002/manaal/miniforge3/etc/profile.d/conda.sh
conda activate child-vocalizations

export TRANSFORMERS_OFFLINE=0   # need to download Bark weights first run
export HF_HUB_OFFLINE=0
unset HF_TOKEN HUGGINGFACE_HUB_TOKEN HF_HUB_TOKEN

mkdir -p logs/synth synth/synth_voice/spec019_bark_smoke

# 1. Unconditioned smoke test (no voice preset)
python synth/synth_voice/bark_smoke.py \
    --out-dir synth/synth_voice/spec019_bark_smoke/unconditioned \
    --n-per-prompt 2 --model suno/bark-small

# 2. With a generic voice preset (smaller speaker, female; closest baseline to "child")
python synth/synth_voice/bark_smoke.py \
    --out-dir synth/synth_voice/spec019_bark_smoke/preset_speaker9 \
    --n-per-prompt 2 --model suno/bark-small \
    --voice-preset v2/en_speaker_9
