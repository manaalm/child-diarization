#!/bin/bash
#SBATCH --job-name=xling_vc
#SBATCH --output=logs/adult/cross_lingual_vc_%A_%a.out
#SBATCH --error=logs/adult/cross_lingual_vc_%A_%a.out
#SBATCH --partition=ou_bcs_normal
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --array=0-3

set -euo pipefail

REPO=/orcd/scratch/orcd/008/manaal/child-adult-diarization
ENV_PY=/orcd/home/002/manaal/miniforge3/envs/knnvc/bin/python
cd "$REPO"

N_SHARDS=4
SHARD="${SLURM_ARRAY_TASK_ID:-0}"

# Offline / cache safety (transformers >=4.57 has_file() bug — see CLAUDE.md).
export TRANSFORMERS_OFFLINE=0
export HF_HUB_OFFLINE=0
unset HF_TOKEN HUGGINGFACE_HUB_TOKEN HF_HUB_TOKEN

OUT_DIR="data/segments/cross_lingual_vc"
OUT_MAN="synth_results/manifests/cross_lingual_vc_manifest.csv"

mkdir -p "$OUT_DIR"
mkdir -p "$(dirname "$OUT_MAN")"

"$ENV_PY" synth/scripts/cross_lingual_tinyvox_vc.py \
    --tinyvox-audio-dir data/tinyvox/audio \
    --children-csv      whisper-modeling/seen_child_splits/train.csv \
    --output-dir        "$OUT_DIR" \
    --output-manifest   "$OUT_MAN" \
    --n-per-target      40 \
    --max-targets       100 \
    --device            cuda \
    --shard-id          "$SHARD" \
    --n-shards          "$N_SHARDS"

echo "[done] cross-lingual VC shard=$SHARD"
