#!/bin/bash
#SBATCH --job-name=cleese_child
#SBATCH --output=logs/adult/cleese_childrenization_%A_%a.out
#SBATCH --error=logs/adult/cleese_childrenization_%A_%a.out
#SBATCH --partition=ou_bcs_normal
#SBATCH --time=8:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --array=0-7

set -euo pipefail

REPO=/orcd/scratch/orcd/008/manaal/child-adult-diarization
ENV_PY=/orcd/home/002/manaal/miniforge3/envs/child-vocalizations/bin/python
cd "$REPO"

N_SHARDS=8
SHARD="${SLURM_ARRAY_TASK_ID:-0}"

OUT_DIR="data/segments/cleese_childrenized"
OUT_MAN="synth_results/manifests/cleese_childrenized_manifest.csv"

mkdir -p "$OUT_DIR"
mkdir -p "$(dirname "$OUT_MAN")"

"$ENV_PY" synth/scripts/cleese_childrenization.py \
    --segment-manifest synth_results/manifests/segment_manifest_v2.csv \
    --output-dir       "$OUT_DIR" \
    --output-manifest  "$OUT_MAN" \
    --source-datasets  "librispeech,providence_adults,playlogue_adults" \
    --max-segments     32000 \
    --seed             42 \
    --shard-id         "$SHARD" \
    --n-shards         "$N_SHARDS"

echo "[done] CLEESE childrenization shard=$SHARD"
