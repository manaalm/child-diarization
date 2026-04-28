#!/bin/bash
#SBATCH --job-name=talknet_finetune
#SBATCH -c 8
#SBATCH -t 10:00:00
#SBATCH -p ou_bcs_normal,pi_satra
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH -o logs/video/talknet_finetune_%j.out
#SBATCH -e logs/video/talknet_finetune_%j.out

# Fine-tune TalkNet-ASD for clip-level child vocalization detection.
# Uses auto-downloadable pretrained TalkNet-ASD checkpoint as backbone.
# Replaces TS-TalkNet (spec-007 T026, checkpoint unavailable from authors).
#
# Phase 1 (5 ep): freeze backbone, train head only
# Phase 2 (15 ep): unfreeze backbone, fine-tune everything
# Total wall time: ~2 h precompute + ~4 h training + ~30 min eval
#
# Output: video_finetuned_talknet_runs/{best_checkpoint.pt, val/test_metrics_tuned.json,
#          val/test_predictions.csv, config.json}

set -euo pipefail

cd /orcd/scratch/orcd/008/manaal/child-adult-diarization

module load ffmpeg/5.1.4
export LD_LIBRARY_PATH="/orcd/software/community/001/spack/pkg/ffmpeg/5.1.4/6kcopsg/lib:${LD_LIBRARY_PATH:-}"

# Use the video/ Python 3.10 uv-managed environment
UV_PYTHON="video/.venv/bin/python"

echo "=== TalkNet child-vocalization fine-tuning ==="
echo "Start: $(date)"
echo "Python: $UV_PYTHON"

$UV_PYTHON video/talknet_child_finetune.py \
    --pretrain-path video/pretrain/talknet_asd.model \
    --train-csv whisper-modeling/seen_child_splits/train.csv \
    --val-csv   whisper-modeling/seen_child_splits/val.csv \
    --test-csv  whisper-modeling/seen_child_splits/test.csv \
    --face-cache-dir pyannote/video_face_cache \
    --crops-dir video/talknet_child_finetuned/crops \
    --output-dir video_finetuned_talknet_runs \
    --lr-head 1e-4 \
    --lr-backbone 1e-5 \
    --phase1-epochs 5 \
    --phase2-epochs 15 \
    --seed 42

echo "=== Done: $(date) ==="
