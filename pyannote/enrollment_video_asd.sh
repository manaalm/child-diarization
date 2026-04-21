#!/bin/bash
#SBATCH -c 4
#SBATCH -t 8:00:00
#SBATCH -p ou_bcs_normal,pi_satra
#SBATCH --mem=40G
#SBATCH --requeue
#SBATCH --gres=gpu:1
#SBATCH -o logs/enrollment/video_asd_%j.out
#SBATCH -e logs/enrollment/video_asd_%j.err

# Run TalkNet-ASD (and optionally TS-TalkNet) enrollment evaluation.
#
# Prerequisites:
#   cd video && uv sync  (done)
#   Repos cloned: video/TalkNet-ASD/, video/TS-TalkNet/  (done)
#   Checkpoints auto-download on first run:
#     - S3FD: video/TalkNet-ASD/model/faceDetector/s3fd/sfd_face.pth
#     - TalkNet: video/pretrain/talknet_asd.model
#   For TS-TalkNet (optional): must have
#     - video/pretrain/ts_talknet.model
#     - video/TS-TalkNet/exps/pretrain.model

set -euo pipefail
source /orcd/home/002/manaal/miniforge3/etc/profile.d/conda.sh
conda activate child-vocalizations
module load ffmpeg/5.1.4
export LD_LIBRARY_PATH="/orcd/software/community/001/spack/pkg/ffmpeg/5.1.4/6kcopsg/lib:${LD_LIBRARY_PATH:-}"

cd /home/manaal/orcd/scratch/child-adult-diarization/pyannote

mkdir -p logs/enrollment

# ── TalkNet-ASD enrollment ─────────────────────────────────────────────────
echo "=== TalkNet-ASD enrollment ==="
python unified.py --diarizer talknet_asd

# ── TS-TalkNet enrollment (only if checkpoints exist) ──────────────────────
TSTALKNET_CKPT="/home/manaal/orcd/scratch/child-adult-diarization/video/pretrain/ts_talknet.model"
ECAPA_CKPT="/home/manaal/orcd/scratch/child-adult-diarization/video/TS-TalkNet/exps/pretrain.model"
if [[ -f "$TSTALKNET_CKPT" && -f "$ECAPA_CKPT" ]]; then
    echo "=== TS-TalkNet enrollment ==="
    python unified.py --diarizer ts_talknet
else
    echo "Skipping TS-TalkNet: checkpoints not found (see tasks.md T076)"
fi
