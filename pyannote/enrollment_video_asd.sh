#!/bin/bash
#SBATCH -c 4
#SBATCH -t 24:00:00
#SBATCH -p ou_bcs_normal,pi_satra
#SBATCH --mem=40G
#SBATCH --requeue
#SBATCH --gres=gpu:1
#SBATCH -o logs/enrollment/video_asd_%j.out
#SBATCH -e logs/enrollment/video_asd_%j.err

# Run TalkNet-ASD (and optionally TS-TalkNet) enrollment evaluation.
# Syncs the video/ uv env, then runs both models via unified.py.
#
# Repos cloned: video/TalkNet-ASD/, video/TS-TalkNet/
# Checkpoints auto-download on first run:
#   - S3FD: video/TalkNet-ASD/model/faceDetector/s3fd/sfd_face.pth
#   - TalkNet: video/pretrain/talknet_asd.model
# For TS-TalkNet (optional): must have
#   - video/pretrain/ts_talknet.model
#   - video/TS-TalkNet/exps/pretrain.model

set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
export PYTHONUNBUFFERED=1
source /orcd/home/002/manaal/miniforge3/etc/profile.d/conda.sh
conda activate child-vocalizations
module load ffmpeg/5.1.4
export LD_LIBRARY_PATH="/orcd/software/community/001/spack/pkg/ffmpeg/5.1.4/6kcopsg/lib:${LD_LIBRARY_PATH:-}"

# ── Preflight: verify BIDS NFS is mounted ─────────────────────────────────
BIDS_CHECK="/orcd/scratch/bcs/001/sensein/sails/BIDS_data/final_bids-dataset"
if [[ ! -d "$BIDS_CHECK" ]]; then
    echo "ERROR: BIDS NFS not accessible at $BIDS_CHECK" >&2
    echo "This node cannot see the SAILS video files. Re-queue on a node with the mount." >&2
    exit 1
fi
echo "BIDS NFS OK: $BIDS_CHECK"

REPO=/home/manaal/orcd/scratch/child-adult-diarization

# ── Sync the isolated Python 3.10 video env ───────────────────────────────
echo "=== uv sync (video env) ==="
cd "$REPO/video"
uv sync
echo "video env ready: $(uv run python --version)"

cd "$REPO/pyannote"
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
