#!/bin/bash
#SBATCH --job-name=babar_backfill_xc
#SBATCH -c 4
#SBATCH -t 8:00:00
#SBATCH -p ou_bcs_normal,pi_satra
#SBATCH --mem=32G
#SBATCH --requeue
#SBATCH --gres=gpu:1
#SBATCH -o logs/babar/backfill_xc_bids_%j.out
#SBATCH -e logs/babar/backfill_xc_bids_%j.err

# Backfill BabAR RTTMs for the 353 BIDS-recovered cross-child val+test clips
# that are not currently in babar/babar_output/rttm/ (these clips have BIDS
# child IDs that were never run through the BabAR pipeline).
#
# Workflow:
#   1. Stage missing wavs into a tmp folder (16 kHz mono — re-resample if needed)
#   2. Run BabAR pipeline on the folder (uses BabAR uv env)
#   3. Copy each RTTM into babar/babar_output/rttm/ with the md5(audio_path)
#      naming the rest of the codebase expects
#   4. Re-run cross_child_bids_role_only_eval.py for BabAR to refresh the
#      BIDS row.

set -euo pipefail

REPO=/orcd/scratch/orcd/008/manaal/child-adult-diarization
cd "$REPO"
mkdir -p logs/babar

source /home/manaal/miniforge3/etc/profile.d/conda.sh
conda activate child-vocalizations

MISSING_LIST=evaluation/babar_rttm_missing_cross_child_bids.txt
STAGE=$REPO/babar/babar_input_staging_xc_bids
OUTPUT=$REPO/babar/babar_output_xc_bids
RTTM_CACHE=$REPO/babar/babar_output/rttm
mkdir -p "$STAGE" "$OUTPUT" "$RTTM_CACHE"

echo "=== BabAR RTTM backfill (job $SLURM_JOB_ID) ==="
echo "Start: $(date)"

# Step 1: stage wavs at 16kHz mono with md5(audio_path)-suffixed filenames
echo "--- staging $(wc -l < "$MISSING_LIST") missing clips ---"
python - <<'PY'
import hashlib, os, shutil, sys
from pathlib import Path
import wave

import torch
import torchaudio

REPO = Path("/orcd/scratch/orcd/008/manaal/child-adult-diarization")
STAGE = REPO / "babar/babar_input_staging_xc_bids"
STAGE.mkdir(parents=True, exist_ok=True)
missing = [l.strip() for l in open(REPO / "evaluation/babar_rttm_missing_cross_child_bids.txt") if l.strip()]
print(f"staging {len(missing)} clips")
def cid(p): return hashlib.md5(p.encode()).hexdigest()
for i, ap in enumerate(missing):
    if not os.path.exists(ap):
        print(f"  [{i}] MISSING source: {ap}", file=sys.stderr); continue
    stem = Path(ap).stem
    target = STAGE / f"{stem}__{cid(ap)}.wav"
    if target.exists() and target.stat().st_size > 100:
        continue
    wav, sr = torchaudio.load(ap)
    if wav.shape[0] > 1: wav = wav.mean(dim=0, keepdim=True)
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)
    torchaudio.save(str(target), wav, 16000)
print(f"staged {sum(1 for _ in STAGE.iterdir() if _.suffix=='.wav')} wavs in {STAGE}")
PY

# Step 2: run BabAR pipeline on the staged folder
echo "--- running BabAR pipeline ---"
cd "$REPO/BabAR"
uv run python src/pipeline.py \
    --wavs   "$STAGE" \
    --output "$OUTPUT" \
    --device gpu

# Step 3: copy resulting RTTMs into the canonical cache
echo "--- copying RTTMs into babar/babar_output/rttm/ ---"
cp -n "$OUTPUT"/rttm/*.rttm "$RTTM_CACHE/" 2>/dev/null || true
ls -1 "$RTTM_CACHE/" | wc -l

cd "$REPO"
conda activate child-vocalizations

# Step 4: refresh BabAR BIDS role-only eval
echo "--- refreshing BabAR BIDS role-only eval ---"
python evaluation/cross_child_bids_role_only_eval.py --diarizer babar

# Clean up staging
echo "--- cleaning up staging dir ---"
rm -rf "$STAGE"

echo "Done: $(date)"
