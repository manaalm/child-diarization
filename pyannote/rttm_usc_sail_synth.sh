#!/bin/bash
#SBATCH -c 4
#SBATCH -t 6:00:00
#SBATCH -p pi_satra,ou_bcs_normal
#SBATCH --mem=40G
#SBATCH --requeue
#SBATCH --gres=gpu:1
#SBATCH -o logs/rttm/usc_sail_synth_%j.out
#SBATCH -e logs/rttm/usc_sail_synth_%j.err

# Frame-level RTTM accuracy for the USC-SAIL Whisper checkpoint trained
# only on synth scenes (spec-016 C1) on Playlogue + Providence.
#
# Same architecture as the original USC-SAIL diarizer; just a different
# training corpus. Reuses whisper-modeling/scripts/infer_long_wav_files.py
# which already writes RTTMs in the unified_rttm.py {stem}__{md5}.rttm
# convention via make_recording_id_and_rttm_name().
#
# window_size=30 is required for transformers >=4.57 (Whisper encoder
# hard-checks mel_features.length == 3000 = 30s @ 16kHz). The synth
# checkpoint was trained with window_size=30 so this matches.

set -euo pipefail
source /orcd/home/002/manaal/miniforge3/etc/profile.d/conda.sh
conda activate child-vocalizations
module load ffmpeg/5.1.4
export LD_LIBRARY_PATH="/orcd/software/community/001/spack/pkg/ffmpeg/5.1.4/6kcopsg/lib:${LD_LIBRARY_PATH:-}"
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

# NOTE: must use realpath (see comment in rttm_pseudo_frame.sh) so that
# infer_long_wav_files.py and unified_rttm.py compute identical cache MD5s.
REPO=$(realpath /home/manaal/orcd/scratch/child-adult-diarization)
AUDIO_PLAY=$REPO/playlogue/audio
RTTM_PLAY=$REPO/playlogue/rttm_norm
AUDIO_PROV=$REPO/providence/audio
RTTM_PROV=$REPO/providence/rttm

# Lightning .ckpt has state_dict wrapped under 'state_dict' key with 'model.'
# prefix on each tensor. We extract a flat state_dict via a one-time conversion:
#   python -c "import torch; ckpt=torch.load('whisper_base_synth/epoch=17-val_loss=0.243.ckpt', map_location='cpu', weights_only=False); torch.save({(k[6:] if k.startswith('model.') else k): v for k,v in ckpt['state_dict'].items()}, 'whisper_base_synth/state_dict.pt')"
CKPT=$REPO/whisper-modeling/checkpoints/whisper_base_synth/state_dict.pt
CACHE=$REPO/whisper-modeling/usc_sail_synth_rttm_cache

cd $REPO

# Build matched (audio,rttm) lists — absolute paths (see comment in
# rttm_pseudo_frame.sh; same MD5-consistency requirement).
python <<PY
import os, glob
REPO = os.path.abspath(".")
def write_list(audio_dir_rel, rttm_dir_rel, out_path, exts=(".wav", ".mp3", ".flac")):
    audio_dir = os.path.join(REPO, audio_dir_rel)
    rttm_dir = os.path.join(REPO, rttm_dir_rel)
    rttm_stems = {os.path.splitext(f)[0] for f in os.listdir(rttm_dir)}
    matched = []
    for ext in exts:
        for ap in sorted(glob.glob(os.path.join(audio_dir, f"*{ext}"))):
            stem = os.path.splitext(os.path.basename(ap))[0]
            if stem in rttm_stems:
                matched.append(ap)
    with open(out_path, "w") as f:
        f.write("\n".join(matched) + "\n")
    print(f"  {out_path}: {len(matched)} absolute paths")
write_list("playlogue/audio",  "playlogue/rttm_norm", "/tmp/audio_playlogue.txt")
write_list("providence/audio", "providence/rttm",     "/tmp/audio_providence.txt")
PY

mkdir -p "$CACHE"

# ---- Inference: populate USC-SAIL synth RTTM cache for both datasets ----
cd $REPO/whisper-modeling
PYTHONPATH=. python scripts/infer_long_wav_files.py \
  --filelist /tmp/audio_playlogue.txt \
  --out_dir  "$CACHE" \
  --model_path "$CKPT" \
  --window_size 30 \
  --stride 15 \
  --device cuda

PYTHONPATH=. python scripts/infer_long_wav_files.py \
  --filelist /tmp/audio_providence.txt \
  --out_dir  "$CACHE" \
  --model_path "$CKPT" \
  --window_size 30 \
  --stride 15 \
  --device cuda

# ---- Eval: frame-level metrics on Playlogue + Providence ----
cd $REPO/pyannote
python unified_rttm.py \
    --dataset playlogue \
    --audio-dir "$AUDIO_PLAY" \
    --rttm-dir  "$RTTM_PLAY" \
    --diarizer  usc_sail_synth

python unified_rttm.py \
    --dataset providence \
    --audio-dir "$AUDIO_PROV" \
    --rttm-dir  "$RTTM_PROV" \
    --diarizer  usc_sail_synth

echo ""
echo "USC-SAIL synth complete on both datasets."
