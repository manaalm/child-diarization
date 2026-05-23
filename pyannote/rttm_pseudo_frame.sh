#!/bin/bash
#SBATCH -c 4
#SBATCH -t 8:00:00
#SBATCH -p pi_satra,ou_bcs_normal
#SBATCH --mem=40G
#SBATCH --requeue
#SBATCH --gres=gpu:1
#SBATCH -o logs/rttm/pseudo_frame_%j.out
#SBATCH -e logs/rttm/pseudo_frame_%j.err

# Frame-level RTTM accuracy for the 3 pseudo-frame WavLM variants on
# Playlogue + Providence. For each (variant × dataset) pair we:
#   1. Run pseudo_frame/infer_long_wav_files.py to populate the RTTM cache
#      (one WavLM load per variant; 10-sec windows; val-tuned threshold).
#   2. Run unified_rttm.py --diarizer pseudo_frame_<variant> to compute
#      frame-level metrics against ground-truth RTTMs.

set -euo pipefail
source /orcd/home/002/manaal/miniforge3/etc/profile.d/conda.sh
conda activate child-vocalizations
module load ffmpeg/5.1.4
export LD_LIBRARY_PATH="/orcd/software/community/001/spack/pkg/ffmpeg/5.1.4/6kcopsg/lib:${LD_LIBRARY_PATH:-}"
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

# NOTE: must use realpath so inference (which writes cache via canonical
# os.getcwd()) and eval (which reads via --audio-dir) hash the same path.
# /home/manaal/orcd/scratch/child-adult-diarization is a symlink to
# /orcd/scratch/orcd/008/manaal/child-adult-diarization → mixing the two
# breaks the {stem}__{md5(audio_path)}.rttm cache key.
REPO=$(realpath /home/manaal/orcd/scratch/child-adult-diarization)
AUDIO_PLAY=$REPO/playlogue/audio
RTTM_PLAY=$REPO/playlogue/rttm_norm
AUDIO_PROV=$REPO/providence/audio
RTTM_PROV=$REPO/providence/rttm

cd $REPO

# Build matched (audio,rttm) lists — only audios with a corresponding RTTM.
# IMPORTANT: paths must be absolute so that infer_long_wav_files.py and
# unified_rttm.py compute the same MD5 cache id (otherwise inference writes
# to one cache key and eval looks under a different one → all evals fail).
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

declare -A VARIANT_CKPT=(
  ["pseudo_frame_baseline"]="$REPO/pseudo_frame/results/wavlm_pseudo_frame/best_checkpoint.pt"
  ["pseudo_frame_synth"]="$REPO/pseudo_frame/results/wavlm_pseudo_frame_synth/best_checkpoint.pt"
  ["pseudo_frame_c1distill"]="$REPO/pseudo_frame/results/wavlm_pseudo_frame_c1distill/best_checkpoint.pt"
)
declare -A VARIANT_CACHE=(
  ["pseudo_frame_baseline"]="$REPO/pseudo_frame/results/wavlm_pseudo_frame/rttm_cache"
  ["pseudo_frame_synth"]="$REPO/pseudo_frame/results/wavlm_pseudo_frame_synth/rttm_cache"
  ["pseudo_frame_c1distill"]="$REPO/pseudo_frame/results/wavlm_pseudo_frame_c1distill/rttm_cache"
)

for variant in pseudo_frame_baseline pseudo_frame_synth pseudo_frame_c1distill; do
  ckpt=${VARIANT_CKPT[$variant]}
  cache=${VARIANT_CACHE[$variant]}
  echo ""
  echo "=========================================================="
  echo "  $variant"
  echo "=========================================================="

  # ---- Inference: populate RTTM cache for both datasets ----
  for ds_audios in /tmp/audio_playlogue.txt /tmp/audio_providence.txt; do
    python pseudo_frame/infer_long_wav_files.py \
      --checkpoint "$ckpt" \
      --audio-list "$ds_audios" \
      --output-dir "$cache" \
      --device cuda
  done

  # ---- Eval: frame-level metrics on Playlogue + Providence ----
  cd $REPO/pyannote
  python unified_rttm.py \
    --dataset playlogue \
    --audio-dir "$AUDIO_PLAY" \
    --rttm-dir  "$RTTM_PLAY" \
    --diarizer  "$variant"

  python unified_rttm.py \
    --dataset providence \
    --audio-dir "$AUDIO_PROV" \
    --rttm-dir  "$RTTM_PROV" \
    --diarizer  "$variant"
  cd $REPO
done

echo ""
echo "All 3 pseudo-frame variants × 2 datasets complete."
