#!/bin/bash
#SBATCH --job-name=cross_diar
#SBATCH --gres=gpu:1
#SBATCH -t 4:00:00
#SBATCH --mem=40G
#SBATCH -c 4
#SBATCH -p ou_bcs_normal,pi_satra
#SBATCH -o logs/evaluation/cross_child_diarizer_%j.out
#SBATCH -e logs/evaluation/cross_child_diarizer_%j.out

# Role-only BabAR / VTC evaluation on the cross-child split.
# 857/908 val+test clips are already cached from the seen-child runs.
# --run-vtc-for-missing handles the ~51 uncached clips via live VTC inference.
# Results: evaluation/cross_child_{babar,vtc,vtc_kchi}_role_only/

set -euo pipefail

echo "Start: $(date)"
echo "Node: $(hostname)  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo none)"

export HF_TOKEN=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
source /home/manaal/miniforge3/etc/profile.d/conda.sh
conda activate child-vocalizations
module load ffmpeg/5.1.4
export LD_LIBRARY_PATH="/orcd/software/community/001/spack/pkg/ffmpeg/5.1.4/6kcopsg/lib:${LD_LIBRARY_PATH:-}"

REPO=/orcd/scratch/orcd/008/manaal/child-adult-diarization
cd "$REPO"
mkdir -p logs/evaluation

echo "--- BabAR cross-child role-only ---"
python evaluation/cross_child_diarizer_eval.py \
    --diarizer babar \
    --splits-dir baselines/splits

echo "--- VTC cross-child role-only ---"
python evaluation/cross_child_diarizer_eval.py \
    --diarizer vtc \
    --splits-dir baselines/splits \
    --run-vtc-for-missing

echo "--- VTC-KCHI cross-child role-only ---"
python evaluation/cross_child_diarizer_eval.py \
    --diarizer vtc_kchi \
    --splits-dir baselines/splits

echo "Done: $(date)"
