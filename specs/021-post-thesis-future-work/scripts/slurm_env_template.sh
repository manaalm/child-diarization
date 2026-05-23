# Shared SLURM env preamble for spec-021 jobs.
# Source this from any new spec-021 SLURM script:
#     source /orcd/scratch/orcd/008/manaal/child-adult-diarization/specs/021-post-thesis-future-work/scripts/slurm_env_template.sh
#
# Encodes the two CLAUDE.md-documented gotchas:
#   1. transformers >=4.57 has_file() network bug -> force offline.
#   2. Public-model 401 when HF_TOKEN is inherited from the env -> unset.

set -euo pipefail

export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

# Strip any inherited HF tokens (Qwen2.5-Omni and similar 401 on public ckpts).
unset HF_TOKEN HUGGINGFACE_HUB_TOKEN HF_HUB_TOKEN

# Activate the default child-vocalizations env. Override CHILD_VOC_ENV beforehand
# if a story needs a different env (e.g. video/ Python 3.10 or joint_asr_diar).
: "${CHILD_VOC_ENV:=child-vocalizations}"

# SLURM nodes don't inherit user's PATH, so source the user miniforge directly
# (matches mil/slurm/train_eval_spec014.sh and pseudo_frame/slurm/*).
: "${CONDA_BASE:=/orcd/home/002/manaal/miniforge3}"
if [ -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]; then
    # shellcheck disable=SC1091
    source "${CONDA_BASE}/etc/profile.d/conda.sh"
    conda activate "${CHILD_VOC_ENV}"
elif command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
    conda activate "${CHILD_VOC_ENV}"
fi

echo "[slurm_env_template] TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE} HF_HUB_OFFLINE=${HF_HUB_OFFLINE} env=${CHILD_VOC_ENV} python=$(which python)"
