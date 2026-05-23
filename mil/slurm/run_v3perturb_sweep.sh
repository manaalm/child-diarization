#!/bin/bash
# Submit Phase B v3 (VTLP + speed perturbation) MIL sweep.
# Clones the spec-016 lanes that responded most to v1->v2:
#   - C3 Whisper-MIL hardneg  (+0.031 AUROC v1->v2 on Whisper)
#   - C4 Whisper-MIL cross-child  (+0.116 AUROC, biggest swing)
# Each runs through the existing train_eval_spec014.sh wrapper (idempotent).
#
# Prerequisite: synth/scripts/build_v3_aug_manifests.py has been run after the
# v3 corpus job finishes, so the *_v3.csv manifests + splits_synth_aug_v3/ exist.

set -euo pipefail

REPO=/home/manaal/orcd/scratch/child-adult-diarization
cd "$REPO"

REQUIRED=(
    "synth_results/manifests/synthetic_hardneg_v3.csv"
    "baselines/splits_synth_aug_v3/train.csv"
)
for p in "${REQUIRED[@]}"; do
    if [[ ! -f "$p" ]]; then
        echo "ERROR: missing $p — run synth/scripts/build_v3_aug_manifests.py first." >&2
        exit 2
    fi
done

for variant in hardneg cross_child; do
    cfg="mil/configs/whisper_mil_${variant}_synth_v3.yaml"
    echo "Submitting $cfg"
    sbatch mil/slurm/train_eval_spec014.sh "$cfg"
done
