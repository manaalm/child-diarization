#!/bin/bash
# Submit Phase A voice-converted MIL ratio sweep.
# 3 configs:
#   - voiceconv_synth_half:    +545 VC positives  (minimal volume)
#   - voiceconv_synth_full:   +1090 VC positives  (all 109 children)
#   - voiceconv_synth_hardneg: +1090 VC pos + 623 hardneg neg (compound)
#
# Each runs through the existing train_eval_spec014.sh wrapper (idempotent —
# skips if test_metrics_tuned.json already present).

set -euo pipefail

REPO=/home/manaal/orcd/scratch/child-adult-diarization
cd "$REPO"

for variant in half full hardneg; do
    cfg="mil/configs/whisper_mil_voiceconv_synth_${variant}.yaml"
    echo "Submitting $cfg"
    sbatch mil/slurm/train_eval_spec014.sh "$cfg"
done
