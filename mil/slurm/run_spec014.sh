#!/bin/bash
# Fire-and-forget orchestrator for spec-014 MIL extensions.
#
# Submits all spec-014 jobs in parallel where possible and writes a manifest
# at mil/spec014_jobs.json that the tracker (mil/scripts/track_spec014.py)
# uses to monitor progress.
#
# Job groups submitted:
#   1. Prototype cache build (US4 prerequisite — not a strict dep, but the TS-MIL
#      jobs will fail-fast in pre-flight if the cache file is missing)
#   2. US1 (3 layer-sum) + US2 (child-adapted) + US3 (2 ACMIL) + US4 (3 TS-MIL)
#      = 9 frame-window MIL training+eval jobs
#   3. US5 + US6: one segment-MIL sweep job (4 new aggregators × 4 frontends = 16
#      new cells, resume-safe so already-done cells are skipped)
#
# Usage:  bash mil/slurm/run_spec014.sh
# (run from repo root or anywhere; the script cd's to repo root)

set -euo pipefail

REPO=/home/manaal/orcd/scratch/child-adult-diarization
cd "$REPO"
mkdir -p logs/mil mil/prototypes

MANIFEST=mil/spec014_jobs.json
TIMESTAMP=$(date +%Y-%m-%dT%H:%M:%S)

echo "[$TIMESTAMP] Submitting spec-014 jobs..."

# Helper: submit a job and return its job ID.
sb() {
    local out
    out=$(sbatch "$@")
    # sbatch prints "Submitted batch job NNNNNN"
    echo "$out" | awk '{print $NF}'
}

# Helper: append (variant, jobid) to a temporary CSV; the manifest is built at the end.
TMP_CSV=$(mktemp)
record() {
    echo "$1,$2,$3,$4" >> "$TMP_CSV"
}

# ------------------------------------------------------------------
# 1) Prototype cache build (US4)
# ------------------------------------------------------------------
if [[ ! -f mil/prototypes/babar_vtc.npz ]]; then
    JID=$(sb mil/slurm/build_prototype_cache.sh babar_vtc \
        whisper-modeling/seen_child_splits/train.csv \
        mil/prototypes/babar_vtc.npz)
    echo "  proto_cache_seen_child   job=$JID"
    record "proto_cache_seen_child" "$JID" "US4" "mil/prototypes/babar_vtc.npz"
else
    echo "  proto_cache_seen_child   already exists, skipping"
fi

# Cross-child prototype cache: only build if cross-child train CSV exists.
if [[ -f baselines/splits/train.csv && ! -f mil/prototypes/babar_vtc_cross_child.npz ]]; then
    JID=$(sb mil/slurm/build_prototype_cache.sh babar_vtc \
        baselines/splits/train.csv \
        mil/prototypes/babar_vtc_cross_child.npz)
    echo "  proto_cache_cross_child  job=$JID"
    record "proto_cache_cross_child" "$JID" "US4" "mil/prototypes/babar_vtc_cross_child.npz"
fi

# ------------------------------------------------------------------
# 2) Frame-window MIL train+eval jobs — 9 configs total
# ------------------------------------------------------------------

# US1: weighted-layer-sum (3 backbones)
for cfg in wavlm_mil_layersum whisper_mil_layersum hubert_large_mil_layersum; do
    JID=$(sb mil/slurm/train_eval_spec014.sh "mil/configs/${cfg}.yaml")
    echo "  ${cfg}  job=$JID"
    record "$cfg" "$JID" "US1" "mil/mil_results/${cfg}/test_metrics_tuned.json"
done

# US2: child-adapted WavLM
JID=$(sb mil/slurm/train_eval_spec014.sh mil/configs/wavlm_mil_child_adapted.yaml)
echo "  wavlm_mil_child_adapted  job=$JID"
record "wavlm_mil_child_adapted" "$JID" "US2" "mil/mil_results/wavlm_mil_child_adapted/test_metrics_tuned.json"

# US3: ACMIL (2 backbones)
for cfg in wavlm_mil_acmil whisper_mil_acmil; do
    JID=$(sb mil/slurm/train_eval_spec014.sh "mil/configs/${cfg}.yaml")
    echo "  ${cfg}  job=$JID"
    record "$cfg" "$JID" "US3" "mil/mil_results/${cfg}/test_metrics_tuned.json"
done

# US4: TS-MIL (3 configs — concat, FiLM, Whisper-concat; cross-child if cache exists)
for cfg in wavlm_mil_tsmil_concat wavlm_mil_tsmil_film whisper_mil_tsmil_concat; do
    JID=$(sb mil/slurm/train_eval_spec014.sh "mil/configs/${cfg}.yaml")
    echo "  ${cfg}  job=$JID"
    record "$cfg" "$JID" "US4" "mil/mil_results/${cfg}/test_metrics_tuned.json"
done

if [[ -f mil/configs/wavlm_mil_tsmil_concat_cross_child.yaml && -f baselines/splits/train.csv ]]; then
    JID=$(sb mil/slurm/train_eval_spec014.sh \
        mil/configs/wavlm_mil_tsmil_concat_cross_child.yaml)
    echo "  wavlm_mil_tsmil_concat_cross_child  job=$JID"
    record "wavlm_mil_tsmil_concat_cross_child" "$JID" "US4" \
        "mil/mil_results/wavlm_mil_tsmil_concat_cross_child/test_metrics_tuned.json"
fi

# ------------------------------------------------------------------
# 3) Segment-MIL sweep (US5 DSMIL + US6 AutoPool/ExpSoftmax/GMAP)
# ------------------------------------------------------------------
JID=$(sb mil/slurm/seg_mil_sweep.sh)
echo "  seg_mil_sweep  job=$JID"
record "seg_mil_sweep" "$JID" "US5+US6" "mil/mil_results/seg_mil/all_configs.json"

# ------------------------------------------------------------------
# Build the manifest JSON
# ------------------------------------------------------------------
python - <<PY
import csv, json, os
rows = []
with open("$TMP_CSV") as f:
    for variant, jobid, story, expected in csv.reader(f):
        rows.append({
            "variant": variant,
            "job_id": jobid,
            "story": story,
            "expected_output": expected,
            "submitted_at": "$TIMESTAMP",
        })
with open("$MANIFEST", "w") as f:
    json.dump({"submitted_at": "$TIMESTAMP", "jobs": rows}, f, indent=2)
print(f"\n[manifest] wrote {len(rows)} jobs → $MANIFEST")
PY

rm -f "$TMP_CSV"

echo ""
echo "Done. Track progress with:"
echo "  python mil/scripts/track_spec014.py"
