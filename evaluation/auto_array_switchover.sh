#!/bin/bash
# Polls every 60s for two conditions; when each lands, cancels its sequential
# job and submits the corresponding parallel array.
#   Condition E (encoder): wavlm_mean test_metrics_tuned.json appears
#     → scancel 14194134; sbatch encoder array
#   Condition M (MIL):     whisper_mil .bids_retrain_done marker appears
#     → scancel 14191408; sbatch MIL array

set -u

REPO=/orcd/scratch/orcd/008/manaal/child-adult-diarization
cd "$REPO"

WAVLM_MEAN_DONE="$REPO/baselines/baseline_results_cross_child_bids/wavlm_mean/test_metrics_tuned.json"
WHISPER_MIL_MARKER="$REPO/mil/mil_results/whisper_mil_cross_child/.bids_retrain_done"

ENCODER_SEQUENTIAL_JOB=14194134
MIL_SEQUENTIAL_JOB=14191408

did_encoder=0
did_mil=0
iter=0
while true; do
    iter=$((iter + 1))
    NOW=$(date '+%Y-%m-%d %H:%M:%S')

    if [ "$did_encoder" -eq 0 ] && [ -f "$WAVLM_MEAN_DONE" ]; then
        echo "[$NOW] [iter=$iter] wavlm_mean lands; cancelling $ENCODER_SEQUENTIAL_JOB and submitting encoder array"
        scancel "$ENCODER_SEQUENTIAL_JOB" 2>&1 || true
        sleep 5
        sbatch baselines/slurm/run_encoder_xc_bids_array.sh
        did_encoder=1
    fi

    if [ "$did_mil" -eq 0 ] && [ -f "$WHISPER_MIL_MARKER" ]; then
        echo "[$NOW] [iter=$iter] whisper_mil lands; cancelling $MIL_SEQUENTIAL_JOB and submitting MIL array"
        scancel "$MIL_SEQUENTIAL_JOB" 2>&1 || true
        sleep 5
        sbatch mil/slurm/train_eval_mil_xc_bids_array.sh
        did_mil=1
    fi

    if [ "$did_encoder" -eq 1 ] && [ "$did_mil" -eq 1 ]; then
        echo "[$NOW] both transitions complete; poller exiting"
        break
    fi

    if [ $((iter % 10)) -eq 0 ]; then
        echo "[$NOW] [iter=$iter] still waiting. encoder_done=$did_encoder  mil_done=$did_mil"
    fi
    sleep 60
done
