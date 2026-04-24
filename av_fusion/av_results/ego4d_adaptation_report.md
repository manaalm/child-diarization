# Ego4D ASD Adaptation Experiment Report

**Feature**: 007-av-extensions  
**Date**: 2026-04-24  
**Status**: Not yet run (access pathway documented)

---

## Purpose

Measure whether ASD models trained/evaluated on Ego4D's egocentric perspective
generalize better to naturalistic child home video than models trained purely on
broadcast data (e.g., AVA-ActiveSpeaker). If yes, Ego4D pretraining is a viable
domain adaptation strategy for improving ASD-based child detection.

---

## Ego4D Access Pathway

1. Register at https://ego4d-data.org (typically 48h approval for academic use)
2. Install the data CLI: `pip install ego4d`
3. Download the AV/AVD benchmark subset (~50h annotated):
   ```bash
   ego4d --output_directory /path/to/ego4d/ --datasets full_scale --benchmarks AV
   ```
4. ASD annotations are at: `ego4d/v2/annotations/av_{train,val}.json`
5. Convert to clip-level CSV compatible with `extract_asd_features.py`:
   ```python
   # Each row: clip_id, audio_path, video_path, label (1=active speaker, 0=not)
   ```

---

## Recommended Experiment Design

| Experiment | Description | Expected Outcome |
|-----------|-------------|-----------------|
| TalkNet zero-shot (Ego4D) | Evaluate TalkNet on 50 Ego4D AVD clips | Baseline AUROC for broadcast-trained model on egocentric data |
| LocoNet zero-shot (Ego4D) | Evaluate LocoNet on 50 Ego4D AVD clips | Comparison: does CVPR 2023 model close the gap? |
| TalkNet zero-shot (child video) | Existing result from 006 pipeline | Reference AUROC on home video domain |
| LocoNet zero-shot (child video) | From 007 pipeline | Comparison baseline |

---

## Expected Result Table (to be filled after running)

| Model | Domain | Adaptation | AUROC |
|-------|--------|-----------|-------|
| TalkNet | Ego4D AVD | zero-shot | — |
| LocoNet | Ego4D AVD | zero-shot | — |
| TalkNet | Child home video | zero-shot | — |
| LocoNet | Child home video | zero-shot | — |

**delta_auroc** = (Ego4D-adapted AUROC on child video) - (base model AUROC on child video)

---

## Rationale for P3 Priority

Ego4D access requires institutional registration and download time (~50h of annotated video).
The primary pipeline (cascade + smoothing + GPT-4o + LocoNet/Light-ASD) does not depend
on Ego4D data. The Ego4D experiment is an additive comparison that quantifies domain gap
but is not blocking for any thesis results. If Ego4D access is not granted within the thesis
timeline, this report (with access pathway documented) serves as the deliverable for SC-006.

---

## Running the Experiment

```bash
# After Ego4D download and CSV conversion:
python av_fusion/scripts/ego4d_experiment.py \
    --ego4d-metadata-csv /path/to/ego4d/av_val_metadata.csv \
    --output av_fusion/av_results/ego4d_eval/ego4d_experiment_results.csv \
    --asd-model talknet \
    --n-clips 50

python av_fusion/scripts/ego4d_experiment.py \
    --ego4d-metadata-csv /path/to/ego4d/av_val_metadata.csv \
    --output av_fusion/av_results/ego4d_eval/ego4d_experiment_results_loconet.csv \
    --asd-model loconet \
    --n-clips 50
```
