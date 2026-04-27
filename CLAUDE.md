# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Child-adult speaker diarization system that classifies speakers (silence, child, adult, overlap) in audio recordings at 20ms frame-level resolution. Based on ICASSP 2025 / Interspeech 2024 papers by Xu et al.

The goal is per-clip child presence detection: given a short audio clip, predict whether a target child is vocalizing. A synthetic scene generator (`synth/`) produces augmented training data by mixing Providence child speech and LibriSpeech adult speech under configurable SNR, RIR, overlap, and scene-type distributions. Nine diarization frontends are compared:
1. **USC-SAIL** — Fine-tuned Whisper + LoRA frame classifier (`whisper-modeling/`)
2. **Pyannote** — `pyannote/speaker-diarization-community-1` model
3. **BabAR** — VTC 2.0 child speech diarizer (full pipeline with phoneme step)
4. **VTC** — VTC 2.0 standalone (no BabAR phoneme step); two variants: `vtc` (KCHI+OCH) and `vtc_kchi` (KCHI only)
5. **VBx** — Variational Bayes HMM speaker diarization using pyannote VAD + ECAPA embeddings; anonymous speaker labels resolved via cosine similarity to target-child prototype
6. **TalkNet-ASD** — Video-audio active speaker detection (SAILS BIDS .mp4 only); child identified as smallest face track
7. **TS-TalkNet** — Speaker-conditioned video-audio ASD; uses a reference clip from the training split for target-child enrollment
8. **EEND-EDA** — End-to-End Neural Diarization with Encoder-Decoder Attractors (ESPnet2); handles overlapping speech natively; anonymous speaker labels resolved via ECAPA cosine similarity
9. **Sortformer** — Sort-based transformer diarization (NeMo/NVIDIA); anonymous speaker labels resolved via ECAPA cosine similarity
10. **Audio LLM Baseline** — Qwen2-Audio-7B-Instruct zero-shot child vocalization detection (`baselines/audio_llm_baseline.py`); prompted "Is there a child vocalizing?" → yes/no logit ratio → threshold-tuned on val

All are evaluated using a shared ECAPA-based speaker enrollment pipeline. The primary evaluation and combined-feature scripts live in the **`pyannote/` folder** (despite the name, it is the multi-diarizer testing suite for the project).

---

## Environment Setup

Each subsystem has its own Python environment — do not mix them:

```bash
# Main USC-SAIL / Whisper model
cd whisper-modeling && pip install -r requirements.txt

# BabAR requires a separate venv (see BabAR/README.md)
# Pyannote has its own requirements; install pyannote.audio separately

# Video ASD (TalkNet-ASD, TS-TalkNet, LocoNet, Light-ASD) — Python 3.10 isolated env
cd video && uv sync
# Clone model repos (gitignore'd — not committed):
git clone https://github.com/TaoRuijie/TalkNet-ASD video/TalkNet-ASD
git clone https://github.com/Jiang-Yidi/TS-TalkNet video/TS-TalkNet
# LocoNet (007-av-extensions, CVPR 2023):
huggingface-cli download Superxixixi/LoCoNet_ASD --local-dir video/LoCoNet_ASD/
# Light-ASD (007-av-extensions, lightweight ASD):
git clone https://github.com/Junhua-Liao/Light-ASD video/Light-ASD
# Download checkpoints to video/pretrain/ per video/SETUP.md:
#   sfd_face.pth (~87 MB), talknet_asd.model, ts_talknet.model

# GPT-4o feature extraction (007-av-extensions):
conda activate child-vocalizations
pip install openai  # if not already installed
# Set OPENAI_API_KEY before running extract_gpt4o_features.py

# EEND-EDA (ESPnet2) — install into child-vocalizations or a separate env
conda activate child-vocalizations
pip install espnet espnet_model_zoo soundfile
# Find a pre-trained EEND-EDA model:
#   python -c "from espnet_model_zoo.downloader import ModelDownloader; \
#              d=ModelDownloader(); [print(r['name']) for r in d.query('diar')]"
# Default model: espnet/horiguchi_INTERSPEECH2022_EEND-EDA-online_6spk (downloads on first run)

# Sortformer (NeMo) — install into child-vocalizations or a separate env
conda activate child-vocalizations
pip install nemo_toolkit[asr]
# Model (diar_sortformer_4spk-v1) downloads from NVIDIA NGC on first run.
```

All audio is assumed to be 16kHz mono. Auto-resampling is handled in `dataset_classes/preprocess.py`.

---

## Key Commands

### Training the USC-SAIL Whisper model

```bash
cd whisper-modeling
# Remove pdb.set_trace() on line ~41 of scripts/main.py before running on a cluster
python scripts/main.py --debug f --config configs/config.yaml
# Outputs: logs/, checkpoints/ (best by val_loss)
```

### Single-file inference

```bash
cd whisper-modeling
python scripts/infer_wav_file.py --wav_file /path/to/audio.wav
# Outputs: child/adult/overlap segment timestamp lists
```

### Batch inference (SLURM)

```bash
sbatch --array=0-155%5 run_usc_playlogue.sh
# Calls infer_long_wav_files.py per wav; writes RTTM files to playlogue/rttm/
# Log files go to logs/adult/*.out — highest-numbered .out file = most recent run
```

### Unified enrollment evaluation (all diarizers)

```bash
# From pyannote/ — the multi-diarizer testing suite
cd pyannote
python unified.py --diarizer usc_sail   # or pyannote / babar / vtc / vtc_kchi / vbx
# Output: {diarizer}_ecapa_enrollment_runs/ with role_only and enrollment metrics
# VBx and VTC require HF_TOKEN and VBx/VTC uv envs set up first (see Gotchas)

# Video ASD frontends (SAILS BIDS data only — requires .mp4 files and video/ env):
python unified.py --diarizer talknet_asd   # → video_asd_ecapa_enrollment_runs/talknet_asd/
python unified.py --diarizer ts_talknet    # → video_asd_ecapa_enrollment_runs/ts_talknet/

# Neural diarization frontends (EEND-EDA + Sortformer):
python unified.py --diarizer eend_eda      # → eend_eda_ecapa_enrollment_runs/
python unified.py --diarizer sortformer    # → sortformer_ecapa_enrollment_runs/
# Or submit SLURM jobs:
sbatch run_eend_eda_enrollment.sh
sbatch run_sortformer_enrollment.sh
```

### Unified RTTM accuracy evaluation

```bash
cd pyannote
python unified_rttm.py --diarizer usc_sail   # or pyannote / babar / vtc / vtc_kchi / vbx
# Frame-level child detection accuracy; converts RTTMs to 10ms masks
```

### BabAR combined-feature models

```bash
cd pyannote
python babar_three.py   # or babar_updated.py
# Trains 8 feature-set combinations × 2 classifiers (LR + GBM)
# Output: /babar_combined_runs/all_model_results.json + per-model prediction CSVs
```

### Error analysis

```bash
cd pyannote
python error_analysis.py       # BabAR combined models
python pyannote_error_analysis.py  # Pyannote enrollment
# Output: per_child_error_rates.csv, false_positives/negatives.csv, thesis_summary.json
```

### AV fusion pipeline (manual-only MVP, no GPU required)

```bash
# Step 1: Build master feature table (manual BIDS annotations, no video extraction needed)
python av_fusion/scripts/build_av_feature_table.py \
  --metadata-csv whisper-modeling/seen_child_splits/master_with_split.csv \
  --audio-scores-val  babar_ecapa_enrollment_runs/enroll_val_predictions.csv \
  --audio-scores-test babar_ecapa_enrollment_runs/enroll_test_predictions.csv \
  --audio-score-col prob \
  --output-dir av_fusion/av_results/manual_only/ \
  --run-name manual_only
# Output: av_fusion/av_results/manual_only/{av_master_features.csv, av_{train,val,test}.csv,
#          feature_manifest.json, split_integrity_report.json}

# Step 2: Train fusion models
python av_fusion/scripts/train_av_fusion.py \
  --feature-dir av_fusion/av_results/manual_only/ \
  --output-dir  av_fusion/av_results/manual_only/models/ \
  --config      av_fusion/configs/av_fusion.yaml \
  --seed 42
# Output: models/{audio_only,video_only,always_fuse_av,gated_av}.pkl +
#         val_metrics.json, visual_eligibility_threshold.json

# Step 3: Evaluate on held-out test
python av_fusion/scripts/evaluate_av_fusion.py \
  --feature-dir av_fusion/av_results/manual_only/ \
  --model-dir   av_fusion/av_results/manual_only/models/ \
  --output-dir  av_fusion/av_results/manual_only/ \
  --plot
# Output: metrics_overall.json, predictions_test.csv, metrics_by_*.csv,
#         figures/{pr_curve,roc_curve,stratified_bar_metrics,visual_eligibility_histogram}.png

# Step 4: Error analysis
python av_fusion/scripts/error_analysis_av.py \
  --predictions-csv av_fusion/av_results/manual_only/predictions_test.csv \
  --feature-dir     av_fusion/av_results/manual_only/ \
  --output-dir      av_fusion/av_results/manual_only/

# Optional: Extract automatic visual features (requires video files; 48h GPU job)
sbatch av_fusion/slurm/run_av_pipeline.sh
# Then re-run steps 1–4 with --visual-features-csv av_fusion/av_results/auto/visual_features.csv

# Optional: Extract ASD features (requires video/ env + TalkNet checkpoint)
python av_fusion/scripts/extract_asd_features.py \
  --metadata-csv  whisper-modeling/seen_child_splits/master_with_split.csv \
  --output        av_fusion/av_results/manual_only/asd_features.csv
```

### 007-av-extensions: Cascaded pipeline, smoothing, GPT-4o, LocoNet/Light-ASD

```bash
# Step 5: Tune cascade thresholds (val set only)
python av_fusion/scripts/train_cascaded_pipeline.py \
  --feature-dir av_fusion/av_results/manual_only/ \
  --output-dir  av_fusion/av_results/manual_only/models/
# Outputs: models/cascade_thresholds.json, cascade_val_stage_breakdown.csv

# Step 6: Evaluate cascade on test set (extends evaluate_av_fusion.py)
python av_fusion/scripts/evaluate_av_fusion.py \
  --feature-dir av_fusion/av_results/manual_only/ \
  --model-dir   av_fusion/av_results/manual_only/models/ \
  --output-dir  av_fusion/av_results/manual_only/ \
  --cascade-breakdown av_fusion/av_results/manual_only/cascade_stage_breakdown.csv
# Added outputs: cascade_stage_breakdown.csv, metrics_cascade_by_stage.csv

# Step 7: Temporal smoothing (auto-tunes bandwidth on val)
python av_fusion/scripts/smooth_predictions.py \
  --predictions     av_fusion/av_results/manual_only/predictions_test.csv \
  --val-predictions av_fusion/av_results/manual_only/predictions_val.csv \
  --output          av_fusion/av_results/manual_only/predictions_test_smoothed.csv \
  --method gaussian

# GPT-4o feature extraction (requires OPENAI_API_KEY; ~$0.66 for all 2183 clips)
export OPENAI_API_KEY=<key>
python av_fusion/scripts/extract_gpt4o_features.py \
  --metadata-csv whisper-modeling/seen_child_splits/master_with_split.csv \
  --output av_fusion/av_results/manual_only/gpt4o_features.csv \
  --dry-run   # print cost estimate first
python av_fusion/scripts/extract_gpt4o_features.py \
  --metadata-csv whisper-modeling/seen_child_splits/master_with_split.csv \
  --output av_fusion/av_results/manual_only/gpt4o_features.csv

# LocoNet ASD features (requires video/LoCoNet_ASD/ cloned + checkpoint downloaded)
python av_fusion/scripts/extract_asd_features.py \
  --metadata-csv whisper-modeling/seen_child_splits/master_with_split.csv \
  --output av_fusion/av_results/manual_only/asd_features_loconet.csv \
  --model loconet \
  --loconet-checkpoint video/LoCoNet_ASD/<checkpoint>.ckpt

# Light-ASD features (requires video/Light-ASD/ cloned)
python av_fusion/scripts/extract_asd_features.py \
  --metadata-csv whisper-modeling/seen_child_splits/master_with_split.csv \
  --output av_fusion/av_results/manual_only/asd_features_light_asd.csv \
  --model light_asd \
  --light-asd-checkpoint video/Light-ASD/weight/pretrain_AVA_CVPR22.pt

# 1kd dataset compatibility check (safe to run even without data)
python av_fusion/scripts/1kd_integration.py \
  --data-dir /path/to/1kd/ \
  --output av_fusion/av_results/manual_only/1kd_integration_report.json
```

### Audio LLM Zero-Shot Baseline (`baselines/audio_llm_baseline.py`)

```bash
# Dry run — print 3 example prompts and exit 0
python baselines/audio_llm_baseline.py --split val --max-clips 5 --dry-run

# Step 1: val-set inference + threshold tuning (submit via SLURM — requires GPU, ~4h)
sbatch baselines/slurm/run_audio_llm_baseline.sh val
# Output: baselines/audio_llm_baseline_runs/qwen2_audio_7b/val_predictions.csv
#         baselines/audio_llm_baseline_runs/qwen2_audio_7b/val_metrics_tuned.json

# Step 2: test-set inference (run after Step 1 completes; loads threshold from val JSON)
sbatch baselines/slurm/run_audio_llm_baseline.sh test
# Output: test_predictions.csv, test_metrics_tuned.json, test_metrics_by_timepoint.csv, config.json

# Optional: 2-shot few-shot variant (same-child training clips as in-context examples)
sbatch baselines/slurm/run_audio_llm_baseline.sh val qwen2_audio_7b_2shot 2
sbatch baselines/slurm/run_audio_llm_baseline.sh test qwen2_audio_7b_2shot 2

# Smoke test (10 clips, no model required flag)
python baselines/audio_llm_baseline.py --split val --max-clips 10 \
  --output-dir /tmp/audio_llm_smoke --cache-dir /tmp/audio_llm_cache_smoke --seed 42
```

### Synthetic Data Generator (`synth/`)

7-step pipeline: build segment manifest → extract segments → generate scenes → make training manifests → train at each ratio → evaluate → error analysis. No GPU required for steps 1–3.

```bash
# Step 1: build Providence + TinyVox + LibriSpeech segment manifest
python synth/scripts/build_segment_manifest.py \
  --providence-dir        providence/ \
  --providence-rttm-dir   providence/rttm/ \
  --tinyvox-dir           data/tinyvox/ \
  --librispeech-dir       /path/to/LibriSpeech/train-clean-100/ \
  --exclude-speakers-csv  whisper-modeling/seen_child_splits/test.csv \
  --output                synth_results/manifests/segment_manifest.csv
# TinyVox adds ~24k Eng-NA child segments (~10 h) with age_band from session YYMMDD

# Step 2: extract segment WAVs
python synth/scripts/extract_segments.py \
  --manifest   synth_results/manifests/segment_manifest.csv \
  --output-dir data/segments/

# Step 3: generate 5000 acoustic scenes with RIR + noise (via SLURM — no GPU)
# RIR and MUSAN paths are baked into synth/configs/default_14_18mo.yaml (spec-009):
#   RIR_DIR  = data/rir/simulated_rirs_16k/   (OpenSLR 26, 60k WAVs)
#   NOISE_DIR = data/noise/musan/noise/        (MUSAN noise subset, SLURM job 12646682)
# Can also override at runtime:
sbatch synth/slurm/run_scene_generation.sh synth/configs/default_14_18mo.yaml
# Or with explicit overrides (if config paths change):
# sbatch synth/slurm/run_scene_generation.sh synth/configs/default_14_18mo.yaml \
#   --rir-dir data/rir/simulated_rirs_16k \
#   --noise-dir data/noise/musan/noise
# Output: synth_results/synthetic_scenes/{wav,rttm,json}/
#         synth_results/manifests/synthetic_manifest.csv

# Step 4: build train manifests at 6 ratios (0×, 0.5×, 1×, 2×, 5×, 10×)
python synth/scripts/generate_training_sets.py \
  --real-train-csv     whisper-modeling/seen_child_splits/train.csv \
  --synthetic-manifest synth_results/manifests/synthetic_manifest.csv \
  --output-dir         synth_results/manifests/

# Steps 5–6: train + evaluate (GPU sweep, 48 h SLURM)
sbatch synth/slurm/run_ratio_sweep.sh synth/configs/default_14_18mo.yaml
# Output: synth_results/augmentation_experiments/default_14_18mo/
#         metrics_by_ratio.csv, metrics_by_age_band.csv, figures/

# Step 7: error analysis (real-only vs. best ratio)
python synth/scripts/error_analysis_synthetic.py \
  --experiment-dir synth_results/augmentation_experiments/default_14_18mo/ \
  --test-csv       whisper-modeling/seen_child_splits/test.csv \
  --output-dir     synth_results/augmentation_experiments/default_14_18mo/

# Optional: distribution quality figures
python synth/scripts/analyze_synthetic_quality.py \
  --synthetic-manifest synth_results/manifests/synthetic_manifest.csv \
  --real-train-csv     whisper-modeling/seen_child_splits/train.csv \
  --output-dir         synth_results/augmentation_experiments/default_14_18mo/figures/
```

### Child-adapted WavLM pretraining (spec-009 US3)

Continued masked-speech-unit pretraining of WavLM-Base+ on 73k Providence/TinyVox child
speech segments (~101 h). Outputs a checkpoint compatible with the frame-window MIL backbone.

```bash
# Step 1: build child WAV list (98k files: TinyVox Eng-NA + Providence segments)
find data/tinyvox/audio -name "phon_Eng-NA_*.wav" > synth_results/child_wavs.txt
find data/segments/child -name "*.wav" >> synth_results/child_wavs.txt

# Step 2: submit pretraining (48h GPU, resumes automatically if checkpoint exists)
sbatch synth/slurm/run_wavlm_pretrain.sh
# Output: synth_results/child_wavlm_checkpoint/step_{N}/ (saved every 5000 steps)
#         synth_results/child_wavlm_checkpoint/training_log.csv
# Logs:   logs/synth/wavlm_pretrain_{SLURM_JOB_ID}.out

# Step 3: wire child-adapted backbone into MIL (edit backbone_path in config)
cp mil/configs/wavlm_mil.yaml mil/configs/wavlm_mil_child_adapted.yaml
# Edit wavlm_mil_child_adapted.yaml: set backbone_path to synth_results/child_wavlm_checkpoint/step_50000

# Step 4: train and evaluate child-adapted MIL (same pipeline as baseline)
sbatch mil/slurm/train_mil.sh mil/configs/wavlm_mil_child_adapted.yaml
sbatch mil/slurm/eval_mil.sh
```

### Frame-window MIL (WavLM-Base+ / Whisper-small)

```bash
# Train one variant (wavlm_mil or whisper_mil):
sbatch mil/slurm/train_mil.sh mil/configs/wavlm_mil.yaml
sbatch mil/slurm/train_mil.sh mil/configs/whisper_mil.yaml
# Output: mil/mil_results/{variant}/best_checkpoint.pt + val_metrics_tuned.json
# Logs: logs/mil/train_{jobid}.out

# Evaluate both checkpoints on the test split:
sbatch mil/slurm/eval_mil.sh
# Output per variant: test_metrics_tuned.json, test_predictions.csv,
#                     test_metrics_by_timepoint.csv, val_metrics_by_timepoint.csv
# Logs: logs/mil/eval_{jobid}.out

# Age-stratified evaluation (after eval_mil.sh completes):
python mil/mil_age_stratified.py \
  --checkpoint mil/mil_results/wavlm_mil/best_checkpoint.pt \
  --config     mil/mil_results/wavlm_mil/config.json \
  --age-group  14_month \
  --manifest   playlogue/manifest.csv
python mil/mil_age_stratified.py \
  --checkpoint mil/mil_results/wavlm_mil/best_checkpoint.pt \
  --config     mil/mil_results/wavlm_mil/config.json \
  --age-group  36_month \
  --manifest   playlogue/manifest.csv
# (Repeat replacing wavlm_mil with whisper_mil)
# Output: mil/mil_results/{variant}/age_stratified/{age_group}/test_metrics_tuned.json
```

### Segment-instance MIL sweep

```bash
# Pre-compute embeddings only (run once, ~1-2 hrs on GPU):
python mil/seg_train.py --config mil/configs/seg_mil_sweep.yaml --precompute-only

# Full sweep (precompute + train all 4 frontends × 7 aggregators = 28 configs, resume-safe):
sbatch mil/slurm/seg_mil_sweep.sh
# Output: mil/mil_results/seg_mil/{frontend}_{aggregator}/ + all_configs.json
# Logs: logs/mil/seg_mil_{jobid}.out

# Weak diarization evaluation (after attention-variant configs complete):
python mil/eval_weak_diarization.py \
  --results-dir mil/mil_results/seg_mil \
  --split-csv whisper-modeling/seen_child_splits/test.csv \
  --rttm-cache whisper-modeling/usc_sail_rttm_cache \
  --output mil/mil_results/seg_mil/weak_diarization_eval.csv
```

---

## Architecture

### `pyannote/` — Multi-diarizer testing suite (primary evaluation hub)

This folder, despite its name, contains the shared evaluation infrastructure for all three diarizers:

**`unified.py`** — Abstract `DiarizationFrontend` base class with seven backends:
- `USCSailFrontend`: runs USC-SAIL Whisper model, extracts CHI segments
- `PyannoteFrontend`: runs pyannote model, maps anonymous SPEAKER_XX → CHI/ADT by GT overlap
- `BabARFrontend`: runs BabAR VTC 2.0 (full pipeline with phoneme step), extracts KCHI segments
- `VTCFrontend`: runs VTC 2.0 standalone (no BabAR phoneme step); `vtc` = KCHI+OCH, `vtc_kchi` = KCHI only
- `VBxFrontend`: runs VBx Variational Bayes HMM speaker diarization (pyannote VAD + ECAPA x-vectors); anonymous speaker clusters scored by cosine similarity to target-child prototype
- `TalkNetASDFrontend` (in `video_asd.py`): calls `video/run_asd.py --model talknet_asd` via subprocess; identifies child as smallest face track; returns [] for audio-only datasets
- `TSTalkNetFrontend` (in `video_asd.py`): calls `video/run_asd.py --model ts_talknet` with `--ref_audio` from the same child's training split; returns [] if no reference clip found or video missing

**`video_asd.py`** — `TalkNetASDFrontend` and `TSTalkNetFrontend`; subprocess bridge to the isolated Python 3.10 `video/` env; RTTM cached under `video_asd_rttm_cache/{model}/`; face tracks cached under `video_face_cache/`.

Shared enrollment pipeline: builds ECAPA duration-weighted child prototypes from training data → cosine similarity scoring → threshold tuning on val → test evaluation. Outputs role-only (duration baseline) and enrollment (embedding) predictions and metrics.

**`unified_rttm.py`** — Frame-level accuracy script (mirrors `unified.py` but evaluates diarization accuracy directly on Playlogue/Providence rather than enrollment classification). Converts RTTM segments → 10ms binary frame masks for evaluation.

**`babar_three.py` / `babar_updated.py`** — Combined feature models using three feature groups:
1. Diarizer features: KCHI segment duration, n_segments, proportion
2. Phoneme features from BabAR: n_utterances, consonant/vowel counts, CV ratio, unique phoneme ratio
3. Embedding features: cosine similarity to ECAPA prototype (mean/max/top-3)

Trains 8 feature-set combinations × LR + GBM = 16 models, plus per-timepoint variants. Requires BabAR RTTM + phoneme CSVs and ECAPA prototypes from a prior enrollment run.

**`unified_age_stratified.py`** — Wraps the `unified.py` enrollment pipeline with per-age-group filtering; reads the `timepoint_norm` column from the seen-child split to filter to `14_month` or `36_month` cohorts; writes per-cohort `test_metrics_tuned.json` and `test_predictions.csv` to `pyannote/{diarizer}_age_stratified/{age_group}/`. CLI: `python pyannote/unified_age_stratified.py --diarizer babar --age-group 14_month`.

**`augmentation_eval.py`** — Retrain enrollment prototypes on a training split augmented with synthetic child speech (reads `registry.jsonl` from `--synthetic-dir`); evaluates on the same val/test splits as the baseline; produces F1/AUROC/AUPRC delta table vs. the unaugmented baseline. CLI: `python pyannote/augmentation_eval.py --diarizer babar --synthetic-dir synth_results/synthetic_scenes/ --output-dir pyannote/babar_augmented/`.

**`proxy_analysis.py`** — Quality proxy metrics on unlabeled core dataset recordings; runs BabAR and USC-SAIL diarizers to estimate child speech duration, segment rate, and SNR proxy; writes per-session CSV for exploratory analysis. CLI: `python pyannote/proxy_analysis.py --core-dir core/audio/ --output-dir pyannote/proxy_results/`.

**`scripts/prepare_age_manifests.py`** — Loads per-dataset annotation sources (Playlogue: `anotated_processed.csv`, Providence: CHAT metadata, Seedlings: Databrary export); assigns `age_group` labels (12_16m / 34_38m / other); outputs `manifest.csv` per dataset matching the `AudioRecording` schema. CLI: `python scripts/prepare_age_manifests.py --dataset {playlogue|providence|seedlings}`.

**`scripts/verify_reproducibility.py`** — Compares committed `config.json` against result files across all result folders; reports hash mismatches. CLI: `python scripts/verify_reproducibility.py`; outputs to `evaluation/reproducibility_report.txt`.

**Note**: `unified.py` is partially redundant with `whisper-modeling/usc_sail_run_enrollment.py` — USC-SAIL enrollment logic exists in both places.

### `mil/` — Multiple Instance Learning module

**`mil_model.py`** — `BackboneExtractor` (frozen WavLM-base+ or Whisper-small) + `GatedABMILHead` (gated attention MIL, Ilse et al. 2018) + `MILModel` composer. Used by the frame-window MIL workflow.

**`mil_train.py`** / **`mil_dataset.py`** — Frame-window MIL: splits audio into 2s windows, embeds each window, trains GatedABMIL head over the bag of windows.

**`mil_evaluate.py`** — Loads a trained checkpoint + val-tuned threshold; runs forward pass over the test split; writes `test_metrics_tuned.json`, `test_predictions.csv`, `test_metrics_by_timepoint.csv`. CLI: `python mil/mil_evaluate.py --checkpoint <pt> --config <json>`.

**`mil_age_stratified.py`** — Age-cohort evaluation: inner-joins test split with a dataset manifest on `audio_path`, filters to a single `age_group` (14_month or 36_month), runs the checkpoint and writes cohort-specific metrics to `mil/mil_results/{variant}/age_stratified/{age_group}/`. CLI: `python mil/mil_age_stratified.py --checkpoint <pt> --config <json> --age-group <group> --manifest <csv>`.

**`mil_utils.py`** — Shared metric helpers: `compute_metrics()`, `tune_threshold()`, `per_timepoint_metrics()`, `save_json()`, `save_csv()`.

**Segment-instance MIL** (`seg_*.py`) — Uses diarizer-proposed speech segments as bag instances instead of fixed windows:
- `seg_embedding_cache.py` — Disk cache keyed on MD5("{audio_path}|{start:.4f}|{end:.4f}"), one `.npy` per segment embedding; shared across all 4 aggregators per frontend.
- `seg_dataset.py` — `SegmentBagDataset`: loads RTTM → segments → WavLM embeddings → (K_max×D, mask, label, meta). `precompute_embeddings()` pre-fills cache.
- `seg_model.py` — Seven aggregators over variable-length bags: `MeanAgg`, `MaxAgg`, `AttnAgg` (ABMIL), `GatedAttnAgg` (wraps `GatedABMILHead`), `NoisyORAgg` (log-space noisy-OR), `TopKAgg` (top-k by score, mean-pool), `TransformerAgg` (CLS token + learned PE + 2-layer pre-norm transformer encoder).
- `seg_train.py` — Sweep training loop (4 frontends × up to 7 aggregators = up to 28 configs); resume-safe; writes per-config results + `all_configs.json` summary; produces age-band metrics (`test_metrics_by_timepoint.csv`) and segment attention weight CSVs for attention/gated_attention/transformer configs.
- `eval_weak_diarization.py` — Standalone script that reads saved attention weight CSVs and RTTM ground-truth files to evaluate how well MIL attention weights correlate with actual child speech (Pearson, Spearman, AUROC). Outputs `weak_diarization_eval.csv` stratified by age band.
- `configs/seg_mil_sweep.yaml` — Sweep config: frontends, aggregators (7 total), RTTM paths, training HPs, seed=42, transformer_config block.
- `slurm/seg_mil_sweep.sh` — SLURM submission script (48h, 1 GPU, 40GB RAM).

### `whisper-modeling/` — USC-SAIL Whisper model

**Model** (`models/whisper.py`): `WhisperWrapper` freezes the Whisper backbone and applies LoRA (default rank=8) to `fc1`/`fc2` of each encoder layer. Classification head maps frame-level encoder outputs → 4 classes (silence/child/adult/overlap) at 20ms resolution.

**Training** (`lightning_modules/classifier.py`): PyTorch Lightning with NLLLoss. Config in `configs/config.yaml` sets paths, LoRA rank, LR (0.001), batch size (64), 10s windows at 50% overlap, max 20 epochs.

**Data** (`dataset_classes/`): `preprocess.py` loads 10s windows (5s stride), maps per-frame labels from annotation CSVs (25ms frame, 20ms stride). Labels `"c"/"child"/"CHI"` → 1, `"a"/"adult"/"ADT"` → 2, silence → 0, overlap → 3.

**Inference post-processing**: majority filter (3-frame window) → merge segments within ~200ms gap → drop segments <50ms.

**Pre-trained checkpoint**: `whisper-base_rank8_pretrained_50k.pt` must be present in `whisper-modeling/` for out-of-the-box inference.

### `baselines/` — Encoder baselines

Three encoder variants (Whisper, WavLM, Fused) × two pooling strategies (mean, attention) → linear classifier. Results cached under `baselines/baseline_results/`.

### `av_fusion/` — Audio-Visual Fusion Pipeline

Experimental AV extension of the audio-only pipeline. Binary clip-level classification: does the target child vocalize? Evaluated on seen-child split.

**Module layout**:
- `scripts/utils.py` — shared helpers: `compute_metrics()`, `tune_threshold_f1/balanced_acc()`, `tune_late_fusion_alpha()`, `assert_split_integrity()`, `save_json()`, `get_repo_root()`
- `scripts/face_utils.py` — `YuNetDetector` (OpenCV FaceDetectorYN), `IouCentroidTracker`, `visual_quality_score()`, `child_candidate_score()`, `compute_visual_eligibility()`
- `scripts/extract_visual_features.py` — Frame sampling + face detection → `visual_features.csv`; idempotent with JSON cache per clip
- `scripts/extract_asd_features.py` — Optional; calls `video/run_asd.py` subprocess to get TalkNet-ASD per-clip scores → `asd_features.csv`
- `scripts/build_av_feature_table.py` — Merges metadata + manual BIDS annotations + audio scores (val/test only) + optional visual/ASD features → `av_{train,val,test}.csv`; asserts split integrity
- `scripts/train_av_fusion.py` — Trains `AudioOnlyModel` (threshold-tuned BabAR prob), `VisualXGBModel` (XGBoost on visual features), and `GatedAVModel` (late fusion with val-tuned alpha + visual eligibility gate); saves four pkl files
- `scripts/evaluate_av_fusion.py` — Loads test CSV + pkl files; computes overall + age band + eligibility + strata metrics; optional `--plot` flag for PR/ROC curves
- `scripts/error_analysis_av.py` — Categorizes clips by failure mode: `av_helped_fp/fn`, `av_hurt_fp/fn`, `off_camera_miss`, `multi_face_ambiguous`
- `configs/av_fusion.yaml` — XGBoost HPs, feature column lists per model class, `seed: 42`, `audio_score_col: prob`
- `slurm/run_av_pipeline.sh` — 48h GPU job for full visual feature extraction
- `face_track_cache/` — per-clip face detection JSON cache (shared between extract_visual_features.py and extract_asd_features.py)

**Architecture** (late fusion):
- Audio scores from BabAR enrollment only exist for val/test (train-set scores not available without leakage); train split uses visual features only
- At inference: `final_prob = alpha * audio_prob + (1-alpha) * visual_prob` for eligible clips; audio-only for ineligible clips (gated model)
- `visual_eligible` flag thresholded from `visual_eligibility_score` on val set using balanced accuracy against `child_of_interest_clear_binary`

**MVP path** — manual BIDS annotations only (no video extraction required):
- `Video_Quality_Child_Face_Visibility`, `Video_Quality_Lighting/Resolution`, `Child_of_interest_clear`, `#_adults`, `#_children` are already present in `seen_child_splits/*.csv`
- `visual_eligibility_score` falls back to `0.6 * manual_face_visibility_norm + 0.4 * manual_quality_norm` when automatic features are absent

**Result layout**: `av_fusion/av_results/{run_name}/` — master features CSV, per-split CSVs, `models/` pkl files, metrics JSONs, predictions CSV, `figures/`

---

## Data Splits

There are **three splits locations** representing different evaluation paradigms:

| Location | Strategy | Size | Used by |
|---|---|---|---|
| `whisper-modeling/seen_child_splits/` | **Within-child** (same 109 children in train/val/test), 60/20/20 | 2183 clips | Enrollment runs (all diarizers), combined feature models |
| `baselines/splits/` | **Cross-child** (97 train / 21 val / 21 test children, disjoint) | 2377 clips | Baseline encoder models |
| `splits/` | Copy/alternate of baselines/splits | 2377 clips | — |

**Split generation**: `make_seen_child_split.py` loads annotations from `/orcd/scratch/bcs/001/sensein/sails/BIDS_data/anotated_processed.csv`, filters to ≥5 clips per child per timepoint (14_month, 36_month), stratifies 60/20/20 within each (child, timepoint) group. Seed=42.

The `seen_child_splits/` approach tests enrollment-based personalization (the model has seen the target child during training); the `baselines/splits/` approach tests generalization to unseen children.

---

## Results Storage

### Enrollment run folders

- `whisper-modeling/usc_sail_enrollment_runs/` — USC-SAIL results
- `pyannote/pyannote_enrollment_runs/` — Pyannote results
- `babar_ecapa_enrollment_runs/` — BabAR basic enrollment
- `babar_combined_runs/` — BabAR combined feature models
- `vtc_ecapa_enrollment_runs/` — VTC 2.0 standalone (KCHI+OCH) enrollment
- `vtc_kchi_ecapa_enrollment_runs/` — VTC 2.0 standalone (KCHI only) enrollment
- `vbx_ecapa_enrollment_runs/` — VBx speaker diarization enrollment
- `video_asd_ecapa_enrollment_runs/talknet_asd/` — TalkNet-ASD video ASD enrollment
- `video_asd_ecapa_enrollment_runs/ts_talknet/` — TS-TalkNet video ASD enrollment
- `mil/mil_results/wavlm_mil/` — Frame-window MIL with WavLM-Base+ backbone; `best_checkpoint.pt`, `config.json`, `val/test_metrics_tuned.json`, `val/test_predictions.csv`, `val/test_metrics_by_timepoint.csv`; `age_stratified/{14_month,36_month}/` after age-stratified eval
- `mil/mil_results/whisper_mil/` — Frame-window MIL with Whisper-small backbone; same layout as `wavlm_mil/`
- `mil/mil_results/seg_mil/` — Segment-instance MIL sweep results (28 configs); `all_configs.json` summary + per-config subdirs
- `synth_results/manifests/` — `segment_manifest.csv`, `synthetic_manifest.csv`, `train_{ratio}x_manifest.csv` files (committed)
- `synth_results/augmentation_experiments/{config_name}/` — per-ratio enrollment results, `metrics_by_ratio.csv`, `metrics_by_age_band.csv`, `error_analysis.csv`, `figures/` (committed); scene WAVs in `synth_results/synthetic_scenes/` are gitignore'd
- `baselines/audio_llm_baseline_runs/{model_slug}/` — Audio LLM baseline results; `val_predictions.csv`, `val_metrics_tuned.json`, `test_predictions.csv`, `test_metrics_tuned.json`, `test_metrics_by_timepoint.csv`, `config.json`; cache files in `baselines/audio_llm_cache/` are gitignore'd

Each folder contains:
- `config.json` — full config
- `child_prototype_stats.csv` — per-child ECAPA prototype status
- `role_only_*` — duration-only baseline (no embeddings): `{threshold_sec, f1, precision, recall, auroc, auprc}`
- `enroll_*` or `test_*` — embedding enrollment results

**Key enrollment test metrics (seen-child split):**

| Diarizer | F1 | Precision | Recall | AUROC | AUPRC |
|---|---|---|---|---|---|
| USC-SAIL | 0.874 | 0.806 | 0.955 | 0.625 | 0.793 |
| Pyannote | 0.858 | 0.820 | 0.899 | 0.661 | 0.826 |
| BabAR | 0.874 | 0.912 | 0.839 | 0.820 | 0.918 |
| VTC (KCHI+OCH) | 0.888 | 0.866 | 0.910 | 0.787 | 0.895 |
| VTC-KCHI | 0.874 | 0.912 | 0.839 | 0.820 | 0.918 |
| VBx | 0.858 | 0.797 | 0.928 | 0.686 | 0.851 |
| TalkNet-ASD | 0.336 | 0.908 | 0.206 | 0.569 | 0.791 |
| WavLM-MIL | 0.882 | 0.807 | 0.973 | 0.771 | 0.893 |
| Whisper-MIL | 0.886 | 0.868 | 0.904 | 0.853 | 0.946 |
| Audio LLM (Qwen2-Audio-7B, zero-shot) | 0.871 | 0.807 | 0.946 | 0.725 | 0.853 |

**BabAR per-timepoint combined features** (`babar_combined_runs/all_model_results.json`):
- 14_month: F1=0.907, AUROC=0.892, AUPRC=0.949
- 36_month: F1=0.891, AUROC=0.865, AUPRC=0.948

### Log files

SLURM job output goes to `logs/adult/*.out` and `logs/seedlings/*.out`. When multiple `.out` files share a base name, **the highest-numbered one is the most recent run** and contains the final results. Logs show per-file diarization output (audio path → frame counts for child/adult/overlap).

### Caches

- `whisper-modeling/usc_sail_rttm_cache/` — cached USC-SAIL RTTM predictions per audio file
- `whisper-modeling/usc_sail_segment_cache/` — cached ECAPA embeddings per segment
- `pyannote/pyannote_rttm_cache/` — cached Pyannote RTTM predictions
- `pyannote/vtc_rttm_cache/` — cached VTC 2.0 standalone RTTM predictions
- `pyannote/vbx_rttm_cache/` — cached VBx RTTM predictions
- `pyannote/video_asd_rttm_cache/` — cached video ASD RTTM predictions (per model: `talknet_asd/`, `ts_talknet/`)
- `pyannote/video_face_cache/` — cached S3FD face track JSON files (shared across video ASD models)

If audio files change, delete the relevant cache directory before re-running.

---

## Important Gotchas

- `scripts/main.py` has a hardcoded `pdb.set_trace()` around line 41 — **remove before any cluster/batch run** or the job will hang waiting for a debugger
- `pyannote/unified.py` and `whisper-modeling/usc_sail_run_enrollment.py` overlap in USC-SAIL enrollment logic; `unified.py` is the more general/current version
- BabAR and Pyannote require separate Python environments; do not install into the main whisper-modeling env
- `babar_three.py` requires BabAR RTTM outputs and phoneme CSVs to already exist before running — it is a downstream model, not a standalone pipeline
- Dataset folders (`playlogue/`, `providence/`, `seedlings/`) contain raw audio and ground-truth RTTMs; `seedlings/` data requires Databrary API credentials via `seedlings_import.py`
- VBx requires HF_TOKEN (same as Pyannote) for `pyannote/segmentation-3.0` and `pyannote/embedding`; set up with `cd VBx && uv sync`
- VTC standalone requires `cd BabAR/VTC && uv sync`; checkpoint must be at `VTC/VTC-2.0/model/best.ckpt`
- VBx RTTM accuracy on Providence is incomplete — `pyannote/eval_results/vbx_providence/` has `per_file_predictions/` but no `aggregate_metrics.json`
- **Video files only exist for SAILS BIDS data** — Providence and Playlogue are audio-only; `talknet_asd` and `ts_talknet` frontends return [] for those datasets (no crash)
- Video ASD repos (`video/TalkNet-ASD/`, `video/TS-TalkNet/`) and checkpoints (`video/pretrain/`) are `.gitignore`'d and must be cloned/downloaded per `video/SETUP.md`
- `video/` env requires Python 3.10 (uv-managed); do not run video ASD scripts from the main whisper-modeling or pyannote envs
- LocoNet (`video/LoCoNet_ASD/`) and Light-ASD (`video/Light-ASD/`) repos and checkpoints are also `.gitignore`'d; see 007-av-extensions setup instructions above
- `extract_gpt4o_features.py` requires `OPENAI_API_KEY` env var; uses `gpt-4o-mini` by default (~$0.66 for 2183 clips at 2 frames each); supports `--dry-run` for cost estimation before API calls
- `train_cascaded_pipeline.py` requires `av_val.csv` from the 006 pipeline to exist; test thresholds come from `cascade_thresholds.json` (val-tuned only)
- `smooth_predictions.py` requires `--val-predictions` when `--param None`; smoothing is scoped within (child_id, timepoint_norm) groups — no cross-child information leakage
- `synth/scripts/build_segment_manifest.py` **must** receive `--exclude-speakers-csv` pointing to the real test split — omitting it leaks test-child speech into training segments
- Synthetic scene WAVs (`synth_results/synthetic_scenes/wav/`) and extracted segments (`data/segments/`) are gitignore'd; only manifests, configs, metrics, and scripts are committed
- `synth/scripts/generate_scenes.py` is CPU-only; do not request a GPU node for scene generation
- Deleting and regenerating only part of a scene set breaks reproducibility — always regenerate the full N scenes for a given config + seed pair
- **Audio LLM prompt cache invalidation** — if the prompt template in `baselines/audio_llm_baseline.py` changes, delete `baselines/audio_llm_cache/{model_slug}/` before rerunning; cached logits were generated with the old prompt and will silently produce wrong results
- **Audio LLM test-before-val guard** — `python baselines/audio_llm_baseline.py --split test` exits with code 2 if `val_metrics_tuned.json` is missing; run val first

## Recent Changes
- **Age-stratified enrollment** (spec-001, 2026-04-27, job 12614919): All 6 diarizers × 2 age cohorts complete. 36_month consistently outperforms 14_month. Key results (`pyannote/{d}_age_stratified/{age}/{age}/test_metrics_tuned.json`): BabAR 12_16m F1=0.865/14_month consistent improvement; VTC 34_38m F1=0.916 (best); USC-SAIL shows largest age gap (14_month F1=0.825 vs 34_38m F1=0.906). Note: outputs are in doubly-nested `{age}/{age}/` subdirs due to unified_age_stratified.py path behavior.
- **Synthetic augmentation null result** (spec-008, 2026-04-27): All 6 augmentation ratios (0×–10×) produce identical enrollment metrics (F1=0.874, AUROC=0.820, AUPRC=0.918). Root cause: ECAPA encoder is frozen; synthetic BabAR RTTM cache reused from baseline run; synthetic ECAPA embeddings are similar to real ones, so prototype averaging is unaffected. Error analysis: 360/441 test clips unchanged; 81 unchanged errors (44 short-vocalization, 23 overlap, 7 adult-background FP). Full results: `synth_results/augmentation_experiments/default_14_18mo/`
- **Audio LLM results** (spec-010, 2026-04-27): Qwen2-Audio-7B-Instruct zero-shot — test F1=0.871, AUROC=0.725, AUPRC=0.853, val F1=0.859, AUROC=0.781, AUPRC=0.898. AUROC/AUPRC below BabAR (delta_auroc=-0.095, delta_auprc=-0.065) but F1 near-identical (delta=-0.003). 14_month F1=0.838, 36_month F1=0.904. Fixed 3 inference bugs: wrong model class, wrong processor kwarg (`audios=`→`audio=`), constrained decoding (`model.generate()`→`model(**inputs)` forward pass + logsumexp). Results: `baselines/audio_llm_baseline_runs/qwen2_audio_7b/`
- 009-synth-rir-noise: Added Python 3.11, `child-vocalizations` conda env + `transformers>=4.45`, `accelerate`, `torchaudio`, `soundfile`, `pandas`, `scikit-learn`, `numpy`; optional: `bitsandbytes` for 4-bit quantization
- evaluation analyses (cross-diarizer): Added `evaluation/build_master_table.py` (bootstrap CIs for 11 diarizers), `evaluation/stat_significance.py` (pairwise AUROC bootstrap tests, 55 pairs), `evaluation/weak_diarization_correlation.py` (attention-weight vs. classification AUROC Spearman), `evaluation/per_diarizer_error_analysis.py` (FP/FN breakdown by interaction/timepoint/n_children for all diarizers), `evaluation/comprehensive_stratified_analysis.py` (12 SAILS annotation factors × 11 diarizers → stratified_analysis/), `evaluation/cross_diarizer_persistent_errors.py` (persistent FP/FN across diarizers, pairwise agreement matrix, unique contributions, per-child error rates, confidence calibration), `evaluation/double_stratification.py` (2-way interaction effects: gestures×interaction, face_vis×diarizer_type, n_children×n_adults, timepoint×face_vis); outputs under `evaluation/cross_diarizer_errors/`, `evaluation/stratified_analysis/`, `evaluation/double_stratification/`

## Active Technologies
- Python 3.11, `child-vocalizations` conda env + `transformers>=4.45`, `accelerate`, `torchaudio`, `soundfile`, `pandas`, `scikit-learn`, `numpy`; optional: `bitsandbytes` for 4-bit quantization (009-synth-rir-noise)
- Per-clip JSON cache at `baselines/audio_llm_cache/{model_slug}/`; result CSVs and JSONs at `baselines/audio_llm_baseline_runs/{model_slug}/`; no database (009-synth-rir-noise)
