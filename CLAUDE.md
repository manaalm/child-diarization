# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview


The goal is per-clip child presence detection: given a short audio clip, predict whether a target child is vocalizing. A synthetic scene generator (`synth/`) produces augmented training data by mixing child speech (Providence + TinyVox + Playlogue) and adult speech (Providence parents + LibriSpeech + Playlogue) under configurable SNR, RIR, overlap, and scene-type distributions. (Note: the original 5000-scene v1 corpus inadvertently used **only Providence parents** for adults — the LibriSpeech and Playlogue sources were added in the v2 rebuild on 2026-04-30; see "Synthetic Data Generator" below.) Nine diarization frontends are compared:
1. **USC-SAIL** — Fine-tuned Whisper + LoRA frame classifier (`whisper-modeling/`)
2. **Pyannote** — `pyannote/speaker-diarization-community-1` model
3. **BabAR** — VTC 2.0 child speech diarizer (full pipeline with phoneme step)
4. **VTC** — VTC 2.0 standalone (no BabAR phoneme step); two variants: `vtc` (KCHI+OCH) and `vtc_kchi` (KCHI only)
5. **VBx** — Variational Bayes HMM speaker diarization using pyannote VAD + ECAPA embeddings; anonymous speaker labels resolved via cosine similarity to target-child prototype
6. **TalkNet-ASD** — Video-audio active speaker detection (SAILS BIDS .mp4 only); child identified as smallest face track
7. **Fine-tuned TalkNet** — TalkNet-ASD backbone fine-tuned for clip-level child vocalization (replaces TS-TalkNet, whose checkpoint was unavailable); phase 1 freezes backbone + trains clip-level pooling head; phase 2 full fine-tune; AV path for clips with cached face tracks, audio-only fallback; `video/talknet_child_finetune.py`
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

# Fine-tune TalkNet-ASD for child vocalization (replaces TS-TalkNet)
# Pretrained checkpoint auto-downloads; face crops precomputed from video_face_cache/
sbatch video/slurm/run_talknet_finetune.sh
# Output: video_finetuned_talknet_runs/{best_checkpoint.pt, val/test_metrics_tuned.json,
#          test_predictions.csv, config.json}
# Manual run (from video/):
# .venv/bin/python talknet_child_finetune.py --skip-precompute  # if crops already cached
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

### Metadata-Conditioned Routing and Ensemble Extensions (spec-012)

CPU-only (US1/US2) and GPU SLURM (US3/US4) experiments extending the ensemble pipeline with BIDS metadata.

```bash
# Verify all 10 system prediction files are loadable (exits 0 if clean)
python evaluation/metadata_router.py --verify

# US1: Metadata-augmented LR/GBM stacker (train on val, eval on test, ~1 min)
python evaluation/metadata_router.py --mode stack
# Output: ensemble_runs/metadata_stack/{test_metrics_tuned.json, feature_importances.json,
#          test_predictions.csv, val_metrics_tuned.json, config.json}

# US2: Rule-based + learned metadata router (~1 min)
python evaluation/metadata_router.py --mode router
# Output: ensemble_runs/metadata_router_rule/  +  ensemble_runs/metadata_router_learned/
#         (same file layout as above; config.json includes routed_system distribution)

# Run both US1 and US2 together:
python evaluation/metadata_router.py --mode all

# US3: Multi-child FP suppressor (requires GPU, ~30 min)
python evaluation/multi_child_suppressor.py --dry-run   # print stratum size, no training
sbatch evaluation/slurm/run_multi_child_suppressor.sh
# Output: mil/mil_results/multi_child_suppressor/{test_metrics_multi_child_only.json,
#          test_metrics_single_child_only.json, test_metrics_tuned.json,
#          test_predictions.csv, emb_cache.npz, config.json}

# US4: Short-vocalization specialized head (requires GPU, ~4h)
python evaluation/short_voc_head.py --dry-run           # print short-voc clip counts, no training
sbatch evaluation/slurm/run_short_voc_head.sh
# Output: mil/mil_results/short_voc_head/{best_checkpoint.pt, test_metrics_short_voc_clips.json,
#          test_metrics_non_short_voc_clips.json, test_metrics_tuned.json,
#          test_predictions.csv, config.json}
```

**Key results (spec-012):**

| Config | F1 | AUROC | delta_F1 | delta_AUROC |
|---|---|---|---|---|
| Baseline (best_audio_mil mean) | 0.893 | 0.878 | — | — |
| Metadata stacker (US1) | 0.901 | 0.900 | +0.009 | +0.022 |
| Rule router (US2) | 0.883 | 0.705 | −0.010 | −0.173 |
| Learned router (US2) | 0.873 | 0.731 | −0.020 | −0.147 |
| Multi-child suppressor (US3) | TBD (SLURM) | TBD | — | — |
| Short-voc head (US4) | TBD (SLURM) | TBD | — | — |

### Self-Distillation: Pseudo-Frame-Label Classifier (`pseudo_frame/`)

WavLM-Base+ frozen frame classifier trained on diarizer-derived pseudo-labels (mean of VTC KCHI + USC-SAIL CHI at 50 Hz). Closes the MIL → frame-detection localization gap.

```bash
# Step 1: build pseudo frame labels (one-shot, ~3 min for all 2183 clips)
python pseudo_frame/build_pseudo_labels.py
# Output: pseudo_frame/pseudo_labels/{<md5>.npy, index.csv}

# Step 2: train + evaluate (single SLURM job, ~5 min total)
sbatch pseudo_frame/slurm/train_pseudo.sh
# Output: pseudo_frame/results/wavlm_pseudo_frame/{best_checkpoint.pt, config.json,
#          training_history.csv, val/test_metrics_tuned.json, test_predictions.csv,
#          frame_localization.json, frame_localization_per_clip.csv}
# Logs:   logs/pseudo_frame/train_<jobid>.out
```

**Key results (pseudo-frame, seen-child test n=441):** F1=0.869, AUROC=**0.831** (+0.060 vs WavLM-MIL), AUPRC=**0.937** (+0.044). Frame-level localization vs held-out test pseudo-labels: mean per-clip Pearson **0.566** (vs MIL attention's 0.084 — ~6.7× gain). Frame Spearman 0.524, frame-AUROC 0.853.

### AV Self-Distillation and Visual-Eligibility Fusion (spec-015)

Four user stories layering on the pseudo-frame classifier and `av_fusion/face_track_cache/`. Shared design: frozen pretrained encoders + tiny fusion + visual-eligibility gating (audio_visual.txt §63, §151, §157).

```bash
# US1: extract per-clip visual eligibility features (CPU, ~3 min)
python pseudo_frame/visual_eligibility.py
# Output: pseudo_frame/visual_features/visual_eligibility.csv (n=2183)

# US1: re-run metadata stacker with visual features
python evaluation/metadata_router.py --mode stack \
  --visual-features pseudo_frame/visual_features/visual_eligibility.csv
# Output: ensemble_runs/metadata_stack_av/{test_metrics_tuned.json, ...}

# US1: stratified ablation (mirror spec-012 ablation)
python evaluation/metadata_stack_av_ablation.py
# Output: ensemble_runs/metadata_stack_av/ablation/

# US2/US3/US4: extract per-clip face/mouth-motion features (CPU SLURM, ~4-6h)
sbatch pseudo_frame/slurm/extract_mouth_motion.sh
# Sharded version for parallelism (split rows 0-2183):
# python pseudo_frame/extract_mouth_motion.py --start-row 0    --end-row 1300 --out shard1.csv
# python pseudo_frame/extract_mouth_motion.py --start-row 1300 --end-row 2183 --out shard2.csv
# Output: pseudo_frame/visual_features/mouth_motion.csv

# US2: AV-HuBERT-style late fusion (CPU, ~30s after extraction)
python pseudo_frame/avhubert_late_fusion.py
# Output: pseudo_frame/results/avhubert_lipfusion/{audio_only,always_fuse,gated_av}/
#          subset_eligible_metrics.json, test_predictions_all.csv, config.json

# US3: speaker-embedding-informed AV (Clarke 2025 simplified, CPU, ~30s)
python pseudo_frame/speaker_informed_asd.py
# Output: pseudo_frame/results/speaker_informed_asd/{test_metrics_tuned.json,
#          multi_child_test_metrics.json, test_predictions.csv, config.json}

# US4: audio→video clip-level pseudo-label distillation (CPU, ~30s)
python pseudo_frame/audio2video_distill.py
# Output: pseudo_frame/results/audio2video_distilled/{test_metrics_tuned.json,
#          visual_student_correlation.json, test_predictions.csv, config.json}
```

**Note on US2 substitution**: AV-HuBERT requires fairseq (non-trivial install on this cluster), so US2 substitutes hand-engineered face/mouth-motion features (face/mouth intensity std + frame-to-frame motion energy on bbox crops). Architectural intent (frozen visual extractor + tiny fusion + visual-eligibility gating) is preserved. Future work: install AV-HuBERT and re-run US2/US4 with real visual-speech embeddings.

### Synthetic Data Generator (`synth/`)

7-step pipeline: build segment manifest → extract segments → generate scenes → make training manifests → train at each ratio → evaluate → error analysis. No GPU required for steps 1–3.

> **v1 vs v2 corpus** (2026-04-30): the original 5000-scene corpus (`synth_results/synthetic_scenes/`, `synthetic_manifest_v1.csv`) was built **without LibriSpeech and without Playlogue** despite both being listed in `synth/configs/default_14_18mo.yaml` — the build_segment_manifest.py invocation in `synth/slurm/run_scene_generation.sh` did not pass `--librispeech-dir` and the script had no `--playlogue-dir` support at all. Adult speech in v1 came entirely from `providence_adults` (the parents in the Providence corpus). The v2 corpus (`synth_results/synthetic_scenes_v2/`, `synthetic_manifest.csv` after re-run, segment manifest `segment_manifest_v2.csv` with 294,745 segments) properly includes LibriSpeech (28,539 adult segs, ~100 h) and Playlogue (24,412 child + 27,558 adult segs from cameron/ew/gleason/vh CHILDES corpora — disjoint from BIDS test). v2 use is preferred for all new spec-016-style augmentation experiments. Use `synth/slurm/run_scene_generation_v2.sh` to regenerate.

```bash
# Step 1: build Providence + TinyVox + Playlogue + LibriSpeech segment manifest
python synth/scripts/build_segment_manifest.py \
  --providence-dir        providence/ \
  --providence-rttm-dir   providence/rttm/ \
  --tinyvox-dir           data/tinyvox/ \
  --playlogue-dir         playlogue/audio/ \
  --playlogue-rttm-dir    playlogue/rttm/ \
  --librispeech-dir       data/LibriSpeech/LibriSpeech/train-clean-100/ \
  --exclude-speakers-csv  whisper-modeling/seen_child_splits/test.csv \
  --output                synth_results/manifests/segment_manifest_v2.csv
# v2 outputs: ~295k segments (providence 74k + providence_adults 91k + tinyvox 25k +
#             librispeech 29k + playlogue 49k + playlogue_adults 28k)
# Original v1 manifest used Providence + TinyVox only (~190k segments, no LibriSpeech, no Playlogue).
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

### Hard-negative MIL (balanced negatives from RTTM)

Addresses class imbalance in the training split (73.8% positive). Extracts 30s windows from
Playlogue/Providence RTTM files where CHI is silent but ≥3s of non-silence is active — these
are harder negatives than silent windows since a speaker is present but not the target child.
Brings pos:neg ratio from 967:344 (~2.8:1) down to 967:967 (1:1).

```bash
# Step 1: extract hard-negative windows (run inside 4-step SLURM script below, or manually)
python mil/scripts/extract_hard_negatives.py \
  --output synth_results/manifests/hard_negatives_manifest.csv \
  --window-sec 30 --stride-sec 15 --min-activity-sec 3 --max-per-file 20 --seed 42
# Output: synth_results/manifests/hard_negatives_manifest.csv
#   Columns: audio_path, start_sec, end_sec, label (=0), child_id, timepoint_norm, source
#   ~612 windows from 33 Playlogue + 579 Providence files (estimated)

# Step 2-4: train + evaluate both variants (single SLURM job)
sbatch mil/slurm/train_mil_hardneg.sh
# Trains: wavlm_mil_hardneg (WavLM-Base+, extra_negatives_cap=623 → 1:1 ratio)
#         whisper_mil_hardneg (Whisper-small, same cap)
# Output: mil/mil_results/{wavlm_mil_hardneg,whisper_mil_hardneg}/
#         best_checkpoint.pt, config.json, val/test_metrics_tuned.json, val/test_predictions.csv
# Logs: logs/mil/hardneg_{jobid}.out  (SLURM job 12770452)
```

Configs: `mil/configs/wavlm_mil_hardneg.yaml` and `mil/configs/whisper_mil_hardneg.yaml`.
Key config keys: `extra_negatives_csv` (path to manifest) and `extra_negatives_cap` (max rows to add).
The `MILBagDataset` supports `start_sec`/`end_sec` columns for slice-loading long files without
reading the full audio into memory.

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

### Synthetic Data Augmentation Extensions (spec-016)

Six independent training-recipe variants routing the existing 5000 synth scenes into pipelines where labels are load-bearing. Builders generate per-pipeline manifests/caches; existing training scripts ingest them via standard configs.

```bash
# Step 1: build all derived synth manifests (single pass)
python synth/scripts/build_synth_aug_manifests.py
# Outputs: synth_results/manifests/synthetic_{hardneg,cross_child_aug,audio_llm_shots,train_aug}.csv

# Step 2: per-candidate prerequisite builders
python synth/scripts/build_cross_child_synth_split.py     # C4: baselines/splits_synth_aug/
python synth/scripts/build_pseudo_synth_split.py          # C2/C5: whisper-modeling/seen_child_splits_synth_aug/
python synth/scripts/build_seg_mil_synth_cache.py         # C5: mil/seg_mil_combined_cache/ (real USC-SAIL + synth GT, 7112 entries)
python synth/scripts/build_usc_sail_synth_data.py         # C1: synth_results/usc_sail_data/{audios,labels}/{train,val}/
python pseudo_frame/build_synth_pseudo_labels.py          # C2: appends 5000 synth GT pseudo-frames to pseudo_frame/pseudo_labels/index.csv

# Step 3: submit per-candidate jobs
sbatch mil/slurm/train_eval_spec014.sh mil/configs/wavlm_mil_hardneg_synth.yaml         # C3 wavlm
sbatch mil/slurm/train_eval_spec014.sh mil/configs/whisper_mil_hardneg_synth.yaml       # C3 whisper
sbatch mil/slurm/train_eval_spec014.sh mil/configs/wavlm_mil_cross_child_synth.yaml     # C4 wavlm
sbatch mil/slurm/train_eval_spec014.sh mil/configs/whisper_mil_cross_child_synth.yaml   # C4 whisper
sbatch pseudo_frame/slurm/train_pseudo.sh pseudo_frame/configs/wavlm_pseudo_synth.yaml  # C2
sbatch mil/slurm/seg_mil_synth.sh                                                       # C5
sbatch baselines/slurm/run_audio_llm_synth_shots.sh val                                 # C6 val
sbatch baselines/slurm/run_audio_llm_synth_shots.sh test                                # C6 test (after val)
sbatch whisper-modeling/run_train_synth.sh                                              # C1 (requires PYTHONPATH=. + window_size 30)

# Audio LLM with universal synth demos (replaces per-child same-speaker demos):
python baselines/audio_llm_baseline.py --split val --n-shot 2 --universal-shots \
    --train-csv synth_results/manifests/synthetic_audio_llm_shots.csv \
    --model-slug qwen2_audio_7b_synth_2shot
```

Tracker: `mil/spec016_jobs.json` (per-job state + metrics + deltas, mirrors `spec014_jobs.json` schema).

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

Three encoder variants (Whisper, WavLM, Fused) × two pooling strategies (mean, attention) → linear classifier. Results cached under `baselines/baseline_results/` (cross-child split) or `baselines/baseline_results_seen_child/` (seen-child split).

**Seen-child mode** (`--seen-child`): reads pre-generated `whisper-modeling/seen_child_splits/{train,val,test}.csv` instead of re-splitting from scratch; enables direct comparison with enrollment-based diarizers on the same 109-child within-child split. Add `--all-experiments` to run all 13 variants (Phase 1–6). Submit via `sbatch baselines/slurm/run_baseline_seen_child.sh`.

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
| `baselines/splits/` | **Cross-child** (97 train / 21 val / 21 test children, disjoint) | 2377 clips | Baseline encoder models (default); `baseline_results/` |
| `whisper-modeling/seen_child_splits/` (reused) | **Within-child** via `--seen-child` flag | 2183 clips | Baseline encoders on seen-child split; `baseline_results_seen_child/` |
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
- `video_finetuned_talknet_runs/` — Fine-tuned TalkNet child vocalization; direct clip-level scores (no RTTM/ECAPA step); `best_checkpoint.pt`, `val/test_metrics_tuned.json`, `test_predictions.csv`, `config.json`
- `mil/mil_results/wavlm_mil/` — Frame-window MIL with WavLM-Base+ backbone; `best_checkpoint.pt`, `config.json`, `val/test_metrics_tuned.json`, `val/test_predictions.csv`, `val/test_metrics_by_timepoint.csv`; `age_stratified/{14_month,36_month}/` after age-stratified eval
- `mil/mil_results/whisper_mil/` — Frame-window MIL with Whisper-small backbone; same layout as `wavlm_mil/`
- `mil/mil_results/seg_mil/` — Segment-instance MIL sweep results (28 configs); `all_configs.json` summary + per-config subdirs
- `synth_results/manifests/` — `segment_manifest.csv`, `synthetic_manifest.csv`, `train_{ratio}x_manifest.csv` files (committed)
- `synth_results/augmentation_experiments/{config_name}/` — per-ratio enrollment results, `metrics_by_ratio.csv`, `metrics_by_age_band.csv`, `error_analysis.csv`, `figures/` (committed); scene WAVs in `synth_results/synthetic_scenes/` are gitignore'd
- `baselines/audio_llm_baseline_runs/{model_slug}/` — Audio LLM baseline results; `val_predictions.csv`, `val_metrics_tuned.json`, `test_predictions.csv`, `test_metrics_tuned.json`, `test_metrics_by_timepoint.csv`, `config.json`; cache files in `baselines/audio_llm_cache/` are gitignore'd
- **spec-016 synth-augmentation result dirs** (per-candidate):
  - `mil/mil_results/{wavlm,whisper}_mil_hardneg_synth/` — C3 MIL with synth-derived hardnegs
  - `mil/mil_results/{wavlm,whisper}_mil_cross_child_synth/` — C4 MIL on cross-child + synth
  - `mil/mil_results/seg_mil_synth/usc_sail_synth_combined_{gated_attention,transformer}/` — C5 seg-MIL with combined real+synth RTTM cache; `all_configs.json` index
  - `pseudo_frame/results/wavlm_pseudo_frame_synth/` — C2 pseudo-frame with synth GT pseudo-labels; standard pseudo-frame layout + `frame_localization.json`
  - `baselines/audio_llm_baseline_runs/qwen2_audio_7b_synth_2shot/` — C6 audio-LLM with universal synth demos
  - `whisper-modeling/checkpoints/whisper_base_synth/` — C1 USC-SAIL synth-only training; completed (job 12849231, 24m, exit 0:0); best ckpt `epoch=17-val_loss=0.235.ckpt`; frame-level acc=0.922 on synth val
  - Manifests at `synth_results/manifests/synthetic_{hardneg,cross_child_aug,audio_llm_shots,train_aug}.csv`; combined RTTM cache at `mil/seg_mil_combined_cache/` (7112 entries)
  - Aggregated job tracker: `mil/spec016_jobs.json` (mirrors `spec014_jobs.json` schema)

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
| LocoNet-ECAPA (video ASD, NEGATIVE) | 0.000 | 0.000 | 0.000 | 0.500 | 0.760 |
| Fine-tuned TalkNet (NEGATIVE) | 0.863 | 0.760 | 1.000 | 0.523 | 0.763 |
| EEND-EDA | 0.844 | 0.772 | 0.931 | 0.528 | 0.781 |
| Sortformer | 0.844 | 0.796 | 0.899 | 0.664 | 0.841 |
| WavLM-MIL | 0.882 | 0.807 | 0.973 | 0.771 | 0.893 |
| Whisper-MIL | 0.886 | 0.868 | 0.904 | 0.853 | 0.946 |
| **Whisper-MIL TS-MIL concat (spec-014 US4)** | **0.896** | **0.856** | **0.940** | **0.869** | **0.944** |
| HuBERT-large MIL layersum (spec-014 US1) | 0.878 | 0.802 | 0.970 | 0.813 | 0.920 |
| WavLM-MIL ACMIL (spec-014 US3, mean, NEGATIVE) | 0.870 | 0.783 | 0.979 | 0.733 | 0.877 |
| WavLM-MIL ACMIL topk (k=2, spec-014 US3 ext) | 0.884 | 0.814 | 0.967 | 0.775 | 0.902 |
| **Whisper-MIL ACMIL max (spec-014 US3 ext)** | **0.891** | **0.867** | **0.916** | **0.842** | **0.936** |
| Whisper-MIL ACMIL topk (k=2, spec-014 US3 ext) | 0.875 | 0.851 | 0.901 | 0.816 | 0.926 |
| WavLM-MIL child-adapted (spec-014 US2, NEGATIVE) | 0.863 | 0.760 | 1.000 | 0.500 | 0.760 |
| Audio LLM (Qwen2-Audio-7B, zero-shot) | 0.871 | 0.807 | 0.946 | 0.725 | 0.853 |
| Audio LLM 2-shot synth demos (spec-016 C6, NEUTRAL) | 0.863 | 0.783 | 0.961 | 0.713 | 0.861 |
| Granite-Speech-1B zero-shot (NEGATIVE — null/random) | 0.863 | 0.761 | 0.997 | 0.454 | 0.726 |
| Cohere-Transcribe ASR gap_ratio (NEGATIVE — null) | 0.863 | 0.760 | 1.000 | 0.500 | 0.760 |
| Pseudo-frame WavLM synth-aug (spec-016 C2, NEGATIVE) | 0.876 | 0.785 | 0.991 | 0.763 | 0.910 |
| WavLM-MIL hardneg synth-aug (spec-016 C3 wavlm) | 0.863 | 0.760 | 1.000 | 0.657 | 0.851 |
| Whisper-MIL hardneg synth-aug (spec-016 C3 whisper) | 0.877 | 0.817 | 0.946 | 0.822 | 0.931 |
| WavLM-MIL cross-child synth-aug (spec-016 C4 wavlm, NEGATIVE) | 0.864 | 0.760 | 1.000 | 0.620 | 0.835 |
| Whisper-MIL cross-child synth-aug (spec-016 C4 whisper, STRONG NEGATIVE) | 0.859 | 0.755 | 0.997 | 0.589 | 0.780 |
| Seg-MIL synth-aug transformer (spec-016 C5, POSITIVE) | 0.871 | 0.778 | 0.991 | 0.637 | 0.829 |
| **USC-SAIL synth-only frame classifier (spec-016 C1, frame-level acc=0.922)** | n/a | n/a | n/a | n/a | n/a |
| Pseudo-frame WavLM C1-self-distill (spec-016 follow-up #8, NEGATIVE) | 0.865 | 0.766 | 0.994 | 0.690 | 0.856 |
| Voice-transfer LR (spec-016 follow-up #1, WavLM mean-feature, NEUTRAL) | 0.871 | 0.786 | 0.976 | 0.750 | 0.893 |
| Parakeet TDT 0.6B (gap_ratio, NEGATIVE) | 0.863 | 0.760 | 1.000 | 0.457 | 0.731 |
| **Ensemble (best_audio_mil, mean)** | **0.893** | — | — | **0.878** | **0.956** |
| Metadata stacker (spec-012 US1) | 0.901 | 0.901 | 0.901 | 0.900 | 0.964 |

**spec-012 Metadata-Conditioned Routing/Ensemble** (`ensemble_runs/`, `mil/mil_results/`):
- `ensemble_runs/metadata_stack/`: US1 stacker — F1=0.901, AUROC=0.900, delta_F1=+0.009
- `ensemble_runs/metadata_router_rule/`: US2 rule router — F1=0.883, AUROC=0.705 (routed_system distribution in config.json)
- `ensemble_runs/metadata_router_learned/`: US2 learned router — F1=0.873, AUROC=0.731
- `mil/mil_results/multi_child_suppressor/`: US3 suppressor — test_metrics_multi_child_only.json + test_metrics_single_child_only.json; emb_cache.npz
- `mil/mil_results/short_voc_head/`: US4 short-voc head — best_checkpoint.pt; test_metrics_short_voc_clips.json; test_metrics_non_short_voc_clips.json

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

- `pyannote/unified.py` and `whisper-modeling/usc_sail_run_enrollment.py` overlap in USC-SAIL enrollment logic; `unified.py` is the more general/current version
- BabAR and Pyannote require separate Python environments; do not install into the main whisper-modeling env
- `babar_three.py` requires BabAR RTTM outputs and phoneme CSVs to already exist before running — it is a downstream model, not a standalone pipeline
- Dataset folders (`playlogue/`, `providence/`, `seedlings/`) contain raw audio and ground-truth RTTMs; `seedlings/` data requires Databrary API credentials via `seedlings_import.py`
- VBx requires HF_TOKEN (same as Pyannote) for `pyannote/segmentation-3.0` and `pyannote/embedding`; set up with `cd VBx && uv sync`
- VTC standalone requires `cd BabAR/VTC && uv sync`; checkpoint must be at `VTC/VTC-2.0/model/best.ckpt`
- VBx RTTM accuracy on Providence: completed; `pyannote/eval_results/vbx_providence/aggregate_metrics.json` reports Micro F1=0.529, Macro F1=0.305, AUROC=0.515 (matches THESIS_MEGADOC §8 table)
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
- **USC-SAIL training requires `PYTHONPATH=.`** — `python scripts/main.py ...` from inside `whisper-modeling/` raises `ModuleNotFoundError: lightning_modules` because `scripts/` (not `.`) is on sys.path. Set `PYTHONPATH=.` before invocation. Encoded into `whisper-modeling/run_train_synth.sh`.
- **USC-SAIL window_size must be 30 on transformers ≥4.57** — Whisper encoder hard-checks `mel_features.length == 3000` (= 30s @ 16kHz). The original `window_size: 10` config produces 1000-frame mels and raises `ValueError: Whisper expects mel input length 3000`. Synth scenes are 30s natively, so set `window_size: 30, batch_size: 16` (memory). The original anfengxu 5k config worked on older transformers where this check was absent.
- **Custom WhisperWrapper API drift on transformers ≥4.57** (whisper-modeling/models/whisper.py) — newer transformers changed Whisper internals: (1) `WhisperAttention.__init__` now requires `config=` arg or `self.config` is None, breaking later `_attn_implementation` access — fixed by passing `config=config`; (2) `WhisperAttention.forward` now returns 2-tuple instead of 3-tuple (dropped `past_key_value`) — fixed by robust unpacking `result[0], result[1]`. Both fixes committed.
- **Granite-Speech requires `<|audio|>` placeholder** — `processor(text=prompt, ...)` raises "Number of audio tokens does not match number of audio features" if prompt lacks the `<|audio|>` token. `score_granite_llm` injects it automatically. Even with the fix, the 1B model produces near-random scores on zero-shot child-vocalization (AUROC≈0.45-0.50) — this is a model-capability ceiling, not a setup bug.
- **Audio model error fallback poisons cache** — `baselines/audio_model_baseline.py` `run_inference` writes `score=0.5` to cache on any per-clip exception. If a buggy model run errors all clips, the cache fills with 431×0.5 entries. Subsequent runs see "all cached", skip model load, compute AUROC=0.5 exactly. After fixing model code, **delete the cache** before resubmitting: `rm baselines/audio_model_cache/{model_slug}{_cross_child}/{val,test}_scores.json`.
- **transformers >=4.57 has_file() network bug — set TRANSFORMERS_OFFLINE=1 in SLURM** (2026-04-30). transformers' `_get_resolved_checkpoint_files()` calls `has_file()` even for fully-cached models, doing a network roundtrip to HF Hub. On compute nodes with intermittent / slow / proxied connections, the response sometimes misparses → misleading `OSError: microsoft/wavlm-base-plus is not a local folder or a valid repository name on 'https://hf.co'`. The cache exists, the network is reachable, but the check fails. Fix: export `TRANSFORMERS_OFFLINE=1` and `HF_HUB_OFFLINE=1` in any SLURM script that loads from `from_pretrained()`. Already added to `pseudo_frame/slurm/train_pseudo.sh`, `mil/slurm/train_eval_spec014.sh`, `mil/slurm/seg_mil_synth.sh`, `baselines/slurm/run_audio_llm_synth_shots.sh`.
- **synth corpus v1 lacked LibriSpeech and Playlogue** (2026-04-30 audit). The original 5000-scene corpus (`synth_results/synthetic_scenes/`) was generated from a manifest containing 0 LibriSpeech segments and 0 Playlogue segments — the SLURM script `synth/slurm/run_scene_generation.sh` did not pass `--librispeech-dir` to `build_segment_manifest.py`, and the script had no `--playlogue-dir` support at all. All v1 adult speech came from `providence_adults` (the parents of the children in Providence). v2 corpus (`synthetic_scenes_v2/`, manifest `segment_manifest_v2.csv` with 294,745 segments) corrects this. Verify the source mix in any new synth corpus by inspecting the `source_dataset` field in scene JSONs (`synth_results/synthetic_scenes_v?/json/*.json` under `source_segments`). Spec-016 results were produced on v1; re-runs should target v2.
- **Canary-Qwen-2.5b NeMo↔HF format mismatch** — `EncDecMultiTaskModel.from_pretrained("nvidia/canary-qwen-2.5b")` fails with `FileNotFoundError: model_config.yaml` because the HF-uploaded model has HF format (config.json + safetensors) but NeMo's loader expects a `.nemo` bundle with `model_config.yaml`. NeMo 2.7.3 doesn't support loading HF-only Canary uploads. Currently blocked; would require either NGC download (NeMo format) or rewriting the loader to use `transformers.AutoModel`.

## Recent Changes
- **Spec-016 follow-ups #8 (C1 self-distillation) and #1 (voice transfer)** (2026-04-29, SLURM jobs 12868860 + 12869264, both COMPLETED): two follow-up experiments to spec-016. **(8) C1 self-distillation NEGATIVE**: ran C1 USC-SAIL synth-only checkpoint (frame-acc 0.922 on synth) over real BIDS audio at 30s windows / 50Hz to generate distilled pseudo-frame labels (`pseudo_frame/pseudo_labels_c1/`); retrained WavLM pseudo-frame classifier on those labels → test F1=0.865 AUROC=**0.690 AUPRC=0.856 frame-Pearson=0.136** (vs baseline pseudo-frame 0.869/0.831/0.937/0.566 → ΔAUROC **−0.141**, ΔPearson **−0.430 collapse**). Root cause: C1 trained only on clean synth produces conservative predictions on real (mean pos rate 0.047 vs real GT ~25%); pseudo-frame student learns near-trivial localization. Result: `pseudo_frame/results/wavlm_pseudo_frame_c1distill/`. **(1) Voice-transfer NEUTRAL**: implemented per-child WavLM mean-feature voice transfer as a feature-space proxy for full voice cloning (full XTTS/SPARC blocked by senselab/coqui-tts numpy ABI conflict — env modification not authorized). Computed per-child WavLM-Base+ mean-pooled prototype from 101 train children with ≥3 positive clips, generic synth prototype from 2509 positive synth scenes, applied linear mean-shift `feat_aug = feat_synth - p_generic + p_child` to generate 1010 voice-transferred positive features (10 × 101 children), trained LR with vs without augmentation → ΔF1 −0.003, **ΔAUROC +0.0004**, ΔAUPRC −0.002 vs no-aug baseline. Effectively a wash. Mean-pooled features are too lossy to capture per-child voice identity; LR is shift-equivariant when same shift applied to all positives. Result: `synth_results/voice_transfer_experiment/results.json` + log `logs/synth/voice_transfer_12869264.out`. Both follow-ups confirm the spec-016 finding that the seen-child WavLM pipeline is information-saturated; further progress requires either richer features (frame-level rather than mean) or a different attack vector entirely.
- **Spec-016 Synth Augmentation Extensions** (2026-04-29, SLURM jobs 12845253–12848196): six independent training-recipe variants routing the existing 5000-scene synth corpus into pipelines where labels are load-bearing. **Mixed results so far** (4 of 8 results in): C2 pseudo-frame synth-aug NEGATIVE (AUROC 0.831→0.763, frame-Pearson 0.566→0.468 — synth swamps real and hurts both clip-level and frame-level transfer); C3a wavlm hardneg synth tiny POSITIVE (AUROC 0.642→0.657, +0.015 — synth-mined hardnegs slightly beat RTTM-mined ones); C5 seg-MIL synth combined cache strong POSITIVE (transformer aggregator: AUROC 0.518→0.637, +0.119; gated_attention: +0.035) — clean synth segments help most where the real-segment baseline was noisiest; C6 audio-LLM 2-shot synth demos NEUTRAL (AUROC 0.725→0.713 vs zero-shot, real-2shot identical at 0.725 — few-shot is a low-leverage axis on Qwen2-Audio for this task regardless of demo source). **Pattern**: synth helps where the baseline pipeline is information-starved (seg-MIL transformer), hurts where the baseline already has high-quality signal (pseudo-frame had Pearson 0.566 from VTC+USC-SAIL averaging). Spec dir: `specs/016-synth-augmentation-extensions/{spec,plan,tasks}.md`. Helper builders in `synth/scripts/build_*` produce all 4 derived manifests/caches; `pseudo_frame/build_synth_pseudo_labels.py` appends 5000 synth GT pseudo-frames to `pseudo_frame/pseudo_labels/index.csv`. C1 USC-SAIL completed (job 12849231 after PYTHONPATH fix + window_size 10→30 for transformers 4.57+ mel-3000 enforcement; 24m runtime, exit 0:0); best checkpoint `whisper-modeling/checkpoints/whisper_base_synth/epoch=17-val_loss=0.235.ckpt`, frame-level acc=0.922 on synth val. Used as input to follow-up self-distillation experiment (see preceding entry — NEGATIVE on real audio).
- **ACMIL branch-aggregation extension** (2026-04-29, SLURM jobs 12839667–12839672): Added `branch_aggregation: "mean"|"max"|"topk_mean"|"gated"` parameter on `ACMILHead` (mil/mil_model.py) plus `branch_topk` and learnable `branch_gate` (init zero → sigmoid 0.5 = mean-equivalent at init); new `forward_branches(h)` method exposes per-branch logits for no-retrain inference. 6 new YAML configs at `mil/configs/{wavlm,whisper}_mil_acmil_{max,gated,topk}.yaml` retrained from scratch. **Best new result: `whisper_mil_acmil_max` F1=0.891 AUROC=0.842 AUPRC=0.936** — +0.091 AUROC vs the original mean-aggregation baseline (which was a NEGATIVE in spec-014). `wavlm_mil_acmil_topk` (k=2) also positive: AUROC=0.775 (+0.042 vs wavlm mean baseline). Whisper gated retrain early-stopped at epoch 8 (instability). Helper script: `mil/slurm/run_acmil_branch_selection.sh <results_dir>` runs `mil/eval_acmil_branch_selection.py` for no-retrain per-branch / best-branch / max-over-branches / topk_mean inference from an existing ACMIL checkpoint (jobs 12844150 wavlm + 12844151 whisper in flight; outputs `branch_selection.{csv,json}` in the results dir).
- **Spec-014 MIL Extensions completed** (2026-04-29, SLURM jobs 12805726–12805739, fire-and-forget orchestrator `mil/slurm/run_spec014.sh`, tracker `mil/scripts/track_spec014.py`): All 11 jobs completed. **Whisper-MIL TS-MIL concat is the only positive frame-window result**: F1=0.896 AUROC=0.869 AUPRC=0.944 (vs Whisper-MIL baseline 0.886/0.853/0.946 → +0.016 AUROC). All other US1/US2/US3 frame-window variants underperform their backbone baselines. Child-adapted WavLM (US2) collapses to AUROC=0.500 (random). New seg-MIL aggregators (US5/US6) marginally improve over gated_attention: ExpSoftmaxPool +0.008 AUROC, DSMIL +0.007, AutoPool +0.005; GMAP regresses (−0.015). HuBERT-large layersum is a useful new model variant (AUROC=0.813, +0.042 vs WavLM-MIL but still below Whisper-MIL). Cross-child TS-MIL (`wavlm_mil_tsmil_concat_cross_child`) intentionally skipped — BabAR/VTC env libtorchcodec/FFmpeg conflict prevents cross-child VTC RTTM cache rebuild; marked `attempt: 99` + `skipped_reason` in `mil/spec014_jobs.json`. Synthetic scene generation (resubmit job 12770080) completed 5000 scenes (manifest: `synth_results/manifests/synthetic_manifest.csv`).
- 014-mil-extensions-attention-and-layers: Added Python 3.11, `child-vocalizations` conda env (same as spec-009 / spec-012) + `torch`, `transformers` (WavLM/HuBERT/Whisper backbones), `numpy`, `pandas`, `scikit-learn` (metrics only); no new Python packages required for US1/US2; US3 introduces no new dependencies (ACMIL is pure PyTorch — clone reference impl from https://github.com/dazhangyu123/ACMIL but rewrite into `mil/mil_model.py` rather than vendoring the package).
- 009-synth-rir-noise: Added Python 3.11, `child-vocalizations` conda env + pandas, scikit-learn (LR, GBM via HistGradientBoosting), numpy, torch + torchaudio (sub-features C/D only), transformers (WavLM backbone for C/D)
- **TinyVox MIL augmentation negative result** (spec-009, 2026-04-28, job 12748294): Adding 15,550 TinyVox Providence clips (padded to 10s, label=1) to WavLM-MIL train split HURTS performance. Test AUROC=0.670 vs baseline 0.771 (delta=-0.101); F1=0.866 vs 0.882 (delta=-0.017); AUPRC=0.819 vs 0.893 (delta=-0.074). Early stopping at epoch 12. Root cause: TinyVox short clips padded with silence create uniform 0-energy windows; model overfits on pad-pattern as a positive signal → worse generalization on real clips. Results: `mil/mil_results/wavlm_mil_tinyvox/`.

## Active Technologies
- Python 3.11, `child-vocalizations` conda env (same as spec-009 / spec-012) + `torch`, `transformers` (WavLM/HuBERT/Whisper backbones), `numpy`, `pandas`, `scikit-learn` (metrics only); no new Python packages required for US1/US2; US3 introduces no new dependencies (ACMIL is pure PyTorch — clone reference impl from https://github.com/dazhangyu123/ACMIL but rewrite into `mil/mil_model.py` rather than vendoring the package). (014-mil-extensions-attention-and-layers)
- New result directories under `mil/mil_results/`: `wavlm_mil_layersum/`, `whisper_mil_layersum/`, `hubert_large_mil_layersum/`, `wavlm_mil_child_adapted/`, `wavlm_mil_acmil/`, `whisper_mil_acmil/` (and combined `wavlm_mil_child_adapted_layersum/` if FR-010 triggers). Each follows the existing MIL output schema: `best_checkpoint.pt`, `config.json`, `val/test_metrics_tuned.json`, `val/test_predictions.csv`, `val/test_metrics_by_timepoint.csv`. New artifacts: `layer_weights.json` (US1), `branch_weights.json` (US3). (014-mil-extensions-attention-and-layers)
- ACMIL branch-aggregation retrain dirs (2026-04-29): `mil/mil_results/{wavlm,whisper}_mil_acmil_{max,gated,topk}/`. Same output schema as parent ACMIL dir, plus `branch_diagnostics_test.json`, `branch_attention_test.csv`, `branch_weights_test.json`. Branch-selection eval (no retrain) writes `branch_selection.{csv,json}` into `mil/mil_results/{wavlm,whisper}_mil_acmil/` via `mil/eval_acmil_branch_selection.py`. (014-mil-extensions-attention-and-layers)
