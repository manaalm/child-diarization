# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Child-adult speaker diarization system that classifies speakers (silence, child, adult, overlap) in audio recordings at 20ms frame-level resolution. Based on ICASSP 2025 / Interspeech 2024 papers by Xu et al.

The goal is per-clip child presence detection: given a short audio clip, predict whether a target child is vocalizing. Nine diarization frontends are compared:
1. **USC-SAIL** — Fine-tuned Whisper + LoRA frame classifier (`whisper-modeling/`)
2. **Pyannote** — `pyannote/speaker-diarization-community-1` model
3. **BabAR** — VTC 2.0 child speech diarizer (full pipeline with phoneme step)
4. **VTC** — VTC 2.0 standalone (no BabAR phoneme step); two variants: `vtc` (KCHI+OCH) and `vtc_kchi` (KCHI only)
5. **VBx** — Variational Bayes HMM speaker diarization using pyannote VAD + ECAPA embeddings; anonymous speaker labels resolved via cosine similarity to target-child prototype
6. **TalkNet-ASD** — Video-audio active speaker detection (SAILS BIDS .mp4 only); child identified as smallest face track
7. **TS-TalkNet** — Speaker-conditioned video-audio ASD; uses a reference clip from the training split for target-child enrollment
8. **EEND-EDA** — End-to-End Neural Diarization with Encoder-Decoder Attractors (ESPnet2); handles overlapping speech natively; anonymous speaker labels resolved via ECAPA cosine similarity
9. **Sortformer** — Sort-based transformer diarization (NeMo/NVIDIA); anonymous speaker labels resolved via ECAPA cosine similarity

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

**Note**: `unified.py` is partially redundant with `whisper-modeling/usc_sail_run_enrollment.py` — USC-SAIL enrollment logic exists in both places.

### `mil/` — Multiple Instance Learning module

**`mil_model.py`** — `BackboneExtractor` (frozen WavLM-base+ or Whisper-small) + `GatedABMILHead` (gated attention MIL, Ilse et al. 2018) + `MILModel` composer. Used by the frame-window MIL workflow.

**`mil_train.py`** / **`mil_dataset.py`** — Frame-window MIL: splits audio into 2s windows, embeds each window, trains GatedABMIL head over the bag of windows.

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
- `mil/mil_results/seg_mil/` — Segment-instance MIL sweep results (16 configs); `all_configs.json` summary + per-config subdirs

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

## Recent Changes
- 007-av-extensions: Added `train_cascaded_pipeline.py` (3-stage cascade with val-tuned thresholds), `smooth_predictions.py` (Gaussian/majority-vote/moving-average temporal smoothing), `extract_gpt4o_features.py` (GPT-4o-mini vision API frame analysis with caching), `ego4d_experiment.py` (zero-shot ASD evaluation), `1kd_integration.py` (schema compatibility check); extended `extract_asd_features.py` with `--model {loconet,light_asd}` and `evaluate_av_fusion.py` with `--cascade-breakdown` and `--smoothed-predictions`; added `av_extensions.yaml` config
- 005-mil-extensions: Added Python 3.11 (conda `child-vocalizations` env — same as existing AV pipeline) + scikit-learn (threshold tuning, logistic regression), pandas, numpy, scipy (Gaussian smoothing), OpenCV (frame sampling), openai (GPT-4o API), tqdm; LocoNet and Light-ASD require the `video/` Python 3.10 uv env as subprocess targets
- 005-mil-extensions: Added Python 3.11 (conda `child-vocalizations` env — same as MIL sweep) + `opencv-python` (YuNet face detector), `xgboost`, `scikit-learn`, `pandas`, `numpy`, `matplotlib`, `seaborn`, `scipy`; optional: `mediapipe` (fallback detector); existing `video/` env for ASD stretch
- 005-mil-extensions: Added Python 3.11 (conda `child-vocalizations`) + PyTorch 2.x, transformers (WavLM-base+), torchaudio, scikit-learn, pandas, PyYAML, numpy, soundfile, scipy (for Pearson/Spearman correlation in eval script)

## Active Technologies
- Python 3.11 (conda `child-vocalizations` env — same as existing AV pipeline) + scikit-learn (threshold tuning, logistic regression), pandas, numpy, scipy (Gaussian smoothing), OpenCV (frame sampling), openai (GPT-4o API), tqdm; LocoNet and Light-ASD require the `video/` Python 3.10 uv env as subprocess targets (005-mil-extensions, 007-av-extensions)
- CSV files for features and predictions; JSON for metrics/configs/cache; `.pkl` for trained models; `av_fusion/gpt4o_cache/` for raw API responses; `av_fusion/face_track_cache/` shared with 006 (005-mil-extensions, 007-av-extensions)
