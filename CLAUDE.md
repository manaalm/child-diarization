# CLAUDE.md

Guidance for Claude Code working in this repo.

## Project Overview

Per-clip child presence detection: given a short audio clip, predict whether a target child is vocalizing. Synthetic scene generator (`synth/`) mixes child speech (Providence + TinyVox + Playlogue) and adult speech (Providence parents + LibriSpeech + Playlogue) under configurable SNR, RIR, overlap. v2 corpus is canonical (see "Synth corpus v1 vs v2" gotcha). Frontends compared:

1. **USC-SAIL** — Whisper + LoRA frame classifier (`whisper-modeling/`)
2. **Pyannote** — `pyannote/speaker-diarization-community-1`
3. **BabAR** — VTC 2.0 child diarizer (full pipeline w/ phoneme step)
4. **VTC** — VTC 2.0 standalone; `vtc` (KCHI+OCH) and `vtc_kchi` (KCHI only)
5. **VBx** — VB-HMM diarization (pyannote VAD + ECAPA); cluster→child via cosine
6. **TalkNet-ASD** / **Fine-tuned TalkNet** — Video ASD (SAILS BIDS .mp4 only); `video/talknet_child_finetune.py`
7. **EEND-EDA** — End-to-end neural diar with attractors (ESPnet2)
8. **Sortformer** — Sort-based transformer diar (NeMo)
9. **Audio LLM** — Qwen2.5-Omni-7B thinker zero/few-shot (`baselines/audio_llm_baseline.py`); v2 headline; v1 Qwen2-Audio preserved at `qwen2_audio_7b/`

All evaluated via shared ECAPA enrollment. Primary eval scripts live in `pyannote/` (multi-diarizer testing suite, despite the name).

---

## Environment Setup

Each subsystem has its own env — do not mix:
- USC-SAIL/Whisper: `cd whisper-modeling && pip install -r requirements.txt`
- BabAR: separate venv (see BabAR/README.md)
- Pyannote: install pyannote.audio separately
- Video ASD (TalkNet, TS-TalkNet, LocoNet, Light-ASD): `cd video && uv sync` (Python 3.10); clone repos & checkpoints per `video/SETUP.md` (gitignored)
- EEND-EDA: `pip install espnet espnet_model_zoo soundfile` (in `child-vocalizations`)
- Sortformer: `pip install nemo_toolkit[asr]`
- GPT-4o features: `pip install openai`; needs `OPENAI_API_KEY`
- VBx: `cd VBx && uv sync`; needs HF_TOKEN
- VTC standalone: `cd BabAR/VTC && uv sync`; checkpoint at `VTC/VTC-2.0/model/best.ckpt`

All audio assumed 16kHz mono (auto-resample in `dataset_classes/preprocess.py`).

---

## Key Commands

### USC-SAIL Whisper
```bash
# Train (remove pdb.set_trace() in scripts/main.py first)
cd whisper-modeling && PYTHONPATH=. python scripts/main.py --debug f --config configs/config.yaml
python scripts/infer_wav_file.py --wav_file /path/to/audio.wav
# Batch SLURM: sbatch --array=0-155%5 run_usc_playlogue.sh
```

### Unified enrollment (all diarizers)
```bash
cd pyannote
python unified.py --diarizer {usc_sail|pyannote|babar|vtc|vtc_kchi|vbx|talknet_asd|ts_talknet|eend_eda|sortformer}
# Or: sbatch run_{eend_eda,sortformer}_enrollment.sh
# Output: {diarizer}_ecapa_enrollment_runs/
```

### Frame-level eval
```bash
cd pyannote && python unified_rttm.py --diarizer <name>        # RTTM accuracy
python evaluation/frame_localization_gt.py                      # 12 systems × 2-3 datasets
python evaluation/onset_tolerance_f1.py                         # 100/250/500/1000 ms tolerances
python synth/scripts/build_synth_holdout_eval.py                # one-time, seed=43
sbatch pyannote/run_synth_holdout_eval.sh <diarizer>            # auto-discovered by metric scripts
```

### BabAR combined-feature models
```bash
cd pyannote && python babar_three.py     # 8 feature combos × LR + GBM
# Output: babar_combined_runs/all_model_results.json
```

### AV fusion (manual-only MVP, no GPU)
```bash
python av_fusion/scripts/build_av_feature_table.py --metadata-csv whisper-modeling/seen_child_splits/master_with_split.csv \
  --audio-scores-val babar_ecapa_enrollment_runs/enroll_val_predictions.csv \
  --audio-scores-test babar_ecapa_enrollment_runs/enroll_test_predictions.csv \
  --audio-score-col prob --output-dir av_fusion/av_results/manual_only/ --run-name manual_only
python av_fusion/scripts/train_av_fusion.py --feature-dir <d> --output-dir <d>/models/ --config av_fusion/configs/av_fusion.yaml --seed 42
python av_fusion/scripts/evaluate_av_fusion.py --feature-dir <d> --model-dir <d>/models/ --output-dir <d> --plot
python av_fusion/scripts/error_analysis_av.py --predictions-csv <d>/predictions_test.csv --feature-dir <d> --output-dir <d>
# Auto visual features: sbatch av_fusion/slurm/run_av_pipeline.sh (48h GPU)
# 007 extensions: train_cascaded_pipeline.py, smooth_predictions.py, extract_gpt4o_features.py,
#   extract_asd_features.py (--model {talknet|loconet|light_asd}), 1kd_integration.py
# Fine-tune TalkNet for child voc: sbatch video/slurm/run_talknet_finetune.sh
```

### Audio LLM baseline
```bash
# Dry run: python baselines/audio_llm_baseline.py --split val --max-clips 5 --dry-run
sbatch baselines/slurm/run_audio_llm_baseline.sh val          # ~4h GPU
sbatch baselines/slurm/run_audio_llm_baseline.sh test         # exits 2 if val_metrics_tuned.json missing
# Few-shot: sbatch run_audio_llm_baseline.sh {val|test} qwen25_omni_7b_2shot 2
# 4th arg = prompt_template (default zero_shot_v1; alt target_child_v1)
# Universal synth demos: --universal-shots --train-csv synth_results/manifests/synthetic_audio_llm_shots.csv
```

### Metadata routing/ensemble (spec-012)
```bash
python evaluation/metadata_router.py --verify         # check 10 system files load
python evaluation/metadata_router.py --mode {stack|router|all}      # CPU, ~1 min
sbatch evaluation/slurm/run_multi_child_suppressor.sh  # US3, GPU ~30 min
sbatch evaluation/slurm/run_short_voc_head.sh          # US4, GPU ~4h
```

### Pseudo-frame self-distillation
```bash
python pseudo_frame/build_pseudo_labels.py             # ~3 min
sbatch pseudo_frame/slurm/train_pseudo.sh              # ~5 min
# Output: pseudo_frame/results/wavlm_pseudo_frame/
```

### AV self-distill & visual-eligibility (spec-015)
```bash
python pseudo_frame/visual_eligibility.py              # CPU ~3 min
python evaluation/metadata_router.py --mode stack --visual-features pseudo_frame/visual_features/visual_eligibility.csv
python evaluation/metadata_stack_av_ablation.py
sbatch pseudo_frame/slurm/extract_mouth_motion.sh      # CPU 4-6h
python pseudo_frame/avhubert_late_fusion.py            # US2 (hand-engineered, AV-HuBERT install blocked)
python pseudo_frame/speaker_informed_asd.py            # US3 Clarke 2025
python pseudo_frame/audio2video_distill.py             # US4
```

### Synthetic scene generation (`synth/`)
```bash
# Step 1: build manifest (REQUIRES --exclude-speakers-csv = real test split, else leakage)
python synth/scripts/build_segment_manifest.py \
  --providence-dir providence/ --providence-rttm-dir providence/rttm/ \
  --tinyvox-dir data/tinyvox/ \
  --playlogue-dir playlogue/audio/ --playlogue-rttm-dir playlogue/rttm/ \
  --librispeech-dir data/LibriSpeech/LibriSpeech/train-clean-100/ \
  --exclude-speakers-csv whisper-modeling/seen_child_splits/test.csv \
  --output synth_results/manifests/segment_manifest_v2.csv
# v2: ~295k segments. RIR/MUSAN paths baked into synth/configs/default_14_18mo.yaml.
python synth/scripts/extract_segments.py --manifest <m> --output-dir data/segments/
sbatch synth/slurm/run_scene_generation_v2.sh synth/configs/default_14_18mo.yaml   # CPU
python synth/scripts/generate_training_sets.py --real-train-csv whisper-modeling/seen_child_splits/train.csv \
  --synthetic-manifest synth_results/manifests/synthetic_manifest.csv --output-dir synth_results/manifests/
sbatch synth/slurm/run_ratio_sweep.sh synth/configs/default_14_18mo.yaml           # GPU 48h
python synth/scripts/error_analysis_synthetic.py --experiment-dir <d> --test-csv whisper-modeling/seen_child_splits/test.csv --output-dir <d>
```

### Synth v4: empirical turn-taking + childrenization (lit-review extensions)
```bash
# 1. Empirical TT fit (literature priors used by default; flip empirical_path + sampling_mode=bootstrap to use JSON).
python synth/scripts/fit_empirical_turn_taking.py \
  --providence-rttm-dir providence/rttm --playlogue-rttm-dir playlogue/rttm \
  --playlogue-manifest playlogue/manifest.csv --age-bands 14_18 34_38 \
  --output synth_results/manifests/empirical_turn_taking.json \
  --write-config-stub synth/configs/empirical_turn_taking_stub.yaml
# 2-4. Voice augmentation:
sbatch synth/slurm/run_world_childrenization.sh                  # WORLD vocoder; CPU array, 8 shards
sbatch synth/slurm/run_cleese_childrenization.sh                 # CLEESE phase-vocoder (no spectral warp); CPU array, 8 shards
sbatch synth/slurm/run_cross_lingual_tinyvox_vc.sh               # kNN-VC (Zhang 2024); GPU array, 4 shards
# 5. Adultification eval (validation only; output: synth_results/adultification_eval/<tag>_<band>mo/).
sbatch synth/slurm/run_adultification_eval.sh 14_18 v3_perturb 600
# 6. End-to-end v4 build (after 2-4 produce manifests).
sbatch synth/slurm/run_v4_pipeline.sh                            # CPU; uses synth/configs/v4_14_18mo.yaml
```

### Child-adapted WavLM pretrain (spec-009 US3)
```bash
find data/tinyvox/audio -name "phon_Eng-NA_*.wav" > synth_results/child_wavs.txt
find data/segments/child -name "*.wav" >> synth_results/child_wavs.txt
sbatch synth/slurm/run_wavlm_pretrain.sh                # 48h GPU; resumes
# Then point mil/configs/wavlm_mil_child_adapted.yaml backbone_path → step_50000
```

### Frame-window MIL & hard-negative MIL
```bash
sbatch mil/slurm/train_mil.sh mil/configs/{wavlm_mil|whisper_mil}.yaml
sbatch mil/slurm/eval_mil.sh
python mil/mil_age_stratified.py --checkpoint <pt> --config <json> --age-group {14_month|36_month} --manifest playlogue/manifest.csv
# Hard-neg variant (1:1 pos:neg via Playlogue/Providence RTTM mining):
python mil/scripts/extract_hard_negatives.py --output synth_results/manifests/hard_negatives_manifest.csv --window-sec 30 --stride-sec 15 --min-activity-sec 3 --max-per-file 20 --seed 42
sbatch mil/slurm/train_mil_hardneg.sh
```

### Segment-instance MIL sweep (4 frontends × 7 aggregators)
```bash
python mil/seg_train.py --config mil/configs/seg_mil_sweep.yaml --precompute-only   # ~1-2h GPU
sbatch mil/slurm/seg_mil_sweep.sh                                                   # resume-safe
python mil/eval_weak_diarization.py --results-dir mil/mil_results/seg_mil --split-csv whisper-modeling/seen_child_splits/test.csv --rttm-cache whisper-modeling/usc_sail_rttm_cache --output mil/mil_results/seg_mil/weak_diarization_eval.csv
```

### Synth augmentation extensions (spec-016)
```bash
python synth/scripts/build_synth_aug_manifests.py
python synth/scripts/build_cross_child_synth_split.py     # C4
python synth/scripts/build_pseudo_synth_split.py          # C2/C5
python synth/scripts/build_seg_mil_synth_cache.py         # C5
python synth/scripts/build_usc_sail_synth_data.py         # C1
python pseudo_frame/build_synth_pseudo_labels.py          # C2
sbatch mil/slurm/train_eval_spec014.sh mil/configs/{wavlm,whisper}_mil_{hardneg,cross_child}_synth.yaml  # C3/C4
sbatch pseudo_frame/slurm/train_pseudo.sh pseudo_frame/configs/wavlm_pseudo_synth.yaml                    # C2
sbatch mil/slurm/seg_mil_synth.sh                                                                          # C5
sbatch baselines/slurm/run_audio_llm_synth_shots.sh {val|test}                                             # C6
sbatch whisper-modeling/run_train_synth.sh                                                                 # C1 (PYTHONPATH=., window_size 30)
# Tracker: mil/spec016_jobs.json
```

---

## Architecture

### `pyannote/` — Multi-diarizer testing suite
- **`unified.py`** — Abstract `DiarizationFrontend` base + 7 backends (USCSail/Pyannote/BabAR/VTC/VBx/TalkNetASD/TSTalkNet). Shared enrollment: ECAPA duration-weighted child prototypes → cosine similarity → val threshold tune → test eval.
- **`unified_rttm.py`** — Frame-level accuracy on Playlogue/Providence (RTTM → 10ms binary masks).
- **`video_asd.py`** — TalkNetASD/TSTalkNet subprocess bridge to `video/` env (Python 3.10); RTTM cache `video_asd_rttm_cache/{model}/`; face cache `video_face_cache/`.
- **`babar_three.py`** / `babar_updated.py` — Combined feature LR/GBM (8 combos × 2 classifiers): diarizer features + phoneme features + ECAPA cosine similarities. Requires prior BabAR RTTM + phoneme CSVs + ECAPA prototypes.
- **`unified_age_stratified.py`** — Per-cohort filter on `timepoint_norm` (14_month/36_month).
- **`augmentation_eval.py`** — Retrain prototypes on synth-augmented training; produce delta table.
- **`proxy_analysis.py`** — Quality proxy on unlabeled core data.
- Note: `unified.py` and `whisper-modeling/usc_sail_run_enrollment.py` overlap; `unified.py` is current.

### `mil/` — Multiple Instance Learning
- **`mil_model.py`** — `BackboneExtractor` (frozen WavLM-base+ or Whisper-small) + `GatedABMILHead` + `MILModel`.
- **`mil_train.py`** / `mil_dataset.py` — Frame-window: 2s windows, embed each, train GatedABMIL.
- **`mil_evaluate.py`** — Loads checkpoint + val threshold; writes `test_metrics_tuned.json`, `test_predictions.csv`, `test_metrics_by_timepoint.csv`.
- **`mil_age_stratified.py`** — Cohort eval; outputs to `age_stratified/{group}/`.
- **`mil_utils.py`** — `compute_metrics()`, `tune_threshold()`, `per_timepoint_metrics()`, `save_json/csv()`.
- **Segment-instance MIL** (`seg_*.py`): per-segment WavLM embedding cache (MD5 keyed `.npy`); `SegmentBagDataset` → bag of segment embeddings; 7 aggregators (mean/max/attn/gated_attn/noisy_or/topk/transformer); sweep config `seg_mil_sweep.yaml`.
- **`eval_weak_diarization.py`** — Reads attention CSVs + RTTM GT → Pearson/Spearman/AUROC, age-stratified.
- **ACMIL extension**: `branch_aggregation: mean|max|topk_mean|gated` + `forward_branches(h)` for no-retrain inference. Branch-selection eval: `mil/eval_acmil_branch_selection.py`.

### `whisper-modeling/` — USC-SAIL
- **Model**: WhisperWrapper freezes backbone + LoRA (rank=8) on encoder fc1/fc2; head → 4 classes (silence/child/adult/overlap) at 20ms.
- **Train**: PyTorch Lightning + NLLLoss; 10s windows / 5s stride; LR 0.001; bs 64; max 20 epochs.
- **Inference post**: majority filter (3-frame) → merge ≤200ms gap → drop <50ms.
- **Pretrained ckpt**: `whisper-base_rank8_pretrained_50k.pt` must be in `whisper-modeling/`.

### `baselines/` — Encoder baselines
3 encoders (Whisper/WavLM/Fused) × 2 poolings (mean/attention) → linear classifier. Cross-child default; `--seen-child` reuses `whisper-modeling/seen_child_splits/`. `--all-experiments` runs 13 variants. SLURM: `run_baseline_seen_child.sh`.

### `av_fusion/` — Audio-Visual fusion
Late fusion: `final_prob = α·audio_prob + (1-α)·visual_prob` for visually eligible clips; audio-only fallback. Train uses visual only (audio scores leak-only on val/test). Modules: `face_utils.py` (YuNetDetector, IouCentroidTracker, eligibility scoring), `extract_visual_features.py`, `extract_asd_features.py` (subprocess to `video/run_asd.py`), `build_av_feature_table.py`, `train_av_fusion.py` (AudioOnly + VisualXGB + GatedAV pkls), `evaluate_av_fusion.py`, `error_analysis_av.py`. **MVP path**: manual BIDS annotations only — `Video_Quality_Child_Face_Visibility/Lighting/Resolution`, `Child_of_interest_clear`, `#_adults`, `#_children` already in splits CSVs; eligibility falls back to `0.6·face_visibility_norm + 0.4·quality_norm`.

---

## Data Splits

| Location | Strategy | Size | Used by |
|---|---|---|---|
| `whisper-modeling/seen_child_splits/` | Within-child (109 children, 60/20/20) | 2183 | All enrollment runs, combined feature, MIL |
| `baselines/splits/` | Cross-child (97/21/21 disjoint) | 2377 | Baseline encoders default |
| seen_child_splits via `--seen-child` | Within-child | 2183 | Baselines on seen-child split |

`make_seen_child_split.py` reads `/orcd/scratch/bcs/001/sensein/sails/BIDS_data/anotated_processed.csv`, filters ≥5 clips/child/timepoint, stratifies seed=42.

---

## Results Storage

### Enrollment / model dirs
- `whisper-modeling/usc_sail_enrollment_runs/`, `pyannote/pyannote_enrollment_runs/`
- `babar_ecapa_enrollment_runs/`, `babar_combined_runs/`, `vtc_ecapa_enrollment_runs/`, `vtc_kchi_ecapa_enrollment_runs/`, `vbx_ecapa_enrollment_runs/`
- `video_asd_ecapa_enrollment_runs/{talknet_asd,ts_talknet}/`, `video_finetuned_talknet_runs/`
- `mil/mil_results/{wavlm_mil,whisper_mil,seg_mil/...}/` — `best_checkpoint.pt`, `config.json`, `val/test_metrics_tuned.json`, `val/test_predictions.csv`, `test_metrics_by_timepoint.csv`; age-stratified subdirs
- `pseudo_frame/results/wavlm_pseudo_frame/` and variants
- `baselines/audio_llm_baseline_runs/{model_slug}/` — `qwen25_omni_7b/` (v2 headline), `qwen2_audio_7b/` (v1)
- `synth_results/manifests/`, `synth_results/augmentation_experiments/{config}/`
- spec-016 result dirs: `mil/mil_results/{wavlm,whisper}_mil_{hardneg,cross_child}_synth/`, `mil/mil_results/seg_mil_synth/usc_sail_synth_combined_{gated_attention,transformer}/`, `pseudo_frame/results/wavlm_pseudo_frame_synth/`, `whisper-modeling/checkpoints/whisper_base_synth/`. Tracker: `mil/spec016_jobs.json`.
- spec-012 dirs: `ensemble_runs/{metadata_stack,metadata_router_rule,metadata_router_learned}/`; `mil/mil_results/{multi_child_suppressor,short_voc_head}/`

Each contains: `config.json`, `child_prototype_stats.csv`, `role_only_*` (duration baseline), `enroll_*`/`test_*` (embedding results).

### Headline test metrics (seen-child split)
*(Diarization rows resynced 2026-05-03 from each system's `enroll_test_metrics.json` — treat JSON as authoritative.)*

| System | F1 | AUROC | AUPRC |
|---|---|---|---|
| USC-SAIL | 0.872 | 0.658 | 0.813 |
| Pyannote | 0.849 | 0.678 | 0.830 |
| BabAR | 0.871 | 0.826 | 0.923 |
| VTC (KCHI+OCH) | 0.885 | 0.813 | 0.914 |
| VTC-KCHI | 0.871 | 0.826 | 0.923 |
| VBx | 0.856 | 0.675 | 0.841 |
| TalkNet-ASD | 0.279 | 0.568 | 0.786 |
| EEND-EDA | 0.841 | 0.521 | 0.767 |
| Sortformer | 0.824 | 0.691 | 0.852 |
| WavLM-MIL | 0.882 | 0.771 | 0.893 |
| Whisper-MIL | 0.886 | 0.853 | 0.946 |
| **Whisper-medium-MIL (backbone sweep)** | **0.904** | **0.873** | **0.951** |
| **Whisper-MIL TS-MIL concat (spec-014)** | **0.896** | **0.869** | **0.944** |
| HuBERT-large MIL layersum | 0.878 | 0.813 | 0.920 |
| **Whisper-MIL ACMIL max** | **0.891** | **0.842** | **0.936** |
| WavLM-MIL ACMIL topk(k=2) | 0.884 | 0.775 | 0.902 |
| Audio LLM Qwen2-Audio-7B 0-shot (v1) | 0.871 | 0.725 | 0.853 |
| **Audio LLM Qwen2.5-Omni-7B 0-shot (v2 headline)** | **0.874** | **0.770** | **0.900** |
| Audio LLM Qwen2.5-Omni 2-shot synth demos (spec-016 C6 v2) | 0.874 | 0.779 | 0.899 |
| **Audio LLM Qwen3-Omni-30B-A3B-Thinking 0-shot (spec-022 US3, 2026-05-12)** | **0.870** | **0.786** | **0.908** |
| Audio LLM Qwen3-Omni-30B-A3B-Thinking 0-shot (all-children-coverage, n=3314) | 0.874 | 0.799 | 0.904 |
| YAMNet 0-shot (spec-022 US3, AudioSet child-voc aggregation) | 0.588 | 0.766 | 0.899 |
| YAMNet 0-shot (all-children-coverage, n=3314) | 0.626 | 0.807 | 0.910 |
| AST (MIT/ast-finetuned-audioset-10-10-0.4593) 0-shot | 0.745 | 0.690 | 0.869 |
| AST 0-shot (all-children-coverage, n=3314) | 0.772 | 0.740 | 0.880 |
| Pseudo-frame WavLM (synth-aug v2) | 0.867 | 0.806 | 0.926 |
| **Whisper-MIL hardneg synth-aug v2 (spec-016 C3)** | **0.894** | **0.854** | **0.944** |
| Whisper-MIL hardneg synth v4 (spec-018, lit-ext, 2026-05-06) | 0.879 | 0.839 | 0.941 |
| Voice-transfer LR (spec-016 #1, NEUTRAL) | 0.871 | 0.750 | 0.893 |
| **Ensemble (best_audio_mil mean)** | **0.893** | **0.878** | **0.956** |
| **Metadata stacker (spec-012 US1)** | **0.905** | **0.904** | **0.966** |

NEGATIVE/null rows pruned — see git log + `evaluation/recomputed_metrics.csv` for full audit (Granite-Speech, Cohere-Transcribe ASR, Parakeet, AV-HuBERT-Large LR, KNN-VC, ACMIL mean, child-adapted WavLM, fine-tuned TalkNet, LocoNet-ECAPA all at/below trivial-predict-all F1=0.864 floor).

**spec-012**: `ensemble_runs/metadata_router_{rule,learned}/` F1=0.883/0.873, AUROC=0.705/0.731.
**spec-018 Phase B cross-child (held-out, n=496)**: Whisper-MIL cross-child synth v3 F1=0.861, AUROC=0.736, AUPRC=0.884 (+0.031 vs v2 0.705). Hardneg lane stays at v2 (v3 NEGATIVE −0.010).
**v4 corpus (lit-review extensions, 2000 scenes, 2026-05-06)**: Whisper-MIL **cross-child synth v4** F1=0.880, **AUROC=0.779**, AUPRC=0.907 — new cross-child lane high (**+0.043 vs v3, +0.074 vs v2**); biggest single-shot lift in spec-018, on 40% the corpus size. **Hardneg synth v4** F1=0.879, AUROC=0.839, AUPRC=0.941 — −0.005 vs v3 (lane stays at v2; voice-saturation prior holds). v3-perturb and v4-lit-review gains stack additively on cross-child (different axes). Dirs: `mil/mil_results/whisper_mil_{hardneg,cross_child}_synth_v4/`. Detail in megadoc §29.8.

**Balanced-accuracy ranking on seen-child test, sorted DESC (spec-022 US2, 2026-05-12)**. Full table at `evaluation/balanced_metrics_summary.csv` (315 systems × extended metric set; trivial-floor F1=0.864, trivial-floor balanced_accuracy=0.5). Top by balanced_accuracy:
- **Whisper-medium-MIL 0.773** (F1 0.904, AUROC 0.873)
- **Whisper-MIL-ACMIL-max 0.737** (F1 0.891, AUROC 0.842)
- **Whisper-MIL 0.735** (F1 0.886, AUROC 0.853)
- whisper_attn (encoder baseline) 0.724 (F1 0.882, AUROC 0.850)
- wavlm_attn 0.662, fused_attn 0.648, Qwen2.5-Omni 0.646, pyannote 0.638
- WavLM-MIL 0.619, Qwen2-Audio 0.615, VTC/VTC-KCHI/BabAR ≈0.60, VBx 0.590
- USC-SAIL 0.553, **Whisper-pseudo-frame 0.552** (F1 0.873 — high F1 because recall≈0.99 at the val-tuned threshold; the model predicts positive on nearly every clip), WavLM-pseudo-frame 0.538, EEND-EDA 0.516, Sortformer 0.510, Joint ASR+diar 0.508
- 136 of 315 systems have balanced_accuracy < 0.6 — those rows are at or near the trivial-predict-all baseline.

**Implication for prior CLAUDE.md claims**: the "Whisper-pseudo-frame is the strongest single audio system" claim (AUROC 0.881) is true on AUROC but the chosen val-tuned threshold (0.45) pushes the model to a recall=0.99 regime where balanced accuracy collapses to 0.55. AUROC says "this model can rank pos vs neg well"; balanced accuracy says "at the chosen operating point it's not better than always-positive". Both are true facts about the same system.

**Within-child 3-fold AUROC — LEGACY (spec-022 US2 audit, 2026-05-12)** (sources: `evaluation/kfold_summary.csv` + per-system `*_kfold3_f{0,1,2}/test_metrics_tuned.json`). **VERDICT: NOT a cross-child generalisation estimate.** Every fold has the same 109 children in train ∩ val ∩ test (per `evaluation/kfold_audit.md` and the docstring of `whisper-modeling/make_kfold_seen_child_split.py` lines 9-11 — "preserves the within-child paradigm"). Variance reported here is clip-level shuffle variance, not held-out-child variance. Spec-022 US2 group-stratified 3-fold split is built at `whisper-modeling/seen_child_splits_groupstrat_3fold/` (130 children disjoint across folds, positive-rate gap 0.025 — within bootstrap noise); per-system retraining pending GPU dispatch. Numbers below remain published for reproducibility:
- **Whisper pseudo-frame 0.884±0.020 (K-FOLD LEADER, single-split 0.881, Δ=+0.003)**
- Whisper-medium-MIL 0.870±0.007, Fused Whisper+WavLM × Whisper-medium 0.861±0.035
- Whisper-MIL TS-MIL concat 0.859±0.019, Whisper-MIL 0.858±0.007
- Fused Whisper+WavLM × Whisper-large-v3 0.858±0.048 (single-split 0.907, Δ=−0.049)
- Whisper-MIL cross-child 0.853±0.012, BabAR/VTC-KCHI 0.838±0.011
- USC-SAIL 0.837±0.012 (single-split 0.658, Δ=+0.179 — pessimistic outlier)
- Fused × Whisper-small 0.823±0.051, VTC 0.811±0.011, wavlm pseudo-frame 0.796±0.036
- Whisper-MIL ACMIL max 0.736±0.061 (single-split 0.842, Δ=−0.106 — overfit)
- VBx 0.707±0.013, Pyannote 0.693±0.012, Sortformer 0.679±0.007, WavLM-MIL 0.637±0.017, EEND-EDA 0.543±0.026

**Per-timepoint posthoc** (spec-022 US5, 2026-05-12): per-timepoint stratification moved to a dedicated posthoc artefact at `evaluation/posthoc_per_timepoint_table.md` + `.csv` (299 systems with BIDS-corrected per-timepoint data; 85 systems flagged at |Δ AUROC 36m−14m| > 0.05, almost all with 36m > 14m). BabAR per-timepoint preserved as the canonical posthoc example (`babar_combined_runs/all_model_results.json`, `pertp_logistic_diarizer_plus_phoneme`): 14_month F1=0.864/AUROC=0.872/AUPRC=0.933; 36_month F1=0.897/AUROC=0.845/AUPRC=0.957; combined F1=0.882/AUROC=0.870/AUPRC=0.949. Headline tables above show combined-timepoint only (PI directive 2026-05-12: per-timepoint is posthoc, not headline).

### Caches
`whisper-modeling/usc_sail_{rttm,segment}_cache/`, `pyannote/{pyannote,vtc,vbx,video_asd}_rttm_cache/`, `pyannote/video_face_cache/`. Delete relevant cache if audio changes.

### Logs
SLURM output → `logs/adult/*.out`, `logs/seedlings/*.out`. **Highest-numbered .out for a given base name = most recent run**.

---

## Important Gotchas

- BabAR/Pyannote/VBx/VTC need **separate envs**; don't install into whisper-modeling env. VBx needs HF_TOKEN. VTC standalone ckpt at `VTC/VTC-2.0/model/best.ckpt`.
- `babar_three.py` is downstream — needs BabAR RTTM + phoneme CSVs + ECAPA prototypes preexisting.
- **Video files only exist for SAILS BIDS** — Providence/Playlogue audio-only; talknet_asd/ts_talknet return [] there.
- `video/` repos and checkpoints (`TalkNet-ASD/`, `TS-TalkNet/`, `LoCoNet_ASD/`, `Light-ASD/`, `pretrain/`) are gitignored.
- VBx RTTM accuracy on Providence: Micro F1=0.529, AUROC=0.515 (`pyannote/eval_results/vbx_providence/aggregate_metrics.json`).
- `extract_gpt4o_features.py`: needs `OPENAI_API_KEY`; gpt-4o-mini default; `--dry-run` for cost.
- `train_cascaded_pipeline.py`: needs `av_val.csv` from 006 pipeline; thresholds val-tuned only.
- `smooth_predictions.py`: smoothing scoped within (child_id, timepoint_norm) — no cross-child leakage.
- `synth/scripts/build_segment_manifest.py` **must** receive `--exclude-speakers-csv=test split` or test-child speech leaks into training.
- Synth scene WAVs (`synth_results/synthetic_scenes*/wav/`) and segments (`data/segments/`) gitignored. `generate_scenes.py` is CPU-only. Don't regenerate partial scene sets — always full N for given config+seed.
- **Audio LLM prompt cache invalidation**: if prompt template changes, delete `baselines/audio_llm_cache/{model_slug}/` first — old logits silently produce wrong results.
- **Audio LLM test-before-val guard**: `--split test` exits 2 if `val_metrics_tuned.json` missing.
- **USC-SAIL training requires `PYTHONPATH=.`** (else `ModuleNotFoundError: lightning_modules`); encoded in `whisper-modeling/run_train_synth.sh`.
- **USC-SAIL `window_size: 30` on transformers ≥4.57** — Whisper hard-checks mel length 3000. Synth scenes are 30s natively; set `batch_size: 16`. Old 5k pretrained ckpt has positional emb [500,512] sized for window_size=10 → state-dict mismatch when forced to 30. Workaround: chunk 30s into 3×10s sub-clips (not implemented).
- **WhisperWrapper API drift on transformers ≥4.57** (`whisper-modeling/models/whisper.py`): WhisperAttention `__init__` needs `config=` arg; `forward` returns 2-tuple. Both fixed.
- **Granite-Speech needs `<|audio|>` placeholder** in prompt — `score_granite_llm` injects automatically. Even fixed, 1B model is near-random (capability ceiling).
- **Audio model error fallback poisons cache**: `audio_model_baseline.py` writes 0.5 on per-clip exception → AUROC=0.5 forever. After fixing, **delete cache** before resubmitting: `rm baselines/audio_model_cache/{model_slug}{_cross_child}/{val,test}_scores.json`.
- **transformers ≥4.57 has_file() network bug — set `TRANSFORMERS_OFFLINE=1` and `HF_HUB_OFFLINE=1` in SLURM** (2026-04-30). Even cached models trigger network roundtrip; misparses on flaky compute nodes → misleading "not a local folder" error. Already in `pseudo_frame/slurm/train_pseudo.sh`, `mil/slurm/train_eval_spec014.sh`, `mil/slurm/seg_mil_synth.sh`, `baselines/slurm/run_audio_llm_synth_shots.sh`.
- **Synth corpus v1 lacked LibriSpeech and Playlogue** (2026-04-30). v1 SLURM didn't pass `--librispeech-dir`; script had no `--playlogue-dir`. All v1 adult speech = `providence_adults`. v2 (`synthetic_scenes_v2/`, `segment_manifest_v2.csv` 294,745 segs) corrects this. Verify any new corpus by inspecting `source_dataset` in scene JSONs. v2 is canonical.
- **Canary-Qwen-2.5b NeMo↔HF mismatch**: HF upload lacks `model_config.yaml` that NeMo's loader expects. Currently blocked.
- **Qwen2.5-Omni `AutoProcessor` requires torchvision** (2026-05-03) — even for audio-only. Fix: `pip install --no-deps torchvision==0.23.0` matched to torch 2.8.0+cu128. `--no-deps` mandatory.
- **Inode quota cleanup recipe** (2026-05-03): scratch capped at 1M files. When `OSError: [Errno 122]` on small-file writes, quickest unblock is `rm -rf data/segments/` (161k regenerable, ~16G, gitignored, recreates in ~3 min CPU). Other regenerables: `mil/seg_embedding_cache{,_synth}/` (81k/99k, GPU-heavy ~1-2h to recompute). DO NOT delete `data/{tinyvox,LibriSpeech,rir}/` (external, not regenerable).
- **Qwen2.5-Omni env-inherited `HF_TOKEN` triggers 401** (2026-05-03) — even with offline flags. Fix: `unset HF_TOKEN HUGGINGFACE_HUB_TOKEN HF_HUB_TOKEN` at top of any SLURM script that loads a public model. Already in `baselines/slurm/run_audio_llm_{baseline,cross_child,synth_shots}.sh`.
- **Empirical RTTM turn-taking ≠ hand-set Gaussian priors** (2026-05-05). `fit_empirical_turn_taking.py` on Providence+Playlogue gives 14_18 mo child-turn mean 3.18s and pause 0.07s — vs hand-set 0.6s/0.8s. RTTM aggregates *utterances* longer than Casillas/Hilbrink conversational turns. Don't blend RTTM-fit with Casillas response-latency values — pick one regime. v4 default = literature priors; empirical opt-in via `sampling_mode: bootstrap`.
- **WORLD/CLEESE childrenization F0 sanity** (2026-05-05). On 5-segment smoke: WORLD F0 median 269 Hz (target 250); CLEESE F0 median 202 Hz (pitch shift capped at 12 semitones). Both peak-normalized. F0 wildly outside [180, 320] Hz → suspect `_estimate_mean_f0` fallback firing on noisy/silent input, not a vocoder bug.
- **Spreadsheet `timepoint` column ≠ BIDS session ID** (2026-05-12, spec-022 US1). `anotated_processed.csv` `timepoint` is missing for 855 of 3145 rows where BIDS `sub-XXX/ses-{01,02}/` resolves the visit unambiguously (ses-01=14_month, ses-02=36_month). Run `cd whisper-modeling && PYTHONPATH=. python make_seen_child_split.py --use-bids-timepoint` (default true) to use the BIDS-derived value. Legacy spreadsheet behaviour: `--no-bids-timepoint`. Mapping module: `whisper-modeling/bids_timepoint.py`. Always check whether a cached split CSV is pre- or post-correction — pre-correction has 2183 rows / 109 children (seen-child); post has 3145 / 130. **All split paradigms now BIDS-corrected** (2026-05-12 polish): `seen_child_splits/` (130/3145), `all_children_splits/test_all.csv` (151/3314), `seen_child_splits_groupstrat_3fold/` (130/3145, children disjoint, US2), `seen_child_splits_kfold_3fold_bids/` (130/3145, within-child paradigm on BIDS data — successor to legacy `seen_child_splits_kfold_3fold/`), `baselines/splits/` (151/3314, cross-child relaxed filter), `baselines/splits_kfold/` (151/3314, cross-child 3-fold). All legacy splits preserved at `*.legacy_pre_bids_022` (Constitution VI).

---

## Recent Headline Findings

(Detailed run-by-run notes live in git log. Below = load-bearing conclusions only.)

- **Audio-scene-analysis baselines: YAMNet + AST land at the imbalance-aware band ceiling (spec-022 US3, 2026-05-12)**: zero-shot YAMNet (TFHub) and AST (`MIT/ast-finetuned-audioset-10-10-0.4593`) via new `baselines/scene_analysis_baseline.py` with AudioSet child-vocalisation aggregation `p_child_voc = max(P[Child speech], P[Babbling], P[Baby cry], P[Children shouting])`. Run on both seen-child test (635 clips, 75.8% pos) and new universal-coverage `all_children_splits/test_all.csv` (3314 clips, 74.3% pos, 151 children — built via `make_seen_child_split.py --build-all-children-split`). **Seen-child test**: YAMNet F1=0.588 BA=0.644 AUROC=0.766; AST F1=0.745 BA=0.650 AUROC=0.690. **All-children coverage**: YAMNet F1=0.626 BA=0.681 AUROC=0.807; AST F1=0.772 BA=0.688 AUROC=0.740. Both **improve slightly** on the broader population (more diverse, slightly lower imbalance). Sits in same band as Qwen2.5-Omni zero-shot (BA=0.65) — confirms zero-shot audio classification ceiling without fine-tuning. AudioSet class mappings + caveats in `baselines/scene_analysis_runs/{yamnet,ast}/README.md`. Qwen3-Omni-30B-A3B-Thinking jobs queued (3 SLURM jobs, ~12-18 GPU-h on A100; weights ~60GB downloading on val job).
- **BIDS-derived timepoints recover +962 rows (spec-022 US1, 2026-05-12)**: switched `make_seen_child_split.py` from spreadsheet `timepoint` column to BIDS `ses-01`/`ses-02` parsing via new `whisper-modeling/bids_timepoint.py`. Net: seen-child split grew **2183→3145 rows / 109→130 children** (+962 / +21). 855 rows had spreadsheet-missing timepoints that BIDS resolves unambiguously; only **3 rows** had actual value disagreement (all 3: BIDS=36_month, spreadsheet=14_month — spreadsheet stale). Per-timepoint metrics on the legacy 441-row test set are nearly unchanged (sub-0.005 absolute on f1/auroc/auprc). **194 newly-included test rows have no cached predictions** — need GPU rerun (US2/US3). `mil/mil_utils.compute_metrics()` extended with `f1_macro`, `f1_weighted`, `balanced_accuracy` (spec-022 US2 FR-007); the imbalance gap is severe — e.g., whisper_pseudo_frame 14m balanced_accuracy=0.55 vs f1=0.83 (model predicts positive on ~99% of clips). 298 of 316 per-timepoint CSVs regenerated; legacy backups at `*.legacy_pre_bids_022`. Per-row provenance in `whisper-modeling/seen_child_splits/bids_correction_provenance.json` and `specs/022-pi-thesis-revisions/changelog.md`. **Within-child 3-fold k-fold numbers in the table below remain pre-correction** — pending US2 group-stratified rerun.
- **Cardinality-confound check, v3 vs v4 cross-child** (2026-05-07, spec-021 US2): subsampled v3 cross-child synth to 2000 scenes (stratified, seed=42) to match v4 budget; retrained Whisper-MIL identical hyperparams. v3-2k test AUROC=**0.629** (F1=0.854, AUPRC=0.823) — **−0.107 vs full v3 (0.736)**, −0.076 vs v2 (0.705). At matched budget, v4 (0.779) beats v3 by **+0.150 AUROC**. POSITIVE: v4 lift is real, not "more synth". (1) v3 STFT-warp needs ≥5000 scenes; at 2000 it regresses below v2. (2) v4 dominates v3 at any matched budget. (3) Spec-018 ships v4 cross-child as headline. Dir: `mil/mil_results/whisper_mil_cross_child_synth_v3_2k/`.
- **Synth v4 lit-review extensions: cross-child new high, hardneg null** (2026-05-06): Whisper-MIL on v4 (2000 scenes, WORLD + CLEESE + cross-lingual TinyVox VC + empirical-TT bootstrap w/ literature-prior fallback). **Cross-child** AUROC=0.779 (+0.043 vs v3, +0.074 vs v2) — biggest single-shot lift in spec-018, on 40% the corpus. **Hardneg** AUROC=0.839 (−0.005 vs v3, predicted null). v3-perturb and v4-lit-review stack additively on cross-child (different axes). Hardneg plateaued at v2 — voice-saturation prior holds.
- **Whisper-pseudo-frame is the strongest single audio system** (2026-05-05): pseudo-frame self-distill (frozen backbone + linear frame head + max-pool clip score, 50 Hz pseudo-labels = mean of VTC-KCHI ∩ USC-SAIL agreement) with `openai/whisper-small` (pads each crop to 30s, encodes, truncates to valid frame count). Test AUROC **0.881** (vs WavLM 0.831, +0.050); 14m AUROC **0.871** (+0.068); frame-Pearson 0.631 (+0.065). Beats Whisper-MIL 0.853 by +0.028. **3-fold k-fold AUROC = 0.884 ± 0.020**. Code: `pseudo_frame/pseudo_model.py`; config `pseudo_frame/configs/whisper_pseudo.yaml`. Whisper > WavLM transfers from clip-level to frame-level.
- **Joint ASR+diar on SAILS, NEGATIVE** (2026-05-05): `AlexXu811/child-adult-joint-asr-diarization`. (1) Zero-shot duration-fraction τ=0.0: F1=0.863 (trivial floor), AUROC=0.554. 14m AUROC=0.466 sub-random. 20% of 872 clips hit Whisper 300-token ceiling. (2) ECAPA enrollment: F1=0.752, AUROC=0.663, AUPRC=0.840 at τ=0.10 — middle band (Pyannote/VBx/Sortformer/USC-SAIL), well below BabAR/VTC-KCHI (0.826). Output: `joint_asr_diar_{sails_runs,ecapa_enrollment_runs}/`.
- **Lit-review-driven synth extensions implemented** (2026-05-05): (1) Empirical TT fit (`empirical_turn_taking.json`): 14_18 mo child-turn 3.18s, pause 0.07s, overlap 0.118; `TurnTakingSimulator` accepts `empirical_path` + `sampling_mode={gaussian,bootstrap}`. (2) `default_{14_18,34_38}mo.yaml` use Hilbrink/Casillas priors; empirical opt-in. (3) Adultification eval (n=600/set on v3): real_child P_child=0.616, real_adult 0.355, v3 synth 0.508 — Δ=0.108 from real child. (4) WORLD childrenization 32k WAVs. (5) CLEESE 32k WAVs. (6) Cross-lingual TinyVox kNN-VC 3,863 WAVs → 100 EN targets. v4 corpus = v2 + WORLD + CLEESE + xling-VC = **362,608 segments**.
- **Spec-018 Phase B (VTLP + speed perturb) MIXED-POSITIVE** (2026-05-05): v3 corpus (5000 scenes), per-segment STFT-warp VTLP α∈[0.9,1.1] + speed∈[0.9,1.1] at p=0.5. **Cross-child +0.031 AUROC** (0.705→0.736). **Hardneg −0.010** (0.854→0.844). Perturbation helps held-out, hurts shared. Ship v3 perturb cross-child only. Code: `synth/audio_utils.py:apply_{vtlp,speed_perturbation,segment_perturbation}`, `synth/configs/v3_perturb_14_18mo.yaml`.
- **Spec-019 Bark zero-shot probe NEGATIVE** (2026-05-05): Bark-small × 16 prompts × 2 settings. F0 OK (mean 241/282 Hz). Relevance probe via Whisper-MIL hardneg: **34% above 0.5** — below ≥60% bar. **4th NEGATIVE/NEUTRAL voice augmentation in a row** (after voice-transfer LR NEUTRAL, child-adapted WavLM AUROC=0.500, audio-level kNN-VC). Seen-child WavLM/Whisper pipeline is information-saturated wrt voice identity.
- **Multi-child suppressor (spec-012 US3) α-sweep confirms NULL** (2026-05-05): val-tuner picked α=1.0; manual α sweep shows MC F1 monotonically degrades 0.843→0.795 (overall AUROC 0.895→0.756). LR head on 283 MC train clips with frozen WavLM mean-pool adds no signal — adult-pretrained ECAPA/WavLM can't disambiguate target vs sibling at 14m. Artifact: `mil/mil_results/multi_child_suppressor/alpha_sweep.json`.
- **Audio LLM v2** (Qwen2.5-Omni-7B thinker, 2026-05-03): replaces Qwen2-Audio-7B. 0-shot AUROC 0.770 (+0.045 vs v1); 2/5-shot equivalent. Cross-child AUROC=0.820 > seen-child (no enrollment). **Spec-016 C6 universal synth shots v2** AUROC=0.779 (+0.066 vs v1, biggest v1→v2 audio-LLM lift). Target-child prompt reframing NEGATIVE (−0.060).
- **USC-SAIL Joint ASR+Diar baseline** (2026-05-01): F1=0.075 frame, AUROC=0.505 synth holdout — random. Whisper-hallucination-on-OOD, 16/200 clips truncate at max_len=300. Env: `joint_asr_diar` (Py3.10, transformers==4.45.2, torch==2.4.1, librosa==0.10, setuptools<81). **Model trained on Playlogue — DO NOT evaluate on Playlogue**. Providence onset-F1@250ms = 0.175.
- **Synth holdout localization eval** (2026-05-01, 8 systems): "30s clip cliff" — clustering diarizers (Pyannote/Sortformer/VBx/EEND-EDA) collapse on 30s; EEND-EDA/VBx emit zero child preds. BabAR leads synth (F1=0.295 @ 250ms), opposite Playlogue collapse — phoneme step destructive only on Playlogue MP3s. VTC most robust. Frame classifiers (VTC, USC-SAIL), not clustering, right for SAILS short-clip localization.
- **Frame-level localization + onset-F1** (2026-05-01): Playlogue — USC-SAIL F1=0.414 best (onset@250ms), Sortformer 0.411. Pseudo-frame WavLM 0.10-0.12 — weaker than diarizers it distills from. Earlier "frame Pearson 0.566" was circular. BabAR Playlogue F1=0.017. VBx ~0 both.
- **Imbalance-aware reanalysis + 3-fold CV** (2026-04-30): test 76% positive → trivial F1=0.864; many "negative" rows at floor. Top-system AUROC CIs overlap [0.82, 0.93] under cluster bootstrap → top-band rankings not statistically defensible. **wavlm_mil 0.771 was +0.13 single-split overestimate** (k-fold 0.637±0.017). **whisper_mil k-fold 0.858±0.007** validates as real winner. **PANNs cnn14_cross_child had leakage** (19/21 test children in train); clean retrain dropped 0.887→0.720. **BabAR ≡ VTC-KCHI on seen-child** (md5 identical) — phoneme step no-op for short BIDS. **VBx collapses to 1 speaker/file** in 100% Playlogue + Providence. CSVs: `evaluation/{recomputed_metrics,child_bootstrap_cis,kfold_summary,kfold_per_fold}.csv`.
- **Sortformer/EEND-EDA on Playlogue/Providence** (2026-04-30): Sortformer best Playlogue diarizer F1=0.565 (>USC-SAIL 0.493, Pyannote 0.482). USC-SAIL synth-only fails long-form (F1=0.031/0.038 — confirms spec-016 C1 NEGATIVE). Pseudo-frame underperforms on Providence.
- **HF token rotation**: old token scrubbed from working tree (was hardcoded in 7 .sh + 35 result JSONs). Scripts now use `: "${HF_TOKEN:?HF_TOKEN must be set}"`. Old token still in git history.
- **Spec-016 v2 corpus rerun** (2026-04-30): synth regen w/ Providence+TinyVox+LibriSpeech+Playlogue (5000 scenes, 18k subsampled segs). v1↔v2 AUROC deltas: C1 −0.010, C2 +0.043, C3 wavlm +0.049, **C3 whisper +0.031**, C4 wavlm −0.054, **C4 whisper +0.116 (biggest swing)**, C5 gated +0.023, C5 transformer −0.046, C6 unchanged. 5/9 improved, 3 regressed. C4 backbone divergence indicates v1's 100%-Providence-parents adult mix was a confound. **v1 results pruned 2026-05-03 from headline**.
- **Spec-016 follow-ups** (2026-04-29): (#8 C1 self-distill) NEGATIVE — synth-trained C1 produces conservative real preds (mean pos 0.047 vs GT ~25%); pseudo-frame student AUROC 0.831→0.690. (#1 voice transfer) NEUTRAL — feature-space mean-shift is shift-equivariant for LR; full XTTS/SPARC blocked by senselab/coqui-tts numpy ABI.
- **Spec-014 MIL extensions** (2026-04-29): only positive frame-window result is **Whisper-MIL TS-MIL concat F1=0.896 AUROC=0.869**. Child-adapted WavLM (US2) collapses to AUROC=0.500. Seg-MIL aggregator deltas vs gated_attn: ExpSoftmaxPool +0.008, DSMIL +0.007, AutoPool +0.005, GMAP −0.015. HuBERT-large layersum 0.813. Cross-child TS-MIL skipped — env libtorchcodec/FFmpeg conflict.
- **ACMIL branch-aggregation** (2026-04-29): `branch_aggregation: mean|max|topk_mean|gated` + `forward_branches(h)`. **Best: `whisper_mil_acmil_max` F1=0.891 AUROC=0.842** (+0.091 vs original mean NEGATIVE). `wavlm_mil_acmil_topk(k=2)` AUROC=0.775 (+0.042 vs wavlm mean baseline).
- **TinyVox MIL augmentation NEGATIVE** (spec-009, 2026-04-28): adding 15,550 padded TinyVox clips to WavLM-MIL train HURTS — AUROC 0.771→0.670. Padding silence creates uniform 0-energy windows; model overfits pad-pattern as positive. Results: `mil/mil_results/wavlm_mil_tinyvox/`.

## Active Technologies
- Python 3.11 `child-vocalizations` conda env (shared by spec-009/012/014/016/017/018/020/021) + torch 2.8.0+cu128, transformers ≥4.45 (set `TRANSFORMERS_OFFLINE=1` / `HF_HUB_OFFLINE=1` for ≥4.57 `has_file()` bug), pandas, scikit-learn, speechbrain (ECAPA), pyannote.audio (separate env), ESPnet (EEND-EDA env). Python 3.10 for `video/` ASD subprocess bridge and `joint_asr_diar` env.
- ACMIL is pure PyTorch (rewritten into `mil/mil_model.py`). Spec-014 dirs: `mil/mil_results/{wavlm,whisper,hubert_large}_mil_layersum/`, `wavlm_mil_child_adapted/`, `{wavlm,whisper}_mil_acmil{,_max,_gated,_topk}/` — standard MIL output schema + `layer_weights.json` (US1), `branch_weights.json`/`branch_diagnostics_test.json`/`branch_attention_test.csv` (US3).
- Spec-021 new deps: `fairseq` (vendored only for wav2vec2 LL_4300 conversion in US2 — converted artefact loads via HF), `mmpose`/`vitpose` (US3 body-cue stream), `crepe`/`world-vocoder` already present from synth v4. Conformal prediction (US7) uses `mapie` or split-conformal in <50 LOC.
- Filesystem-only — no database. Results under canonical dirs (`mil/mil_results/`, `evaluation/`, `av_fusion/av_results/`, `models/`, `synth_results/manifests/`). SLURM logs to `logs/adult/*.out`.
- Python 3.11 in the `child-vocalizations` conda env (per CLAUDE.md). Python 3.10 for `joint_asr_diar` env is not needed for this spec. + pandas, numpy, scikit-learn 1.7.2 (already provides `StratifiedGroupKFold`, `balanced_accuracy_score`, `f1_score(average='weighted')`); transformers ≥4.45 with `TRANSFORMERS_OFFLINE=1`/`HF_HUB_OFFLINE=1` env vars for ≥4.57 (per CLAUDE.md gotcha); torchaudio for waveform loading; matplotlib for the US4 encoder-pipeline figure; HuggingFace `transformers` for AST (`MIT/ast-finetuned-audioset-10-10-0.4593`) and Qwen 3.5-Omni; TFHub `tensorflow_hub` + `tensorflow` for YAMNet (in a sibling env to avoid TF↔PyTorch ABI conflicts). (021-post-thesis-future-work)
- filesystem-only. Result CSVs and JSONs under canonical dirs per CLAUDE.md; new artefacts under `evaluation/` (US2), `baselines/scene_analysis_runs/` (US3), `baselines/audio_llm_baseline_runs/qwen35_omni_7b/` (US3), `whisper-modeling/all_children_splits/` (US3), `docs/per_model_training_data.csv` (US4), and `specs/022-pi-thesis-revisions/{bids_vs_spreadsheet_diff.csv, changelog.md}` (US1). (021-post-thesis-future-work)

## Recent Changes
- 021-post-thesis-future-work: Added Python 3.11 in the `child-vocalizations` conda env (per CLAUDE.md). Python 3.10 for `joint_asr_diar` env is not needed for this spec. + pandas, numpy, scikit-learn 1.7.2 (already provides `StratifiedGroupKFold`, `balanced_accuracy_score`, `f1_score(average='weighted')`); transformers ≥4.45 with `TRANSFORMERS_OFFLINE=1`/`HF_HUB_OFFLINE=1` env vars for ≥4.57 (per CLAUDE.md gotcha); torchaudio for waveform loading; matplotlib for the US4 encoder-pipeline figure; HuggingFace `transformers` for AST (`MIT/ast-finetuned-audioset-10-10-0.4593`) and Qwen 3.5-Omni; TFHub `tensorflow_hub` + `tensorflow` for YAMNet (in a sibling env to avoid TF↔PyTorch ABI conflicts).
