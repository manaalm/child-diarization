# Tasks: Synthetic Child-Adult Scene Generator

**Input**: Design documents from `specs/008-synthetic-child-scenes/`
**Plan**: [plan.md](plan.md) | **Spec**: [spec.md](spec.md)
**Data model**: [data-model.md](data-model.md) | **Contracts**: [contracts/](contracts/)

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel with other [P] tasks in the same phase
- **[US#]**: User story this task belongs to
- Tests included only for acceptance-critical integration points (not TDD)

---

## Phase 1: Setup

**Purpose**: Create project skeleton and shared infrastructure

- [x] T001 Create `synth/` package with subdirectories: `synth/__init__.py`, `synth/configs/`, `synth/scripts/`, `synth/slurm/`; create `synth_results/manifests/`, `synth_results/synthetic_scenes/{wav,rttm,json}/`, `synth_results/augmentation_experiments/`; create `tests/synth/` with `__init__.py`
- [x] T002 [P] Add `data/`, `synth_results/synthetic_scenes/wav/`, `synth_results/synthetic_scenes/rttm/`, and `synth_results/synthetic_scenes/json/` to `.gitignore` (keep `synth_results/manifests/` and `synth_results/augmentation_experiments/` committed)
- [x] T003 [P] Create `tests/synth/conftest.py` with a pytest fixture that generates a minimal in-memory segment manifest (3 child segments, 2 adult segments, all `usable_for_training=true`, `split=train`) as a pandas DataFrame for use by downstream unit tests

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core library modules required by all user stories. Must complete before any US phase begins.

**⚠️ CRITICAL**: `scene_generator.py` depends on all four modules below.

- [x] T004 Implement `synth/audio_utils.py` with five functions: `resample_to_16k(wav, sr) -> np.ndarray`; `peak_normalize(wav) -> np.ndarray`; `apply_crossfade(wav, crossfade_samples) -> np.ndarray`; `convolve_rir(wav, rir_wav) -> np.ndarray` (scipy FFT convolution); `mix_at_snr(speech_wav, noise_wav, snr_db) -> np.ndarray` (RMS-based SNR scaling with peak normalization after mix). All functions operate on 1-D float32 numpy arrays at 16 kHz.
- [x] T005 [P] Implement `synth/manifest.py` with three functions: `load_manifest(csv_path) -> pd.DataFrame` (reads segment manifest CSV, validates required columns per `contracts/segment-manifest.md`, asserts no speaker_id appears in both train and test rows); `filter_usable(df, age_band=None, source_datasets=None) -> pd.DataFrame` (filters to `usable_for_training=true` and optional age_band/dataset filters); `sample_segment(df, rng) -> dict` (random-sample one row, return as dict). Raise `ValueError` with descriptive message if any integrity constraint fails.
- [x] T006 [P] Implement `synth/labels.py` with three functions: `write_rttm(tracks: list[dict], scene_id: str, path: str)` (writes standard RTTM with standardized labels `TARGET_CHILD`, `ADULT_0`, etc. per `contracts/rttm-output.md`); `write_clip_labels_row(scene_meta: dict) -> dict` (returns a dict matching all columns in `contracts/clip-labels.md`); `write_scene_metadata(scene_meta: dict, path: str)` (writes JSON matching `contracts/scene-metadata.md` schema).
- [x] T007 Implement `synth/turn_taking.py` with class `TurnTakingSimulator`: constructor takes `age_band`, `overlap_prob`, `n_turns_min`, `n_turns_max`, and per-role duration distributions; method `sample_turns(rng) -> list[dict]` returns an ordered list of turn dicts `{speaker_role, duration_sec, pause_before_sec}` where `pause_before_sec` is negative when overlap occurs (implements Markov speaker sequence with alternating TARGET_CHILD/ADULT transitions and configurable pause/overlap sampling per `research.md` D2 defaults).

---

## Phase 3: User Story 1 — Generate Labeled Synthetic Training Scenes (P1) 🎯 MVP

**Goal**: Given a segment manifest CSV and a scene config YAML, generate WAV + RTTM + JSON + clip-labels CSV for N synthetic scenes.

**Independent Test**: Run `python synth/scripts/generate_scenes.py --config synth/configs/default_14_18mo.yaml --manifest synth_results/manifests/segment_manifest.csv --n-scenes 10 --output-dir synth_results/synthetic_scenes/` and verify all 10 scenes have matching WAV/RTTM/JSON files and that RTTM labels are consistent with `target_child_vocalized` in the clip manifest.

- [x] T008 [US1] Implement `synth/scene_generator.py` with class `SceneComposer`: constructor takes config dict and segment manifest DataFrame; method `compose(scene_id, rng) -> dict` assembles a scene timeline using `TurnTakingSimulator`, samples segments from the manifest, applies `audio_utils` transforms (crossfade, RIR, SNR mixing), and returns a scene_meta dict with all fields from `contracts/scene-metadata.md`; method `write(scene_meta, output_dir)` calls `labels.write_rttm`, `labels.write_scene_metadata`, and saves the mixed WAV via soundfile. Scene duration is padded/truncated to exactly `scene_duration_sec` seconds.
- [x] T009 [P] [US1] Create `synth/configs/default_14_18mo.yaml` using the exact schema and default values specified in `contracts/scene-config.md` Example section: `scene.duration_sec=30`, `scene.n_scenes=5000`, `target_age_band=14_18_months`, `sampling.positive_scene_probability=0.50`, `sampling.overlap_probability=0.25`, `snr_db_min=0`, `snr_db_max=25`, `apply_rir_probability=0.7`, `random_seed=42`.
- [x] T010 [P] [US1] Create `synth/configs/default_34_38mo.yaml` identical to `default_14_18mo.yaml` except: `target_age_band=34_38_months`, `turn_taking.child_turn_duration_mean_sec=1.8`, `turn_taking.child_turn_duration_std_sec=0.8`, `turn_taking.pause_mean_sec=0.6`, `sampling.overlap_probability=0.20` (per `research.md` D2 age-band defaults).
- [x] T011 [US1] Implement `synth/scripts/generate_scenes.py` as a CLI script with argparse: `--config` (YAML path), `--manifest` (CSV path), `--n-scenes` (int, overrides config value), `--output-dir` (base output dir), `--seed` (int, overrides config seed). Loads config + manifest, instantiates `SceneComposer`, generates all scenes with per-scene seed = global_seed + index, writes WAV/RTTM/JSON per scene, and writes `{output_dir}/../manifests/synthetic_manifest.csv` accumulating all clip-label rows. Prints progress every 100 scenes.
- [x] T012 [US1] Add integration smoke-test to `tests/synth/test_integration.py`: generate 10 scenes using `default_14_18mo.yaml` with a tiny mock manifest (5 child segments, 3 adult segments, all 1-second WAVs filled with random noise); verify (a) all 10 WAV files exist and have duration ≥ 29.9 s; (b) each RTTM file has ≥ 1 SPEAKER line; (c) `target_child_vocalized` in clip manifest matches presence of `TARGET_CHILD` in the RTTM for every scene; (d) re-running with same seed produces bitwise-identical WAV files.
- [x] T013 [P] [US1] Create `synth/slurm/run_scene_generation.sh`: SLURM script requesting 1 CPU node, 24 GB RAM, 24h walltime; activates `child-vocalizations` conda env; accepts `--config` as `$1`; runs `generate_scenes.py` with full manifest and 5000 scenes; writes logs to `logs/synth/scene_gen_${SLURM_JOB_ID}.out`.

---

## Phase 4: User Story 2 — Build and Filter Segment Manifest (P1)

**Goal**: From Providence RTTMs and LibriSpeech metadata, produce a filtered `segment_manifest.csv` respecting split integrity.

**Independent Test**: Run `build_segment_manifest.py` with Providence only; verify output CSV has all required columns, zero `speaker_id` values that appear in both `train` and `test` rows, and all `split=test` rows have `usable_for_training=false`.

- [x] T014 [US2] Implement `synth/scripts/build_segment_manifest.py` CLI: `--providence-dir`, `--providence-rttm-dir`, `--librispeech-dir`, `--exclude-speakers-csv` (real test split CSV; reads `child_id` column and marks matching Providence speaker segments as `usable_for_training=false`, `split=test`), `--output` (CSV path), `--min-duration-sec` (default 0.3), `--quality-threshold` (default 0.4). For each Providence child RTTM, extract CHI segments into the manifest with `speaker_role=target_child`, `age_band` inferred from filename timepoint, quality score = composite proxy (RMS energy + duration score + silence ratio). For LibriSpeech, assign `speaker_role=adult`, `age_band=adult`. Writes manifest CSV and prints per-dataset counts + split integrity summary.
- [x] T015 [P] [US2] Implement `synth/scripts/extract_segments.py` CLI: `--manifest` (CSV), `--output-dir` (base dir for `child/` and `adult/` subdirs), `--sample-rate` (default 16000). For each row in manifest where `usable_for_training=true`, loads source audio, extracts `[start_time_sec, end_time_sec]`, resamples to 16 kHz mono using `audio_utils.resample_to_16k`, saves to `{output_dir}/{speaker_role}/{segment_id}.wav`. Updates `audio_path` column in the manifest CSV in-place. Skips already-extracted files (idempotent).
- [x] T016 [US2] Add `tests/synth/test_manifest.py`: test `manifest.load_manifest` raises `ValueError` when a `speaker_id` appears in both `split=train` and `split=test` rows; test `filter_usable` returns only `usable_for_training=true` rows; test that `build_segment_manifest.py` called with `--exclude-speakers-csv` pointing to a CSV with one `child_id` correctly sets all matching Providence segments to `usable_for_training=false`.

---

## Phase 5: User Story 3 — Synthetic-to-Real Ratio Experiments (P2)

**Goal**: Produce training manifests at 6 ratios, run the enrollment pipeline on each, and evaluate all on the real test set.

**Independent Test**: Run `generate_training_sets.py` and verify the `train_0x_manifest.csv` contains only real rows (no `is_synthetic=true` rows), and `train_1x_manifest.csv` has approximately equal real and synthetic counts.

- [x] T017 [US3] Implement `synth/scripts/generate_training_sets.py` CLI: `--real-train-csv` (existing `whisper-modeling/seen_child_splits/train.csv`), `--synthetic-manifest` (clip-labels CSV from scene generation), `--ratios` (list of floats, default `0 0.5 1 2 5 10`), `--output-dir`, `--seed`. For each ratio, samples `round(ratio * len(real_rows))` synthetic rows from the synthetic manifest (stratified by age_band to match real distribution), concatenates with all real rows, adds `is_synthetic` column, and writes `train_{ratio}x_manifest.csv` matching the schema in `contracts/training-manifest.md`. Prints row counts per ratio.
- [x] T018 [P] [US3] Implement `synth/scripts/train_with_synthetic.py` CLI: `--manifest-dir`, `--ratios`, `--output-dir`. For each ratio manifest, calls the existing BabAR enrollment script (`pyannote/unified.py --diarizer babar`) with the augmented training CSV substituted for `whisper-modeling/seen_child_splits/train.csv` (via a temp copy or `--train-csv` override if the script supports it). Saves enrollment model outputs to `synth_results/augmentation_experiments/{config_name}/ratio_{r}x/`. Documents the exact invocation in inline comments.
- [x] T019 [US3] Implement `synth/scripts/evaluate_synthetic_augmentation.py` CLI: `--experiment-dir`, `--test-csv` (real held-out test CSV), `--output-dir`, `--plot` (flag). For each `ratio_{r}x/` subdir in experiment-dir, loads enrollment predictions and computes F1, Precision, Recall, AUROC, AUPRC (using existing metric functions from `av_fusion/scripts/utils.py`). Writes `metrics_by_ratio.csv` and `metrics_by_age_band.csv`. If `--plot`, generates `figures/synthetic_ratio_vs_auprc.png` and `figures/synthetic_ratio_vs_f1.png` (line plots with ratio on x-axis).
- [x] T020 [P] [US3] Create `synth/slurm/run_ratio_sweep.sh`: SLURM script requesting 1 GPU node (for BabAR enrollment), 40 GB RAM, 48h walltime; activates `child-vocalizations`; accepts `--config` as `$1`; sequentially runs Steps 4–6 from quickstart.md for all 6 ratios; writes logs to `logs/synth/ratio_sweep_${SLURM_JOB_ID}.out`.

---

## Phase 6: User Story 4 — Hard-Negative and Stress Scenes (P2)

**Goal**: Generate adult-only, background-speech, short-vocalization, and low-SNR scene sets for targeted failure-mode augmentation.

**Independent Test**: Generate 20 scenes from `hard_negatives.yaml`; verify all have `target_child_vocalized=0` and RTTM contains no `TARGET_CHILD` line.

- [x] T021 [US4] Extend `synth/scene_generator.py` `SceneComposer.compose()` to support all 8 scene types from the spec: `adult_only_negative` (sample only ADULT segments, no TARGET_CHILD), `background_speech_negative` (sample BACKGROUND_SPEECH label from MUSAN speech subset), `silence_noise_negative` (noise only, no RTTM speakers), `hard_overlap_positive` (force ≥1 overlap event involving TARGET_CHILD), `hard_overlap_negative` (ADULT speaks during overlap window, TARGET_CHILD absent), `short_vocalization_positive` (cap TARGET_CHILD segment dur at `short_threshold_sec`), `low_snr_positive` (override SNR range to `[snr_db_min, snr_db_min + 5]`). Scene type is selected by sampling from the scene-type probability distribution in config.
- [x] T022 [P] [US4] Create `synth/configs/hard_negatives.yaml`: sets `positive_scene_probability=0.0`, `adult_only_negative_probability=0.50`, `background_speech_negative_probability=0.40`, `noise_only_negative_probability=0.10`, `overlap_probability=0.0`, `n_scenes=2000`.
- [x] T023 [P] [US4] Create `synth/configs/overlap_stress.yaml`: sets `positive_scene_probability=0.60`, `adult_only_negative_probability=0.40`, `overlap_probability=0.90`, `n_scenes=2000`.
- [x] T024 [P] [US4] Create `synth/configs/low_snr_stress.yaml`: sets `snr_db_min=-5`, `snr_db_max=5`, `positive_scene_probability=0.60`, `apply_noise_probability=1.0`, `n_scenes=2000`.

---

## Phase 7: User Story 5 — Synthetic vs. Real Distribution Analysis (P3)

**Goal**: Produce distribution comparison plots and embedding-space visualization confirming synthetic scenes are acoustically similar to real clips.

**Independent Test**: Run `analyze_synthetic_quality.py` with a small synthetic manifest (50 scenes) and real train CSV; verify at least 4 PNG figures are produced.

- [x] T025 [US5] Implement `synth/scripts/analyze_synthetic_quality.py` CLI: `--synthetic-manifest`, `--real-train-csv`, `--output-dir`, `--encoder-model` (default `microsoft/wavlm-base-plus`). Computes and plots: (1) duration distribution histograms (real vs. synthetic, by age band); (2) loudness distribution (RMS dB); (3) SNR distribution (synthetic only, from manifest); (4) child/adult duration ratio; (5) optional UMAP of frozen WavLM embeddings of 200 random real vs. 200 synthetic clips (requires `umap-learn`; skips gracefully if not installed). Saves all figures to `{output_dir}/figures/` as PNGs.
- [x] T026 [P] [US5] Implement `synth/scripts/error_analysis_synthetic.py` CLI: `--experiment-dir`, `--test-csv`, `--output-dir`. For the real-only (0×) model and the best-performing synthetic ratio, loads predictions, then categorizes test clips into 8 failure modes per spec: `real_only_fp_fixed`, `real_only_fn_fixed`, `new_fp_introduced`, `new_fn_introduced`, `short_vocalization_error`, `overlap_error`, `adult_background_fp`, `error_by_age_band`. Writes `error_analysis.csv` with per-clip categorization and counts summary.

---

## Phase 8: Polish & Cross-Cutting Concerns

**Purpose**: Documentation, integration, and thesis readiness

- [x] T027 Write `synth/README.md` covering: prerequisites (data downloads, conda env), step-by-step quickstart matching `quickstart.md`, cache invalidation rules, gotchas (exclude-speakers-csv, no GPU for generation, gitignore for data/), and pointers to each config file.
- [x] T028 [P] Update `CLAUDE.md` with a new "Synthetic Data Generator" section: add `synth/` to Project Overview, document the 7-step pipeline and key commands (`build_segment_manifest.py`, `generate_scenes.py`, `evaluate_synthetic_augmentation.py`), add `synth_results/` to Results Storage, add cache invalidation gotcha, add the `synth/slurm/` SLURM scripts to batch inference section.
- [x] T029 [P] Add a "Synthetic Augmentation Experiments" section to `results_summary.md` with a placeholder table for metrics-by-ratio results (AUROC, AUPRC, F1 by ratio and age band) to be filled in after experiments run.

---

## Phase 9: Ratio Sweep Post-Processing

**Completed**: SLURM job 12613912 completed. NULL RESULT: all 6 ratios (0x–10x) produce identical metrics (F1=0.874, AUROC=0.820, AUPRC=0.918). ECAPA encoder is frozen; synthetic embeddings match real ones; prototype averaging is unaffected.

- [x] T030 [US3] Run `synth/scripts/evaluate_synthetic_augmentation.py` once all ratio subdirs are complete: `python synth/scripts/evaluate_synthetic_augmentation.py --experiment-dir synth_results/augmentation_experiments/default_14_18mo/ --test-csv whisper-modeling/seen_child_splits/test.csv --output-dir synth_results/augmentation_experiments/default_14_18mo/ --plot`; outputs `metrics_by_ratio.csv`, `metrics_by_age_band.csv`, `figures/synthetic_ratio_vs_{auprc,f1}.png`
- [x] T031 [P] [US3] Run `synth/scripts/error_analysis_synthetic.py`: `python synth/scripts/error_analysis_synthetic.py --experiment-dir synth_results/augmentation_experiments/default_14_18mo/ --test-csv whisper-modeling/seen_child_splits/test.csv --output-dir synth_results/augmentation_experiments/default_14_18mo/`; outputs `error_analysis.csv`
- [x] T032 [P] Update CLAUDE.md with final synthetic augmentation metrics by ratio once T030 completes; commit `metrics_by_ratio.csv`, `metrics_by_age_band.csv`, and figures

---

## Dependencies (Story Completion Order)

```
Phase 1 (Setup)
  └─► Phase 2 (Foundational: audio_utils, manifest, labels, turn_taking)
        └─► Phase 3 (US1: generate scenes) ──────────────────────────────►┐
        └─► Phase 4 (US2: build manifest)  ──► feeds manifest into US1   ─┤
              └─► Phase 5 (US3: ratio experiments)  ◄── depends on US1    ┤
              └─► Phase 6 (US4: hard-negative configs) ◄── extends US1    ┤
                    └─► Phase 7 (US5: quality analysis) ◄── needs scenes  ┘
                          └─► Phase 8 (Polish)
```

**US2 (manifest builder) must run before US1 in practice** (scenes need a populated manifest), but implementation of US1 and US2 code can proceed in parallel since they depend on different foundational modules.

---

## Parallel Execution Within Phases

**Phase 2** (all four modules are independent):
- T004 `audio_utils.py` ║ T005 `manifest.py` ║ T006 `labels.py` ║ T007 `turn_taking.py`

**Phase 3** (after T008 scene_generator):
- T009 `default_14_18mo.yaml` ║ T010 `default_34_38mo.yaml`
- T012 integration test ║ T013 SLURM script (after T011 generate_scenes.py)

**Phase 4**:
- T015 `extract_segments.py` ║ T016 manifest tests (after T014 build_segment_manifest)

**Phase 5**:
- T018 `train_with_synthetic.py` ║ T020 SLURM script (after T017 generate_training_sets)

**Phase 6**:
- T022 `hard_negatives.yaml` ║ T023 `overlap_stress.yaml` ║ T024 `low_snr_stress.yaml` (after T021)

**Phase 8**:
- T028 CLAUDE.md ║ T029 results_summary.md (after T027)

---

## Implementation Strategy

**MVP** (minimum to answer the primary research question): Complete Phases 1–5 in order. This delivers scene generation, manifest building, ratio training sets, and real-test evaluation. US4 and US5 are enhancements.

**Suggested MVP order**:
1. Phases 1–2 (setup + foundational modules) — ~2 days
2. Phase 4 (manifest builder + extractor) — ~1 day, produces input data for generation
3. Phase 3 (scene generator + default configs + CLI) — ~2 days
4. Phase 5 (ratio experiments + evaluation) — ~2 days

**Total task count**: 29 tasks across 8 phases
**Parallelizable tasks**: 16 marked [P]
**Per-story task counts**: US1 → 6, US2 → 3, US3 → 4, US4 → 4, US5 → 2, Polish → 3
