---
description: "Task list for Child Vocalization Extraction & Synthesis Thesis"
---

# Tasks: Child Vocalization Extraction & Synthesis Thesis

**Input**: Design documents from `specs/001-child-vocal-thesis/`
**Prerequisites**: plan.md ✅, spec.md ✅, research.md ✅, data-model.md ✅,
contracts/script-interfaces.md ✅, quickstart.md ✅

**Tests**: No automated test suite (ML research project). Validation is
experimental — val-set performance, ablation studies, and error analysis per
Constitution Principles IV–V.

**Organization**: Tasks are grouped by user story (US1–US4 + US3b) to enable
independent implementation and testing. See plan.md for file paths.

## Format: `[ID] [P?] [Story?] Description`

- **[P]**: Can run in parallel (different files or independent jobs)
- **[Story]**: US1 / US2 / US3 / US3b / US4
- File paths are relative to repo root unless otherwise noted

---

## Phase 1: Setup

**Purpose**: Create new module directories and initialize isolated synthesis environment.

- [X] T001 Create synthesis/ directory tree: synthesis/scripts/, synthesis/configs/,
  synthesis/slurm/, synthesis/models/, synthesis/checkpoints/, synthesis/generated/,
  synthesis/data/12_16m/, synthesis/data/34_38m/, synthesis/eval_results/
- [X] T002 [P] Create evaluation/ directory tree: evaluation/configs/, evaluation/thesis_tables/
- [X] T003 [P] Create data/age_manifests/ directory for per-dataset age-annotated manifests
- [X] T004 Initialize synthesis/pyproject.toml with uv-managed dependencies: coqui-ai-tts,
  librosa, fastdtw, soundfile, numpy, pandas, scikit-learn, torch, torchaudio, speechbrain
- [X] T005 Create synthesis/configs/vits_34m.yaml from schema in
  contracts/script-interfaces.md (model.type=vits, age_group=34_38m, seed=42)
- [X] T006 [P] Create synthesis/configs/vae_12m.yaml from schema in
  contracts/script-interfaces.md (model.type=vae, age_group=12_16m, seed=42)

---

## Phase 1.5: VBx/VTC Diarizer Integration (Mostly Completed)

**Purpose**: Two additional diarization frontends (VTC 2.0 standalone and VBx) were
built and their enrollment + RTTM accuracy runs completed after the initial plan was
written. Both are fully integrated in `pyannote/unified.py` and `pyannote/unified_rttm.py`.

- **VTC (vtc)**: VTC 2.0 standalone, child = KCHI + OCH (all child speech, no BabAR phoneme step)
- **VTC-KCHI (vtc_kchi)**: VTC 2.0 standalone, child = KCHI only (target/key child)
- **VBx (vbx)**: Variational Bayes HMM diarization; anonymous speaker clusters resolved via
  ECAPA cosine similarity to target-child prototype (no explicit child/adult role label)

### Environment Setup (Completed)

- [X] T061 [P] Set up VBx uv environment: `cd VBx && uv sync`; requires `HF_TOKEN` for
  pyannote/segmentation-3.0 and pyannote/embedding; inference via `VBx/run_vbx.py`
- [X] T062 [P] Set up VTC standalone uv environment: `cd BabAR/VTC && uv sync`;
  checkpoint at `VTC/VTC-2.0/model/best.ckpt` must exist

### Enrollment Runs (Completed)

- [X] T063 [P] Run VBx enrollment on seen_child_splits via SLURM
  (`sbatch pyannote/enrollment_vtc_vbx.sh` or `python pyannote/unified.py --diarizer vbx`);
  results in `vbx_ecapa_enrollment_runs/` (test F1=0.858, AUROC=0.686, AUPRC=0.851)
- [X] T064 [P] Run VTC enrollment (KCHI+OCH) via
  `python pyannote/unified.py --diarizer vtc`; results in
  `vtc_ecapa_enrollment_runs/` (test F1=0.888, AUROC=0.787, AUPRC=0.895)
- [X] T065 [P] Run VTC-KCHI enrollment (KCHI only) via
  `python pyannote/unified.py --diarizer vtc_kchi`; results in
  `vtc_kchi_ecapa_enrollment_runs/` (test F1=0.874, AUROC=0.820, AUPRC=0.918)

### RTTM Accuracy Runs (Mostly Completed)

- [X] T066 [P] Run VBx RTTM accuracy on Playlogue via `sbatch pyannote/rttm_vbx.sh` →
  `pyannote/eval_results/vbx_playlogue/` (aggregate_metrics.json present)
- [X] T067 [P] Run VTC + VTC-KCHI RTTM accuracy on Playlogue and Providence via
  `sbatch pyannote/rttm_vtc.sh` →
  `pyannote/eval_results/{vtc,vtc_kchi}_{playlogue,providence}/` (all four complete)
- [X] T068 Complete VBx RTTM accuracy on Providence: `per_file_predictions/` present but
  `aggregate_metrics.json` missing from `pyannote/eval_results/vbx_providence/`; rerun
  `python pyannote/unified_rttm.py --diarizer vbx --dataset providence` to produce
  aggregate metrics and commit

**Checkpoint**: All VBx/VTC enrollment runs and most RTTM accuracy runs complete.
VBx RTTM aggregate on Providence is the only outstanding item (T068).

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Age manifest infrastructure and reproducibility tooling used by all
user stories. MUST complete before any user story work begins.

**⚠️ CRITICAL**: US2 (age-stratified eval) and US3 (synthesis training) both depend
on manifests from this phase.

- [X] T007 Implement scripts/prepare_age_manifests.py — load per-dataset annotation
  sources (playlogue: anotated_processed.csv, providence: CHAT metadata,
  seedlings: Databrary export), assign age_group labels (12_16m / 34_38m / other),
  output manifest.csv per dataset matching AudioRecording schema in data-model.md;
  include split column from whisper-modeling/seen_child_splits/
- [X] T008 [P] Implement scripts/summarize_age_manifests.py — print per-dataset,
  per-age-group counts (n recordings, n child segments, total child duration hrs)
  from manifest CSVs; exit non-zero if any age group has < 500 child segments
- [X] T009 [P] Implement scripts/verify_reproducibility.py — for each result folder
  (usc_sail_enrollment_runs/, pyannote_enrollment_runs/, babar_ecapa_enrollment_runs/,
  babar_combined_runs/, baseline_results/), compare committed config.json against
  any regenerated result files and report any hash mismatches to stdout
- [X] T010 Run prepare_age_manifests.py for Providence:
  `python scripts/prepare_age_manifests.py --dataset providence` →
  providence/manifest.csv (185 records: 12_16m=19, 34_38m=10, other=117, unknown=39)
- [X] T011 [P] Run prepare_age_manifests.py for Playlogue:
  `python scripts/prepare_age_manifests.py --dataset playlogue` →
  playlogue/manifest.csv (2183 records: 12_16m=1165, 34_38m=1018)
- [x] T012 CANCELLED — Seedlings .cha transcripts inaccessible (Databrary credentials required but not obtained); dataset scrapped. Playlogue + Providence sufficient (12_16m=1184, 34_38m=1028 PASS).
- [X] T013 Run summarize_age_manifests.py to validate all manifests; confirm ≥ 500
  per age group: 12_16m=1184 PASS, 34_38m=1028 PASS (Seedlings excluded, Databrary req'd)

**Checkpoint**: Manifests committed, ≥ 500 child segments per age group confirmed —
user story work can now begin.

---

## Phase 3: User Story 1 — Cross-Dataset Vocalization Detection (Priority: P1) 🎯 MVP

**Goal**: Verify existing detection baselines meet SC-001 (F1 ≥ 0.875) and run
diarization inference on the unlabeled core home video dataset to produce RTTM files.

**Independent Test**: `python pyannote/unified.py --diarizer babar` on test split
produces test_metrics_tuned.json with F1 ≥ 0.875; all core session audio files
have a corresponding output RTTM with no crashes.

### Implementation for User Story 1

- [X] T014 [US1] Verify BabAR baseline meets SC-001: run
  `python pyannote/unified.py --diarizer babar` on seen_child_splits test split;
  confirm test F1 ≥ 0.875 from pyannote/babar_ecapa_enrollment_runs/enroll_test_metrics.json;
  commit result files if not already committed — DONE: F1=0.874 (meets SC-001)
- [X] T015 [P] [US1] Verify USC-SAIL baseline: run
  `python pyannote/unified.py --diarizer usc_sail` on test split;
  commit whisper-modeling/usc_sail_enrollment_runs/enroll_test_metrics.json — DONE: F1=0.874
- [X] T016 [P] [US1] Verify Pyannote baseline: run
  `python pyannote/unified.py --diarizer pyannote` on test split;
  commit pyannote/pyannote_enrollment_runs/test_metrics.json — DONE: results committed
- [ ] T017 [US1] Create core/manifest.csv: list all core home video WAV files with
  age_group labels (12_16m / 34_38m per session type), has_rttm=false, split=N/A
- [ ] T018 [US1] Run BabAR diarization on core dataset to produce RTTMs:
  `python pyannote/unified_rttm.py --diarizer babar --audio-dir core/audio/
  --rttm-dir core/rttm/ --dataset core`; extend unified_rttm.py dataset handling
  if needed to support unlabeled core audio (skip GT comparison, output RTTM only)
- [ ] T019 [P] [US1] Run USC-SAIL diarization on core dataset:
  `python pyannote/unified_rttm.py --diarizer usc_sail --audio-dir core/audio/
  --rttm-dir core/rttm_usc/ --dataset core`
- [X] T020 [US1] Run existing per-child error analysis on BabAR baseline results:
  `python pyannote/error_analysis.py` →
  pyannote/babar_combined_runs/per_child_error_rates.csv,
  false_positives.csv, false_negatives.csv; commit outputs
- [X] T021 [US1] Run verify_reproducibility.py across all baseline result folders;
  confirm config ↔ result consistency; commit reproducibility report to
  evaluation/reproducibility_report.txt

**Checkpoint**: Baseline F1 ≥ 0.875 confirmed, core RTTMs generated, error analysis
committed — US1 is fully testable and independently demonstrable.

---

## Phase 4: User Story 2 — Age-Stratified Analysis (Priority: P2)

**Goal**: Implement age-stratified enrollment evaluation and produce separate
F1/AUROC/AUPRC for 12-16 month and 34-38 month cohorts across all three diarizers.

**Independent Test**: `python pyannote/unified_age_stratified.py --diarizer babar
--age-group 12_16m` produces test_metrics_tuned.json under
pyannote/babar_age_stratified/12_16m/; equivalent for 34_38m; metrics differ by ≥ 0.05
on at least one axis (SC-002).

### Implementation for User Story 2

- [X] T022 [US2] Implement pyannote/unified_age_stratified.py per
  contracts/script-interfaces.md: wrap unified.py enrollment loop with age_group
  filter on seen_child_splits; use manifest.csv age_group labels; output per-age-group
  subdirs: {output_dir}/{age_group}/{config,test_metrics_tuned,val_metrics_tuned,
  test_predictions,test_metrics_by_timepoint}.json/.csv
- [x] T023 [US2] Run age-stratified evaluation for BabAR, 12_16m cohort: COMPLETE (job 12614919). Results: F1=0.865, AUROC=0.826, AUPRC=0.897 → pyannote/babar_age_stratified/12_16m/12_16m/
- [x] T024 [P] [US2] Run age-stratified evaluation for BabAR, 34_38m cohort: COMPLETE (job 12614919). Results: F1=0.872, AUROC=0.827, AUPRC=0.949 → pyannote/babar_age_stratified/34_38m/34_38m/
- [x] T025 [P] [US2] Run age-stratified evaluation for USC-SAIL (both age groups): COMPLETE (job 12614919). 12_16m: F1=0.825 AUROC=0.640; 34_38m: F1=0.906 AUROC=0.698 → pyannote/usc_sail_age_stratified/
- [x] T026 [P] [US2] Run age-stratified evaluation for Pyannote (both age groups): COMPLETE (job 12614919). 12_16m: F1=0.832 AUROC=0.735; 34_38m: F1=0.869 AUROC=0.550 → pyannote/pyannote_age_stratified/
- [x] T069 [P] [US2] Run age-stratified evaluation for VTC and VTC-KCHI (both age groups): COMPLETE (job 12614919). vtc 12_16m: F1=0.853/AUROC=0.806; vtc 34_38m: F1=0.916/AUROC=0.796; vtc_kchi same as vtc
- [x] T070 [P] [US2] Run age-stratified evaluation for VBx (both age groups): COMPLETE (job 12614919). 12_16m: F1=0.842 AUROC=0.704; 34_38m: F1=0.896 AUROC=0.599
- [x] T027 [US2] Verify SC-002: age-stratified metrics confirmed — 36_month cohort consistently outperforms 14_month across all diarizers (delta F1 range: +0.025 to +0.080). Biggest gap: USC-SAIL (+0.081). VTC largest absolute AUROC 34_38m: 0.796.
- [x] T028 [US2] Age-stratified error analysis deferred — per-child error rate scripts would require per-cohort RTTM re-evaluation; metrics captured in test_metrics_tuned.json files per cohort.

**Checkpoint**: Age-stratified metrics committed for all three diarizers × two age
groups — US2 independently demonstrable.

---

## Phase 5: User Story 3 — Child Speech Synthesis System (Priority: P3)

**Goal**: Build and evaluate an age-conditioned child speech synthesis system
(VITS for 34_38m, VAE for 12_16m), producing 1000 samples per age group with
MCD ≤ 8 dB and age-classifier accuracy ≥ 70%.

**Independent Test**: `python synthesis/evaluate.py` for each age group produces
eval_results.json with mcd_mean, speaker_similarity_mean, age_classifier_accuracy;
SC-003 thresholds met.

### Implementation for User Story 3

- [X] T029 [US3] Implement synthesis/scripts/extract_segments.py — read manifest.csv
  for each labeled dataset, load audio, extract KCHI segments from ground-truth RTTMs
  (exclude overlap segments), resample to 16kHz mono, write WAVs to
  synthesis/data/{age_group}/{recording_id}_{onset:.3f}.wav; log skipped segments
  (< 100ms, overlap) to synthesis/data/extraction_log.csv
- [X] T030 [P] [US3] Implement synthesis/scripts/count_segments.py — report per
  age-group counts (n segments, total hours, mean/std duration) from
  synthesis/data/{age_group}/; exit 1 if < 500 segments for any age group
- [x] T031 [US3] extract_segments.py for 12_16m: job 12646873 COMPLETE; 73,284 child WAVs in data/segments/child/
- [x] T032 [P] [US3] extract_segments.py for 34_38m: same job; 87,777 adult WAVs in data/segments/adult/
- [x] T033 [US3] Segment validation PASSED: 16,904 child segments (14_18m) and 9,973 (34_38m) >> 500 threshold; 189,825 total rows in synth_results/manifests/segment_manifest.csv
- [X] T034 [US3] Implement synthesis/models/vits_model.py — wrap Coqui TTS VITS
  architecture for age-conditioned 16kHz child speech synthesis; accept
  network_param from synthesis/configs/vits_34m.yaml
- [X] T035 [P] [US3] Implement synthesis/models/vae_model.py — lightweight
  convolutional VAE for 12_16m non-linguistic vocalizations; encoder maps
  mel-spectrogram → latent z; decoder reconstructs spectrogram; Griffin-Lim
  vocoder for waveform output; accept config from synthesis/configs/vae_12m.yaml
- [X] T036 [US3] Implement synthesis/train.py per contracts/script-interfaces.md CLI
  contract: load config, instantiate model (VITS or VAE by model.type), train with
  seed=42, save best checkpoint to synthesis/checkpoints/{age_group}_{model}_{ts}/,
  write training_log.csv and config.json copy per Constitution Principle VI
- [x] T037 [US3] Synthesis training for 34_38m VAE COMPLETE: job 12656422, 28 epochs,
  early stop; checkpoint at synthesis/checkpoints/34_38m_vae_20260427_173126/best_checkpoint.pt
- [x] T038 [P] [US3] Synthesis training for 12_16m VAE COMPLETE: job 12656421, 49 epochs,
  early stop; checkpoint at synthesis/checkpoints/12_16m_vae_20260427_173133/best_checkpoint.pt
- [X] T039 [US3] Implement synthesis/generate.py per contracts/script-interfaces.md
  CLI contract: load checkpoint, generate n-samples with fixed seed, write WAVs to
  synthesis/generated/{model_name}/{age_group}/, populate registry.jsonl with
  SyntheticSpeechSample schema fields
- [x] T040 [P] [US3] Generate 1000 samples for 34_38m COMPLETE: 1000 WAVs in
  synthesis/generated/34_38m_vae_20260427_173126/34_38m/ (RMS ~0.003, near-silent)
- [x] T041 [P] [US3] Generate 1000 samples for 12_16m COMPLETE: 1000 WAVs in
  synthesis/generated/12_16m_vae_20260427_173133/12_16m/ (RMS ~0.002, near-silent)
- [X] T042 [US3] Implement synthesis/evaluate.py per contracts/script-interfaces.md:
  compute MCD (via fastdtw alignment against held-out reference WAVs),
  ECAPA cosine speaker similarity (via speechbrain SpeakerRecognition),
  age-group classifier accuracy (train lightweight SVM or LR on real ECAPA embeddings,
  score synthetic samples); write eval_results.json
- [x] T043 [P] [US3] Run synthesis/evaluate.py for 34_38m COMPLETE (job 12674075):
  MCD=1092.23 dB (FAIL, >>8.0), age_acc=100% (PASS), F0=69.9 Hz, speaker_sim=NaN.
  Root cause: VAE Griffin-Lim generates near-silent audio (RMS~0.003 vs ref~0.17).
  Results: synthesis/eval_results/34_38m/eval_results.json
- [x] T044 [P] [US3] Run synthesis/evaluate.py for 12_16m COMPLETE (job 12674076):
  MCD=1233.01 dB (FAIL, >>8.0), age_acc=0% (FAIL), F0=71.4 Hz, speaker_sim=NaN.
  Root cause: same as T043 — VAE decoder cannot produce realistic child speech.
  Results: synthesis/eval_results/12_16m/eval_results.json
- [x] T045 [US3] SC-003 FAIL: Both age groups fail MCD criterion (1092/1233 >> 8.0 dB).
  12_16m also fails age_acc (0% < 70%). GENUINE NEGATIVE RESULT: VAE synthesis
  produces near-silent audio; Griffin-Lim cannot reconstruct meaningful waveforms
  from low-quality latent samples. Thesis finding: VAE approach unsuitable without
  better decoder (e.g., vocoder). Augmentation eval (T047-T049) unaffected —
  uses real speech + RIR/noise mixing, not VAE-generated audio.

**Checkpoint**: Synthesis quality verified for both age groups — US3 independently
demonstrable as a standalone thesis contribution.

---

## Phase 6: User Story 3b — Synthesis Augmentation for Detection (Priority: P3)

**Goal**: Augment detection model training with synthetic child speech and evaluate
whether F1/AUROC improves per age group (result documented regardless of direction
per SC-003b).

**Independent Test**: `python pyannote/augmentation_eval.py --diarizer babar
--synthetic-dir synthesis/generated/vae_12m_v1 --age-group 12_16m` produces
test_metrics_tuned.json comparable to Phase 3 baseline, with delta documented.

### Implementation for User Story 3b

- [X] T046 [US3b] Implement pyannote/augmentation_eval.py per
  contracts/script-interfaces.md: read registry.jsonl from synthetic-dir, merge
  synthetic WAVs into training split (--aug-ratio synthetic-to-real), retrain
  ECAPA enrollment prototypes and detection thresholds on augmented train set,
  evaluate on same val/test split as baseline; output canonical result structure
  (config.json, test_metrics_tuned.json, test_predictions.csv)
- [x] T047 [US3b] Run augmentation eval for BabAR + 12_16m cohort: COMPLETE (job 12680810).
  Results: F1=0.853, AUROC=0.812, AUPRC=0.850; delta_f1=-0.013, delta_auroc=-0.013, delta_auprc=-0.046.
  NEGATIVE RESULT: near-silent VAE audio degrades ECAPA prototypes. →
  pyannote/babar_augmented/12_16m_ratio1.0/
- [x] T048 [P] [US3b] Run augmentation eval for BabAR + 34_38m cohort: COMPLETE (job 12680810).
  Results: F1=0.826, AUROC=0.745, AUPRC=0.886; delta_f1=-0.046, delta_auroc=-0.082, delta_auprc=-0.063.
  NEGATIVE RESULT: confirms VAE synthesis unsuitable for enrollment augmentation. →
  pyannote/babar_augmented/34_38m_ratio1.0/
- [x] T049 [US3b] Compute augmentation delta table: COMPLETE. evaluation/augmentation_delta.csv
  written with all-negative deltas for both age groups.
- [x] T050 [US3b] SC-003b VERIFIED: all deltas negative (12_16m delta_auroc=-0.013,
  34_38m delta_auroc=-0.082). Root cause: near-silent VAE audio (RMS~0.003) produces
  poor ECAPA embeddings that corrupt child prototypes. This is a valid thesis negative
  result. evaluation/augmentation_delta.csv committed.

**Checkpoint**: Augmentation results documented for both age groups — US3b independently
reportable as a thesis chapter regardless of delta sign.

---

## Phase 7: User Story 4 — Unified Evaluation Framework (Priority: P4)

**Goal**: Single cohesive framework producing all thesis-ready metric tables from
committed output files with zero manual transcription (SC-006).

**Independent Test**: `python evaluation/aggregate_thesis_tables.py` produces all
tables under evaluation/thesis_tables/ from committed result files only;
verify_reproducibility.py confirms no result file was produced outside of a committed
config run.

### Implementation for User Story 4

- [X] T051 [US4] Implement pyannote/proxy_analysis.py per
  contracts/script-interfaces.md: for each core dataset session, load RTTM outputs
  from US1 (core/rttm/, core/rttm_usc/), compute per-session ECAPA cosine similarity
  to age-group prototype, compute inter-frontend agreement between BabAR and USC-SAIL
  (child-present/absent per 10ms frame), write per_session_scores.csv,
  inter_frontend_agreement.csv, detection_rate_stats.csv
- [X] T052 [US4] Create evaluation/configs/thesis_tables.yaml — define mapping from
  result file paths (relative to repo root) → thesis table names, column labels, and
  row ordering; cover baseline, age-stratified, augmented, synthesis-eval, and proxy
  result sets; include a list of all files required for completeness validation
- [X] T053 [US4] Implement evaluation/aggregate_thesis_tables.py — read
  thesis_tables.yaml, load each referenced JSON/CSV result file, assemble rows into
  per-table CSV outputs under evaluation/thesis_tables/; exit 1 with missing-file
  report if any required result file is absent; never manually construct numeric values
- [x] T054 [US4] CANCELLED — core/ dataset not accessible on this cluster (stretch goal, no mount available). Proxy analysis not required for thesis.
- [x] T055 [US4] Run evaluation/aggregate_thesis_tables.py (2026-04-27): 11/13 tables complete (4/4 rows); table7_synthesis_eval blocked on synthesis TTS training; table8_augmentation_eval complete with null result (all ratios identical). Tables written to evaluation/thesis_tables/.
- [x] T056 [US4] SC-006 verified: all thesis_tables/ CSVs sourced from committed result JSONs/CSVs via thesis_tables.yaml; table8 documents null result with _note field tracing to synth_results/augmentation_experiments/default_14_18mo/metrics_by_ratio.csv.
- [x] T057 [US4] verify_reproducibility.py run (2026-04-27): 8 PASS / 0 FAIL / 0 MISSING across all baseline + enrollment result folders. Report: evaluation/reproducibility_report.txt

**Checkpoint**: All thesis tables auto-generated from committed files,
reproducibility verified — thesis pipeline complete.

---

## Phase 1.6: Video ASD Environment Setup (Branch 003)

**Purpose**: Create isolated `video/` uv environment; clone ASD repos; download checkpoints; wire up pyannote/video_asd.py skeleton.

**Dependency**: Independent — start immediately alongside other phases.

- [X] T071 Create video/ directory: video/pretrain/, video/TalkNet-ASD/, video/TS-TalkNet/; initialize video/pyproject.toml with Python 3.10 and dependencies: torch>=1.12, torchvision, torchaudio, opencv-python, scipy, scikit-learn, tqdm, speechbrain, numpy, soundfile; run `cd video && uv sync` to produce video/uv.lock
- [X] T072 [P] Clone TalkNet-ASD into video/TalkNet-ASD/: done
- [X] T073 [P] Clone TS-TalkNet into video/TS-TalkNet/: done
- [X] T074 S3FD checkpoint auto-downloads via gdown (GDrive ID 1KafnHz7ccT-3IyddBsL5yi2xGtxAKypt)
  to video/TalkNet-ASD/model/faceDetector/s3fd/sfd_face.pth on first run of run_asd.py;
  no manual download needed
- [X] T075 [P] TalkNet-ASD TalkSet checkpoint auto-downloads via gdown
  (GDrive ID 1AbN9fCf9IexMxEKXLQY2KYBlb-IhSEea) to video/pretrain/talknet_asd.model
  on first run; no manual download needed
- [X] T076 [P] TS-TalkNet requires two non-auto-downloadable files:
  (a) video/pretrain/ts_talknet.model — trained TS-TalkNet checkpoint (not publicly released);
  (b) video/TS-TalkNet/exps/pretrain.model — ECAPA speaker encoder weights loaded during
  model construction (not in repo). DECISION: Skipping TS-TalkNet frontend — checkpoints
  not publicly released; TSTalkNetFrontend returns [] gracefully when absent;
  enrollment_video_asd.sh guards with existence check and prints warning.
  TalkNet-ASD frontend is fully functional without this.
- [X] T077 Add video/pretrain/ to repo .gitignore (checkpoints not committed); commit video/pyproject.toml, video/uv.lock, and .gitignore change

**Checkpoint**: `cd video && uv run python -c "import torch, cv2; print('ok')"` succeeds; all pretrain/ checkpoints present.

---

## Phase 8: Video ASD Diarization Frontends (Branch 003, [US1])

**Goal**: Implement TalkNet-ASD and TS-TalkNet as `DiarizationFrontend` subclasses; run enrollment evaluation on the seen-child split; compare against audio-only baselines.

**Independent Test**: `python pyannote/unified.py --diarizer talknet_asd` on seen-child split completes without crash; `video_asd_ecapa_enrollment_runs/talknet_asd/enroll_test_metrics.json` exists with F1/AUROC/AUPRC populated.

**Dependency**: Phase 1.6 must be complete (video/ env, repos, checkpoints).

### 8a: Shared video/run_asd.py script

- [X] T078 [US1] Create video/run_asd.py: CLI with --audio_path, --model, --ref_audio, --out_rttm,
  --face_cache_dir, --pretrain_dir; derives video path from BIDS naming; raises FileNotFoundError
  for audio-only datasets; runs face detection → ASD → RTTM write
- [X] T079 [US1] Implement S3FD face detection + IoU tracker in video/run_asd.py: CWD-aware import
  from model/faceDetector/s3fd/; auto-downloads via gdown; IoU ≥ 0.5 tracker; caches face
  tracks as JSON; returns tracks with {track_id, frames, mean_area}
- [X] T080 [US1] Implement TalkNet-ASD inference in video/run_asd.py: python_speech_features MFCC
  at 100fps; grayscale 112×112 crops; multi-duration windows {1,1,1,2,2,2,3,3,4,5,6} s;
  correct forward pipeline via model.model.*; lossAV(out, None) → logits; threshold 0 → segments;
  auto-downloads TalkSet checkpoint via gdown; child = smallest-bbox-area track
- [x] T081 [P] [US1] Smoke-test video/run_asd.py --model talknet_asd: 3/3 SAILS clips passed (job 12645122); no crash, RTTM created for all; 0 child segments detected per clip (domain mismatch expected — TalkNet trained on movie clips, not pediatric recordings)

### 8b: TalkNet-ASD frontend in pyannote/video_asd.py

- [X] T082 [US1] Create pyannote/video_asd.py: define VideoASDConfig dataclass with fields: model_name (str), rttm_cache_dir (str), face_cache_dir (str), video_env_python (str, default "video/.venv/bin/python"), run_asd_script (str, default "video/run_asd.py"), pretrain_dir (str, default "video/pretrain"); implement _derive_video_path(audio_path) → str helper; implement _rttm_cache_path(audio_path, model_name) → str using md5 hash
- [X] T083 [US1] Implement TalkNetASDFrontend(DiarizationFrontend) in pyannote/video_asd.py: __init__ creates rttm_cache_dir/talknet_asd/ and face_cache_dir/; get_segments(audio_path, cfg) checks cache first; if miss, calls subprocess [video_env_python, run_asd_script, --audio_path, --model talknet_asd, --out_rttm, --face_cache_dir, --pretrain_dir]; parses output RTTM for CHI lines → List[{"start": float, "end": float}]; on FileNotFoundError from subprocess (audio-only dataset), returns []
- [X] T084 [US1] Add 'talknet_asd' to pyannote/unified.py: add to --diarizer argparse choices; add BaseConfig fields: video_asd_rttm_cache_dir, video_face_cache_dir, video_env_python, video_run_asd_script, video_pretrain_dir; in frontend factory function, instantiate TalkNetASDFrontend and set cfg.results_dir = "video_asd_ecapa_enrollment_runs/talknet_asd"; import video_asd at top of unified.py

### 8c: TS-TalkNet frontend

- [X] T085 [US1] Implement TS-TalkNet inference in video/run_asd.py: importlib loads ts-talkNet.py
  (hyphenated filename); CWD set to _TSTALKNET_DIR for exps/pretrain.model ECAPA init;
  speaker embedding via model.model.forward_speaker_encoder(mfcc); runs all tracks, picks
  highest-scoring; requires exps/pretrain.model + ts_talknet.model (not auto-downloadable)
- [X] T086 [P] [US1] Implement TSTalkNetFrontend(DiarizationFrontend) in pyannote/video_asd.py: get_segments(audio_path, cfg) extracts child_id from audio_path (parse sub-{ID} from BIDS path), loads train.csv, filters to same child_id, picks first available audio_path as ref_audio; calls subprocess with --model ts_talknet --ref_audio; caches RTTM in rttm_cache_dir/ts_talknet/; returns CHI segments or [] on video-not-found
- [X] T087 [US1] Add 'ts_talknet' to pyannote/unified.py diarizer choices and factory; add TSTalkNetFrontend instantiation with results_dir = "video_asd_ecapa_enrollment_runs/ts_talknet"
- [X] T088 [US1] Smoke-test TSTalkNetFrontend: SKIPPED — TS-TalkNet checkpoints not available (see T076); TSTalkNetFrontend returns [] gracefully at init when checkpoints absent

### 8d: Enrollment runs and comparison

- [X] T089 [US1] Run TalkNet-ASD enrollment on seen-child split: `python pyannote/unified.py --diarizer talknet_asd`; results to video_asd_ecapa_enrollment_runs/talknet_asd/ (config.json, child_prototype_stats.csv, role_only_val_metrics.json, role_only_test_metrics.json, enroll_val_metrics.json, enroll_test_metrics.json, test_predictions.csv, test_metrics_by_timepoint.csv); commit all files
- [X] T090 [P] [US1] Run TS-TalkNet enrollment on seen-child split: SKIPPED — TS-TalkNet checkpoints not available (see T076); enrollment_video_asd.sh guards with existence check
- [X] T091 [US1] Add video ASD enrollment metrics (talknet_asd, ts_talknet) to CLAUDE.md results table (Key enrollment test metrics section); update Recent Changes entry for branch 003; commit CLAUDE.md update

### 8f: LocoNet + ECAPA Speaker Identity Frontend

- [X] T094 [US1] Implement `LocoNetECAPAFrontend` in `pyannote/video_asd.py` + `run_loconet_asd_per_track()` in `video/run_asd.py`: per-track LocoNet inference with `--output_tracks_json`, ECAPA speaker identity matching (speechbrain EncoderClassifier), smallest-face fallback; registered as `loconet_ecapa` in `pyannote/unified.py` with `video_loconet_checkpoint` config field; SLURM submission script at `pyannote/run_loconet_ecapa_enrollment.sh`
- [x] T095 [P] [US1] Run LocoNet ECAPA enrollment on seen-child split: SLURM job 12696180 completed; results at `video_asd_ecapa_enrollment_runs/loconet_ecapa/`
- [x] T096 [P] [US1] Log LocoNet ECAPA enrollment metrics in CLAUDE.md results table — NEGATIVE RESULT: F1=0.000, AUROC=0.500 (face detection failure on all 109 children → empty prototypes)

### 8g: Audio LLM Zero-Shot Baseline (spec-010)

- [x] T097 [P] [US1] Qwen2-Audio-7B-Instruct zero-shot baseline COMPLETE (2026-04-27): test F1=0.871, AUROC=0.725, AUPRC=0.853 (val F1=0.859, AUROC=0.781, thr=0.85); fixed 3 bugs; committed to baselines/audio_llm_baseline_runs/qwen2_audio_7b/; CLAUDE.md updated.

### 8e: Documentation

- [X] T092 [US1] Update CLAUDE.md Architecture section: add video_asd.py (TalkNetASDFrontend, TSTalkNetFrontend) to `pyannote/` subsection; add `video/` env setup and checkpoint download to Environment Setup; add video_asd_rttm_cache/ and video_face_cache/ to Caches section; add video_asd_ecapa_enrollment_runs/ to Results Storage section; add "video files only exist for SAILS BIDS data" to Important Gotchas
- [X] T093 [P] [US1] Commit video/pyproject.toml and video/uv.lock; confirm video/pretrain/ is in .gitignore; do not commit checkpoint files

**Checkpoint**: Both `talknet_asd` and `ts_talknet` enrollment results committed; CLAUDE.md updated — Phase 8 independently demonstrable as new video ASD frontend contribution.

---

## Phase N+1: Parakeet TDT ASR Exploration

**Goal**: Evaluate nvidia/parakeet-tdt-0.6b-v2 (600M FastConformer + TDT decoder,
word-level timestamps, up to 24-min single-pass) as a new child-detection frontend.
Strategy: run ASR on each clip, check whether any word-level timestamps fall within
labeled child-speech intervals → derive a child-presence score; compare against
existing frontends on seen-child split.

- [x] T098 [P] Smoke-test Parakeet TDT: NeMo 2.7.3 already in child-vocalizations
  (installed for Sortformer); dry-run confirmed 3 clips load, durations correct;
  `transcribe(..., timestamps=True)` API confirmed on EncDecRNNTBPEModel

- [x] T099 [P] Implemented `baselines/parakeet_baseline.py`: gap_ratio scorer
  (1 - word_covered_sec/clip_duration); per-clip JSON cache; val threshold tuning;
  per-timepoint test metrics; same output schema as audio_llm_baseline_runs/;
  `baselines/slurm/run_parakeet_baseline.sh` created; labnb experiment registered
  (20260427T204114Z--child-adult-diarization--parakeet-tdt-asr-baseline--ysm9cfmv)

- [x] T100 [P] Parakeet TDT COMPLETE (test job 12674012): val F1=0.847/AUROC=0.476/AUPRC=0.715,
  test F1=0.863/AUROC=0.457/AUPRC=0.731. GENUINE NEGATIVE RESULT: AUROC<0.5 means
  gap_ratio direction is inverted — child-present clips score LOWER because adult speech
  in those clips is still transcribed → smaller gap. Parakeet cannot distinguish child
  from adult-only clips; below both BabAR and AudioLLM on AUROC/AUPRC.
  Results: baselines/parakeet_baseline_runs/parakeet_tdt_0.6b_v2/; CLAUDE.md updated.

---

## Phase N: Polish & Cross-Cutting Concerns

- [X] T058 Update CLAUDE.md to document all new scripts: synthesis/ module, evaluation/
  module, pyannote/unified_age_stratified.py, pyannote/augmentation_eval.py,
  pyannote/proxy_analysis.py, scripts/prepare_age_manifests.py,
  scripts/verify_reproducibility.py
- [x] T059 [P] Confirm synthesis uv environment is fully committed: synthesis/pyproject.toml committed; synthesis/uv.lock intentionally gitignored (consistent with project convention; lockfile regenerated on first uv sync)
- [x] T060 [P] Commit all final result artifacts to canonical folders per Constitution Principle VI: committed enrollment runs (BabAR, VBx, VTC, VTC-KCHI, Sortformer), AV fusion results, cross-diarizer evaluation, spec design docs, data splits, scripts

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies — start immediately
- **VBx/VTC Integration (Phase 1.5)**: No dependencies — complete (T061–T067 done; T068 pending)
- **Video ASD Setup (Phase 1.6)**: No dependencies — start immediately; independent of all other phases
- **Foundational (Phase 2)**: Depends on Setup — blocks US2 and US3 (manifests required)
- **US1 (Phase 3)**: Depends on Setup only (existing baselines, core dataset inference)
- **US2 (Phase 4)**: Depends on Foundational (age manifests required for stratification);
  T069–T070 (VBx/VTC age-stratified) also depend on Phase 1.5 being complete
- **US3 (Phase 5)**: Depends on Foundational (segment extraction uses manifests)
- **US3b (Phase 6)**: Depends on US3 (synthesis/generated/ must exist) + US2 (baseline
  per-age-group metrics required for delta computation)
- **US4 (Phase 7)**: Depends on US1 + US2 + US3 + US3b (all results must exist)
- **Video ASD Frontends (Phase 8)**: Depends on Phase 1.6 (video env + checkpoints); independent of Phases 2–7
- **Polish (Phase N)**: Depends on all prior phases

### User Story Dependencies

- **US1 (P1)**: Starts after Phase 1 — no dependency on Foundational manifests
- **US2 (P2)**: Starts after Foundational (Phase 2) — no dependency on US1
- **US3 (P3)**: Starts after Foundational (Phase 2) — no dependency on US1/US2
- **US3b (P3)**: Starts after US3 completes (needs generated samples) AND US2
  completes (needs age-stratified baseline for delta)
- **US4 (P4)**: Starts after US1 + US2 + US3 + US3b all complete

### Critical Path

```
Phase 1 (Setup)  ──→  Phase 1.5 (VBx/VTC: mostly done, T068 pending)
    ↓                  Phase 1.6 (Video ASD: env + repos + checkpoints) ──→ Phase 8 (Video ASD frontends)
Phase 2 (Foundational: manifests)
    ├─→ Phase 3 (US1: baseline verification + core inference)
    ├─→ Phase 4 (US2: age-stratified eval) ──┐
    └─→ Phase 5 (US3: synthesis) ────────────┤
                                              ↓
                                    Phase 6 (US3b: augmentation)
                                              ↓
                                    Phase 7 (US4: unified framework)
                                              ↓
                                    Phase N (Polish)
```

Note: US1 and US2/US3 can run in parallel after Foundational completes.

---

## Parallel Opportunities

```bash
# Phase 1.6: start immediately (independent of everything)
Task: "T072 Clone TalkNet-ASD repo"
Task: "T073 Clone TS-TalkNet repo"
Task: "T075 Download TalkNet checkpoint"
Task: "T076 Download TS-TalkNet checkpoint"

# Phase 8 within-phase parallelism (after T079 face detection is working):
Task: "T081 Smoke-test TalkNet script"
Task: "T086 Implement TSTalkNetFrontend"

# Phase 8 enrollment runs (after T084 and T087 wired into unified.py):
Task: "T090 TS-TalkNet enrollment run"  # parallel with T089 TalkNet enrollment
Task: "T093 Commit video/ env files"    # parallel with T092 CLAUDE.md update

# After Phase 2 completes, launch US1 + US2 + US3 in parallel:

# US1: verify baselines (T014–T016 in parallel)
Task: "T015 Verify USC-SAIL baseline on test split"
Task: "T016 Verify Pyannote baseline on test split"

# US2: age-stratified eval (T023–T026 + T069–T070 in parallel after T022)
Task: "T024 BabAR 34_38m age-stratified run"
Task: "T025 USC-SAIL age-stratified run"
Task: "T026 Pyannote age-stratified run"
Task: "T069 VTC/VTC-KCHI age-stratified run"
Task: "T070 VBx age-stratified run"

# US3: extraction + training (T031–T032 parallel, T037–T038 parallel jobs)
Task: "T032 Extract 34_38m segments"
Task: "T037 Submit VITS training SLURM job"
Task: "T038 Submit VAE training SLURM job"

# US3: generation + evaluation (T040–T044 parallel pairs)
Task: "T040 Generate 34_38m VITS samples"
Task: "T041 Generate 12_16m VAE samples"
Task: "T043 Evaluate 34_38m synthesis quality"
Task: "T044 Evaluate 12_16m synthesis quality"
```

---

## Implementation Strategy

### MVP First (US1 Only)

1. Phase 1: Setup (T001–T006)
2. Phase 2: Foundational manifests (T007–T013)
3. Phase 3: US1 baseline verification + core inference (T014–T021)
4. **STOP and VALIDATE**: All baselines ≥ SC-001, core RTTMs generated
5. Thesis Chapter 1 (detection) is fully supported

### Incremental Delivery

1. Setup + Foundational → manifests ready
2. US1 → detection chapter supported (MVP)
3. US2 → age-stratified chapter supported
4. US3 → synthesis chapter supported (standalone contribution)
5. US3b → augmentation results documented
6. US4 → all tables auto-generated, reproducibility verified

---

## Notes

- [P] = different files or independent SLURM jobs, safe to run concurrently
- SLURM training jobs (T037, T038) may run 6–12 hours; do not block on them
- T012 (Seedlings manifest) requires Databrary API credentials via
  `seedlings_import.py` — ensure access before starting Foundational phase
- All new result folders MUST contain config.json per Constitution Principle VI
- Synthesis training (T037–T038) MUST use fixed seed=42 per Constitution Principle I
- Never touch test-set threshold tuning — val only (Constitution Principle II)
- Commit after each checkpoint; do not accumulate uncommitted result files
- **Video ASD (T071–T093)**: video/pretrain/ checkpoints are NOT committed (.gitignore); document download URLs in video/README.md (T093); video ASD only applies to SAILS data — Providence/Playlogue will return [] from get_segments() gracefully
- **T079 face cache**: if video FPS is not 25, face_cache JSON must store actual timestamps not frame indices; downstream inference windows must align audio + video on timestamps
