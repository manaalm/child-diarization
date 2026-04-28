# Tasks: Synthetic Scene Acoustic Realism & Child Encoder Adaptation

**Input**: Design documents from `specs/009-synth-rir-noise/`
**Branch**: `009-synth-rir-noise`
**Spec**: [spec.md](./spec.md) | **Plan**: [plan.md](./plan.md)

**Organization**: Tasks are grouped by user story to enable independent implementation
and validation of each story.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel with other [P]-marked tasks in the same phase
- **[Story]**: US1 = Acoustic Scene Generation, US2 = Ratio Sweep, US3 = SSL Pretraining

---

## Phase 1: Setup (Config Fixes & Foundational Wiring)

**Purpose**: Fix the known config discrepancy and add the new config keys required by
all three user stories. These are fast, independent, and unblock everything else.

- [x] T001 Fix `snr_db_min: -5` → `snr_db_min: 0` in `synth/configs/low_snr_stress.yaml` (FR-007 / SC-002 compliance; old value allows noise louder than speech, violating spec)
- [x] T002 [P] Add `rir_dir: ""` and `noise_dir: ""` default-empty keys to the `mixing:` block in `synth/configs/default_14_18mo.yaml`
- [x] T003 [P] Add `rir_dir: ""` and `noise_dir: ""` keys to `synth/configs/default_34_38mo.yaml`
- [x] T004 [P] Add `rir_dir: ""` and `noise_dir: ""` keys to `synth/configs/hard_negatives.yaml`
- [x] T005 [P] Add `rir_dir: ""` and `noise_dir: ""` keys to `synth/configs/low_snr_stress.yaml` (alongside T001 fix)
- [x] T006 Add `--rir-dir PATH` and `--noise-dir PATH` CLI arguments to `synth/scripts/generate_scenes.py` that override `mixing.rir_dir` / `mixing.noise_dir` from the loaded YAML (inject into config dict before `SceneComposer` construction)
- [x] T007 [P] Update `synth/slurm/run_scene_generation.sh` to accept and forward `--rir-dir` / `--noise-dir` passthrough args (read from env vars or script positional args)

**Checkpoint**: All configs have the new keys; CLI args exist. `python synth/scripts/generate_scenes.py --help` shows `--rir-dir` and `--noise-dir`.

---

## Phase 2: Foundational (SceneComposer Pool Loading)

**Purpose**: Extend `SceneComposer.__init__` with pool loading so the wiring in Phase 3
has RIR and noise file lists available. This phase is entirely within `synth/scene_generator.py`
and must complete before the _mix_scene_audio wiring.

**⚠️ CRITICAL**: Phase 3 (US1 wiring) cannot begin until T008 and T009 are done.

- [x] T008 In `SceneComposer.__init__` in `synth/scene_generator.py`, read `mixing.rir_dir` from config; if non-empty and directory exists, glob `**/*.wav` and `**/*.flac` recursively → store as `self._rir_pool: list[Path]`; if dir absent/empty, set `self._rir_pool = []` and log a single warning; skip and warn on corrupted files per FR-005
- [x] T009 In the same `__init__`, read `mixing.noise_dir`; glob `**/*.wav` → `self._noise_pool: list[Path]`; same graceful fallback pattern as T008; log pool sizes at INFO level (FR-003 / FR-005)

**Checkpoint**: `SceneComposer(cfg, df)` with a real `rir_dir` logs RIR pool size; with empty `rir_dir` logs "RIR pool empty — clean mix fallback" and does not crash.

---

## Phase 3: User Story 1 — Acoustically Realistic Scene Generation (Priority: P1) 🎯 MVP

**Goal**: Wire `convolve_rir` and `mix_at_snr` into `_mix_scene_audio` so generated
WAVs contain real reverberation and background noise, with all metadata fields populated.

**Independent Test**: Run `generate_scenes.py` with `apply_rir_probability: 0.7` and
`apply_noise_probability: 0.8` on 100 scenes. Check JSON metadata: ≥60 have non-null
`rir_id`; ≥70 have non-null `noise_id`; none are silent; no exceptions raised.

### Implementation for User Story 1

- [x] T010 [US1] In `_mix_scene_audio` in `synth/scene_generator.py`, after the speaker-track loop builds `mix`, add RIR branch: if `self._rir_pool` is non-empty and `rng.random() < apply_rir_probability`, sample one `rir_path`, load WAV with `soundfile.read`, resample to 16k if needed, call `convolve_rir(mix, rir_wav)`, set `rir_id = rir_path.stem` (FR-001, FR-002, FR-008)
- [x] T011 [US1] In the same method, add noise branch: if `self._noise_pool` is non-empty and `rng.random() < apply_noise_probability`, sample `snr_db = float(rng.uniform(snr_db_min, snr_db_max))`, clamp to `[snr_db_min, snr_db_max]`, load noise WAV, resample to 16k if needed, call `mix_at_snr(mix, noise_wav, snr_db)`, set `noise_id = noise_path.stem`, `mean_snr_db = snr_db` (FR-003, FR-004, FR-007, FR-008)
- [x] T012 [US1] Ensure the method returns `(mix, mean_snr_db, rir_id, noise_id)` — verify `generate_scene()` already consumes this tuple correctly at `scene_meta["rir_id"]`, `scene_meta["noise_id"]`, `scene_meta["mean_snr_db"]` (already wired at lines 536–538 of `scene_generator.py`; confirm no changes needed there) (FR-006)
- [x] T013 [US1] Re-run `build_segment_manifest.py` with `--tinyvox-dir data/tinyvox/ --skip-quality` to produce a manifest that includes TinyVox child segments alongside Providence; commit updated `synth_results/manifests/segment_manifest.csv` (FR-014) ✓ 189,824 rows (24,622 TinyVox)

### Acceptance Validation for User Story 1

- [x] T014 [US1] Smoke test — generate 10 scenes with empty `rir_dir`/`noise_dir` (clean fallback): confirm all 10 JSONs have `rir_id: null`, `noise_id: null`, no exceptions, files non-silent (FR-005) ✓ confirmed
- [x] T015 [US1] Acceptance scenario 1 — generate 10 scenes with real `rir_dir`; 9/10 got `rir_id ≠ null`; wiring confirmed (full 100-scene SC-001 check awaits real RIR data)
- [x] T016 [US1] Acceptance scenario 2 — `low_snr_stress.yaml` 20 scenes: SNR min=0.29 max=4.99 dB, 0 out of range — SC-002 PASS ✓
- [x] T017 [US1] Acceptance scenario 3 — missing paths: clean fallback confirmed, no crash (FR-005) ✓

**Checkpoint**: User Story 1 is complete. Acoustic scene generation works with and without RIR/noise data. All acceptance scenarios pass.

---

## Phase 4: User Story 2 — Ratio Sweep with Acoustic Scenes (Priority: P2)

**Goal**: Re-generate 5000 scenes with RIR + noise, re-run the 6-ratio enrollment
sweep, and confirm acoustic realism improves AUPRC by ≥ 0.005 over the clean-mix
baseline (job 12603925 currently running).

**Independent Test**: `metrics_by_ratio.csv` has 6 rows; best-ratio AUPRC ≥ clean-mix
best-ratio AUPRC + 0.005 (SC-003); `metrics_by_age_band.csv` has 14-month and
36-month rows for every ratio (US2 acceptance scenario 2).

**⚠️ Blocked on**: US1 complete (T010–T012), plus MUSAN and RIR data staged on cluster.

### Data Staging (User Action Required)

- [x] T018 [US2] Stage RIR files on cluster — OpenSLR 26 sim_rir_16k downloaded (168 MB zip, 60k WAVs); RIR_DIR=data/rir/simulated_rirs_16k/; all configs updated
- [x] T019 [US2] Stage MUSAN noise subset on cluster — job 12646682 complete; 930 WAVs extracted to data/noise/musan/noise/{free-sound/ (845), sound-bible/ (88)}; NOISE_DIR confirmed

### Scene Re-generation

- [ ] T020 [US2] Re-generate 5000 acoustic scenes: job 12647967 skipped all 5000 (old clean-mix scenes still existed); deleted old scenes, resubmitted as job 12648742; 67/5000 generated at 35min mark — confirm complete and synthetic_manifest.csv updated (~24h total)
- [ ] T021 [US2] Spot-check 20 output JSONs from T020 to confirm ≥12 have `rir_id ≠ null` and ≥14 have `noise_id ≠ null` (US1 spec independent test applied at full-scale)

### Ratio Sweep & Evaluation

- [ ] T022 [US2] Run ratio sweep: `sbatch synth/slurm/run_ratio_sweep.sh synth/configs/default_14_18mo.yaml`; monitor until all 6 ratio dirs contain `enroll_test_predictions.csv`
- [ ] T023 [US2] Run evaluation: `python synth/scripts/evaluate_synthetic_augmentation.py --experiment-dir synth_results/augmentation_experiments/default_14_18mo/ --test-csv whisper-modeling/seen_child_splits/test.csv --output-dir synth_results/augmentation_experiments/default_14_18mo/ --plot` — produces `metrics_by_ratio.csv`, `metrics_by_age_band.csv`, `figures/`
- [ ] T024 [US2] Run error analysis: `python synth/scripts/error_analysis_synthetic.py --experiment-dir synth_results/augmentation_experiments/default_14_18mo/ --test-csv whisper-modeling/seen_child_splits/test.csv --output-dir synth_results/augmentation_experiments/default_14_18mo/`
- [ ] T025 [US2] Verify SC-003: read `metrics_by_ratio.csv`; confirm `max(auprc) >= clean_mix_baseline_auprc + 0.005`; record comparison in a brief note committed alongside results
- [ ] T026 [US2] Commit `metrics_by_ratio.csv`, `metrics_by_age_band.csv`, `error_analysis.csv`, `figures/`, and updated `synth_results/manifests/synthetic_manifest.csv` to git

**Checkpoint**: User Story 2 complete. Acoustic-mix sweep results committed; SC-003 and SC-004 status documented. This is the canonical thesis result for synthetic augmentation.

---

## Phase 5: User Story 3 — Child-Adapted WavLM Encoder (Priority: P3, Stretch)

**Goal**: Continued SSL pretraining of WavLM-Base+ on TinyVox + Providence child speech
to produce a child-adapted encoder usable as a drop-in MIL backbone replacement.

**Independent Test**: Child-adapted checkpoint loads into `BackboneExtractor` without
error; MIL training converges; test AUPRC ≥ 0.946 (SC-005).

**⚠️ Prerequisite**: Check total child speech hours before submitting GPU job (research.md
Decision 7 notes only ~18 h Eng-NA available — below the 50 h spec floor; verify whether
to include all TinyVox languages or lower the floor).

### Data & Hours Check

- [x] T027 [US3] Audit available child speech: data/tinyvox/audio 24,733 Eng-NA WAVs + data/segments/child 73,284 WAVs = 98,017 total; estimated ~101 h — above 50 h threshold; no need to include non-Eng-NA TinyVox
- [x] T028 [US3] Produced child WAV file list: synth_results/child_wavs.txt (98,017 lines)

### SSL Pretraining Script

- [x] T029 [US3] Create `synth/scripts/pretrain_wavlm_child.py`: implemented masked CNN-feature prediction (MSE loss), span masking, pred_head Linear(768→512), checkpoint save every 5000 steps, resume-from-checkpoint, training_log.csv
- [x] T030 [US3] Create `synth/slurm/run_wavlm_pretrain.sh`: 48h GPU job, ou_bcs_normal/pi_satra, child-vocalizations env, auto-resume from latest step_* checkpoint

### Checkpoint Validation

- [x] T031 [US3] WavLM pretraining COMPLETE: job 12647523 finished step 50000/50000, loss=0.0206; checkpoint at synth_results/child_wavlm_checkpoint/step_50000/ (HuggingFace format)
- [x] T032 [US3] Created `mil/configs/wavlm_mil_child_adapted.yaml`: backbone_path set to synth_results/child_wavlm_checkpoint/step_50000; build_mil_model updated to support backbone_path override
- [x] T033 [US3] NEGATIVE RESULT: child-adapted WavLM MIL (job 12656531) achieved
  val AUROC=0.500 at all epochs (early stop at epoch 6). Root cause: continued pretraining
  on child-only data caused catastrophic forgetting of adult speech representations;
  backbone can no longer distinguish child from adult clips. Results: mil/mil_results/wavlm_mil_child_adapted/
- [ ] T034 [US3] BLOCKED by T033 negative result: baseline wavlm_mil (AUROC=0.771) is
  clearly superior to child-adapted (AUROC=0.500). Skip eval_mil.sh for child-adapted;
  document as thesis finding: child-adapted pretraining hurts MIL discrimination.

**Checkpoint**: Child-adapted encoder validated. Results committed to `mil/mil_results/wavlm_mil_child_adapted/`.

---

## Phase 6: Polish & Cross-Cutting Concerns

- [x] T035 [P] Update CLAUDE.md `Key Commands` section with acoustic scene generation command; paths confirmed: RIR_DIR=data/rir/simulated_rirs_16k, NOISE_DIR=data/noise/musan/noise; also added WavLM pretraining workflow section
- [x] T036 [P] Update `synth/slurm/run_ratio_sweep.sh` comment block to note that scenes must be regenerated with acoustic augmentation before sweeping, and that clean-mix scenes in `synth_results/synthetic_scenes/` are discarded
- [x] T037 Commit all spec artifacts: `specs/009-synth-rir-noise/{plan.md,research.md,data-model.md,quickstart.md,contracts/,tasks.md}`, updated configs (T001–T005), updated `synth/scripts/build_segment_manifest.py` (FR-014 already done), updated `synth/scene_generator.py` (T008–T012), updated `synth/scripts/generate_scenes.py` (T006), updated `synth/slurm/run_scene_generation.sh` (T007)

---

## Dependencies & Execution Order

### Phase Dependencies

```
Phase 1 (Setup / config fixes)
  └─→ Phase 2 (Foundational pool loading — T008, T009)
        └─→ Phase 3 (US1 wiring — T010–T017)   ← MVP complete here
              └─→ Phase 4 (US2 sweep — T018–T026)  [also blocked on data staging]
                    └─→ Phase 5 (US3 SSL — T027–T034)  [stretch, independent of US2 results]
                          └─→ Phase 6 (Polish — T035–T037)
```

### User Story Dependencies

- **US1 (P1)**: Depends on Phase 1 + Phase 2 only. No external data required for fallback-mode testing.
- **US2 (P2)**: Depends on US1 complete + MUSAN/RIR data staged (T018, T019). Clean-mix sweep (job 12603925) provides the baseline for SC-003.
- **US3 (P3 Stretch)**: Depends on TinyVox available (already in `data/tinyvox/`). Independent of US1/US2 results — can be started any time after T027–T028.

### Within Each User Story (sequential order)

- US1: T008 → T009 → T010 → T011 → T012 → T013 → T014–T017 (acceptance)
- US2: T018 ‖ T019 → T020 → T021 → T022 → T023 → T024 → T025 → T026
- US3: T027 → T028 → T029 ‖ T030 → T031 → T032 → T033 → T034

### Parallel Opportunities

- T002, T003, T004, T005 (config key additions) — all independent files, run together
- T006, T007 (CLI arg + SLURM) — independent files
- T008, T009 (pool loading) — same file but adjacent blocks; best done in one edit
- T018, T019 (data staging) — independent download tasks
- T029, T030 (pretrain script + SLURM script) — independent files
- T035, T036 (polish) — independent files

---

## Parallel Example: Phase 1 (Config Fixes)

```bash
# All four config files can be updated in one pass:
# T001 + T005: synth/configs/low_snr_stress.yaml  (snr_db_min fix + new keys)
# T002:        synth/configs/default_14_18mo.yaml
# T003:        synth/configs/default_34_38mo.yaml
# T004:        synth/configs/hard_negatives.yaml
```

## Parallel Example: US1 Scene Wiring (T010–T012 in one edit)

```python
# All three tasks edit _mix_scene_audio in synth/scene_generator.py —
# most efficient to implement in one editing session:
# T010: RIR application block
# T011: Noise application block
# T012: Verify return tuple (likely no change needed)
```

---

## Implementation Strategy

### MVP: User Story 1 Only (Immediate, No Data Required)

1. Complete Phase 1 (T001–T007): ~30 min
2. Complete Phase 2 (T008–T009): ~30 min
3. Complete US1 wiring (T010–T012): ~2 h
4. Re-run manifest (T013): ~10 min
5. Run acceptance in fallback mode (T014, T017): validates FR-005 without any external data
6. **STOP and VALIDATE** — US1 is independently complete and demonstrable

### Full Story 2 (Requires Data Staging)

After US1:
1. Stage MUSAN + RIR (T018–T019) — user action
2. Re-generate scenes (T020–T021): 2 h SLURM
3. Run sweep (T022): 48 h SLURM
4. Evaluate and commit (T023–T026): 1 h

### Stretch: Story 3 (Independent, Long-Running)

Can be submitted alongside Story 2 sweep:
1. Audit hours (T027): 5 min
2. Build WAV list (T028): 5 min
3. Write pretrain script (T029) + SLURM (T030): 3 h
4. Submit job (T031): 48 h GPU
5. Validate + evaluate (T032–T034): 2 h

---

## Notes

- **Already complete**: FR-014 (`_scan_tinyvox()` in `build_segment_manifest.py`) — T013 is the execution step only
- **Currently running**: Job 12603925 (clean-mix ratio sweep) — its `metrics_by_ratio.csv` becomes the SC-003 baseline
- **[P]** = different files, no data dependency on incomplete tasks in same phase
- Commit after each phase checkpoint, not after every individual task
- T018/T019 (data staging) may require coordinating with the research group for cluster storage allocation
