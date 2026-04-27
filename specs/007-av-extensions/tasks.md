# Tasks: AV Extended Experiments — 007-av-extensions

**Input**: Design documents from `specs/007-av-extensions/`
**Prerequisites**: 006 pipeline complete — `av_fusion/av_results/{run_name}/av_{train,val,test}.csv` must exist; BabAR enrollment predictions in `babar_ecapa_enrollment_runs/enroll_{val,test}_predictions.csv`

**Organization**: Tasks grouped by user story for independent implementation and testing.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies on incomplete tasks)
- **[Story]**: Which user story this task belongs to

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Config and cache directory structure needed by all scripts

- [X] T001 Create `av_fusion/configs/av_extensions.yaml` with all config sections per contracts/cli_scripts.md: seed, asd_models (loconet + light_asd checkpoints/batch_size/device), gpt4o (model/sample_rate/max_tokens/temperature/cache_dir), cascade (vad_feature/child_id_feature/grid lists), temporal_smoothing (default_method/bandwidth_grids/window_grids/group_cols)
- [X] T002 [P] Add `av_fusion/gpt4o_cache/` to `.gitignore` (alongside existing `av_fusion/face_track_cache/` entry) and create `av_fusion/gpt4o_cache/.gitkeep` so the directory is tracked by git but contents are not

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: No additional foundational code required — all user stories build on the existing 006 pipeline. T001 (config) is the only shared prerequisite. All Phase 3+ tasks may begin after T001.

**Checkpoint**: Config ready — all user story implementation can now begin.

---

## Phase 3: User Story 1 — Cascaded Detection Pipeline (Priority: P1) 🎯 MVP

**Goal**: Three-stage cascade (VAD → child ID → AV fusion) with per-clip stage logging, val-tuned thresholds, and a stage breakdown table for the test set.

**Independent Test**: Run `python av_fusion/scripts/train_cascaded_pipeline.py --feature-dir av_fusion/av_results/manual_only/ --output-dir av_fusion/av_results/manual_only/models/`; confirm `cascade_thresholds.json` exists; run `python av_fusion/scripts/evaluate_av_fusion.py ... --cascade-breakdown ...`; confirm `metrics_cascade_by_stage.csv` has non-NaN AUROC row for each stage.

- [X] T003 [US1] Implement `av_fusion/scripts/train_cascaded_pipeline.py` — argparse for --feature-dir, --output-dir, --vad-feature (default: kchi_total_dur), --child-id-feature (default: prob), --seed; load av_val.csv; grid search over vad_threshold ∈ [0.0,0.1,0.2,0.3,0.4,0.5,0.75,1.0,1.5,2.0] and child_id_threshold ∈ [0.1,0.2,...,0.9] maximizing val F1; assign cascade_stage per clip (1=vad_speech_detected=False → final_prob=0.0, 2=child_id_score below child_id_threshold, 3=AV fusion); write models/cascade_thresholds.json (vad_threshold, child_id_threshold, val_f1, val_auroc) and cascade_val_stage_breakdown.csv (CascadeStageRecord schema from data-model.md)
- [X] T004 [US1] Extend `av_fusion/scripts/evaluate_av_fusion.py` — add `--cascade-breakdown` arg (path to cascade_stage_breakdown.csv) and `cascaded_av` model class; load cascade_thresholds.json from model-dir; apply three-stage logic to test set writing cascade_stage_breakdown.csv (per-clip stage, final_prob, vad_threshold, child_id_threshold); compute AUROC/F1 broken down by cascade_stage (1/2/3) and write metrics_cascade_by_stage.csv; ensure all existing 006 model paths still work unchanged

**Checkpoint**: Cascade pipeline end-to-end — train thresholds → evaluate on test → stage breakdown table.

---

## Phase 4: User Story 2 — Temporal Smoothing (Priority: P1)

**Goal**: Post-processing layer that applies Gaussian, majority-vote, or moving-average smoothing within each (child_id, session) group; val-tuned bandwidth; smoothed metrics logged alongside raw.

**Independent Test**: Run `python av_fusion/scripts/smooth_predictions.py --predictions predictions_test.csv --val-predictions predictions_val.csv --output predictions_test_smoothed.csv --method gaussian --param None`; confirm output CSV has `prob_smoothed` column; confirm printed val F1 raw vs smoothed.

- [X] T005 [US2] Implement `av_fusion/scripts/smooth_predictions.py` — argparse for --predictions, --output, --method {gaussian,majority_vote,moving_average}, --param (float or None; None = auto-tune on val), --val-predictions (required when --param None), --group-cols (default: child_id,timepoint_norm); load predictions CSV; group by group-cols; sort within each group by clip_position if present else clip_id lexicographic; apply gaussian (scipy.ndimage.gaussian_filter1d on prob column), majority_vote (rolling mode with window=param), or moving_average (pd.Series.rolling(window=int(param)).mean()); when --param None, grid-search bandwidth/window on val set (grids from av_extensions.yaml: gaussian_bandwidth_grid, majority_vote_window_grid) maximizing F1; write SmoothedPredictionRecord CSV (data-model.md schema) with prob_raw, prob_smoothed, smoothing_method, smoothing_param, session_id, clip_position columns added; print val F1 (raw) vs val F1 (smoothed) as diagnostic
- [X] T006 [US2] Extend `av_fusion/scripts/evaluate_av_fusion.py` — add `--smoothed-predictions` arg; when provided, compute AUROC/F1/precision/recall from prob_smoothed vs label column; write metrics_smoothed.csv with raw and smoothed metric rows side-by-side; handle missing prob_smoothed column with clear error message pointing to smooth_predictions.py

**Checkpoint**: Temporal smoothing independently testable; smoothed + raw metrics reported side-by-side.

---

## Phase 5: User Story 3 — GPT-4o Vision Features (Priority: P2)

**Goal**: Extract structured child-detection outputs from GPT-4o-mini over sampled video frames; per-clip features (child_visible_gpt4o, child_vocalizing_gpt4o, visual_quality_gpt4o, gpt4o_reasoning); resumable with frame-level JSON cache; dry-run cost estimate before any API calls.

**Independent Test**: Run `python av_fusion/scripts/extract_gpt4o_features.py --metadata-csv ... --output gpt4o_features.csv --max-clips 50`; confirm CSV has GPT4oFeatureRow columns; NaN rows for audio-only clips; re-running is idempotent.

- [X] T007 [US3] Implement argument parsing, metadata loading, and video availability check in `av_fusion/scripts/extract_gpt4o_features.py` — argparse for --metadata-csv, --output, --model {gpt-4o-mini,gpt-4o}, --sample-rate (default: 2), --cache-dir (default: av_fusion/gpt4o_cache/), --max-clips, --dry-run; require OPENAI_API_KEY env var (raise clear error if missing); load metadata CSV; identify audio-only rows (missing or NaN video_path) → write NaN GPT4oFeatureRow immediately; apply --max-clips cap to video clips only; print cost estimate (n_frames × ~1000 tokens × per-token price) before any API calls; confirm with [y/N] prompt (skip prompt in --dry-run mode, just print and exit)
- [X] T008 [US3] Implement frame extraction and cache logic in `av_fusion/scripts/extract_gpt4o_features.py` — use cv2.VideoCapture to sample --sample-rate evenly-spaced frames from each video clip; encode each frame as base64 JPEG (quality=85); check per-frame cache file at av_fusion/gpt4o_cache/{clip_id}_{frame_idx}.json before making API call; skip already-cached frames; handle OpenCV failures (video unreadable, 0-frame clips) by writing NaN row and logging warning
- [X] T009 [US3] Implement OpenAI API call with structured output and exponential backoff in `av_fusion/scripts/extract_gpt4o_features.py` — call openai.chat.completions.create with response_format={"type":"json_object"} and system prompt enforcing schema {child_visible: yes|no|uncertain, child_vocalizing: yes|no|uncertain, n_children_visible: 0-3, visual_quality: good|medium|poor, notes: str}; exponential backoff (wait 2^attempt seconds) up to 5 retries on RateLimitError/APIError; on malformed JSON: log warning, write NaN for all structured fields, save raw text to gpt4o_reasoning; save raw API response dict to av_fusion/gpt4o_cache/{clip_id}_{frame_idx}.json
- [X] T010 [US3] Implement per-clip aggregation and output writing in `av_fusion/scripts/extract_gpt4o_features.py` — aggregate frame-level results to GPT4oFeatureRow (data-model.md schema): child_visible_gpt4o = fraction of yes frames, child_vocalizing_gpt4o = fraction of yes frames, n_children_visible_mean = mean count, visual_quality_gpt4o = mean score (good=1.0/medium=0.5/poor=0.0), gpt4o_reasoning = concatenated notes, n_frames_sampled = non-error frame count, n_frames_api_error = error count, model_used, cost_usd_estimate; write output CSV incrementally (append-safe; re-running skips clips already in output file using clip_id deduplication)

**Checkpoint**: GPT-4o features extracted for SAILS BIDS clips; NaN for audio-only; idempotent re-runs.

---

## Phase 6: User Story 4 — LocoNet and Light-ASD Frontends (Priority: P2)

**Goal**: Two additional ASD model backends selectable via `--model {loconet,light_asd}` in the existing `extract_asd_features.py`; same ASDFeatureRow output schema as TalkNet; inference routed through the `video/` Python 3.10 subprocess env.

**Independent Test**: Run `python av_fusion/scripts/extract_asd_features.py --model loconet --metadata-csv ... --output asd_features_loconet.csv`; confirm same column schema as TalkNet output; run with missing checkpoint to confirm FileNotFoundError with setup instructions.

- [X] T011 [US4] Extend argparse in `av_fusion/scripts/extract_asd_features.py` — add --model {talknet,loconet,light_asd} (default: talknet), --loconet-checkpoint (required when model=loconet), --light-asd-checkpoint (required when model=light_asd), --face-cache-dir (default: av_fusion/face_track_cache/), --batch-size (default: 16), --device (default: cuda); validate checkpoint existence at startup — raise FileNotFoundError("LocoNet checkpoint not found at {path}. Download with: huggingface-cli download Superxixixi/LoCoNet_ASD --local-dir video/LoCoNet_ASD/") for loconet; raise FileNotFoundError("Light-ASD checkpoint not found at {path}. Clone with: git clone https://github.com/Junhua-Liao/Light-ASD video/Light-ASD") for light_asd
- [X] T012 [US4] Implement LocoNet inference branch in `av_fusion/scripts/extract_asd_features.py` — call `video/run_asd.py --model loconet --checkpoint {loconet_checkpoint} --face-cache-dir {face_cache_dir} --input-csv {temp_manifest} --output-csv {temp_output}` as subprocess (same pattern as existing TalkNet subprocess call); parse per-clip output CSV columns to ASDFeatureRow schema (max_asd_score_any_face, mean_asd_score_any_face, max_asd_score_target_candidate, mean_asd_score_target_candidate, fraction_frames_active_speaker, n_active_speaker_tracks, asd_confidence_summary); add asd_model=loconet column; NaN row on missing video with warning
- [X] T013 [US4] Implement Light-ASD inference branch in `av_fusion/scripts/extract_asd_features.py` — call `video/run_asd.py --model light_asd --checkpoint {light_asd_checkpoint} --face-cache-dir {face_cache_dir} --input-csv {temp_manifest} --output-csv {temp_output}` as subprocess; parse per-clip output CSV to ASDFeatureRow schema with asd_model=light_asd column; NaN row on missing video with warning
- [X] T014 [US4] Add LocoNet runner to `video/run_asd.py` — add `--model loconet` branch that loads LoCoNet model from video/LoCoNet_ASD/ using its pretrained_model.py; reads face tracks from face-cache-dir JSON (same format as TalkNet uses); runs LocoNet forward pass in batches of --batch-size; outputs per-clip CSV with same column names as existing TalkNet output (max_asd_score_any_face, mean_asd_score_any_face, max_asd_score_target_candidate, mean_asd_score_target_candidate, fraction_frames_active_speaker, n_active_speaker_tracks, asd_confidence_summary)
- [X] T015 [US4] Add Light-ASD runner to `video/run_asd.py` — add `--model light_asd` branch that loads Light-ASD from video/Light-ASD/ using its model.py; reads face tracks from face-cache-dir JSON; runs Light-ASD forward pass in batches of --batch-size; outputs per-clip CSV with same column names as TalkNet output

**Checkpoint**: LocoNet and Light-ASD ASD features extractable; same schema as TalkNet; subprocess isolation via video/ env preserved.

---

## Phase 7: User Story 5 — Ego4D Reference Experiment (Priority: P3)

**Goal**: Zero-shot ASD evaluation on Ego4D AVD subset; before/after comparison with base TalkNet; exits gracefully with documentation if data not found.

**Independent Test**: Produce `ego4d_experiment_results.csv` or `ego4d_adaptation_report.md` with documented rationale if data unavailable.

- [X] T016 [US5] Create `av_fusion/scripts/ego4d_experiment.py` — argparse for --ego4d-metadata-csv (path to Ego4D AV annotation CSV), --output (ego4d_experiment_results.csv path), --asd-model {talknet,loconet}, --n-clips (default: 50, for zero-shot eval subset), --run-name; if --ego4d-metadata-csv path does not exist: write a Ego4DExperimentRecord row with adaptation_type=not_run, notes explaining access requirements (ego4d-data.org registration + `pip install ego4d` + `ego4d --datasets full_scale --benchmarks AV`), exit 0; if data exists: load N clips, run zero-shot ASD via subprocess to extract_asd_features.py, compute AUROC against Ego4D active-speaker labels, write Ego4DExperimentRecord (data-model.md schema) with experiment_id, asd_model, adaptation_type=zero_shot, ego4d_subset, val_auroc_home_video=NaN (not tested on child video yet), baseline_auroc=ASD AUROC on Ego4D
- [X] T017 [US5] Create `av_fusion/av_results/ego4d_adaptation_report.md` template — document: which Ego4D subset to request (AV/AVD, ~50h annotated), Python CLI install (`pip install ego4d`), download command, expected AUROC comparison table format (TalkNet zero-shot vs LocoNet zero-shot vs LocoNet Ego4D-finetuned), rationale for why Ego4D addresses domain gap; this file is committed as the standalone artifact when the experiment cannot be run

**Checkpoint**: Ego4D pathway documented; script exits gracefully whether or not data is accessible.

---

## Phase 8: User Story 6 — 1kd Dataset Compatibility Check (Priority: P3)

**Goal**: Schema compatibility check against existing clip format; JSON report covering access status, compatible columns, age-range overlap; exit 0 always; documentation fallback if data not found.

**Independent Test**: Run `python av_fusion/scripts/1kd_integration.py --data-dir /nonexistent/ --output 1kd_report.json`; confirm report written with status=not_found and notes; exit code 0.

- [X] T018 [US6] Implement `av_fusion/scripts/1kd_integration.py` — argparse for --data-dir, --output, --dry-run; check if --data-dir exists; if not: write JSON {status: not_found, n_clips: 0, missing_columns: [], age_range_overlap: [], notes: "..."}, exit 0; if exists: load annotation CSV from data-dir; check for required columns (clip_id, child_id, audio_path, label, timepoint); identify missing columns; compute age range from timepoint values; check overlap with existing dataset age ranges (14_month, 36_month); count compatible clips
- [X] T019 [US6] Add documentation fallback and report writing to `av_fusion/scripts/1kd_integration.py` — always write JSON report with status (compatible|incompatible|not_found), n_clips, missing_columns list, age_range_overlap list, notes string; when not_found or incompatible: set notes to human-readable explanation citing known 1kd datasets (1000 Days project — Brown University/NICHD naturalistic home recordings; access via institutional agreement), relevant publications, and data request pathway; --dry-run flag: check schema only, do not copy any files, just write report

**Checkpoint**: 1kd compatibility check runnable with or without data access; JSON report produced in all cases.

---

## Phase 9: Polish & Cross-Cutting Concerns

**Purpose**: Feature table integration and documentation updates

- [X] T020 Update `av_fusion/scripts/build_av_feature_table.py` — add optional `--gpt4o-features-csv` arg (path to gpt4o_features.csv); if provided: left-join GPT4oFeatureRow columns (child_visible_gpt4o, child_vocalizing_gpt4o, n_children_visible_mean, visual_quality_gpt4o) into av_{train,val,test}.csv on clip_id; NaN for clips without GPT-4o features (audio-only); add optional `--asd-features-csv` arg(s) (one per model, e.g. --asd-features-csv loconet:path/to/asd_features_loconet.csv); if provided: left-join max_asd_score_target_candidate as column asd_{model}_max_score; all existing behavior unchanged when new args absent
- [X] T021 [P] Update CLAUDE.md — add 007-av-extensions to Recent Changes and Active Technologies sections; add new scripts to Key Commands (extract_gpt4o_features.py, train_cascaded_pipeline.py, smooth_predictions.py, extended extract_asd_features.py); add LocoNet and Light-ASD to setup instructions under Video ASD section; add av_fusion/av_results/{run_name}/ extended file layout (gpt4o_features.csv, asd_features_loconet.csv, asd_features_light_asd.csv, cascade_stage_breakdown.csv, predictions_test_smoothed.csv, ego4d_experiment_results.csv, 1kd_integration_report.json)

**Checkpoint**: All 007 tasks complete; feature table optionally extended with GPT-4o and new ASD features; CLAUDE.md reflects new commands and output layout.

---

## Phase 10: LocoNet + ECAPA Speaker Identity Frontend (Branch 005-mil-extensions)

**Goal**: Replace the smallest-face heuristic with per-track LocoNet ASD + ECAPA speaker similarity for target-child identification; alternative to TS-TalkNet when its checkpoint is unavailable.

- [X] T022 Implement `run_loconet_asd_per_track()` in `video/run_asd.py` with `--output_tracks_json` flag — runs LocoNet independently on every face track, returns per-track JSON with `{track_id, mean_area, segments:[{start,end}]}`; wired into `--model loconet` branch alongside existing smallest-face RTTM output
- [X] T023 Implement `LocoNetECAPAFrontend` in `pyannote/video_asd.py` — loads ECAPA via speechbrain on init; checks/fills per-track JSON cache via `run_asd.py --output_tracks_json`; embeds reference audio and each track's active speech segments; picks best-cosine-similarity track as target child; falls back to smallest-face when no reference available; registered as `loconet_ecapa` in `pyannote/unified.py` with `video_loconet_checkpoint` config field
- [ ] T024 Run LocoNet ECAPA enrollment: `sbatch pyannote/run_loconet_ecapa_enrollment.sh` (SLURM job 12615544, 24h); results to `video_asd_ecapa_enrollment_runs/loconet_ecapa/` with `enroll_test_metrics.json`
- [ ] T025 [P] Log LocoNet ECAPA enrollment results in CLAUDE.md results table (F1/AUROC/AUPRC) once T024 completes
- [ ] T026 [P] TS-TalkNet checkpoint acquisition: contact `jiang_yidi@outlook.com` for `video/pretrain/ts_talknet.model` and `video/TS-TalkNet/exps/pretrain.model`; once received, run `sbatch pyannote/run_ts_talknet_enrollment.sh`; results to `video_asd_ecapa_enrollment_runs/ts_talknet/`

**Checkpoint**: LocoNet ECAPA enrollment results committed; TS-TalkNet checkpoint acquired or acquisition documented.

---

## Dependencies & Execution Order

### Phase Dependencies

- **Phase 1 (Setup)**: No dependencies — start immediately
- **Phase 2 (Foundational)**: N/A — only T001 needed; already in Phase 1
- **Phase 3 (US1)**: Requires T001 (config) and existing 006 av_{train,val,test}.csv
- **Phase 4 (US2)**: Requires T001; T006 has a soft dependency on T004 (same file — evaluate_av_fusion.py)
- **Phase 5 (US3)**: Requires T001 (GPT-4o config section); fully independent of US1/US2
- **Phase 6 (US4)**: Requires T001 (ASD model config section); fully independent of US1/US2/US3
- **Phase 7 (US5)**: Requires US4 scripts to exist (runs extract_asd_features.py); otherwise independent
- **Phase 8 (US6)**: No dependencies on other stories
- **Phase 9 (Polish)**: T020 requires US3 (gpt4o_features.csv schema) and US4 (ASDFeatureRow schema); T021 requires all prior phases

### User Story Dependencies

- **US1 (P1)**: Can start after T001 — no other story dependency
- **US2 (P1)**: Can start after T001 — T006 modifies same file (evaluate_av_fusion.py) as T004; T006 must run after T004
- **US3 (P2)**: Can start after T001 — independent of US1/US2
- **US4 (P2)**: Can start after T001 — independent of US1/US2/US3
- **US5 (P3)**: Soft dependency on US4 (calls extract_asd_features.py subprocess); fully fallback-safe without it
- **US6 (P3)**: No dependencies

### Same-File Constraints (must be sequential)

- T004 → T006: both modify `av_fusion/scripts/evaluate_av_fusion.py`
- T012 → T013: both add branches to `av_fusion/scripts/extract_asd_features.py` (after T011 adds args)
- T014 → T015: both add runner functions to `video/run_asd.py`

### Parallel Opportunities

- T003 [US1] and T005 [US2] and T007 [US3] and T011 [US4] can all start simultaneously after T001
- T007, T008, T009, T010 [US3] are sequential (same file) but US3 is independent of US1/US2/US4
- T016 and T017 [US5] are independent (different files)
- T018 and T019 [US6] are sequential (same file) but US6 is independent of all other stories
- T021 [P] (CLAUDE.md) can run in parallel with T020

---

## Parallel Example: US1 + US2 + US3 + US4 (after T001)

```bash
# Start all P1 stories simultaneously:
Task A: "Implement train_cascaded_pipeline.py (T003)"
Task B: "Implement smooth_predictions.py (T005)"

# Start all P2 stories simultaneously (different files):
Task C: "Implement extract_gpt4o_features.py argument parsing (T007)"
Task D: "Extend extract_asd_features.py argparse (T011)"

# After Task A, Task B, Task C complete:
Task E: "Extend evaluate_av_fusion.py for cascade (T004)"  # after T003
Task F: "Extend evaluate_av_fusion.py for smoothing (T006)"  # after T004
```

---

## Implementation Strategy

### MVP First (US1 Only)

1. Complete Phase 1: T001 (config)
2. Complete Phase 3: T003 + T004
3. **STOP and VALIDATE**: `train_cascaded_pipeline.py` runs; `evaluate_av_fusion.py --cascade-breakdown` runs; `metrics_cascade_by_stage.csv` shows non-NaN AUROC per stage
4. MVP deliverable: cascaded pipeline with stage breakdown table

### Incremental Delivery

1. T001 → Foundation ready
2. T003–T004 (US1) → Cascade pipeline + evaluation → validate SC-001
3. T005–T006 (US2) → Temporal smoothing → validate SC-002
4. T007–T010 (US3) → GPT-4o features → validate SC-003, SC-004
5. T011–T015 (US4) → LocoNet/Light-ASD → validate SC-005
6. T016–T017 (US5) → Ego4D experiment (best-effort) → validate SC-006
7. T018–T019 (US6) → 1kd check → validate SC-007
8. T020–T021 → Feature table integration + CLAUDE.md → validate SC-008

### Priority Order

US1 (cascade) and US2 (smoothing) are P1 and can be parallelized. Both must be complete before Polish (T020 references their outputs). US3 and US4 are P2 and are independent from each other. US5 and US6 are P3 and purely additive.

---

## Notes

- No test tasks generated — no tests were requested in the specification
- [P] tasks can be implemented concurrently (different files, no dependency)
- Each US phase is independently testable per the spec's Independent Test criteria
- T004 and T006 both modify `evaluate_av_fusion.py` — implement T004 first, then T006 as an additive extension
- T014 and T015 both modify `video/run_asd.py` — implement sequentially; check existing TalkNet function signature first to match output format
- av_extensions.yaml (T001) is the sole shared prerequisite; all story scripts read from it
- GPT-4o extraction (US3) requires OPENAI_API_KEY at runtime; set before running T009
- LocoNet and Light-ASD checkpoints must be downloaded before T012/T013 can be tested end-to-end
