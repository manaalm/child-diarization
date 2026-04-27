# Tasks: Audio-Visual Target-Child Vocalization Detection

**Input**: Design documents from `specs/006-av-child-vocalization/`
**Branch**: `006-av-child-vocalization`

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (US1–US6)

---

## Phase 1: Setup

**Purpose**: Create the `av_fusion/` module skeleton and config before any implementation.

- [X] T001 Create `av_fusion/` directory structure per plan.md: `scripts/`, `configs/`, `slurm/`, `av_results/.gitkeep`, `face_track_cache/.gitkeep`
- [X] T002 Create `av_fusion/configs/av_fusion.yaml` with: model HPs (XGBoost `n_estimators=100, max_depth=3, learning_rate=0.1`; LR `C=1.0`), feature column lists per model class (audio_only, visual_only, always_fuse, gated_av), `seed: 42`, `sample_fps: 2`, `visual_eligibility_threshold_method: balanced_accuracy`, `audio_score_col: enroll_proba`
- [X] T003 [P] Create `av_fusion/slurm/run_av_pipeline.sh` SLURM script: `#SBATCH -t 48:00:00 --gres=gpu:1 -p ou_bcs_normal,pi_satra --mem=40G -c 4 -o logs/av_fusion/av_pipeline_%j.out`; sources conda env `child-vocalizations`; runs `python av_fusion/scripts/extract_visual_features.py` with face cache

---

## Phase 2: Foundational (Blocking Prerequisite)

**Purpose**: Shared utility module and verified data availability. Required before any user story.

**⚠️ CRITICAL**: US1–US6 cannot proceed until T004–T005 are complete.

- [X] T004 Create `av_fusion/scripts/utils.py` with shared helpers: `compute_metrics(y_true, y_score, threshold)` → dict with AUROC/AUPRC/F1/precision/recall/balanced_accuracy; `tune_threshold_f1(y_true, y_score)` → (threshold, val_f1); `tune_threshold_balanced_acc(y_true, y_score)` → (threshold, val_bacc); `assert_split_integrity(df)` → raises ValueError if any child_id spans multiple splits; `save_json(d, path)`, `load_feature_csv(path)` with NaN-safe dtypes
- [X] T005 Verify audio baseline availability: inspect `babar_ecapa_enrollment_runs/` for a prediction CSV containing `enroll_proba` column and `audio_path` or `clip_id` join key; document the exact filename in `av_fusion/configs/av_fusion.yaml` under `audio_scores_csv:` — if BabAR predictions aren't available, use WavLM baseline from `baselines/baseline_results/wavlm_attn/val_predictions.csv` instead

**Checkpoint**: Foundation ready — shared utils implemented, audio baseline path confirmed.

---

## Phase 3: User Story 1 — Visual Feature Extraction Pipeline (Priority: P1)

**Goal**: Run face detection + tracking on SAILS BIDS video clips; produce `visual_features.csv` with one row per clip.

**Independent Test**: Run `python av_fusion/scripts/extract_visual_features.py --metadata-csv whisper-modeling/seen_child_splits/test.csv --output /tmp/vf_test.csv --sample-fps 1` on a 10-clip subset; confirm `/tmp/vf_test.csv` has 10 rows with `n_face_tracks`, `visual_eligibility_score`, `off_camera_likely_score` columns and no crash for clips with `BidsProcessed = NaN`.

- [X] T006 [P] [US1] Implement `av_fusion/scripts/face_utils.py`: `YuNetDetector` class (wraps `cv2.FaceDetectorYN`; returns list of `(x,y,w,h,conf)` per frame); `IouCentroidTracker` class (assigns detections to tracks by IoU overlap ≥ 0.3; returns dict of `track_id → [(frame_idx, bbox)]`); `visual_quality_score(frames)` → float (mean Laplacian variance normalized to [0,1] clipped at 300); `child_candidate_score(tracks)` → float (fraction of frames where smallest-area face track is present); eligibility formula: `0.40 * child_visible + 0.25 * track_fraction + 0.20 * quality + 0.15 * confidence`
- [X] T007 [US1] Implement `av_fusion/scripts/extract_visual_features.py` — CLI per `contracts/cli_contracts.md` (Script 1); reads `BidsProcessed` column for video path; samples frames at `--sample-fps`; calls `face_utils`; computes all `AutomaticVisualFeatures` fields from `data-model.md`; clips with missing video get NaN for all face fields and `off_camera_likely_score = 1.0`; writes per-clip detection JSON to `--face-cache-dir` (skip if cached); assembles and writes `visual_features.csv`; is idempotent (skip clips already in cache)

**Checkpoint**: US1 complete — `visual_features.csv` produced; spot-check that clips where `Child_of_interest_clear == "yes"` tend to have high `visual_eligibility_score`.

---

## Phase 4: User Story 2 — Audio-Visual Feature Table Assembly (Priority: P1)

**Goal**: Merge metadata, labels, manual BIDS annotations, audio baseline scores, and (optionally) visual features into `av_master_features.csv`; assert split integrity.

**Independent Test**: Run `python av_fusion/scripts/build_av_feature_table.py --metadata-csv whisper-modeling/seen_child_splits/master_with_split.csv --audio-scores-csv <baseline_csv> --audio-score-col enroll_proba --output-dir /tmp/av_test/`; confirm `split_integrity_report.json` has `leakage_detected: false`; confirm `av_master_features.csv` has same row count as metadata CSV and includes `child_of_interest_clear_binary`, `manual_face_visibility_norm`, `visual_eligibility_score` columns.

**Note**: US2 MVP requires only the metadata CSV + audio scores — `--visual-features-csv` is optional. This is the fast-track path for immediate model training.

- [X] T008 [US2] Implement `av_fusion/scripts/build_av_feature_table.py` — CLI per `contracts/cli_contracts.md` (Script 3); load metadata CSV; extract ManualVisualAnnotation fields; compute derived fields: `child_of_interest_clear_binary` (1 if "yes"), `manual_face_visibility_norm` (`Video_Quality_Child_Face_Visibility / 10`), `manual_quality_norm` (`(Lighting + Resolution) / 20`), `n_people_total` (`#_adults + #_children`), `multi_person_clip` (1 if > 1), `age_band` ("14_18_months" or "34_38_months" from `timepoint_norm`); resolve `video_path` from `BidsProcessed` (fallback to `BidsRaw`, else None); join audio scores on `audio_path`; optionally join `visual_features.csv` and `asd_features.csv` on `clip_id`; compute `visual_eligibility_score` — if automatic features present use formula from `face_utils`, else use `0.6 * manual_face_visibility_norm + 0.4 * manual_quality_norm`; call `assert_split_integrity(df)`; write `av_master_features.csv`, `av_train/val/test.csv`, `feature_manifest.json`, `split_integrity_report.json`

**Checkpoint**: US2 complete — master feature table assembled; `split_integrity_report.json` confirms no leakage; US3 can proceed immediately.

---

## Phase 5: User Story 3 — Fusion Model Training (Priority: P2)

**Goal**: Train audio-only, video-only, always-fuse AV, and gated AV classifiers on training split; tune thresholds on validation split.

**Independent Test**: Run `python av_fusion/scripts/train_av_fusion.py --feature-dir <outdir> --output-dir <outdir>/models --config av_fusion/configs/av_fusion.yaml --seed 42`; confirm 4 pkl files exist; load each and call `predict_proba(av_val_df.head(5))` without error; confirm `val_metrics.json` has non-NaN AUROC for all four models.

- [X] T009 [US3] Implement `av_fusion/scripts/train_av_fusion.py` — CLI per `contracts/cli_contracts.md` (Script 4); define feature column sets per model class from `av_fusion.yaml` config; fit XGBoost (primary, NaN-safe) and LogisticRegression (interpretable) per model class using train split with `class_weight='balanced'` (LR) or `scale_pos_weight` (XGBoost); tune binary classification threshold per model on val using `tune_threshold_f1()` from `utils.py`; tune `visual_eligibility_threshold` on val using `tune_threshold_balanced_acc()` on `child_of_interest_clear_binary` as a proxy for visual eligibility; the gated AV model pkl is identical to always_fuse but `predict()` checks `visual_eligible` at inference — implement a `GatedAVModel` wrapper class that wraps always_fuse pkl + threshold and applies audio_only score when `visual_eligible == 0`; save `audio_only.pkl`, `video_only.pkl`, `always_fuse_av.pkl`, `gated_av.pkl`, `visual_eligibility_threshold.json` (`{"threshold": float, "val_balanced_acc": float}`), `val_metrics.json`, `config.json` to `--output-dir`

**Checkpoint**: US3 complete — four models trained; val AUROC values confirm audio-only baseline is established; gated model produces different predictions from always-fuse on visually ineligible clips.

---

## Phase 6: User Story 4 — Evaluation and Stratified Reporting (Priority: P2)

**Goal**: Evaluate all four models on held-out test split; produce full stratified metrics and thesis-ready figures.

**Independent Test**: Run `python av_fusion/scripts/evaluate_av_fusion.py --feature-dir <outdir> --model-dir <outdir>/models --output-dir <outdir> --plot`; confirm `metrics_overall.json` has AUROC/AUPRC/F1 for all four model classes; confirm `metrics_by_age_band.csv` has rows for both age bands; confirm `figures/roc_curve.png` exists.

- [X] T010 [US4] Implement `av_fusion/scripts/evaluate_av_fusion.py` — CLI per `contracts/cli_contracts.md` (Script 5); load `av_test.csv` and all four pkl files; load `visual_eligibility_threshold.json`; apply `visual_eligible` flag to test clips (score ≥ threshold); run each model's `predict_proba` on test set (gated model switches to audio_only probability when `visual_eligible == 0`); compute all metrics via `compute_metrics()` from `utils.py`; write `metrics_overall.json` and `predictions_test.csv` (columns: `clip_id`, `child_id`, `age_band`, `visual_eligible`, `label`, `proba_audio_only`, `proba_video_only`, `proba_always_fuse`, `proba_gated_av`, `pred_audio_only`, `pred_video_only`, `pred_always_fuse`, `pred_gated_av`)
- [X] T011 [US4] Add stratified metrics and figures in `av_fusion/scripts/evaluate_av_fusion.py`: compute metrics per `age_band` (14_18_months, 34_38_months) → `metrics_by_age_band.csv`; per `visual_eligible` (0, 1) → `metrics_by_visual_eligibility.csv`; per strata (off_camera: `off_camera_likely_score > 0.7`; multi_person: `multi_person_clip == 1`; low_quality: `manual_quality_norm < 0.5`) → `metrics_by_strata.csv`; with `--plot` flag: generate `pr_curve.png` (precision-recall curves for all 4 models), `roc_curve.png` (ROC curves), `stratified_bar_metrics.png` (AUROC bar chart by stratum and model), `visual_eligibility_histogram.png` (score distribution by label); all figures to `figures/` subdirectory

**Checkpoint**: US4 complete — test metrics available; gated vs. always-fuse comparison on visually eligible subset visible; null or conditional result clearly reported.

---

## Phase 7: User Story 5 — Error Analysis (Priority: P3)

**Goal**: Categorize test clips by failure mode (AV-helped, AV-hurt, off-camera miss, multi-face) and produce structured tables for thesis discussion.

**Independent Test**: Run `python av_fusion/scripts/error_analysis_av.py --predictions-csv <outdir>/predictions_test.csv --feature-dir <outdir> --output-dir <outdir>`; confirm `error_analysis_examples.csv` exists; confirm `error_analysis_summary.json` has entries for all 6 error categories (some may have n=0).

- [X] T012 [P] [US5] Implement `av_fusion/scripts/error_analysis_av.py` — CLI per `contracts/cli_contracts.md` (Script 6); load `predictions_test.csv` and `av_test.csv`; define audio-only prediction as binary via val-tuned threshold (loaded from `models/val_metrics.json`); define error mode categories: `av_helped_fp` (audio pred=1, label=0, gated pred=0), `av_helped_fn` (audio pred=0, label=1, gated pred=1), `av_hurt_fp` (audio pred=0 or TN, gated pred=1, label=0), `av_hurt_fn` (audio pred=1 or TP, gated pred=0, label=1), `off_camera_miss` (label=1 AND `off_camera_likely_score > 0.7` AND gated pred=0), `multi_face_ambiguous` (`multi_person_clip == 1` AND audio pred ≠ gated pred AND label=1); for each category select top `--n-examples` clips sorted by abs(proba_audio_only - proba_gated_av); write `error_analysis_examples.csv` (columns: `clip_id`, `child_id`, `age_band`, `error_mode`, `label`, `proba_audio_only`, `proba_gated_av`, `visual_eligible`, `off_camera_likely_score`, `multi_person_clip`, `visual_eligibility_score`, `manual_face_visibility_norm`); write `error_analysis_summary.json` (count + mean proba_delta per mode)

**Checkpoint**: US5 complete — failure mode breakdown available; identifies which audio-only failure modes are fixed vs. worsened by AV fusion.

---

## Phase 8: User Story 6 — Optional ASD Feature Extraction (Priority: P4)

**Goal**: Optionally compute TalkNet-ASD scores per clip and feed them into the master feature table for improved fusion.

**Independent Test**: Run `python av_fusion/scripts/extract_asd_features.py --metadata-csv whisper-modeling/seen_child_splits/test.csv --output /tmp/asd_test.csv` on a 5-clip subset with known face tracks; confirm `/tmp/asd_test.csv` has 5 rows and non-NaN `max_asd_score_any_face` for clips where faces were detected.

- [X] T013 [P] [US6] Implement `av_fusion/scripts/extract_asd_features.py` — CLI per `contracts/cli_contracts.md` (Script 2); reuse subprocess pattern from `pyannote/video_asd.py` (call `video/run_asd.py --model talknet_asd --video <path> --audio <path>`) with the isolated `video/` Python 3.10 uv env; load per-clip TalkNet output scores; identify child-candidate track as smallest-face track ID (from `visual_features.csv` if provided via `--visual-features-csv`, else re-detect); compute all `ASDFeatures` fields from `data-model.md`; clips with no detected faces have all scores 0.0; exit code 1 if `video/pretrain/talknet_asd.model` checkpoint not found; write `asd_features.csv`

**Checkpoint**: US6 complete — ASD features available; re-run build_av_feature_table.py with `--asd-features-csv` to add ASD columns; re-run training to compare ASD vs. no-ASD fusion.

---

## Phase 9: Polish & Cross-Cutting Concerns

**Purpose**: Documentation and thesis table wiring.

- [X] T014 [P] Update CLAUDE.md `av_fusion/` section: document module architecture (6 scripts, face_utils, utils, face_track_cache), key commands (all 5 pipeline steps), result layout (`av_results/{run}/`), and note that manual BIDS annotations from split CSV enable a no-video MVP; add to Key Commands block
- [X] T015 [P] Verify `evaluation/configs/thesis_tables.yaml` format and add `table_av_fusion` entry pointing to `av_fusion/av_results/{run}/metrics_overall.json` for thesis table generation; run `python evaluation/aggregate_thesis_tables.py --skip-missing` to confirm no crash

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies — start immediately
- **Foundational (Phase 2)**: Depends on Phase 1
- **US1 (Phase 3)**: Depends on T004 (face_utils needs utils.py); T006 and T007 are sequential within US1
- **US2 (Phase 4)**: Depends on T004 only — **does NOT require US1 completion for MVP** (manual annotations path)
- **US3 (Phase 5)**: Depends on US2 completion (needs av_train.csv/av_val.csv)
- **US4 (Phase 6)**: Depends on US3 completion (needs trained pkl files)
- **US5 (Phase 7)**: Depends on US4 completion (needs predictions_test.csv)
- **US6 (Phase 8)**: Depends on T004 only; independent of US1–US5; feeds back into US2 if ASD features desired
- **Polish (Phase 9)**: Depends on US4 completion

### MVP Execution Order (Fast-Track without Video Processing)

```
Phase 1 → Phase 2 → US2 (T008) → US3 (T009) → US4 (T010, T011) → US5 (T012)
```
Skip US1 and US6 entirely for the first experiment run. Use manual BIDS annotations as all visual features.

### Full Execution Order

```
Phase 1 → Phase 2 → US1 (T006, T007) + US2 (T008) [parallel]
         → US3 (T009) → US4 (T010, T011) → US5 (T012)
         → US6 (T013) [optional, run in parallel with US1 after Phase 2]
```

### User Story Dependencies

- **US1 (P1)**: Depends on T004 only
- **US2 (P1)**: Depends on T004 only (manual-only MVP path); optionally consumes US1 output
- **US3 (P2)**: Depends on US2 completion
- **US4 (P2)**: Depends on US3 completion
- **US5 (P3)**: Depends on US4 completion
- **US6 (P4)**: Depends on T004 only; independent of all other stories

### Parallel Opportunities

- T003 (SLURM script) ‖ T004 (utils.py) ‖ T005 (baseline verification) — all in Phase 2
- T006 (face_utils.py) ‖ T008 (build_av_feature_table.py) — after T004 completes, US1 and US2 MVP can start simultaneously
- T012 (error_analysis_av.py) ‖ T013 (extract_asd_features.py) — different files, independent
- T014 (CLAUDE.md) ‖ T015 (thesis_tables.yaml) — different files

---

## Parallel Example: US1 + US2 MVP (Recommended Start)

```bash
# After Phase 2 completes, launch US1 and US2 simultaneously:
Task: "Implement face_utils.py (T006)"         # US1 visual utils
Task: "Implement build_av_feature_table.py (T008)"  # US2 MVP fast-track

# US2 can be run and tested before US1 completes — just skip --visual-features-csv
```

---

## Implementation Strategy

### MVP (US2 + US3 + US4 — manual annotations only)

1. Phase 1+2: Setup and foundation
2. T008 (US2): Build master feature table with manual annotations — no video needed
3. T009 (US3): Train four model classes — proves the pipeline works end-to-end
4. T010–T011 (US4): Evaluate — produces thesis-ready stratified results
5. **STOP and VALIDATE**: Check if audio-only is strong baseline; check if manual-annotation AV helps on visually eligible subset
6. If promising: run US1 to add automatic face features and retrain

### Full Scope (US1–US5)

1. MVP above → T006–T007 (US1 face extraction) → re-run T008 with `--visual-features-csv` → retrain (T009) → re-evaluate (T010–T011) → compare manual-only vs. manual+auto
2. T012 (US5 error analysis) → identify failure modes
3. Optional: T013 (US6 ASD) → re-run US2–US4 with ASD features

### Out of Scope

- End-to-end AV model training from scratch
- AV-HuBERT / VideoMAE embeddings (too costly for 1500-clip training set)
- Manual frame-level annotation at scale (only small diagnostic subset)
- Real-time deployment optimization
