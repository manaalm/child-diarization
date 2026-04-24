# Feature Specification: AV Extended Experiments — Cascaded Detection, GPT-4o, LocoNet/AS-Net, Ego4D, Temporal Smoothing

**Feature Branch**: `007-av-extensions`  
**Created**: 2026-04-24  
**Status**: Draft  
**Project**: MIT EECS MEng Thesis — Audio-Visual Target-Child Vocalization Detection  
**Extends**: `006-av-child-vocalization` — builds on the existing AV pipeline with new models, datasets, and processing strategies

---

## User Scenarios & Testing *(mandatory)*

### User Story 1 — Cascaded Detection Pipeline (Priority: P1)

A researcher runs a three-stage cascaded detector where each stage filters or scores the clip independently before combining evidence: (1) a voice activity detector flags whether any vocalization is present; (2) a target-child classifier determines whether the vocalizing speaker is the target child; (3) an audio-video fusion step incorporates visual evidence to produce the final probability. Each stage's output is logged so the contribution of each component can be measured.

**Why this priority**: The cascaded design is the most architecturally novel contribution in this extension. It separates the "is anyone speaking?" question from "is it the target child?" question — a decomposition that matches how researchers think about the problem and may better handle cases where audio-only is ambiguous. It is also independently evaluable: the VAD stage alone can be scored, the child ID stage alone can be scored, and so can the full cascade.

**Independent Test**: Run `train_cascaded_pipeline.py`; confirm three saved stage artifacts exist; run `evaluate_av_fusion.py --model cascaded_av`; confirm `metrics_overall.json` contains a `cascaded_av` row with non-NaN AUROC.

**Acceptance Scenarios**:

1. **Given** a clip where no voice is detected by the VAD stage, **When** the cascade runs, **Then** the clip is scored as non-target-child-vocalized without running the child ID or fusion stages, and this early-exit is logged per clip.
2. **Given** a clip where VAD detects speech, **When** the child ID stage runs, **Then** it produces a scalar probability that the detected speech belongs to the target child, using the existing ECAPA enrollment embeddings.
3. **Given** a clip where the child ID stage is uncertain (probability near 0.5), **When** the AV fusion stage runs, **Then** visual features (face visibility, eligibility) are incorporated to update the final probability.
4. **Given** a full cascade run on the test set, **When** per-stage outputs are examined, **Then** a summary table shows what fraction of clips were resolved at each stage (VAD no-speech, child ID confident, AV fusion needed).

---

### User Story 2 — Temporal Smoothing of Clip Predictions (Priority: P1)

A researcher applies temporal smoothing over the clip-level prediction sequence within each recording session. Adjacent clips from the same session are smoothed using a sliding window majority vote, a Gaussian kernel, or a simple CRF-style model trained on the val set. The smoothed probabilities are saved alongside the raw predictions for comparison.

**Why this priority**: Clip-level predictions from acoustic/visual models are noisy and may flip between adjacent clips even when the underlying child activity is continuous. Temporal smoothing is a lightweight post-processing step that can improve both precision and recall, requires no additional labeled data, and applies to any model in the pipeline. It is independently testable without retraining any model.

**Independent Test**: Run `smooth_predictions.py --predictions predictions_test.csv --method gaussian`; confirm `predictions_test_smoothed.csv` is written with a `prob_smoothed` column; confirm that metrics computed from `prob_smoothed` are logged alongside raw metrics.

**Acceptance Scenarios**:

1. **Given** raw clip-level predictions from any model, **When** temporal smoothing is applied, **Then** the smoothed sequence has fewer single-clip isolated spikes compared to the raw sequence, measured by a reduction in runs of alternating 0-1-0 or 1-0-1 predictions.
2. **Given** a recording session with a long contiguous span of child vocalizations, **When** temporal smoothing is applied, **Then** clips in the middle of that span that were false negatives in the raw predictions are more likely to be corrected.
3. **Given** smoothing is tuned only on the val set (kernel width or CRF weights), **When** the same parameters are applied to the test set, **Then** no test labels are used for parameter selection.
4. **Given** the smoothed and raw predictions, **When** both are evaluated, **Then** both AUROC and F1 (at tuned threshold) are reported side by side for each model so the effect of smoothing can be directly compared.

---

### User Story 3 — GPT-4o Vision for Child Detection in Frame (Priority: P2)

A researcher uses GPT-4o's vision capability to analyze sampled video frames from each clip and obtain structured answers about child presence: whether a child is visible, whether the child appears to be vocalizing, an estimate of the child's age band, and a visual quality assessment. These structured outputs are used as an alternative or supplement to programmatic face detection features, particularly for cases where the face detector fails or where contextual reasoning about the scene is needed.

**Why this priority**: Programmatic face detectors (YuNet, RetinaFace) fail on small, side-facing, or partially occluded infant/toddler faces. GPT-4o's visual reasoning can provide contextual cues (child body posture, toys present, interaction with adult) that no face detector can produce. As a diagnostic and complementary feature source, it can reveal what information is theoretically available in the video and establish an upper-bound on visual feature quality.

**Independent Test**: Run `extract_gpt4o_features.py --sample-rate 2` (2 frames per clip) on a 50-clip subset; confirm `gpt4o_features.csv` contains structured columns including `child_visible_gpt4o`, `child_vocalizing_gpt4o`, `visual_quality_gpt4o`, and `gpt4o_reasoning`; confirm no API errors crashed the run.

**Acceptance Scenarios**:

1. **Given** a clip where a toddler is visibly playing on camera, **When** GPT-4o analyzes sampled frames, **Then** `child_visible_gpt4o` is True and `child_vocalizing_gpt4o` is True or Unknown (not False with high confidence).
2. **Given** a clip that is entirely dark or shows only a wall/floor, **When** GPT-4o analyzes it, **Then** `child_visible_gpt4o` is False and `visual_quality_gpt4o` is low.
3. **Given** a frame where a child and adult are both visible, **When** GPT-4o responds, **Then** the structured output distinguishes the target child from other visible people using size/age cues.
4. **Given** GPT-4o features merged into `av_master_features.csv`, **When** the fusion model is trained using `child_visible_gpt4o` as a feature, **Then** the model can be trained without crashing; if GPT-4o features are missing for some clips, those features are treated as NaN.
5. **Given** cost constraints, **When** GPT-4o features are extracted, **Then** the script only samples `--sample-rate` frames per clip (default: 2) and logs the estimated API cost before proceeding.

---

### User Story 4 — LocoNet and AS-Net as ASD Frontends (Priority: P2)

A researcher integrates LocoNet (Local Context Network for Active Speaker Detection) and AS-Net as additional ASD model options within the existing ASD feature extraction pipeline. Each model produces per-clip ASD scores (probability that the most child-likely face track is actively speaking) that are used as features in the fusion models. Results are compared across TalkNet, LocoNet, and AS-Net to determine which ASD model best correlates with target-child vocalization in naturalistic child home videos.

**Why this priority**: TalkNet-ASD was primarily trained on adult broadcast data (movies, TV) and may not generalize well to naturalistic child home recordings. LocoNet and AS-Net are more recent ASD models that incorporate richer temporal context or different architectures. Comparing them is a direct ASD model ablation that contributes to understanding whether better ASD models translate to better child detection.

**Independent Test**: Run `extract_asd_features.py --model loconet` and `--model as_net` on a 50-clip subset; confirm per-model `asd_features_{model}.csv` files are written with the same column schema as the TalkNet output; confirm both can be merged into the master feature table without conflict.

**Acceptance Scenarios**:

1. **Given** LocoNet checkpoints are available, **When** `extract_asd_features.py --model loconet` runs, **Then** `asd_features_loconet.csv` is produced with all required ASD columns and the same row count as the input metadata.
2. **Given** AS-Net checkpoints are available, **When** `extract_asd_features.py --model as_net` runs, **Then** `asd_features_as_net.csv` is produced identically.
3. **Given** ASD features from TalkNet, LocoNet, and AS-Net are all available, **When** fusion models are trained with each, **Then** a comparison table of test AUROC per ASD model is produced showing which ASD frontend contributes most to fusion quality.
4. **Given** a clip where a child is visibly speaking, **When** all three ASD models score it, **Then** LocoNet and AS-Net `max_asd_score_target_candidate` values are recorded alongside TalkNet scores in a side-by-side diagnostic CSV.
5. **Given** an ASD model's checkpoint is missing, **When** `extract_asd_features.py` is run with that model, **Then** a clear FileNotFoundError with setup instructions is raised rather than a silent empty output.

---

### User Story 5 — Ego4D Dataset Integration for ASD Pretraining Reference (Priority: P3)

A researcher documents the Ego4D dataset as a reference/pretraining resource for ASD models and, if feasible within compute budget, fine-tunes or evaluates ASD model checkpoints on a subset of Ego4D's active speaker annotations before applying them to child home video clips. The experiment quantifies whether models adapted on Ego4D's egocentric perspective generalize better to naturalistic home video than models trained purely on broadcast data.

**Why this priority**: Ego4D's egocentric perspective (wearable camera, informal settings, variable lighting, multiple speakers) is the closest public large-scale dataset to naturalistic home recordings. Using it as a pretraining source for ASD models addresses the domain gap between broadcast-trained ASD models and home video conditions. However, it requires compute and storage investment that may not be feasible; this story is P3 because the minimum viable experiment (feature extraction + fusion) can proceed without it.

**Independent Test**: Produce `ego4d_adaptation_report.md` documenting: which Ego4D subset was used, which ASD model was adapted, before/after ASD scores on a held-out home-video validation subset, and a conclusion on whether Ego4D pretraining is worth the compute cost.

**Acceptance Scenarios**:

1. **Given** access to Ego4D's active speaker annotations (AV subset), **When** an ASD model is evaluated zero-shot on a 50-clip subset of child home video, **Then** a baseline ASD AUROC against manual child vocalization labels is recorded.
2. **Given** an Ego4D-adapted ASD model, **When** it is used to extract features via `extract_asd_features.py --model ego4d_adapted`, **Then** its ASD scores are compared to the base TalkNet model scores in a side-by-side CSV.
3. **Given** the experiment is not feasible (insufficient compute, inaccessible data), **When** this story is reported, **Then** a written rationale is included in the thesis appendix explaining why Ego4D was considered but not used, citing specific constraints.

---

### User Story 6 — 1kd Project Dataset Integration (Priority: P3)

A researcher investigates the 1000 Days (1kd) project dataset as a potential supplementary source of naturalistic home recordings with child and adult speech annotations. If the dataset is accessible and compatible with the existing split structure, clips from 1kd are merged into the feature table as additional training or evaluation data; if not accessible, the dataset is documented as a relevant related resource.

**Why this priority**: Additional naturalistic child home recording data directly addresses the primary limiting factor of the current pipeline: the small training set (~1,500 clips). However, access restrictions, annotation format differences, or age-range mismatches may prevent integration. P3 priority reflects that integration depends on conditions outside the researcher's control.

**Independent Test**: Produce `1kd_integration_report.md` documenting: dataset access status, annotation format compatibility, number of compatible clips, and either a merged dataset evaluation or a documented rationale for non-integration.

**Acceptance Scenarios**:

1. **Given** 1kd dataset access is granted, **When** clip metadata is examined, **Then** a compatibility check confirms whether clip_id, child_id, audio_path, label, and timepoint columns can be mapped to the existing schema.
2. **Given** compatible 1kd clips, **When** they are merged into the master feature table, **Then** no cross-dataset child_id conflicts exist; 1kd children are assigned to a separate child_id namespace; split assignment respects group-wise child isolation.
3. **Given** a merged dataset including 1kd clips, **When** all models are re-evaluated, **Then** results are reported both with and without 1kd data so the contribution of the additional data can be measured.
4. **Given** 1kd access is not available, **When** this story is reported, **Then** the access requirements, relevant publications, and data request process are documented for future researchers.

---

### Edge Cases

- What happens when GPT-4o returns a malformed or unexpected response for a frame? → The parser logs a warning and falls back to NaN for all structured fields from that clip; the pipeline continues.
- What happens when the cascade VAD stage produces very high sensitivity (misses almost nothing)? → The child ID stage processes nearly all clips; the cascade degrades gracefully to a two-stage system rather than crashing.
- What happens when temporal smoothing is applied across clips from different sessions or different children? → Smoothing is always scoped within a single (child_id, session_id) group; no information crosses child or session boundaries.
- What happens when LocoNet/AS-Net produce different face track IDs than TalkNet for the same clip? → ASD features are stored per-model independently; track ID normalization is not required since all models use the same face detection output.
- What happens when the Ego4D or 1kd datasets are unavailable? → Experiments are documented as infeasible with a written rationale; the rest of the pipeline proceeds unaffected.
- What happens when GPT-4o API costs exceed a reasonable budget mid-extraction? → The script supports `--max-clips N` to cap spending; completed features are saved incrementally so interrupted runs can resume from the last saved clip.
- What happens when the cascaded pipeline's VAD stage is too aggressive (marks too many clips as no-speech)? → Recall drops; this is reported explicitly in the cascade stage breakdown, allowing the threshold to be adjusted on the val set.

---

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The cascaded pipeline MUST implement three explicit, separately logged stages: VAD (any speech present), child ID (is the speaker the target child), and AV fusion (incorporate visual evidence); each stage's output probability MUST be saved per clip.
- **FR-002**: The cascaded pipeline MUST allow per-stage threshold tuning on the validation set; test-set thresholds MUST be identical to the val-tuned values.
- **FR-003**: Temporal smoothing MUST be applied as a post-processing step on top of any model's raw predictions without requiring model retraining; smoothing parameters MUST be tuned only on the val set.
- **FR-004**: The GPT-4o feature extractor MUST produce structured per-clip outputs with at minimum: `child_visible_gpt4o` (bool/float), `child_vocalizing_gpt4o` (bool/float), `visual_quality_gpt4o` (float), `gpt4o_reasoning` (text).
- **FR-005**: GPT-4o feature extraction MUST be resumable and idempotent: re-running the script must skip already-processed clips using a JSON or CSV cache.
- **FR-006**: GPT-4o feature extraction MUST log estimated API cost (tokens × price) before processing and support a `--dry-run` flag that prints cost without making API calls.
- **FR-007**: LocoNet and AS-Net MUST be implemented as drop-in additions to `extract_asd_features.py`, selectable via `--model {loconet,as_net,talknet}`; the output schema MUST be identical across all ASD models.
- **FR-008**: ASD model comparison MUST produce a single summary table reporting, for each ASD model, the correlation between `max_asd_score_target_candidate` and the ground truth label on the test set (AUROC, Pearson r).
- **FR-009**: Ego4D integration (if feasible) MUST NOT modify existing train/val/test splits for the primary dataset; Ego4D clips MUST be added only to the training portion and clearly flagged with a `dataset_source` column.
- **FR-010**: 1kd dataset integration MUST be conditional on access; the script MUST exit gracefully with a clear message if the data path does not exist.
- **FR-011**: All new model artifacts (cascaded stages, smoothing parameters, GPT-4o features) MUST be stored in `av_fusion/av_results/{run_name}/` following the existing output layout.
- **FR-012**: The cascaded pipeline MUST be evaluable with the existing `evaluate_av_fusion.py` script by adding `cascaded_av` as a model class; no separate evaluation script is required.

### Key Entities

- **CascadedPipeline**: Three-stage sequential model with separately saved VAD stage, child ID stage, and AV fusion stage; produces per-clip stage scores and a final probability.
- **TemporalSmoother**: Stateless post-processing layer that takes a sequence of raw probabilities (ordered by session and clip position) and returns a smoothed sequence; parameterized by method and window/bandwidth.
- **GPT4oFeatureRow**: Per-clip output from GPT-4o vision queries — structured JSON parsed into tabular form; includes child visibility flag, vocalizing flag, quality score, and free-text reasoning.
- **ASDFrontend**: Abstraction over TalkNet, LocoNet, and AS-Net; all frontends share the same input (audio path + face track cache) and output schema (per-clip ASD score CSV).
- **Ego4DAdaptedModel**: An ASD model fine-tuned or evaluated on Ego4D active speaker annotations before being applied to child home video; compared against the base TalkNet model.

---

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: The cascaded pipeline produces a `stage_breakdown.csv` showing, for each test clip, which stage made the final decision and the probability at that stage — enabling quantification of how often each stage is the deciding factor.
- **SC-002**: Temporal smoothing produces measurable reduction in prediction instability: the number of isolated single-clip sign changes (0→1→0 or 1→0→1) decreases by at least 20% on the val set compared to raw predictions.
- **SC-003**: GPT-4o features are extracted for at least 80% of SAILS BIDS clips without API errors; the remaining ≤20% are documented by failure reason (rate limit, missing frame, parsing error).
- **SC-004**: A fusion model trained with `child_visible_gpt4o` as an additional feature is evaluated on the test set and its AUROC is reported alongside the model trained without it, so the marginal value of GPT-4o features can be directly measured.
- **SC-005**: LocoNet and AS-Net ASD scores are extracted for all SAILS BIDS clips and a comparison table of test AUROC (ASD score vs. ground truth label, not fusion model AUROC) is produced for TalkNet vs. LocoNet vs. AS-Net.
- **SC-006**: Ego4D pretraining experiment (if run) produces a quantitative before/after comparison of ASD model performance on the child home video val set; if not run, a written rationale of ≥200 words is produced explaining the decision.
- **SC-007**: 1kd integration (if accessible) increases training set size by a documented amount and produces a comparison of model performance with vs. without 1kd data.
- **SC-008**: All new experiments are runnable from a single SLURM script or documented sequence of commands; no manual intermediate steps are required beyond dataset access.

---

## Assumptions

- GPT-4o API access is available and costs are manageable for ~1,500 clips × 2 frames at ~$0.01–$0.05 per clip; the default is to sample 2 frames per clip unless `--sample-rate` is overridden.
- LocoNet and AS-Net checkpoints are publicly available or obtainable; if not, those frontends are marked as unavailable and skipped with a clear error message rather than crashing.
- Ego4D requires an account and data use agreement; access is not assumed to be available; the story is implemented as a best-effort experiment.
- 1kd project refers to longitudinal naturalistic child home recording data; if the exact dataset is ambiguous or access-gated, the story resolves to documentation only.
- Temporal smoothing assumes clips from the same recording session appear in contiguous order in the predictions CSV, sorted by (child_id, session_id, clip_position); if this ordering is not present, the smoother logs a warning and applies clip-level smoothing only.
- The cascaded VAD stage reuses the existing audio diarization output (e.g., BabAR/VTC RTTM child segments) to determine whether any speech is present; a separate dedicated VAD model is not required unless the existing diarizers are unavailable.
- GPT-4o vision queries use the existing SAILS BIDS `.mp4` file path; for Providence/Playlogue (audio-only), GPT-4o features are NaN and the script skips those clips gracefully.
- All new experiments share the existing seen-child train/val/test split from `whisper-modeling/seen_child_splits/`; no new split generation is needed.
