# Feature Specification: Audio-Visual Self-Distillation and Visual-Eligibility-Aware Fusion

**Feature Branch**: `015-av-self-distill-fusion`
**Created**: 2026-04-29
**Status**: Draft
**Lit-review source**: `audio_visual.txt` (audio-visual literature review for child vocalization detection)

## Overview

Four AV experiments motivated by `audio_visual.txt` and tightly coupled to existing infrastructure. The shared design principle is the literature's clearest recommendation for small-data child home video: **frozen pretrained encoders + tiny fusion + visual-eligibility gating, not end-to-end AV training** (audio_visual.txt §151, §157). All four user stories deliberately avoid the failure modes of the existing TalkNet-ASD / LocoNet / fine-tuned TalkNet null results.

The feature also leverages two assets the lit review highlights as comparative advantages of this project:
1. The just-completed **pseudo-frame audio classifier** (`pseudo_frame/`, AUROC=0.831, frame Pearson=0.566) as a strong audio teacher.
2. Per-child **ECAPA prototypes** built during seen-child enrollment, which let us replicate Clarke et al. (2025) "speaker-embedding-informed AV" without re-engineering the speaker side.

User stories are listed in priority order (lowest cost / highest expected payoff first).

---

## User Scenarios & Testing *(mandatory)*

### User Story 1 — Computed Visual-Eligibility Features in the Metadata Stacker (Priority: P1)

A researcher wants to test whether *automatic* visual-quality features (face count, face area, face confidence, body keypoint visibility) add signal to the spec-012 metadata stacker. Currently the stacker uses only manual BIDS metadata (`#_adults`, `#_children`, `Context`, `Interaction_with_child`, `timepoint`); the spec-012 ablation showed metadata contributes only ~9% of LR coefficient mass and the gain is concentrated on easy strata (36-month, single-child). Visual-availability features may add signal on the harder strata where the stacker currently regresses (`Child_of_interest=no`, n=39 clips with F1 −0.043).

**Why this priority**: Lowest implementation cost (CPU-only, ~1 day), reuses cached face tracks at `av_fusion/face_track_cache/`, no GPU. If even computed visual-availability features fail to improve the stacker, that is a clean negative result confirming the SAILS BIDS visual stream is fundamentally weak — strengthening the framing for US2-US4. The lit review (§154) explicitly recommends "visual gating itself as a first-class experiment".

**Independent Test**: Re-run `evaluation/metadata_router.py --mode stack` with the augmented feature set. Pass when test AUROC and stratified `Child_of_interest=no` AUROC are reported as deltas vs. the metadata-only baseline (F1=0.901, AUROC=0.900).

**Acceptance Scenarios**:

1. **Given** cached face tracks for all 2183 seen-child clips, **When** `pseudo_frame/visual_eligibility.py` extracts per-clip face statistics (face_count_max, face_count_mean, face_area_max, face_area_mean, face_confidence_mean, face_track_coverage_ratio, body_visibility_proxy), **Then** a CSV is written with one row per clip and the same audio_path key as the metadata stacker.
2. **Given** the augmented feature CSV, **When** `evaluation/metadata_router.py --mode stack --visual-features <csv>` is run, **Then** `ensemble_runs/metadata_stack_av/test_metrics_tuned.json` is written with F1/AUROC/AUPRC and per-feature LR coefficients reported.
3. **Given** a completed run, **When** stratified metrics are computed by `Child_of_interest_clear` (yes/no/unclear), `n_children` (1 vs ≥2), and `timepoint_norm`, **Then** a CSV reports stack vs base deltas — independent of overall direction. The result is documented as positive, neutral, or negative.

---

### User Story 2 — AV-HuBERT Frozen Lip-ROI + Late Fusion + Visual-Eligibility Gating (Priority: P1)

A researcher wants to test whether mouth motion adds anything beyond the audio-only pseudo-frame classifier when restricted to the visually eligible subset. The lit review (§67) explicitly identifies AV-HuBERT as "the best candidate if you want one frozen visual-speech encoder to test whether mouth motion adds anything beyond audio." The thesis framing recommended at audio_visual.txt §157 is exactly the comparison this US implements: audio-only baseline vs. always-fuse AV vs. gated AV.

**Why this priority**: Highest-leverage AV experiment in the doc for this exact problem regime. Released checkpoints exist; slots into existing `av_fusion/` directory; uses the new pseudo-frame audio score as the fusion partner.

**Independent Test**: Three configurations evaluated on the same test clips:
- `audio_only`: pseudo-frame classifier (already built).
- `always_fuse`: late fusion of audio + AV-HuBERT visual score for all clips (visual_score=0.5 fallback when no track).
- `gated_av`: same fusion, but only for clips above a val-tuned visual-eligibility threshold; audio-only fallback otherwise.

Plus stratified evaluation on the visually-eligible subset only. Pass when results are reported in all four cells (3 configs × 2 sets = 6 metric blocks).

**Acceptance Scenarios**:

1. **Given** AV-HuBERT pretrained checkpoint and cached face tracks, **When** `pseudo_frame/extract_avhubert_features.py` runs over the 2183 clips, **Then** per-clip mouth-ROI sequences are saved + a clip-level visual score is produced via mean-pool of AV-HuBERT frame embeddings → linear head trained on val.
2. **Given** audio scores from `pseudo_frame/results/wavlm_pseudo_frame/test_predictions.csv` and visual scores from US2 step 1, **When** late fusion is applied with val-tuned α, **Then** `pseudo_frame/results/avhubert_lipfusion/{always_fuse,gated_av}/test_metrics_tuned.json` are written.
3. **Given** the gated_av run, **When** subset metrics are computed on visually-eligible clips only, **Then** a delta-AUROC vs. audio-only is reported with the n of the eligible subset.

---

### User Story 3 — Speaker-Embedding-Informed AV (Clarke 2025) (Priority: P2)

A researcher wants to replicate Clarke et al. (2025), which adds speaker-comparison information to TalkNet/Light-ASD on Ego4D and reports +14.5% / +10.3% relative mAP. The bridge in this project is direct: the audio side already has per-child ECAPA prototypes from enrollment. Compute, per face track in each clip, an ASD score (probability the on-screen face is speaking) **multiplied by** the cosine similarity of the matching audio segment to the target-child ECAPA prototype. This produces a "is this on-screen person the **target child** speaking" score rather than a generic "is any visible person speaking" score. Aggregate across face tracks → clip score → late-fuse with audio.

**Why this priority**: The lit review (§28) calls this paper "arguably the most promising conceptual bridge between your audio-side experience with child-robust speaker cues and the video literature." Most labs would have to build the speaker side; here it's free.

**Independent Test**: Run on test clips with at least one face track AND a valid ECAPA prototype for the target child. Compare against TalkNet-ASD baseline (F1=0.336) and audio-only pseudo-frame classifier (AUROC=0.831).

**Acceptance Scenarios**:

1. **Given** cached face tracks + per-track temporal alignment to audio, **When** `pseudo_frame/speaker_informed_asd.py` computes (ASD score × ECAPA cosine) per face track per frame, **Then** a per-clip max- and mean-pool score is produced with metadata for which face track was selected.
2. **Given** per-clip scores, **When** late-fused with the audio pseudo-frame score, **Then** test metrics are written to `pseudo_frame/results/speaker_informed_asd/test_metrics_tuned.json`.
3. **Given** the run, **When** evaluated on the multi-child stratum (n_children≥2, n=94, where the spec-012 multi-child suppressor was a null result), **Then** a delta vs. audio-only is reported. Hypothesis: this is the stratum where speaker-conditioning helps most.

---

### User Story 4 — Audio → Video Pseudo-Label Distillation (Priority: P2)

A researcher wants to train an AV-HuBERT visual frame head where the supervision is the pseudo-frame audio classifier's per-frame score. The audio teacher transfers its frame-level localization knowledge to a video-only model. At inference, fuse the audio-distilled visual frame score with the original audio score. This is a novel direction for this project (most AV work is video → audio); it combines US2's frozen AV-HuBERT with the just-finished pseudo-frame audio classifier.

**Why this priority**: Original contribution that connects two newly-built systems. Not in the lit review explicitly, but follows directly from §149-§151 (small-data AV needs strong supervision signal) and from the project's now-built audio teacher.

**Independent Test**: Train on visually-eligible clips only (face track exists + face_confidence > τ); evaluate clip-level + frame-level localization on the eligible subset. Compare against US2 (which uses linear-on-mean-pool, not frame-level training).

**Acceptance Scenarios**:

1. **Given** AV-HuBERT visual embeddings + pseudo-frame audio scores aligned at 25 fps (visual) ⇄ 50 Hz (audio) → 25 Hz common rate, **When** a small frame head (LayerNorm → Linear → GELU → Linear) is trained with frame-BCE against audio-distilled targets, **Then** `pseudo_frame/results/audio2video_distilled/best_checkpoint.pt` is written with per-epoch val metrics.
2. **Given** the trained model, **When** evaluated on the eligible-subset test split, **Then** clip-level F1/AUROC/AUPRC AND mean per-clip frame-level Pearson/Spearman/AUROC vs. test pseudo-labels are written.
3. **Given** late fusion with audio at the frame level, **When** clip-level metrics are recomputed via max-pool, **Then** the fused-AV-frame model is reported as a 12th system that can be optionally added to the spec-012 metadata stacker (US1's framework).

---

### Edge Cases

- Clips with **zero face tracks**: all visual features = 0; eligibility flag = 0; gated_av falls back to audio-only.
- Clips where **face track exists but ECAPA prototype is missing for the child**: US3 falls back to ASD-only score (no speaker conditioning); document fallback rate.
- AV-HuBERT mouth-ROI extraction failures: documented per-clip in a manifest; clip is treated as ineligible.
- `n_adults≥2` clips (n=15 in test): too few for stratified analysis at this granularity; report jointly with `n_children≥2` instead.
- Negative clips with face tracks: pseudo-label is forced to zero (clip-level supervision is exact); gating still applies.

---

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: All four US implementations MUST share the same seen-child split (`whisper-modeling/seen_child_splits/`) used by spec-012 and the pseudo-frame classifier. Test set n=441.
- **FR-002**: Visual-eligibility features (US1, US2, US3, US4) MUST be derived from the existing cached face tracks at `av_fusion/face_track_cache/` (keyed by md5(audio_path)) without re-running detection.
- **FR-003**: AV-HuBERT (US2, US4) MUST use the public Meta checkpoint (`facebook/av_hubert_base_lrs3_iter5` or equivalent) and remain **frozen** during training. Only a small head (≤500k params) may be trained.
- **FR-004**: Speaker-informed AV (US3) MUST reuse the per-child ECAPA prototypes already built by `pyannote/unified.py` for seen-child enrollment.
- **FR-005**: Each US MUST report (a) overall test metrics, (b) stratified metrics by `Child_of_interest_clear`, `n_children`, and `timepoint_norm`, (c) a comparison row vs. the relevant baseline (metadata stacker for US1; pseudo-frame classifier for US2/US4; TalkNet-ASD + pseudo-frame for US3).
- **FR-006**: Each US MUST write `config.json`, `test_metrics_tuned.json`, `test_predictions.csv`, `val_metrics_tuned.json` to its designated output folder.
- **FR-007**: Threshold tuning MUST be done on val only, never test.
- **FR-008**: All randomness MUST use seed=42.

### Key Entities

- **Visual eligibility manifest** (US1, FR-002): per-clip JSON/CSV with face_count_max/mean, face_area_max/mean, face_confidence_mean, face_track_coverage_ratio, body_visibility_proxy.
- **Mouth-ROI cache** (US2, US4): per-clip per-frame 96×96 grayscale crops aligned to face tracks (cf. AV-HuBERT default preprocessing).
- **AV-HuBERT visual embedding cache** (US2, US4): per-clip (T, 768) tensor at 25 fps; cached on disk to amortize the GPU-heavy forward pass.
- **Speaker-informed score table** (US3): per-face-track score = ASD × ECAPA-prototype-cosine, per frame, with track-level pool to clip score.

---

## Success Criteria *(mandatory)*

**SC-001 (US1)**: A delta-AUROC and delta-F1 vs. the spec-012 metadata-only stacker is reported, with stratified breakdowns. Sign of delta is unconstrained (positive or negative) — the *experiment* is the success criterion.

**SC-002 (US2)**: Three configurations (audio-only, always-fuse, gated-AV) are evaluated. Hypothesis is met if `gated_av` AUROC > `audio_only` AUROC on the visually-eligible subset (n typically ~300-350).

**SC-003 (US3)**: Speaker-informed AV outperforms naive TalkNet-ASD (F1=0.336, AUROC=0.569) on the multi-child stratum. Optionally: speaker-informed AV provides any non-zero lift over audio-only when fused.

**SC-004 (US4)**: Audio→video distillation produces a video-only frame model with mean per-clip frame Pearson > 0 (positive correlation with the audio teacher on held-out clips). True success: per-clip frame Pearson > 0.3 (half of the audio model's 0.566).

**SC-005 (cross-cutting)**: All four USs are documented in `results_summary.md` (new sections), `THESIS_MEGADOC.md` §17d (new), and the labnb notebook with full results, caveats, and links to artifacts.
