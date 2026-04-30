# Implementation Plan: AV Self-Distillation and Visual-Eligibility Fusion

**Branch**: `015-av-self-distill-fusion` | **Date**: 2026-04-29 | **Spec**: `specs/015-av-self-distill-fusion/spec.md`

## Summary

Four AV experiments grounded in `audio_visual.txt` and tightly coupled to existing infrastructure. Shared design: frozen pretrained encoders + tiny fusion + visual-eligibility gating. US1 is CPU-only and almost free; US2-US4 require ~4-12h GPU jobs each. The pseudo-frame audio classifier (frozen WavLM + 199k-param head, AUROC=0.831) is the audio teacher / fusion partner for all of US2-US4.

| US | Priority | Compute | Wall-time | Hypothesis |
|---|---|---|---|---|
| US1 | P1 | CPU | ~2h | Computed visual features add small lift to spec-012 stacker, especially on `Child_of_interest=no` |
| US2 | P1 | GPU 4-8h | ~1d | Gated AV fusion ≥ audio-only on visually-eligible subset |
| US3 | P2 | GPU 4h | ~1d | ASD × ECAPA outperforms ASD-alone on multi-child clips |
| US4 | P2 | GPU 8h | ~2d | Video-only model trained on audio teacher reaches frame Pearson > 0.3 |

## Technical Context

**Language/Version**: Python 3.10 (`child-vocalizations` conda env at `/home/manaal/miniforge3/envs/child-vocalizations/`).

**Primary Dependencies**:
- Existing: torch 2.8+cu128, torchaudio, transformers 4.57+, numpy, pandas, scikit-learn, opencv-python (YuNet face detection), mediapipe (Pose).
- New: `fairseq` for AV-HuBERT (ships with the AV-HuBERT codebase) OR `torchaudio.models` if AV-HuBERT becomes part of upstream torchaudio. Fall back to a HuggingFace mirror (`Lin-Y123/AV-HuBERT` etc.) if direct fairseq install is brittle on the cluster.
- Mouth-ROI cropping: `dlib` (68-landmark) or `mediapipe.face_mesh` (468-landmark) — MediaPipe is faster and already available; use it.

**Storage**: Results under canonical folders. Caches:
- `pseudo_frame/visual_features/` (US1)
- `pseudo_frame/mouth_roi_cache/` (US2, US4) — per-clip .npy of (T, 96, 96) uint8
- `pseudo_frame/avhubert_emb_cache/` (US2, US4) — per-clip .npy of (T, 768) float32
- `pseudo_frame/results/{metadata_stack_av,avhubert_lipfusion,speaker_informed_asd,audio2video_distilled}/` — one folder per US

**Testing**: Each script must support `--limit N` or `--dry-run` for smoke testing 5-10 clips before SLURM submission.

**Target Platform**: SLURM cluster (NVIDIA A100s on `pi_satra,ou_bcs_normal`).

**Project Type**: ML experiment pipeline.

**Performance Goals**:
- US1: <2 hours wall (no GPU).
- US2 visual feature extraction: ≤4h GPU for 2183 clips.
- US3 inference: ≤4h GPU.
- US4 training: ≤8h GPU.

**Constraints**: seed=42; val-only threshold tuning; no test leakage; config.json committed with every result.

**Scale/Scope**: 2183 clips × ~25-40 fps × 10-30s = ~3-6 GB of cached visual features.

## Constitution Check

| Principle | Status | Notes |
|-----------|--------|-------|
| I. Reproducibility | ✅ PASS | seed=42 in all configs; SLURM scripts log job ID; config.json committed alongside results |
| II. Split discipline | ✅ PASS | All US use seen-child split; no cross-contamination |
| III. Baseline-first | ✅ PASS | All US compare against well-defined existing baselines (spec-012 stacker for US1; pseudo-frame classifier for US2/US4; TalkNet-ASD for US3) |
| IV. Metrics | ✅ PASS | F1, precision, recall, AUROC, AUPRC + per-timepoint + stratified by `Child_of_interest_clear` and `n_children` for each US |
| V. Ablations | ✅ PASS | US2 always-fuse vs gated; US3 speaker-conditioned vs not; US4 audio→video vs audio+video parallel |
| VI. Thesis sync | ✅ PASS | results_summary.md §11d, THESIS_MEGADOC.md §17d; per-US labnb entries |
| VII. Documentation | ✅ PASS | Each script docstring; CLAUDE.md updated after results |
| File deletion | ✅ PASS | No deletions; only new files |

## Project Structure

### Documentation (this feature)

```
specs/015-av-self-distill-fusion/
├── plan.md          # This file
├── spec.md          # User stories + requirements
├── tasks.md         # Phase 2 task breakdown
└── research.md      # AV-HuBERT install / mouth-ROI extraction notes (deferred)
```

### Source Code

```
pseudo_frame/
├── visual_eligibility.py            # US1: extract face stats from face_track_cache
├── extract_mouth_roi.py             # US2/US4: per-clip mouth crops from face tracks
├── extract_avhubert_features.py     # US2/US4: AV-HuBERT forward pass + cache
├── avhubert_late_fusion.py          # US2: train α + visual-eligibility threshold on val; eval test
├── speaker_informed_asd.py          # US3: per-track ASD × ECAPA-cos
├── audio2video_distill.py           # US4: train visual frame head with audio pseudo-labels
├── configs/
│   ├── avhubert_lipfusion.yaml      # US2
│   ├── speaker_informed_asd.yaml    # US3
│   └── audio2video_distill.yaml     # US4
├── slurm/
│   ├── extract_avhubert.sh          # US2/US4 setup
│   ├── train_lipfusion.sh           # US2
│   ├── speaker_informed.sh          # US3
│   └── train_audio2video.sh         # US4
└── results/
    ├── metadata_stack_av/           # US1 output
    ├── avhubert_lipfusion/          # US2: {audio_only, always_fuse, gated_av}/ subdirs
    ├── speaker_informed_asd/        # US3
    └── audio2video_distilled/       # US4

evaluation/
└── metadata_router.py               # extended: --visual-features <csv> flag (US1)
```

## Phase 0: Research

Defer detailed research notes to `research.md`. Key open questions:
- AV-HuBERT install path on this cluster (fairseq vs HuggingFace mirror). Fallback: extract via `torchaudio.pipelines` if available; otherwise pin a known-working fairseq commit.
- Mouth-ROI extraction: MediaPipe Face Mesh provides 468 landmarks; mouth ROI is the bounding box of landmarks 78-308 (or the standard AV-HuBERT 96×96 crop centered on the mouth midpoint).
- For US3, the per-frame ECAPA score requires re-segmenting audio aligned with face track timestamps. Reuse the per-segment cache at `whisper-modeling/usc_sail_segment_cache/` if its segment boundaries align; otherwise compute on-the-fly with the same ECAPA model.

## Phase 1: Data model

| Entity | Source | Output schema |
|---|---|---|
| Visual eligibility manifest | `av_fusion/face_track_cache/<md5>.json` | CSV: audio_path, face_count_max, face_count_mean, face_area_max, face_area_mean, face_confidence_mean, face_track_coverage_ratio, n_distinct_tracks |
| Mouth ROI cache | face tracks + raw video | `pseudo_frame/mouth_roi_cache/<md5>.npy` shape (T, 96, 96) uint8 |
| AV-HuBERT visual embedding | mouth ROI + AV-HuBERT model | `pseudo_frame/avhubert_emb_cache/<md5>.npy` shape (T, 768) float32 |
| Speaker-informed score | face tracks + audio segment + ECAPA prototype | per-track CSV: track_id, t_start, t_end, asd_score, ecapa_cos, joint_score |

## Phase 2: Tasks

See `tasks.md` for the granular breakdown.

## Risk register

1. **AV-HuBERT install brittleness on the cluster** — fairseq pinning may take a half day. Mitigation: try HuggingFace mirror first (`pip install transformers + checkpoint repo`), fall back to fairseq.
2. **MediaPipe mouth-ROI quality on toddler faces** — `audio_visual.txt` §108 explicitly warns that toddler mouth extraction is unreliable in naturalistic home video. Mitigation: report extraction success rate; gate fusion on extraction confidence.
3. **Face track cache may be incomplete** — verify coverage early in US1; document any gaps.
4. **Pseudo-frame teacher may transfer noise** to US4 — model could overfit to audio score artifacts. Mitigation: train only on visually-eligible clips with face track coverage > 50%.
5. **Negative results dominate** (TalkNet, LocoNet, fine-tuned TalkNet are all null) — design each US to be informative even if it null-results, by reporting subset-restricted metrics.

## Out of scope (deferred to a future feature)

- Cross-attention fusion of WavLM + AV-HuBERT (Tier 2 in the lit-review prioritization).
- ASDnB-style body-cue features (Tier 2). MediaPipe Pose extraction for body motion is half-implemented in US1's `body_visibility_proxy` but not used as a fusion signal.
- Self-supervised AV pretraining on Providence (Tier 3, infrastructure-heavy).
- Frame-level evaluation on Providence with human phonological labels (Tier 3, the thesis-credibility move).
