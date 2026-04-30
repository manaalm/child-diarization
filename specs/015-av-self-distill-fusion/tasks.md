# Tasks: AV Self-Distillation and Visual-Eligibility Fusion

**Feature**: `015-av-self-distill-fusion` | **Spec**: `specs/015-av-self-distill-fusion/spec.md` | **Plan**: `specs/015-av-self-distill-fusion/plan.md`

Granular tasks. Each task is sized for ≤4h of focused work and has explicit success criteria. Tasks within a US can be done in any order unless an explicit dependency arrow is shown.

---

## US1 — Computed Visual-Eligibility Features in the Metadata Stacker (P1, CPU)

### T1.1 — `pseudo_frame/visual_eligibility.py`: extract face statistics from cache

**Input**: `av_fusion/face_track_cache/<md5(audio_path)>.json` — list of tracks with `frames: [{frame_idx, timestamp, bbox, score}]`.
**Output**: `pseudo_frame/visual_features/visual_eligibility.csv` with columns: `audio_path, face_count_max, face_count_mean, face_area_max_norm, face_area_mean_norm, face_confidence_mean, face_track_coverage_ratio, n_distinct_tracks`.
**Done when**: CSV has 2183 rows; smoke run on 10 clips matches manual cross-check.

### T1.2 — Extend `evaluation/metadata_router.py`: `--visual-features <csv>` flag

**Change**: load the CSV from T1.1 and merge by audio_path; add the columns to META_FEATS in `--mode stack`.
**Output**: when run with `--visual-features ...`, writes to `ensemble_runs/metadata_stack_av/`.
**Done when**: `python evaluation/metadata_router.py --mode stack --visual-features pseudo_frame/visual_features/visual_eligibility.csv` produces a `test_metrics_tuned.json` and `feature_importances.json`.

### T1.3 — Stratified eval (mirror spec-012 ablation)

**Reuse**: `evaluation/metadata_stack_ablation.py` pattern. Same strata: `Child_of_interest_clear`, `n_children`, `timepoint_norm`, `n_adults`.
**Output**: `ensemble_runs/metadata_stack_av/ablation/{lr_coefficients.csv, stratified_metrics.csv, score_correlation.json}`.
**Done when**: stratified table is in the same shape as `ensemble_runs/metadata_stack/ablation/stratified_metrics.csv`, with three rows per stratum (delta vs. metadata-only stacker).

### T1.4 — Document

**Update**: `results_summary.md` §11d (new), `THESIS_MEGADOC.md` §17d.1, labnb.

---

## US2 — AV-HuBERT Frozen Lip-ROI + Late Fusion + Visual-Eligibility Gating (P1, GPU)

### T2.1 — Install AV-HuBERT

**Try**: HuggingFace `Lin-Y123/AV-HuBERT` or `nguyenvulebinh/AV-HuBERT` if available.
**Fallback**: pin fairseq commit, install AV-HuBERT codebase from `facebookresearch/av_hubert`.
**Done when**: a single-clip forward pass produces (T, 768) features with no env errors. Smoke verified on a 10s clip.

### T2.2 — `pseudo_frame/extract_mouth_roi.py`: mouth crops from face tracks

**Input**: face tracks (T1.1 cache) + raw video `.mp4` from BIDS path.
**Output**: `pseudo_frame/mouth_roi_cache/<md5>.npy` shape (T, 96, 96) uint8.
**Method**: For each frame in each track, run MediaPipe Face Mesh on the bbox crop, extract mouth region (landmarks 78-308 bounding box), resize to 96×96 grayscale. Per-clip select the longest face track as the "child candidate" track (audio_visual.txt §40 warns against assuming smallest=child; revisit using ECAPA agreement in US3).
**Done when**: 2183 .npy files produced; smoke check on 10 clips visualizes mouth crops at 5 random frames.

### T2.3 — `pseudo_frame/extract_avhubert_features.py`: visual embedding cache

**Input**: T2.2 mouth ROI cache.
**Output**: `pseudo_frame/avhubert_emb_cache/<md5>.npy` shape (T, 768) float32 at 25 fps.
**Method**: forward AV-HuBERT visual stream only on the mouth-ROI sequence; extract last hidden state.
**Submission**: `sbatch pseudo_frame/slurm/extract_avhubert.sh` (4-8h GPU).
**Done when**: 2183 .npy files produced; mean coverage > 80% (some clips will fail extraction due to absent video or empty face tracks).

### T2.4 — `pseudo_frame/avhubert_late_fusion.py`: train α + visual-eligibility threshold

**Input**: AV-HuBERT visual embeddings (T2.3) + audio scores from `pseudo_frame/results/wavlm_pseudo_frame/{val,test}_predictions.csv` + visual eligibility manifest (T1.1).
**Method**:
- Train a tiny linear head (768 → 1) on val to produce a clip-level visual score from mean-pooled AV-HuBERT features.
- Tune α ∈ [0,1] on val for `score_av = α · audio + (1-α) · visual` (always_fuse).
- Tune visual_eligibility_threshold on val using balanced accuracy of `eligibility_score >= τ` against the visually-eligible label (face_count_max ≥ 1 AND face_confidence_mean ≥ 0.6 AND face_track_coverage_ratio ≥ 0.3).
- gated_av: `score_av if eligibility >= τ else audio`.
**Output**: three result subdirs in `pseudo_frame/results/avhubert_lipfusion/{audio_only,always_fuse,gated_av}/` each with `val_metrics_tuned.json`, `test_metrics_tuned.json`, `test_predictions.csv`, `config.json`.
**Done when**: all three runs produce metrics; subset-on-eligible metrics also written.

### T2.5 — Document

**Update**: `results_summary.md` §11d.2, `THESIS_MEGADOC.md` §17d.2, labnb.

---

## US3 — Speaker-Embedding-Informed AV (P2, GPU)

### T3.1 — Per-track audio segment alignment

**Input**: face tracks + clip audio.
**Output**: per-clip CSV mapping (track_id, t_start, t_end) → audio segment hash (matching the cache at `whisper-modeling/usc_sail_segment_cache/` if compatible, else compute on-the-fly).
**Done when**: 2183 CSVs produced; smoke check on 10 clips shows reasonable track-to-audio alignment.

### T3.2 — `pseudo_frame/speaker_informed_asd.py`: ASD × ECAPA scoring

**Input**: T3.1 alignment + AV-HuBERT visual embeddings (T2.3) + per-child ECAPA prototype from `pyannote/usc_sail_segment_cache/proto.json` or equivalent.
**Method**:
- For each face track, compute mean ASD score (use AV-HuBERT-based "is speaking" head from US2 OR a separate Light-ASD model — choose simpler).
- For each face track's matching audio segment, compute ECAPA cosine similarity to the target-child prototype.
- Joint score = ASD × cos.
- Aggregate: max-pool over face tracks → clip score.
**Output**: `pseudo_frame/results/speaker_informed_asd/test_predictions.csv` with columns: audio_path, label, asd_score_max, ecapa_cos_max, joint_score_max, n_tracks, selected_track_id.

### T3.3 — Late fusion + evaluation

**Method**: fuse joint_score with audio pseudo-frame score via val-tuned α; same eligibility gating as US2.
**Output**: `pseudo_frame/results/speaker_informed_asd/test_metrics_tuned.json` + stratified metrics, especially on `n_children≥2` (the failure stratum where this is hypothesized to help most).

### T3.4 — Document

**Update**: `results_summary.md` §11d.3, `THESIS_MEGADOC.md` §17d.3, labnb.

---

## US4 — Audio → Video Pseudo-Label Distillation (P2, GPU)

### T4.1 — `pseudo_frame/audio2video_distill.py` dataset

**Input**: AV-HuBERT visual embeddings at 25 fps (T2.3) + pseudo-frame audio scores at 50 Hz (already cached from `pseudo_frame/results/wavlm_pseudo_frame/`, derived from forward pass) + visual eligibility flag.
**Method**: Downsample audio scores to 25 Hz (mean of pairs of adjacent frames). Use as soft per-frame target. Train only on visually-eligible clips (eligibility_score ≥ τ from T2.4) to avoid teaching the visual head from non-visual evidence.
**Done when**: dataset returns (B, T, 768), (B, T) pairs at 25 Hz with masking.

### T4.2 — Frame head + training loop

**Architecture**: AV-HuBERT (frozen) → LayerNorm → Linear(768→256) → GELU+Dropout → Linear(256→1) → frame logits at 25 Hz.
**Loss**: BCEWithLogitsLoss against soft target ∈ [0, 1]; pos_weight=3.0 (mirroring pseudo-frame).
**Confidence weighting**: weight by audio teacher's confidence: `2 · |target − 0.5|` so the visual head learns from frames where the audio model itself is confident.
**Train**: AdamW lr=1e-3, 25 epochs, patience=5, val tunes clip threshold (max-pool).
**Submission**: `sbatch pseudo_frame/slurm/train_audio2video.sh` (8h GPU).

### T4.3 — Evaluation

**Clip-level**: max-pool frame probs → val-tuned threshold → F1/AUROC/AUPRC on test (eligible subset only). Comparison row vs. US2 mean-pool baseline.
**Frame-level localization**: mean per-clip Pearson/Spearman/AUROC vs. test pseudo-frame audio scores (downsampled to 25 Hz).
**Output**: `pseudo_frame/results/audio2video_distilled/{best_checkpoint.pt, val_metrics_tuned.json, test_metrics_tuned.json, frame_localization.json, test_predictions.csv, config.json}`.

### T4.4 — Optional: add as 12th system in spec-012 stacker

**Defer**: only if SC-004 is met (frame Pearson > 0.3). Add `audio2video_distilled_prob` to `_SYSTEM_PATHS` in `evaluation/metadata_router.py` and re-run `--mode stack`.

### T4.5 — Document

**Update**: `results_summary.md` §11d.4, `THESIS_MEGADOC.md` §17d.4, labnb.

---

## Cross-cutting tasks

### TX.1 — labnb experiment registration

Register one experiment per US at submission time:
```
python /home/manaal/.claude/skills/labnb/scripts/register_experiment.py \
  --lab-root /home/manaal/.local/state/lab-notebook \
  --project-root /orcd/scratch/orcd/008/manaal/child-adult-diarization \
  --project-slug child-adult-diarization \
  --experiment-slug spec-015-us<N>-<short-name> \
  --objective "<US objective>" \
  --entry-kind experiment --metric-name auroc --direction higher \
  --overall-budget <e.g. 4h> --loop-budget <e.g. 4h>
```

### TX.2 — Final summary table

Produce a single comparison table across all four USs in `results_summary.md` §11d.5 with rows: pseudo-frame baseline; metadata stacker (spec-012); US1; US2 audio_only / always_fuse / gated_av; US3 (overall + multi-child); US4. Columns: F1, Precision, Recall, AUROC, AUPRC, n.

### TX.3 — Update CLAUDE.md "Recent Changes"

Append entry summarizing spec-015 outcomes (positive, neutral, or negative — no a priori framing).

---

## Dependency graph

```
T1.1 → T1.2 → T1.3 → T1.4
T2.1 → T2.2 → T2.3 → T2.4 → T2.5
                  └→ T3.1 → T3.2 → T3.3 → T3.4
                  └→ T4.1 → T4.2 → T4.3 → T4.4 → T4.5
```

T2.3 (AV-HuBERT cache) is the longest pole and the dependency for US3 + US4. Submit T2.3 SLURM as early as possible.

## Estimated wall-time (sequential)

| Phase | Tasks | Wall-time |
|---|---|---|
| US1 | T1.1-T1.4 | ~2-3h (CPU) |
| US2 setup | T2.1-T2.3 (mouth ROI + AV-HuBERT cache) | ~1d (GPU 4-8h) |
| US2 train+eval | T2.4-T2.5 | ~3h |
| US3 | T3.1-T3.4 | ~6-8h (after T2.3) |
| US4 | T4.1-T4.5 | ~12h (after T2.3) |

Parallelism: US3 and US4 can run in parallel after T2.3 completes.
