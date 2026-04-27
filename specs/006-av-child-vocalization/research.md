# Research: Audio-Visual Target-Child Vocalization Detection

**Feature**: 006-av-child-vocalization  
**Date**: 2026-04-24  
**Status**: Complete — all decisions resolved

---

## Decision 1: Visual Feature Source — Manual Annotations vs. Automatic Extraction

**Decision**: Use the manual BIDS annotations already present in `whisper-modeling/seen_child_splits/` as the primary visual feature source for MVP. Extend with automatic frame-level extraction as a separate, optional step.

**Rationale**: The existing split CSVs contain per-clip human-scored fields (`Video_Quality_Child_Face_Visibility`, `Video_Quality_Child_Body_Visibility`, `Video_Quality_Lighting`, `Video_Quality_Resolution`, `Video_Quality_Motion`, `Child_of_interest_clear`, `#_adults`, `#_children`, `Body_Parts_Visible`, `Angle_of_Body`). These are gold-standard annotations that directly answer whether the child is visible and whether video quality is usable. Using them as features gives a reliable upper-bound estimate of what automatic detection could achieve and enables fusion experiments to run immediately without GPU video processing.

**Alternatives considered**:
- Automatic-only: would require face detection on all clips before any experiment could run; riskier given child face detection reliability issues; deferred to optional stretch step.
- Manual + automatic combined: correct long-term strategy; the fast-track MVP uses manual annotations, and automatic features are added as a second pass if needed.

---

## Decision 2: Automatic Face Detection Model

**Decision**: YuNet (OpenCV `cv2.FaceDetectorYN`) as the default; MediaPipe Face Detection as fallback. Do NOT require InsightFace/ONNX separately for the MVP.

**Rationale**: YuNet is bundled with OpenCV 4.8+ (already present in the conda env), requires no extra installs, runs in real-time on CPU, and supports small faces reasonably well. MediaPipe (also CPU-capable) provides a fast fallback. RetinaFace/SCRFD offer better small-face recall but require InsightFace installation, which adds environment complexity. For a thesis system with existing manual annotations, YuNet is sufficient to compute automatic face track stats as supplementary features.

**Alternatives considered**:
- RetinaFace (InsightFace): better recall on small faces but requires `insightface` package, which conflicts with some conda environments.
- MediaPipe: good but deprecated face detection API in v0.10+; use as fallback only.
- MTCNN: slower than YuNet, no benefit for this use case.

---

## Decision 3: Face Tracking

**Decision**: Simple IoU-based centroid tracker implemented in ~100 lines (no extra package dependency). ByteTrack/BoT-SORT are overkill for clips ≤30 seconds with low face count.

**Rationale**: For short naturalistic home clips (typically 10–60s), the number of simultaneously visible faces is ≤3. A centroid tracker that assigns detections to the nearest existing track by IoU intersection handles this well without the engineering overhead of multi-object tracking libraries. The tracker only needs to produce: track IDs, track durations, per-track bounding box statistics.

**Alternatives considered**:
- ByteTrack: excellent for crowded scenes, but clips have at most 3–4 faces; overkill.
- BoT-SORT: even heavier, not worth the complexity.
- No tracking (per-frame stats only): sufficient for eligibility features but loses track-duration signal.

---

## Decision 4: Child Candidate Identification Heuristic

**Decision**: Identify the target-child candidate as the face track with the smallest median bounding-box area across the clip. Secondary signal: `Child_of_interest_clear == "yes"` from manual annotations confirms the heuristic is plausible. Use `#_children == 1` as a prerequisite for high-confidence child-face assignment.

**Rationale**: Toddlers and infants are physically smaller than adults; in a home recording their face bounding box is typically smaller than adult faces in the same frame. This heuristic is imperfect (fails when camera is close-up on child face, or when sibling is also present) but is the simplest viable approach for naturalistic data. Manual `Child_of_interest_clear` provides a validation signal.

**Alternatives considered**:
- Target-child face enrollment (ECAPA-based face recognition): would require face crops of the target child from the training split; much higher complexity and likely unreliable on infant faces.
- "Looking at camera" heuristic: face pose estimator for gaze direction; too unreliable for small/blurry child faces.
- Body-size estimation: could use person detection + bounding box height; added complexity, saved for stretch.

---

## Decision 5: Visual Eligibility Score Formula

**Decision**: Compute a composite `visual_eligibility_score` as a weighted average of normalized component scores:

```
eligibility = (
    0.40 * child_visible_score +       # is a child-sized face tracked?
    0.25 * track_fraction_score +      # face track covers ≥ X% of clip
    0.20 * quality_score +             # lighting × resolution, normalized
    0.15 * detection_confidence_score  # mean detector confidence
)
```

When using manual annotations as features, map as follows:
- `child_visible_score` = `Video_Quality_Child_Face_Visibility / 10` if `Child_of_interest_clear == "yes"`, else `Video_Quality_Child_Face_Visibility / 20`
- `quality_score` = `(Video_Quality_Lighting + Video_Quality_Resolution) / 20`
- `track_fraction_score` and `detection_confidence_score` filled by automatic extraction if run, else estimated from `Child_of_interest_clear`.

Binary `visual_eligible` threshold tuned on the validation split by maximizing the difference in label distribution between eligible and ineligible clips (not by optimizing F1 — we want a threshold that genuinely separates usable from unusable video, not one that is tuned toward the vocalization label).

**Alternatives considered**:
- Single-feature gate (face detected ≥ 50% of frames): too crude; doesn't account for quality or child identity.
- Logistic regression eligibility model trained on manual annotations: more principled but risks target leakage if the eligibility model learns the vocalization label implicitly.

---

## Decision 6: Fusion Model Class

**Decision**: XGBoost (`xgboost.XGBClassifier`) as the primary fusion model. Logistic regression as a secondary interpretable baseline. Shallow MLP as stretch.

**Rationale**: XGBoost handles NaN values natively (important for clips with missing video), is robust at ~1,000 training examples, provides feature importance for thesis interpretation, and avoids overfitting with small `max_depth=3` and `n_estimators=100`. Logistic regression provides an interpretable coefficient baseline.

**Training feature set** (per clip, ~25 features):
- Audio features: `existing_audio_score`, `existing_audio_proba` (from best audio baseline)
- Manual visual: `Video_Quality_Child_Face_Visibility`, `Video_Quality_Lighting`, `Video_Quality_Resolution`, `Video_Quality_Motion`, `Child_of_interest_clear_binary`, `n_adults`, `n_children`
- Automatic visual (if extracted): `n_face_tracks`, `max_face_track_fraction_clip`, `child_visible_score`, `off_camera_likely_score`, `visual_eligibility_score`
- Clip metadata: `age_band_binary` (0=14mo, 1=34mo)

**Alternatives considered**:
- Random forest: similar to XGBoost but slightly lower performance on tabular data; use as ablation.
- MLP (2-layer): promising but higher variance at 1500 examples without careful regularization.
- End-to-end AV model: out of scope per spec constraints.

---

## Decision 7: Audio Baseline Score Source

**Decision**: Use the BabAR combined-feature model's enrollment probability (`enroll_proba`) from `babar_ecapa_enrollment_runs/` as the primary audio score input to fusion, because it is the strongest single-model audio baseline (AUROC ~0.820 overall, ~0.892 at 14mo). Provide a `--audio-score-col` argument to allow switching to other baselines (e.g., WavLM).

**Alternatives considered**:
- WavLM direct classifier: competitive but BabAR consistently outperforms on this dataset.
- Multiple audio scores as features: valid but increases feature count; save for ablation.

---

## Decision 8: Output Directory Structure

**Decision**: New top-level module `av_fusion/` with the following layout:
```
av_fusion/
├── scripts/               # 6 pipeline scripts
├── configs/               # av_fusion.yaml sweep config
├── slurm/                 # SLURM submission script
├── av_results/            # canonical results output
│   └── {run_name}/        # per-run subdirectory
│       ├── config.json
│       ├── visual_features.csv
│       ├── av_master_features.csv
│       ├── models/
│       ├── metrics_overall.json
│       ├── metrics_by_age_band.csv
│       ├── metrics_by_visual_eligibility.csv
│       ├── metrics_by_failure_mode.csv
│       ├── predictions_test.csv
│       ├── error_analysis_examples.csv
│       └── figures/
└── face_track_cache/      # per-clip face detection cache (avoid re-detection)
```

**Rationale**: Mirrors the `mil/` module structure that already exists. Run-name subdirectories allow multiple experiments (e.g., manual-annotations-only vs. manual+automatic) to coexist. Face track cache avoids re-running expensive video processing.

---

## Decision 9: ASD Strategy

**Decision**: Treat ASD as optional stretch. If run, reuse existing `video/run_asd.py` TalkNet-ASD infrastructure from feature 004 (already integrated). The ASD script calls `video/run_asd.py --model talknet_asd` via subprocess (same pattern as `video_asd.py` in `pyannote/`).

**Alternatives considered**:
- Light-ASD: lighter model but would require a new integration not yet present in the repo.
- AV-HuBERT: requires face crop + audio alignment; high engineering cost; deferred.
- VideoMAE v2: generic video encoder; doesn't provide ASD scores directly.

---

## Decision 10: Video Path Source

**Decision**: Video paths (`BidsProcessed` column) are already present in `whisper-modeling/seen_child_splits/*.csv`. Use `BidsProcessed` as the primary video path; fall back to `BidsRaw` if the processed file is missing. Clips where both are absent have `video_path = None` and are treated as visually ineligible.

**Rationale**: The BIDS preprocessed videos are already used by the existing TalkNet/TS-TalkNet frontend in `pyannote/video_asd.py`. No new video path resolution logic is needed.

---

## Decision 11: Null Result Handling

**Decision**: The null result (AV does not improve overall performance) is treated as a first-class outcome. Stratified evaluation is designed to detect whether improvement exists within eligible subsets even when absent globally. The hypothesis is that improvement, if any, will appear in the `visual_eligible=True` subset and at the older age band (34–38 months).

**Framing**: A null global result paired with positive subset results is scientifically defensible and publishable as: "AV fusion helps conditionally but naturalistic home videos lack sufficient usable visual evidence for global improvement."
