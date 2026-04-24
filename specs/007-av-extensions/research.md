# Research: AV Extended Experiments — 007-av-extensions

**Date**: 2026-04-24  
**Feature**: `specs/007-av-extensions/spec.md`

---

## Decision 1: LocoNet ASD Model

**Decision**: Use `SJTUwxz/LoCoNet_ASD` (CVPR 2023) as the primary new ASD frontend.

**Rationale**: Publicly available code and pretrained checkpoint (HuggingFace: `Superxixixi/LoCoNet_ASD`). Outperforms TalkNet on AVA-ActiveSpeaker benchmark. Uses long-short temporal context via a two-stream architecture; more robust to temporal motion than TalkNet. Compatible with Python 3.11.

**Integration notes**:
- Requires pre-extracted face tracks (same as TalkNet — reuses existing `video/face_track_cache/`)
- Input: stacked face crops (grayscale, 112×112) + matching audio; AVA-style feature format
- Output: per-clip ASD scores per face track (convert to `max_asd_score_target_candidate`)
- `conda activate child-vocalizations && pip install -r requirements.txt` from the LocoNet repo
- Checkpoint: download from `Superxixixi/LoCoNet_ASD` on HuggingFace (≈200MB)
- Must add LocoNet repo to `video/` directory (gitignore'd like TalkNet-ASD)

**Alternatives considered**:
- TalkNet-ASD: already implemented, serves as baseline; mAP ≈90.8 on AVA (adult broadcast)
- Light-ASD: more lightweight but lower accuracy ceiling
- LocoNet chosen over TalkNet for thesis comparison value (newer, stronger baseline)

---

## Decision 2: AS-Net

**Decision**: AS-Net (Radman & Laaksonen 2024, Aalto University) has **no public code or checkpoint**. Use **Light-ASD** (github.com/Junhua-Liao/Light-ASD) as the second ASD comparison system instead.

**Rationale**: AS-Net paper is academic-only; contacting Aalto authors would introduce timeline uncertainty. Light-ASD is a recent (2023), publicly available, lightweight ASD model that outperforms TalkNet while being faster — makes it a practical second comparison point.

**Light-ASD notes**:
- GitHub: `Junhua-Liao/Light-ASD`
- Pretrained checkpoint available in the repo (≈10MB)
- Same input/output format as TalkNet (face crops + audio → per-frame scores)
- Can be called from the same `extract_asd_features.py --model light_asd` interface

**Alternatives considered**:
- AS-Net (Radman & Laaksonen): no public code — excluded
- MAAS (Multi-granularity Active Speaker): complex setup, less established
- Light-ASD chosen for simplicity, public availability, and compatibility with existing face-track pipeline

---

## Decision 3: GPT-4o Frame Analysis

**Decision**: Use `gpt-4o-mini` as the default model for child detection in frames; `gpt-4o` available as override via `--model` flag.

**Rationale**: GPT-4o-mini is 10× cheaper ($0.15/1M input tokens vs $2.50 for gpt-4o) and sufficient for binary/structured classification tasks. At ~1000 tokens per JPEG frame and 3000 frames total, total cost ≈ $0.45–$1.50 with gpt-4o-mini (vs ~$7.50 with gpt-4o). Rate limits are manageable with async batching and exponential backoff.

**API design**:
- Input: base64-encoded JPEG frame extracted via OpenCV at 2 frames/clip (configurable)
- System prompt: structured JSON schema enforced via `response_format: {"type": "json_object"}`
- Output schema per frame:
  ```json
  {
    "child_visible": "yes" | "no" | "uncertain",
    "child_vocalizing": "yes" | "no" | "uncertain",
    "n_children_visible": 0-3,
    "visual_quality": "good" | "medium" | "poor",
    "notes": "optional free text"
  }
  ```
- Per-clip aggregation: majority vote across sampled frames for `child_visible_gpt4o`; max score for `child_vocalizing_gpt4o`
- Caching: save raw API response per frame to `av_fusion/gpt4o_cache/{clip_id}_{frame_idx}.json`; skip on re-run

**Alternatives considered**:
- Full gpt-4o: same capability for this task, 10× more expensive — excluded as default
- Claude Vision / Gemini Vision: possible but OpenAI is the most stable API for structured JSON output and the most widely documented

---

## Decision 4: Cascaded Pipeline Architecture

**Decision**: Three-stage cascade using existing components as stages, no new model training required for the VAD stage.

**Stage design**:
1. **VAD stage**: Reuse BabAR/VTC RTTM outputs — if `child_total_dur > 0` in the RTTM, speech is detected. Configurable with a minimum duration threshold tuned on val set. This avoids training a separate VAD model from scratch.
2. **Child ID stage**: ECAPA cosine similarity score from the existing enrollment pipeline (BabAR or VTC). This is already computed; the cascade gate is a threshold on the enrollment score.
3. **AV fusion stage**: The existing `GatedAVModel` from 006 — uses visual eligibility gating + late fusion.

**Per-clip stage selection**: Each clip gets a `cascade_stage` column (1, 2, or 3) indicating which stage produced the final prediction. This enables the stage breakdown table required by SC-001.

**Training**: Only the thresholds between stages need tuning (2 thresholds on val set: `vad_threshold`, `child_id_threshold`). No new model weights required for a first version.

**Alternatives considered**:
- Training a dedicated deep VAD model: unnecessary overhead given existing RTTM outputs
- Single-stage end-to-end: already implemented in 006; cascade is a new decomposition
- Two-stage only (skip AV): falls back to this gracefully when visual_eligible=False

---

## Decision 5: Temporal Smoothing

**Decision**: Implement three smoothing methods selectable via `--method`: `gaussian` (default), `majority_vote`, and `moving_average`. Tune window/bandwidth on val set only.

**Rationale**: Gaussian smoothing is parameter-efficient (one bandwidth parameter), has a principled probabilistic interpretation, and handles irregular session lengths gracefully. Majority vote is easily interpretable and good for thesis comparison. CRF was considered but adds training complexity that is not justified by the small dataset size.

**Implementation**: Smooth within (child_id, recording_date) groups. The predictions CSV must have a `clip_position` or `clip_order` column; if absent, infer from row order within each (child_id, timepoint) group. Apply smoothing to raw probabilities, not binary labels, to preserve threshold flexibility.

**Alternatives considered**:
- CRF (Conditional Random Field): strongest theoretically, but requires learning transition parameters; not justified for a ~1500 clip dataset
- HMM post-processing: similar concern as CRF
- Gaussian chosen as default for simplicity and no additional training

---

## Decision 6: Ego4D Integration

**Decision**: Treat Ego4D as an optional pretraining dataset for ASD models; implement as a documented experiment, not a pipeline dependency.

**Access**: Freely available with academic license registration at ego4d-data.org (48h approval). Python CLI: `pip install ego4d`. ASD/AVD benchmark: ~50h annotated (572 clips).

**Recommended ASD models trained/evaluated on Ego4D**:
- `Ego4d_TalkNet_ASD` (zcxu-eric): TalkNet adapted for egocentric video; public checkpoint
- PAIR-Net: mAP 76.6% on Ego4D; more recent; check for public code
- GateFusion: mAP 77.8% on Ego4D; more recent

**For the thesis experiment**: Zero-shot evaluate TalkNet and LocoNet on Ego4D AVD val set to measure egocentric domain gap; optionally fine-tune LocoNet on Ego4D before applying to child home video.

**Alternatives considered**:
- Full Ego4D fine-tuning: likely impractical within SLURM job budget (3000h+ of Ego4D)
- Using Ego4D data for training from scratch: out of scope

---

## Decision 7: 1kd Project

**Decision**: Treat as a dataset integration requiring manual investigation of access pathway; implement as conditional on data availability.

**Finding**: "1kd" likely refers to one of several child development longitudinal projects:
- **1000 Days project** (Brown University / NICHD): naturalistic home recordings; access requires institutional agreement
- **1000 Days from Home** (UK Biobank linked): age 0–3 longitudinal; restricted
- Could refer to a local MIT lab dataset (the user should clarify the exact project name)

**Approach**: Create `scripts/1kd_integration.py` that accepts a data directory and checks for schema compatibility; output a JSON compatibility report. The script is a stub that documents the integration pathway without assuming access. The spec's fallback scenario (documentation only) applies.

**Action needed**: User should confirm which specific "1kd" dataset they mean; the code handles the integration pathway generically.

---

## Summary of Model/Tool Choices

| Component | Chosen | Alternative Rejected |
|---|---|---|
| New ASD model 1 | LocoNet (SJTUwxz/LoCoNet_ASD) | AS-Net (no public code) |
| New ASD model 2 | Light-ASD (Junhua-Liao/Light-ASD) | AS-Net (same reason) |
| GPT-4o vision | gpt-4o-mini default | gpt-4o (10× cost, same quality) |
| VAD stage | Existing BabAR/VTC RTTM | Separate VAD model (overkill) |
| Child ID stage | ECAPA enrollment score | Re-train speaker ID model |
| Temporal smoothing | Gaussian kernel | CRF (training overhead) |
| Ego4D | Optional / pretraining reference | Required dependency |
| 1kd | Conditional / stub | Required dependency |
