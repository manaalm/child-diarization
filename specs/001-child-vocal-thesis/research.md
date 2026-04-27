# Research: Child Vocalization Extraction & Synthesis Thesis

**Phase 0 Output** | **Date**: 2026-04-17 | **Feature**: 001-child-vocal-thesis

---

## 1. Synthesis Framework Selection

**Decision**: Two-track synthesis pipeline based on developmental stage:
- **34-38 months (toddler speech)**: Fine-tune a VITS-based neural TTS model via
  Coqui TTS on labeled child speech segments extracted from Providence/Playlogue/Seedlings.
- **12-16 months (infant vocalizations)**: VAE or VQVAE on mel spectrograms (not
  text-conditioned), since 12-16 month vocalizations are pre-linguistic (babbling,
  vowel chains, proto-words) and cannot be meaningfully modeled with standard TTS.

**Rationale**:
- VITS (Casanova et al., 2021) is the leading end-to-end neural TTS model that supports
  multi-speaker fine-tuning with relatively small corpora (hours, not thousands of hours).
- Coqui TTS provides accessible fine-tuning scripts and VITS checkpoints; separation
  from the whisper-modeling environment (per constitution) is natural.
- For 12-16 months, text-conditioned TTS is conceptually incorrect (infants produce
  vocalizations, not words). A VAE on mel spectrograms learns the distribution of
  infant vocalization acoustics directly and can sample novel vocalizations.
  This is a novel contribution in itself — few papers address non-linguistic infant
  speech synthesis.

**Alternatives considered**:
- *Pitch/formant-shifting voice conversion (adult → child)* — Simple and fast but
  produces artifacts; does not model developmental acoustic differences between 12-16
  and 34-38 months; less scientifically defensible as a thesis contribution.
- *SoundStorm / VALL-E / XTTS-v2* — More powerful large-scale models, but require
  far more training data and compute; Providence/Playlogue/Seedlings child speech
  corpus may be insufficient; adds implementation complexity without clear gain.
- *AudioLM* — Requires large codebook pre-training (encodec); overkill for the
  available data scale; time-to-result risk is high for a thesis timeline.

---

## 2. Age Conditioning Mechanism

**Decision**: Speaker-embedding-based age conditioning. Age groups (12-16m, 34-38m)
are treated as distinct speaker roles in ECAPA embedding space. For VITS fine-tuning,
the multi-speaker conditioning vector is a mean ECAPA prototype computed per age group
from the training split (analogous to per-child prototypes in the existing enrollment
pipeline). For the VAE (12-16m), age is a one-hot label concatenated to the latent code.

**Rationale**: Reuses the existing ECAPA-TDNN enrollment infrastructure from
`pyannote/unified.py`. Age groups become "macro-speakers" — the conditioning vector
encodes the mean acoustic identity of that developmental stage. This ensures
architectural consistency with the detection pipeline and allows cross-evaluation
(e.g., using detection similarity scores as synthesis quality proxies).

**Alternatives considered**:
- *Explicit one-hot age label injected as auxiliary input to VITS discriminator* —
  Requires architectural modification; less compatible with pre-trained VITS checkpoints.
- *Separate model per age group* — Simpler to train but wastes data by not sharing
  parameters; harder to compare age-group outputs on equal footing.

---

## 3. TinyVox Dataset Integration

**Decision**: Tentatively include TinyVox as supplementary training data for detection
models, pending verification. TinyVox is a corpus of very short (< 2s) labeled speech
vocalizations covering infant/child age ranges. Its key value is expanding the 12-16
month vocalization coverage.

**Action items before use**:
1. Verify TinyVox age labels are available at clip level and cover 12-16 month range.
2. Confirm annotation format can be converted to RTTM (or frame-label CSV compatible
   with existing `dataset_classes/preprocess.py`).
3. Verify audio is (or can be resampled to) 16kHz mono.
4. Document license/access requirements in dataset provenance section of thesis.

**Rationale**: The 12-16 month age group has sparser coverage in Providence/Playlogue
(most child language acquisition corpora focus on 18m+ when words emerge). TinyVox
specifically targets very early vocalizations. If compatible, it directly addresses the
data scarcity that motivates the synthesis work.

**Contingency**: If TinyVox is not compatible or unavailable, synthesis augmentation
becomes even more critical for the 12-16 month cohort, which strengthens the synthesis
contribution rather than weakening it.

---

## 4. Synthesis Quality Metrics

**Decision**: Four-metric evaluation suite for synthesis quality:
1. **MCD (Mel Cepstral Distortion)** — primary spectral quality metric; compare
   generated mel cepstrum vs. held-out real child speech (aligned via DTW).
   Target: MCD ≤ 8 dB (per SC-003).
2. **ECAPA Speaker Similarity** — cosine similarity between the ECAPA embedding of
   generated audio and the age-group prototype; confirms age-group identity preservation.
3. **Age-Group Accuracy** — an age-group classifier (trained on real speech) applied
   to synthetic samples; confirms age conditioning is effective. Target: ≥ 70% accuracy
   (per SC-003).
4. **F0 Distribution Statistics** — compare pitch (fundamental frequency) histograms
   between generated and real speech per age group. Children have higher and more
   variable F0 than adults; this verifies the synthesis captures that.

**Rationale**: MCD is the field standard for TTS quality. Speaker similarity reuses
existing infrastructure. Age-group accuracy directly tests the conditioning hypothesis.
F0 statistics provide an interpretable, human-understandable quality check that will
translate well into thesis figures.

**Note on MOS**: Large-scale human Mean Opinion Score evaluation is out of scope for
a single-student thesis (requires paid annotation or lab recruitment). The four
objective metrics above constitute a sufficient and reproducible evaluation.

---

## 5. Age Metadata Availability in Training Datasets

**Decision**: Age metadata extraction per dataset:
- **Providence**: CHAT-format transcripts include child age at recording; sessions
  map to approximate months (e.g., `child_age` field). Filter to sessions ≥12 and
  ≤16 months, and ≥34 and ≤38 months.
- **Playlogue**: Existing `anotated_processed.csv` already filters to 14_month and
  36_month timepoints (consistent with seen_child_splits). Use these directly as
  12-16m and 34-38m proxies.
- **Seedlings**: Month-level metadata available via Databrary API (`seedlings_import.py`).
  Sessions range across 6, 10, 14, 18, 24, 30, 36 months — extract 14-month as 12-16m
  proxy and 36-month as 34-38m proxy.
- **TinyVox**: Verify age labels exist; if they use numeric age (months), bin accordingly.

**Key assumption confirmed**: The existing splits (`seen_child_splits/`) already use
14_month and 36_month as the two age-group identifiers, so the new age-stratified
analysis builds directly on that existing classification rather than requiring new
annotation.

---

## 6. Augmentation Strategy

**Decision**:
- Use the same `seen_child_splits/` train/val/test partition as all existing experiments.
- Augment only the **training split** with synthetic samples; val and test remain real-only.
- Augmentation ratio: start with 1:1 (one synthetic per real training sample per age
  group); tune ratio on val F1 (0.5:1, 1:1, 2:1 options).
- Mix at the clip level (not frame level): each synthetic clip is treated as a new
  training example with label = child-present (for positive class), appended to the
  training manifest.

**Rationale**: 
- Same splits ensure a clean apples-to-apples comparison with baseline enrollment results.
- Training-only augmentation is standard practice; augmenting val/test would contaminate
  the evaluation.
- Clip-level mixing is simpler and more compatible with existing `dataset_classes/` loading.

---

## 7. Core Dataset Proxy Analysis Design

**Decision**: Three proxy measures computed on each core dataset session:
1. **Enrollment cosine similarity score** — compute ECAPA embedding for each detected
   vocalization segment (from all three frontends) and compare to the age-group prototype.
   Report distribution of scores per recording as a qualitative detection confidence map.
2. **Inter-frontend agreement** — segment-level agreement between USC-SAIL, Pyannote,
   and BabAR on child-present/absent classification (based on RTTM overlap > 50%).
   A high agreement score suggests the core dataset detections are reliable.
3. **Detection rate statistics** — proportion of frames labeled as child per recording,
   per age group. Compare qualitatively to known ground-truth rates from labeled datasets
   to sanity-check whether the core dataset shows expected developmental trends
   (12-16m sessions might have lower vocalization rates than 34-38m).

**Presentation**: All proxy results are clearly labeled "qualitative supplementary
analysis" in the thesis, not used to support primary quantitative claims (per FR-009,
SC-004, and Constitution Principle II).

---

## 9. Video-Audio ASD Model Integration (Branch 003)

**Decision**: Add video-audio active speaker detection (ASD) as a new diarization frontend category. Integrate TalkNet-ASD and TS-TalkNet first; LoCoNet as stretch goal.

### 9a. SAILS Video File Locations & Naming

SAILS BIDS data lives at `/orcd/scratch/bcs/001/sensein/sails/BIDS_data/final_bids-dataset/derivatives/preprocessed/`. Both `splits/` (cross-child) and `whisper-modeling/seen_child_splits/` (seen-child) CSVs have a `BidsProcessed` column (video path) and `audio_path` column (WAV path).

**Conversion rule** (audio path → video path):
```python
video_path = audio_path.replace("_audio.wav", "_desc-processed_beh.mp4")
```

**Example**:
- Audio: `.../sub-D9N0U7M9X3_ses-02_task-..._run-01_audio.wav`
- Video: `.../sub-D9N0U7M9X3_ses-02_task-..._run-01_desc-processed_beh.mp4`

Confirmed accessible via direct filesystem check. **Important**: video files exist only for SAILS (BIDS dataset). Providence and Playlogue are audio-only; video ASD frontends must raise a descriptive error if the derived video path does not exist.

**Alternatives considered**: Parse `BidsProcessed` column directly. Rejected because `get_segments(audio_path, cfg)` only receives the audio path; in-frontend derivation avoids changing the shared API.

### 9b. Model Selection

| Model | GitHub | Selected | Rationale |
|---|---|---|---|
| TalkNet-ASD | TaoRuijie/TalkNet-ASD | Yes (primary) | Base pipeline; well-documented; establishes face detection + ASD skeleton |
| TS-TalkNet | Jiang-Yidi/TS-TalkNet | Yes (primary) | Only ASD model with speaker enrollment — aligns with existing ECAPA prototype paradigm |
| LoCoNet | SJTUwxz/LoCoNet_ASD | Stretch goal | CVPR 2024 SOTA (95.2% mAP); add after base models work |
| EASEE | N/A (no public repo) | No | Deprioritized — no implementation available |
| AS-Net | N/A (paper only) | No | Deprioritized — no implementation available |
| EG4D | EGO4D/audio-visual | No | Benchmark framework for egocentric camera — less relevant for SAILS home videos |

### 9c. Face Detection Pipeline

**Decision**: S3FD detector (bundled with TalkNet-ASD repo). Face tracks persisted as JSON cache at `pyannote/video_face_cache/` — shared across ASD models to avoid re-running detection.

**Child-face identification strategy**:
- **TS-TalkNet**: speaker enrollment built-in — identifies the child by voice similarity automatically.
- **TalkNet-ASD** (no enrollment): use smallest-face-in-frame heuristic as child proxy. Document this as a known limitation (fails in group scenes or when child is not visible).

### 9d. Integration Architecture

New file `pyannote/video_asd.py` implements `DiarizationFrontend` subclasses:
- `TalkNetASDFrontend` — wraps TalkNet-ASD subprocess (separate `video/` uv env)
- `TSTalkNetFrontend` — wraps TS-TalkNet subprocess; passes reference audio from train split for enrollment
- Both return `List[{"start": float, "end": float}]` (child vocalization segments), identical to existing frontends

Cache: `pyannote/video_asd_rttm_cache/{model_name}/`
Results: `video_asd_ecapa_enrollment_runs/{model_name}/`

### 9e. Python Environment

New `uv`-managed environment at `video/` (top-level, separate from all existing envs):
- Python 3.10, PyTorch ≥ 1.12 with CUDA
- opencv-python, scipy, scikit-learn, tqdm
- speechbrain (for TS-TalkNet ECAPA; installed separately)
- Model checkpoints downloaded to `video/pretrain/` (not committed; documented in setup)

**Unresolved**: TS-TalkNet uses its own ECAPA encoder internally at inference time for speaker enrollment. The shared `unified.py` ECAPA pipeline still operates downstream on the returned segments. These two ECAPA usages are independent — TS-TalkNet's in-model enrollment maps a reference clip to the target speaker; `unified.py`'s ECAPA computes cosine similarity scores for the enrollment evaluation metric. Confirmed: no coupling issue.

---

## 8. Framework Architecture Decision

**Decision**: Extend the existing `pyannote/` evaluation suite rather than building
a new top-level evaluation module. Add three new scripts:
- `pyannote/unified_age_stratified.py` — wraps `unified.py` with age-group filtering
- `pyannote/augmentation_eval.py` — runs detection experiments with synthetic augmentation
- `pyannote/proxy_analysis.py` — core dataset proxy metric computation

Create a new top-level `synthesis/` module (separate uv environment) for:
- `synthesis/train.py`, `synthesis/generate.py`, `synthesis/evaluate.py`

**Rationale**: The existing `pyannote/` infrastructure already handles the multi-diarizer
evaluation loop, ECAPA prototypes, and metrics computation. Extension via new scripts
is far less risky than building a parallel framework. The synthesis module requires
different heavy dependencies (Coqui TTS, custom VAE) that justify a separate environment.
