# Research: Synthetic Child-Adult Scene Generator

**Date**: 2026-04-24 | **Plan**: [plan.md](plan.md)

All NEEDS CLARIFICATION items resolved. No unknowns remain.

---

## D1: Audio Mixing Approach

**Decision**: Mix-to-SNR with peak normalization after final mix.

**Rationale**: Mix-to-SNR is the standard approach in speech augmentation literature (SpecAugment, SpeechBrain data recipes, MUSAN mixing scripts). Scale each source signal so that the ratio of its RMS energy to the background (noise + other speakers) matches the sampled target SNR. Apply peak normalization at the end to avoid clipping. This is CPU-efficient (no neural components), reproducible given the same seed, and directly configurable via the scene YAML.

**Alternatives considered**:
- LUFS-normalized loudness matching: more perceptually accurate but adds a librosa dependency and per-segment loudness computation step; not needed for downstream task utility over perceptual quality.
- SDR-based mixing: relevant for source separation evaluation but overkill for data augmentation where we just need controllable child-to-noise ratios.

---

## D2: Turn-Taking Simulator

**Decision**: Discrete-time Markov chain over {TARGET_CHILD, ADULT_0, PAUSE} states with configurable age-band-specific transition matrices and duration distributions.

**Rationale**: Published work on synthetic conversation generation (e.g., LibriMix, Fisher corpus simulations) uses Markov-chain speaker sequencing. For parent-child interaction, CHILDES and Soderstrom et al. (2008) show that turn transitions are predominantly adult→child and child→adult with short pauses (median 0.5–1.5 s). A Markov chain captures this without requiring an ML model. Duration for each turn is independently sampled from a log-normal or truncated normal distribution (configurable per age band and speaker role).

**Alternatives considered**:
- Poisson point process for turn onsets: simpler but doesn't naturally encode the alternating adult/child structure.
- Learned neural turn-taking model: accurate but requires training data and adds a stretch-goal dependency; unnecessary for the MVP research question.
- Fixed round-robin alternation: too rigid; doesn't reproduce realistic variation in pause duration or overlap probability.

**Age-band defaults** (drawn from CHILDES turn-taking literature):
- 14–18 months: child turns short (mean 0.6 s, σ=0.3 s), high variability, low lexical density; adult turns longer (mean 3.5 s, σ=1.5 s); pause mean 0.8 s.
- 34–38 months: child turns longer (mean 1.8 s, σ=0.8 s); adult turns similar; pause mean 0.6 s; higher overlap probability (0.20 vs 0.12 for 14–18 months).

---

## D3: Room Impulse Response Application

**Decision**: Offline FFT-based convolution using scipy.signal.fftconvolve, applied per-speaker before mixing.

**Rationale**: FFT convolution is the standard for applying pre-recorded RIRs in data augmentation (used by RIRS_NOISES, SpeechBrain, Lhotse). Each speaker in the scene is convolved with a sampled RIR (optionally the same room for all speakers, or different RIRs to simulate different positions). Applying RIR before mixing ensures RTTM labels remain exact — convolution does not change segment boundaries.

**RIR sources** (priority order for MVP):
1. RIRS_NOISES (real-room and simulated RIRs, freely downloadable, ~1.3 GB): primary source.
2. MIT IR Survey (real-room RIRs, downloadable): secondary for room diversity.
3. BUT ReverbDB (optional): additional variety.
4. Synthetic RIRs via pyroomacoustics (stretch): parameterized small-room simulation; requires additional dependency.

**Alternatives considered**:
- Time-domain convolution: correct but 10–100× slower than FFT convolution for long RIRs; not suitable for CPU batch generation.
- GPU-accelerated convolution: unnecessary; CPU FFT convolution on 30-second clips with typical RIR lengths (<1 s) completes in milliseconds.

---

## D4: Segment Quality Proxy

**Decision**: Use a composite quality score: `0.5 × energy_score + 0.3 × duration_score + 0.2 × silence_ratio_score` where each component is 0–1 scaled. Default threshold: 0.4. Segments from corpora with explicit quality annotations (e.g., TinyVox phonetic transcripts) use the corpus-provided score directly.

**Rationale**: Providence and CHILDES-derived corpora do not provide per-segment perceptual quality scores. A proxy based on RMS energy (normalized within dataset), duration (0.3–10 s is usable range), and silence ratio (fraction of frames below a noise floor threshold) correlates well with listener-rated quality in prior data-cleaning work. The threshold of 0.4 is conservative; it is a config parameter so researchers can adjust.

**Alternatives considered**:
- DNSMOS (neural MOS estimator): accurate but requires a model inference call per segment; too slow for manifest building on large corpora; could be added as an optional quality scorer.
- Manual auditing: not scalable to 560k TinyVox clips; reserved for spot-checking a random sample.

---

## D5: Child Segment Source Priority

**Decision**: Providence child segments (already in repo with RTTM annotations) are the bootstrapping source. TinyVox is added in a second pass.

**Rationale**: Providence child RTTMs are already parsed by `pyannote/unified_rttm.py`; `extract_segments.py` can reuse that parsing logic. This means zero new data infrastructure is needed to start generating scenes. TinyVox adds diversity (560k clips, 5 languages) but requires a download and its RTTM-equivalent format (phonetically timed CHA files) needs a dedicated parser.

**Source inventory**:

| Dataset | Access | License | Age range | Annotation | Use |
|---------|--------|---------|-----------|------------|-----|
| Providence | Already in repo (`providence/`) | CHILDES/TalkBank; academic use | 1–5 years | Existing RTTM | Child + adult segments; primary MVP source |
| TinyVox | TalkBank/PhonBank (download via PhonBank API) | CC BY | 0–6 years, 5 languages | Phonetically segmented CHA | Diverse child clips; second-pass source |
| LibriSpeech train-clean-100 | OpenSLR (download) | CC BY 4.0 | Adults | Word-aligned | Clean adult speech baseline |
| MUSAN speech subset | OpenSLR (download) | CC BY 4.0 | Mixed | None | Background speech / TV-like negatives |
| RIRS_NOISES | OpenSLR (download) | CC BY 4.0 | N/A | N/A | Room impulse responses |
| MUSAN noise/music | OpenSLR (download) | CC BY 4.0 | N/A | N/A | Home background noise |

**Restricted / do-not-use for reproducible core**: MyST (LDC agreement), ECOLANG (varies by institution), SAILS BIDS data (IRB-gated; used only as real held-out test set, never as synthetic source).

---

## D6: Overlap Handling

**Decision**: Overlap is modeled by advancing the scene cursor backward by a sampled overlap duration when the overlap condition fires. Both overlapping segments retain their original audio; they are mixed additively. RTTM correctly records both speaker intervals simultaneously.

**Rationale**: Additive mixing of overlapping segments is how all real mixed-speech data is generated. The RTTM standard natively supports overlapping intervals (two SPEAKER lines covering the same time range), which is what pyannote and EEND-EDA already consume. No special post-processing is needed.

**Implementation**: When the turn-taking simulator selects an overlap event, it returns a negative pause duration (i.e., the next segment starts before the current one ends). `SceneComposer.place_segment()` handles this by using `start = max(0, cursor + pause)` where pause can be negative.

---

## D7: Integration with Existing Pipeline

**Decision**: Synthetic scenes plug into the existing pipeline by augmenting `whisper-modeling/seen_child_splits/train.csv` with synthetic rows. No existing script is rewritten.

**Rationale**: The enrollment pipeline (`pyannote/unified.py`, `babar_ecapa_enrollment_runs/`) reads `train.csv` to build ECAPA prototypes. Adding synthetic rows expands the training pool without changing any evaluation code. The `train_with_synthetic.py` script produces augmented CSVs at each ratio; `evaluate_synthetic_augmentation.py` calls the existing evaluation scripts against those augmented CSVs.

**Augmented CSV columns that must match existing format**:
- `audio_path`, `label` (0/1), `child_id`, `timepoint_norm`, `split`
- Synthetic rows use `child_id = synthetic_{scene_id}` and `timepoint_norm` matching the age band of the source child segments.

---

## D8: RTTM Label Standardization

**Decision**: Use uppercase standardized labels: `TARGET_CHILD`, `ADULT_0`, `ADULT_1`, `OTHER_CHILD_0`, `BACKGROUND_SPEECH`. These are synthetic-data-specific labels; downstream scripts that consume synthetic RTTMs must map them to their expected label set (e.g., `CHI` for BabAR-style evaluation).

**Rationale**: The existing pipeline uses `CHI`, `KCHI`, `ADT` etc. from specific diarizers. Using distinct labels for synthetic scenes makes it unambiguous whether an RTTM came from real vs. synthetic data. A one-line mapping in `evaluate_synthetic_augmentation.py` converts between label sets.

---

## D9: Synthetic Scene Duration

**Decision**: Default 30 seconds, matching the approximate duration of existing SAILS BIDS clips. Configurable.

**Rationale**: The seen-child split clips average ~30 s (SAILS BIDS recordings are structured as 30-second clips). Matching this duration ensures the downstream clip classifier sees synthetic clips of the same length as real training clips, avoiding distributional shift in clip length.

---

## D10: Split Integrity Enforcement for Synthetic Sources

**Decision**: Providence child speakers assigned to the real test split in `seen_child_splits/test.csv` are excluded from synthetic training segment sourcing. The manifest builder reads the split file and sets `usable_for_training = false` for those speaker IDs.

**Rationale**: A Providence child speaker who appears in the real test set must not contribute to synthetic training data, even indirectly via their vocalization style. This enforces Constitution Principle II strictly.

**Implementation**: `build_segment_manifest.py` accepts `--exclude-speakers-csv` pointing to the test split CSV; it reads the `child_id` column and marks matching Providence speaker segments as `usable_for_training = false` and `split = test`. Non-Providence sources (TinyVox, LibriSpeech) have no overlap with the SAILS test set and are always `usable_for_training = true` for the training pool.
