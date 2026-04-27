# Research: Synthetic Scene Acoustic Realism & Child Encoder Adaptation

**Date**: 2026-04-26 | **Feature**: 009-synth-rir-noise

---

## Decision 1: RIR Application Strategy — Per-Track vs. Mixed-Signal

**Question**: Should RIR be applied to each speaker track independently before mixing
(more realistic — each speaker in the same room), or to the already-mixed signal?

**Decision**: Apply RIR to the already-mixed signal.

**Rationale**: Applying per-track is more physically accurate, but the existing
`_mix_scene_audio` builds a single summed waveform before returning it — splitting
into per-track application would require significant refactoring. For the enrollment
task, the key effect is that the mixed signal sounds reverberant, not precise
per-speaker physics. Single-mix RIR application is the common approach in data
augmentation literature (e.g., Ko et al. 2017, RIR augmentation for ASR).

**Alternatives considered**:
- Per-track RIR: More correct but requires track-level audio storage. Deferred to
  future work if per-speaker RIR is needed.
- Randomized per-track RIR (different rooms per speaker): Maximum diversity but
  physically unrealistic. Rejected.

---

## Decision 2: RIR and Noise File Scanning — Init-Time vs. Pre-Built Lookup

**Question**: Should the RIR/noise file lists be built at `SceneComposer` init time
(scan once per job), or maintained in a separate pre-built lookup table (FR-009)?

**Decision**: Scan at `SceneComposer.__init__` time; no separate pre-built lookup
table.

**Rationale**: 5000 scenes × O(100–500 RIR files) is trivial to re-scan each job.
A pre-built CSV lookup adds an extra file to maintain and provides no benefit at
this scale. FR-009's intent ("scan and register into a lookup table usable by the
generator") is satisfied by an in-memory list populated at init.

**Alternatives considered**:
- Pre-built `rir_manifest.csv`: More flexible for large RIR sets (>10k files) but
  overkill here. If the RIR pool grows beyond ~1k files, revisit.

---

## Decision 3: RIR and MUSAN Dataset Locations

**Question**: Where are the RIR and MUSAN noise files on the cluster?

**Finding**: A cluster search at depth ≤5 under `/orcd/scratch/bcs/001` and
`/orcd/scratch/orcd` found no directories named `musan`, `MUSAN`, `rir`, or `RIR`.
The files are not pre-staged.

**Decision**: Treat as "data not yet available." The graceful fallback (FR-005) means
the generator can be coded and tested without them; scene re-generation with acoustic
augmentation is gated on the user staging the data.

**Action required (user)**: Before running T8 (scene re-generation):
1. MUSAN (music + noise subset, ~17 GB):
   - Download from [OpenSLR](https://www.openslr.org/17/) or copy from a group share
   - Set `--noise-dir /path/to/musan/noise` (and optionally `/musan/music`)
2. RIR files (one of):
   - OpenSLR 26 — "Room Impulse Responses and Noises" (~1 GB):
     `wget https://www.openslr.org/resources/26/sim_rir_16k.zip`
   - MIT IR Survey (~500 measured RIRs, free):
     `git clone https://mcdermottlab.mit.edu/Reverb/IR_Survey.git`
   - Set `--rir-dir /path/to/rir_files/`

**Workaround**: If data is unavailable before the thesis deadline, Story 1 can be
demonstrated with graceful-fallback mode (clean mix, `rir_id=null`, `noise_id=null`)
and FR-005 acceptance still passes.

---

## Decision 4: Config Discrepancy — `low_snr_stress.yaml` SNR Range

**Finding**: `low_snr_stress.yaml` has `snr_db_min: -5`, but FR-007/SC-002 require
SNR ∈ [0, 5] dB for positive scenes. Negative SNR means noise louder than speech,
which may be useful for robustness testing but violates the spec as written.

**Decision**: Change `snr_db_min: -5` → `snr_db_min: 0` to match FR-007/SC-002.

**Rationale**: The spec and success criterion are unambiguous. If very-low-SNR
robustness is a separate research question, create a distinct config (e.g.,
`extreme_noise_stress.yaml`) rather than modifying this one.

---

## Decision 5: SNR Sampling and Clamping

**Question**: How to handle SNR clamping (edge case from spec)?

**Decision**: Sample `snr_db = rng.uniform(snr_db_min, snr_db_max)` then clamp to
`[snr_db_min, snr_db_max]` before calling `mix_at_snr`. Since `rng.uniform` already
draws from the closed-open interval `[low, high)`, clamping is a safety net for
floating-point edge cases, not a common code path.

---

## Decision 6: Noise ID Format

**Question**: What should `noise_id` contain — filename stem, full path, or a
MUSAN-style category?

**Decision**: Use the filename stem (e.g., `noise-free-sound-0000`). This is
reproducible, short, and captures the specific file. The MUSAN category (noise vs.
music) is recoverable from the parent directory if needed and is implicit in
`noise_dir` choice.

---

## Decision 7: SSL Pretraining Approach (Stretch — Story 3)

**Question**: How to implement continued WavLM masked-speech-prediction on TinyVox?

**Decision**: Use HuggingFace `transformers` with `WavLMForPreTraining` or
`Wav2Vec2ForPreTraining` (same objective as WavLM's GSLM-based masked prediction).
Load `microsoft/wavlm-base-plus` as starting checkpoint, run continued pretraining
on TinyVox (+ optionally Providence) WAVs for a configurable number of steps.

**Rationale**: HuggingFace provides a documented continued-pretraining path for
WavLM that avoids re-implementing the masking objective. The resulting checkpoint
uses the standard `WavLMModel` interface, which `BackboneExtractor` in `mil_model.py`
already loads via `WavLMModel.from_pretrained()` — zero MIL code change needed.

**Prerequisites**:
- TinyVox Eng-NA: ~10 h child speech (available at `data/tinyvox/`)
- Providence: ~8 h child speech (available; covered by existing RTTM-segmented segments)
- Total: ~18 h — above the 50-hour minimum flagged in the spec as the hard floor.
  **Risk**: 18 h < 50 h spec threshold. Must check whether spec floor is achievable
  with full TinyVox (not just Eng-NA) or whether the 50 h floor should be lowered
  for a stretch goal.

**Alternatives considered**:
- BabyHuBERT: published but model weights not publicly released as of 2026-04.
- Continuing Wav2Vec 2.0: similar approach but WavLM-Base+ is the current backbone.
- Full retraining from scratch: requires >1000 h of child speech; infeasible.

---

## Decision 8: Avoid `build_segment_manifest.py` --skip-quality for TinyVox

**Decision**: Default quality scoring (duration-only `min(1.0, dur/1.0)`) is used
for TinyVox because TinyVox segments are pre-vetted phoneme clips. The audio-based
quality check (RMS + silence ratio) is not needed and would add significant runtime
with no expected benefit.

**Rationale**: TinyVox files are already segmented and aligned; RMS-based scoring
could penalize legitimate but short child vocalizations.
