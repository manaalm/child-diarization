# Feature Specification: Synthetic Scene Acoustic Realism & Child Encoder Adaptation

**Feature Branch**: `009-synth-rir-noise`
**Created**: 2026-04-26
**Status**: Draft

## User Scenarios & Testing *(mandatory)*

### User Story 1 — Acoustically Realistic Scene Generation (Priority: P1)

A researcher runs the synthetic scene generator to augment a child speech enrollment
system. Currently the generated WAVs are clean speech mixes (no room acoustics, no
background noise) even though the config requests RIR and noise application. The
researcher needs the generated scenes to sound like real home/daycare recordings so
the augmented enrollment model generalises to unseen conditions.

**Why this priority**: The cut-and-paste + turn-taking scaffold already exists. The
only thing blocking the strongest version of the primary synthesis path is the
unstubbed RIR convolution and noise mixing code paths. This is the highest-ROI
change: no new infrastructure, no training, minimal compute.

**Independent Test**: Run `generate_scenes.py` with `apply_rir_probability: 0.7` and
`apply_noise_probability: 0.8`. Sample 20 output WAVs and confirm that (a) at least
12 of them exhibit audible reverberation, (b) at least 14 exhibit audible background
noise, and (c) no output is silent. This can be validated before any downstream
training.

**Acceptance Scenarios**:

1. **Given** a scene config with `apply_rir_probability: 0.7`, **When** 100 scenes are
   generated, **Then** roughly 70 ± 10 have RIR metadata populated (`rir_id ≠ null`)
   and audibly contain reverberation.
2. **Given** a scene config with `apply_noise_probability: 0.8` and MUSAN noise files
   available, **When** 100 scenes are generated, **Then** roughly 80 ± 10 scenes have
   `noise_id ≠ null` and SNR values within the configured `[snr_db_min, snr_db_max]`
   range.
3. **Given** a `low_snr_stress.yaml` config, **When** scenes are generated, **Then**
   all positive scenes have `mean_snr_db` within the low-SNR band (0–5 dB).
4. **Given** no noise files are present (MUSAN unavailable), **When** scenes are
   generated with `apply_noise_probability > 0`, **Then** the generator falls back
   gracefully (clean mix, `noise_id = null`), logs a warning, and does not crash.
5. **Given** no RIR files are present, **When** scenes are generated with
   `apply_rir_probability > 0`, **Then** the generator falls back gracefully
   (`rir_id = null`) and does not crash.

---

### User Story 2 — Re-run Augmentation Experiment With Acoustic Scenes (Priority: P2)

After scenes are regenerated with RIR + noise, the researcher re-runs the 6-ratio
enrollment experiment (ratios 0× – 10×) to measure whether acoustic realism produces
a larger AUPRC improvement over the clean-mix baseline.

**Why this priority**: The downstream experiment is the scientific contribution. The
generator fix (Story 1) is a prerequisite; this story is the payoff.

**Independent Test**: `metrics_by_ratio.csv` exists with 6 rows; AUPRC at the best
ratio exceeds the AUPRC from the clean-mix run stored in the same file at ratio 0×.

**Acceptance Scenarios**:

1. **Given** acoustically realistic scenes generated in Story 1, **When**
   `run_ratio_sweep.sh` completes, **Then** `metrics_by_ratio.csv` shows at least one
   ratio with AUPRC ≥ 0.92 (above the current clean-mix best of 0.918).
2. **Given** the sweep results, **When** `error_analysis_synthetic.py` runs, **Then**
   age-band breakdown shows 14-month and 36-month rows for every ratio.
3. **Given** both clean-mix and acoustic-mix experiment directories, **When** metrics
   are compared, **Then** the acoustic experiment is the canonical result reported
   in the thesis.

---

### User Story 3 — Child-Adapted WavLM Encoder via Continued SSL Pretraining (Priority: P3)

A researcher wants a child-speech-aware encoder as a drop-in replacement for the
frozen WavLM-Base+ backbone used in the MIL pipeline. The encoder should be pretrained
by continuing WavLM masked-speech-prediction on TinyVox (and Providence) child speech,
producing lower frame-level error on child phonemes than the original adult-trained
model.

**Why this priority**: Stretch goal; requires cluster allocation and SSL training
infrastructure not yet present. Valuable as a clean thesis contribution ("child
BabyHuBERT replacement") but blocked on infrastructure setup.

**Independent Test**: A child-adapted checkpoint exists at a known path. When it
replaces `wavlm_mil`'s backbone, MIL test AUPRC does not decrease and ideally
improves. Frame-level phoneme error on held-out Providence child speech is lower
than the baseline WavLM-Base+ checkpoint.

**Acceptance Scenarios**:

1. **Given** TinyVox + Providence child audio (≥ 100 hours), **When** continued
   SSL pretraining runs for a configurable number of steps, **Then** a checkpoint
   file is saved and the training loss curves are logged.
2. **Given** the child-adapted checkpoint, **When** it is loaded into the existing
   MIL backbone slot (zero code change to MIL training), **Then** training runs
   without error and converges.
3. **Given** both baseline and child-adapted MIL checkpoints evaluated on the test
   split, **When** AUPRC values are compared, **Then** the child-adapted model
   achieves AUPRC ≥ the WavLM-Base+ baseline (no regression).

---

### Edge Cases

- What happens when RIR files exist but all are corrupted / wrong length?
  Generator must skip bad files, log a warning per file, and continue.
- What if the MUSAN noise directory exists but is empty?
  Graceful fallback to clean mix with logged warning.
- What if SNR sampling produces a value outside `[snr_db_min, snr_db_max]` due to
  floating point? Clamp to range before applying.
- What if TinyVox total duration is insufficient for continued SSL pretraining?
  Pipeline must report data hours at startup and refuse to run below a configurable
  minimum (e.g., 50 hours).
- What if the child-adapted model has a different feature-dimension than WavLM-Base+?
  The MIL head must remain compatible; document any dimension requirements.

## Requirements *(mandatory)*

### Functional Requirements

**Primary path (Stories 1–2)**

- **FR-001**: The scene generator MUST apply room impulse responses (from a
  configurable directory of RIR WAV files) to speaker tracks before mixing, with
  probability equal to `apply_rir_probability` in the scene config.
- **FR-002**: Each applied RIR MUST be randomly selected from the available RIR pool;
  the selected file ID MUST be recorded in `scene_meta["rir_id"]`.
- **FR-003**: The scene generator MUST add background noise at a randomly sampled SNR
  within `[snr_db_min, snr_db_max]`, with probability equal to
  `apply_noise_probability`.
- **FR-004**: The selected noise file ID MUST be recorded in `scene_meta["noise_id"]`
  and the actual SNR in `scene_meta["mean_snr_db"]`.
- **FR-005**: When no RIR or noise files are available, the generator MUST fall back
  to the current behaviour (clean mix) without raising an exception.
- **FR-006**: Generated scene JSON metadata MUST include `rir_id`, `noise_id`, and
  `mean_snr_db` fields (null when not applied).
- **FR-007**: The `low_snr_stress.yaml` config MUST produce scenes where all
  positive-scene SNR values fall within 0–5 dB.
- **FR-008**: Existing helper functions `convolve_rir` and `mix_at_snr` in
  `audio_utils.py` MUST be wired into `_mix_scene_audio`; no new audio processing
  primitives are required.
- **FR-009**: A manifest-building step (`build_segment_manifest.py` or a new sibling
  script) MUST scan and register available RIR and noise files into a lookup table
  usable by the generator.
- **FR-014**: `build_segment_manifest.py` MUST support a `--tinyvox-dir` argument
  that scans TinyVox pre-segmented phoneme WAVs (filename format
  `phon_{lang}_{corpus}_{speaker}_{session}_{start_ms}_{end_ms}.wav`) and appends
  them to the segment manifest alongside Providence child segments. Speakers present
  in `--exclude-speakers-csv` MUST be marked `usable_for_training=false` to prevent
  leakage. Only English-NA (`Eng-NA`) files are included by default; the filter must
  be configurable. Age band is inferred from the YYMMDD session field using the same
  logic as Providence.

**Stretch path (Story 3)**

- **FR-010**: A new training script MUST accept a pre-trained WavLM checkpoint and
  continue masked-speech-unit prediction on a provided list of child speech WAV files.
- **FR-011**: The script MUST support resuming from the latest saved checkpoint.
- **FR-012**: The resulting checkpoint MUST be loadable by the existing MIL
  `BackboneExtractor` class without code changes to `mil_model.py`.
- **FR-013**: Training MUST be launchable via a SLURM script targeting the
  `ou_bcs_normal,pi_satra` partitions with GPU.

### Key Entities

- **RIR file**: A WAV containing a measured or simulated room impulse response;
  identified by file path and a short canonical ID.
- **Noise file**: A WAV from MUSAN (noise or music subset) used as additive
  background; identified by file path and MUSAN category.
- **Scene config**: YAML controlling scene generation probabilities and mixing
  parameters; extended with RIR/noise directory paths.
- **Child-adapted checkpoint**: A WavLM model checkpoint fine-tuned on child speech;
  consumable by `BackboneExtractor` with feature dimension 768.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: After Story 1, at least 70% of generated scenes with RIR enabled
  have a non-null `rir_id` in their JSON metadata.
- **SC-002**: After Story 1, all scenes generated with `low_snr_stress.yaml` have
  `mean_snr_db` between 0 and 5 dB (inclusive).
- **SC-003**: After Story 2, the best-ratio AUPRC from the acoustic-mix experiment
  is at least 0.005 above the clean-mix best-ratio AUPRC (effect size sufficient
  to report).
- **SC-004**: After Story 2, the full 6-ratio sweep completes within the 48-hour
  SLURM walltime.
- **SC-007**: After FR-014 is implemented, running `build_segment_manifest.py`
  with `--tinyvox-dir data/tinyvox/` adds at least 20,000 child segments
  (`source_dataset=tinyvox`) to the manifest with no test-child leakage
  (`usable_for_training=false` for all excluded speakers).
- **SC-005 (stretch)**: The child-adapted MIL model achieves test AUPRC ≥ 0.946
  (matching or exceeding the current Whisper-MIL baseline) on the seen-child split.
- **SC-006 (stretch)**: SSL pretraining completes within a single 48-hour GPU job.

## Assumptions

- RIR files from MIT IR Survey or RIRs & Noises dataset are available at a
  configurable path on the cluster (not committed to the repo).
- MUSAN noise and music WAVs are available at a configurable path on the cluster.
- TinyVox data is already present at `data/tinyvox/audio/` on the cluster
  (~64 k pre-segmented phoneme WAVs; Eng-NA subset ≈ 24 k files, ~10 hours).
  It is used as an additional child speech source for scene generation (Stories 1–2)
  and as pretraining data (Story 3 stretch).
- For Story 3, continued pretraining uses the same masked-speech-unit prediction
  objective as WavLM; no new objective design is needed.
- The existing `convolve_rir` and `mix_at_snr` functions in `audio_utils.py` are
  correct implementations; they only need to be called.
- The 5000 already-generated clean-mix scenes are discarded and regenerated;
  they are gitignored and not committed.
- Story 3 is time-permitting and does not block the thesis submission.
