# Data Model: Synthetic Scene Acoustic Realism & Child Encoder Adaptation

**Date**: 2026-04-26 | **Feature**: 009-synth-rir-noise

---

## Entities

### 1. SceneComposer (runtime object)

Extended from base MVP. New fields added to `__init__`:

| Field | Type | Source | Description |
|-------|------|--------|-------------|
| `_rir_pool` | `list[Path]` | scanned from `mixing.rir_dir` | RIR WAV files available at job start; `[]` if dir absent |
| `_noise_pool` | `list[Path]` | scanned from `mixing.noise_dir` | Noise WAV files available at job start; `[]` if dir absent |
| `_rir_dir` | `str` | `mixing.rir_dir` config key | Configurable path; empty string → clean mix |
| `_noise_dir` | `str` | `mixing.noise_dir` config key | Configurable path; empty string → clean mix |

---

### 2. Scene Config (YAML) — Extended Mixing Block

New keys in the `mixing:` section of all scene configs:

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `rir_dir` | str | `""` | Path to directory of RIR WAV/FLAC files; empty = no RIR |
| `noise_dir` | str | `""` | Path to directory of noise WAV files (e.g., MUSAN); empty = no noise |
| `apply_rir_probability` | float | `0.0` | Fraction of scenes that receive RIR; already present in configs |
| `apply_noise_probability` | float | `0.0` | Fraction of scenes that receive noise; already present |
| `snr_db_min` | float | `0` | Lower bound of SNR sampling range (dB) |
| `snr_db_max` | float | `25` | Upper bound of SNR sampling range (dB) |

---

### 3. Scene Metadata JSON — Extended Fields

The per-scene JSON written by `write_scene_metadata()` already reserves these fields
(currently always `null`). After this feature they become non-null when applicable:

| Field | Type | Values | Description |
|-------|------|--------|-------------|
| `rir_id` | str or null | `"<filename_stem>"` or `null` | ID of the applied RIR file; null if not applied |
| `noise_id` | str or null | `"<filename_stem>"` or `null` | ID of the applied noise file; null if not applied |
| `mean_snr_db` | float or null | float in `[snr_db_min, snr_db_max]` | Actual SNR used; null if no noise applied |

---

### 4. Segment Manifest Row — Extended Sources

TinyVox rows (FR-014, already implemented) add new `source_dataset` values:

| Column | Value (TinyVox) | Notes |
|--------|-----------------|-------|
| `source_dataset` | `"tinyvox"` | Distinguishes from `"providence"` / `"librispeech"` |
| `speaker_role` | `"target_child"` | Same role as Providence child segments |
| `age_band` | derived from YYMMDD session field | Same logic as Providence |
| `audio_path` | absolute path to `phon_*.wav` | Each file is its own segment (start=0.0) |
| `start_time_sec` | `0.0` | File is pre-segmented |
| `end_time_sec` | `(end_ms - start_ms) / 1000` | Computed from filename |

---

### 5. RIR File (referenced entity)

Not stored in the manifest; scanned at runtime into `SceneComposer._rir_pool`.

| Attribute | Description |
|-----------|-------------|
| `path` | Absolute path to WAV/FLAC on cluster |
| `rir_id` | `path.stem` — unique identifier for logging |
| `sample_rate` | Any; resampled to 16 kHz before convolution |
| `duration` | Any; `convolve_rir` handles arbitrary-length RIRs via FFT convolution |

Validity constraints:
- Must be loadable by `soundfile.read()`
- Duration < 10 s recommended (long RIRs slow FFT convolution; warn but don't skip)
- Corrupted/empty files skipped with per-file warning; do not crash

---

### 6. Noise File (referenced entity)

Not stored in manifest; scanned at runtime into `SceneComposer._noise_pool`.

| Attribute | Description |
|-----------|-------------|
| `path` | Absolute path to WAV on cluster |
| `noise_id` | `path.stem` |
| `sample_rate` | Any; resampled to 16 kHz if needed |
| `duration` | Any; `mix_at_snr` loop-tiles short files to match scene length |

---

### 7. Metrics by Ratio Table (`metrics_by_ratio.csv`)

Produced by `evaluate_synthetic_augmentation.py`. One row per augmentation ratio:

| Column | Type | Description |
|--------|------|-------------|
| `ratio` | str | `"0x"`, `"0.5x"`, `"1x"`, `"2x"`, `"5x"`, `"10x"` |
| `f1` | float | Test F1 at tuned threshold |
| `precision` | float | Test precision |
| `recall` | float | Test recall |
| `auroc` | float | Test AUROC |
| `auprc` | float | Test AUPRC |
| `threshold` | float | Val-tuned threshold |

SC-003 is satisfied when `max(auprc) >= baseline_0x_auprc + 0.005`.

---

### 8. Child-Adapted Checkpoint (Stretch — Story 3)

Produced by `pretrain_wavlm_child.py`; consumed by `mil/mil_model.py`
`BackboneExtractor` with no code changes.

| Attribute | Constraint |
|-----------|------------|
| Format | HuggingFace `WavLMModel` `save_pretrained()` directory |
| Feature dimension | 768 (must match WavLM-Base+; no MIL head change needed) |
| Path | Configurable; e.g., `synth_results/child_wavlm_checkpoint/` |
| Load check | `WavLMModel.from_pretrained(path)` must not raise |

---

## State Transitions

```
Scene generation pipeline:

  segment_manifest.csv ──→ SceneComposer ──→ WAV + RTTM + JSON per scene
       ↑                         ↑
  _scan_tinyvox()          _mix_scene_audio()
  (DONE)                   (to be wired: RIR + noise)

Augmentation experiment:

  acoustic scenes ──→ enrollment (BabAR ECAPA) ──→ metrics_by_ratio.csv
                ↑
  compare vs. clean-mix baseline (job 12603925)
```
