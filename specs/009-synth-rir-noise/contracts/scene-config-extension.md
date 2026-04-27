# Contract: Scene Config YAML — Acoustic Augmentation Extension

**Feature**: 009-synth-rir-noise | **Version**: 1.0

This extends the existing scene config contract (defined in `synth/contracts/scene-config.md`)
with the new keys required for acoustic augmentation.

---

## New Keys in `mixing:` Block

```yaml
mixing:
  # --- existing keys (unchanged) ---
  snr_db_min: 0           # float, dB; lower SNR sampling bound
  snr_db_max: 25          # float, dB; upper SNR sampling bound
  apply_rir_probability: 0.7   # float [0, 1]; probability a scene gets RIR
  apply_noise_probability: 0.8 # float [0, 1]; probability a scene gets noise
  crossfade_ms: 20.0      # float, ms
  peak_normalize: true    # bool
  save_stems: false       # bool

  # --- NEW in 009-synth-rir-noise ---
  rir_dir: ""             # str; path to dir of RIR WAV/FLAC files; "" → no RIR
  noise_dir: ""           # str; path to dir of noise WAV files (MUSAN layout); "" → no noise
```

---

## Validation Rules

| Key | Rule |
|-----|------|
| `rir_dir` | If non-empty, directory must exist at job start; missing → warning + fallback (not error) |
| `noise_dir` | Same as above |
| `snr_db_min` | Must be ≤ `snr_db_max` |
| `apply_rir_probability` | Float in [0.0, 1.0] |
| `apply_noise_probability` | Float in [0.0, 1.0] |

---

## CLI Override

`generate_scenes.py` accepts `--rir-dir PATH` and `--noise-dir PATH` that override
the YAML keys at runtime. This allows cluster paths to be injected without editing
committed config files.

Priority: CLI arg > YAML key > empty string (no augmentation).

---

## Backward Compatibility

Existing configs without `rir_dir`/`noise_dir` keys behave identically to before —
both pools default to `[]` and the generator produces clean mixes. No breaking change.

---

## `low_snr_stress.yaml` — Corrected SNR Range

Per FR-007/SC-002, `snr_db_min` must be `0` (not `-5`) to ensure all positive
scenes have SNR ∈ [0, 5] dB. This is a **spec-breaking correction** to the existing
config; scenes generated with the old `-5` floor must be discarded and regenerated.
