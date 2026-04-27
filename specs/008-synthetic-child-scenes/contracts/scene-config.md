# Contract: Scene Configuration YAML

**Files**: `synth/configs/*.yaml`
**Consumed by**: `synth/scripts/generate_scenes.py`

---

## Schema

```yaml
project:
  name: string                    # run identifier; used in scene IDs and output dirs
  sample_rate: int                # must be 16000
  random_seed: int                # global seed; per-scene seed = random_seed + scene_index

scene:
  duration_sec: float             # default 30.0; must be > 0
  n_scenes: int                   # number of scenes to generate; must be > 0
  target_age_band: string         # "14_18_months" or "34_38_months"
  max_speakers: int               # max simultaneous speakers in one scene; default 3

sources:
  child_segments:                 # list of dataset names to draw child segments from
    - string                      # e.g. "providence", "tinyvox"
  adult_segments:                 # list of dataset names for adult segments
    - string                      # e.g. "providence_adults", "librispeech"
  noise:                          # list of noise source names
    - string                      # e.g. "musan_noise", "musan_music"
  rirs:                           # list of RIR source names
    - string                      # e.g. "rirs_noises", "mit_ir_survey"

sampling:
  positive_scene_probability: float            # [0,1]; scenes where TARGET_CHILD vocalizes
  adult_only_negative_probability: float       # [0,1]; adult speech only, no child
  background_speech_negative_probability: float # [0,1]; TV/far-field speech negatives
  noise_only_negative_probability: float       # [0,1]; silence or household noise only
  # above four must sum to 1.0
  overlap_probability: float                   # [0,1]; prob any turn pair overlaps
  short_child_vocalization_probability: float  # [0,1]; child segment dur < short_threshold_sec
  short_threshold_sec: float                   # default 0.5; max dur for "short" vocalizations
  other_child_probability: float               # [0,1]; include OTHER_CHILD_0 in positive scenes

turn_taking:
  # 14_18_months band defaults
  child_turn_duration_mean_sec: float          # default 0.6
  child_turn_duration_std_sec: float           # default 0.3
  adult_turn_duration_mean_sec: float          # default 3.5
  adult_turn_duration_std_sec: float           # default 1.5
  pause_mean_sec: float                        # default 0.8
  pause_std_sec: float                         # default 0.3
  # 34_38_months overrides (optional; if absent, same as above)
  child_turn_duration_mean_sec_34_38: float    # default 1.8
  child_turn_duration_std_sec_34_38: float     # default 0.8
  pause_mean_sec_34_38: float                  # default 0.6
  n_turns_min: int                             # default 2
  n_turns_max: int                             # default 20

mixing:
  snr_db_min: float                            # default 0; child-to-background SNR floor
  snr_db_max: float                            # default 25
  apply_rir_probability: float                 # [0,1]; default 0.7
  apply_noise_probability: float               # [0,1]; default 0.8
  crossfade_ms: float                          # default 20.0
  peak_normalize: bool                         # default true; normalize to avoid clipping
  save_stems: bool                             # default false; save per-speaker dry WAVs

labels:
  generate_rttm: bool                          # default true
  generate_clip_labels: bool                   # default true
  target_child_label: string                   # default "TARGET_CHILD"
```

## Constraints

- `positive_scene_probability + adult_only_negative_probability + background_speech_negative_probability + noise_only_negative_probability` must equal 1.0 (validated at load time).
- `snr_db_min ≤ snr_db_max`.
- `sample_rate` must equal 16000.
- `scene_duration_sec` must be > 0.
- `random_seed` must be an integer ≥ 0.

## Example: default_14_18mo.yaml

```yaml
project:
  name: default_14_18mo
  sample_rate: 16000
  random_seed: 42

scene:
  duration_sec: 30.0
  n_scenes: 5000
  target_age_band: 14_18_months
  max_speakers: 3

sources:
  child_segments: [providence, tinyvox]
  adult_segments: [providence_adults, librispeech]
  noise: [musan_noise, musan_music]
  rirs: [rirs_noises, mit_ir_survey]

sampling:
  positive_scene_probability: 0.50
  adult_only_negative_probability: 0.25
  background_speech_negative_probability: 0.15
  noise_only_negative_probability: 0.10
  overlap_probability: 0.25
  short_child_vocalization_probability: 0.30
  short_threshold_sec: 0.5
  other_child_probability: 0.10

turn_taking:
  child_turn_duration_mean_sec: 0.6
  child_turn_duration_std_sec: 0.3
  adult_turn_duration_mean_sec: 3.5
  adult_turn_duration_std_sec: 1.5
  pause_mean_sec: 0.8
  pause_std_sec: 0.3
  n_turns_min: 2
  n_turns_max: 20

mixing:
  snr_db_min: 0
  snr_db_max: 25
  apply_rir_probability: 0.7
  apply_noise_probability: 0.8
  crossfade_ms: 20.0
  peak_normalize: true
  save_stems: false

labels:
  generate_rttm: true
  generate_clip_labels: true
  target_child_label: TARGET_CHILD
```
