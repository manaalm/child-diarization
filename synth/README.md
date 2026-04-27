# Synthetic Child-Adult Scene Generator

Generates synthetic 30-second audio scenes by mixing real child and adult speech segments under configurable SNR, RIR, overlap, and scene-type distributions. Used to augment training data for the BabAR enrollment pipeline.

---

## Prerequisites

**Conda environment** (same as `av_fusion/` and `mil/`):
```bash
conda activate child-vocalizations
# Required packages: numpy, pandas, scipy, soundfile, pyyaml, tqdm, librosa (optional)
```

**Data** (all gitignore'd — not committed):

| Path | Contents | How to get |
|---|---|---|
| `data/segments/child/` | Extracted Providence CHI segments (WAV) | Step 2 below |
| `data/segments/adult/` | Extracted LibriSpeech segments (WAV) | Step 2 below |
| `data/noise/musan/` | MUSAN noise + music (WAV) | `wget https://openslr.org/resources/17/musan.tar.gz && tar -xf musan.tar.gz -C data/noise/` |
| `data/rirs/RIRS_NOISES/` | Room impulse responses | `wget https://openslr.org/resources/28/rirs_noises.zip && unzip rirs_noises.zip -d data/rirs/` |

**Existing data already in repo**:
- Providence RTTM files: `providence/rttm/`
- Real train/val/test splits: `whisper-modeling/seen_child_splits/`

---

## Step-by-Step Quickstart

### Step 1 — Build Segment Manifest

Scans Providence RTTMs and LibriSpeech audio to build a manifest of usable segments.

```bash
python synth/scripts/build_segment_manifest.py \
  --providence-dir        providence/ \
  --providence-rttm-dir   providence/rttm/ \
  --librispeech-dir       /path/to/LibriSpeech/train-clean-100/ \
  --exclude-speakers-csv  whisper-modeling/seen_child_splits/test.csv \
  --output                synth_results/manifests/segment_manifest.csv \
  --min-duration-sec      0.3 \
  --quality-threshold     0.4
```

`--exclude-speakers-csv` must point to the real **test** split. Any Providence child who appears in the test split is marked `usable_for_training=false` and `split=test` to prevent data leakage.

### Step 2 — Extract Segments

Copies segment audio to `data/segments/` and updates `audio_path` in the manifest.

```bash
python synth/scripts/extract_segments.py \
  --manifest    synth_results/manifests/segment_manifest.csv \
  --output-dir  data/segments/ \
  --sample-rate 16000
```

Idempotent — already-extracted files are skipped.

### Step 3 — Generate Synthetic Scenes

```bash
# Smoke test (50 scenes):
python synth/scripts/generate_scenes.py \
  --config     synth/configs/default_14_18mo.yaml \
  --manifest   synth_results/manifests/segment_manifest.csv \
  --n-scenes   50 \
  --output-dir synth_results/synthetic_scenes/

# Full run (5 000 scenes via SLURM — no GPU needed):
sbatch synth/slurm/run_scene_generation.sh synth/configs/default_14_18mo.yaml
```

Output: `synth_results/synthetic_scenes/{wav,rttm,json}/` and
`synth_results/manifests/synthetic_manifest.csv`.

Idempotent — existing WAV/JSON files are skipped; the manifest is appended.

### Step 4 — Generate Training Manifests

```bash
python synth/scripts/generate_training_sets.py \
  --real-train-csv     whisper-modeling/seen_child_splits/train.csv \
  --synthetic-manifest synth_results/manifests/synthetic_manifest.csv \
  --ratios             0 0.5 1 2 5 10 \
  --output-dir         synth_results/manifests/ \
  --seed               42
```

Output: `synth_results/manifests/train_{0,0.5,1,2,5,10}x_manifest.csv`.

### Step 5 — Train at Each Ratio

```bash
python synth/scripts/train_with_synthetic.py \
  --manifest-dir synth_results/manifests/ \
  --ratios       0 0.5 1 2 5 10 \
  --output-dir   synth_results/augmentation_experiments/default_14_18mo/

# Or submit the SLURM sweep (GPU node, 48 h):
sbatch synth/slurm/run_ratio_sweep.sh synth/configs/default_14_18mo.yaml
```

### Step 6 — Evaluate on Real Test Set

```bash
python synth/scripts/evaluate_synthetic_augmentation.py \
  --experiment-dir synth_results/augmentation_experiments/default_14_18mo/ \
  --test-csv       whisper-modeling/seen_child_splits/test.csv \
  --output-dir     synth_results/augmentation_experiments/default_14_18mo/ \
  --plot
```

Output: `metrics_by_ratio.csv`, `metrics_by_age_band.csv`,
`figures/synthetic_ratio_vs_auprc.png`, `figures/synthetic_ratio_vs_f1.png`.

### Step 7 — Error Analysis

```bash
python synth/scripts/error_analysis_synthetic.py \
  --experiment-dir synth_results/augmentation_experiments/default_14_18mo/ \
  --test-csv       whisper-modeling/seen_child_splits/test.csv \
  --output-dir     synth_results/augmentation_experiments/default_14_18mo/
```

Output: `error_analysis.csv`, `error_counts.json`, `error_by_age_band.csv`.

### Optional — Synthetic Quality Figures

```bash
python synth/scripts/analyze_synthetic_quality.py \
  --synthetic-manifest synth_results/manifests/synthetic_manifest.csv \
  --real-train-csv     whisper-modeling/seen_child_splits/train.csv \
  --output-dir         synth_results/augmentation_experiments/default_14_18mo/figures/
```

---

## Config Files

| Config | Purpose |
|---|---|
| `synth/configs/default_14_18mo.yaml` | Standard 30 s scenes, 14–18 month age band, 5 000 scenes |
| `synth/configs/default_34_38mo.yaml` | Same as above but 34–38 month age band, longer turns |
| `synth/configs/hard_negatives.yaml` | 100 % negative scenes (adult-only + background + noise), 2 000 scenes |
| `synth/configs/overlap_stress.yaml` | High overlap rate (90 %), 2 000 scenes |
| `synth/configs/low_snr_stress.yaml` | Low SNR (−5 to +5 dB), 2 000 scenes |

---

## Cache Invalidation

| What changed | Action |
|---|---|
| Source audio or RTTM annotations | Delete `data/segments/`, re-run Steps 1–3 |
| `segment_manifest.csv` | Re-run Steps 3–7 |
| Scene config YAML | Re-run Steps 3–7 (config name creates a new output subdir) |
| Random seed | Re-run Steps 3–7 |
| Real train / test CSV | Re-run Steps 4–7 |

Never partially regenerate a scene set. Always regenerate all N scenes for a given config to preserve reproducibility.

---

## Gotchas

- **`--exclude-speakers-csv` is mandatory** in `build_segment_manifest.py`. Omitting it will include test-split children in training segments, violating the data-split contract.
- **No GPU needed** for scene generation (`generate_scenes.py`). Do not waste a GPU allocation on Steps 1–3.
- **Scene WAVs are gitignore'd** — only manifests, configs, metrics, and scripts are committed.
- **Always use `conda activate child-vocalizations`** before running any script in this module.
- **Idempotency**: `generate_scenes.py` skips existing WAV files. If you need to regenerate (e.g., after changing a config), delete the output directory first.
- **Providence age encoding**: filenames like `alex_010427.rttm` encode the child's age as YYMMDD (1 year 4 months 27 days), not a calendar date. `build_segment_manifest.py` handles this correctly via `_parse_age_months`.
- **LibriSpeech format**: `build_segment_manifest.py` scans `.flac` files recursively. Ensure `train-clean-100` is fully extracted before running.
