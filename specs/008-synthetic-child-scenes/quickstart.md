# Quickstart: Synthetic Child-Adult Scene Generator

**Date**: 2026-04-24 | Reproduces the MVP pipeline end-to-end from raw data to evaluation.

---

## Prerequisites

1. **Conda environment**: `conda activate child-vocalizations` (same env as `av_fusion/` and `mil/`)
2. **Data**: Populate `data/` (gitignore'd):
   - `data/segments/` — populated by Step 2 (extract_segments)
   - `data/noise/` — download MUSAN: `wget https://openslr.org/resources/17/musan.tar.gz && tar -xf musan.tar.gz -C data/noise/`
   - `data/rirs/` — download RIRS_NOISES: `wget https://openslr.org/resources/28/rirs_noises.zip && unzip rirs_noises.zip -d data/rirs/`
3. **Providence**: Already present at `providence/` in repo root.
4. **LibriSpeech**: download `train-clean-100` from `https://openslr.org/resources/12/` to a local path; pass via `--librispeech-dir`.

---

## Step 1 — Build Segment Manifest

```bash
conda activate child-vocalizations
cd /orcd/scratch/orcd/008/manaal/child-adult-diarization

python synth/scripts/build_segment_manifest.py \
  --providence-dir        providence/ \
  --providence-rttm-dir   providence/rttm/ \
  --librispeech-dir       /path/to/LibriSpeech/train-clean-100/ \
  --exclude-speakers-csv  whisper-modeling/seen_child_splits/test.csv \
  --output                synth_results/manifests/segment_manifest.csv \
  --min-duration-sec      0.3 \
  --quality-threshold     0.4
# Output: synth_results/manifests/segment_manifest.csv
# Prints: per-dataset counts, split integrity report
```

---

## Step 2 — Extract Segments

```bash
python synth/scripts/extract_segments.py \
  --manifest  synth_results/manifests/segment_manifest.csv \
  --output-dir data/segments/ \
  --sample-rate 16000
# Output: data/segments/child/*.wav, data/segments/adult/*.wav
# Updates audio_path column in manifest to point to extracted WAVs
```

---

## Step 3 — Generate Synthetic Scenes

```bash
# Small smoke test (50 scenes, both age bands):
python synth/scripts/generate_scenes.py \
  --config  synth/configs/default_14_18mo.yaml \
  --manifest synth_results/manifests/segment_manifest.csv \
  --n-scenes 50 \
  --output-dir synth_results/synthetic_scenes/

# Full run (5000 scenes, via SLURM):
sbatch synth/slurm/run_scene_generation.sh \
  --config synth/configs/default_14_18mo.yaml
# Output: synth_results/synthetic_scenes/{wav,rttm,json}/
#         synth_results/manifests/synthetic_manifest.csv
```

**Verify outputs**:
```bash
# Check 10 random scenes have matching WAV + RTTM + JSON
python -c "
import pandas as pd, os
df = pd.read_csv('synth_results/manifests/synthetic_manifest.csv').sample(10, random_state=1)
for _, row in df.iterrows():
    assert os.path.exists(row.audio_path), row.audio_path
    assert os.path.exists(row.rttm_path), row.rttm_path
    # verify label consistency
    has_child = any('TARGET_CHILD' in l for l in open(row.rttm_path))
    assert has_child == bool(row.target_child_vocalized), row.synthetic_scene_id
print('All 10 spot-checks passed.')
"
```

---

## Step 4 — Generate Training Manifests

```bash
python synth/scripts/generate_training_sets.py \
  --real-train-csv        whisper-modeling/seen_child_splits/train.csv \
  --synthetic-manifest    synth_results/manifests/synthetic_manifest.csv \
  --ratios                0 0.5 1 2 5 10 \
  --output-dir            synth_results/manifests/ \
  --seed                  42
# Output: synth_results/manifests/train_{0,0.5,1,2,5,10}x_manifest.csv
```

---

## Step 5 — Train at Each Ratio

```bash
# Runs enrollment pipeline (BabAR) with augmented training manifests:
python synth/scripts/train_with_synthetic.py \
  --manifest-dir  synth_results/manifests/ \
  --ratios        0 0.5 1 2 5 10 \
  --output-dir    synth_results/augmentation_experiments/default_14_18mo/

# Or submit SLURM sweep (trains all 6 ratios sequentially on GPU node):
sbatch synth/slurm/run_ratio_sweep.sh \
  --config synth/configs/default_14_18mo.yaml
```

---

## Step 6 — Evaluate on Held-Out Real Test Set

```bash
python synth/scripts/evaluate_synthetic_augmentation.py \
  --experiment-dir  synth_results/augmentation_experiments/default_14_18mo/ \
  --test-csv        whisper-modeling/seen_child_splits/test.csv \
  --output-dir      synth_results/augmentation_experiments/default_14_18mo/ \
  --plot
# Output: metrics_by_ratio.csv, metrics_by_age_band.csv
#         figures/synthetic_ratio_vs_auprc.png
#         figures/synthetic_ratio_vs_der.png
```

---

## Step 7 — Error Analysis

```bash
python synth/scripts/error_analysis_synthetic.py \
  --experiment-dir  synth_results/augmentation_experiments/default_14_18mo/ \
  --test-csv        whisper-modeling/seen_child_splits/test.csv \
  --output-dir      synth_results/augmentation_experiments/default_14_18mo/
# Output: error_analysis.csv with per-failure-mode breakdown
```

---

## Optional: Synthetic Quality Analysis

```bash
python synth/scripts/analyze_synthetic_quality.py \
  --synthetic-manifest synth_results/manifests/synthetic_manifest.csv \
  --real-train-csv     whisper-modeling/seen_child_splits/train.csv \
  --output-dir         synth_results/augmentation_experiments/default_14_18mo/figures/
# Output: duration_distribution.png, snr_distribution.png,
#         real_vs_synthetic_embedding_umap.png
```

---

## Cache Invalidation

If any of the following change, **delete and regenerate** the affected outputs:

| What changed | Delete |
|---|---|
| Source audio files or RTTM annotations | `data/segments/`, then re-run Steps 1–3 |
| `segment_manifest.csv` | Re-run Steps 3–7 |
| Scene config YAML | Re-run Steps 3–7 (new config name creates new output dir) |
| Random seed | Re-run Steps 3–7 |
| Real train/test CSV | Re-run Steps 4–7 |

**Never** partially regenerate scenes: always regenerate the full scene set for a given config to preserve reproducibility.

---

## Gotchas

- `build_segment_manifest.py` must receive `--exclude-speakers-csv` pointing to the real test CSV, or Constitution Principle II will be violated.
- Scene WAVs are not committed (they live in `data/` or `synth_results/synthetic_scenes/wav/`, which is gitignore'd). Only the manifests and metrics are committed.
- The `generate_scenes.py` script is CPU-only; running it on a GPU node wastes allocation.
- Verify `conda activate child-vocalizations` before running any script in this module.
