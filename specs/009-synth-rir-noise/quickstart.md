# Quickstart: Acoustic Scene Generation (009-synth-rir-noise)

**Prerequisites**: `child-vocalizations` conda env; Providence audio; TinyVox at
`data/tinyvox/`; SLURM access on `ou_bcs_normal` partition.

---

## Step 0 — Stage RIR and Noise Data (one-time, user action required)

```bash
# Option A: OpenSLR 26 RIRs (~1 GB)
wget https://www.openslr.org/resources/26/sim_rir_16k.zip -P /path/to/data/
unzip /path/to/data/sim_rir_16k.zip -d /path/to/data/rir_files/

# Option B: MUSAN noise + music (~17 GB total; noise subset alone ~1 GB)
wget https://www.openslr.org/resources/17/musan.tar.gz -P /path/to/data/
tar -xzf /path/to/data/musan.tar.gz -C /path/to/data/

# Note the paths — you will need them in Steps 2–3 below.
RIR_DIR=/path/to/data/rir_files/
NOISE_DIR=/path/to/data/musan/noise/
```

If neither dataset is available, the generator falls back to clean mix (FR-005).

---

## Step 1 — Rebuild Segment Manifest (includes TinyVox)

```bash
conda activate /orcd/home/002/manaal/miniforge3/envs/child-vocalizations
cd /orcd/scratch/orcd/008/manaal/child-adult-diarization

python synth/scripts/build_segment_manifest.py \
  --providence-dir        providence/ \
  --providence-rttm-dir   providence/rttm/ \
  --tinyvox-dir           data/tinyvox/ \
  --librispeech-dir       /path/to/LibriSpeech/train-clean-100/ \
  --exclude-speakers-csv  whisper-modeling/seen_child_splits/test.csv \
  --output                synth_results/manifests/segment_manifest.csv \
  --skip-quality
```

Expected output: ~24k TinyVox Eng-NA child segments + Providence + LibriSpeech adult.

---

## Step 2 — Generate Acoustic Scenes (SLURM)

```bash
sbatch synth/slurm/run_scene_generation.sh \
  synth/configs/default_14_18mo.yaml \
  --rir-dir  "$RIR_DIR" \
  --noise-dir "$NOISE_DIR"
# Output: synth_results/synthetic_scenes/{wav,rttm,json}/ (gitignored)
#         synth_results/manifests/synthetic_manifest.csv (committed)
```

To test graceful fallback (no RIR/noise):
```bash
python synth/scripts/generate_scenes.py \
  --config synth/configs/default_14_18mo.yaml \
  --manifest synth_results/manifests/segment_manifest.csv \
  --output-dir synth_results/synthetic_scenes/ \
  --n-scenes 10
# Should produce 10 clean-mix scenes; rir_id=null, noise_id=null in JSON
```

---

## Step 3 — Verify Acoustic Realism (SC-001, SC-002)

```bash
# Spot-check 20 output JSONs for non-null rir_id
python -c "
import json, glob, random
jsons = glob.glob('synth_results/synthetic_scenes/json/*.json')
sample = random.sample(jsons, min(20, len(jsons)))
n_rir = sum(1 for p in sample if json.load(open(p)).get('rir_id') is not None)
n_noise = sum(1 for p in sample if json.load(open(p)).get('noise_id') is not None)
print(f'RIR applied: {n_rir}/20 (expect ~14)  Noise applied: {n_noise}/20 (expect ~16)')
"

# Verify low_snr_stress scenes have SNR in [0,5]
python -c "
import json, glob
jsons = glob.glob('synth_results/synthetic_scenes/low_snr_stress/json/*.json')
snrs = [json.load(open(p)).get('mean_snr_db') for p in jsons if json.load(open(p)).get('mean_snr_db') is not None]
print(f'SNR range: {min(snrs):.2f} – {max(snrs):.2f} dB  (expect 0.0 – 5.0)')
"
```

---

## Step 4 — Re-run Ratio Sweep with Acoustic Scenes

```bash
sbatch synth/slurm/run_ratio_sweep.sh synth/configs/default_14_18mo.yaml
# Output: synth_results/augmentation_experiments/default_14_18mo/
#         metrics_by_ratio.csv, metrics_by_age_band.csv, figures/
```

---

## Step 5 — Compare vs. Clean-Mix Baseline

```bash
conda activate /orcd/home/002/manaal/miniforge3/envs/child-vocalizations

python synth/scripts/evaluate_synthetic_augmentation.py \
  --experiment-dir synth_results/augmentation_experiments/default_14_18mo/ \
  --test-csv       whisper-modeling/seen_child_splits/test.csv \
  --output-dir     synth_results/augmentation_experiments/default_14_18mo/ \
  --plot

python synth/scripts/error_analysis_synthetic.py \
  --experiment-dir synth_results/augmentation_experiments/default_14_18mo/ \
  --test-csv       whisper-modeling/seen_child_splits/test.csv \
  --output-dir     synth_results/augmentation_experiments/default_14_18mo/
```

SC-003 passes if `max(metrics_by_ratio.auprc) >= baseline_0x_auprc + 0.005`.

---

## Stretch: Child-Adapted WavLM Pretraining

```bash
# Collect child speech file list
find data/tinyvox/audio -name "phon_Eng-NA_*.wav" > /tmp/child_wavs.txt
find data/segments/child -name "*.wav" >> /tmp/child_wavs.txt
wc -l /tmp/child_wavs.txt  # expect > 20k files; report data-hours at startup

sbatch synth/slurm/run_wavlm_pretrain.sh \
  --wav-list   /tmp/child_wavs.txt \
  --base-model microsoft/wavlm-base-plus \
  --output-dir synth_results/child_wavlm_checkpoint/ \
  --max-steps  50000
# Logs: logs/synth/wavlm_pretrain_{jobid}.out
```

After pretraining, test as MIL backbone:
```bash
sbatch mil/slurm/train_mil.sh mil/configs/wavlm_mil_child_adapted.yaml
# mil/configs/wavlm_mil_child_adapted.yaml: identical to wavlm_mil.yaml
# except backbone_path: synth_results/child_wavlm_checkpoint/
```

SC-005 passes if child-adapted MIL test AUPRC ≥ 0.946 (Whisper-MIL baseline).
