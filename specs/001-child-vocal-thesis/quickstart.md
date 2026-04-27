# Quickstart: Child Vocalization Extraction & Synthesis Thesis

**Phase 1 Output** | **Date**: 2026-04-17

This guide walks through running the full experimental pipeline end-to-end. Each section
corresponds to one experimental contribution. All steps assume you are in the repo root:
`/orcd/scratch/orcd/008/manaal/child-adult-diarization/`.

---

## Prerequisites

```bash
# 1. Verify existing baselines still run (sanity check before new experiments)
cd pyannote
python unified.py --diarizer babar        # Should match F1=0.874 from CLAUDE.md
cd ..

# 2. Confirm seen-child splits exist
ls whisper-modeling/seen_child_splits/    # Should show train.csv, val.csv, test.csv

# 3. Confirm RTTM caches are valid (re-run if audio has changed)
ls whisper-modeling/usc_sail_rttm_cache/ | head -5
```

---

## Step 1: Prepare Age-Stratified Data Manifests

Extract age metadata from each labeled dataset and produce split manifests filtered
by age group.

```bash
# Generate per-dataset age-annotated manifests
python scripts/prepare_age_manifests.py \
    --playlogue-csv BIDS_data/anotated_processed.csv \
    --providence-dir providence/ \
    --seedlings-dir seedlings/ \
    --splits-dir whisper-modeling/seen_child_splits/ \
    --output-dir data/age_manifests/

# Verify counts per age group
python scripts/summarize_age_manifests.py --manifest-dir data/age_manifests/
# Expected: ≥500 child segments per age group across train split
```

---

## Step 2: Age-Stratified Enrollment Evaluation

Run enrollment evaluation separately for each age group using all three diarization
frontends.

```bash
cd pyannote

# Run for each diarizer × age group combination
for DIARIZER in usc_sail pyannote babar; do
    for AGE in 12_16m 34_38m all; do
        python unified_age_stratified.py \
            --diarizer $DIARIZER \
            --age-group $AGE \
            --output-dir ${DIARIZER}_age_stratified/
    done
done

# Results written to pyannote/{diarizer}_age_stratified/{age_group}/
```

**Expected outputs per run**: `test_metrics_tuned.json`, `val_metrics_tuned.json`,
`test_predictions.csv`, `test_metrics_by_timepoint.csv`

---

## Step 3: Set Up Synthesis Environment

The synthesis module requires a separate uv-managed environment.

```bash
cd synthesis/
uv sync                     # Installs Coqui TTS, torch, torchaudio per pyproject.toml
uv run python -c "import TTS; print(TTS.__version__)"   # Verify Coqui TTS installed
cd ..
```

---

## Step 4: Extract Child Speech Segments for Synthesis Training

Extract clean child speech segments (no overlap with adult) from labeled datasets
for synthesis model training.

```bash
python synthesis/scripts/extract_segments.py \
    --manifest-dir data/age_manifests/ \
    --output-dir synthesis/data/segments/ \
    --min-duration 0.5 \
    --exclude-overlap true

# Summary: count segments per age group
python synthesis/scripts/count_segments.py --segments-dir synthesis/data/segments/
```

---

## Step 5: Train Synthesis Models

Train separate models for each age group.

```bash
cd synthesis/

# 34-38 month model (VITS-based)
uv run python train.py \
    --config configs/vits_34m.yaml \
    --age-group 34_38m \
    --seed 42

# 12-16 month model (VAE-based)
uv run python train.py \
    --config configs/vae_12m.yaml \
    --age-group 12_16m \
    --seed 42

# Checkpoints saved to synthesis/checkpoints/{age_group}_*/
```

**SLURM option** for long-running training:
```bash
sbatch synthesis/slurm/train_synthesis.sh --age-group 34_38m
```

---

## Step 6: Generate Synthetic Samples

```bash
cd synthesis/

# Generate 1000 samples per age group (adjust n-samples based on training data size)
uv run python generate.py \
    --checkpoint checkpoints/vits_34m_v1/best_checkpoint.pt \
    --age-group 34_38m \
    --n-samples 1000 \
    --seed 42

uv run python generate.py \
    --checkpoint checkpoints/vae_12m_v1/best_checkpoint.pt \
    --age-group 12_16m \
    --n-samples 1000 \
    --seed 42

# Generated samples + registry.jsonl in synthesis/generated/{model_name}/{age_group}/
```

---

## Step 7: Evaluate Synthesis Quality

```bash
cd synthesis/

# Build age-group ECAPA prototypes (run once per model)
cd ../pyannote
python build_age_prototypes.py \
    --manifest-dir ../data/age_manifests/ \
    --output-dir ../synthesis/prototypes/
cd ../synthesis/

# Evaluate each model
for AGE in 12_16m 34_38m; do
    uv run python evaluate.py \
        --generated-dir generated/{model_name}/${AGE}/ \
        --reference-dir data/segments/${AGE}/test/ \
        --age-group $AGE \
        --prototype-path prototypes/${AGE}_prototype.pt \
        --output-path results/synthesis_eval_${AGE}.json
done
```

**Expected**: `results/synthesis_eval_{age_group}.json` with MCD, speaker similarity,
age-group accuracy fields. Commit these files.

---

## Step 8: Augmentation Experiments

```bash
cd pyannote/

for DIARIZER in usc_sail pyannote babar; do
    for AGE in 12_16m 34_38m all; do
        python augmentation_eval.py \
            --diarizer $DIARIZER \
            --synthetic-dir ../synthesis/generated/ \
            --age-group $AGE \
            --aug-ratio 1.0 \
            --output-dir ${DIARIZER}_augmented/
    done
done
```

---

## Step 9: Proxy Analysis on Core Dataset

```bash
cd pyannote/

python proxy_analysis.py \
    --core-dir /path/to/core_dataset_audio/ \
    --prototype-dir ../synthesis/prototypes/ \
    --output-dir core_proxy_analysis/
```

---

## Step 10: Aggregate Thesis Tables

```bash
# Collect all results into thesis-ready tables
python evaluation/aggregate_thesis_tables.py \
    --results-dirs pyannote/ \
    --synthesis-results synthesis/results/ \
    --output-dir evaluation/thesis_tables/

# Output: evaluation/thesis_tables/
#   ├── table1_detection_by_diarizer.csv
#   ├── table2_age_stratified.csv
#   ├── table3_synthesis_quality.csv
#   └── table4_augmentation_delta.csv
```

**Commit all files in `evaluation/thesis_tables/` before writing thesis chapter.**

---

## Reproducibility Verification

To verify all results are reproducible from committed configs:

```bash
python scripts/verify_reproducibility.py \
    --results-dirs pyannote/ synthesis/results/ \
    --check-configs true

# Prints: [PASS] or [FAIL] for each result folder's config match
```
