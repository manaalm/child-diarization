# Script Interface Contracts

**Phase 1 Output** | **Date**: 2026-04-17 | **Feature**: 001-child-vocal-thesis

These contracts define the CLI interfaces for all new scripts introduced in this
feature. Existing scripts (`unified.py`, `unified_rttm.py`, etc.) are not modified
here — only new scripts are specified.

---

## synthesis/train.py

Trains a synthesis model for a given age group.

```
python synthesis/train.py [OPTIONS]

Required:
  --config PATH         Path to synthesis training YAML config
  --age-group STR       Age group to train for: "12_16m" or "34_38m"

Optional:
  --output-dir PATH     Output dir for checkpoints (default: synthesis/checkpoints/)
  --seed INT            Random seed (default: 42)
  --resume PATH         Path to checkpoint to resume from

Outputs:
  synthesis/checkpoints/{age_group}_{model_name}_{timestamp}/
    ├── best_checkpoint.pt
    ├── config.json        # Copy of config + resolved paths
    └── training_log.csv
```

**Exit codes**: 0 = success, 1 = config error, 2 = data error, 3 = training failure.

---

## synthesis/generate.py

Generates synthetic child speech samples from a trained model.

```
python synthesis/generate.py [OPTIONS]

Required:
  --checkpoint PATH     Path to trained model checkpoint
  --age-group STR       Age group to generate for: "12_16m" or "34_38m"
  --n-samples INT       Number of samples to generate

Optional:
  --output-dir PATH     Output dir for generated WAVs (default: synthesis/generated/)
  --seed INT            Random seed (default: 42)
  --duration-range STR  Min,max duration in seconds (default: "1.0,5.0")
  --prototype-path PATH ECAPA prototype embedding for speaker conditioning

Outputs:
  synthesis/generated/{model_name}/{age_group}/
    ├── sample_{000001}.wav
    ├── ...
    └── registry.jsonl    # One JSON record per sample (SyntheticSpeechSample schema)
```

**Exit codes**: 0 = success, 1 = checkpoint error, 2 = output error.

---

## synthesis/evaluate.py

Evaluates synthesis quality for a set of generated samples.

```
python synthesis/evaluate.py [OPTIONS]

Required:
  --generated-dir PATH  Directory of generated WAV files
  --reference-dir PATH  Directory of held-out real child speech WAVs
  --age-group STR       Age group being evaluated: "12_16m" or "34_38m"

Optional:
  --prototype-path PATH ECAPA age-group prototype for speaker similarity
  --age-classifier PATH Path to trained age-group classifier checkpoint
  --output-path PATH    Output JSON with all metric scores (default: eval_results.json)

Outputs (written to --output-path):
  {
    "mcd_mean": float,
    "mcd_std": float,
    "speaker_similarity_mean": float,
    "age_classifier_accuracy": float,
    "f0_stats": { "mean": float, "std": float, "median": float }
  }
```

**Exit codes**: 0 = success, 1 = input error, 2 = metric computation failure.

---

## pyannote/unified_age_stratified.py

Runs enrollment evaluation stratified by age group, extending unified.py.

```
python pyannote/unified_age_stratified.py [OPTIONS]

Required:
  --diarizer STR        Diarizer frontend: usc_sail | pyannote | babar

Optional:
  --age-group STR       Filter to age group: all | 12_16m | 34_38m (default: all)
  --output-dir PATH     Results output dir (default: pyannote/{diarizer}_age_stratified/)
  --splits-dir PATH     Path to seen_child_splits dir
  --seed INT            Random seed (default: 42)

Outputs (per age_group):
  {output_dir}/{age_group}/
    ├── config.json
    ├── test_metrics_tuned.json      # F1, precision, recall, AUROC, AUPRC
    ├── test_metrics_by_timepoint.csv
    ├── test_predictions.csv
    └── val_metrics_tuned.json
```

---

## pyannote/augmentation_eval.py

Trains and evaluates detection with synthetic data augmentation.

```
python pyannote/augmentation_eval.py [OPTIONS]

Required:
  --diarizer STR        Diarizer frontend: usc_sail | pyannote | babar
  --synthetic-dir PATH  Directory of synthetic WAVs + registry.jsonl

Optional:
  --age-group STR       Age group to augment: all | 12_16m | 34_38m (default: all)
  --aug-ratio FLOAT     Synthetic-to-real ratio (default: 1.0)
  --output-dir PATH     Results output dir (default: pyannote/{diarizer}_augmented/)
  --seed INT            Random seed (default: 42)

Outputs (same structure as unified.py):
  {output_dir}/{age_group}_ratio{aug_ratio}/
    ├── config.json
    ├── test_metrics_tuned.json
    ├── test_predictions.csv
    └── val_metrics_tuned.json
```

---

## pyannote/proxy_analysis.py

Computes proxy quality metrics on unlabeled core dataset recordings.

```
python pyannote/proxy_analysis.py [OPTIONS]

Required:
  --core-dir PATH       Directory of core dataset WAV files

Optional:
  --prototype-dir PATH  Directory of age-group ECAPA prototypes (.pt files)
  --output-dir PATH     Output dir (default: pyannote/core_proxy_analysis/)
  --age-group STR       Expected age group for each core session file (via manifest)

Outputs:
  {output_dir}/
    ├── config.json
    ├── per_session_scores.csv    # cosine similarity per session
    ├── inter_frontend_agreement.csv
    └── detection_rate_stats.csv
```

---

## Config File Schema (synthesis/configs/*.yaml)

```yaml
# Synthesis training config (common fields)
model:
  type: vits | vae              # Architecture selection
  age_group: "12_16m" | "34_38m"
  n_speakers: 1                 # 1 = age-group-conditioned, >1 = per-child

data:
  train_manifest: path/to/train_manifest.csv
  val_manifest: path/to/val_manifest.csv
  sample_rate: 16000
  max_duration_secs: 5.0
  min_duration_secs: 0.5

training:
  seed: 42
  batch_size: 32
  max_epochs: 100
  learning_rate: 0.0002
  early_stopping_patience: 10

output:
  checkpoint_dir: synthesis/checkpoints/
  log_interval: 100
```
