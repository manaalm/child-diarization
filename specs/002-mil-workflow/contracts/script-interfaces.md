# Script Interface Contracts: Multiple Instance Learning Workflow

**Generated**: 2026-04-23

All scripts live under `mil/` relative to the repo root.

---

## mil_train.py

Train an ABMIL model on the seen-child split.

### CLI

```
python mil/mil_train.py --config mil/configs/wavlm_mil.yaml
python mil/mil_train.py --config mil/configs/whisper_mil.yaml
```

### Arguments

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--config` | Yes | — | Path to YAML config file |

### Config File Schema (YAML)

```yaml
variant_name: "wavlm_mil"           # used as results subdirectory name
backbone: "microsoft/wavlm-base-plus"
backbone_layer: -1                  # -1 = final encoder layer
window_sec: 2.0
stride_sec: 1.0
mil_hidden_dim: 256
mil_dropout: 0.25
lr: 1.0e-3
epochs: 20
patience: 5
batch_size: 8
pos_weight: null                    # float or null
seed: 42
split_dir: "whisper-modeling/seen_child_splits"
device: "cuda"
```

### Behavior

1. Load `split_dir/train.csv` and `split_dir/val.csv`; filter `audio_exists == True`.
2. For each clip, extract windowed InstanceFeatures on-the-fly (no disk cache).
3. Train ABMIL head for `epochs` epochs; save checkpoint whenever val F1 improves.
4. Apply early stopping after `patience` epochs without val F1 improvement.
5. Sweep threshold 0.05–0.95 on val set; pick threshold maximizing val F1.
6. Write result files to `mil/mil_results/{variant_name}/`.
7. Print per-epoch train loss, val loss, val F1 to stdout.

### Outputs

```
mil/mil_results/{variant_name}/
├── config.json              # copy of resolved config
├── training_history.csv     # epoch, train_loss, val_loss, val_f1, val_auroc
├── best_checkpoint.pt       # best weights by val F1
├── val_metrics_tuned.json   # metrics at tuned threshold
└── val_predictions.csv      # score + prediction per val clip
```

### Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Training completed; results written |
| 1 | Config file not found or invalid |
| 2 | Split CSV missing or no audio_exists rows |
| 3 | GPU not available when device=cuda |

---

## mil_evaluate.py

Evaluate a trained checkpoint on the test split and write final result files.

### CLI

```
python mil/mil_evaluate.py \
    --checkpoint mil/mil_results/wavlm_mil/best_checkpoint.pt \
    --config     mil/mil_results/wavlm_mil/config.json
```

### Arguments

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--checkpoint` | Yes | — | Path to `best_checkpoint.pt` |
| `--config` | Yes | — | Path to `config.json` from the training run |

### Behavior

1. Load config from `--config`; load weights from `--checkpoint`.
2. Load `split_dir/test.csv` (filter `audio_exists == True`).
3. Produce clip-level `score` and `prediction` for every test clip using the
   `threshold` stored in `val_metrics_tuned.json` (same threshold used at train time).
4. Compute F1, precision, recall, AUROC, AUPRC on test set.
5. Compute per-timepoint metrics grouped by `timepoint_norm`.
6. Write outputs to the same `mil/mil_results/{variant_name}/` directory.

### Outputs (added to existing result folder)

```
mil/mil_results/{variant_name}/
├── test_metrics_tuned.json           # F1, precision, recall, AUROC, AUPRC, threshold
├── test_predictions.csv              # audio_path, child_id, timepoint_norm, label, score, prediction
├── test_metrics_by_timepoint.csv     # timepoint, f1, precision, recall, auroc, auprc, n
└── val_metrics_by_timepoint.csv      # same for val set (produced if not already present)
```

### Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Evaluation completed |
| 1 | Checkpoint or config not found |
| 2 | Test split CSV missing |

---

## mil_age_stratified.py

Evaluate a trained checkpoint restricted to one age cohort.

### CLI

```
python mil/mil_age_stratified.py \
    --checkpoint mil/mil_results/wavlm_mil/best_checkpoint.pt \
    --config     mil/mil_results/wavlm_mil/config.json \
    --age-group  12_16m \
    --manifest   playlogue/manifest.csv
```

### Arguments

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--checkpoint` | Yes | — | Trained model weights |
| `--config` | Yes | — | Training config JSON |
| `--age-group` | Yes | — | `12_16m` or `34_38m` |
| `--manifest` | Yes | — | Age-annotated manifest CSV (from `scripts/prepare_age_manifests.py`) |

### Behavior

1. Load config and checkpoint.
2. Load `split_dir/test.csv`; inner-join with `--manifest` on `audio_path`; filter
   to rows where `age_group == --age-group` and `split == "test"`.
3. Produce predictions and metrics for the filtered subset.
4. Write outputs to `mil/mil_results/{variant_name}/age_stratified/{age_group}/`.

### Outputs

```
mil/mil_results/{variant_name}/age_stratified/{age_group}/
├── test_metrics_tuned.json
├── test_predictions.csv
└── test_metrics_by_timepoint.csv
```

---

## Config Files

### mil/configs/wavlm_mil.yaml

```yaml
variant_name: "wavlm_mil"
backbone: "microsoft/wavlm-base-plus"
backbone_layer: -1
window_sec: 2.0
stride_sec: 1.0
mil_hidden_dim: 256
mil_dropout: 0.25
lr: 1.0e-3
epochs: 20
patience: 5
batch_size: 8
pos_weight: null
seed: 42
split_dir: "whisper-modeling/seen_child_splits"
device: "cuda"
```

### mil/configs/whisper_mil.yaml

```yaml
variant_name: "whisper_mil"
backbone: "openai/whisper-small"
backbone_layer: -1
window_sec: 2.0
stride_sec: 1.0
mil_hidden_dim: 256
mil_dropout: 0.25
lr: 1.0e-3
epochs: 20
patience: 5
batch_size: 8
pos_weight: null
seed: 42
split_dir: "whisper-modeling/seen_child_splits"
device: "cuda"
```

---

## SLURM Script: mil/slurm/train_mil.sh

```bash
#!/bin/bash
#SBATCH -c 4
#SBATCH -t 8:00:00
#SBATCH -p ou_bcs_normal,pi_satra
#SBATCH --mem=40G
#SBATCH --gres=gpu:1
#SBATCH -o logs/mil/train_%j.out
#SBATCH -e logs/mil/train_%j.err
```

Called as:
```bash
sbatch mil/slurm/train_mil.sh mil/configs/wavlm_mil.yaml
sbatch mil/slurm/train_mil.sh mil/configs/whisper_mil.yaml
```

---

## Integration with Thesis Tables

`evaluation/configs/thesis_tables.yaml` must be updated to include MIL result paths:

```yaml
# MIL enrollment results (seen-child split)
mil_wavlm:
  test_metrics: mil/mil_results/wavlm_mil/test_metrics_tuned.json
  test_by_timepoint: mil/mil_results/wavlm_mil/test_metrics_by_timepoint.csv
  predictions: mil/mil_results/wavlm_mil/test_predictions.csv

mil_whisper:
  test_metrics: mil/mil_results/whisper_mil/test_metrics_tuned.json
  test_by_timepoint: mil/mil_results/whisper_mil/test_metrics_by_timepoint.csv
  predictions: mil/mil_results/whisper_mil/test_predictions.csv
```
