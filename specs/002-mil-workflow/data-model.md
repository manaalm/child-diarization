# Data Model: Multiple Instance Learning Workflow

**Generated**: 2026-04-23

---

## Entities

### Bag (AudioClip)

One training or evaluation example. A bag is a single audio clip with a known
clip-level binary label.

| Field | Type | Description |
|-------|------|-------------|
| `audio_path` | str | Absolute path to 16kHz mono WAV file |
| `child_id` | str | BIDS subject ID (e.g., `A1H3H9Y3T1`); used for per-child error analysis |
| `timepoint_norm` | str | `"14_month"` or `"36_month"`; for per-timepoint metrics |
| `label` | int | `1` = child present, `0` = child absent |
| `split` | str | `"train"`, `"val"`, or `"test"` |

Source: `whisper-modeling/seen_child_splits/{train,val,test}.csv`

---

### Instance (AudioWindow)

One fixed-length audio window extracted from a Bag. Instances have no individual
label — only the bag label is known.

| Field | Type | Description |
|-------|------|-------------|
| `window_idx` | int | Zero-based index within the parent bag |
| `start_sec` | float | Window start time in seconds |
| `end_sec` | float | Window end time in seconds (= `start_sec` + window size) |
| `waveform` | Tensor `(1, T)` | Raw 16kHz audio samples |

Derived at runtime; not persisted to disk. Window size: 2 s, stride: 1 s.
Clips shorter than 2 s are zero-padded to form one instance.

---

### InstanceFeature

Dense vector representation of a single Instance, produced by a frozen pre-trained
audio encoder. This is what the ABMIL model receives as input.

| Field | Type | Description |
|-------|------|-------------|
| `embedding` | Tensor `(D,)` | Mean-pooled frame-level features; D=768 for both WavLM-base+ and Whisper-small |
| `window_idx` | int | Back-reference to parent Instance |

Produced on-the-fly during forward pass; not cached between training epochs.

---

### AttentionWeight

Scalar importance assigned by the ABMIL head to each InstanceFeature within a Bag.

| Field | Type | Description |
|-------|------|-------------|
| `weight` | float ∈ [0, 1] | Softmax-normalized attention score (sum over bag = 1) |
| `window_idx` | int | Corresponding instance index |

Stored in `test_predictions.csv` (optional top-k attention column) for interpretability.

---

### MILPrediction (clip-level)

The final output for one Bag during evaluation.

| Field | Type | Description |
|-------|------|-------------|
| `audio_path` | str | Source clip path |
| `child_id` | str | Child identifier |
| `timepoint_norm` | str | `"14_month"` or `"36_month"` |
| `label` | int | Ground-truth label |
| `score` | float ∈ [0, 1] | Continuous child presence probability |
| `prediction` | int | Threshold-tuned binary prediction (`0` or `1`) |

Written to `mil/mil_results/{variant}/test_predictions.csv`.

---

### MILRunConfig

All configuration fields for one training run, persisted as `config.json`.

| Field | Type | Description |
|-------|------|-------------|
| `variant_name` | str | e.g., `"wavlm_mil"`, `"whisper_mil"` |
| `backbone` | str | HuggingFace model name (WavLM or Whisper) |
| `backbone_layer` | int | Which encoder layer to extract features from (−1 = final) |
| `window_sec` | float | Instance window length in seconds (default: 2.0) |
| `stride_sec` | float | Window stride in seconds (default: 1.0) |
| `mil_hidden_dim` | int | Hidden dimension of ABMIL attention MLP (default: 256) |
| `mil_dropout` | float | Dropout rate applied to instance embeddings (default: 0.25) |
| `lr` | float | Learning rate for MIL head (default: 1e-3) |
| `epochs` | int | Max training epochs (default: 20) |
| `patience` | int | Early stopping patience on val F1 (default: 5) |
| `batch_size` | int | Bags per batch (default: 8) |
| `pos_weight` | float or null | BCE positive class weight; null = unweighted |
| `seed` | int | Random seed (always 42) |
| `split_dir` | str | Path to `seen_child_splits/` directory |
| `device` | str | `"cuda"` or `"cpu"` |

---

### MILResultFolder

Directory structure for one completed training+evaluation run.

```
mil/mil_results/{variant_name}/
├── config.json                    # MILRunConfig, serialized
├── training_history.csv           # epoch, train_loss, val_loss, val_f1 per epoch
├── best_checkpoint.pt             # Best model weights (by val F1)
├── val_metrics_tuned.json         # F1, precision, recall, AUROC, AUPRC + threshold
├── test_metrics_tuned.json        # Same fields, test set
├── val_predictions.csv            # MILPrediction rows for val set
├── test_predictions.csv           # MILPrediction rows for test set
├── val_metrics_by_timepoint.csv   # Metrics grouped by timepoint_norm (val)
└── test_metrics_by_timepoint.csv  # Metrics grouped by timepoint_norm (test)
```

**File schemas** (matching unified.py output for thesis table compatibility):

`*_metrics_tuned.json`:
```json
{"f1": 0.0, "precision": 0.0, "recall": 0.0, "auroc": 0.0, "auprc": 0.0,
 "threshold": 0.5, "val_f1_at_threshold": 0.0}
```

`*_predictions.csv`: columns `audio_path, child_id, timepoint_norm, label, score, prediction`

`*_metrics_by_timepoint.csv`: columns `timepoint, f1, precision, recall, auroc, auprc, n`

`training_history.csv`: columns `epoch, train_loss, val_loss, val_f1, val_auroc`

---

## State Transitions

```
AudioClip (CSV row)
    → windowed → [AudioWindow, ...]
    → encoded  → [InstanceFeature, ...]
    → attended → weighted bag embedding (1 × D)
    → classified → MILPrediction.score ∈ [0, 1]
    → thresholded → MILPrediction.prediction ∈ {0, 1}
```

---

## Validation Rules

- `window_sec` MUST be ≤ audio clip duration for at least 1 instance; shorter clips
  are padded, not skipped.
- `stride_sec` MUST be < `window_sec` (positive overlap required).
- `seed` MUST be `42` per Constitution Principle I.
- `split_dir` MUST point to `whisper-modeling/seen_child_splits/`; cross-child splits
  (`baselines/splits/`) MUST NOT be used for MIL training or evaluation.
- `test_predictions.csv` MUST contain exactly one row per clip in the test split.
- `config.json` MUST be committed before any result files derived from that config.
