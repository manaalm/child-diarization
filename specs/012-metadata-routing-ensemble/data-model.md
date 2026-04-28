# Data Model: Metadata-Conditioned Routing and Ensemble Extensions

## Core Entities

### ClipRecord
A single audio clip with ground-truth label and metadata.

| Field | Source | Type | Notes |
|---|---|---|---|
| `audio_path` | `master_with_split.csv` | str | Join key across all files |
| `label` | `master_with_split.csv` | int (0/1) | Ground truth |
| `split` | `master_with_split.csv` | str | train/val/test |
| `timepoint_norm` | `master_with_split.csv` | str | 14_month / 36_month |
| `n_adults` | `#_adults` col | int | Cast from string; NaN → 0 |
| `n_children` | `#_children` col | int | Cast from string; NaN → 1 |
| `context` | `Context` col | str | 9 categories; lowercase strip |
| `has_interaction` | `Interaction_with_child` col | int (0/1) | yes→1, else→0 |
| `location` | `Location` col | str | 6 categories |
| `face_visibility` | `Video_Quality_Child_Face_Visibility` | float | 1–10 scale; NaN → 5.0 |

### SystemScore
Per-system per-clip predicted probability. 10 systems; joined on `audio_path`.

| Field | Type | Notes |
|---|---|---|
| `audio_path` | str | Join key |
| `{system}_prob` | float [0,1] | Normalized to [0,1]; MIL "score" renamed to "prob" |
| Missing clips | float | Imputed with 0.5 (neutral prior) for audio_llm missing ~16 clips |

Score column normalization: all sources produce sigmoid-transformed [0,1] scores. No further normalization required.

### RouterOutput
Result of applying a router to a clip.

| Field | Type | Notes |
|---|---|---|
| `audio_path` | str | |
| `label` | int | |
| `routed_system` | str | Which system/rule was applied |
| `score` | float | The score from the routed system or sub-ensemble |
| `prediction` | int | threshold-tuned binary |

### StackerOutput
Result of the metadata-augmented stacker.

| Field | Type | Notes |
|---|---|---|
| `audio_path` | str | |
| `label` | int | |
| `lr_score` | float | LR stacker probability |
| `gbm_score` | float | GBM stacker probability |
| `prediction` | int | from best model, val-tuned threshold |

## Feature Engineering

### Metadata features for stacker (7 binary/continuous)
```
n_adults_int       = int(#_adults), NaN→0
n_children_int     = int(#_children), NaN→1
n_adults_ge2       = (n_adults_int >= 2).astype(int)
n_children_ge2     = (n_children_int >= 2).astype(int)
context_unknown    = (context == "unknown").astype(int)
has_interaction    = (Interaction_with_child == "yes").astype(int)
timepoint_is_36m   = (timepoint_norm == "36_month").astype(int)
```

### Rule-based router rules (priority order)
```python
def route(row) -> str:
    if "unknown" in row.context.lower():
        return "sortformer"
    if row.n_adults_int >= 2:
        return "mean(wavlm_mil, eend_eda)"
    if row.n_children_int >= 2:
        return "whisper_mil"
    if row.n_children_int == 1:
        return "vtc"
    return "best_audio_mil_mean"   # fallback
```

## Output File Contracts

### `test_metrics_tuned.json` (all sub-features)
```json
{
  "f1": float,
  "precision": float,
  "recall": float,
  "auroc": float,
  "auprc": float,
  "threshold": float,
  "n": int,
  "baseline_f1": float,       // best_audio_mil mean = 0.893
  "baseline_auroc": float,    // 0.878
  "delta_f1": float,          // this - baseline
  "delta_auroc": float
}
```

### `feature_importances.json` (metadata_stack only)
```json
{
  "lr_coefficients": {"feature_name": float, ...},
  "gbm_feature_importances": {"feature_name": float, ...}
}
```

### `config.json` (all sub-features)
```json
{
  "sub_feature": "A_rule" | "A_learned" | "B" | "C" | "D",
  "systems_used": [...],
  "metadata_features": [...],
  "router_rules": {...},   // A only
  "model_type": "lr" | "gbm" | "linear_head" | "cnn_head",
  "seed": 42,
  "val_threshold": float,
  "created": "2026-04-28"
}
```
