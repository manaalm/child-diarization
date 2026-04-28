# Data Model: Segment-Instance MIL

**Feature**: 004-segment-instance-mil
**Date**: 2026-04-23

---

## Entities

### SegmentInstance
Represents one diarizer-proposed speech segment embedded with the frozen WavLM backbone.

| Field | Type | Description |
|-------|------|-------------|
| `audio_path` | str | Absolute path to the source audio file |
| `start` | float | Segment start time (seconds) |
| `end` | float | Segment end time (seconds) |
| `embedding` | np.ndarray (D,) | Mean-pooled WavLM-base+ frame vectors over [start, end] |
| `attention_weight` | float or None | Aggregator-assigned weight; None for MeanAgg/MaxAgg |

**Validation rules**:
- `end > start`; minimum duration enforced by `cfg.min_seg_dur_sec` (same as ECAPA pipeline)
- `embedding` must be non-NaN and finite
- `attention_weight` in [0, 1] and sums to 1.0 over instances in a bag (when not None)

---

### SegmentBag
The full bag of `SegmentInstance` objects for one audio clip, used as input to the MIL aggregator.

| Field | Type | Description |
|-------|------|-------------|
| `audio_path` | str | Clip identifier |
| `child_id` | str | Target child ID (for grouping in evaluation) |
| `timepoint_norm` | str | "14_month" or "36_month" |
| `label` | int | 0 = child absent, 1 = child present |
| `instances` | List[SegmentInstance] | 0 to ~30 instances; empty list allowed |
| `frontend_name` | str | Diarizer that proposed these segments |

**Validation rules**:
- `label` in {0, 1}
- Empty `instances` list is valid; model predicts score 0.0 for empty bags

---

### MILConfiguration
Uniquely identifies one of the 16 experimental cells.

| Field | Type | Description |
|-------|------|-------------|
| `frontend_name` | str | One of: `usc_sail`, `pyannote`, `babar_vtc`, `vbx` |
| `aggregator_type` | str | One of: `mean`, `max`, `attention`, `gated_attention` |
| `encoder_name` | str | `wavlm-base-plus` (fixed for primary sweep) |
| `pool_method` | str | `mean` (within-segment pooling; fixed) |
| `seed` | int | 42 |

---

### MILResult
Per-clip prediction and per-segment attention weights for one MILConfiguration on val or test split.

| Field | Type | Description |
|-------|------|-------------|
| `audio_path` | str | Clip audio path |
| `child_id` | str | Target child ID |
| `timepoint_norm` | str | "14_month" or "36_month" |
| `label` | int | Ground-truth label |
| `prob` | float | Model-predicted child presence probability [0, 1] |
| `pred` | int | Thresholded prediction (0 or 1) |
| `n_instances` | int | Number of bag instances (0 for empty bags) |
| `top_seg_start` | float or None | Start time of highest-attention segment |
| `top_seg_end` | float or None | End time of highest-attention segment |
| `top_seg_weight` | float or None | Attention weight of highest-attention segment |

**Storage**: One CSV per (configuration, split): `{frontend}_{aggregator}/val_predictions.csv` and `test_predictions.csv`

---

### ConfigSummaryEntry
One row in the 16-row `all_configs.json` summary.

| Field | Type | Description |
|-------|------|-------------|
| `frontend` | str | Frontend name |
| `aggregator` | str | Aggregator type |
| `val_f1` | float | Val-split F1 at tuned threshold |
| `val_auroc` | float | Val-split AUROC |
| `val_auprc` | float | Val-split AUPRC |
| `test_f1` | float | Test-split F1 |
| `test_precision` | float | Test-split precision |
| `test_recall` | float | Test-split recall |
| `test_auroc` | float | Test-split AUROC |
| `test_auprc` | float | Test-split AUPRC |
| `threshold` | float | Similarity threshold tuned on val |
| `n_train_bags` | int | Number of training clips |
| `n_empty_bags_train` | int | Empty bags in training set |
| `config_path` | str | Relative path to per-config `config.json` |

---

## State Transitions

```
RTTM Cache (per frontend)
        ↓ load segments
SegmentBag collection
        ↓ WavLM forward pass + mean pool per segment
SegmentEmbeddingCache (per frontend, persisted to disk)
        ↓ bag assembly
SegmentBag with embeddings
        ↓ MIL aggregator forward pass
clip-level score [0,1]
        ↓ threshold (tuned on val)
MILResult (pred, prob, attention weights)
        ↓ aggregate over clips
ConfigSummaryEntry → all_configs.json
```

---

## Relationships

- One `SegmentBag` contains 0–N `SegmentInstance` objects
- One `MILConfiguration` maps to exactly one trained `AggregatorHead` and one results directory
- `SegmentEmbeddingCache` is shared across all 4 `MILConfiguration` objects with the same `frontend_name`
- `all_configs.json` contains one `ConfigSummaryEntry` per `MILConfiguration` (16 total)

---

## File Layout

```
mil/
├── seg_dataset.py            # SegmentBagDataset: RTTM → embeddings → bags
├── seg_model.py              # MeanAgg, MaxAgg, AttnAgg, GatedAttnAgg (wraps existing GatedABMILHead)
├── seg_embedding_cache.py    # Disk-backed segment embedding cache
├── seg_train.py              # Sweep script: 16-config training + evaluation loop
├── configs/
│   └── seg_mil_sweep.yaml    # Sweep config: frontends, aggregators, training HPs, paths
└── mil_results/
    └── seg_mil/
        ├── all_configs.json
        └── {frontend}_{aggregator}/
            ├── config.json
            ├── val_predictions.csv
            ├── test_predictions.csv
            ├── val_metrics.json
            └── test_metrics.json
```

No external API contracts are required; this is a training/evaluation script module, not a service.
