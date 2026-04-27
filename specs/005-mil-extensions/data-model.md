# Data Model: MIL Extensions

## Entities

### Aggregator (extended)

Extends the existing set in `mil/seg_model.py`.

| Name | Class | Key Parameters | Returns Weights? |
|------|-------|---------------|-----------------|
| `mean` | `MeanAgg` | — | No |
| `max` | `MaxAgg` | — | No |
| `attention` | `AttnAgg` | `attn_dim=256` | Yes (K_max) |
| `gated_attention` | `GatedAttnAgg` | `attn_dim=256` | Yes (K_max) |
| `noisy_or` | `NoisyORAgg` | — | No |
| `top_k` | `TopKAgg` | `k=3` | No |
| `transformer` | `TransformerAgg` | `num_layers=2, num_heads=4, ffn_dim=1536, dropout=0.3, weight_decay=0.01` | Yes (K_max, from CLS cross-attention) |

All aggregators implement: `forward(bag: Tensor[K_max, D], mask: Tensor[K_max]) → (logit: Tensor[], weights: Tensor[K_max] | None)`

---

### PerAgeBandMetrics

A new result structure emitted by the extended sweep per config, written alongside the existing `test_metrics.json`.

Fields:
- `frontend`: str
- `aggregator`: str
- `timepoint`: str (`14_month` | `36_month`)
- `f1`, `precision`, `recall`, `auroc`, `auprc`: float
- `n_clips`: int
- `threshold`: float (from val-tuned threshold applied to each band)

Written to: `mil/mil_results/seg_mil/{frontend}_{aggregator}/test_metrics_by_timepoint.csv` (column per band, row per metric — matches existing format).

---

### WeakDiarizationResult

Output of `mil/eval_weak_diarization.py`. One row per (frontend, aggregator, timepoint).

Fields:
- `frontend`: str
- `aggregator`: str
- `timepoint`: str
- `pearson_r`: float
- `pearson_pval`: float
- `spearman_rho`: float
- `spearman_pval`: float
- `auroc_ranking`: float  ← AUROC treating attention weight as ranking score vs. GT child fraction ≥ 0.5
- `n_segments`: int
- `n_clips`: int

Written to: `mil/mil_results/seg_mil/weak_diarization_eval.csv`

---

### TransformerConfig (logged in config.json)

Additional fields in `config.json` when `aggregator == "transformer"`:

```json
{
  "aggregator": "transformer",
  "transformer_num_layers": 2,
  "transformer_num_heads": 4,
  "transformer_ffn_dim": 1536,
  "transformer_dropout": 0.3,
  "transformer_weight_decay": 0.01
}
```

---

### AllConfigsEntry (extended)

The existing `all_configs.json` entries gain two new optional fields:

```json
{
  "frontend": "...",
  "aggregator": "...",
  "val_f1": ...,
  "test_f1": ...,
  "test_precision": ...,
  "test_recall": ...,
  "test_auroc": ...,
  "test_auprc": ...,
  "test_auroc_14month": ...,
  "test_auroc_36month": ...
}
```

The `test_auroc_14month` and `test_auroc_36month` fields are added once age-band inference is implemented (US2). Existing entries from the 16-config baseline will have these fields populated retroactively when age-band inference runs.

---

## File Layout (new/modified files)

```text
mil/
├── seg_model.py              MODIFIED — add NoisyORAgg, TopKAgg, TransformerAgg; extend build_aggregator()
├── seg_train.py              MODIFIED — add age-band inference after test eval; pass transformer HPs from config
├── eval_weak_diarization.py  NEW — reads attention weight CSVs + RTTMs, outputs weak_diarization_eval.csv
├── configs/
│   └── seg_mil_sweep.yaml    MODIFIED — extend aggregators: [mean, max, attention, gated_attention, noisy_or, top_k, transformer]
└── slurm/
    └── seg_mil_sweep.sh      MODIFIED — increase wall time to 48h for extended sweep (7 aggregators × 4 frontends = 28 configs)
```
