# Quickstart: Segment-Instance MIL

**Feature**: 004-segment-instance-mil
**Branch**: `004-segment-instance-mil`

---

## Prerequisites

1. **RTTM caches must exist** for all four target frontends. Verify:
   ```
   whisper-modeling/usc_sail_rttm_cache/
   pyannote/pyannote_rttm_cache/
   pyannote/vtc_rttm_cache/         (BabAR-VTC segments)
   pyannote/vbx_rttm_cache/
   ```
   If any is empty, run the corresponding enrollment pipeline first (`python pyannote/unified.py --diarizer <name>`).

2. **Environment**: Activate `child-vocalizations` conda env (Python 3.11):
   ```bash
   conda activate child-vocalizations
   ```

3. **WavLM model** is auto-downloaded by `transformers` on first run; no manual checkpoint needed.

---

## Step 1: Pre-compute Segment Embeddings (One-Time)

Pre-extract and cache WavLM embeddings for all segments from all four frontends. This is the most expensive step (~1–2 hrs on GPU) and needs to run only once:

```bash
cd /home/manaal/orcd/scratch/child-adult-diarization
python mil/seg_train.py --config mil/configs/seg_mil_sweep.yaml --precompute-only
```

This writes to `mil/seg_embedding_cache/{frontend_name}/`. All 16 training runs share this cache.

---

## Step 2: Run the 16-Configuration Sweep

```bash
cd /home/manaal/orcd/scratch/child-adult-diarization
python mil/seg_train.py --config mil/configs/seg_mil_sweep.yaml
```

Trains and evaluates all 16 (frontend × aggregator) configurations sequentially.
Output goes to `mil/mil_results/seg_mil/`. Progress is printed per configuration.

---

## Step 3: Inspect Results

```bash
# Summary table (16 rows, sorted by test AUROC)
python -c "
import json, pandas as pd
d = json.load(open('mil/mil_results/seg_mil/all_configs.json'))
df = pd.DataFrame(d).sort_values('test_auroc', ascending=False)
print(df[['frontend','aggregator','test_f1','test_auroc','test_auprc']].to_string())
"

# Per-clip predictions + attention weights for one config
python -c "
import pandas as pd
df = pd.read_csv('mil/mil_results/seg_mil/vbx_gated_attention/test_predictions.csv')
print(df.head())
"
```

---

## Step 4: SLURM Batch Run

For the full sweep on the cluster:
```bash
cd /home/manaal/orcd/scratch/child-adult-diarization
sbatch mil/slurm/seg_mil_sweep.sh
```

Expected wall time: ~4–6 hours for precomputation + all 16 training runs (GPU required).

---

## Step 5: Add Results to Thesis Tables

After results are committed, regenerate the comparison table:
```bash
python evaluation/aggregate_thesis_tables.py
```

The `table_segment_mil` entry in `evaluation/configs/thesis_tables.yaml` sources from `mil/mil_results/seg_mil/all_configs.json`.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `KeyError: frontend_name` in seg_dataset | RTTM cache missing | Run unified.py for that frontend first |
| Empty bag warning for all clips | Wrong rttm_cache path in sweep config | Check `rttm_cache_dir` in `seg_mil_sweep.yaml` under the relevant frontend |
| NaN loss at epoch 1 | All bags empty (embedding extractor broken) | Run `--precompute-only` first; check RTTM cache has .rttm files |
| CUDA OOM | WavLM forward pass on large segment | Reduce GPU memory usage by setting CUDA_VISIBLE_DEVICES or splitting precompute |
