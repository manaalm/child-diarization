# Quickstart: MIL Workflow

**Prerequisites**: repo checked out, `child-vocalizations` conda env active,
`whisper-modeling/seen_child_splits/` committed, GPU node available.

---

## 1. Train MIL (both backbone variants)

Submit SLURM jobs for the two backbone variants:

```bash
mkdir -p logs/mil
sbatch mil/slurm/train_mil.sh mil/configs/wavlm_mil.yaml
sbatch mil/slurm/train_mil.sh mil/configs/whisper_mil.yaml
```

Each job runs for up to 8 hours. Monitor progress:

```bash
tail -f logs/mil/train_<jobid>.out
```

Results land in:
```
mil/mil_results/wavlm_mil/    ← WavLM backbone
mil/mil_results/whisper_mil/  ← Whisper backbone
```

Each folder will contain `config.json`, `training_history.csv`,
`best_checkpoint.pt`, `val_metrics_tuned.json`, `val_predictions.csv`.

---

## 2. Evaluate on Test Split

After training completes, run evaluation for each variant:

```bash
python mil/mil_evaluate.py \
    --checkpoint mil/mil_results/wavlm_mil/best_checkpoint.pt \
    --config     mil/mil_results/wavlm_mil/config.json

python mil/mil_evaluate.py \
    --checkpoint mil/mil_results/whisper_mil/best_checkpoint.pt \
    --config     mil/mil_results/whisper_mil/config.json
```

This adds `test_metrics_tuned.json`, `test_predictions.csv`, and
`test_metrics_by_timepoint.csv` to each result folder.

---

## 3. Age-Stratified Evaluation

```bash
for AGE in 12_16m 34_38m; do
  python mil/mil_age_stratified.py \
      --checkpoint mil/mil_results/wavlm_mil/best_checkpoint.pt \
      --config     mil/mil_results/wavlm_mil/config.json \
      --age-group  $AGE \
      --manifest   playlogue/manifest.csv

  python mil/mil_age_stratified.py \
      --checkpoint mil/mil_results/whisper_mil/best_checkpoint.pt \
      --config     mil/mil_results/whisper_mil/config.json \
      --age-group  $AGE \
      --manifest   playlogue/manifest.csv
done
```

Results land in `mil/mil_results/{variant}/age_stratified/{age_group}/`.

---

## 4. Update Thesis Tables Config

Add MIL result paths to `evaluation/configs/thesis_tables.yaml`
(see `contracts/script-interfaces.md` for the exact YAML snippet), then regenerate:

```bash
python evaluation/aggregate_thesis_tables.py
```

MIL rows should appear in the comparative baseline table alongside USC-SAIL,
BabAR, VTC, etc.

---

## 5. Commit Results

```bash
git add mil/mil_results/
git add mil/configs/ mil/slurm/ mil/*.py
git commit -m "feat: add MIL workflow results (wavlm_mil, whisper_mil)"
```

---

## Expected Results (targets)

| Variant | F1 (target) | AUROC | AUPRC |
|---------|-------------|-------|-------|
| wavlm_mil | ≥ 0.850 | — | — |
| whisper_mil | ≥ 0.850 | — | — |

Actual values are filled in after the training runs complete.

---

## Troubleshooting

**`RuntimeError: CUDA out of memory`**: Reduce `batch_size` in the YAML config
(try 4 → 2). Bags vary in length; a single long clip creates many windows.

**`0 training clips after filtering`**: Check that `audio_exists == True` rows exist
in `seen_child_splits/train.csv`. Run:
```bash
python -c "import pandas as pd; df=pd.read_csv('whisper-modeling/seen_child_splits/train.csv'); print(df.audio_exists.value_counts())"
```

**Backbone download on first run**: WavLM-base+ and Whisper-small are downloaded from
HuggingFace Hub (~300 MB each) on first use. Ensure internet access from the compute node
or pre-cache with `python -c "from transformers import WavLMModel; WavLMModel.from_pretrained('microsoft/wavlm-base-plus')"` on a login node.
