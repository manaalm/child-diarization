# Quickstart — spec-014 MIL Extensions

End-to-end recipe for running the three user stories. All commands assume the repo root is the working directory and `child-vocalizations` conda env is active.

---

## Prerequisites

- Existing seen-child split CSV at `whisper-modeling/seen_child_splits/{train,val,test}.csv`.
- Existing baseline runs `mil/mil_results/wavlm_mil/`, `whisper_mil/`, `hubert_large_mil/` (for delta comparisons).
- For US2: child-adapted WavLM checkpoint at `synth_results/child_wavlm_checkpoint/step_50000/` (produced by `synth/slurm/run_wavlm_pretrain.sh`).
- SLURM access on ORCD with 1× A100 GPU per job.

```bash
# Sanity check: required files exist
test -f mil/mil_results/wavlm_mil/test_metrics_tuned.json   || echo "missing wavlm_mil baseline"
test -d synth_results/child_wavlm_checkpoint/step_50000     || echo "missing child-adapted checkpoint"
test -f mil/configs/wavlm_mil_child_adapted.yaml            || echo "missing US2 config"
```

---

## Step 0: Backward-compatibility regression check (run once after code changes)

Re-run a known-good baseline against the new code; assert reproducibility within ±0.005 AUROC.

```bash
sbatch mil/slurm/train_mil.sh mil/configs/wavlm_mil.yaml
# After completion, diff the test metrics against the committed baseline:
python -c "
import json
new = json.load(open('mil/mil_results/wavlm_mil/test_metrics_tuned.json'))
# (Compare against the committed baseline values: F1 0.882, AUROC 0.771)
print(f\"AUROC delta = {new['auroc'] - 0.771:+.4f}\")
print(f\"F1 delta    = {new['f1'] - 0.882:+.4f}\")
"
# PASS criterion: |AUROC delta| <= 0.005 AND |F1 delta| <= 0.01
```

If this fails, do not proceed — the new code has changed behavior on existing configs.

---

## Step 1: US1 — Weighted-Layer-Sum

Train three layer-sum variants in parallel; each is a standalone SLURM job.

```bash
sbatch mil/slurm/train_mil.sh mil/configs/wavlm_mil_layersum.yaml         # ~24 h
sbatch mil/slurm/train_mil.sh mil/configs/whisper_mil_layersum.yaml       # ~24 h
sbatch mil/slurm/train_mil.sh mil/configs/hubert_large_mil_layersum.yaml  # ~36 h (larger backbone)
```

After all three complete, run evaluation:

```bash
sbatch mil/slurm/eval_mil.sh
# Output per run: test_metrics_tuned.json, test_predictions.csv,
#                 test_metrics_by_timepoint.csv, layer_weights.json
```

Inspect which layers the model selected:

```bash
for run in wavlm_mil_layersum whisper_mil_layersum hubert_large_mil_layersum; do
    echo "=== $run ==="
    python -c "
import json
w = json.load(open('mil/mil_results/$run/layer_weights.json'))
top = sorted(w.items(), key=lambda kv: -kv[1])[:5]
print('Top 5 layers:', top)
"
done
```

**Expected sanity**: top layer should not always be the last layer. If `layer_weights.json` shows a one-hot at the final layer, the implementation collapsed back to the baseline.

---

## Step 2: US2 — Child-Adapted WavLM Wired Into MIL

Pre-flight check + train + eval:

```bash
# Pre-flight: assert checkpoint exists
test -f synth_results/child_wavlm_checkpoint/step_50000/config.json || \
    { echo "ERROR: pretrain not finished; submit synth/slurm/run_wavlm_pretrain.sh first"; exit 2; }

# Train
sbatch mil/slurm/train_mil.sh mil/configs/wavlm_mil_child_adapted.yaml

# Eval (after training completes)
sbatch mil/slurm/eval_mil.sh

# Compare to off-the-shelf WavLM baseline:
python -c "
import json
adapted = json.load(open('mil/mil_results/wavlm_mil_child_adapted/test_metrics_tuned.json'))
baseline = json.load(open('mil/mil_results/wavlm_mil/test_metrics_tuned.json'))
for k in ['f1','auroc','auprc']:
    print(f'{k}: {adapted[k]:.4f} vs baseline {baseline[k]:.4f}  (delta = {adapted[k]-baseline[k]:+.4f})')
"
```

If US1 layer-sum is positive, also run the combined config:

```bash
sbatch mil/slurm/train_mil.sh mil/configs/wavlm_mil_child_adapted_layersum.yaml
```

---

## Step 3: US3 — ACMIL Head Drop-In

```bash
sbatch mil/slurm/train_mil.sh mil/configs/wavlm_mil_acmil.yaml      # ~36 h
sbatch mil/slurm/train_mil.sh mil/configs/whisper_mil_acmil.yaml    # ~36 h
sbatch mil/slurm/eval_mil.sh
```

Then run weak-diarization eval per branch:

```bash
python mil/eval_weak_diarization.py \
    --results-dir mil/mil_results/wavlm_mil_acmil \
    --rttm-cache whisper-modeling/usc_sail_rttm_cache \
    --split-csv whisper-modeling/seen_child_splits/test.csv \
    --output mil/mil_results/wavlm_mil_acmil/branch_alignment.csv

# Inspect: do branches differ in alignment with GT?
python -c "
import pandas as pd
df = pd.read_csv('mil/mil_results/wavlm_mil_acmil/branch_alignment.csv')
print(df.groupby('branch')[['pearson','auroc']].mean())
# Healthy MBA: branches differ in alignment by >0.02 AUROC.
# Collapsed MBA: all branches identical → diversity loss did not bite.
"
```

---

## Step 4: Cross-child evaluation (any of US1/US2/US3 that show positive deltas on seen-child)

For any positive-delta seen-child run, repeat the experiment on the cross-child split to verify generalization. The Whisper-MIL cross-child baseline is 0.876 AUROC (higher than seen-child 0.853); regressions there would reverse the seen-child win.

```bash
# Add `--split cross_child` or use the `*_cross_child.yaml` companion configs
# (follow whatever convention spec-002 / spec-005 already established for cross-child runs)
```

---

## Step 5: Update results table and CLAUDE.md

```bash
# Append rows to the seen-child results table in results_summary.md
# Format: | Variant | F1 | Precision | Recall | AUROC | AUPRC | delta_AUROC vs baseline |

# Append a Recent Changes entry to CLAUDE.md mirroring the format of prior entries:
# - **Spec-014 US1/US2/US3 result** (spec-014, 2026-MM-DD, job NNNNNN): <one-line summary>
#   <result numbers and delta>. Root cause / interpretation.
#   Results: `mil/mil_results/<run_name>/`.
```

---

## Troubleshooting

- **OOM during training**: reduce `batch_size` in the YAML (default 16 → 8). Layer-sum mode does not change peak memory; ACMIL with n_branches=5 adds ~5% overhead.
- **`layer_weights.json` shows one-hot at last layer**: model collapsed to baseline. Check `softmax(layer_weights)` is being computed every forward and that gradients flow through `layer_weights` (it must be in the optimizer's param groups).
- **ACMIL diversity loss → 0 with branches identical**: increase `acmil_mba_diversity_weight` (try 0.5, 1.0); the cosine penalty needs to be loud enough to fight gradient pull toward a single mode.
- **`AutoModel.from_pretrained` cannot load child-adapted checkpoint**: confirm `synth_results/child_wavlm_checkpoint/step_50000/` contains both `config.json` and `pytorch_model.bin`. If only a `state_dict.pt` exists, the pretraining script needs to write a HF-format directory (see `synth/slurm/run_wavlm_pretrain.sh`).
- **Existing baseline regresses after spec-014 code changes**: re-run Step 0 regression check; if the delta exceeds threshold, bisect the changes between `BackboneExtractor`/`build_mil_model` and the head factory.

---

## Definition of Done

- [ ] Step 0 regression check passes (baseline reproduces within ±0.005 AUROC).
- [ ] All three US runs (and any conditional combined runs) complete with output schemas per `data-model.md` §6.
- [ ] `results_summary.md` updated with deltas vs. existing baselines on both seen-child and cross-child.
- [ ] `CLAUDE.md` Recent Changes entries added for each US.
- [ ] `layer_weights.json` (US1) shows non-trivial layer selection; `branch_weights.json` (US3) shows non-collapsed branches.
- [ ] No test-data leakage: thresholds tuned on val only; eval timestamps confirm val ran before test.
