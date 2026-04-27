# Quickstart: Audio LLM Zero-Shot Baseline

**Feature**: 010-audio-llm-baseline
**Date**: 2026-04-27

This guide covers the end-to-end workflow for running Qwen2-Audio-7B-Instruct as a zero-shot child vocalization detector on the seen-child test split.

---

## Prerequisites

```bash
conda activate child-vocalizations
pip install "transformers>=4.45" accelerate soundfile torchaudio
# Optional (for 4-bit quantization):
pip install bitsandbytes

# HuggingFace token (for gated models, if applicable):
export HF_TOKEN=<your_token>
```

The model weights (~15 GB) download automatically to `~/.cache/huggingface/` on first run. Ensure the compute node has internet access or pre-download with:
```bash
huggingface-cli download Qwen/Qwen2-Audio-7B-Instruct
```

---

## Step 1: Validate the Setup (Dry Run)

```bash
python baselines/audio_llm_baseline.py \
  --split val \
  --max-clips 5 \
  --dry-run
```

Expected output: prints 3 prompt templates to stdout, then exits with code 0.

---

## Step 2: Run Val-Set Inference + Threshold Tuning

```bash
python baselines/audio_llm_baseline.py \
  --split val \
  --model Qwen/Qwen2-Audio-7B-Instruct \
  --model-slug qwen2_audio_7b \
  --output-dir baselines/audio_llm_baseline_runs/qwen2_audio_7b \
  --cache-dir baselines/audio_llm_cache/qwen2_audio_7b \
  --seed 42
```

Or via SLURM:
```bash
sbatch baselines/slurm/run_audio_llm_baseline.sh --split val
```

**Runtime estimate**: ~2–4 hours on A100 (40GB) for 437 val clips at ~10–30s per clip.

**Outputs**:
```
baselines/audio_llm_baseline_runs/qwen2_audio_7b/
├── val_predictions.csv
└── val_metrics_tuned.json   ← threshold written here
```

---

## Step 3: Run Test-Set Inference

```bash
python baselines/audio_llm_baseline.py \
  --split test \
  --model Qwen/Qwen2-Audio-7B-Instruct \
  --model-slug qwen2_audio_7b \
  --output-dir baselines/audio_llm_baseline_runs/qwen2_audio_7b \
  --cache-dir baselines/audio_llm_cache/qwen2_audio_7b \
  --seed 42
```

**IMPORTANT**: Step 2 must complete before Step 3 (`val_metrics_tuned.json` must exist).

Or via SLURM (run after Step 2 completes):
```bash
sbatch baselines/slurm/run_audio_llm_baseline.sh --split test
```

**Runtime estimate**: ~6–8 hours on A100 for 1309 test clips.

**Outputs**:
```
baselines/audio_llm_baseline_runs/qwen2_audio_7b/
├── test_predictions.csv
├── test_metrics_tuned.json
├── test_metrics_by_timepoint.csv
└── config.json
```

---

## Step 4: (Optional) Few-Shot Variant

```bash
python baselines/audio_llm_baseline.py \
  --split val \
  --n-shot 2 \
  --model-slug qwen2_audio_7b_2shot \
  --output-dir baselines/audio_llm_baseline_runs/qwen2_audio_7b_2shot \
  --cache-dir baselines/audio_llm_cache/qwen2_audio_7b_2shot \
  --seed 42
```

Then repeat Step 3 with `--model-slug qwen2_audio_7b_2shot` and `--split test`.

---

## Step 5: Compare Against Other Baselines

```bash
python evaluation/build_master_table.py  # includes audio LLM row if result folder present
# or directly compare:
python -c "
import json
babar = json.load(open('babar_ecapa_enrollment_runs/enroll_test_metrics.json'))
llm = json.load(open('baselines/audio_llm_baseline_runs/qwen2_audio_7b/test_metrics_tuned.json'))
for m in ['f1','auroc','auprc']:
    delta = llm[m] - babar[m]
    print(f'{m}: BabAR={babar[m]:.3f}, AudioLLM={llm[m]:.3f}, delta={delta:+.3f}')
"
```

---

## Smoke Test (Verify Correctness Without Full Run)

```bash
python baselines/audio_llm_baseline.py \
  --split val \
  --max-clips 10 \
  --output-dir /tmp/audio_llm_smoke \
  --cache-dir /tmp/audio_llm_cache_smoke \
  --seed 42

# Expected: 10 rows in val_predictions.csv, prob column in [0,1], no NaN in n_shot column
python -c "
import pandas as pd
df = pd.read_csv('/tmp/audio_llm_smoke/val_predictions.csv')
assert len(df) == 10
assert df['prob'].between(0.0, 1.0).all()
assert df['n_shot'].notna().all()
print('Smoke test passed.')
"
```

---

## Resuming an Interrupted Run

The cache ensures idempotency. Simply re-run the same command:

```bash
python baselines/audio_llm_baseline.py --split val ...  # same args
```

Clips already in `--cache-dir` are loaded instantly; only uncached clips are re-inferred.

---

## Gotchas

- **Step 3 before Step 2**: Will exit with code 2 and print `val_metrics_tuned.json not found — run --split val first`.
- **Degenerate outputs**: If `[WARNING] Degenerate predictions detected` appears, check the model prompt format — the model may not understand the audio input or is outputting multi-word responses.
- **VRAM**: 7B at bfloat16 requires ~18GB. If node has only 16GB, add `--quantize-4bit` (reduces to ~5GB, slight accuracy impact).
- **Cache invalidation**: If the prompt template changes, delete `baselines/audio_llm_cache/{model_slug}/` before rerunning — cached logits were generated with the old prompt.
- **Download failures**: If HF download times out, run `huggingface-cli download Qwen/Qwen2-Audio-7B-Instruct` from a login node (which has internet) before submitting the SLURM job.
