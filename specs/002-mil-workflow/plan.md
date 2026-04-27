# Implementation Plan: Multiple Instance Learning Workflow

**Branch**: `002-mil-workflow` | **Date**: 2026-04-23 | **Spec**: [spec.md](spec.md)
**Input**: Feature specification from `specs/002-mil-workflow/spec.md`

---

## Summary

Train and evaluate a Gated Attention-Based MIL (ABMIL) child presence detector using
only clip-level binary labels — no frame-level annotations or diarization front-end
required. Two backbone variants (WavLM-base+ and Whisper-small, frozen) produce
instance embeddings from 2-second audio windows; the ABMIL head learns to attend to
child-vocalization windows. Results are evaluated on the seen-child test split and
reported in the same format as the enrollment-based diarization baselines, enabling
direct thesis table comparison.

---

## Technical Context

**Language/Version**: Python 3.11
**Primary Dependencies**: PyTorch, torchaudio, transformers (WavLM-base+ / Whisper-small),
scikit-learn, pandas, numpy, PyYAML — all present in the existing `child-vocalizations`
conda environment; no new packages required.
**Storage**: Local NFS (`/orcd/scratch/`); results in `mil/mil_results/{variant}/`;
HuggingFace model cache for backbone weights.
**Testing**: Manual validation — train on seen-child split, verify val/test metric
files match expected schema; no automated test suite (ML research project, per
Constitution Principle IV).
**Target Platform**: SLURM GPU cluster (single A100 or V100 node); Linux.
**Project Type**: ML research pipeline / CLI scripts.
**Performance Goals**: Training completes in ≤ 8 hours per variant on one GPU.
**Constraints**: Must use seen-child split exclusively; seed=42; frozen backbone;
result files must match unified.py schema exactly.
**Scale/Scope**: 2183 clips (seen-child split); ~29 windows per 30 s clip → ~63k
instances total; two backbone variants.

---

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-checked post-design.*

| Principle | Status | Notes |
|-----------|--------|-------|
| I. Reproducibility | ✓ PASS | seed=42 in every config; config.json committed alongside results; existing conda env (no new env needed — same packages as baselines/) |
| II. Data Integrity | ✓ PASS | seen_child_splits/ used exclusively; val for threshold tuning; test touched once for final reporting |
| III. Baseline-First | ✓ PASS | MIL compared against enrollment-based baselines on same seen-child split (USC-SAIL F1=0.874, BabAR F1=0.874, VTC F1=0.888). Cross-child encoder baselines (baselines/baseline_results/) use a different split paradigm and serve as reference context, not direct competitors |
| IV. Rigorous Evaluation | ✓ PASS | F1, precision, recall, AUROC, AUPRC reported; per-timepoint (14_month, 36_month) breakdown in test_metrics_by_timepoint.csv; threshold reported |
| V. Ablations & Error Analysis | ✓ PASS | Two backbone variants = controlled ablation; per-child error rates required post-run (via error_analysis.py equivalent) |
| VI. Thesis Sync | ✓ PASS | All result files committed under mil/mil_results/; thesis_tables.yaml updated; no manual transcription |
| VII. Documentation | ✓ PASS | Docstrings on all scripts; CLAUDE.md updated with mil/ section; Gotchas documented in quickstart.md |

**No constitution violations. No Complexity Tracking entries required.**

---

## Project Structure

### Documentation (this feature)

```text
specs/002-mil-workflow/
├── plan.md                      ← this file
├── research.md                  ← Phase 0 decisions (complete)
├── data-model.md                ← entities and schemas (complete)
├── quickstart.md                ← integration guide (complete)
├── contracts/
│   └── script-interfaces.md    ← CLI contracts and config schemas
├── checklists/
│   └── requirements.md
└── tasks.md                     ← Phase 2 output (created by /speckit.tasks)
```

### Source Code (repository root)

```text
mil/
├── mil_model.py              # ABMIL model: FeatureExtractor + GatedABMIL + ClassifierHead
├── mil_dataset.py            # MILBagDataset: window extraction + feature caching logic
├── mil_train.py              # Training entry point: loads config, trains, writes results
├── mil_evaluate.py           # Evaluation entry point: loads checkpoint, writes test metrics
├── mil_age_stratified.py     # Age-cohort evaluation: filters by manifest age_group
├── configs/
│   ├── wavlm_mil.yaml        # WavLM-base+ backbone config (window=2s, stride=1s, seed=42)
│   └── whisper_mil.yaml      # Whisper-small backbone config
├── slurm/
│   └── train_mil.sh          # SLURM GPU job: activates conda env, calls mil_train.py
└── mil_results/              # Created at runtime; one subdirectory per variant
    └── {variant_name}/       # e.g., wavlm_mil/, whisper_mil/
        ├── config.json
        ├── training_history.csv
        ├── best_checkpoint.pt
        ├── val_metrics_tuned.json
        ├── val_predictions.csv
        ├── val_metrics_by_timepoint.csv
        ├── test_metrics_tuned.json
        ├── test_predictions.csv
        ├── test_metrics_by_timepoint.csv
        └── age_stratified/
            ├── 12_16m/
            │   ├── test_metrics_tuned.json
            │   ├── test_predictions.csv
            │   └── test_metrics_by_timepoint.csv
            └── 34_38m/
                └── ...

# Existing files modified:
evaluation/configs/thesis_tables.yaml   # add mil_wavlm and mil_whisper entries
CLAUDE.md                               # add mil/ section to Architecture and Results Storage
```

**Structure Decision**: New top-level `mil/` module, parallel to `baselines/`,
`synthesis/`, and `evaluation/`. This keeps MIL independently runnable and avoids
adding complexity to the already large `baselines/baseline_encoders.py`. The `mil/`
conda env is the existing `child-vocalizations` (no new uv environment created).

---

## Key Design Decisions (from research.md)

| Decision | Choice | Rationale |
|----------|--------|-----------|
| MIL architecture | Gated ABMIL (Ilse et al. 2018) | Interpretable attention; stable on imbalanced bags |
| Backbones | WavLM-base+ and Whisper-small (frozen) | Matches strongest existing baselines; controlled comparison |
| Window size / stride | 2 s / 1 s (50% overlap) | Captures 2–4 utterances per window; interpretable attention |
| Intra-window pooling | Mean-pool over frames | Matches MeanPooling in baselines; no extra parameters |
| Environment | Existing `child-vocalizations` conda env | Same packages as baselines/; avoids redundant backbone downloads |
| Result folder schema | Identical to unified.py outputs | thesis_tables.yaml integration with zero code changes |
| Comparison baseline | Enrollment-based systems on seen-child split | Same split → fair comparison; cross-child baselines reference-only |
