# Implementation Plan: Metadata-Conditioned Routing and Ensemble Extensions

**Branch**: `012-metadata-routing-ensemble` | **Date**: 2026-04-28 | **Spec**: [spec.md](./spec.md)

## Summary

Four targeted post-hoc ensemble extensions exploiting BIDS scene metadata and stratified error analysis. Sub-features A/B (router and metadata-augmented stacker) are CPU-only, no new training data, run in minutes. Sub-features C/D (multi-child suppressor, short-voc head) require short GPU jobs. All use the seen-child test split (441 clips). Baseline to beat: `best_audio_mil` mean ensemble F1=0.893, AUROC=0.878, AUPRC=0.956.

## Technical Context

**Language/Version**: Python 3.11, `child-vocalizations` conda env  
**Primary Dependencies**: pandas, scikit-learn (LR, GBM via HistGradientBoosting), numpy, torch + torchaudio (sub-features C/D only), transformers (WavLM backbone for C/D)  
**Storage**: CSV predictions in-place; new results written to `ensemble_runs/metadata_router_rule/`, `ensemble_runs/metadata_router_learned/`, `ensemble_runs/metadata_stack/`; suppressor/short-voc head results in `mil/mil_results/multi_child_suppressor/`, `mil/mil_results/short_voc_head/`  
**Testing**: Manual: verify output JSON metrics + no test-set leakage; automated: assert val/test split integrity before writing test metrics  
**Target Platform**: Linux cluster (SLURM); A/B run on login node or short CPU job; C/D via SLURM GPU job  
**Project Type**: Research pipeline (offline experiment scripts, not a service)  
**Performance Goals**: A/B complete in <5 minutes; C/D GPU training <2h  
**Constraints**: No test-data leakage; all thresholds tuned on val; results committed alongside config JSON; split integrity assertions before any test evaluation  
**Scale/Scope**: 441 test clips, 431 val clips, 1311 train clips; 10 available system score columns

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Principle | Status | Notes |
|---|---|---|
| I. Reproducibility | ✅ PASS | seed=42 throughout; YAML/JSON configs committed alongside results |
| II. Split Discipline | ✅ PASS | All thresholds tuned on val only; test touched once for final reporting; assertion added in code |
| III. Baseline-First | ✅ PASS | Explicitly compared against best_audio_mil mean (0.893 F1) and all_available LR (0.897 F1) baselines |
| IV. Rigorous Evaluation | ✅ PASS | Report F1, AUROC, AUPRC, threshold, plus per-timepoint breakdown; stratum-specific metrics for C |
| V. Ablations & Error | ✅ PASS | Rule-based and learned router both evaluated (A); per-stratum before/after for C; FP rate on non-short-voc clips for D |
| VI. Thesis Sync | ✅ PASS | Results written to committed result folders; config.json saved alongside |
| VII. Documentation | ✅ PASS | CLAUDE.md updated with new scripts and result locations |

## Project Structure

### Documentation (this feature)

```text
specs/012-metadata-routing-ensemble/
├── plan.md              ← this file
├── research.md          ← Phase 0 output
├── data-model.md        ← Phase 1 output
└── tasks.md             ← Phase 2 output (/speckit.tasks)
```

### Source Code

```text
evaluation/
├── metadata_router.py          # Sub-features A + B (router + metadata stacker)
├── multi_child_suppressor.py   # Sub-feature C (FP suppressor for n_children≥2)
└── short_voc_head.py           # Sub-feature D (fine-grained head for <0.5s vocalizations)

ensemble_runs/
├── metadata_router_rule/       # A: rule-based router results
│   ├── config.json
│   ├── test_metrics_tuned.json
│   ├── val_metrics_tuned.json
│   └── test_predictions.csv
├── metadata_router_learned/    # A: learned router results
│   └── [same layout]
└── metadata_stack/             # B: metadata-augmented stacker results
    ├── config.json
    ├── feature_importances.json
    ├── test_metrics_tuned.json
    ├── val_metrics_tuned.json
    └── test_predictions.csv

mil/mil_results/
├── multi_child_suppressor/     # C results
│   ├── config.json
│   ├── test_metrics_tuned.json
│   ├── test_metrics_multi_child_only.json   # stratum-specific
│   ├── test_metrics_single_child_only.json  # guard against regression
│   └── test_predictions.csv
└── short_voc_head/             # D results
    ├── config.json
    ├── best_checkpoint.pt
    ├── test_metrics_tuned.json
    ├── test_metrics_short_voc_clips.json    # target stratum
    ├── test_metrics_non_short_voc_clips.json
    └── test_predictions.csv

evaluation/slurm/
├── run_multi_child_suppressor.sh   # GPU job for C
└── run_short_voc_head.sh           # GPU job for D
```

---

## Phase 0: Research ✅

See [research.md](./research.md). All unknowns resolved:
- System prediction paths and score column names confirmed for all 10 available systems
- Metadata column names confirmed (`#_adults`, `#_children`, `Context` — not the names assumed in the spec)
- Router rules derived from actual stratified F1 tables
- Pyannote path identified as non-standard: `pyannote/pyannote_enrollment_runs/test_predictions.csv`

## Phase 1: Design ✅

See [data-model.md](./data-model.md). Key contracts:
- `ClipRecord` — join key: `audio_path`; metadata parsed from `#_adults`/`#_children` string cols
- `SystemScore` — 10 systems × 441 clips; MIL "score" col renamed to "prob"; audio_llm imputed with 0.5 for ~16 missing clips
- `RouterOutput` / `StackerOutput` — unified output format with delta vs. baseline in JSON
- All result directories documented above with file layouts
- Constitution II compliance: threshold tuning uses only val; test metrics written in a single final pass
