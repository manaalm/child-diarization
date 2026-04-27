# Implementation Plan: Synthetic Child-Adult Scene Generator

**Branch**: `008-synthetic-child-scenes` | **Date**: 2026-04-24 | **Spec**: [spec.md](spec.md)

## Summary

Build a configurable synthetic audio scene generator that composes real child vocalization segments and adult speech into labeled multi-speaker parent-child clips. The system produces WAV + RTTM + clip-label outputs targeting known failure modes in the existing WavLM/BabAR/ECAPA pipeline: missed short toddler vocalizations, false positives from adult/sibling/TV speech, and poor performance under overlap and low SNR. Downstream value is measured on held-out real test clips at multiple synthetic-to-real training ratios.

---

## Technical Context

**Language/Version**: Python 3.11 — conda `child-vocalizations` env (same as existing `av_fusion/` and `mil/` pipeline; no new env required for MVP)
**Primary Dependencies**: numpy, scipy, soundfile, torchaudio, pandas, scikit-learn, PyYAML, tqdm, matplotlib, seaborn — all already installed in `child-vocalizations`; optional librosa (F0/formant analysis in quality script); optional pyroomacoustics (synthetic RIR generation, stretch only)
**Storage**: WAV files + plain-text CSV/JSON/RTTM; no database
**Testing**: pytest — unit tests for audio composition, label generation, and manifest integrity; integration test runs 10 scenes end-to-end and verifies label–audio consistency
**Target Platform**: Linux SLURM cluster (same node setup as existing jobs); MVP is CPU-only; no GPU required
**Project Type**: CLI pipeline scripts organized as a `synth/` package at repo root (mirrors `mil/`, `av_fusion/` layout)
**Performance Goals**: ≥5,000 scenes in < 4 hours on a single CPU node
**Constraints**: All output audio 16 kHz mono; seed=42 default; configs committed alongside results; no neural TTS in MVP
**Scale/Scope**: 5,000–50,000 synthetic scenes per ratio experiment; augmenting ~1,311 real training clips (seen-child split)

---

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-checked after Phase 1 design. — Constitution v1.1.0*

| Principle | Check | Status |
|-----------|-------|--------|
| **I. Reproducibility** | Scene configs YAML committed alongside results; `random_seed` required in every config; `child-vocalizations` conda env; cache invalidated when source segments or config hash changes | PASS |
| **II. Data Integrity** | Manifest builder enforces speaker-level split assignment with assertion; no test-set speakers in synthetic training pool; seen-child split only; evaluation reports state split | PASS |
| **III. Baseline-First** | 0× ratio (real-only) is mandatory first evaluation; compared against existing BabAR/VTC/WavLM enrollment baselines on same split | PASS |
| **IV. Metrics** | All five metrics (F1, Precision, Recall, AUROC, AUPRC) per ratio; per-age-band breakdown; threshold tuned on val only | PASS |
| **V. Ablations** | 13-ablation sweep; `error_analysis_synthetic.py` covers all eight failure-mode categories; all results saved to CSV | PASS |
| **VI. Thesis Sync** | Results committed to `synth_results/`; config.json alongside each experiment; figures from committed CSVs | PASS |
| **VII. Documentation** | CLAUDE.md update is a required final task; all scripts have docstrings; cache invalidation gotchas documented in README | PASS |

**No violations. Gate passes.**

---

## Project Structure

### Documentation (this feature)

```text
specs/008-synthetic-child-scenes/
├── plan.md              # This file
├── research.md          # Phase 0 decisions
├── data-model.md        # Phase 1 entity definitions
├── quickstart.md        # Phase 1 reproduction guide
├── contracts/
│   ├── segment-manifest.md
│   ├── scene-config.md
│   ├── rttm-output.md
│   ├── clip-labels.md
│   ├── scene-metadata.md
│   └── training-manifest.md
├── checklists/
│   └── requirements.md
└── tasks.md             # Phase 2 output (/speckit.tasks — NOT created here)
```

### Source Code (repository root)

```text
synth/
├── configs/
│   ├── default_14_18mo.yaml
│   ├── default_34_38mo.yaml
│   ├── hard_negatives.yaml
│   ├── overlap_stress.yaml
│   └── low_snr_stress.yaml
├── scripts/
│   ├── build_segment_manifest.py
│   ├── extract_segments.py
│   ├── generate_scenes.py
│   ├── generate_training_sets.py
│   ├── train_with_synthetic.py
│   ├── evaluate_synthetic_augmentation.py
│   ├── analyze_synthetic_quality.py
│   └── error_analysis_synthetic.py
├── slurm/
│   ├── run_scene_generation.sh
│   └── run_ratio_sweep.sh
├── manifest.py
├── scene_generator.py
├── turn_taking.py
├── audio_utils.py
├── labels.py
└── README.md

data/                          # gitignore'd; user-populated
├── segments/child/
├── segments/adult/
├── noise/
└── rirs/

synth_results/                 # committed
├── synthetic_scenes/{wav,rttm,json}/
├── manifests/
└── augmentation_experiments/{config_name}/
    ├── config.json
    ├── metrics_by_ratio.csv
    ├── metrics_by_age_band.csv
    ├── error_analysis.csv
    └── figures/

tests/synth/
├── test_audio_utils.py
├── test_labels.py
├── test_manifest.py
├── test_turn_taking.py
└── test_integration.py
```

**Structure Decision**: Single package under `synth/` mirrors existing `mil/` and `av_fusion/` layout. Core library modules at package root; CLI entry points in `scripts/`; SLURM wrappers in `slurm/`. Results committed to `synth_results/`; raw data gitignore'd in `data/`.

---

## Complexity Tracking

*No constitution violations — table not required.*
