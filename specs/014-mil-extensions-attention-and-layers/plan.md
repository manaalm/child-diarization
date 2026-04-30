# Implementation Plan: MIL Extensions — Weighted-Layer-Sum, Child-Adapted Backbone, ACMIL

**Branch**: `014-mil-extensions-attention-and-layers` | **Date**: 2026-04-29 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/specs/014-mil-extensions-attention-and-layers/spec.md`

## Summary

Six MIL extensions targeting the gated-ABMIL frame-window plateau (WavLM 0.771 / Whisper 0.853 test AUROC, seen-child) and the segment-instance MIL plateau (best babar_vtc gated-attn 0.808 from spec-005). US1–US3 modify `mil/mil_model.py` (frame-window MIL); US5/US6 modify `mil/seg_model.py` (segment-instance MIL); US4 spans both via a new `TSMILHead` class and a prototype-cache build script. (US1) **Learnable weighted-layer-sum** over WavLM/Whisper/HuBERT hidden_states replacing the single-layer read at `mil_model.py:65,68`. (US2) **Child-adapted WavLM wired into MIL** — completes spec-009 US3. (US3) **ACMIL head drop-in** (Zhang et al., ECCV 2024) — Multiple Branch Attention + Stochastic Top-K Instance Masking. (US4) **TS-MIL: target-speaker conditional MIL head** — concat or FiLM injection of per-(child, timepoint) ECAPA prototype, with a new `mil/scripts/build_prototype_cache.py` to dump prototypes built by `pyannote/unified.py:559` to disk for MIL training. (US5) **DSMIL dual-stream aggregator** (Li et al. CVPR 2021) for segment-instance MIL — max-instance stream + cosine-distance attention stream with averaged BCE losses. (US6) **Adaptive pooling operators** — AutoPool (McFee 2018), ExpSoftmaxPool, and GMAP added to `mil/seg_model.py:build_aggregator()`. All six are gated behind config flags with backward-compatible defaults; existing baselines reproduce within ±0.005 AUROC of committed numbers.

## Technical Context

**Language/Version**: Python 3.11, `child-vocalizations` conda env (same as spec-009 / spec-012)
**Primary Dependencies**: `torch`, `transformers` (WavLM/HuBERT/Whisper backbones), `numpy`, `pandas`, `scikit-learn` (metrics only); no new Python packages required for US1/US2; US3 introduces no new dependencies (ACMIL is pure PyTorch — clone reference impl from https://github.com/dazhangyu123/ACMIL but rewrite into `mil/mil_model.py` rather than vendoring the package).
**Storage**: New result directories under `mil/mil_results/`: `wavlm_mil_layersum/`, `whisper_mil_layersum/`, `hubert_large_mil_layersum/`, `wavlm_mil_child_adapted/`, `wavlm_mil_acmil/`, `whisper_mil_acmil/` (and combined `wavlm_mil_child_adapted_layersum/` if FR-010 triggers). Each follows the existing MIL output schema: `best_checkpoint.pt`, `config.json`, `val/test_metrics_tuned.json`, `val/test_predictions.csv`, `val/test_metrics_by_timepoint.csv`. New artifacts: `layer_weights.json` (US1), `branch_weights.json` (US3).
**Testing**: Manual validation of (a) numeric correctness on a small subset (verify `softmax(layer_weights) @ hidden_states` matches a sanity-check NumPy implementation), (b) split integrity (no test data touched during training; threshold tuned on val only), (c) backward compatibility (`wavlm_mil` baseline still trains and matches prior numbers when re-run with the new code).
**Target Platform**: ORCD SLURM cluster — 1× A100 GPU per training run, 24 h walltime per US1/US2 config, 36 h per US3 config (extra time for ACMIL diversity-loss tuning).
**Project Type**: Research pipeline (offline experiment scripts), single project layout — extends `mil/` package.
**Performance Goals**: US1 wall-clock ≤ baseline `wavlm_mil` ×1.05 (single forward over all hidden_states is cheap); US3 wall-clock ≤ baseline ×1.3 (n_branches=5 attention branches share a backbone); per-epoch GPU memory headroom ≥ 4 GB for n_branches=5 at batch_size=16.
**Constraints**: No test-data leakage (all thresholds tuned on val); seed=42 logged; backward-compatible defaults so the existing `wavlm_mil` and `whisper_mil` re-runs reproduce within ±0.005 AUROC of the committed baseline.
**Scale/Scope**: Same data scope as spec-002/004/005 — 1311 train / 431 val / 441 test seen-child clips, plus the 1517/505/505 cross-child split. Three US × 2–3 backbones × 2 splits ≈ 12–15 SLURM jobs total.

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

Constitution v1.1.0 — Principles I–VII compliance for spec-014:

- **I. Reproducibility & Environment** — PASS. Each new config under `mil/configs/` is committed alongside the resulting `config.json` per FR-006/FR-008; seed=42 default carried forward; SLURM job IDs logged in CLAUDE.md Recent Changes per FR-009. No new venv required (reuses `child-vocalizations` from spec-009).
- **II. Data Integrity & Split Discipline** — PASS. All three US use existing seen-child and cross-child splits unchanged; no train/val/test mixing. The acceptance scenarios explicitly reference the existing splits; threshold tuning on val only is asserted in FR-002 / US1 acceptance #2.
- **III. Baseline-First Development** — PASS. Each US compares against an already-committed baseline: US1 vs. `wavlm_mil`/`whisper_mil`/`hubert_large_mil` last-layer baselines; US2 vs. off-the-shelf `wavlm_mil`; US3 vs. gated-ABMIL on the same backbone+split. The spec calls out Whisper-MIL 0.853, WavLM-MIL 0.771 etc. as numeric reference points.
- **IV. Rigorous Evaluation & Appropriate Metrics** — PASS. All five primary metrics (F1, P, R, AUROC, AUPRC) come for free via `mil_utils.compute_metrics`. Per-timepoint breakdown is mandated by FR-008 and US1 acceptance #2 / US2 acceptance #3 / US3 weak-diarization checkout. Frame-level (RTTM) eval is not a goal of MIL extensions; covered by `mil/eval_weak_diarization.py` for US3 attention-alignment only.
- **V. Mandatory Ablations & Error Analysis** — PASS. US3 mandates per-branch alignment numbers and a comparison against gated-ABMIL on the same data (US3 acceptance #4–5). Per-age-band metrics required by FR-008. US1 layer_weights inspection itself is an ablation surface (which layer dominated?). One gap: no formal error-analysis script update for US1/US2 — mitigated by reuse of `error_analysis.py` / `pyannote_error_analysis.py` style tooling already in the repo; no new infra required.
- **VI. Thesis Synchronization** — PASS. FR-008 mandates `results_summary.md` updates with deltas and per-timepoint rows; FR-009 mandates `CLAUDE.md` Recent Changes entries mirroring the format used for prior negative results (TinyVox, hardneg, multi-child suppressor).
- **VII. Documentation & Honest Reporting** — PASS. New head class will carry a docstring referencing the ACMIL paper; `BackboneExtractor` weighted-layer-sum mode will document the leading-conv-feature skip behavior. Limitations call-outs (cross-child Whisper baseline 0.876 must not regress, child-adapted backbone may help most at 14_month) are baked into the spec acceptance criteria.

**File deletion discipline (Development Standards)** — PASS. Spec produces only new files; existing `GatedABMILHead`, existing configs, and existing result directories are preserved. No deletes planned.

**Verdict**: All gates PASS. No Constitution violations to justify in Complexity Tracking.

## Project Structure

### Documentation (this feature)

```text
specs/014-mil-extensions-attention-and-layers/
├── plan.md              # This file
├── spec.md              # Feature specification (already written)
├── research.md          # Phase 0 — research notes resolving design questions
├── data-model.md        # Phase 1 — entity / class / config schemas
└── quickstart.md        # Phase 1 — copy-paste recipe for running the three US
```

`contracts/` directory is intentionally omitted: this is an internal research pipeline, no external API surface. Per the plan-template guidance ("Skip if project is purely internal"), interface contracts are documented inline in `data-model.md` (head-class signatures, config keys).

### Source Code (repository root)

This feature touches a single existing module (`mil/`) and adds new YAML configs. No new top-level packages.

```text
mil/
├── mil_model.py            # MODIFIED: BackboneExtractor.layer_aggregation; new ACMILHead class; head factory
├── mil_train.py            # MODIFIED: optional diversity-loss term in training loop; layer_weights.json dump
├── mil_evaluate.py         # MODIFIED: branch_weights.json dump for ACMIL eval; otherwise unchanged
├── mil_dataset.py          # UNCHANGED
├── mil_utils.py            # UNCHANGED
├── eval_weak_diarization.py # MODIFIED (or wrapper): per-branch attention alignment for ACMIL
├── configs/
│   ├── wavlm_mil_layersum.yaml          # NEW (US1)
│   ├── whisper_mil_layersum.yaml        # NEW (US1)
│   ├── hubert_large_mil_layersum.yaml   # NEW (US1)
│   ├── wavlm_mil_child_adapted.yaml     # EXISTS (US2 — re-uses)
│   ├── wavlm_mil_child_adapted_layersum.yaml  # NEW (US2 conditional, FR-010)
│   ├── wavlm_mil_acmil.yaml             # NEW (US3)
│   └── whisper_mil_acmil.yaml           # NEW (US3)
├── slurm/
│   ├── train_mil.sh        # UNCHANGED — already config-parameterized
│   └── eval_mil.sh         # MODIFIED: enumerate new run dirs (or generalize via glob)
└── mil_results/
    ├── wavlm_mil_layersum/                    # NEW output dir (US1)
    ├── whisper_mil_layersum/                  # NEW output dir (US1)
    ├── hubert_large_mil_layersum/             # NEW output dir (US1)
    ├── wavlm_mil_child_adapted/               # NEW output dir (US2)
    ├── wavlm_mil_child_adapted_layersum/      # NEW output dir (US2 conditional)
    ├── wavlm_mil_acmil/                       # NEW output dir (US3)
    └── whisper_mil_acmil/                     # NEW output dir (US3)
```

**Structure Decision**: Single-project research-pipeline layout, in line with all prior MIL specs (002/004/005). Edits are confined to `mil/mil_model.py`, `mil/mil_train.py`, `mil/mil_evaluate.py`, `mil/eval_weak_diarization.py`, plus new YAML configs and SLURM job invocations. Results land in canonical `mil/mil_results/` per CLAUDE.md.

## Complexity Tracking

> No Constitution violations to justify. Table omitted.

## Phase 0 Output

See `research.md` for the resolved design decisions (layer-aggregation skip-first behavior; ACMIL diversity-loss form; STKIM annealing schedule; child-adapted checkpoint loading path).

## Phase 1 Output

See `data-model.md` for class/config schemas and `quickstart.md` for the end-to-end run recipe.
