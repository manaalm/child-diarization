# Implementation Plan: Segment-Instance MIL with Attention Aggregation

**Branch**: `004-segment-instance-mil` | **Date**: 2026-04-23 | **Spec**: [spec.md](spec.md)

## Summary

Treat diarizer-proposed speech segments as bag instances for MIL. For each clip, pool frozen WavLM-base+ frame embeddings over each segment span to produce a per-segment embedding; then train one of four aggregator heads (mean, max, attention, gated-attention) over the variable-length bag to predict child presence at the clip level. Sweep 4 frontends × 4 aggregators = 16 configurations, all using the same seen-child split and evaluation protocol as ECAPA enrollment, producing a directly comparable results table for the thesis.

---

## Technical Context

**Language/Version**: Python 3.11 (conda `child-vocalizations` env)
**Primary Dependencies**: PyTorch, torchaudio, transformers (`wavlm-base-plus`), speechbrain (for ECAPA comparison), pandas, scikit-learn, PyYAML
**Storage**: Disk-based segment embedding cache at `mil/seg_embedding_cache/`; results JSON/CSV at `mil/mil_results/seg_mil/`
**Testing**: Manual smoke-test with a single (frontend, aggregator) pair; full 16-config run on SLURM
**Target Platform**: Linux cluster (SLURM), GPU required for WavLM forward pass
**Project Type**: Research training/evaluation script module
**Performance Goals**: Full 16-config sweep completes within 24 hours on a single GPU (SC-003)
**Constraints**: Frozen backbone (no fine-tuning); segment embedding cache must fit on scratch disk (estimated <5 GB for 2183 clips × 4 frontends × 1024-dim)
**Scale/Scope**: 2183 clips (1311 train / 431 val / 441 test), 4 frontends, 4 aggregators, 16 result entries

---

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Principle | Status | Notes |
|-----------|--------|-------|
| I — Reproducibility & Environment | ✅ PASS | seed=42 in sweep config; uv env reused; config.json committed per run |
| II — Data Integrity & Split Discipline | ✅ PASS | Uses `seen_child_splits/` only; no test-set tuning; threshold tuned on val |
| III — Baseline-First Development | ✅ PASS | Directly compared against ECAPA enrollment (same protocol); MeanAgg and MaxAgg serve as non-trainable baselines within the matrix |
| IV — Rigorous Evaluation & Metrics | ✅ PASS | F1+precision+recall+AUROC+AUPRC reported; per-timepoint breakdown required; threshold reported |
| V — Mandatory Ablations & Error Analysis | ✅ PASS | All 16 cells are the ablation; attention weights serve as the interpretability/error-analysis output |
| VI — Thesis Synchronization | ✅ PASS | Results committed to `mil/mil_results/seg_mil/`; `all_configs.json` wired into `thesis_tables.yaml` |
| VII — Documentation & Honest Reporting | ✅ PASS | New scripts require docstrings; CLAUDE.md updated; limitations noted (seen-child split may overestimate real-world performance) |
| File deletion discipline | ✅ PASS | No existing files deleted; new files added only |

**Post-design re-check**: All gates pass. No violations. No Complexity Tracking entries required.

---

## Project Structure

### Documentation (this feature)

```text
specs/004-segment-instance-mil/
├── plan.md              ← this file
├── research.md          ← Phase 0: all design decisions
├── data-model.md        ← Phase 1: entities and file layout
├── quickstart.md        ← Phase 1: how to run
├── checklists/
│   └── requirements.md  ← spec quality checklist
└── tasks.md             ← Phase 2 output (via /speckit.tasks)
```

### Source Code (repository root)

```text
mil/
├── mil_dataset.py          (existing — frame-window MIL dataset)
├── mil_model.py            (existing — GatedABMILHead, BackboneExtractor, MILModel)
├── mil_train.py            (existing — frame-window training entry point)
├── mil_utils.py            (existing — compute_metrics, tune_threshold, per_timepoint_metrics)
├── mil_evaluate.py         (existing)
├── mil_age_stratified.py   (existing)
│
├── seg_dataset.py          NEW — SegmentBagDataset
│                               Reads RTTM cache for a given frontend; loads audio;
│                               runs frozen WavLM forward pass; mean-pools per segment.
│                               Returns List[SegmentBag].
│
├── seg_embedding_cache.py  NEW — SegmentEmbeddingCache
│                               Disk-backed cache keyed on MD5(audio_path|start|end).
│                               Stores numpy arrays as .npy files under
│                               mil/seg_embedding_cache/{frontend_name}/.
│
├── seg_model.py            NEW — Aggregator heads
│                               MeanAgg, MaxAgg, AttnAgg (standard ABMIL),
│                               GatedAttnAgg (wraps existing GatedABMILHead).
│                               All expose .forward(bag_tensor, mask) → (score, weights).
│
├── seg_train.py            NEW — Sweep training entry point
│                               Reads seg_mil_sweep.yaml; iterates 16 configs;
│                               trains each aggregator head; evaluates on val/test;
│                               writes per-config results and all_configs.json.
│
├── configs/
│   ├── wavlm_mil.yaml      (existing)
│   ├── whisper_mil.yaml    (existing)
│   └── seg_mil_sweep.yaml  NEW — sweep config
│                               Lists frontends, aggregators, rttm_cache_paths,
│                               encoder name, training HPs, seed, output dir.
│
├── slurm/
│   ├── (existing scripts)
│   └── seg_mil_sweep.sh    NEW — SLURM submission script for full sweep
│
└── mil_results/
    └── seg_mil/            NEW — results root
        ├── all_configs.json
        └── {frontend}_{aggregator}/   (16 subdirectories)
            ├── config.json
            ├── val_predictions.csv
            ├── test_predictions.csv
            ├── val_metrics.json
            └── test_metrics.json

evaluation/
└── configs/
    └── thesis_tables.yaml  MODIFY — add table_segment_mil entry

CLAUDE.md                   MODIFY — add seg_mil/ to Results Storage and Key Commands
```

**Structure Decision**: All new code is co-located with the existing `mil/` module, following its established naming pattern (`mil_*.py` → `seg_*.py`). No new top-level directory is created. The `mil_results/seg_mil/` subdirectory mirrors the enrollment run folder convention used across the project.

---

## Implementation Phases

### Phase A: Segment Embedding Extraction

**Files**: `seg_embedding_cache.py`, `seg_dataset.py`

- `SegmentEmbeddingCache`: wraps a directory; `get(audio_path, start, end)` → ndarray or None; `put(...)` → saves .npy. Key is MD5 of `{audio_path}|{start:.4f}|{end:.4f}`. Thread-safe for single-process use.
- `SegmentBagDataset`: accepts a frontend name, RTTM cache dir, split DataFrame, and embedding cache. For each clip: load RTTM file → segment list → for each segment: check cache miss → load audio slice → WavLM forward pass → mean pool → store in cache → assemble bag tensor (K × D, zero-padded) + mask (K,). Returns `(bag_tensor, mask, label, metadata_dict)`.
- `precompute_embeddings(frontend, rttm_dir, df, cache, encoder)`: standalone function that pre-fills the cache for all clips in df; used by `--precompute-only` flag.

**Key design notes**:
- WavLM model is loaded once and shared across all bags in a dataset; not reloaded per clip
- Audio slicing: `torchaudio.load` with frame offsets; resample to 16kHz if needed (mirrors `dataset_classes/preprocess.py`)
- WavLM frame stride is 20ms; segment shorter than 1 frame → zero-vector embedding (logged as warning)
- Empty bags (K=0) return a (1 × D) all-zeros tensor with mask=0 so the aggregator can handle them uniformly

---

### Phase B: Aggregator Heads

**File**: `seg_model.py`

Four aggregator classes, each taking `(bag: Tensor[K, D], mask: Tensor[K]) → (score: Tensor[1], weights: Tensor[K] or None)`:

| Class | Trainable Params | Attention Output |
|-------|-----------------|-----------------|
| `MeanAgg(embed_dim, hidden_dim)` | linear head only | None (uniform) |
| `MaxAgg(embed_dim, hidden_dim)` | linear head only | None (argmax) |
| `AttnAgg(embed_dim, attn_dim, hidden_dim)` | V, U matrices + linear head | softmax weights |
| `GatedAttnAgg(embed_dim, attn_dim, hidden_dim)` | V, U, V_gate, U_gate + linear head | gated softmax |

`GatedAttnAgg` wraps the existing `GatedABMILHead` from `mil/mil_model.py` with a shared linear head.

All heads output raw logit (for BCEWithLogitsLoss during training) and sigmoid probability at inference. Masking: masked instances contribute zero to the attention numerator.

---

### Phase C: Training Sweep

**File**: `seg_train.py`

```
for frontend in [usc_sail, pyannote, babar_vtc, vbx]:
    precompute embeddings (skip if cache exists)
    for aggregator in [mean, max, attention, gated_attention]:
        build SegmentBagDataset for train/val/test
        instantiate AggregatorHead
        train with Adam, BCEWithLogitsLoss, 20 epochs, early stop on val AUROC
        evaluate on val → tune threshold → evaluate on test
        save results to mil/mil_results/seg_mil/{frontend}_{aggregator}/
append row to all_configs.json
```

- Config loaded from `seg_mil_sweep.yaml`; config dict committed to `config.json` alongside results
- Logging: one line per epoch (loss, val AUROC); one line per completed config
- All 16 configs run in a single script invocation; partial runs are resumable (skip configs whose results dir already has `test_metrics.json`)

---

### Phase D: Evaluation & Thesis Integration

**Files**: `mil/mil_utils.py` (reused), `evaluation/configs/thesis_tables.yaml` (modified), `CLAUDE.md` (modified)

- `mil_utils.py` `compute_metrics()` and `per_timepoint_metrics()` are reused without modification
- Per-timepoint breakdown (14_month, 36_month) written to `val_metrics_by_timepoint.json` per config
- `thesis_tables.yaml`: add `table_segment_mil` with rows sourced from `all_configs.json` using `key_map` to align field names to thesis table columns

---

## Complexity Tracking

No constitution violations to justify.
