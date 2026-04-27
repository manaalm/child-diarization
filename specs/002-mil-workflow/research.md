# Research: Multiple Instance Learning Workflow

**Status**: Complete — all decisions resolved
**Generated**: 2026-04-23

---

## Decision 1: MIL Aggregation Architecture

**Decision**: Gated Attention-Based MIL (ABMIL), following Ilse et al. 2018.

**Rationale**:
- Standard ABMIL assigns a scalar attention weight `a_k` to each instance (window)
  via a two-layer MLP → softmax; the bag embedding is `z = Σ a_k h_k`.
- Gated ABMIL adds a sigmoid gate for improved stability on imbalanced bags:
  `a_k ∝ tanh(V h_k) ⊙ σ(U h_k)` — prevents attention collapse when
  most windows contain no child speech.
- Chosen over max-pooling MIL because the thesis requires interpretable attention
  maps (which windows the model attends to) for the error analysis section.
- Chosen over transformer-based aggregation to keep the parameter count low
  relative to the frozen backbone (MIL head: ~200k parameters vs. backbone ~90M).

**Alternatives considered**:
- Max-pooling MIL: simpler but not interpretable; discards the full attention
  distribution and cannot characterize failure modes.
- Mean-pooling MIL: equivalent to existing attentive-pooling baselines in
  `baseline_encoders.py`; not novel.
- Transformer aggregator: high parameter overhead; risk of overfitting on 2183 clips.

---

## Decision 2: Backbone Selection

**Decision**: Two variants, each frozen:
- **WavLM-base+** (`microsoft/wavlm-base-plus`, 768-dim hidden, 24 encoder layers)
- **Whisper-small** (`openai/whisper-small`, 768-dim hidden, 12 encoder layers)

**Rationale**:
- These two backbones are the strongest in the existing cross-child baselines
  (Whisper-small F1=0.884/0.882 mean/attn; WavLM-base+ F1=0.870/0.874).
- Using the same pretrained weights ensures the backbone comparison is controlled —
  the only variable is the MIL aggregation vs. single-vector pooling.
- Frozen backbone: matches the baseline protocol (freeze_backbone=True, 0 unfrozen
  layers); prevents overfitting given the small clip count.

**Alternatives considered**:
- Whisper-base: smaller; lower baseline performance; not worth an additional run slot.
- Fine-tuning backbone jointly: risks overfitting; not comparable to frozen baselines.
- ECAPA-TDNN: already used as the enrollment speaker encoder — would conflate the
  MIL classification role with the embedding role.

---

## Decision 3: Instance (Window) Definition

**Decision**: Fixed-length 2-second windows, 1-second stride (50% overlap).

**Rationale**:
- 2 s captures 2–4 child vocalization events (mean child utterance duration ~0.5 s
  in SAILS data) while keeping the per-window feature dimensionality tractable.
- 1 s stride ensures ~90% of utterances appear in at least one complete window;
  utterances at window boundaries appear in the overlapping window.
- At 16kHz, 2 s = 32,000 samples; Whisper processes in 480,000-sample (30 s)
  chunks but feature extraction uses only the first 2 s with 20 ms frame stride
  → 100 frames per window, manageable for batching.
- Window count per clip: a 30 s clip produces 29 windows at 2 s/1 s stride.
  Clips shorter than 2 s are padded to 2 s and treated as a single instance.

**Alternatives considered**:
- 1 s windows, 0.5 s stride: higher instance count per bag (~60 per 30 s clip);
  each window too short to contain more than 1–2 child utterances; attention harder
  to learn.
- 5 s windows: lower instance count (~6 per 30 s clip); insufficient bag size for
  stable attention learning; too coarse for interpretable error analysis.
- Diarizer-detected segments as instances: hybrid approach that couples MIL to a
  diarization front-end; violates the goal of a diarization-free baseline.

---

## Decision 4: Instance Feature Extraction

**Decision**: Mean-pool frame-level backbone features within each window → single
dense vector per instance.

**Rationale**:
- Mean-pooling within a window is fast, parameter-free, and consistent with the
  `MeanPooling` class already used in `baseline_encoders.py`.
- The ABMIL attention layer handles cross-window aggregation; intra-window pooling
  does not need to be learned separately.
- For Whisper-small: encoder outputs at the second-to-last layer (standard practice
  for feature extraction) → mean over time → 768-dim instance vector.
- For WavLM-base+: hidden states from the final transformer layer → mean over
  time → 768-dim instance vector.

---

## Decision 5: Environment

**Decision**: Use the existing `child-vocalizations` conda environment (same as
`baselines/baseline_encoders.py`). No new uv-managed sub-environment created.

**Rationale**:
- MIL requires identical packages to the baselines: `torch`, `torchaudio`,
  `transformers` (WavLM + Whisper), `scikit-learn`, `pandas`, `numpy`.
- Constitution Principle I requires uv for NEW subsystems; MIL is an extension of
  the existing baselines subsystem and shares its dependency set.
- Adding a separate uv env for MIL would require redownloading the 3+ GB backbone
  checkpoints; the conda env already has them cached.
- SLURM scripts activate `child-vocalizations` (same as whisper.sh in baselines/).

---

## Decision 6: Training Protocol

**Decision**: Binary cross-entropy, Adam, lr=1e-3 (MIL head only), frozen backbone,
20 epochs max, early stopping on val F1 (patience=5), threshold tuned on val.

**Rationale**:
- Matches baseline training protocol as closely as possible for comparability.
- Head-only training: 200k MIL parameters vs. 90M backbone — backbone LR at 0 is
  most analogous to the frozen-backbone baselines.
- Threshold tuning on val (sweep 0.05–0.95): same as unified.py sweep logic, ensures
  fair comparison with enrollment-based systems.
- Optional class-weighting (`pos_weight = n_neg / n_pos`) for imbalanced training
  sets, configurable per run.

---

## Decision 7: Result Folder Location

**Decision**: `mil/mil_results/{variant_name}/` with the same file set as
enrollment-based systems:
```
config.json
training_history.csv
val_metrics_tuned.json
test_metrics_tuned.json
test_predictions.csv
test_metrics_by_timepoint.csv
```

**Rationale**:
- Identical file names and column schemas to `unified.py` outputs allow
  `evaluation/aggregate_thesis_tables.py` to read MIL results with zero code changes
  (FR-008 / SC-002).
- `test_predictions.csv` columns: `audio_path, child_id, timepoint_norm, label,
  score, prediction` — same as enrollment CSVs.
- `test_metrics_by_timepoint.csv` columns: `timepoint, f1, precision, recall,
  auroc, auprc, n` — same as `per_timepoint_metrics()` in `unified.py`.

---

## Baseline Comparison Protocol

MIL runs on the **seen-child split** (`whisper-modeling/seen_child_splits/`) and is
directly comparable to the enrollment-based systems (USC-SAIL, BabAR, VTC, VBx).
The cross-child encoder baselines in `baselines/baseline_results/` use a different
split paradigm and serve as reference context only — they are NOT a fair head-to-head
comparison with MIL.

This satisfies Constitution Principle III: MIL is compared against the strongest
systems on the same evaluation protocol (seen-child, same metrics, same split).
