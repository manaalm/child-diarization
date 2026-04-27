# Research: MIL Extensions

## Decision 1: Noisy-OR Aggregation Formula

**Decision**: Implement noisy-OR as probabilistic bag-level prediction using log-space product for numerical stability.

**Formula**: Each instance gets a scalar logit via a linear head. The bag probability is `1 - ∏(1 - σ(logit_k))` over valid (non-masked) instances. In log-space: `log_bag_complement = Σ log(1 - σ(logit_k))`, then `bag_logit = logaddexp(0, log_bag_complement)`. Masked (padding) positions contribute neutral probability 1.0 (i.e., `log(1-p) = 0`).

**Rationale**: Noisy-OR assumes instances are independent evidence for the bag label — if any one instance is strongly positive, the bag is positive. This matches the hypothesis that for 14-month clips with sparse babbles, a single high-confidence child segment should be sufficient evidence.

**Alternatives considered**: Simple max-pooling of instance probabilities (less principled than noisy-OR for the independence assumption).

---

## Decision 2: Top-k Aggregation

**Decision**: Score each instance with a per-instance linear head, select top-k by score (masking padding with -inf), mean-pool the embeddings of those k instances, then apply a final linear head.

**k-clamping**: `k_actual = min(k_config, n_valid)` where `n_valid = mask.sum()`. Default `k=3` is appropriate for average bag sizes of 5–30.

**Rationale**: Top-k captures the hypothesis that the strongest positive instances drive the prediction — useful when the child is vocalizing in only a subset of segments. Mean-pooling selected instances is more stable than scoring the pooled embedding.

**Alternatives considered**: Soft-max-weighted top-k selection (blurs the discrete selection; harder to interpret).

---

## Decision 3: Transformer MIL Architecture

**Decision**: 2-layer transformer encoder (4 heads, FFN dim=1536, attention dropout=0.2, FFN dropout=0.3), with a learned [CLS] token prepended to the sorted bag, pre-norm residuals, and CLS-token output fed to a linear classifier.

**Positional encoding**: Learned positional embeddings (not sinusoidal). Bag sizes 5–30 are too short for sinusoidal PE to be meaningful; learned PE adapts to the specific temporal structure of the data.

**Segment ordering**: Segments are sorted by start time before positional encoding, so position reflects temporal order within the clip.

**Regularization**: dropout=0.3 in FFN layers, weight decay=0.01, early stopping on val loss.

**Rationale**: 2 layers + 4 heads is the minimal configuration that allows cross-instance attention while staying below the overfitting risk threshold at ~1,500 training samples. The [CLS] token pattern allows the model to learn task-specific aggregation implicitly.

**Alternatives considered**: Mean-pool over output tokens (reduces expressiveness; CLS is better for classification). 3+ layers (overfits at 1,500 samples). Sinusoidal PE (requires fixed maximum sequence length and doesn't adapt).

---

## Decision 4: Weak Diarization Evaluation Metric

**Decision**: For each test segment, compute ground-truth child-speech fraction = (duration of child speaker segments that overlap with [start, end]) / (end - start). Then report:
- Pearson correlation between attention weights and GT fraction
- Spearman correlation (rank-based, robust to non-linearity)
- AUROC treating GT child fraction ≥ 0.5 as binary positive label and attention weight as ranking score

Results stratified by `timepoint` (`14_month` vs. `36_month`).

**Child speaker identification in RTTMs**: For RTTM files from BabAR-VTC and USC-SAIL, speaker labels include `KCHI`, `CHI`, or `CHILD` for child speech. For Pyannote RTTMs, speaker labels are anonymous (`SPEAKER_00`, etc.) — use the ground-truth RTTM from the same dataset (usc_sail or vtc) as reference rather than the Pyannote RTTM for this evaluation.

**Rationale**: Pearson/Spearman measure how well the model's internal scoring correlates with actual child speech density — a soft evaluation. AUROC measures ranking quality as a binary classifier. Stratifying by age band tests the Zhu et al. 2021 extension hypothesis.

**Alternatives considered**: Frame-level precision/recall (requires converting segment-level attention to frame-level scores, introducing an arbitrary threshold; the correlation approach is more principled for a ranking evaluation).

---

## Decision 5: Age-Band Stratification Approach

**Decision**: Filter test set rows by the `timepoint` column in `whisper-modeling/seen_child_splits/test.csv` (values: `14_month`, `36_month`). Run inference on the full test set once and post-filter predictions by timepoint — no retraining needed.

**Implementation**: After inference, the prediction DataFrames already include per-clip metadata. Join with split CSV on audio path to get `timepoint`, then compute metrics on each subset.

**Rationale**: The test split has 234 clips at 14_month and 207 clips at 36_month — large enough for reliable per-band AUROC estimation.

---

## Decision 6: Scope Boundaries

- **US3 (Transformer MIL)** runs as part of the same sweep config after US1/US2 aggregators are implemented. It is not a separate sweep — all aggregators share the same embedding cache and training loop.
- **US5 (TinyVox scorer)** and **US6 (end-to-end)** are explicitly gated; no scaffolding is built for them in this implementation cycle.
- The existing 16-config results are not re-run; new aggregators add rows to `all_configs.json` incrementally via resume-safe logic.
- No new RTTM caches or embedding caches are required — all new aggregators share `mil/seg_embedding_cache/`.
