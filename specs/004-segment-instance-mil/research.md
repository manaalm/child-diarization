# Research: Segment-Instance MIL

**Feature**: 004-segment-instance-mil
**Date**: 2026-04-23

---

## Decision 1: Backbone Encoder

**Decision**: WavLM-base+ (`microsoft/wavlm-base-plus`), frozen, as the sole encoder for the 16-cell matrix.

**Rationale**: The baseline encoder experiments (`baselines/baseline_results/`) show WavLM-base+ consistently outperforms Whisper-small on this dataset for mean and attention pooling. Using one encoder for the primary sweep keeps the matrix clean (4 frontends × 4 aggregators = 16 cells, not 32). Whisper-small is documented as a secondary ablation for future work.

**Alternatives considered**:
- Whisper-small: viable, already used by USC-SAIL. Excluded from primary sweep to keep cell count at 16.
- Both encoders as a 32-cell matrix: would dilute the frontend-vs-aggregator story; kept as future extension.

---

## Decision 2: Within-Segment Instance Pooling

**Decision**: Mean pooling over encoder frame vectors within each segment span.

**Rationale**: Mean pooling adds zero trainable parameters, making the instance representation a pure function of the frozen backbone. This isolates the variable being studied (MIL aggregator quality and diarizer quality) from within-segment pooling choices. Attention-based within-segment pooling is documented as a secondary ablation.

**Alternatives considered**:
- Attentive within-segment pooling: adds parameters to the instance extractor; conflates two design axes.
- Max pooling within segment: sensitive to noise spikes; less stable for short segments.

---

## Decision 3: Environment

**Decision**: Reuse the conda `child-vocalizations` environment (Python 3.11) already used by `pyannote/unified.py`. No new `uv` environment needed because the segment MIL training only requires packages already installed (torch, torchaudio, speechbrain for ECAPA, transformers for WavLM, sklearn).

**Rationale**: The `mil/` module already runs in the same environment. Adding a new environment for a module that has identical dependencies violates constitution Principle I (reproducibility) without benefit.

**Alternatives considered**:
- Separate `uv` env: unnecessary — no dependency conflicts exist; would add setup friction.

---

## Decision 4: The 4 MIL Aggregators

**Decision**: Implement four aggregators over the bag of K per-segment embeddings:
1. **MeanAgg**: unweighted mean → linear head (no trainable params in aggregator itself)
2. **MaxAgg**: element-wise max → linear head
3. **AttnAgg** (standard ABMIL, Ilse et al. 2018): `V = tanh(W·h); a = softmax(W_a·V); z = Σ a_k h_k`
4. **GatedAttnAgg** (GABMIL): extends AttnAgg with a gating term; already implemented as `GatedABMILHead` in `mil/mil_model.py`

**Rationale**: MeanAgg and MaxAgg are non-trainable aggregators; they serve as the ablation baseline within this matrix (equivalent to no MIL learning, just pooling). AttnAgg and GatedAttnAgg learn attention over instances. Gated attention is the ABMIL variant shown in the original paper to be more selective. All four are standard MIL variants.

**Alternatives considered**:
- TransMIL (transformer over instances): more complex, not yet motivated by a gap in AttnAgg performance.
- CLAM: adds a clustering auxiliary loss; adds complexity not justified by current baselines.

---

## Decision 5: Empty-Bag Handling

**Decision**: Clips where the diarizer found zero segments receive a fixed predicted score of `0.0` (child absent) and are included in evaluation.

**Rationale**: Empty bags occur when a diarizer assigns no speech to any speaker in the clip. For enrollment purposes this means there is no evidence of the child, so the model should predict absence. Imputing embeddings would bias the model. Including these clips in evaluation is required for a fair comparison with ECAPA enrollment (which also returns score 0 for clips with no segments).

**Alternatives considered**:
- Excluding empty-bag clips from evaluation: inflates metrics artificially; inconsistent with ECAPA protocol.
- Imputing a zero-vector instance: misleads the attention mechanism into treating silence as meaningful content.

---

## Decision 6: Training Protocol

**Decision**:
- Binary cross-entropy loss on clip labels
- Adam optimizer, LR=1e-3 for the MIL head (backbone frozen, no backbone LR)
- 20 epochs with early stopping on val AUROC (patience=5)
- Batch size=32 clips
- Threshold tuned on val split (maximize F1), applied to test split
- Seed=42 for all runs

**Rationale**: Matches the training protocol used in `mil/configs/wavlm_mil.yaml`. Frozen backbone means only the aggregator head trains. Early stopping on AUROC is preferred over loss because class imbalance in the dataset makes loss a less reliable stopping criterion.

**Alternatives considered**:
- Fine-tuning the backbone: excluded per spec Assumptions; adds cost and conflates backbone quality with aggregator quality.
- Fixed number of epochs: less robust; early stopping protects against overfitting on small-head models.

---

## Decision 7: Segment Embedding Cache Location

**Decision**: Cache per-segment WavLM embeddings at `mil/seg_embedding_cache/{frontend_name}/`. Key: MD5 of `{audio_path}|{start:.4f}|{end:.4f}`. Each frontend gets its own subdirectory since segment boundaries differ per frontend.

**Rationale**: Embedding extraction (frozen WavLM forward pass over each segment) is the most expensive step. Caching allows all 4 aggregator variants for a given frontend to share the same precomputed embeddings, cutting compute by 4×. Cache is keyed on segment boundaries so it invalidates automatically if the RTTM changes.

**Alternatives considered**:
- Clip-level embedding cache (entire audio): would not capture segment-level pooling correctly.
- In-memory cache within a single run: doesn't persist across training runs for different aggregators.

---

## Decision 8: Results Layout and Thesis Table Integration

**Decision**:
- Per-configuration results at `mil/mil_results/seg_mil/{frontend}_{aggregator}/` (16 subdirectories)
- Summary at `mil/mil_results/seg_mil/all_configs.json`
- Add `table_segment_mil` entry to `evaluation/configs/thesis_tables.yaml` sourcing from `all_configs.json`

**Rationale**: Mirrors the existing enrollment run folder convention (`pyannote/pyannote_enrollment_runs/`, etc.). One summary file allows the thesis table generator to produce the full 16-row comparison table without reading 16 subdirectories.

**Alternatives considered**:
- A single flat results CSV: harder to query per-configuration; doesn't match existing folder convention.
- Storing results inside `pyannote/`: wrong — segment MIL is a training-based method, not a diarization frontend output.

---

## Resolved NEEDS CLARIFICATION Items

None were present in the spec. All design decisions were resolved above.
