# Feature Specification: MIL Extensions — Weighted-Layer-Sum, Child-Adapted Backbone, ACMIL

**Feature Branch**: `014-mil-extensions-attention-and-layers`
**Created**: 2026-04-29
**Status**: Draft

## Overview

Six MIL extensions chosen from the recent-literature review (`related_works.MD`) and the unfilled gaps in the existing MIL stack (`mil/`, `specs/002-mil-workflow/`, `specs/004-segment-instance-mil/`, `specs/005-mil-extensions/`). The first three are isolated drop-in modifications to the frame-window MIL path; the latter three are deeper architectural changes spanning frame-window MIL (US4 TS-MIL) and segment-instance MIL (US5 DSMIL, US6 adaptive pooling). Each is independently evaluable and shares the existing SLURM and metrics tooling.

The six extensions:

- **US1 (P1) — Learnable Weighted-Layer-Sum**: Replace the single-layer (`hidden_states[-1]`) feature read in `BackboneExtractor` with a SUPERB/DiariZen-style learnable softmax over all transformer layers. Motivation: Pasad/Shi/Livescu (ICASSP 2023) place speaker-discriminative information in WavLM-Large layers 6–12 / HuBERT-base layers 4–8, not the final layer; DiariZen, pyannote 3.x, and WeSpeaker all combine layers via a learnable scalar weighted sum.
- **US2 (P1) — Child-Adapted WavLM Wired Into MIL**: The 50k-step child-adapted WavLM pretraining (`synth/slurm/run_wavlm_pretrain.sh`) and the matching MIL config (`mil/configs/wavlm_mil_child_adapted.yaml`) already exist but the resulting MIL run has not been integrated into `results_summary.md`. Run the pipeline end-to-end and produce a comparable test row.
- **US3 (P2) — ACMIL Drop-In Head**: Replace `GatedABMILHead` in `mil/mil_model.py` with a configurable ACMIL head (Zhang et al., ECCV 2024, https://github.com/dazhangyu123/ACMIL). Two independently switchable components: **Multiple Branch Attention (MBA)** with a diversity regularizer, and **Stochastic Top-K Instance Masking (STKIM)**. Motivation: small-data attention concentration is the most likely failure mode behind the gated-ABMIL plateau (test AUROC 0.853 / 0.771 for Whisper / WavLM frame-window) and the transformer-aggregator collapse documented in `specs/005-mil-extensions/`.
- **US4 (P2) — TS-MIL: Target-Speaker Conditional MIL Head**: Inject the per-(child, timepoint) ECAPA prototype (already built by all enrollment frontends — see `pyannote/unified.py:559` `build_child_prototypes`) into the MIL head, conditioning attention on the target child's voice. Two flavors: **concat** (project the 192-d prototype, concatenate to each instance embedding before attention) and **FiLM** (`gamma, beta = MLP(prototype)`; `h_k = gamma * h_k + beta`). Motivation: this is the TS-VAD-style enrollment recipe (Medennikov ICASSP 2020, Bertamini *Res Dev Disabil* 2025) expressed inside the MIL aggregation step — none of the existing MIL configs use the target-child prototype, even though the enrollment infrastructure already builds one per (child, timepoint) pair.
- **US5 (P2) — DSMIL Dual-Stream Aggregator (segment-instance path)**: Add `DSMILAgg` to `mil/seg_model.py:build_aggregator()`. DSMIL (Li et al. CVPR 2021) keeps two streams: (a) a max-instance stream identifying the highest-scored "critical instance" with a max pool head, and (b) an attention stream where each non-critical instance's attention weight is computed as a function of its *distance* (in feature space) to the critical instance. Both streams produce a bag prediction, and the loss is the mean of the two BCE losses. Motivation: instance-level co-training adds an auxiliary supervised signal — currently the loss is bag-only BCE; this is the analogue of the mean-teacher / self-distillation moves that DCASE 2024 SED systems leaned on.
- **US6 (P2) — Adaptive / Learnable Pooling Operators (segment-instance path)**: Add three new aggregators to `mil/seg_model.py:build_aggregator()`: **AutoPool** (McFee, Salamon, Bello, ICASSP 2018; learnable softmax temperature interpolating between mean and max), **ExpSoftmaxPool** (exponentially-weighted softmax pooling — Wang et al. arXiv:1810.09050 §III.E baseline), and **GMAP** (Gated Multi-Head Attention Pooling — Hong et al.; multi-head attention with learned query gating). Motivation: the spec-005 sweep included only fixed pooling functions (mean, max, attention, gated, noisy-OR, top-k, transformer); adaptive operators consistently outperform fixed pooling on weakly-supervised SED benchmarks.

---

## User Scenarios & Testing *(mandatory)*

### User Story 1 — Learnable Weighted-Layer-Sum (Priority: P1)

A researcher wants to know whether the frame-window MIL backbone (`BackboneExtractor` in `mil/mil_model.py`) is leaving information on the table by reading only `hidden_states[self.layer]` (default `-1`). The hypothesis is that a learnable softmax over all transformer layers will outperform the last-layer-only read for both WavLM-Base+ and HuBERT-Large, because speaker-discriminative information peaks in middle layers (Pasad et al. 2023).

**Why this priority**: Lowest implementation cost, strongest literature support, and the change is local to `BackboneExtractor`. DiariZen reports learnable layer-sum as a default ingredient. The existing MIL training loop, dataset, and SLURM scripts do not need to change.

**Independent Test**: Train two configs end-to-end on the seen-child split — `wavlm_mil` (last layer; existing baseline) and a new `wavlm_mil_layersum`. Pass when the new config writes `test_metrics_tuned.json` to `mil/mil_results/wavlm_mil_layersum/` with valid F1/AUROC/AUPRC and the val-tuned threshold is applied to test.

**Acceptance Scenarios**:

1. **Given** an updated `BackboneExtractor` with a `layer_aggregation: weighted_sum` mode and a new `wavlm_mil_layersum.yaml` config, **When** training runs, **Then** the learned per-layer softmax weights are saved alongside `best_checkpoint.pt` and inspectable as a `layer_weights.json` file in the run directory.
2. **Given** the new config completes, **When** test AUROC is compared to the `wavlm_mil` baseline (0.771 seen-child / 0.690 cross-child), **Then** the delta is reported in `results_summary.md`. The change is considered successful if delta_AUROC > 0 on the seen-child split, and the cross-child run is reported regardless of direction.
3. **Given** an analogous `whisper_mil_layersum.yaml` and `hubert_large_mil_layersum.yaml`, **When** all three layer-sum variants train, **Then** layer weights are inspected to identify which layers carry the strongest signal per backbone.
4. **Given** existing `wavlm_mil` checkpoints, **When** the new variant reads from them, **Then** the new code is backwards-compatible: configs without `layer_aggregation` continue to use the single-layer behavior unchanged.

---

### User Story 2 — Child-Adapted WavLM Wired Into Frame-Window MIL (Priority: P1)

A researcher wants the test-row that closes the loop on spec-009 US3: continued masked-speech-unit pretraining of WavLM-Base+ on 73k Providence/TinyVox child segments produced a checkpoint at `synth_results/child_wavlm_checkpoint/step_50000/`, and `mil/configs/wavlm_mil_child_adapted.yaml` already points `backbone_path` to that checkpoint. But `results_summary.md` does not yet have the corresponding child-adapted MIL row, so the question "does child-adapted SSL pretraining help downstream MIL?" is unresolved in this codebase. Three priors say it should help: Bertamini 2025 (30 s of in-domain adaptation gives big wins), Lahiri/Feng/Narayanan 2023–24 USC-SAIL WavLM child adaptation, and Al Futaisi et al. 2025 (task-specific pretraining can beat generic SSL on small child-speech problems).

**Why this priority**: All artifacts already exist; the only missing step is the train+eval run and integration into the results table. Cheapest possible win if the prior holds. If the prior fails, the negative result itself is publishable (parallel to the TinyVox-augmentation negative result already in `CLAUDE.md`).

**Independent Test**: Submit the existing SLURM training job using the existing config; on completion, run `mil/slurm/eval_mil.sh` against the new checkpoint and confirm the `child_adapted` row appears in `results_summary.md`.

**Acceptance Scenarios**:

1. **Given** the child-adapted WavLM checkpoint at `synth_results/child_wavlm_checkpoint/step_50000/` exists and is readable, **When** `sbatch mil/slurm/train_mil.sh mil/configs/wavlm_mil_child_adapted.yaml` is submitted, **Then** training completes within 24 h on a single GPU and writes `mil/mil_results/wavlm_mil_child_adapted/best_checkpoint.pt`.
2. **Given** training completes, **When** evaluation runs, **Then** `test_metrics_tuned.json`, `test_predictions.csv`, `test_metrics_by_timepoint.csv`, and `val_metrics_tuned.json` are written and the test row is added to `results_summary.md` with deltas vs. the off-the-shelf `wavlm_mil` baseline (AUROC 0.771).
3. **Given** the per-timepoint metrics, **When** 14_month and 36_month rows are compared, **Then** the child-adapted backbone's gain (or loss) is reported per age band, since child-adapted pretraining is hypothesized to help most at the younger age.
4. **Given** the checkpoint is missing or incomplete (pretraining not finished), **When** the SLURM job starts, **Then** it exits with a clear error pointing to `synth/slurm/run_wavlm_pretrain.sh` rather than silently falling back to the off-the-shelf backbone.

---

### User Story 3 — ACMIL Head Drop-In (Priority: P2)

A researcher wants to address the small-data attention-concentration failure mode that likely caps the gated-ABMIL head: with ~1300–1500 training clips, a single attention vector overfits to a tiny subset of instances per bag. ACMIL (Zhang et al., ECCV 2024) attacks this with two switchable components — Multiple Branch Attention (MBA) parallel attention branches with a diversity regularizer, and Stochastic Top-K Instance Masking (STKIM) which randomly suppresses the top-K attention positions during training. Both are direct generalizations of the existing `GatedABMILHead` and slot into both the frame-window MIL path (`mil/mil_model.py`) and the segment-instance MIL path (`seg_model.py`).

**Why this priority**: Strong literature support (ECCV 2024 SOTA on three histopathology benchmarks; matched problem regime: small bags, small training sets, weak labels). Higher implementation cost than US1/US2 because it adds a new head class and at least one regularization-loss term. Risk: ACMIL was validated on visual MIL — translation to audio is a thesis-credible study even if it doesn't beat gated-ABMIL.

**Independent Test**: Train `wavlm_mil_acmil` and `whisper_mil_acmil` configs on the seen-child split. Pass when `test_metrics_tuned.json` and a `branch_weights.json` (one row per attention branch) are written and val-tuned thresholds are applied to test.

**Acceptance Scenarios**:

1. **Given** a new `ACMILHead` class registered alongside `GatedABMILHead`, **When** a config sets `head: acmil` with `n_branches: 5` and `stkim_p: 0.5`, **Then** training produces non-trivial branch diversity (no two branches collapse to identical attention) and the diversity regularizer term is logged per epoch.
2. **Given** STKIM is enabled, **When** training runs, **Then** the masking is applied only at training time (not at val/test), and the masking probability is annealed or held constant per the config — both modes must be supported.
3. **Given** the new head trains to convergence, **When** test AUROC is compared to `gated_attention` (frame-window WavLM 0.771 / Whisper 0.853), **Then** the delta is reported. ACMIL is considered successful if delta_AUROC > 0 on at least one of (WavLM, Whisper) seen-child or cross-child runs.
4. **Given** weak-diarization eval (`mil/eval_weak_diarization.py`), **When** ACMIL attention is compared to gated-ABMIL attention against ground-truth child-speech RTTM, **Then** ACMIL attention is at least as well-aligned (Pearson and AUROC) as gated-ABMIL on both age bands; if ACMIL bag-level AUROC improves but attention alignment degrades, that tradeoff is documented.
5. **Given** an existing `gated_attention` run, **When** the same dataset, seed, and training schedule are used for ACMIL, **Then** the comparison is apples-to-apples and the seed is logged in `config.json`.

---

---

### User Story 4 — TS-MIL: Target-Speaker Conditional MIL Head (Priority: P2)

A researcher wants the MIL head to know *which child* it is supposed to detect. The existing `GatedABMILHead` learns one global notion of "child-vocalization-ness"; the proposed TS-MIL head additionally takes a per-clip ECAPA prototype `p ∈ R^192` (the duration-weighted L2-normalized embedding of the target child at the corresponding timepoint, identical to the enrollment prototype built by `pyannote/unified.py:559` `build_child_prototypes`). The head conditions its attention/embedding computation on `p`, so two clips with identical audio but different target children can produce different scores. Two head variants:

- **TS-Concat**: `h_k' = [h_k ; W_p · p]` — project the 192-d prototype to a small dimension, concatenate to every instance embedding, then run gated attention as usual.
- **TS-FiLM**: `gamma, beta = MLP(p)`; `h_k' = gamma * h_k + beta` — feature-wise linear modulation of every instance embedding by the prototype before attention.

**Why this priority**: This is the TS-VAD recipe (Medennikov 2020) expressed inside the MIL aggregation, supported empirically by Bertamini 2025's finding that 30 s of in-domain target speaker enrollment dramatically improves clinical child diarization. The enrollment infrastructure that builds these prototypes already exists in `pyannote/unified.py`; spec-014 needs to (a) cache them to disk so the MIL training loop can read them and (b) add the TSMILHead class.

**Independent Test**: Train `wavlm_mil_tsmil_concat` (and optionally `wavlm_mil_tsmil_film`) on the seen-child split. Pass when (i) the run loads prototypes for all train+val+test (child, timepoint) pairs without missing keys, (ii) `test_metrics_tuned.json` is written, and (iii) val-tuned threshold is applied to test.

**Acceptance Scenarios**:

1. **Given** the prototype cache file produced by a new `mil/scripts/build_prototype_cache.py` (FR-018), **When** the TS-MIL training loop runs, **Then** it loads the prototype tensor for each clip's (child_id, timepoint_norm) key and asserts no missing keys among labelled clips with valid prototypes.
2. **Given** training completes, **When** test AUROC is compared to `wavlm_mil` baseline (0.771 seen-child / 0.690 cross-child), **Then** the delta is reported in `results_summary.md`.
3. **Given** the FiLM variant is also run, **When** both flavors complete, **Then** their results are compared head-to-head (concat vs FiLM) and the better-performing variant is recorded with the rationale (capacity vs locality of conditioning).
4. **Given** the cross-child split, **When** TS-MIL is trained on cross-child train clips and evaluated on cross-child test clips with prototypes built from cross-child train, **Then** the TS-MIL gain (or loss) over `wavlm_mil_cross_child` (test AUROC 0.690) is recorded — TS-MIL is hypothesized to help most in the cross-child regime where the model cannot rely on having seen the target child during training.

---

### User Story 5 — DSMIL Dual-Stream Aggregator (Priority: P2)

A researcher wants to add an instance-level auxiliary supervision signal to segment-instance MIL by adopting Li et al.'s Dual-Stream MIL (CVPR 2021). DSMIL keeps two parallel streams that share embeddings but use different aggregation:

- **Max stream**: scores every instance with a linear head, identifies the highest-scored "critical" instance `m = argmax_k score_k`, and uses `score_m` as one bag-level prediction `logit_max`.
- **Attention stream**: for each instance, computes attention as a function of the distance between its embedding and the critical instance's embedding (`a_k ∝ q · cos(h_k, h_m)`), then aggregates as `z = sum_k a_k * h_k` and produces `logit_attn = head(z)`.

The bag loss is `(BCE(logit_max, y) + BCE(logit_attn, y)) / 2`, and the final prediction is `(sigmoid(logit_max) + sigmoid(logit_attn)) / 2`. The max stream provides direct instance-level pressure ("there must be at least one instance scoring high"), and the attention stream provides relational context ("instances similar to the critical one matter more").

**Why this priority**: Adds an instance-level auxiliary signal without adding new data; should help with sparse-positive bags from `usc_sail` (median 21 segments, max 2911) where attention-only aggregation gets diluted. Pure PyTorch, ~80 lines.

**Independent Test**: Add `dsmil` to `aggregators` in `mil/configs/seg_mil_sweep.yaml`; rerun the sweep (resume-safe) so DSMIL is added across all four frontends. Pass when each `mil/mil_results/seg_mil/{frontend}_dsmil/` directory contains `test_metrics_tuned.json` and `all_configs.json` gains four DSMIL rows.

**Acceptance Scenarios**:

1. **Given** the `DSMILAgg` class is registered in `seg_model.py:build_aggregator()`, **When** the sweep runs with `dsmil` added to the aggregators list, **Then** four new run directories appear (one per frontend) with the standard segment-MIL output schema.
2. **Given** the run completes, **When** test AUROC is compared to the best gated-attention frontend (babar_vtc 0.808), **Then** the delta is recorded. DSMIL is considered successful if delta_AUROC > 0 on at least one frontend.
3. **Given** DSMIL emits two logits (max, attention), **When** logging is enabled, **Then** the per-clip `attention_max_logit` and `attention_stream_logit` are saved alongside the final score in `test_predictions.csv`, allowing post-hoc inspection of which stream contributed each correct/incorrect prediction.
4. **Given** the existing seven aggregators (mean/max/attention/gated/noisy_or/top_k/transformer) plus the three from US6, **When** DSMIL is added, **Then** the eleven-aggregator sweep completes without OOM on the existing 40 GB GPU configuration.

---

### User Story 6 — Adaptive / Learnable Pooling Operators (Priority: P2)

A researcher wants to add three adaptive pooling operators to the segment-instance MIL sweep, each a known performer on weakly-supervised SED benchmarks (the closest analogue task in audio):

- **AutoPool** (McFee, Salamon, Bello, ICASSP 2018): `pool(x) = sum_k softmax(alpha * x_k) * x_k`, with `alpha` a learnable scalar. `alpha = 0` → mean pool; `alpha → infinity` → max pool. Smoothly interpolates and is data-driven.
- **ExpSoftmaxPool** (Wang et al. arXiv:1810.09050 §III.E): `pool(x) = (sum_k exp(x_k) * x_k) / (sum_k exp(x_k))` — exponentially-weighted softmax over instance scores, no learnable parameter beyond the linear head.
- **GMAP — Gated Multi-Head Attention Pooling** (Hong et al.): multi-head attention pooling where each head has a learnable query, and the head outputs are gated by a sigmoid before being averaged. Generalization of single-head ABMIL with explicit multi-modal coverage of the bag.

**Why this priority**: Cheap (each ~30–60 lines in `seg_model.py:build_aggregator()`); the spec-005 sweep covered only fixed-form pooling. Adaptive pooling is the highest-confidence bet from the SED literature for moving past a fixed-pool plateau.

**Independent Test**: Add `auto_pool`, `exp_softmax_pool`, `gmap` to `aggregators` in `mil/configs/seg_mil_sweep.yaml`; rerun the sweep (resume-safe). Pass when each new `mil/mil_results/seg_mil/{frontend}_{aggregator}/` exists with the standard segment-MIL output schema and `all_configs.json` gains 12 new rows (3 aggregators × 4 frontends).

**Acceptance Scenarios**:

1. **Given** AutoPool is added, **When** the sweep runs, **Then** the learned `alpha` scalar per (frontend, run) is saved to `config.json` (final value) and is non-zero, indicating the model exploited the parameter.
2. **Given** GMAP is added with `n_heads=4`, **When** the sweep runs, **Then** per-head attention weights are saved to a wide CSV `head_attention.csv` for downstream interpretability.
3. **Given** all three new aggregators complete, **When** their test AUROC is compared against the existing seven on the same frontend, **Then** the comparison is recorded in `all_configs.json` and the per-aggregator ranking is reported.
4. **Given** ExpSoftmaxPool's saturating exponential, **When** training runs on a frontend with sparse positive instances (`usc_sail` median 21 segs/clip), **Then** no NaN losses appear in `training_history.csv` (the score head must be initialized so logits stay in a numerically safe range; e.g., zero-init).

---

### Edge Cases

- **WavLM hidden-states shape**: `hidden_states` is a tuple of length `num_hidden_layers + 1` (the leading entry is the conv-feature output, not a transformer layer). The weighted sum MUST exclude index 0 by default, configurable via `layer_aggregation_skip_first: true`.
- **Whisper hidden-states**: Whisper's encoder hidden_states tuple has the same convention; same fix applies. Verify shape on the first batch and assert.
- **Layer-weight initialization**: Initialize the layer weights to a uniform softmax (zeros logits) so training starts at the equal-weighted average, not at any single layer. This makes the first epoch comparable to a "mean of all layers" baseline.
- **ACMIL at K=1 instance**: The MBA diversity loss is undefined when only one branch fires or only one instance is present (degenerate softmax). Mask the diversity term out when N_instances == 1 or when batch contains only positives.
- **STKIM with very small bags**: When N_instances ≤ K, skip STKIM for that bag (no instances left to mask).
- **Child-adapted checkpoint format**: The checkpoint produced by `synth/slurm/run_wavlm_pretrain.sh` is a state-dict; ensure `BackboneExtractor` can load it via `from_pretrained` directory or a `torch.load` fallback. Document which path is used.
- **Cross-child split**: Run all three US on cross-child as well as seen-child where existing baselines exist. The cross-child Whisper-MIL number (0.876) is higher than seen-child (0.853); a layer-sum or ACMIL change must not regress that.
- **Reproducibility**: Each run MUST log seed, config hash, and (for layer-sum) initial and final layer weights to `config.json`.
- **TS-MIL prototype-cache freshness**: If the prototype cache file is older than the train CSV it was built from, training MUST refuse to start and ask the user to rebuild the cache. Stale prototypes are a silent leakage hazard.
- **TS-MIL missing prototype**: If a clip's `(child_id, timepoint_norm)` has no prototype (e.g., `child_prototype_stats.csv` `status != ok`), the clip is dropped from training/eval with a warning. Negative clips (label=0) without a prototype are still scored using a zero-vector prototype so the model sees them.
- **DSMIL with empty bags**: When `mask.sum() == 0`, both streams output zero logits and the loss term contributes 0 (existing pattern in `MaxAgg`/`AttnAgg`).
- **AutoPool initialization**: `alpha` is initialized to 0.0 so the first epoch is a mean pool. The sign of `alpha` is unconstrained at inference but logged.

---

## Requirements *(mandatory)*

### Functional Requirements

**Sub-feature US1 — Weighted-Layer-Sum (P1)**

- **FR-001**: `BackboneExtractor` MUST accept a new `layer_aggregation` field with values `last` (default, current behavior) or `weighted_sum`. When `weighted_sum`, a learnable parameter `layer_weights: nn.Parameter(num_layers)` is registered and `softmax(layer_weights) @ stacked_hidden_states` is returned.
- **FR-002**: Layer weights MUST be excluded from the frozen-backbone parameter freeze and trainable jointly with the MIL head; `requires_grad=True` for `layer_weights` even when `model.eval()` is set on the backbone.
- **FR-003**: The leading conv-feature entry of `hidden_states` MUST be skipped by default (configurable via `layer_aggregation_skip_first`).
- **FR-004**: At end of training, the final softmax(layer_weights) vector MUST be saved to `{run_dir}/layer_weights.json` for inspection.
- **FR-005**: Three new configs are required: `wavlm_mil_layersum.yaml`, `whisper_mil_layersum.yaml`, `hubert_large_mil_layersum.yaml`. Each derives from the corresponding non-layersum baseline by adding `layer_aggregation: weighted_sum` and bumping `run_name`.

**Sub-feature US2 — Child-Adapted WavLM (P1)**

- **FR-006**: An end-to-end run of `mil/configs/wavlm_mil_child_adapted.yaml` MUST be submitted via the existing `mil/slurm/train_mil.sh` and `mil/slurm/eval_mil.sh`, producing all standard MIL output files in `mil/mil_results/wavlm_mil_child_adapted/`.
- **FR-007**: Training MUST verify the child-adapted checkpoint exists at `synth_results/child_wavlm_checkpoint/step_50000/` (or a configurable step) before launching; missing checkpoint → exit code 2 with a pointer to the pretraining script.
- **FR-008**: `results_summary.md` MUST be updated with a new row for child-adapted WavLM-MIL alongside the existing rows, including overall and per-timepoint deltas vs. the off-the-shelf `wavlm_mil` baseline.
- **FR-009**: A short note in `CLAUDE.md` Recent Changes MUST record the result (positive or negative) including job ID and key delta numbers, mirroring the format used for prior negative results (TinyVox, hardneg, multi-child suppressor).
- **FR-010**: If the layer-sum result from US1 is positive, ALSO run a `wavlm_mil_child_adapted_layersum.yaml` config that combines both — child-adapted backbone with weighted-layer-sum — to test for compound gains.

**Sub-feature US3 — ACMIL Head (P2)**

- **FR-011**: A new `ACMILHead` class MUST be added to `mil/mil_model.py` with constructor parameters: `in_dim`, `hidden_dim` (default 256), `n_branches` (default 5), `stkim_p` (default 0.5; 0.0 disables STKIM), `mba_diversity_weight` (default 0.1; 0.0 disables diversity regularizer), `dropout`.
- **FR-012**: Forward pass MUST return `(logit, attn, branch_attn, diversity_loss)`; the existing `GatedABMILHead` interface returns `(logit, attn)` and the training loop MUST be updated to optionally accept a third `diversity_loss` term.
- **FR-013**: STKIM MUST be applied **only when `self.training` is True**; at eval/test, the attention is computed without masking.
- **FR-014**: The MBA diversity loss MUST regularize against attention-vector cosine similarity across branches: `L_div = mean over (i<j) of cos(A_i, A_j)^2` (or equivalent — final form documented in code with citation to ACMIL paper).
- **FR-015**: A factory function MUST allow `head: gated_abmil | acmil` selection from config; default remains `gated_abmil` for backward compatibility.
- **FR-016**: At least one `wavlm_mil_acmil.yaml` and one `whisper_mil_acmil.yaml` MUST be added under `mil/configs/`.
- **FR-017**: `mil/eval_weak_diarization.py` MUST be extended (or wrapped by a new script) to read ACMIL multi-branch attention and report alignment statistics either per-branch or for the average across branches.

**Sub-feature US4 — TS-MIL Head (P2)**

- **FR-018**: A new script `mil/scripts/build_prototype_cache.py` MUST build per-(child_id, timepoint_norm) ECAPA prototypes from the labelled positive clips of a given split CSV (default: `whisper-modeling/seen_child_splits/train.csv`) and save them as a `.npz` file at `mil/prototypes/{frontend}.npz` with keys `f"{child_id}__{timepoint_norm}"` mapping to L2-normalized 192-d float32 vectors. The script reuses the duration-weighted aggregation logic in `pyannote/unified.py:559` `build_child_prototypes`.
- **FR-019**: A new `TSMILHead` class MUST be added to `mil/mil_model.py` with `mode: "concat" | "film"`, `prototype_dim: int = 192`, `prototype_proj_dim: int = 64` (concat mode), and the existing GatedABMILHead-style attention layer. Forward signature: `forward(h: (N, in_dim), prototype: (prototype_dim,)) -> (logit, attn)`.
- **FR-020**: `mil/mil_train.py` MUST accept a new config key `prototype_cache: str` (path to the .npz file). When set, the training loop loads prototypes per clip and passes them to `MILModel.forward` alongside the bag of windows. When unset, `MILModel.forward` retains its current bag-only signature (`gated_abmil`, `acmil`).
- **FR-021**: For clips where no prototype exists for the (child_id, timepoint_norm) key, training/eval MUST drop the clip and log a single warning per missing key (not per-clip). Missing-prototype counts MUST be saved to `{run_dir}/missing_prototypes.json`.

**Sub-feature US5 — DSMIL Aggregator (P2)**

- **FR-022**: A new `DSMILAgg(nn.Module)` class MUST be added to `mil/seg_model.py` implementing the dual-stream architecture (Li et al. CVPR 2021 §3.2): max-instance stream + cosine-distance attention stream, two BCE losses averaged.
- **FR-023**: `build_aggregator()` in `seg_model.py:246` MUST register `dsmil` as a valid aggregator name.
- **FR-024**: `mil/seg_train.py` MUST detect when the chosen aggregator is `dsmil` and average the two BCE losses (max-stream loss + attention-stream loss). Final `score` for predictions CSV is the average of the two sigmoid outputs.

**Sub-feature US6 — Adaptive Pooling Aggregators (P2)**

- **FR-025**: Three new aggregator classes MUST be added to `mil/seg_model.py`: `AutoPoolAgg`, `ExpSoftmaxPoolAgg`, `GMAPAgg`.
- **FR-026**: `build_aggregator()` MUST register `auto_pool`, `exp_softmax_pool`, `gmap` as valid aggregator names.
- **FR-027**: `mil/configs/seg_mil_sweep.yaml` MUST include the three new aggregators in its `aggregators` list (for a total of 11 aggregators); the sweep is resume-safe so adding new aggregators triggers only the new (frontend × aggregator) cells.
- **FR-028**: `AutoPoolAgg` MUST log the final learned `alpha` scalar in the run's `config.json` for inspection.

### Key Entities

- **`hidden_states`**: Tuple from HuggingFace transformer encoders (length = `num_hidden_layers + 1`; leading entry is conv-feature embedding).
- **`layer_weights`**: 1D `nn.Parameter` of length `num_hidden_layers` (or `num_hidden_layers + 1` if leading entry retained). Trained jointly with MIL head.
- **`branch_attn`**: `(N_branches, N_instances)` matrix of per-branch attention weights produced by ACMIL MBA.
- **`diversity_loss`**: Scalar regularizer term encouraging branches to attend to different instances.
- **Child-adapted backbone**: `synth_results/child_wavlm_checkpoint/step_50000/` (or matching step), output of `synth/slurm/run_wavlm_pretrain.sh`.

---

## Success Criteria

A successful spec-014 produces, at minimum:

- Three new MIL run directories under `mil/mil_results/`: `wavlm_mil_layersum/`, `wavlm_mil_child_adapted/`, `wavlm_mil_acmil/` (and the matching Whisper + HuBERT-Large variants where backbones differ).
- A `results_summary.md` table that compares all three extensions against existing `wavlm_mil`/`whisper_mil` baselines on both seen-child and cross-child splits, with overall and per-timepoint deltas.
- A `CLAUDE.md` Recent Changes entry per US documenting the result (positive or negative), job IDs, and key delta numbers.
- For US1: `layer_weights.json` per run showing which transformer layers the model selected, ideally identifying a non-final layer as a top contributor (consistent with the literature prior).
- For US3: `branch_weights.json` and weak-diarization alignment numbers per branch.

A null result on any single US is acceptable so long as the comparison is documented; combined null results across all three would be a publishable observation that the gated-ABMIL + last-layer + off-the-shelf-WavLM baseline is a strong floor for this dataset size.

---

## Out of Scope

- Segment-instance MIL extensions (covered by spec-005).
- Audio-visual MIL or AV-conditioned MIL (covered by spec-006/007).
- Mean-teacher / self-distillation with unlabeled HomeBank — strong follow-up but distinct (Tier-2 in the research note); leave for spec-015.
- Target-Speaker MIL (TS-MIL with ECAPA prototype injection) — strong follow-up; leave for spec-015.
- Adaptive / auto-pool / exponential-softmax pooling operators — strong follow-up; leave for spec-015 if any of the three US here disappoint.
- New diarizer frontends or re-training the diarizer suite.

---

## References

- Pasad, Shi, Livescu (ICASSP 2023) — layer-wise probing of WavLM/HuBERT (related_works.MD §1.2).
- Chen et al. (IEEE JSTSP 2022) — WavLM-based diarization with weighted-layer features (related_works.MD §2.2).
- Polok, Landini, Burget et al. (BUT 2024, DiariZen) — production WavLM weighted-layer-sum diarization recipe (related_works.MD §2.2).
- Bertamini et al. (Res Dev Disabil 2025) — 30 s of in-domain adaptation for child speech (related_works.MD §1.2).
- Lahiri, Feng, Bishop, Narayanan (Interspeech 2023; ICASSP 2024) — USC-SAIL WavLM-based child-adult diarization (related_works.MD §1.2).
- Al Futaisi et al. (Frontiers in Digital Health 2025) — task-specific pretraining beating wav2vec 2.0 fine-tuning on small child-speech tasks (related_works.MD §1.2).
- Zhang et al. (ECCV 2024) — Attention-Challenging MIL (https://arxiv.org/abs/2311.07125; code at https://github.com/dazhangyu123/ACMIL).
- Ilse, Tomczak, Welling (ICML 2018) — Gated ABMIL (current `GatedABMILHead` in `mil/mil_model.py`).
- Wang et al. (arXiv:1810.09050) — comparison of five MIL pooling functions for SED.
