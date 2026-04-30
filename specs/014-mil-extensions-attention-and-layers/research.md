# Phase 0 Research — spec-014 MIL Extensions

Resolves the open design questions from `spec.md` and `plan.md` Technical Context.

---

## R1 — Weighted-Layer-Sum: which layers, how to skip the conv-feature output

**Decision**: Compute `softmax(layer_weights) @ stack(hidden_states[1:])` over all *transformer* hidden states, excluding `hidden_states[0]` (the conv-feature embedding) by default. Configurable via `layer_aggregation_skip_first: true`.

**Rationale**:
- HuggingFace `WavLMModel`/`HubertModel`/`Wav2Vec2Model` return `hidden_states` as a tuple of length `num_hidden_layers + 1`; index 0 is the output of the conv feature extractor (post-projection, pre-transformer), indices 1..L are transformer-block outputs. WhisperEncoder follows the same convention.
- DiariZen, pyannote 3.x WavLM frontend, and the SUPERB benchmark all weighted-sum across transformer outputs only. Including the conv-feature stream is unconventional and would dilute the speaker-discriminative signal Pasad et al. document at layers 4–8 (HuBERT-base) / 6–12 (WavLM-Large).
- Initialize `layer_weights = zeros(L)` so `softmax` starts uniform — first epoch is the equal-weighted mean, then training selects the dominant layers. Avoids any prior bias toward the last layer.

**Alternatives considered**:
- *Last-layer + skip-connection sum* (concat last + mid). Rejected: doubles `embed_dim`, breaks downstream attention head dimensions, requires retraining from scratch. Marginal expected gain over softmax weighted-sum.
- *Hard layer selection (pick best layer on val)*. Rejected: brittle and per-backbone; the learnable softmax recovers this if the val-best layer dominates anyway, while preserving graceful degradation to layer mixtures.
- *Per-frame attention over layers* (attention pooling instead of fixed weights). Rejected: extra parameters, no SUPERB-evidence backing for the diarization-style frame-mean pipeline used here. Reserve for future spec.

**Gradient handling**: Backbone params remain frozen (`requires_grad=False`); `layer_weights` is registered as a separate `nn.Parameter` so it joins the trainable param list returned by the head. The `model.eval()` call inside `BackboneExtractor.__init__` only affects backbone modules (dropout/BN); `layer_weights` is unaffected.

**Sanity check**: After training, log `softmax(layer_weights)` and assert it is *not* a one-hot at the last layer — if it is, the implementation is silently equivalent to the baseline.

---

## R2 — ACMIL diversity-loss form (MBA branch regularization)

**Decision**: Use the cosine-similarity penalty from the ACMIL paper (Eq. 3 in Zhang et al. ECCV 2024):
```
L_div = (1 / (n_branches * (n_branches - 1))) * sum_{i != j} cos(A_i, A_j)
```
where `A_i` is the (N_instances,) softmax-normalized attention vector of branch `i`. Sign is positive — minimizing this *reduces* between-branch agreement.

**Rationale**:
- ACMIL's reference implementation uses a cosine penalty over the post-softmax attention vectors. We replicate this exactly to stay faithful to the published method; deviating (e.g., L2, KL-div) introduces a confound between method-faithfulness and per-dataset tuning.
- The `mba_diversity_weight: 0.1` default mirrors the paper's `lambda_div` setting on Camelyon16; tune on val if ACMIL underperforms gated-ABMIL.

**Alternatives considered**:
- *Squared cosine* (`cos^2`). Rejected: more sensitive to noise; cosine penalty already enforces non-redundancy without amplifying tiny perturbations.
- *Determinantal point process* on attention vectors. Rejected: 5× more compute per batch, no documented gain at this scale.

**Edge case**: When `N_instances == 1`, all attention vectors collapse to `[1.0]` and pairwise cosine similarity is 1 by construction. Mask the diversity term out for single-instance bags (rare for frame-window MIL where every clip yields ≥1 window, common-ish for empty-bag segment-instance configs).

---

## R3 — STKIM (Stochastic Top-K Instance Masking) schedule

**Decision**: Hold `stkim_p = 0.5` constant during training; do not anneal. K = `max(1, floor(0.1 * N_instances))` per the paper, capped at K=10. Apply only when `self.training` is True.

**Rationale**:
- ACMIL paper holds `p` constant; the annealing variant tested in their ablation (linearly decay to 0 over training) gave smaller gains and required tuning the schedule. Constant `p` is the parameter-free default.
- K = 10% of N instances bounds the masked fraction independent of bag size. Cap at K=10 prevents overly aggressive masking for the few large bags from `usc_sail` (median 21, max 2911 segments per clip).
- At eval time, full attention is used — STKIM is a regularizer only.

**Alternatives considered**:
- *Anneal `p` from 0.5 → 0 over epochs*. Rejected: extra hyperparameter, paper shows marginal effect.
- *Mask in attention space rather than instance space* (zero out top-K attention values, renormalize). Rejected: this is exactly what the paper-faithful STKIM already does; the attention is renormalized after the masked positions have their weight redistributed.

**Implementation note**: In the paper, masked attention is set to 0 and the remainder is *not* re-softmaxed (the bag embedding is just the un-normalized weighted sum minus the masked positions). Replicate this — re-softmax would change the regularizer's effect.

---

## R4 — Child-adapted WavLM checkpoint loading path

**Decision**: Resolve `backbone_path` via the existing `BackboneExtractor.__init__` logic (already supports a local path, see `mil/mil_model.py:148` `cfg.get("backbone_path", cfg["backbone"])`). The pretraining script writes a HuggingFace-format directory at `synth_results/child_wavlm_checkpoint/step_50000/` that `AutoModel.from_pretrained` can read directly.

**Rationale**:
- `synth/slurm/run_wavlm_pretrain.sh` already produces `pytorch_model.bin` + `config.json` per the spec-009 US3 implementation, in standard HuggingFace layout.
- No code change needed in `BackboneExtractor`; `wavlm_mil_child_adapted.yaml` already sets `backbone_path: synth_results/child_wavlm_checkpoint/step_50000`.
- Pre-flight check: training script asserts `os.path.isdir(backbone_path) and os.path.isfile(f"{backbone_path}/config.json")` before launching; missing checkpoint → exit code 2 with message pointing to `synth/slurm/run_wavlm_pretrain.sh` (per FR-007).

**Alternatives considered**:
- *Load from a step checkpoint that's not yet final (e.g., step_20000)*. Rejected for the headline run; supported via config override if intermediate-step ablation is wanted later.
- *Bake the child-adapted checkpoint into a HuggingFace model card upload*. Rejected as out-of-scope for spec-014; reproducibility lives in `synth_results/` paths.

---

## R5 — Backward-compatibility contract

**Decision**: Add `layer_aggregation: last` and `head: gated_abmil` as the default values when keys are absent from a config, so existing configs (`wavlm_mil.yaml`, `whisper_mil.yaml`, `hubert_large_mil.yaml`, `wav2vec2_large_mil.yaml`) reproduce numerically. Existing baselines must be re-runnable to within ±0.005 AUROC of the committed numbers.

**Rationale**:
- Constitution Principle I (Reproducibility) and III (Baseline-First) require that prior baselines remain valid reference points. Any silent change in default behavior would invalidate the entire `results_summary.md` table.
- The spec's FR-015 explicitly requires `head: gated_abmil` as the default factory selection.

**Verification step**: Before merging spec-014 code, run `wavlm_mil` on the seen-child split with the updated codebase and diff `test_metrics_tuned.json` against the committed baseline. Acceptance: AUROC delta ≤ 0.005, F1 delta ≤ 0.01.

---

## R6 — Eval script changes for ACMIL multi-branch attention

**Decision**: Extend `mil/eval_weak_diarization.py` to read `branch_weights.json` (if present) and compute alignment metrics per-branch and for the mean across branches. For non-ACMIL runs, behavior is unchanged.

**Rationale**:
- `eval_weak_diarization.py` already consumes attention CSVs from segment-instance MIL runs; ACMIL's per-branch attention can be saved alongside the bag-level attention as a wide CSV (`branch_0_weight, branch_1_weight, …, branch_n_weight, mean_weight`).
- A per-branch alignment number is the cleanest test that branches are doing different things — if all branches give identical alignment scores, MBA collapsed and the diversity loss didn't bite.

**Alternatives considered**:
- *Separate `eval_acmil_attention.py` script*. Rejected: duplicates 80% of `eval_weak_diarization.py` logic. The modification is ~30 lines of new branch-iteration code in the existing script.

---

## R7 — Test-data leakage prevention

**Decision**: Reuse the already-validated split-integrity assertions from spec-002 (`mil/mil_train.py` reads `whisper-modeling/seen_child_splits/{train,val,test}.csv` directly; no in-script re-splitting). For US1/US2/US3, add a one-line assertion in `mil_evaluate.py` confirming val-tuned threshold is loaded from a JSON file written *before* test inference begins.

**Rationale**:
- This is already enforced by the existing `mil/mil_evaluate.py` flow; it is restated here as a guard against accidental regressions during the new edits.

---

## R8 — SLURM walltime sizing for new runs

**Decision**: Reuse `mil/slurm/train_mil.sh` (24 h walltime, 1× A100, 40 GB RAM) for US1 and US2; for US3, request 36 h to allow extra epochs if ACMIL converges more slowly due to the diversity regularizer. Memory unchanged (40 GB suffices for n_branches=5 ACMIL at batch_size=16, per ACMIL paper Table 4 disk/memory footprint).

**Rationale**:
- US1 forward pass through hidden_states is computed once per batch (HF returns all layers via `output_hidden_states=True` regardless), so the layer-sum is essentially free.
- US3 ACMIL adds 5× small linear heads over 768-dim embeddings — negligible memory, ~5% per-step overhead.

---

---

## R9 — TS-MIL prototype cache format and conditioning flavor

**Decision**: `mil/scripts/build_prototype_cache.py` produces `mil/prototypes/{frontend}.npz` with keys `f"{child_id}__{timepoint_norm}"` mapping to L2-normalized 192-d float32 vectors (matching SpeechBrain ECAPA `speechbrain/spkrec-ecapa-voxceleb`). Default frontend for the cache: `babar_vtc` (best segmenter for prototype quality per spec-002 results). Both **concat** and **FiLM** conditioning flavors are implemented; `concat` is the default because it adds parameters monotonically (preserves the gated-ABMIL inductive bias); FiLM is offered as an ablation.

**Rationale**:
- `pyannote/unified.py:559` `build_child_prototypes` already implements duration-weighted prototype construction; the new script is a thin CLI wrapper that dumps the `prototypes` dict to .npz.
- 192 dims matches the SpeechBrain ECAPA output dim; no dim mismatch between cache files across frontends.
- Concat-mode adds parameters via a single `nn.Linear(192, prototype_proj_dim=64)` projection — cheap, well-understood capacity. FiLM gates each dimension independently (`gamma, beta = MLP(p)`, both 768-d for WavLM), more parameters but stronger conditioning. Run both and compare per US4 acceptance #3.

**Alternatives considered**:
- *Build prototypes inside `mil_train.py` per run.* Rejected: adds 5-10 minutes per run for ECAPA inference; cache makes runs idempotent and sweepable. Also risks subtle differences across runs if frontend RTTM caches change.
- *Cross-attention conditioning (prototype as query, instances as keys/values).* Rejected as out-of-scope; concat/FiLM are simpler and more common in the TS-VAD literature. Reserve for follow-up spec.

---

## R10 — DSMIL critical-instance similarity metric

**Decision**: Use cosine similarity in the original (un-projected) embedding space — `a_k = q_W @ tanh(W_attn @ (h_k - h_m))` per Li et al. CVPR 2021 §3.2 Eq. 4 with the difference-then-project formulation. The two streams share the linear instance-scoring head that identifies the critical instance.

**Rationale**:
- Faithful to the CVPR 2021 paper.
- Difference-formulation (`h_k - h_m` rather than concat or dot product) makes the critical instance a centroid; attention is highest when `h_k` is close to `h_m`, lowest when far.

**Alternatives considered**:
- *Dot-product `h_k . h_m`.* Rejected: not what the DSMIL paper does; introduces a confound between method-faithfulness and per-dataset tuning.
- *Learned distance metric.* Rejected: extra parameters and tuning surface for marginal gain at this scale.

---

## R11 — AutoPool / ExpSoftmaxPool / GMAP design choices

**Decision**:
- **AutoPool** (McFee, Salamon, Bello, ICASSP 2018): single learnable `alpha` scalar shared across embedding dims, initialized to 0.0; pool over instance scores (post-linear-head), not over raw embeddings. This is the original "scalar AutoPool" formulation.
- **ExpSoftmaxPool**: pool over instance scores (`pool(s) = sum_k exp(s_k) * s_k / sum_k exp(s_k)`); no learnable parameter beyond the score head; clamp logits to `[-10, 10]` before exp to prevent overflow.
- **GMAP**: 4 attention heads, each with independent `(V_h, q_h)` parameters; each head produces a softmax-attention bag embedding; head outputs are gated by `sigmoid(g_h)` (one learnable scalar gate per head) and averaged. Final linear head produces the bag logit.

**Rationale**:
- Scalar AutoPool is simpler and more stable than the per-dim (vector-AutoPool) formulation; the McFee paper showed scalar AutoPool wins on most tasks.
- ExpSoftmaxPool needs the clamp to avoid `exp(very_large_logit) = inf` on `usc_sail` bags with thousands of instances; clamp at ±10 is the standard SED-baseline trick.
- GMAP at 4 heads matches the spec-005 transformer aggregator's head count (parameter parity for fair comparison).

**Alternatives considered**:
- *Per-dim AutoPool.* Rejected: harder to train, no consistent win in McFee 2018.
- *Soft-AutoPool* (replace softmax with `(alpha * x_k) / sum_k (alpha * x_k)` directly). Rejected: less stable.
- *GMAP without gating, just multi-head attention.* Rejected: gating is the discriminating feature in Hong et al.; without it, GMAP collapses to 4-head ABMIL.

---

## Summary of resolved unknowns

| Question | Decision |
|---|---|
| Skip leading conv-feature in weighted-layer-sum? | YES (configurable, default true) |
| ACMIL diversity loss form | Pairwise cosine over softmax attention vectors, lambda=0.1 |
| STKIM schedule | Constant p=0.5; K=max(1, floor(0.1·N)), cap 10 |
| Child-adapted checkpoint loading | HF `from_pretrained` via existing `backbone_path` config key |
| Default values for new config keys | `layer_aggregation: last`, `head: gated_abmil` (backward-compat) |
| Per-branch alignment in weak-diarization eval | Extend `eval_weak_diarization.py`, no new script |
| Test-leakage prevention | Existing val→test ordering; one-line assertion in eval |
| US3 SLURM walltime | 36 h (vs. 24 h for US1/US2) |
| TS-MIL prototype cache format | `.npz` keyed by `f"{child_id}__{timepoint_norm}"` → 192-d L2-normalized float32; default frontend babar_vtc |
| TS-MIL conditioning flavor | concat (default) and FiLM (ablation) — run both, compare |
| DSMIL similarity formulation | Cosine-attention-of-difference-vector per Li et al. CVPR 2021 §3.2 Eq. 4 |
| AutoPool formulation | Scalar `alpha` initialized 0.0; pool over instance scores |
| ExpSoftmaxPool numerical safety | Clamp logits to ±10 before exp |
| GMAP heads | 4, with per-head sigmoid gating before averaging |

No NEEDS CLARIFICATION items remain.
