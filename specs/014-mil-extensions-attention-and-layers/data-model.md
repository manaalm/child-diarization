# Phase 1 Data Model — spec-014 MIL Extensions

Class signatures, config schemas, and output-file schemas for the three user stories.

---

## 1. `BackboneExtractor` (modified) — `mil/mil_model.py`

**New constructor parameters** (added to existing signature):

```python
class BackboneExtractor(nn.Module):
    def __init__(
        self,
        backbone_name: str,
        layer: int = -1,                                 # legacy; used when layer_aggregation == "last"
        sample_rate: int = 16000,
        layer_aggregation: str = "last",                 # NEW: "last" | "weighted_sum"
        layer_aggregation_skip_first: bool = True,       # NEW: skip hidden_states[0] (conv-feature)
    ) -> None:
        ...
        if self.layer_aggregation == "weighted_sum":
            num_layers = self.model.config.num_hidden_layers
            n_to_combine = num_layers if layer_aggregation_skip_first else num_layers + 1
            self.layer_weights = nn.Parameter(torch.zeros(n_to_combine))
        else:
            self.layer_weights = None
```

**Forward semantics change**: when `layer_aggregation == "weighted_sum"`, replace the current

```python
hidden = out.hidden_states[self.layer]   # mil_model.py:65,68
```

with

```python
all_h = torch.stack(out.hidden_states[1:] if self.skip_first else out.hidden_states, dim=0)
# all_h: (L, B, T, D)
w = torch.softmax(self.layer_weights, dim=0).view(-1, 1, 1, 1)
hidden = (w * all_h).sum(dim=0)          # (B, T, D)
```

**Trainability**: backbone params remain frozen; `self.layer_weights` is the *only* backbone-side trainable parameter.

**Persistence**: at end of training, `mil_train.py` writes `softmax(layer_weights).cpu().tolist()` to `{run_dir}/layer_weights.json` keyed by layer index.

---

## 2. `ACMILHead` (new) — `mil/mil_model.py`

```python
class ACMILHead(nn.Module):
    """Attention-Challenging MIL head (Zhang et al., ECCV 2024).

    Components:
      - Multiple Branch Attention (MBA) — n_branches parallel gated-ABMIL
        attention vectors with a cosine-similarity diversity penalty.
      - Stochastic Top-K Instance Masking (STKIM) — at training time,
        zero out top-K attention positions per branch with prob stkim_p.

    Forward returns:
      logit:          scalar pre-sigmoid bag-level score (mean over branches' bag heads).
      attn:           (N,) mean attention across branches (for downstream weak-diarization eval).
      branch_attn:    (n_branches, N) per-branch attention vectors (post-STKIM at train time).
      diversity_loss: scalar regularizer term (or 0 if mba_diversity_weight == 0).
    """
    def __init__(
        self,
        in_dim: int = 768,
        hidden_dim: int = 256,
        n_branches: int = 5,
        stkim_p: float = 0.5,
        stkim_k_frac: float = 0.1,
        stkim_k_cap: int = 10,
        mba_diversity_weight: float = 0.1,
        dropout: float = 0.25,
    ) -> None: ...

    def forward(
        self, h: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: ...
```

**Initialization**: each branch is an independent `(V_i, U_i, w_i, head_i)` quadruple matching the existing `GatedABMILHead`. Per-branch bag heads' logits are mean-reduced for the final bag-level score.

**STKIM application** (training only):
1. For each branch `i`, compute attention `A_i ∈ R^N` via gated-ABMIL.
2. Sample `mask_active ~ Bernoulli(stkim_p)`. If active, zero out the top-K positions of `A_i` (K = `min(stkim_k_cap, max(1, floor(stkim_k_frac * N)))`) without re-softmaxing.
3. Compute branch bag embedding `z_i = sum_k A_i[k] * h[k]`.

**MBA diversity loss**:
```python
A_stack = torch.stack(branch_attn, dim=0)   # (n_branches, N)
A_norm = F.normalize(A_stack, p=2, dim=1)
sim = A_norm @ A_norm.T                       # (n_branches, n_branches)
off_diag_mask = ~torch.eye(n_branches, dtype=torch.bool, device=h.device)
L_div = sim[off_diag_mask].mean()
```
Multiply by `mba_diversity_weight` before adding to total loss.

---

## 3. Head factory — `mil/mil_model.py`

```python
def build_mil_model(cfg: dict) -> MILModel:
    backbone_src = cfg.get("backbone_path", cfg["backbone"])
    backbone = BackboneExtractor(
        backbone_name=backbone_src,
        layer=cfg.get("backbone_layer", -1),
        layer_aggregation=cfg.get("layer_aggregation", "last"),         # NEW default
        layer_aggregation_skip_first=cfg.get("layer_aggregation_skip_first", True),
        sample_rate=16000,
    )
    head_kind = cfg.get("head", "gated_abmil")                          # NEW default
    if head_kind == "gated_abmil":
        head = GatedABMILHead(
            in_dim=backbone.embed_dim,
            hidden_dim=cfg.get("mil_hidden_dim", 256),
            dropout=cfg.get("mil_dropout", 0.25),
        )
    elif head_kind == "acmil":
        head = ACMILHead(
            in_dim=backbone.embed_dim,
            hidden_dim=cfg.get("mil_hidden_dim", 256),
            n_branches=cfg.get("acmil_n_branches", 5),
            stkim_p=cfg.get("acmil_stkim_p", 0.5),
            stkim_k_frac=cfg.get("acmil_stkim_k_frac", 0.1),
            stkim_k_cap=cfg.get("acmil_stkim_k_cap", 10),
            mba_diversity_weight=cfg.get("acmil_mba_diversity_weight", 0.1),
            dropout=cfg.get("mil_dropout", 0.25),
        )
    else:
        raise ValueError(f"Unknown head: {head_kind}")
    return MILModel(backbone=backbone, mil_head=head)
```

---

## 4. Training-loop adapter — `mil/mil_train.py`

`MILModel.forward` returns `(logit, attn)` for `gated_abmil` and `(logit, attn, branch_attn, div_loss)` for `acmil`. Wrap the forward call:

```python
out = model(windows)
if isinstance(out, tuple) and len(out) == 4:
    logit, attn, branch_attn, div_loss = out
else:
    logit, attn = out
    div_loss = 0.0

bce = F.binary_cross_entropy_with_logits(logit, label_tensor, pos_weight=pos_weight)
loss = bce + div_loss
```

Log `bce` and `div_loss` separately to the training history CSV under columns `loss_bce` and `loss_div` (the latter is 0 for non-ACMIL runs).

---

## 5. New config files — `mil/configs/`

### `wavlm_mil_layersum.yaml`

Inherits all keys from `wavlm_mil.yaml`. Adds:

```yaml
run_name: wavlm_mil_layersum
layer_aggregation: weighted_sum
layer_aggregation_skip_first: true
```

### `whisper_mil_layersum.yaml`

Inherits from `whisper_mil.yaml`. Same two new keys.

### `hubert_large_mil_layersum.yaml`

Inherits from `hubert_large_mil.yaml`. Same two new keys.

### `wavlm_mil_child_adapted.yaml`

Already exists. Spec-014 ensures it is run end-to-end and integrated into `results_summary.md` (no schema change).

### `wavlm_mil_child_adapted_layersum.yaml` *(conditional, FR-010)*

Created only if US1 layer-sum delta is positive. Identical to `wavlm_mil_child_adapted.yaml` plus:

```yaml
run_name: wavlm_mil_child_adapted_layersum
layer_aggregation: weighted_sum
layer_aggregation_skip_first: true
```

### `wavlm_mil_acmil.yaml`

Inherits from `wavlm_mil.yaml`. Adds:

```yaml
run_name: wavlm_mil_acmil
head: acmil
acmil_n_branches: 5
acmil_stkim_p: 0.5
acmil_stkim_k_frac: 0.1
acmil_stkim_k_cap: 10
acmil_mba_diversity_weight: 0.1
```

### `whisper_mil_acmil.yaml`

Same overrides on top of `whisper_mil.yaml`.

---

## 6. Output-file schemas

For every run, the existing schema is preserved:

```
mil/mil_results/{run_name}/
├── best_checkpoint.pt
├── config.json                       # full config dump
├── training_history.csv              # NEW columns: loss_bce, loss_div (0 for non-ACMIL)
├── val_metrics_tuned.json
├── val_predictions.csv
├── val_metrics_by_timepoint.csv
├── test_metrics_tuned.json
├── test_predictions.csv
└── test_metrics_by_timepoint.csv
```

Plus new artifacts:

| File | When | Schema |
|---|---|---|
| `layer_weights.json` | US1 (any layersum config) | `{"layer_index": float, ...}` — softmax-normalized weights, length L |
| `branch_weights.json` | US3 (any acmil config) | `{"branch_0": {...}, "branch_1": {...}}` — per-branch summary stats (mean/std attention; alignment to GT if available) |
| `branch_attention.csv` | US3 (per-clip CSV for weak-diar eval) | columns: `audio_path, instance_idx, start_sec, end_sec, branch_0_weight, ..., branch_{n-1}_weight, mean_weight` |

---

## 7. `TSMILHead` (new) — `mil/mil_model.py` (US4)

```python
class TSMILHead(nn.Module):
    """Target-Speaker MIL head — gated-ABMIL conditioned on a child prototype.

    mode="concat":  h_k' = [h_k; W_p · prototype]  (concat-then-attend)
    mode="film":    gamma, beta = MLP(prototype); h_k' = gamma * h_k + beta

    Forward signature:
      forward(h: (N, in_dim), prototype: (prototype_dim,)) -> (logit, attn)
    """
    def __init__(
        self,
        in_dim: int = 768,
        hidden_dim: int = 256,
        prototype_dim: int = 192,
        prototype_proj_dim: int = 64,
        mode: str = "concat",          # "concat" | "film"
        dropout: float = 0.25,
    ) -> None: ...
```

The training loop calls `model(windows, prototype=prototype_tensor)`; for non-TS heads (`gated_abmil`, `acmil`), the `prototype` kwarg is ignored.

---

## 8. Prototype cache file (new)

Produced by `mil/scripts/build_prototype_cache.py`. Format:

- File path: `mil/prototypes/{frontend}.npz` (default `babar_vtc.npz`)
- Keys: `f"{child_id}__{timepoint_norm}"` (string)
- Values: `np.ndarray(192,)` `float32`, L2-normalized

Companion: `mil/prototypes/{frontend}_stats.csv` (mirrors `child_prototype_stats.csv` from enrollment runs: `child_id, timepoint_norm, n_segments, status`).

---

## 9. New segment-MIL aggregator classes — `mil/seg_model.py` (US5, US6)

### `DSMILAgg(nn.Module)` (US5)

```python
class DSMILAgg(nn.Module):
    """Dual-stream MIL (Li et al. CVPR 2021).

    Returns:
      logit_max:   max-stream bag logit (the score of the critical instance).
      logit_attn:  attention-stream bag logit (cosine-distance attention).
      attn_weights: (K_max,) attention weights from the attention stream.
    Loss is mean(BCE(logit_max, y), BCE(logit_attn, y)); final score = mean(sigmoids).
    """
    def __init__(self, embed_dim: int, attn_dim: int = 256) -> None: ...
    def forward(
        self, bag: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ...
```

### `AutoPoolAgg(nn.Module)` (US6)

```python
class AutoPoolAgg(nn.Module):
    """Scalar AutoPool (McFee, Salamon, Bello ICASSP 2018).

    pool(s) = sum_k softmax(alpha * s_k) * s_k     # alpha learnable scalar
    """
    def __init__(self, embed_dim: int) -> None: ...
    def forward(self, bag, mask): ...
```

### `ExpSoftmaxPoolAgg(nn.Module)` (US6)

```python
class ExpSoftmaxPoolAgg(nn.Module):
    """Exponential-softmax pool (Wang 1810.09050 §III.E).

    pool(s) = sum_k exp(s_k) * s_k / sum_k exp(s_k); s_k clamped to ±10.
    """
```

### `GMAPAgg(nn.Module)` (US6)

```python
class GMAPAgg(nn.Module):
    """Gated Multi-Head Attention Pooling (Hong et al.).

    For each head h:
        a_h = softmax(q_h^T tanh(V_h x))
        z_h = sum_k a_h[k] x[k]
    Final z = mean_h(sigmoid(g_h) * z_h); logit = head(z).
    """
    def __init__(self, embed_dim: int, n_heads: int = 4, attn_dim: int = 256) -> None: ...
```

`build_aggregator()` is extended to dispatch `dsmil`, `auto_pool`, `exp_softmax_pool`, `gmap`.

---

## 10. New configs

| File | US | Purpose |
|---|---|---|
| `mil/configs/wavlm_mil_tsmil_concat.yaml` | US4 | TS-MIL concat mode on WavLM-Base+ seen-child |
| `mil/configs/wavlm_mil_tsmil_film.yaml` | US4 | TS-MIL FiLM mode (ablation) |
| `mil/configs/whisper_mil_tsmil_concat.yaml` | US4 | TS-MIL concat on Whisper-small |
| `mil/configs/wavlm_mil_tsmil_concat_cross_child.yaml` | US4 | TS-MIL concat on cross-child split |
| `mil/configs/seg_mil_sweep.yaml` (modified) | US5+US6 | Add `dsmil`, `auto_pool`, `exp_softmax_pool`, `gmap` to `aggregators` |

Each TS-MIL config inherits its base from `wavlm_mil.yaml` / `whisper_mil.yaml` and adds:
```yaml
head: tsmil
tsmil_mode: concat                         # or "film"
prototype_cache: mil/prototypes/babar_vtc.npz
prototype_proj_dim: 64                     # concat-mode only
run_name: wavlm_mil_tsmil_concat
```

---

## 11. New output files

| File | US | Schema |
|---|---|---|
| `{run_dir}/missing_prototypes.json` | US4 | `{"missing_count": N, "missing_keys": [...]}` |
| `{run_dir}/predictions.csv` `attention_max_logit, attention_stream_logit` columns | US5 | per-stream raw logits added to existing CSV |
| `{run_dir}/config.json` `final_alpha` field | US6 (AutoPool) | learned scalar logged at end of training |
| `{run_dir}/head_attention.csv` | US6 (GMAP) | per-clip per-head attention weights wide CSV |

---

## 7. Backward-compatibility invariants

- `head` key absent → `gated_abmil` (current behavior).
- `layer_aggregation` key absent → `last` (current behavior).
- All existing configs (`wavlm_mil.yaml`, `whisper_mil.yaml`, `hubert_large_mil.yaml`, `wav2vec2_large_mil.yaml`, `wavlm_mil_hardneg.yaml`, `whisper_mil_hardneg.yaml`, `wavlm_mil_tinyvox.yaml`, `wavlm_mil_cross_child.yaml`, `whisper_mil_cross_child.yaml`) load and run unchanged.
- A re-run of `wavlm_mil` after spec-014 code lands MUST reproduce the committed `mil/mil_results/wavlm_mil/test_metrics_tuned.json` to within AUROC ±0.005, F1 ±0.01 (R5 in research.md).
