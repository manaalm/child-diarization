"""Aggregator heads for segment-instance MIL.

Purpose: Four MIL aggregators over variable-length bags of segment embeddings.
Inputs:  bag_tensor (K_max × D), mask (K_max,) bool.
Outputs: (logit: Tensor scalar, weights: Tensor[K_max] or None).
"""

import os
import sys
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)

from mil.mil_model import GatedABMILHead


class MeanAgg(nn.Module):
    """Masked mean pooling over instances, then a linear head."""

    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.head = nn.Linear(embed_dim, 1)

    def forward(self, bag: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, None]:
        # bag: (K_max, D), mask: (K_max,)
        k = mask.sum().clamp(min=1)
        pooled = (bag * mask.unsqueeze(1).float()).sum(dim=0) / k  # (D,)
        logit = self.head(pooled).squeeze()
        return logit, None


class MaxAgg(nn.Module):
    """Masked element-wise max pooling over instances, then a linear head."""

    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.head = nn.Linear(embed_dim, 1)

    def forward(self, bag: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, None]:
        # Replace masked positions with -inf before max
        masked_bag = bag.clone()
        masked_bag[~mask] = float("-inf")
        # If all instances are masked (empty bag), use zero vector
        if mask.sum() == 0:
            pooled = torch.zeros(bag.shape[1], device=bag.device, dtype=bag.dtype)
        else:
            pooled = masked_bag.max(dim=0).values  # (D,)
            # Replace any remaining -inf with 0 (shouldn't occur if mask.sum()>0)
            pooled = pooled.nan_to_num(neginf=0.0)
        logit = self.head(pooled).squeeze()
        return logit, None


class AttnAgg(nn.Module):
    """Standard ABMIL attention aggregation (Ilse et al. 2018).

    a_k = softmax_over_k( w^T · tanh(V · h_k) )
    z   = Σ_k a_k · h_k
    logit = head(z)
    """

    def __init__(self, embed_dim: int, attn_dim: int = 256) -> None:
        super().__init__()
        self.V = nn.Linear(embed_dim, attn_dim)
        self.w = nn.Linear(attn_dim, 1, bias=False)
        self.head = nn.Linear(embed_dim, 1)

    def forward(
        self, bag: torch.Tensor, mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # bag: (K_max, D), mask: (K_max,)
        scores = self.w(torch.tanh(self.V(bag))).squeeze(-1)  # (K_max,)
        # Mask out padding before softmax
        scores = scores.masked_fill(~mask, float("-inf"))
        if mask.sum() == 0:
            weights = torch.zeros_like(scores)
            z = torch.zeros(bag.shape[1], device=bag.device, dtype=bag.dtype)
        else:
            weights = F.softmax(scores, dim=0)  # (K_max,)
            z = (weights.unsqueeze(1) * bag).sum(dim=0)  # (D,)
        logit = self.head(z).squeeze()
        return logit, weights


class GatedAttnAgg(nn.Module):
    """Gated ABMIL aggregation (wraps existing GatedABMILHead).

    Uses the same gated attention formula as mil_model.py but with
    a masked softmax to handle variable-length bags.
    """

    def __init__(self, embed_dim: int, attn_dim: int = 256) -> None:
        super().__init__()
        # GatedABMILHead has its own linear head; we wrap it
        self._inner = GatedABMILHead(in_dim=embed_dim, hidden_dim=attn_dim)

    def forward(
        self, bag: torch.Tensor, mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # bag: (K_max, D), mask: (K_max,)
        if mask.sum() == 0:
            d = bag.shape[1]
            dummy = torch.zeros(1, d, device=bag.device, dtype=bag.dtype)
            logit, weights_dummy = self._inner(dummy)
            return logit, torch.zeros(bag.shape[0], device=bag.device, dtype=bag.dtype)

        # Select only real instances, run GatedABMILHead, then map weights back
        real_bag = bag[mask]  # (K, D)
        logit, weights_real = self._inner(real_bag)  # weights_real: (K,)

        # Scatter attention weights back into K_max positions
        weights = torch.zeros(bag.shape[0], device=bag.device, dtype=bag.dtype)
        weights[mask] = weights_real
        return logit, weights


class NoisyORAgg(nn.Module):
    """Probabilistic noisy-OR bag aggregation.

    P(bag=1) = 1 - prod_k(1 - sigma(logit_k)) over valid instances.
    Computed in log-space for numerical stability.
    """

    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.head = nn.Linear(embed_dim, 1)

    def forward(self, bag: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, None]:
        instance_logits = self.head(bag).squeeze(-1)  # (K_max,)
        probs = torch.sigmoid(instance_logits)  # (K_max,)
        # Masked positions contribute neutral probability 1.0 → log(1-1)=log(0) avoided by masking
        # Use log(1 - p_k) only for valid positions; sum; then bag_logit = logaddexp(0, sum)
        log_complement = torch.log((1.0 - probs).clamp(min=1e-8))  # (K_max,)
        log_complement = log_complement.masked_fill(~mask, 0.0)  # neutral for padding
        log_bag_complement = log_complement.sum()
        bag_logit = torch.logaddexp(torch.zeros(1, device=bag.device, dtype=bag.dtype).squeeze(), log_bag_complement)
        return bag_logit, None


class TopKAgg(nn.Module):
    """Top-k instance aggregation.

    Score each instance, select top-k (clamped to bag size), mean-pool their
    embeddings, then apply a final linear head.
    """

    def __init__(self, embed_dim: int, k: int = 3) -> None:
        super().__init__()
        self.k = k
        self.score_head = nn.Linear(embed_dim, 1)
        self.head = nn.Linear(embed_dim, 1)

    def forward(self, bag: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, None]:
        n_valid = int(mask.sum().clamp(min=1).item())
        k_actual = min(self.k, n_valid)
        scores = self.score_head(bag).squeeze(-1)  # (K_max,)
        scores = scores.masked_fill(~mask, float("-inf"))
        _, topk_idx = torch.topk(scores, k=k_actual, dim=0)
        pooled = bag[topk_idx].mean(dim=0)  # (D,)
        logit = self.head(pooled).squeeze()
        return logit, None


class TransformerAgg(nn.Module):
    """Transformer MIL aggregation with CLS token and learned positional embeddings.

    2-layer pre-norm transformer encoder. CLS token aggregates bag context.
    Attention weights are derived post-hoc from CLS-to-instance similarity for
    interpretability and sum to 1 over valid instances.
    """

    def __init__(
        self,
        embed_dim: int,
        num_layers: int = 2,
        num_heads: int = 4,
        ffn_dim: int = 1536,
        dropout: float = 0.3,
        k_max: int = 64,
    ) -> None:
        super().__init__()
        self.cls_token = nn.Parameter(torch.zeros(1, embed_dim))
        self.pos_embed = nn.Embedding(k_max + 1, embed_dim)  # position 0 = CLS
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True,  # pre-norm for stability
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Linear(embed_dim, 1)
        self._k_max = k_max
        nn.init.normal_(self.cls_token, std=0.02)

    def forward(
        self, bag: torch.Tensor, mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        K_max_padded, D = bag.shape

        # Collect valid (non-padded) indices, capped at self._k_max.
        # This avoids both the pos_embed OOB (dataset K_max >> k_max) and
        # quadratic attention over thousands of padding tokens.
        valid_idx = mask.nonzero(as_tuple=False).squeeze(-1)  # indices of real segments
        if valid_idx.shape[0] > self._k_max:
            valid_idx = valid_idx[: self._k_max]
        n_valid = valid_idx.shape[0]

        if n_valid > 0:
            bag_local = bag[valid_idx]  # (n_valid, D) — only real segments
        else:
            bag_local = bag.new_zeros(0, D)

        # Prepend CLS; add positional embeddings (positions 0..n_valid, always in-bounds)
        cls = self.cls_token.to(bag.dtype)                        # (1, D)
        seq = torch.cat([cls, bag_local], dim=0)                  # (n_valid+1, D)
        pos_ids = torch.arange(n_valid + 1, device=bag.device)   # always <= k_max
        seq = seq + self.pos_embed(pos_ids)

        # No padding mask needed — seq contains only real tokens + CLS
        enc = self.encoder(seq.unsqueeze(0))  # (1, n_valid+1, D)
        enc = enc.squeeze(0)                  # (n_valid+1, D)

        cls_out = enc[0]    # (D,)
        bag_out = enc[1:]   # (n_valid, D)

        logit = self.head(cls_out).squeeze()

        # Scatter interpretability weights back into K_max_padded positions
        scale = D ** -0.5
        weights = bag.new_zeros(K_max_padded)
        if n_valid > 0:
            scores = (cls_out @ bag_out.T) * scale  # (n_valid,)
            local_weights = F.softmax(scores, dim=0)
            weights[valid_idx] = local_weights

        return logit, weights


class DSMILAgg(nn.Module):
    """Dual-Stream MIL (Li et al. CVPR 2021) — spec-014 US5.

    Two streams share instance embeddings:
      - Max stream: scores every instance with a linear head, identifies the highest-
        scored "critical" instance m, and uses score_m as one bag logit (logit_max).
      - Attention stream: for each instance k, computes attention as a function of the
        difference vector h_k - h_m projected through a small MLP, then aggregates the
        bag embedding z = sum_k a_k h_k and produces logit_attn.

    Loss (computed in seg_train.py): mean(BCE(logit_max), BCE(logit_attn)).
    Final score (in seg_train.py): mean(sigmoid(logit_max), sigmoid(logit_attn)).

    Forward returns:
      logit_max:    scalar bag logit from the max stream.
      logit_attn:   scalar bag logit from the attention stream.
      attn_weights: (K_max,) attention weights of the attention stream (zero on
                    masked positions).
    """

    def __init__(self, embed_dim: int, attn_dim: int = 256) -> None:
        super().__init__()
        self.score_head = nn.Linear(embed_dim, 1)            # used by both streams
        self.W_attn = nn.Linear(embed_dim, attn_dim)
        self.q_attn = nn.Linear(attn_dim, 1, bias=False)
        self.bag_head = nn.Linear(embed_dim, 1)

    def forward(
        self, bag: torch.Tensor, mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        K_max, D = bag.shape
        if mask.sum() == 0:
            zero = torch.zeros((), device=bag.device, dtype=bag.dtype)
            return zero, zero, torch.zeros(K_max, device=bag.device, dtype=bag.dtype)

        instance_scores = self.score_head(bag).squeeze(-1)   # (K_max,)
        instance_scores = instance_scores.masked_fill(~mask, float("-inf"))

        # Critical instance (max stream)
        m_idx = int(torch.argmax(instance_scores).item())
        logit_max = instance_scores[m_idx]                    # scalar
        h_m = bag[m_idx]                                       # (D,)

        # Attention stream: attention as fn of distance to critical instance
        diffs = bag - h_m.unsqueeze(0)                         # (K_max, D)
        attn_scores = self.q_attn(torch.tanh(self.W_attn(diffs))).squeeze(-1)  # (K_max,)
        attn_scores = attn_scores.masked_fill(~mask, float("-inf"))
        weights = F.softmax(attn_scores, dim=0)               # (K_max,)

        z = (weights.unsqueeze(1) * bag).sum(dim=0)            # (D,)
        logit_attn = self.bag_head(z).squeeze()
        return logit_max, logit_attn, weights


class AutoPoolAgg(nn.Module):
    """Scalar AutoPool (McFee, Salamon, Bello, ICASSP 2018) — spec-014 US6.

    Pools instance scores via a learnable temperature `alpha`:
        weights_k = softmax(alpha * s_k) over valid instances
        bag_score = sum_k weights_k * s_k
    alpha = 0  → mean pool
    alpha → ∞  → max pool
    Initialized to alpha = 0.0 so first epoch is a mean pool.
    """

    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.score_head = nn.Linear(embed_dim, 1)
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, bag: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, None]:
        scores = self.score_head(bag).squeeze(-1)             # (K_max,)
        if mask.sum() == 0:
            return torch.zeros((), device=bag.device, dtype=bag.dtype), None
        # Apply alpha to raw (finite) scores first, then mask before softmax.
        # This avoids 0 * -inf = NaN at alpha=0.
        weighted = self.alpha * scores
        weighted = weighted.masked_fill(~mask, float("-inf"))
        weights = F.softmax(weighted, dim=0)
        bag_score = (weights * scores.masked_fill(~mask, 0.0)).sum()
        return bag_score, None

    def alpha_value(self) -> float:
        return float(self.alpha.detach().cpu().item())


class ExpSoftmaxPoolAgg(nn.Module):
    """Exponential softmax pool (Wang et al. arXiv:1810.09050 §III.E) — spec-014 US6.

    pool(s) = sum_k exp(s_k) * s_k / sum_k exp(s_k), with s_k clamped to ±10
    for numerical safety on bags with thousands of instances.
    """

    _LOGIT_CLAMP = 10.0

    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.score_head = nn.Linear(embed_dim, 1)

    def forward(self, bag: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, None]:
        scores = self.score_head(bag).squeeze(-1)
        scores = scores.clamp(min=-self._LOGIT_CLAMP, max=self._LOGIT_CLAMP)
        scores_masked = scores.masked_fill(~mask, float("-inf"))
        if mask.sum() == 0:
            return torch.zeros((), device=bag.device, dtype=bag.dtype), None
        weights = F.softmax(scores_masked, dim=0)
        bag_score = (weights * scores.masked_fill(~mask, 0.0)).sum()
        return bag_score, None


class GMAPAgg(nn.Module):
    """Gated Multi-Head Attention Pooling (Hong et al.) — spec-014 US6.

    For each head h:
        a_h = softmax(q_h^T tanh(V_h x))                # (K_max,) attention
        z_h = sum_k a_h[k] * x[k]                       # (D,) bag embedding
    Final z = mean_h(sigmoid(g_h) * z_h); logit = head(z).

    Each head has independent V_h, q_h linear layers; a single learnable scalar
    gate g_h per head; final linear bag head.

    Reported attention weights are the mean over heads (post-gating reweight).
    """

    def __init__(self, embed_dim: int, n_heads: int = 4, attn_dim: int = 256) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.V = nn.ModuleList([nn.Linear(embed_dim, attn_dim) for _ in range(n_heads)])
        self.q = nn.ModuleList([nn.Linear(attn_dim, 1, bias=False) for _ in range(n_heads)])
        self.gate = nn.Parameter(torch.zeros(n_heads))    # sigmoid(0) = 0.5 init
        self.head = nn.Linear(embed_dim, 1)

    def forward(
        self, bag: torch.Tensor, mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if mask.sum() == 0:
            d = bag.shape[1]
            zero_w = torch.zeros(bag.shape[0], device=bag.device, dtype=bag.dtype)
            zero_logit = torch.zeros((), device=bag.device, dtype=bag.dtype)
            return zero_logit, zero_w

        gates = torch.sigmoid(self.gate)                  # (n_heads,)
        bag_embeddings = []
        all_attns = []
        for h in range(self.n_heads):
            scores = self.q[h](torch.tanh(self.V[h](bag))).squeeze(-1)  # (K_max,)
            scores = scores.masked_fill(~mask, float("-inf"))
            attn = F.softmax(scores, dim=0)
            z_h = (attn.unsqueeze(1) * bag).sum(dim=0)    # (D,)
            bag_embeddings.append(z_h * gates[h])
            all_attns.append(attn)
        z = torch.stack(bag_embeddings, dim=0).mean(dim=0)  # (D,)
        logit = self.head(z).squeeze()
        attn_mean = torch.stack(all_attns, dim=0).mean(dim=0)  # (K_max,)
        return logit, attn_mean

    def head_attentions(
        self, bag: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """Return (n_heads, K_max) per-head attention vectors for interpretability."""
        if mask.sum() == 0:
            return torch.zeros(self.n_heads, bag.shape[0], device=bag.device, dtype=bag.dtype)
        out = []
        for h in range(self.n_heads):
            scores = self.q[h](torch.tanh(self.V[h](bag))).squeeze(-1)
            scores = scores.masked_fill(~mask, float("-inf"))
            out.append(F.softmax(scores, dim=0))
        return torch.stack(out, dim=0)


def build_aggregator(
    agg_name: str,
    embed_dim: int,
    attn_dim: int = 256,
    k: int = 3,
    transformer_config: Optional[dict] = None,
) -> nn.Module:
    if agg_name == "mean":
        return MeanAgg(embed_dim)
    if agg_name == "max":
        return MaxAgg(embed_dim)
    if agg_name == "attention":
        return AttnAgg(embed_dim, attn_dim)
    if agg_name == "gated_attention":
        return GatedAttnAgg(embed_dim, attn_dim)
    if agg_name == "noisy_or":
        return NoisyORAgg(embed_dim)
    if agg_name == "top_k":
        return TopKAgg(embed_dim, k=k)
    if agg_name == "transformer":
        tc = transformer_config or {}
        return TransformerAgg(
            embed_dim,
            num_layers=tc.get("num_layers", 2),
            num_heads=tc.get("num_heads", 4),
            ffn_dim=tc.get("ffn_dim", 1536),
            dropout=tc.get("dropout", 0.3),
            k_max=tc.get("k_max", 64),
        )
    if agg_name == "dsmil":
        return DSMILAgg(embed_dim, attn_dim)
    if agg_name == "auto_pool":
        return AutoPoolAgg(embed_dim)
    if agg_name == "exp_softmax_pool":
        return ExpSoftmaxPoolAgg(embed_dim)
    if agg_name == "gmap":
        return GMAPAgg(embed_dim, n_heads=4, attn_dim=attn_dim)
    raise ValueError(f"Unknown aggregator: {agg_name!r}")
