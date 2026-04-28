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
    raise ValueError(f"Unknown aggregator: {agg_name!r}")
