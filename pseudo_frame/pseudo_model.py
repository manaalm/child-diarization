"""WavLM-Base+ frame classifier for self-distilled child-vocalization detection.

Architecture:
  WavLM-Base+ (frozen)        → (B, T, 768)
  Layer norm                  → (B, T, 768)
  Linear(768 → hidden)        → (B, T, hidden)
  GELU + dropout              → (B, T, hidden)
  Linear(hidden → 1)          → (B, T, 1)
  → squeeze → frame logits     (B, T)

Output rate is 50 Hz (20 ms / frame).
Sigmoid gives per-frame probability of target-child speech.
Clip-level score = max-pool over frames (or top-k mean).
"""
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel


class FrameHead(nn.Module):
    def __init__(self, in_dim: int = 768, hidden_dim: int = 256, dropout: float = 0.2) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(in_dim)
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        x = self.norm(h)
        x = F.gelu(self.fc1(x))
        x = self.drop(x)
        return self.fc2(x).squeeze(-1)  # (B, T)


class PseudoFrameModel(nn.Module):
    """Frozen WavLM-Base+ backbone + per-frame logistic head."""

    def __init__(
        self,
        backbone_name: str = "microsoft/wavlm-base-plus",
        backbone_layer: int = -1,
        hidden_dim: int = 256,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.backbone = AutoModel.from_pretrained(backbone_name)
        self.backbone_layer = backbone_layer
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.backbone.eval()
        in_dim = self.backbone.config.hidden_size  # 768 for WavLM-Base+
        self.head = FrameHead(in_dim=in_dim, hidden_dim=hidden_dim, dropout=dropout)

    @torch.no_grad()
    def encode(self, waveform: torch.Tensor) -> torch.Tensor:
        """Run frozen backbone. waveform: (B, T) → (B, T_frames, D)."""
        out = self.backbone(waveform, output_hidden_states=True)
        return out.hidden_states[self.backbone_layer]

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """waveform: (B, T) → frame_logits (B, T_frames)."""
        h = self.encode(waveform)
        return self.head(h)

    def clip_score(self, frame_probs: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        """Aggregate frame probs to clip-level via masked max + top-k mean.

        Returns max-pool clip score (most discriminative for child-presence).
        """
        masked = frame_probs.masked_fill(valid < 0.5, float("-inf"))
        return masked.max(dim=1).values
