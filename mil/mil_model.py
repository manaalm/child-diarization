"""ABMIL model for clip-level child presence detection.

Architecture:
  BackboneExtractor  — frozen WavLM-base+ or Whisper-small
  GatedABMILHead     — gated attention pooling over instance embeddings (Ilse et al. 2018)
  MILModel           — composes both; processes a bag of audio windows
"""

from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import WavLMModel, WhisperModel, WhisperProcessor


class BackboneExtractor(nn.Module):
    """Frozen pre-trained audio encoder; produces frame-level embeddings."""

    def __init__(self, backbone_name: str, layer: int = -1, sample_rate: int = 16000) -> None:
        super().__init__()
        self.backbone_name = backbone_name
        self.layer = layer
        self.sample_rate = sample_rate
        self._is_whisper = "whisper" in backbone_name.lower()

        if self._is_whisper:
            self.model = WhisperModel.from_pretrained(backbone_name)
            self.processor = WhisperProcessor.from_pretrained(backbone_name)
            self.embed_dim = self.model.config.d_model
        else:
            self.model = WavLMModel.from_pretrained(backbone_name)
            self.embed_dim = self.model.config.hidden_size

        for p in self.model.parameters():
            p.requires_grad = False
        self.model.eval()

    @torch.no_grad()
    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """Encode a batch of waveforms → frame-level embeddings.

        Args:
            waveform: (B, 1, T) or (B, T) at self.sample_rate.

        Returns:
            Tensor (B, T_frames, embed_dim).
        """
        if waveform.dim() == 3:
            waveform = waveform.squeeze(1)  # (B, T)

        device = waveform.device

        if self._is_whisper:
            # Whisper needs mel-spectrogram input
            waveform_np = waveform.cpu().float().numpy()
            inputs = self.processor(
                waveform_np,
                sampling_rate=self.sample_rate,
                return_tensors="pt",
            )
            input_features = inputs["input_features"].to(device)
            out = self.model.encoder(input_features, output_hidden_states=True)
            hidden = out.hidden_states[self.layer]  # (B, T_frames, D)
        else:
            out = self.model(waveform, output_hidden_states=True)
            hidden = out.hidden_states[self.layer]  # (B, T_frames, D)

        return hidden


class GatedABMILHead(nn.Module):
    """Gated Attention-Based MIL (Ilse et al. 2018).

    A_k = softmax(w^T * (tanh(V * h_k) ⊙ σ(U * h_k)))
    z   = Σ_k A_k * h_k
    logit = head(z)
    """

    def __init__(self, in_dim: int = 768, hidden_dim: int = 256, dropout: float = 0.25) -> None:
        super().__init__()
        self.V = nn.Linear(in_dim, hidden_dim)
        self.U = nn.Linear(in_dim, hidden_dim)
        self.w = nn.Linear(hidden_dim, 1, bias=False)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(in_dim, 1)

    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Aggregate N instance embeddings into a bag-level prediction.

        Args:
            h: (N, in_dim) instance embeddings.

        Returns:
            logit: scalar (pre-sigmoid bag-level score).
            attn:  (N,) softmax attention weights.
        """
        h = self.drop(h)
        A = self.w(torch.tanh(self.V(h)) * torch.sigmoid(self.U(h)))  # (N, 1)
        A = F.softmax(A, dim=0)                                         # (N, 1)
        z = (A * h).sum(dim=0)                                          # (in_dim,)
        logit = self.head(z).squeeze()
        return logit, A.squeeze(1)


class MILModel(nn.Module):
    """Full MIL pipeline: backbone → mean-pool per window → GatedABMIL."""

    def __init__(self, backbone: BackboneExtractor, mil_head: GatedABMILHead) -> None:
        super().__init__()
        self.backbone = backbone
        self.mil_head = mil_head

    def forward(self, windows: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Process one bag of audio windows.

        Args:
            windows: List of (1, T) tensors, one per window.

        Returns:
            logit: scalar pre-sigmoid score.
            attn:  (N,) attention weights over windows.
        """
        device = next(self.mil_head.parameters()).device
        instance_embeddings = []

        for w in windows:
            w = w.unsqueeze(0).to(device)           # (1, 1, T)
            frames = self.backbone(w)                # (1, T_frames, D)
            emb = frames.mean(dim=1).squeeze(0)     # (D,)
            instance_embeddings.append(emb)

        h = torch.stack(instance_embeddings, dim=0)  # (N, D)
        return self.mil_head(h)

    def predict_bag(self, windows: List[torch.Tensor]) -> Tuple[float, List[float]]:
        """Return (score, attn_weights) with no grad. score ∈ [0, 1]."""
        self.eval()
        with torch.no_grad():
            logit, attn = self.forward(windows)
        return float(torch.sigmoid(logit).item()), attn.cpu().tolist()


def build_mil_model(cfg: dict) -> MILModel:
    """Instantiate a MILModel from a config dict."""
    # backbone_path overrides backbone (HuggingFace model name) with a local checkpoint
    backbone_src = cfg.get("backbone_path", cfg["backbone"])
    backbone = BackboneExtractor(
        backbone_name=backbone_src,
        layer=cfg.get("backbone_layer", -1),
        sample_rate=16000,
    )
    head = GatedABMILHead(
        in_dim=backbone.embed_dim,
        hidden_dim=cfg.get("mil_hidden_dim", 256),
        dropout=cfg.get("mil_dropout", 0.25),
    )
    return MILModel(backbone=backbone, mil_head=head)
