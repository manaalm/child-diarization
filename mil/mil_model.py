"""ABMIL model for clip-level child presence detection.

Architecture:
  BackboneExtractor  — frozen WavLM-base+ or Whisper-small. Optional learnable
                       softmax-weighted sum over transformer layers (spec-014 US1).
  GatedABMILHead     — gated attention pooling over instance embeddings (Ilse et al. 2018).
  ACMILHead          — Attention-Challenging MIL (Zhang et al. ECCV 2024): multi-branch
                       attention with cosine-diversity regularizer + Stochastic Top-K
                       Instance Masking. (spec-014 US3)
  TSMILHead          — Target-Speaker MIL: gated-ABMIL conditioned on a per-(child,
                       timepoint) ECAPA prototype via concat or FiLM. (spec-014 US4)
  MILModel           — composes backbone + head; processes a bag of audio windows.
"""

from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, WavLMModel, WhisperModel, WhisperProcessor


class BackboneExtractor(nn.Module):
    """Frozen pre-trained audio encoder; produces frame-level embeddings.

    Two layer-aggregation modes:
      - "last" (default, backward-compatible): return hidden_states[self.layer].
      - "weighted_sum": return softmax(layer_weights) @ hidden_states[1:] (or [:]
        if not skipping the conv-feature output). The layer_weights parameter is
        the only trainable backbone-side parameter; all transformer params remain
        frozen. Initialized to zeros so softmax is uniform on epoch 1.
    """

    def __init__(
        self,
        backbone_name: str,
        layer: int = -1,
        sample_rate: int = 16000,
        layer_aggregation: str = "last",
        layer_aggregation_skip_first: bool = True,
    ) -> None:
        super().__init__()
        self.backbone_name = backbone_name
        self.layer = layer
        self.sample_rate = sample_rate
        self.layer_aggregation = layer_aggregation
        self.layer_aggregation_skip_first = layer_aggregation_skip_first
        self._is_whisper = "whisper" in backbone_name.lower()

        if self._is_whisper:
            self.model = WhisperModel.from_pretrained(backbone_name)
            self.processor = WhisperProcessor.from_pretrained(backbone_name)
            self.embed_dim = self.model.config.d_model
            num_hidden_layers = self.model.config.encoder_layers
        else:
            # AutoModel covers WavLM, HuBERT, wav2vec2, and other HF speech encoders
            self.model = AutoModel.from_pretrained(backbone_name)
            self.embed_dim = self.model.config.hidden_size
            num_hidden_layers = self.model.config.num_hidden_layers

        for p in self.model.parameters():
            p.requires_grad = False
        self.model.eval()

        if layer_aggregation == "weighted_sum":
            n_to_combine = num_hidden_layers if layer_aggregation_skip_first else num_hidden_layers + 1
            # Trainable scalar weights, one per layer. softmax in forward().
            self.layer_weights = nn.Parameter(torch.zeros(n_to_combine))
            self._n_layers_combined = n_to_combine
        else:
            self.layer_weights = None
            self._n_layers_combined = 0

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

        # Backbone encoding is no-grad (frozen); layer_weights gradient flows through
        # the post-hoc softmax-weighted sum below.
        with torch.no_grad():
            if self._is_whisper:
                waveform_np = waveform.cpu().float().numpy()
                inputs = self.processor(
                    waveform_np,
                    sampling_rate=self.sample_rate,
                    return_tensors="pt",
                )
                input_features = inputs["input_features"].to(device)
                out = self.model.encoder(input_features, output_hidden_states=True)
            else:
                out = self.model(waveform, output_hidden_states=True)
            hidden_states = out.hidden_states  # tuple length L+1 (HF speech) or L (Whisper encoder)

        if self.layer_aggregation == "weighted_sum":
            # HF speech (WavLM/HuBERT/wav2vec2): hidden_states[0] is conv-feature output;
            # hidden_states[1:] are the transformer layers. Whisper encoder hidden_states
            # already start at the first transformer layer, but we keep the same
            # skip_first switch for symmetry — set to False for Whisper if needed.
            if self.layer_aggregation_skip_first and len(hidden_states) > self._n_layers_combined:
                stack = torch.stack(list(hidden_states[1:]), dim=0)
            else:
                stack = torch.stack(list(hidden_states), dim=0)
            # stack: (L_combined, B, T, D)
            w = torch.softmax(self.layer_weights, dim=0).view(-1, 1, 1, 1)
            hidden = (w * stack).sum(dim=0)  # (B, T, D)
        else:
            hidden = hidden_states[self.layer]  # (B, T, D)

        return hidden

    def layer_weights_softmax(self) -> Optional[List[float]]:
        """Return softmax(layer_weights) as a Python list, or None if not in weighted_sum mode."""
        if self.layer_weights is None:
            return None
        with torch.no_grad():
            return torch.softmax(self.layer_weights, dim=0).cpu().tolist()


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
        h = self.drop(h)
        A = self.w(torch.tanh(self.V(h)) * torch.sigmoid(self.U(h)))  # (N, 1)
        A = F.softmax(A, dim=0)
        z = (A * h).sum(dim=0)
        logit = self.head(z).squeeze()
        return logit, A.squeeze(1)


class ACMILHead(nn.Module):
    """Attention-Challenging MIL (Zhang et al. ECCV 2024).

    Two components:
      - Multiple Branch Attention (MBA): n_branches parallel gated-ABMIL attention
        vectors with a cosine-similarity diversity regularizer.
      - Stochastic Top-K Instance Masking (STKIM): with prob stkim_p, zero out the
        top-K attention positions per branch (training only), without re-softmax.

    Forward returns:
      logit:           bag logit (mean of per-branch bag logits).
      attn:            (N,) mean attention across branches.
      branch_attn:     (n_branches, N) per-branch attention vectors (post-STKIM at train).
      diversity_loss:  scalar (already multiplied by mba_diversity_weight).
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
        branch_aggregation: str = "mean",
        branch_topk: int = 2,
    ) -> None:
        super().__init__()
        self.n_branches = n_branches
        self.stkim_p = stkim_p
        self.stkim_k_frac = stkim_k_frac
        self.stkim_k_cap = stkim_k_cap
        self.mba_diversity_weight = mba_diversity_weight
        self.drop = nn.Dropout(dropout)
        self.branch_aggregation = branch_aggregation
        self.branch_topk = branch_topk

        self.V = nn.ModuleList([nn.Linear(in_dim, hidden_dim) for _ in range(n_branches)])
        self.U = nn.ModuleList([nn.Linear(in_dim, hidden_dim) for _ in range(n_branches)])
        self.w = nn.ModuleList([nn.Linear(hidden_dim, 1, bias=False) for _ in range(n_branches)])
        self.bag_heads = nn.ModuleList([nn.Linear(in_dim, 1) for _ in range(n_branches)])

        # Per-branch learned gate (only used when branch_aggregation == "gated").
        # Init to zero → sigmoid(0)=0.5 for all branches → equivalent to mean at init.
        if branch_aggregation == "gated":
            self.branch_gate = nn.Parameter(torch.zeros(n_branches))

    def _branch_attention(self, h: torch.Tensor, branch_idx: int) -> torch.Tensor:
        """Return (N,) softmax-normalized attention for one branch."""
        scores = self.w[branch_idx](
            torch.tanh(self.V[branch_idx](h)) * torch.sigmoid(self.U[branch_idx](h))
        ).squeeze(-1)  # (N,)
        return F.softmax(scores, dim=0)

    def _stkim_mask(self, attn: torch.Tensor) -> torch.Tensor:
        """STKIM: with prob stkim_p, zero out top-K attention positions (no re-softmax)."""
        if not self.training or self.stkim_p <= 0.0:
            return attn
        if torch.rand(1).item() >= self.stkim_p:
            return attn
        n = attn.shape[0]
        if n <= 1:
            return attn
        k = max(1, min(self.stkim_k_cap, int(self.stkim_k_frac * n)))
        if k >= n:
            return attn
        topk = torch.topk(attn, k=k, dim=0).indices
        masked = attn.clone()
        masked[topk] = 0.0
        return masked

    def forward_branches(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return per-branch bag logits and per-branch attention without aggregation.

        Used by `mil/eval_acmil_branch_selection.py` to evaluate each branch
        individually (no retraining needed). Mirrors the inner loop of `forward`
        but skips the mean and the diversity loss computation.
        """
        h_drop = self.drop(h)
        branch_logits = []
        branch_attns = []
        for b in range(self.n_branches):
            attn = self._branch_attention(h_drop, b)
            attn = self._stkim_mask(attn)
            z = (attn.unsqueeze(1) * h_drop).sum(dim=0)
            branch_logits.append(self.bag_heads[b](z).squeeze())
            branch_attns.append(attn)
        return torch.stack(branch_logits, dim=0), torch.stack(branch_attns, dim=0)

    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.drop(h)
        N = h.shape[0]
        branch_logits = []
        branch_attns = []
        for b in range(self.n_branches):
            attn = self._branch_attention(h, b)        # (N,)
            attn = self._stkim_mask(attn)
            z = (attn.unsqueeze(1) * h).sum(dim=0)     # (D,)
            branch_logits.append(self.bag_heads[b](z).squeeze())
            branch_attns.append(attn)

        branch_logits_t = torch.stack(branch_logits, dim=0)  # (B,)
        agg = getattr(self, "branch_aggregation", "mean")
        if agg == "max":
            logit = branch_logits_t.max()
        elif agg == "topk_mean":
            k = max(1, min(self.n_branches, getattr(self, "branch_topk", 2)))
            logit = torch.topk(branch_logits_t, k=k, dim=0).values.mean()
        elif agg == "gated":
            # learned per-branch gating (sigmoid over per-branch scalar parameter)
            gate = torch.sigmoid(self.branch_gate)  # (B,)
            logit = (gate * branch_logits_t).sum() / (gate.sum() + 1e-9)
        else:
            logit = branch_logits_t.mean()  # scalar
        branch_attn = torch.stack(branch_attns, dim=0)    # (n_branches, N)
        attn_mean = branch_attn.mean(dim=0)               # (N,)

        # Cosine-similarity diversity loss across branches (Zhang et al. Eq. 3)
        if self.n_branches > 1 and N > 1:
            A_norm = F.normalize(branch_attn, p=2, dim=1)             # (n_branches, N)
            sim = A_norm @ A_norm.T                                    # (n_branches, n_branches)
            off = ~torch.eye(self.n_branches, dtype=torch.bool, device=h.device)
            div_loss = sim[off].mean() * self.mba_diversity_weight
        else:
            div_loss = torch.zeros((), device=h.device, dtype=h.dtype)

        return logit, attn_mean, branch_attn, div_loss


class TSMILHead(nn.Module):
    """Target-Speaker MIL head — gated-ABMIL conditioned on an ECAPA prototype.

    mode="concat":  h_k' = [h_k ; W_p · prototype]  (concat the projected prototype
                    onto every instance embedding before the gated-attention head).
    mode="film":    gamma, beta = MLP(prototype); h_k' = gamma * h_k + beta
                    (feature-wise linear modulation per Perez et al. 2018).

    Forward signature:
      forward(h: (N, in_dim), prototype: (prototype_dim,)) -> (logit, attn)
    """

    def __init__(
        self,
        in_dim: int = 768,
        hidden_dim: int = 256,
        prototype_dim: int = 192,
        prototype_proj_dim: int = 64,
        mode: str = "concat",
        dropout: float = 0.25,
    ) -> None:
        super().__init__()
        if mode not in ("concat", "film"):
            raise ValueError(f"TSMILHead mode must be 'concat' or 'film', got {mode!r}")
        self.mode = mode
        self.prototype_dim = prototype_dim
        self.in_dim = in_dim

        if mode == "concat":
            self.prototype_proj = nn.Linear(prototype_dim, prototype_proj_dim)
            attn_in = in_dim + prototype_proj_dim
        else:  # film
            self.film_mlp = nn.Sequential(
                nn.Linear(prototype_dim, 256),
                nn.ReLU(),
                nn.Linear(256, 2 * in_dim),  # outputs gamma and beta concatenated
            )
            attn_in = in_dim

        # Gated-ABMIL attention over the conditioned embeddings
        self.V = nn.Linear(attn_in, hidden_dim)
        self.U = nn.Linear(attn_in, hidden_dim)
        self.w = nn.Linear(hidden_dim, 1, bias=False)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(attn_in, 1)

    def forward(
        self, h: torch.Tensor, prototype: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # h: (N, in_dim), prototype: (prototype_dim,)
        if self.mode == "concat":
            p_proj = self.prototype_proj(prototype)            # (prototype_proj_dim,)
            p_expanded = p_proj.unsqueeze(0).expand(h.shape[0], -1)  # (N, prototype_proj_dim)
            h_cond = torch.cat([h, p_expanded], dim=1)         # (N, in_dim + proj_dim)
        else:  # film
            film = self.film_mlp(prototype)                    # (2 * in_dim,)
            gamma, beta = film[: self.in_dim], film[self.in_dim:]  # each (in_dim,)
            h_cond = gamma.unsqueeze(0) * h + beta.unsqueeze(0)   # (N, in_dim)

        h_cond = self.drop(h_cond)
        A = self.w(torch.tanh(self.V(h_cond)) * torch.sigmoid(self.U(h_cond)))  # (N, 1)
        A = F.softmax(A, dim=0)
        z = (A * h_cond).sum(dim=0)
        logit = self.head(z).squeeze()
        return logit, A.squeeze(1)


class MILModel(nn.Module):
    """Full MIL pipeline: backbone → mean-pool per window → MIL head.

    The head's signature determines the model's forward signature:
      - GatedABMILHead/(legacy 2-tuple): returns (logit, attn).
      - ACMILHead (4-tuple): returns (logit, attn, branch_attn, div_loss).
      - TSMILHead (2-tuple, requires prototype kwarg): returns (logit, attn).
    """

    def __init__(self, backbone: BackboneExtractor, mil_head: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone
        self.mil_head = mil_head

    def _embed_windows(self, windows: List[torch.Tensor]) -> torch.Tensor:
        """Run a list of (1, T) windows through the backbone; return (N, D) instance embeddings."""
        device = next(self.mil_head.parameters()).device
        instance_embeddings = []
        for w in windows:
            w = w.unsqueeze(0).to(device)            # (1, 1, T)
            frames = self.backbone(w)                 # (1, T_frames, D)
            emb = frames.mean(dim=1).squeeze(0)      # (D,)
            instance_embeddings.append(emb)
        return torch.stack(instance_embeddings, dim=0)  # (N, D)

    def forward(
        self,
        windows: List[torch.Tensor],
        prototype: Optional[torch.Tensor] = None,
    ) -> Union[
        Tuple[torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ]:
        h = self._embed_windows(windows)
        if isinstance(self.mil_head, TSMILHead):
            if prototype is None:
                raise ValueError("TSMILHead requires a prototype tensor.")
            return self.mil_head(h, prototype)
        if isinstance(self.mil_head, ACMILHead):
            return self.mil_head(h)
        return self.mil_head(h)

    def predict_bag(
        self,
        windows: List[torch.Tensor],
        prototype: Optional[torch.Tensor] = None,
    ) -> Tuple[float, List[float]]:
        """Return (score, attn_weights) with no grad. score ∈ [0, 1]."""
        self.eval()
        with torch.no_grad():
            out = self.forward(windows, prototype=prototype)
        if len(out) == 4:
            logit, attn, _, _ = out
        else:
            logit, attn = out
        return float(torch.sigmoid(logit).item()), attn.cpu().tolist()


def build_mil_model(cfg: dict) -> MILModel:
    """Instantiate a MILModel from a config dict.

    Backward-compat: cfg without `head` key defaults to gated_abmil; cfg without
    `layer_aggregation` defaults to the legacy single-layer read.
    """
    backbone_src = cfg.get("backbone_path", cfg["backbone"])
    backbone = BackboneExtractor(
        backbone_name=backbone_src,
        layer=cfg.get("backbone_layer", -1),
        sample_rate=16000,
        layer_aggregation=cfg.get("layer_aggregation", "last"),
        layer_aggregation_skip_first=cfg.get("layer_aggregation_skip_first", True),
    )

    head_kind = cfg.get("head", "gated_abmil")
    if head_kind == "gated_abmil":
        head: nn.Module = GatedABMILHead(
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
            branch_aggregation=cfg.get("acmil_branch_aggregation", "mean"),
            branch_topk=cfg.get("acmil_branch_topk", 2),
        )
    elif head_kind == "tsmil":
        head = TSMILHead(
            in_dim=backbone.embed_dim,
            hidden_dim=cfg.get("mil_hidden_dim", 256),
            prototype_dim=cfg.get("prototype_dim", 192),
            prototype_proj_dim=cfg.get("prototype_proj_dim", 64),
            mode=cfg.get("tsmil_mode", "concat"),
            dropout=cfg.get("mil_dropout", 0.25),
        )
    else:
        raise ValueError(f"Unknown head: {head_kind!r}")

    return MILModel(backbone=backbone, mil_head=head)
