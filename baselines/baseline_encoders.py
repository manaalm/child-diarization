import os
import json
import random
from dataclasses import dataclass, asdict, replace, field
from typing import Optional, Dict, Any, Tuple, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchaudio
from torch.utils.data import Dataset, DataLoader
from transformers import (
    WavLMModel,
    Wav2Vec2FeatureExtractor,
    WhisperModel,
    WhisperProcessor,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
)


# =========================================================
# Config
# =========================================================

@dataclass
class Config:
    annotations_csv: str = "/orcd/scratch/bcs/001/sensein/sails/BIDS_data/anotated_processed.csv"
    results_root: str = "./baseline_results"

    # --- NEW: model_type now supports "fused" ---
    model_type: str = "whisper"   # "whisper", "wavlm", or "fused"
    whisper_name: str = "openai/whisper-small"
    wavlm_name: str = "microsoft/wavlm-base-plus"

    sample_rate: int = 16000
    max_seconds: Optional[float] = 90.0
    batch_size: int = 8
    num_workers: int = 4

    lr_head: float = 1e-3
    lr_backbone: float = 1e-5
    weight_decay: float = 1e-4
    epochs: int = 20

    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    freeze_backbone: bool = True
    unfreeze_last_n_layers: int = 0

    # --- CHANGED: pooling now supports "stats" ---
    pooling: str = "mean"   # "mean", "attn", or "stats"
    # --- NEW: layer-weighted sum toggle ---
    use_layer_weights: bool = False

    use_timepoint_feature: bool = True
    dropout: float = 0.3
    hidden_dim: int = 256

    positive_class_weight: Optional[float] = None
    gradient_clip: float = 1.0

    split_dir: str = "./splits"
    split_seed: int = 42
    train_frac: float = 0.70
    val_frac: float = 0.15
    test_frac: float = 0.15

    # --- NEW: data augmentation flags ---
    speed_perturb: bool = False          # 0.9x / 1.1x speed perturbation
    random_crop: bool = False            # random window each epoch instead of first N seconds

    # --- NEW: per-timepoint threshold tuning ---
    per_timepoint_threshold: bool = False

    experiment_name: str = "whisper_mean"
    save_path: str = "whisper_mean_best_model.pt"


CFG = Config()


# =========================================================
# Metadata prep  (unchanged)
# =========================================================

def bidsprocessed_to_audio_path(bids_processed_path: str) -> str:
    if pd.isna(bids_processed_path):
        return ""
    s = str(bids_processed_path).strip()
    suffix = "_desc-processed_beh.mp4"
    if not s.endswith(suffix):
        return ""
    return s[:-len(suffix)] + "_audio.wav"


def normalize_timepoint(tp: str) -> Optional[str]:
    if pd.isna(tp):
        return None
    tp = str(tp).strip()
    if tp in {"14_month", "36_month"}:
        return tp
    return None


def vocalizations_to_label(v) -> Optional[int]:
    if pd.isna(v):
        return None
    s = str(v).strip().lower()
    if s == "yes":
        return 1
    if s == "no":
        return 0
    try:
        x = float(s)
        if x == 1:
            return 1
        if x == 0:
            return 0
    except Exception:
        pass
    return None


def build_master_dataframe(cfg: Config) -> pd.DataFrame:
    df = pd.read_csv(cfg.annotations_csv)

    required_cols = ["BidsProcessed", "ID", "timepoint", "Vocalizations"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in annotations CSV: {missing}")

    out = df.copy()
    out["audio_path"] = out["BidsProcessed"].apply(bidsprocessed_to_audio_path)
    out["child_id"] = out["ID"].astype(str).str.strip()
    out["timepoint_norm"] = out["timepoint"].apply(normalize_timepoint)
    out["label"] = out["Vocalizations"].apply(vocalizations_to_label)

    out = out[out["timepoint_norm"].notna()].copy()
    out = out[out["audio_path"].astype(str) != ""].copy()
    out = out[out["child_id"].astype(str) != ""].copy()
    out = out[out["label"].notna()].copy()
    out["label"] = out["label"].astype(int)

    out["audio_exists"] = out["audio_path"].apply(os.path.exists)
    out = out[out["audio_exists"]].copy()

    out["timepoint_feature"] = out["timepoint_norm"].map({
        "14_month": 0.0,
        "36_month": 1.0,
    })

    out = out.reset_index(drop=True)
    return out


# =========================================================
# Reusable group split by child ID  (unchanged)
# =========================================================

def make_reusable_group_split(
    df: pd.DataFrame,
    split_dir: str,
    seed: int = 42,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
) -> pd.DataFrame:
    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-8
    os.makedirs(split_dir, exist_ok=True)

    groups = df["child_id"].values
    idx = np.arange(len(df))

    gss1 = GroupShuffleSplit(n_splits=1, train_size=train_frac, random_state=seed)
    train_idx, temp_idx = next(gss1.split(idx, groups=groups))
    train_df = df.iloc[train_idx].copy()
    temp_df = df.iloc[temp_idx].copy()

    rel_val = val_frac / (val_frac + test_frac)
    gss2 = GroupShuffleSplit(n_splits=1, train_size=rel_val, random_state=seed + 1)
    temp_groups = temp_df["child_id"].values
    temp_idx2 = np.arange(len(temp_df))
    val_idx2, test_idx2 = next(gss2.split(temp_idx2, groups=temp_groups))

    val_df = temp_df.iloc[val_idx2].copy()
    test_df = temp_df.iloc[test_idx2].copy()

    train_df["split"] = "train"
    val_df["split"] = "val"
    test_df["split"] = "test"

    full = pd.concat([train_df, val_df, test_df], axis=0).sort_index().reset_index(drop=True)

    full.to_csv(os.path.join(split_dir, "master_with_split.csv"), index=False)
    train_df.to_csv(os.path.join(split_dir, "train.csv"), index=False)
    val_df.to_csv(os.path.join(split_dir, "val.csv"), index=False)
    test_df.to_csv(os.path.join(split_dir, "test.csv"), index=False)

    summary = {
        "seed": seed,
        "n_total_rows": int(len(full)),
        "n_train_rows": int(len(train_df)),
        "n_val_rows": int(len(val_df)),
        "n_test_rows": int(len(test_df)),
        "n_train_children": int(train_df["child_id"].nunique()),
        "n_val_children": int(val_df["child_id"].nunique()),
        "n_test_children": int(test_df["child_id"].nunique()),
        "train_timepoints": train_df["timepoint_norm"].value_counts().to_dict(),
        "val_timepoints": val_df["timepoint_norm"].value_counts().to_dict(),
        "test_timepoints": test_df["timepoint_norm"].value_counts().to_dict(),
        "train_labels": train_df["label"].value_counts().to_dict(),
        "val_labels": val_df["label"].value_counts().to_dict(),
        "test_labels": test_df["label"].value_counts().to_dict(),
    }
    with open(os.path.join(split_dir, "split_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    return full


def load_or_create_split(cfg: Config) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    split_master = os.path.join(cfg.split_dir, "master_with_split.csv")
    if os.path.exists(split_master):
        full = pd.read_csv(split_master)
    else:
        df = build_master_dataframe(cfg)
        full = make_reusable_group_split(
            df,
            split_dir=cfg.split_dir,
            seed=cfg.split_seed,
            train_frac=cfg.train_frac,
            val_frac=cfg.val_frac,
            test_frac=cfg.test_frac,
        )

    train_df = full[full["split"] == "train"].reset_index(drop=True)
    val_df = full[full["split"] == "val"].reset_index(drop=True)
    test_df = full[full["split"] == "test"].reset_index(drop=True)
    return train_df, val_df, test_df


# =========================================================
# Dataset  — CHANGED: added speed_perturb + random_crop
# =========================================================

class ChildVocalizationDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        sample_rate: int = 16000,
        max_seconds: Optional[float] = 30.0,
        speed_perturb: bool = False,
        random_crop: bool = False,
    ):
        self.df = df.reset_index(drop=True).copy()
        self.sample_rate = sample_rate
        self.max_seconds = max_seconds
        self.speed_perturb = speed_perturb
        self.random_crop = random_crop

    def __len__(self):
        return len(self.df)

    def _load_audio(self, path: str) -> torch.Tensor:
        wav, sr = torchaudio.load(path)

        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)

        if sr != self.sample_rate:
            wav = torchaudio.functional.resample(wav, sr, self.sample_rate)

        wav = wav.squeeze(0)

        # --- NEW: speed perturbation (0.9x or 1.1x) ---
        if self.speed_perturb:
            factor = random.choice([0.9, 1.0, 1.1])
            if factor != 1.0:
                new_sr = int(self.sample_rate * factor)
                wav = torchaudio.functional.resample(wav, self.sample_rate, new_sr)
                wav = torchaudio.functional.resample(wav, new_sr, self.sample_rate)

        # --- CHANGED: random_crop vs. front-truncation ---
        if self.max_seconds is not None:
            max_len = int(self.sample_rate * self.max_seconds)
            if wav.numel() > max_len:
                if self.random_crop:
                    start = random.randint(0, wav.numel() - max_len)
                    wav = wav[start : start + max_len]
                else:
                    wav = wav[:max_len]

        return wav

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        wav = self._load_audio(row["audio_path"])

        return {
            "waveform": wav,
            "label": torch.tensor(float(row["label"]), dtype=torch.float32),
            "child_id": str(row["child_id"]),
            "timepoint_feature": torch.tensor(float(row["timepoint_feature"]), dtype=torch.float32),
            "audio_path": row["audio_path"],
            "timepoint": row["timepoint_norm"],
        }


# =========================================================
# Collators  (unchanged)
# =========================================================

class WhisperCollator:
    def __init__(self, processor: WhisperProcessor, sample_rate: int = 16000):
        self.processor = processor
        self.sample_rate = sample_rate

    def __call__(self, batch):
        waveforms = [item["waveform"].numpy() for item in batch]
        proc = self.processor(waveforms, sampling_rate=self.sample_rate, return_tensors="pt")
        return {
            "inputs": proc.input_features,
            "labels": torch.stack([item["label"] for item in batch]),
            "timepoint_features": torch.stack([item["timepoint_feature"] for item in batch]),
            "child_ids": [item["child_id"] for item in batch],
            "audio_paths": [item["audio_path"] for item in batch],
            "timepoints": [item["timepoint"] for item in batch],
        }


class WavLMCollator:
    def __init__(self, processor: Wav2Vec2FeatureExtractor, sample_rate: int = 16000):
        self.processor = processor
        self.sample_rate = sample_rate

    def __call__(self, batch):
        waveforms = [item["waveform"].numpy() for item in batch]
        proc = self.processor(
            waveforms,
            sampling_rate=self.sample_rate,
            return_tensors="pt",
            padding=True,
            return_attention_mask=True,
        )
        return {
            "inputs": proc.input_values,
            "attention_mask": proc.get("attention_mask", None),
            "labels": torch.stack([item["label"] for item in batch]),
            "timepoint_features": torch.stack([item["timepoint_feature"] for item in batch]),
            "child_ids": [item["child_id"] for item in batch],
            "audio_paths": [item["audio_path"] for item in batch],
            "timepoints": [item["timepoint"] for item in batch],
        }


# --- NEW: FusedCollator runs both processors ---
class FusedCollator:
    """Produces both Whisper mel features and WavLM waveform inputs."""
    def __init__(self, whisper_processor, wavlm_processor, sample_rate: int = 16000):
        self.whisper_proc = whisper_processor
        self.wavlm_proc = wavlm_processor
        self.sample_rate = sample_rate

    def __call__(self, batch):
        waveforms_np = [item["waveform"].numpy() for item in batch]

        whisper_out = self.whisper_proc(waveforms_np, sampling_rate=self.sample_rate, return_tensors="pt")
        wavlm_out = self.wavlm_proc(
            waveforms_np,
            sampling_rate=self.sample_rate,
            return_tensors="pt",
            padding=True,
            return_attention_mask=True,
        )

        return {
            "whisper_inputs": whisper_out.input_features,
            "wavlm_inputs": wavlm_out.input_values,
            "wavlm_attention_mask": wavlm_out.get("attention_mask", None),
            "labels": torch.stack([item["label"] for item in batch]),
            "timepoint_features": torch.stack([item["timepoint_feature"] for item in batch]),
            "child_ids": [item["child_id"] for item in batch],
            "audio_paths": [item["audio_path"] for item in batch],
            "timepoints": [item["timepoint"] for item in batch],
        }


# =========================================================
# NEW: Layer-weighted sum
# =========================================================

class LayerWeightedSum(nn.Module):
    """Learnable weighted combination of all transformer layers."""
    def __init__(self, num_layers: int):
        super().__init__()
        self.weights = nn.Parameter(torch.zeros(num_layers))

    def forward(self, hidden_states: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        # hidden_states: tuple of (B, T, D), one per layer
        w = torch.softmax(self.weights, dim=0)
        stacked = torch.stack(hidden_states, dim=0)   # (L, B, T, D)
        return (w[:, None, None, None] * stacked).sum(dim=0)  # (B, T, D)


# =========================================================
# Pooling — ADDED StatisticalPooling
# =========================================================

class AttentivePooling(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.score = nn.Linear(dim, 1)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        scores = self.score(x).squeeze(-1)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        weights = torch.softmax(scores, dim=-1)
        pooled = torch.bmm(weights.unsqueeze(1), x).squeeze(1)
        return pooled

    @property
    def output_dim_multiplier(self) -> int:
        return 1


class MeanPooling(nn.Module):
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if mask is None:
            return x.mean(dim=1)
        mask = mask.float().unsqueeze(-1)
        x = x * mask
        denom = mask.sum(dim=1).clamp(min=1e-6)
        return x.sum(dim=1) / denom

    @property
    def output_dim_multiplier(self) -> int:
        return 1


class StatisticalPooling(nn.Module):
    """Mean + std pooling — doubles the output dimension."""
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if mask is not None:
            mask_f = mask.float().unsqueeze(-1)
            x = x * mask_f
            lengths = mask_f.sum(dim=1).clamp(min=1e-6)
            mean = x.sum(dim=1) / lengths
            var = ((x - mean.unsqueeze(1)) * mask_f).pow(2).sum(dim=1) / lengths
        else:
            mean = x.mean(dim=1)
            var = x.var(dim=1, unbiased=False)
        std = (var + 1e-6).sqrt()
        return torch.cat([mean, std], dim=-1)

    @property
    def output_dim_multiplier(self) -> int:
        return 2


def make_pooling(pooling: str, dim: int) -> nn.Module:
    """Factory that returns the pooling module."""
    if pooling == "attn":
        return AttentivePooling(dim)
    elif pooling == "stats":
        return StatisticalPooling()
    else:
        return MeanPooling()


# =========================================================
# Models — CHANGED: layer weights + stats pooling support
# =========================================================

class ClipClassifierHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 256, dropout: float = 0.3, use_timepoint_feature: bool = True):
        super().__init__()
        self.use_timepoint_feature = use_timepoint_feature
        total_in = input_dim + (1 if use_timepoint_feature else 0)
        self.net = nn.Sequential(
            nn.Linear(total_in, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, pooled: torch.Tensor, timepoint_feature: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.use_timepoint_feature:
            pooled = torch.cat([pooled, timepoint_feature.unsqueeze(-1)], dim=-1)
        return self.net(pooled).squeeze(-1)


class WhisperDirectModel(nn.Module):
    def __init__(self, model_name: str, pooling: str, hidden_dim: int, dropout: float,
                 use_timepoint_feature: bool, freeze_backbone: bool,
                 use_layer_weights: bool = False):
        super().__init__()
        self.backbone = WhisperModel.from_pretrained(model_name)
        d = self.backbone.config.d_model
        num_layers = self.backbone.config.encoder_layers  # 12 for whisper-small

        # --- NEW: optional layer-weighted sum ---
        self.use_layer_weights = use_layer_weights
        if use_layer_weights:
            self.layer_mix = LayerWeightedSum(num_layers)

        self.pool = make_pooling(pooling, d)
        pool_out_dim = d * self.pool.output_dim_multiplier

        self.head = ClipClassifierHead(
            input_dim=pool_out_dim,          # <-- accounts for stats doubling
            hidden_dim=hidden_dim,
            dropout=dropout,
            use_timepoint_feature=use_timepoint_feature,
        )
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward(self, input_features: torch.Tensor, timepoint_feature: Optional[torch.Tensor] = None):
        # --- CHANGED: request all hidden states when using layer weights ---
        out = self.backbone.encoder(
            input_features=input_features,
            output_hidden_states=self.use_layer_weights,
        )

        if self.use_layer_weights:
            h = self.layer_mix(out.hidden_states[1:])  # skip embedding layer
        else:
            h = out.last_hidden_state

        pooled = self.pool(h)
        return self.head(pooled, timepoint_feature)


class WavLMDirectModel(nn.Module):
    def __init__(self, model_name: str, pooling: str, hidden_dim: int, dropout: float,
                 use_timepoint_feature: bool, freeze_backbone: bool,
                 use_layer_weights: bool = False):
        super().__init__()
        self.backbone = WavLMModel.from_pretrained(model_name)
        d = self.backbone.config.hidden_size
        num_layers = self.backbone.config.num_hidden_layers  # 12 for wavlm-base-plus

        # --- NEW: optional layer-weighted sum ---
        self.use_layer_weights = use_layer_weights
        if use_layer_weights:
            self.layer_mix = LayerWeightedSum(num_layers)

        self.pool = make_pooling(pooling, d)
        pool_out_dim = d * self.pool.output_dim_multiplier

        self.head = ClipClassifierHead(
            input_dim=pool_out_dim,          # <-- accounts for stats doubling
            hidden_dim=hidden_dim,
            dropout=dropout,
            use_timepoint_feature=use_timepoint_feature,
        )
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward(
        self,
        input_values: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        timepoint_feature: Optional[torch.Tensor] = None,
    ):
        # --- CHANGED: request all hidden states when using layer weights ---
        out = self.backbone(
            input_values=input_values,
            attention_mask=attention_mask,
            output_hidden_states=self.use_layer_weights,
        )

        if self.use_layer_weights:
            h = self.layer_mix(out.hidden_states[1:])  # skip embedding layer
        else:
            h = out.last_hidden_state

        feature_attention_mask = None
        if attention_mask is not None:
            feature_attention_mask = self.backbone._get_feature_vector_attention_mask(
                h.shape[1], attention_mask
            )

        pooled = self.pool(h, feature_attention_mask)
        return self.head(pooled, timepoint_feature)


# =========================================================
# NEW: Fused model (Whisper + WavLM)
# =========================================================

class FusedModel(nn.Module):
    """Run both encoders, pool each independently, concat, classify."""
    def __init__(
        self,
        whisper_name: str,
        wavlm_name: str,
        pooling: str,
        hidden_dim: int,
        dropout: float,
        use_timepoint_feature: bool,
        freeze_backbone: bool,
        use_layer_weights: bool = False,
    ):
        super().__init__()

        # --- Whisper encoder ---
        self.whisper = WhisperModel.from_pretrained(whisper_name)
        d_w = self.whisper.config.d_model
        n_w = self.whisper.config.encoder_layers

        # --- WavLM encoder ---
        self.wavlm = WavLMModel.from_pretrained(wavlm_name)
        d_v = self.wavlm.config.hidden_size
        n_v = self.wavlm.config.num_hidden_layers

        # --- Layer mixing ---
        self.use_layer_weights = use_layer_weights
        if use_layer_weights:
            self.whisper_mix = LayerWeightedSum(n_w)
            self.wavlm_mix = LayerWeightedSum(n_v)

        # --- Pooling (one per encoder) ---
        self.whisper_pool = make_pooling(pooling, d_w)
        self.wavlm_pool = make_pooling(pooling, d_v)

        whisper_pool_dim = d_w * self.whisper_pool.output_dim_multiplier
        wavlm_pool_dim = d_v * self.wavlm_pool.output_dim_multiplier
        concat_dim = whisper_pool_dim + wavlm_pool_dim

        self.head = ClipClassifierHead(
            input_dim=concat_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            use_timepoint_feature=use_timepoint_feature,
        )

        if freeze_backbone:
            for p in self.whisper.parameters():
                p.requires_grad = False
            for p in self.wavlm.parameters():
                p.requires_grad = False

    def forward(
        self,
        whisper_inputs: torch.Tensor,
        wavlm_inputs: torch.Tensor,
        wavlm_attention_mask: Optional[torch.Tensor] = None,
        timepoint_feature: Optional[torch.Tensor] = None,
    ):
        # Whisper
        w_out = self.whisper.encoder(
            input_features=whisper_inputs,
            output_hidden_states=self.use_layer_weights,
        )
        if self.use_layer_weights:
            h_w = self.whisper_mix(w_out.hidden_states[1:])
        else:
            h_w = w_out.last_hidden_state
        pooled_w = self.whisper_pool(h_w)

        # WavLM
        v_out = self.wavlm(
            input_values=wavlm_inputs,
            attention_mask=wavlm_attention_mask,
            output_hidden_states=self.use_layer_weights,
        )
        if self.use_layer_weights:
            h_v = self.wavlm_mix(v_out.hidden_states[1:])
        else:
            h_v = v_out.last_hidden_state

        feat_mask = None
        if wavlm_attention_mask is not None:
            feat_mask = self.wavlm._get_feature_vector_attention_mask(
                h_v.shape[1], wavlm_attention_mask
            )
        pooled_v = self.wavlm_pool(h_v, feat_mask)

        # Concatenate and classify
        fused = torch.cat([pooled_w, pooled_v], dim=-1)
        return self.head(fused, timepoint_feature)


def unfreeze_last_n_layers(model: nn.Module, model_type: str, n: int):
    if n <= 0:
        return
    if model_type == "fused":
        # Unfreeze last n layers of both encoders
        for enc in [model.whisper.encoder.layers, model.wavlm.encoder.layers]:
            for layer in enc[-n:]:
                for p in layer.parameters():
                    p.requires_grad = True
    else:
        layers = model.backbone.encoder.layers
        for layer in layers[-n:]:
            for p in layer.parameters():
                p.requires_grad = True


# =========================================================
# Build loaders / optim — CHANGED: supports fused + augmentation flags
# =========================================================

def build_model_and_loaders(cfg: Config, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame):
    # Only training set gets augmentation
    train_ds = ChildVocalizationDataset(
        train_df, cfg.sample_rate, cfg.max_seconds,
        speed_perturb=cfg.speed_perturb,
        random_crop=cfg.random_crop,
    )
    val_ds = ChildVocalizationDataset(val_df, cfg.sample_rate, cfg.max_seconds)
    test_ds = ChildVocalizationDataset(test_df, cfg.sample_rate, cfg.max_seconds)

    if cfg.model_type == "whisper":
        processor = WhisperProcessor.from_pretrained(cfg.whisper_name)
        collate_fn = WhisperCollator(processor, cfg.sample_rate)
        model = WhisperDirectModel(
            model_name=cfg.whisper_name,
            pooling=cfg.pooling,
            hidden_dim=cfg.hidden_dim,
            dropout=cfg.dropout,
            use_timepoint_feature=cfg.use_timepoint_feature,
            freeze_backbone=cfg.freeze_backbone,
            use_layer_weights=cfg.use_layer_weights,
        )

    elif cfg.model_type == "wavlm":
        processor = Wav2Vec2FeatureExtractor.from_pretrained(cfg.wavlm_name)
        collate_fn = WavLMCollator(processor, cfg.sample_rate)
        model = WavLMDirectModel(
            model_name=cfg.wavlm_name,
            pooling=cfg.pooling,
            hidden_dim=cfg.hidden_dim,
            dropout=cfg.dropout,
            use_timepoint_feature=cfg.use_timepoint_feature,
            freeze_backbone=cfg.freeze_backbone,
            use_layer_weights=cfg.use_layer_weights,
        )

    # --- NEW: fused model path ---
    elif cfg.model_type == "fused":
        whisper_processor = WhisperProcessor.from_pretrained(cfg.whisper_name)
        wavlm_processor = Wav2Vec2FeatureExtractor.from_pretrained(cfg.wavlm_name)
        collate_fn = FusedCollator(whisper_processor, wavlm_processor, cfg.sample_rate)
        model = FusedModel(
            whisper_name=cfg.whisper_name,
            wavlm_name=cfg.wavlm_name,
            pooling=cfg.pooling,
            hidden_dim=cfg.hidden_dim,
            dropout=cfg.dropout,
            use_timepoint_feature=cfg.use_timepoint_feature,
            freeze_backbone=cfg.freeze_backbone,
            use_layer_weights=cfg.use_layer_weights,
        )
    else:
        raise ValueError("cfg.model_type must be 'whisper', 'wavlm', or 'fused'")

    unfreeze_last_n_layers(model, cfg.model_type, cfg.unfreeze_last_n_layers)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=cfg.num_workers, pin_memory=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                            num_workers=cfg.num_workers, pin_memory=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False,
                             num_workers=cfg.num_workers, pin_memory=True, collate_fn=collate_fn)

    return model, train_loader, val_loader, test_loader


def build_optimizer(model: nn.Module, cfg: Config):
    head_params, backbone_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        # layer_mix weights count as head-lr since they're new trainable params
        if "head" in name or "layer_mix" in name:
            head_params.append(p)
        else:
            backbone_params.append(p)

    groups = []
    if head_params:
        groups.append({"params": head_params, "lr": cfg.lr_head})
    if backbone_params:
        groups.append({"params": backbone_params, "lr": cfg.lr_backbone})

    return torch.optim.AdamW(groups, weight_decay=cfg.weight_decay)


# =========================================================
# Metrics / prediction helpers
# =========================================================

def compute_metrics(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)

    metrics = {
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
    }

    try:
        metrics["auroc"] = float(roc_auc_score(y_true, y_prob))
    except Exception:
        metrics["auroc"] = float("nan")

    try:
        metrics["auprc"] = float(average_precision_score(y_true, y_prob))
    except Exception:
        metrics["auprc"] = float("nan")

    return metrics


def _run_forward(model, batch, device, model_type):
    """Shared forward-pass logic for any model_type."""
    timepoint_features = batch["timepoint_features"].to(device)

    if model_type == "whisper":
        inputs = batch["inputs"].to(device)
        logits = model(inputs, timepoint_feature=timepoint_features)

    elif model_type == "wavlm":
        inputs = batch["inputs"].to(device)
        attn = batch["attention_mask"]
        if attn is not None:
            attn = attn.to(device)
        logits = model(inputs, attention_mask=attn, timepoint_feature=timepoint_features)

    elif model_type == "fused":
        w_in = batch["whisper_inputs"].to(device)
        v_in = batch["wavlm_inputs"].to(device)
        v_mask = batch["wavlm_attention_mask"]
        if v_mask is not None:
            v_mask = v_mask.to(device)
        logits = model(w_in, v_in, wavlm_attention_mask=v_mask, timepoint_feature=timepoint_features)

    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    return logits


@torch.no_grad()
def collect_predictions(model, loader, criterion, device, model_type):
    model.eval()
    total_loss = 0.0
    rows = []

    for batch in loader:
        labels = batch["labels"].to(device)
        logits = _run_forward(model, batch, device, model_type)

        loss = criterion(logits, labels)
        probs = torch.sigmoid(logits)
        total_loss += loss.item() * labels.size(0)

        labels_np = labels.cpu().numpy()
        probs_np = probs.cpu().numpy()

        for i in range(len(labels_np)):
            rows.append({
                "audio_path": batch["audio_paths"][i],
                "child_id": batch["child_ids"][i],
                "timepoint": batch["timepoints"][i],
                "label": int(labels_np[i]),
                "prob": float(probs_np[i]),
            })

    pred_df = pd.DataFrame(rows)
    avg_loss = total_loss / len(loader.dataset)
    return pred_df, avg_loss


def tune_threshold_for_f1(pred_df: pd.DataFrame) -> Tuple[float, Dict[str, float]]:
    y_true = pred_df["label"].to_numpy()
    y_prob = pred_df["prob"].to_numpy()

    thresholds = np.linspace(0.05, 0.95, 181)
    best_t = 0.5
    best_metrics = compute_metrics(y_true, y_prob, threshold=best_t)
    best_f1 = best_metrics["f1"]

    for t in thresholds:
        m = compute_metrics(y_true, y_prob, threshold=float(t))
        if m["f1"] > best_f1:
            best_f1 = m["f1"]
            best_t = float(t)
            best_metrics = m

    return best_t, best_metrics


# --- NEW: per-timepoint threshold tuning ---
def tune_per_timepoint_thresholds(pred_df: pd.DataFrame) -> Dict[str, float]:
    """Returns {timepoint_str: best_threshold} tuned independently per group."""
    thresholds = {}
    for tp, sub in pred_df.groupby("timepoint"):
        best_t, _ = tune_threshold_for_f1(sub)
        thresholds[str(tp)] = best_t
    return thresholds


def apply_per_timepoint_thresholds(pred_df: pd.DataFrame, tp_thresholds: Dict[str, float]) -> pd.DataFrame:
    """Apply a different threshold per timepoint group."""
    out = pred_df.copy()
    out["pred_label"] = 0
    for tp, t in tp_thresholds.items():
        mask = out["timepoint"] == tp
        out.loc[mask, "pred_label"] = (out.loc[mask, "prob"] >= t).astype(int)
    return out


def add_pred_labels(pred_df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    out = pred_df.copy()
    out["pred_label"] = (out["prob"] >= threshold).astype(int)
    return out


def per_timepoint_metrics(pred_df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    rows = []
    for tp, sub in pred_df.groupby("timepoint"):
        y_true = sub["label"].to_numpy()
        y_prob = sub["prob"].to_numpy()
        m = compute_metrics(y_true, y_prob, threshold=threshold)
        m["timepoint"] = tp
        m["n"] = int(len(sub))
        rows.append(m)
    return pd.DataFrame(rows)


def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


# =========================================================
# Train / eval — CHANGED: uses _run_forward helper
# =========================================================

def train_one_epoch(model, loader, optimizer, criterion, device, model_type, gradient_clip):
    model.train()
    total_loss = 0.0

    for batch in loader:
        optimizer.zero_grad()
        labels = batch["labels"].to(device)
        logits = _run_forward(model, batch, device, model_type)

        loss = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        optimizer.step()

        total_loss += loss.item() * labels.size(0)

    return total_loss / len(loader.dataset)


# =========================================================
# One experiment — CHANGED: per-timepoint threshold option
# =========================================================

def run_experiment(cfg: Config, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame):
    exp_dir = os.path.join(cfg.results_root, cfg.experiment_name)
    os.makedirs(exp_dir, exist_ok=True)

    save_json(asdict(cfg), os.path.join(exp_dir, "config.json"))

    model, train_loader, val_loader, test_loader = build_model_and_loaders(cfg, train_df, val_df, test_df)
    model = model.to(cfg.device)

    optimizer = build_optimizer(model, cfg)

    if cfg.positive_class_weight is not None:
        pos_weight = torch.tensor([cfg.positive_class_weight], device=cfg.device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        criterion = nn.BCEWithLogitsLoss()

    best_val_f1_at_05 = -1.0
    history = []

    for epoch in range(1, cfg.epochs + 1):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, cfg.device, cfg.model_type, cfg.gradient_clip
        )

        val_pred_df, val_loss = collect_predictions(model, val_loader, criterion, cfg.device, cfg.model_type)
        val_metrics_05 = compute_metrics(
            val_pred_df["label"].to_numpy(),
            val_pred_df["prob"].to_numpy(),
            threshold=0.5,
        )
        val_metrics_05["loss"] = float(val_loss)

        row = {"epoch": epoch, "train_loss": float(train_loss), **val_metrics_05}
        history.append(row)

        print(
            f"[{cfg.experiment_name}] Epoch {epoch:02d} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"f1@0.5={val_metrics_05['f1']:.4f} | "
            f"prec={val_metrics_05['precision']:.4f} | "
            f"rec={val_metrics_05['recall']:.4f} | "
            f"auroc={val_metrics_05['auroc']:.4f} | "
            f"auprc={val_metrics_05['auprc']:.4f}"
        )

        if val_metrics_05["f1"] > best_val_f1_at_05:
            best_val_f1_at_05 = val_metrics_05["f1"]
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": asdict(cfg),
                    "best_val_metrics_at_05": val_metrics_05,
                    "epoch": epoch,
                },
                cfg.save_path,
            )
            print(f"[{cfg.experiment_name}] Saved best checkpoint to {cfg.save_path}")

    pd.DataFrame(history).to_csv(os.path.join(exp_dir, "training_history.csv"), index=False)

    # --- Reload best checkpoint ---
    ckpt = torch.load(cfg.save_path, map_location=cfg.device)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(cfg.device)

    # --- NEW: log layer weights if used ---
    if cfg.use_layer_weights:
        _save_layer_weights(model, cfg, exp_dir)

    # --- Validation predictions + threshold tuning ---
    val_pred_df, val_loss = collect_predictions(model, val_loader, criterion, cfg.device, cfg.model_type)

    # Global threshold (always computed for comparison)
    tuned_threshold, val_tuned_metrics = tune_threshold_for_f1(val_pred_df)

    # --- NEW: per-timepoint thresholds ---
    tp_thresholds = None
    if cfg.per_timepoint_threshold:
        tp_thresholds = tune_per_timepoint_thresholds(val_pred_df)
        save_json(tp_thresholds, os.path.join(exp_dir, "per_timepoint_thresholds.json"))
        val_pred_df = apply_per_timepoint_thresholds(val_pred_df, tp_thresholds)
        print(f"[{cfg.experiment_name}] Per-timepoint thresholds: {tp_thresholds}")
    else:
        val_pred_df = add_pred_labels(val_pred_df, tuned_threshold)

    val_pred_df.to_csv(os.path.join(exp_dir, "val_predictions.csv"), index=False)

    val_overall_metrics = compute_metrics(
        val_pred_df["label"].to_numpy(),
        val_pred_df["prob"].to_numpy(),
        threshold=tuned_threshold,
    )
    val_overall_metrics["loss"] = float(val_loss)
    val_overall_metrics["threshold"] = float(tuned_threshold)
    save_json(val_overall_metrics, os.path.join(exp_dir, "val_metrics_tuned.json"))

    val_tp_df = per_timepoint_metrics(val_pred_df, tuned_threshold)
    val_tp_df.to_csv(os.path.join(exp_dir, "val_metrics_by_timepoint.csv"), index=False)

    # --- Test predictions ---
    test_pred_df, test_loss = collect_predictions(model, test_loader, criterion, cfg.device, cfg.model_type)

    if cfg.per_timepoint_threshold and tp_thresholds is not None:
        test_pred_df = apply_per_timepoint_thresholds(test_pred_df, tp_thresholds)
    else:
        test_pred_df = add_pred_labels(test_pred_df, tuned_threshold)

    test_pred_df.to_csv(os.path.join(exp_dir, "test_predictions.csv"), index=False)

    test_overall_metrics = compute_metrics(
        test_pred_df["label"].to_numpy(),
        test_pred_df["prob"].to_numpy(),
        threshold=tuned_threshold,
    )
    test_overall_metrics["loss"] = float(test_loss)
    test_overall_metrics["threshold"] = float(tuned_threshold)
    save_json(test_overall_metrics, os.path.join(exp_dir, "test_metrics_tuned.json"))

    test_tp_df = per_timepoint_metrics(test_pred_df, tuned_threshold)
    test_tp_df.to_csv(os.path.join(exp_dir, "test_metrics_by_timepoint.csv"), index=False)

    print(f"\n[{cfg.experiment_name}] Tuned threshold on val: {tuned_threshold:.3f}")
    print(f"[{cfg.experiment_name}] Final test metrics: {test_overall_metrics}")


def _save_layer_weights(model, cfg, exp_dir):
    """Log the learned layer-mixing weights for analysis."""
    weights_dict = {}
    if cfg.model_type == "fused":
        for name, attr in [("whisper", "whisper_mix"), ("wavlm", "wavlm_mix")]:
            mix = getattr(model, attr, None)
            if mix is not None:
                w = torch.softmax(mix.weights, dim=0).detach().cpu().tolist()
                weights_dict[name] = {f"layer_{i}": round(v, 4) for i, v in enumerate(w)}
    else:
        mix = getattr(model, "layer_mix", None)
        if mix is not None:
            w = torch.softmax(mix.weights, dim=0).detach().cpu().tolist()
            weights_dict[cfg.model_type] = {f"layer_{i}": round(v, 4) for i, v in enumerate(w)}

    if weights_dict:
        save_json(weights_dict, os.path.join(exp_dir, "learned_layer_weights.json"))
        print(f"[{cfg.experiment_name}] Learned layer weights: {weights_dict}")


# =========================================================
# Main — expanded experiment grid
# =========================================================

def main():
    os.makedirs(CFG.results_root, exist_ok=True)
    os.makedirs(CFG.split_dir, exist_ok=True)

    train_df, val_df, test_df = load_or_create_split(CFG)

    print(f"Train rows: {len(train_df)} | children: {train_df['child_id'].nunique()}")
    print(f"Val rows:   {len(val_df)} | children: {val_df['child_id'].nunique()}")
    print(f"Test rows:  {len(test_df)} | children: {test_df['child_id'].nunique()}")

    # ---- Phase 1: your original 4 baselines (unchanged) ----
    baselines: List[Config] = [
        replace(CFG, experiment_name="whisper_mean", model_type="whisper", pooling="mean",
                save_path=os.path.join(CFG.results_root, "whisper_mean", "best_model.pt")),
        replace(CFG, experiment_name="whisper_attn", model_type="whisper", pooling="attn",
                save_path=os.path.join(CFG.results_root, "whisper_attn", "best_model.pt")),
        replace(CFG, experiment_name="wavlm_mean", model_type="wavlm", pooling="mean",
                batch_size=2, num_workers=2,
                save_path=os.path.join(CFG.results_root, "wavlm_mean", "best_model.pt")),
        replace(CFG, experiment_name="wavlm_attn", model_type="wavlm", pooling="attn",
                batch_size=1, num_workers=2,
                save_path=os.path.join(CFG.results_root, "wavlm_attn", "best_model.pt")),
    ]

    # ---- Phase 2: + layer-weighted sum (attn pooling since whisper_attn was best) ----
    layer_weighted: List[Config] = [
        replace(CFG, experiment_name="whisper_attn_lw", model_type="whisper", pooling="attn",
                use_layer_weights=True,
                save_path=os.path.join(CFG.results_root, "whisper_attn_lw", "best_model.pt")),
        replace(CFG, experiment_name="wavlm_attn_lw", model_type="wavlm", pooling="attn",
                use_layer_weights=True, batch_size=1, num_workers=2,
                save_path=os.path.join(CFG.results_root, "wavlm_attn_lw", "best_model.pt")),
    ]

    # ---- Phase 3: + layer-weighted sum + stats pooling ----
    # stats replaces attn here — test whether richer pooling beats attention
    lw_stats: List[Config] = [
        replace(CFG, experiment_name="whisper_stats_lw", model_type="whisper", pooling="stats",
                use_layer_weights=True,
                save_path=os.path.join(CFG.results_root, "whisper_stats_lw", "best_model.pt")),
        replace(CFG, experiment_name="wavlm_stats_lw", model_type="wavlm", pooling="stats",
                use_layer_weights=True, batch_size=2, num_workers=2,
                save_path=os.path.join(CFG.results_root, "wavlm_stats_lw", "best_model.pt")),
    ]

    # ---- Phase 4: fused model WITHOUT layer weights (they didn't learn) ----
    fused: List[Config] = [
        replace(CFG, experiment_name="fused_attn", model_type="fused", pooling="attn",
                use_layer_weights=False, batch_size=1, num_workers=2,
                per_timepoint_threshold=True,
                save_path=os.path.join(CFG.results_root, "fused_attn", "best_model.pt")),
    ]

    # ---- Phase 5: unfreezing top 2 layers on best configs ----
    unfrozen: List[Config] = [
        replace(CFG, experiment_name="whisper_attn_unfreeze2", model_type="whisper", pooling="attn",
                use_layer_weights=False, unfreeze_last_n_layers=2, per_timepoint_threshold=True,
                save_path=os.path.join(CFG.results_root, "whisper_attn_unfreeze2", "best_model.pt")),
        replace(CFG, experiment_name="fused_attn_unfreeze2", model_type="fused", pooling="attn",
                use_layer_weights=False, unfreeze_last_n_layers=2, batch_size=1, num_workers=2,
                per_timepoint_threshold=True,
                save_path=os.path.join(CFG.results_root, "fused_attn_unfreeze2", "best_model.pt")),
    ]

    # ---- Phase 6: no-new-parameters improvements on whisper_attn baseline ----
    no_new_params: List[Config] = [
        # Per-timepoint thresholds only (same model, just smarter eval)
        replace(CFG, experiment_name="whisper_attn_ptt", model_type="whisper", pooling="attn",
                per_timepoint_threshold=True,
                save_path=os.path.join(CFG.results_root, "whisper_attn_ptt", "best_model.pt")),
        # Speed perturbation + random crop augmentation
        replace(CFG, experiment_name="whisper_attn_aug", model_type="whisper", pooling="attn",
                speed_perturb=True, random_crop=True,
                save_path=os.path.join(CFG.results_root, "whisper_attn_aug", "best_model.pt")),
        # Both augmentation + per-timepoint thresholds
        replace(CFG, experiment_name="whisper_attn_aug_ptt", model_type="whisper", pooling="attn",
                speed_perturb=True, random_crop=True, per_timepoint_threshold=True,
                save_path=os.path.join(CFG.results_root, "whisper_attn_aug_ptt", "best_model.pt")),
    ]

    # Skip phases already completed
    # all_experiments = baselines + layer_weighted + lw_stats + fused + unfrozen + no_new_params
    # all_experiments = baselines + fused + unfrozen + no_new_params
    all_experiments = no_new_params

    for exp_cfg in all_experiments:
        print("\n" + "=" * 80)
        print(f"Running experiment: {exp_cfg.experiment_name}")
        print("=" * 80)
        run_experiment(exp_cfg, train_df, val_df, test_df)


if __name__ == "__main__":
    main()