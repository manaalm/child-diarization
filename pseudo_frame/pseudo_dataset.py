"""Dataset for frame-level pseudo-label training.

Each example yields:
  - waveform tensor: (T_audio,) float32 at 16 kHz
  - frame mask:      (T_frames,) float32 in [0, 1]
  - label:           clip-level int (used for max-pool clip score at eval)
  - meta:            audio_path, child_id, timepoint_norm

Frame rate = WavLM-Base+ output rate = 50 Hz (20 ms / frame).

For training, a random `crop_sec`-length crop is taken when the audio is longer.
For eval, the full audio is returned (caller chunks if needed).
"""
from __future__ import annotations

import os
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torchaudio
from torch.utils.data import Dataset


SAMPLE_RATE = 16000
FRAME_STEP_SEC = 0.02
FRAME_RATE = int(round(1.0 / FRAME_STEP_SEC))   # 50 Hz


class PseudoFrameDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        index_df: pd.DataFrame,
        crop_sec: Optional[float] = 10.0,
        deterministic: bool = False,
        sample_rate: int = SAMPLE_RATE,
    ) -> None:
        idx = index_df.set_index("audio_path")
        # Inner-join: only keep clips with cached pseudo-labels
        df = df[df["audio_path"].isin(idx.index)].reset_index(drop=True)
        df["npy_path"] = df["audio_path"].map(idx["npy_path"])
        df["n_sources"] = df["audio_path"].map(idx["n_sources"])
        self.records = df
        self.crop_sec = crop_sec
        self.deterministic = deterministic
        self.sample_rate = sample_rate

    def __len__(self) -> int:
        return len(self.records)

    def _load_audio(self, audio_path: str) -> torch.Tensor:
        wav, sr = torchaudio.load(audio_path)
        if sr != self.sample_rate:
            wav = torchaudio.functional.resample(wav, sr, self.sample_rate)
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        return wav.squeeze(0)  # (T,)

    def __getitem__(self, idx: int):
        row = self.records.iloc[idx]
        wav = self._load_audio(str(row["audio_path"]))
        mask = np.load(str(row["npy_path"])).astype(np.float32)
        return self._slice(wav, mask, row)

    def _slice(self, wav: torch.Tensor, mask: np.ndarray, row):
        # Align: 1 frame = FRAME_STEP_SEC seconds = (sr * FRAME_STEP_SEC) audio samples
        n_audio = wav.shape[0]
        n_frames_audio = n_audio // int(self.sample_rate * FRAME_STEP_SEC)
        # Pin mask to the audio frame count
        if len(mask) < n_frames_audio:
            mask = np.pad(mask, (0, n_frames_audio - len(mask)), constant_values=0)
        elif len(mask) > n_frames_audio:
            mask = mask[:n_frames_audio]

        if self.crop_sec is not None:
            crop_samples = int(self.crop_sec * self.sample_rate)
            crop_frames = int(self.crop_sec * FRAME_RATE)
            if n_audio > crop_samples:
                if self.deterministic:
                    s_aud = 0
                else:
                    # Random crop with frame alignment
                    max_frame_start = n_frames_audio - crop_frames
                    f_start = int(np.random.randint(0, max_frame_start + 1))
                    s_aud = f_start * int(self.sample_rate * FRAME_STEP_SEC)
                wav = wav[s_aud:s_aud + crop_samples]
                f_start = s_aud // int(self.sample_rate * FRAME_STEP_SEC)
                mask = mask[f_start:f_start + crop_frames]
            elif n_audio < crop_samples:
                # Pad both
                pad = crop_samples - n_audio
                wav = torch.cat([wav, torch.zeros(pad, dtype=wav.dtype)])
                pad_f = crop_frames - len(mask)
                if pad_f > 0:
                    mask = np.pad(mask, (0, pad_f), constant_values=0)

        return {
            "waveform": wav.float(),
            "mask": torch.from_numpy(mask).float(),
            "label": int(row["label"]),
            "audio_path": str(row["audio_path"]),
            "child_id": str(row["child_id"]),
            "timepoint_norm": str(row["timepoint_norm"]),
            "n_sources": int(row.get("n_sources", 0)),
        }


def collate(batch):
    """Stack with right-padding for variable-length batches."""
    max_T = max(b["waveform"].shape[0] for b in batch)
    max_F = max(b["mask"].shape[0] for b in batch)
    waves = torch.zeros(len(batch), max_T)
    masks = torch.zeros(len(batch), max_F)
    valid = torch.zeros(len(batch), max_F)
    for i, b in enumerate(batch):
        waves[i, :b["waveform"].shape[0]] = b["waveform"]
        masks[i, :b["mask"].shape[0]] = b["mask"]
        valid[i, :b["mask"].shape[0]] = 1.0
    return {
        "waveform": waves,
        "mask": masks,
        "valid": valid,
        "label": torch.tensor([b["label"] for b in batch], dtype=torch.float32),
        "audio_path": [b["audio_path"] for b in batch],
        "child_id": [b["child_id"] for b in batch],
        "timepoint_norm": [b["timepoint_norm"] for b in batch],
        "n_sources": [b["n_sources"] for b in batch],
    }
