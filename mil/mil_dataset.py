"""MILBagDataset: load audio clips as bags of fixed-length windows."""

from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import torchaudio
from torch.utils.data import Dataset


class MILBagDataset(Dataset):
    """Each example is a bag (clip) of fixed-length audio windows.

    Args:
        df: DataFrame from seen_child_splits CSV, pre-filtered to audio_exists==True.
            Required columns: audio_path, label, child_id, timepoint_norm.
        window_sec: Window length in seconds (default 2.0).
        stride_sec: Window stride in seconds (default 1.0).
        sample_rate: Target sample rate; audio is resampled if needed (default 16000).
        pad_to_sec: If set, zero-pad audio to at least this many seconds after loading.
            Useful for short TinyVox clips so they produce the same number of windows
            as full-length training clips.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        window_sec: float = 2.0,
        stride_sec: float = 1.0,
        sample_rate: int = 16000,
        pad_to_sec: float | None = None,
    ) -> None:
        self.records = df.reset_index(drop=True)
        self.window_samples = int(window_sec * sample_rate)
        self.stride_samples = int(stride_sec * sample_rate)
        self.sample_rate = sample_rate
        self.pad_samples = int(pad_to_sec * sample_rate) if pad_to_sec else None

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict:
        row = self.records.iloc[idx]
        start_sec = float(row["start_sec"]) if "start_sec" in row and pd.notna(row["start_sec"]) else None
        end_sec   = float(row["end_sec"])   if "end_sec"   in row and pd.notna(row["end_sec"])   else None
        waveform = self._load_audio(str(row["audio_path"]), start_sec, end_sec)
        windows = self._make_windows(waveform)
        return {
            "windows": windows,
            "label": int(row["label"]),
            "child_id": str(row["child_id"]),
            "timepoint_norm": str(row["timepoint_norm"]),
            "audio_path": str(row["audio_path"]),
        }

    def _load_audio(self, path: str, start_sec: float | None = None, end_sec: float | None = None) -> torch.Tensor:
        """Load audio as (1, T) mono tensor at self.sample_rate.

        If start_sec/end_sec are given, loads only that slice (used for hard-negative
        windows extracted from long Playlogue/Providence recordings).
        """
        if start_sec is not None:
            info = torchaudio.info(path)
            sr_native = info.sample_rate
            frame_offset = int(start_sec * sr_native)
            num_frames = int((end_sec - start_sec) * sr_native) if end_sec is not None else -1
            wav, sr = torchaudio.load(path, frame_offset=frame_offset, num_frames=num_frames)
        else:
            wav, sr = torchaudio.load(path)
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        if sr != self.sample_rate:
            wav = torchaudio.functional.resample(wav, sr, self.sample_rate)
        if self.pad_samples and wav.shape[1] < self.pad_samples:
            wav = torch.nn.functional.pad(wav, (0, self.pad_samples - wav.shape[1]))
        return wav  # (1, T)

    def _make_windows(self, waveform: torch.Tensor) -> List[torch.Tensor]:
        """Slice waveform into overlapping windows; pad short clips."""
        total = waveform.shape[1]
        if total < self.window_samples:
            # Pad to exactly one window
            pad = self.window_samples - total
            waveform = torch.nn.functional.pad(waveform, (0, pad))
            return [waveform]

        windows = []
        start = 0
        while start + self.window_samples <= waveform.shape[1]:
            windows.append(waveform[:, start : start + self.window_samples])
            start += self.stride_samples

        # Include a final partial window if there's remaining audio
        remainder = waveform.shape[1] - (start - self.stride_samples + self.window_samples)
        if remainder > 0 and start < waveform.shape[1]:
            chunk = waveform[:, start:]
            pad = self.window_samples - chunk.shape[1]
            chunk = torch.nn.functional.pad(chunk, (0, pad))
            windows.append(chunk)

        return windows if windows else [torch.zeros(1, self.window_samples)]


def mil_collate_fn(batch: List[Dict]) -> Dict:
    """Collate a list of bag dicts into a batch dict.

    Windows are NOT stacked (bags have variable counts); returned as list-of-lists.
    """
    return {
        "windows": [item["windows"] for item in batch],
        "labels": torch.tensor([item["label"] for item in batch], dtype=torch.float32),
        "child_ids": [item["child_id"] for item in batch],
        "timepoint_norms": [item["timepoint_norm"] for item in batch],
        "audio_paths": [item["audio_path"] for item in batch],
    }
