"""Disk-backed segment embedding cache for segment-instance MIL.

Purpose: Cache frozen WavLM-base+ segment embeddings to avoid redundant
         forward passes across the 4 aggregator variants that share the
         same frontend.
Inputs:  audio_path, segment start/end times (seconds).
Outputs: numpy arrays stored as .npy files under
         mil/seg_embedding_cache/{frontend_name}/.
Side effects: Creates cache directories on first write.
"""

import hashlib
import os
from typing import Optional

import numpy as np


def _seg_cache_key(audio_path: str, start: float, end: float) -> str:
    raw = f"{audio_path}|{start:.4f}|{end:.4f}"
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


class SegmentEmbeddingCache:
    """Disk-backed cache: (audio_path, start, end) → np.ndarray."""

    def __init__(self, cache_dir: str) -> None:
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    def _path(self, audio_path: str, start: float, end: float) -> str:
        key = _seg_cache_key(audio_path, start, end)
        return os.path.join(self.cache_dir, f"{key}.npy")

    def get(self, audio_path: str, start: float, end: float) -> Optional[np.ndarray]:
        p = self._path(audio_path, start, end)
        if os.path.exists(p):
            return np.load(p)
        return None

    def put(self, audio_path: str, start: float, end: float, embedding: np.ndarray) -> None:
        p = self._path(audio_path, start, end)
        np.save(p, embedding)
