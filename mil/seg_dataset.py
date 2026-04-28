"""Segment-bag dataset for segment-instance MIL.

Purpose: Assemble per-clip bags of WavLM-base+ segment embeddings from
         diarizer RTTM outputs. Each clip becomes a bag of K instance
         embeddings, one per speaker segment proposed by the frontend.
Inputs:  RTTM cache directory (one .rttm per audio file), split DataFrame,
         SegmentEmbeddingCache.
Outputs: (bag_tensor [K_max × D], mask [K_max], label, metadata_dict)
         via PyTorch Dataset interface.
Side effects: Writes embeddings to SegmentEmbeddingCache on cache miss.
"""

import argparse
import hashlib
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import WavLMModel

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)

from mil.seg_embedding_cache import SegmentEmbeddingCache

_WAVLM_SR = 16000
_WAVLM_FRAME_STRIDE_S = 0.02  # 20 ms


def _rttm_cache_path(audio_path: str, rttm_cache_dir: str) -> str:
    stem = os.path.splitext(os.path.basename(audio_path))[0]
    cid = hashlib.md5(audio_path.encode("utf-8")).hexdigest()
    return os.path.join(rttm_cache_dir, f"{stem}__{cid}.rttm")


def _load_rttm_segments(rttm_path: str, min_dur: float) -> List[Dict[str, float]]:
    """Read all SPEAKER lines from an RTTM; return [{start, end, label}]."""
    segs = []
    if not os.path.exists(rttm_path):
        return segs
    with open(rttm_path) as fh:
        for line in fh:
            line = line.strip()
            if not line or not line.startswith("SPEAKER"):
                continue
            parts = line.split()
            if len(parts) < 8:
                continue
            start, dur = float(parts[3]), float(parts[4])
            if dur < min_dur:
                continue
            segs.append({"start": start, "end": start + dur, "label": parts[7]})
    return segs


def _load_audio_segment(audio_path: str, start: float, end: float) -> Optional[torch.Tensor]:
    """Load a contiguous audio slice [start, end] as a (1, T) float32 tensor at 16 kHz."""
    try:
        import soundfile as sf
        import torchaudio

        data, sr = sf.read(audio_path, dtype="float32", always_2d=True)
        wav = torch.from_numpy(data.T)  # (channels, T)
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        if sr != _WAVLM_SR:
            wav = torchaudio.functional.resample(wav, sr, _WAVLM_SR)
        s_fr = max(0, int(start * _WAVLM_SR))
        e_fr = min(wav.shape[1], int(end * _WAVLM_SR))
        if e_fr <= s_fr:
            return None
        return wav[:, s_fr:e_fr]  # (1, T)
    except Exception:
        return None


def _embed_segment(
    model: WavLMModel,
    wav: torch.Tensor,
    device: torch.device,
    layer: int = -1,
) -> np.ndarray:
    """Run frozen WavLM forward pass on a (1, T) tensor; mean-pool frames → (D,)."""
    wav = wav.to(device)
    with torch.no_grad():
        out = model(wav, output_hidden_states=True)
        hidden = out.hidden_states[layer]  # (1, T_frames, D)
    emb = hidden.mean(dim=1).squeeze(0).cpu().numpy()  # (D,)
    return emb.astype(np.float32)


class SegmentBagDataset(Dataset):
    """One bag per clip; bags are variable-length collections of segment embeddings.

    Returns (bag_tensor, mask, label, metadata_dict).
    bag_tensor: (K_max, D) zero-padded
    mask: (K_max,) bool — True for real instances, False for padding
    Empty bags → all-zeros tensor, all-False mask.
    """

    def __init__(
        self,
        frontend_name: str,
        rttm_cache_dir: str,
        df: pd.DataFrame,
        embed_cache: SegmentEmbeddingCache,
        model: Optional[WavLMModel],
        device: torch.device,
        min_seg_dur: float = 0.4,
        layer: int = -1,
    ) -> None:
        self.frontend_name = frontend_name
        self.rttm_cache_dir = rttm_cache_dir
        self.df = df.reset_index(drop=True)
        self.embed_cache = embed_cache
        self.model = model
        self.device = device
        self.min_seg_dur = min_seg_dur
        self.layer = layer

        # Pre-build bag structure (segments per clip)
        self._bags: List[List[Dict]] = []
        for _, row in self.df.iterrows():
            rttm = _rttm_cache_path(row["audio_path"], rttm_cache_dir)
            segs = _load_rttm_segments(rttm, min_seg_dur)
            self._bags.append(segs)

        # Determine K_max and embed_dim
        self._k_max = max((len(b) for b in self._bags), default=1)
        # Peek embed_dim from cache or model
        self._embed_dim = self._resolve_embed_dim()

    def _resolve_embed_dim(self) -> int:
        if self.model is not None:
            return self.model.config.hidden_size
        # Try to find any cached embedding to infer dim
        for i, row in self.df.iterrows():
            for seg in self._bags[i]:
                cached = self.embed_cache.get(row["audio_path"], seg["start"], seg["end"])
                if cached is not None:
                    return cached.shape[0]
        # WavLM-base+ default
        return 768

    def _get_embedding(self, audio_path: str, seg: Dict) -> np.ndarray:
        cached = self.embed_cache.get(audio_path, seg["start"], seg["end"])
        if cached is not None:
            return cached
        if self.model is None:
            return np.zeros(self._embed_dim, dtype=np.float32)
        wav = _load_audio_segment(audio_path, seg["start"], seg["end"])
        if wav is None or wav.shape[1] == 0:
            emb = np.zeros(self._embed_dim, dtype=np.float32)
        else:
            emb = _embed_segment(self.model, wav, self.device, self.layer)
        self.embed_cache.put(audio_path, seg["start"], seg["end"], emb)
        return emb

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int, dict]:
        row = self.df.iloc[idx]
        segs = self._bags[idx]
        k = len(segs)
        d = self._embed_dim
        k_max = self._k_max

        bag = np.zeros((k_max, d), dtype=np.float32)
        mask = np.zeros(k_max, dtype=bool)

        for i, seg in enumerate(segs):
            bag[i] = self._get_embedding(row["audio_path"], seg)
            mask[i] = True

        # metadata for predictions CSV
        top_idx = int(np.argmax(np.zeros(k_max))) if k == 0 else None
        meta = {
            "audio_path": row["audio_path"],
            "child_id": row["child_id"],
            "timepoint_norm": row["timepoint_norm"],
            "label": int(row["label"]),
            "n_instances": k,
            "segs": segs,  # kept for top_seg extraction post-forward
        }

        return (
            torch.from_numpy(bag),
            torch.from_numpy(mask),
            int(row["label"]),
            meta,
        )


def precompute_embeddings(
    frontend_name: str,
    rttm_cache_dir: str,
    df: pd.DataFrame,
    embed_cache: SegmentEmbeddingCache,
    model: WavLMModel,
    device: torch.device,
    min_seg_dur: float = 0.4,
    layer: int = -1,
) -> None:
    """Pre-fill the embedding cache for all clips in df.

    Used by --precompute-only flag in seg_train.py. Iterates unique audio paths,
    loads each audio file once, and embeds all segments for that file.
    """
    audio_paths = df["audio_path"].unique()
    n_clips = len(audio_paths)
    n_cached = 0
    n_computed = 0

    for clip_idx, audio_path in enumerate(audio_paths):
        rttm = _rttm_cache_path(audio_path, rttm_cache_dir)
        segs = _load_rttm_segments(rttm, min_seg_dur)
        if not segs:
            continue

        # Load audio once for all segments in this clip
        try:
            import soundfile as sf
            import torchaudio

            data, sr = sf.read(audio_path, dtype="float32", always_2d=True)
            wav_full = torch.from_numpy(data.T)
            if wav_full.shape[0] > 1:
                wav_full = wav_full.mean(dim=0, keepdim=True)
            if sr != _WAVLM_SR:
                wav_full = torchaudio.functional.resample(wav_full, sr, _WAVLM_SR)
        except Exception as exc:
            print(f"  WARNING: could not load {audio_path}: {exc}", flush=True)
            continue

        for seg in segs:
            if embed_cache.get(audio_path, seg["start"], seg["end"]) is not None:
                n_cached += 1
                continue
            s_fr = max(0, int(seg["start"] * _WAVLM_SR))
            e_fr = min(wav_full.shape[1], int(seg["end"] * _WAVLM_SR))
            if e_fr <= s_fr:
                emb = np.zeros(model.config.hidden_size, dtype=np.float32)
            else:
                wav_seg = wav_full[:, s_fr:e_fr].to(device)
                with torch.no_grad():
                    out = model(wav_seg, output_hidden_states=True)
                    hidden = out.hidden_states[layer]
                emb = hidden.mean(dim=1).squeeze(0).cpu().numpy().astype(np.float32)
            embed_cache.put(audio_path, seg["start"], seg["end"], emb)
            n_computed += 1

        if (clip_idx + 1) % 100 == 0:
            print(
                f"  [{frontend_name}] {clip_idx + 1}/{n_clips} clips — "
                f"{n_computed} computed, {n_cached} cache hits",
                flush=True,
            )

    print(
        f"  [{frontend_name}] precompute done: {n_computed} new embeddings, "
        f"{n_cached} cache hits",
        flush=True,
    )


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    if args.smoke_test:
        import tempfile

        from transformers import WavLMModel

        print("Smoke test: SegmentBagDataset and SegmentEmbeddingCache")
        with tempfile.TemporaryDirectory() as tmp:
            cache = SegmentEmbeddingCache(tmp)
            # test put/get round-trip
            arr = np.random.randn(768).astype(np.float32)
            cache.put("/fake/path.wav", 1.0, 2.5, arr)
            got = cache.get("/fake/path.wav", 1.0, 2.5)
            assert got is not None and np.allclose(arr, got), "Cache round-trip failed"
            assert cache.get("/fake/path.wav", 1.0, 2.6) is None, "Wrong key should miss"
        print("Cache round-trip OK")
        print("Smoke test passed.")
