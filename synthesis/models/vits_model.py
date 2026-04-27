"""
VITS-based synthesis wrapper for 34-38 month toddler speech.

Wraps Coqui TTS's VITS model for fine-tuning on child speech segments.
The Coqui TTS VITS implementation provides multi-speaker conditioning via
speaker embeddings; here we treat the age-group prototype (ECAPA embedding)
as the single speaker conditioning vector.

Requires: uv run pip install coqui-ai-tts (via synthesis/pyproject.toml)
"""

import os
from pathlib import Path
from typing import Optional

import torch
import torchaudio


class VITSModel:
    """
    Thin wrapper around Coqui TTS VITS for fine-tuning on child speech.

    Training is handled by Coqui TTS trainer infrastructure. This class
    provides a consistent interface for:
      - Loading a pretrained or fine-tuned VITS checkpoint
      - Generating audio from (optionally) text input or purely from latent noise
      - Saving/loading checkpoints in a format compatible with synthesis/train.py

    For non-linguistic infant-adjacent toddler speech (34-38m), we use
    phoneme-level input derived from basic text tokens.
    """

    def __init__(self, config_path: str, checkpoint_path: Optional[str] = None,
                 device: str = "cpu"):
        self.config_path = config_path
        self.device = device
        self._model = None
        self._ap = None  # AudioProcessor

        if checkpoint_path and os.path.exists(checkpoint_path):
            self._load(checkpoint_path)

    def _load(self, checkpoint_path: str):
        try:
            from TTS.tts.configs.vits_config import VitsConfig
            from TTS.tts.models.vits import Vits
            from TTS.utils.audio import AudioProcessor

            config = VitsConfig()
            config.load_json(self.config_path)
            self._ap = AudioProcessor.init_from_config(config)
            self._model = Vits.init_from_config(config)
            state = torch.load(checkpoint_path, map_location=self.device)
            if "model" in state:
                self._model.load_state_dict(state["model"])
            else:
                self._model.load_state_dict(state)
            self._model.to(self.device)
            self._model.eval()
        except ImportError:
            raise ImportError(
                "Coqui TTS not installed. Run: cd synthesis && uv sync"
            )

    def is_loaded(self) -> bool:
        return self._model is not None

    @torch.no_grad()
    def generate(self, n_samples: int = 1, text: str = "a",
                 speaker_embedding: Optional[torch.Tensor] = None) -> torch.Tensor:
        if not self.is_loaded():
            raise RuntimeError("Model not loaded. Call _load() first.")

        wavs = []
        for _ in range(n_samples):
            outputs = self._model.synthesize(
                text=text,
                config=self._model.config,
                speaker_id=None,
                d_vector=speaker_embedding.to(self.device) if speaker_embedding is not None else None,
            )
            wav = torch.tensor(outputs["wav"], dtype=torch.float32)
            wavs.append(wav)
        return torch.stack(wavs)

    def save_checkpoint(self, path: str, optimizer=None, epoch: int = 0):
        state = {"model": self._model.state_dict(), "epoch": epoch}
        if optimizer is not None:
            state["optimizer"] = optimizer.state_dict()
        torch.save(state, path)

    @classmethod
    def from_pretrained_coqui(cls, model_name: str = "tts_models/en/ljspeech/vits",
                               config_path: str = "",
                               device: str = "cpu") -> "VITSModel":
        try:
            from TTS.api import TTS as CoquiTTS
        except ImportError:
            raise ImportError("Coqui TTS not installed. Run: cd synthesis && uv sync")

        tts = CoquiTTS(model_name=model_name, progress_bar=False)
        instance = cls(config_path=config_path, device=device)
        instance._model = tts.synthesizer.tts_model
        instance._ap = tts.synthesizer.output_sample_rate
        return instance
