"""
nemo_diar.py — EEND-EDA and Sortformer DiarizationFrontend implementations.

EENDEDAFrontend   — End-to-End Neural Diarization with Encoder-Decoder Attractors
                    via ESPnet2. Handles overlapping speech natively; outputs
                    anonymous spk1/spk2/... labels.

SortformerFrontend — Sort-based transformer diarization via NeMo (NVIDIA).
                    Outputs anonymous speaker_0/speaker_1/... labels.

Both produce all-speaker RTTM segments and rely on the shared ECAPA
enrollment pipeline to identify the target child by cosine similarity.

Setup
-----
EEND-EDA (ESPnet2):
    pip install espnet espnet_model_zoo soundfile
    # Find a pre-trained EEND-EDA model at https://github.com/espnet/espnet_model_zoo
    # List available diarization models:
    #   python -c "from espnet_model_zoo.downloader import ModelDownloader; \\
    #              d=ModelDownloader(); print(d.query('diar'))"
    # Set cfg.eend_eda_model_tag to the chosen model tag or a local directory
    # containing train_config.yaml and a *.pth checkpoint.
    # Default: espnet/horiguchi_INTERSPEECH2022_EEND-EDA-online_6spk

Sortformer (NeMo):
    pip install nemo_toolkit[asr]
    # The model (diar_sortformer_4spk-v1) downloads from NGC on first run.
    # Set cfg.sortformer_model to a different NGC model name if needed.

Both may be installed into the existing child-vocalizations conda env or
separate venvs.  Configure cfg.eend_eda_env_python / cfg.sortformer_env_python
to point to the desired Python interpreter (default: "python").
"""

import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _audio_cache_id(audio_path: str) -> str:
    return hashlib.md5(audio_path.encode("utf-8")).hexdigest()


def _parse_rttm_all(rttm_path: str) -> List[Dict[str, float]]:
    """Return all segments from an RTTM, regardless of speaker label."""
    segs = []
    if not os.path.exists(rttm_path):
        return segs
    with open(rttm_path) as f:
        for line in f:
            line = line.strip()
            if not line or not line.startswith("SPEAKER"):
                continue
            parts = line.split()
            if len(parts) < 8:
                continue
            start, dur = float(parts[3]), float(parts[4])
            if dur > 0:
                segs.append({"start": start, "end": start + dur, "dur": dur})
    return segs


def _run_inference_subprocess(cmd: List[str], label: str) -> None:
    result = subprocess.run(cmd, capture_output=False, check=False)
    if result.returncode != 0:
        raise RuntimeError(
            f"{label} inference subprocess failed (exit {result.returncode}).\n"
            f"Command: {' '.join(cmd)}"
        )


# ---------------------------------------------------------------------------
# EEND-EDA frontend
# ---------------------------------------------------------------------------

class EENDEDAFrontend:
    """
    EEND-EDA diarization via ESPnet2.

    Calls pyannote/run_espnet_diar.py as a subprocess so ESPnet can live
    in its own Python env without touching the main conda environment.
    Anonymous speaker labels (spk1, spk2, …) are returned; ECAPA enrollment
    identifies the target child by cosine similarity.

    Usage after setup:
        python unified.py --diarizer eend_eda
    """

    def __init__(self, cfg):
        self.cfg = cfg
        os.makedirs(cfg.eend_eda_rttm_cache_dir, exist_ok=True)
        self._script = os.path.join(_THIS_DIR, "run_espnet_diar.py")
        if not os.path.exists(self._script):
            raise FileNotFoundError(
                f"EEND-EDA inference script not found: {self._script}\n"
                "Expected at pyannote/run_espnet_diar.py"
            )

    def _rttm_cache_path(self, audio_path: str) -> str:
        cid = _audio_cache_id(audio_path)
        stem = Path(audio_path).stem
        return os.path.join(self.cfg.eend_eda_rttm_cache_dir, f"{stem}__{cid}.rttm")

    def prepare(self, audio_paths: List[str]):
        """Batch-run EEND-EDA on all files without a cached RTTM."""
        missing = [p for p in audio_paths
                   if not os.path.exists(self._rttm_cache_path(p))]
        if not missing:
            print("EEND-EDA: all RTTM files already cached.")
            return

        print(f"EEND-EDA: running on {len(missing)} audio file(s)...")
        with tempfile.TemporaryDirectory() as tmp:
            list_path = os.path.join(tmp, "audio_list.txt")
            with open(list_path, "w") as f:
                f.write("\n".join(missing) + "\n")

            rttm_tmp = os.path.join(tmp, "rttms")
            os.makedirs(rttm_tmp)

            device = "cuda" if "cuda" in self.cfg.device else "cpu"
            cmd = [
                self.cfg.eend_eda_env_python, self._script,
                "--audio-list", list_path,
                "--output-dir", rttm_tmp,
                "--model-tag", self.cfg.eend_eda_model_tag,
                "--num-spks", str(self.cfg.eend_eda_num_spks),
                "--device", device,
            ]
            _run_inference_subprocess(cmd, "EEND-EDA")

            for ap in missing:
                stem = Path(ap).stem
                src = os.path.join(rttm_tmp, f"{stem}.rttm")
                dst = self._rttm_cache_path(ap)
                if os.path.exists(src):
                    shutil.copy2(src, dst)
                else:
                    open(dst, "w").close()

    def get_segments(self, audio_path: str, cfg) -> List[Dict[str, float]]:
        rttm = self._rttm_cache_path(audio_path)
        if not os.path.exists(rttm):
            self.prepare([audio_path])
        segs = _parse_rttm_all(rttm)
        return [s for s in segs if s["dur"] >= cfg.min_seg_dur_sec]


# ---------------------------------------------------------------------------
# Sortformer frontend
# ---------------------------------------------------------------------------

class SortformerFrontend:
    """
    Sortformer diarization via NeMo (NVIDIA).

    Calls pyannote/run_nemo_diar.py as a subprocess so NeMo can live in
    its own Python env.  Anonymous SPEAKER_XX labels from the output RTTM
    are all passed to the ECAPA enrollment step.

    Usage after setup:
        python unified.py --diarizer sortformer
    """

    def __init__(self, cfg):
        self.cfg = cfg
        os.makedirs(cfg.sortformer_rttm_cache_dir, exist_ok=True)
        self._script = os.path.join(_THIS_DIR, "run_nemo_diar.py")
        if not os.path.exists(self._script):
            raise FileNotFoundError(
                f"Sortformer inference script not found: {self._script}\n"
                "Expected at pyannote/run_nemo_diar.py"
            )

    def _rttm_cache_path(self, audio_path: str) -> str:
        cid = _audio_cache_id(audio_path)
        stem = Path(audio_path).stem
        return os.path.join(self.cfg.sortformer_rttm_cache_dir, f"{stem}__{cid}.rttm")

    def prepare(self, audio_paths: List[str]):
        """Batch-run Sortformer on all files without a cached RTTM."""
        missing = [p for p in audio_paths
                   if not os.path.exists(self._rttm_cache_path(p))]
        if not missing:
            print("Sortformer: all RTTM files already cached.")
            return

        print(f"Sortformer: running on {len(missing)} audio file(s)...")
        with tempfile.TemporaryDirectory() as tmp:
            list_path = os.path.join(tmp, "audio_list.txt")
            with open(list_path, "w") as f:
                f.write("\n".join(missing) + "\n")

            rttm_tmp = os.path.join(tmp, "rttms")
            os.makedirs(rttm_tmp)

            device = "cuda" if "cuda" in self.cfg.device else "cpu"
            cmd = [
                self.cfg.sortformer_env_python, self._script,
                "--audio-list", list_path,
                "--output-dir", rttm_tmp,
                "--model", self.cfg.sortformer_model,
                "--max-speakers", str(self.cfg.sortformer_max_speakers),
                "--device", device,
            ]
            _run_inference_subprocess(cmd, "Sortformer")

            for ap in missing:
                stem = Path(ap).stem
                src = os.path.join(rttm_tmp, f"{stem}.rttm")
                dst = self._rttm_cache_path(ap)
                if os.path.exists(src):
                    shutil.copy2(src, dst)
                else:
                    open(dst, "w").close()

    def get_segments(self, audio_path: str, cfg) -> List[Dict[str, float]]:
        rttm = self._rttm_cache_path(audio_path)
        if not os.path.exists(rttm):
            self.prepare([audio_path])
        segs = _parse_rttm_all(rttm)
        return [s for s in segs if s["dur"] >= cfg.min_seg_dur_sec]
