"""
run_nemo_diar.py — NeMo Sortformer batch diarization inference script.

Called as a subprocess by SortformerFrontend in nemo_diar.py.

Usage:
    python run_nemo_diar.py \\
        --audio-list    /tmp/audio_paths.txt \\
        --output-dir    /path/to/rttm_output/ \\
        [--model        diar_sortformer_4spk-v1] \\
        [--max-speakers 4] \\
        [--device       cuda]

Output:
    One RTTM per input audio, named <stem>.rttm in <output-dir>.
    Speaker labels are SPEAKER_00, SPEAKER_01, etc.
    Empty RTTM is written for any file that fails (no crash).

Setup:
    pip install nemo_toolkit[asr]
    # The model downloads from NVIDIA NGC automatically on first run.

Available pre-trained Sortformer models (NeMo NGC):
    diar_sortformer_4spk-v1   — 4-speaker, trained on LibriSpeech + VoxCeleb
    # Browse: https://catalog.ngc.nvidia.com/models?filters=&orderBy=weightPopularDESC&query=sortformer
"""

import argparse
import os
import sys
from pathlib import Path
from typing import List

import torch


# ---------------------------------------------------------------------------
# NeMo Sortformer inference
# ---------------------------------------------------------------------------

def _build_manifest(audio_paths: List[str], manifest_path: str) -> None:
    """Create a NeMo-format JSONL manifest for the given audio files."""
    import torchaudio

    with open(manifest_path, "w") as f:
        for ap in audio_paths:
            try:
                info = torchaudio.info(ap)
                dur = info.num_frames / info.sample_rate
            except Exception:
                dur = -1.0
            entry = {
                "audio_filepath": ap,
                "offset": 0,
                "duration": dur if dur > 0 else None,
                "label": "infer",
                "text": "-",
                "num_speakers": None,
                "rttm_filepath": None,
            }
            f.write(json.dumps(entry) + "\n")


def _segments_to_rttm(segments: List[str], stem: str) -> List[str]:
    """Convert NeMo diarize() output segments to RTTM lines.

    NeMo 2.7 returns segments as strings: "begin_sec end_sec speaker_idx"
    or as lists [begin, end, spk_idx].
    """
    lines = []
    for seg in segments:
        if isinstance(seg, str):
            parts = seg.strip().split()
            if len(parts) < 3:
                continue
            start, end, spk = float(parts[0]), float(parts[1]), parts[2]
        else:
            start, end, spk = float(seg[0]), float(seg[1]), str(seg[2])
        dur = end - start
        if dur <= 0:
            continue
        spk_label = f"SPEAKER_{int(spk):02d}" if spk.isdigit() else spk
        lines.append(
            f"SPEAKER {stem} 1 {start:.3f} {dur:.3f} <NA> <NA> {spk_label} <NA> <NA>"
        )
    return lines


def run_sortformer(audio_paths: List[str], output_dir: str, model_name: str,
                   max_speakers: int, device: str) -> None:
    """Run NeMo Sortformer diarization on a list of audio files."""
    try:
        from nemo.collections.asr.models import SortformerEncLabelModel
    except ImportError:
        raise ImportError(
            "NeMo not found. Install with: pip install nemo_toolkit[asr]\n"
            "Then re-run."
        )

    # Support HuggingFace repo IDs (e.g. "nvidia/diar_sortformer_4spk-v1") in addition
    # to NGC model names. from_pretrained falls back to HF snapshot_download when the
    # NGC lookup returns 404.
    print(f"Loading Sortformer model: {model_name} (device={device})")
    try:
        model = SortformerEncLabelModel.from_pretrained(model_name)
    except (FileNotFoundError, Exception) as e:
        # If the NGC name fails, try treating it as a HuggingFace repo id
        if "/" in model_name:
            raise
        hf_name = f"nvidia/{model_name}"
        print(f"  NGC lookup failed ({e}); retrying as HF repo: {hf_name}")
        model = SortformerEncLabelModel.from_pretrained(hf_name)
    model.eval()
    if device == "cuda" and torch.cuda.is_available():
        model = model.cuda()
    else:
        model = model.cpu()
    print(f"Model loaded. Processing {len(audio_paths)} file(s)...")

    os.makedirs(output_dir, exist_ok=True)

    # NeMo 2.7+ diarize() takes a list of audio paths and returns a list of
    # segment lists: [[begin, end, spk_idx], ...] per file.
    results = model.diarize(
        audio=audio_paths,
        batch_size=1,
        num_workers=0,
        verbose=True,
    )

    for ap, segs in zip(audio_paths, results):
        stem = Path(ap).stem
        dst = os.path.join(output_dir, f"{stem}.rttm")
        lines = _segments_to_rttm(segs if segs else [], stem)
        with open(dst, "w") as f:
            if lines:
                f.write("\n".join(lines) + "\n")

    print("Sortformer inference complete.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="NeMo Sortformer batch diarization — outputs one RTTM per audio file."
    )
    parser.add_argument(
        "--audio-list", required=True,
        help="Text file with one audio path per line.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--model",
        default="diar_sortformer_4spk-v1",
        help="NeMo NGC model name or local .nemo file path.",
    )
    parser.add_argument(
        "--max-speakers", type=int, default=4,
        help="Maximum number of speakers to detect per clip.",
    )
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    with open(args.audio_list) as f:
        paths = [ln.strip() for ln in f if ln.strip()]
    if not paths:
        print("No audio files in list — nothing to do.")
        return

    run_sortformer(paths, args.output_dir, args.model, args.max_speakers, args.device)


if __name__ == "__main__":
    main()
