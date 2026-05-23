"""infer_long_wav_files.py — write per-frame pseudo-frame RTTMs for long audio.

Loads a PseudoFrameModel checkpoint and runs `chunked_inference` (10-sec
non-overlapping windows, 50 Hz frame output) over each input audio file,
thresholds at the val-tuned threshold stored in the checkpoint, then writes
one RTTM per audio file under <output_dir>/<stem>__<md5(audio_path)>.rttm.

Used by the unified_rttm.py pseudo_frame_* frontends to evaluate the
pseudo-frame head as a frame-level diarizer on Playlogue / Providence,
alongside USC-SAIL, Pyannote, BabAR, VTC, VBx, EEND-EDA, Sortformer.

Usage:
    python pseudo_frame/infer_long_wav_files.py \\
        --checkpoint pseudo_frame/results/wavlm_pseudo_frame/best_checkpoint.pt \\
        --audio-list /tmp/audio_paths.txt \\
        --output-dir pseudo_frame/results/wavlm_pseudo_frame/rttm_cache/ \\
        [--threshold-override 0.5]   # default: read from checkpoint
"""

import argparse
import hashlib
import os
import sys
from pathlib import Path
from typing import List

import numpy as np
import torch
import torchaudio

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_THIS_DIR)
sys.path.insert(0, _REPO)

from pseudo_frame.pseudo_model import PseudoFrameModel  # noqa: E402
from pseudo_frame.pseudo_evaluate import chunked_inference  # noqa: E402

SAMPLE_RATE = 16000
FRAME_HOP_SEC = 0.02  # 50 Hz output


def _audio_cache_id(audio_path: str) -> str:
    return hashlib.md5(audio_path.encode("utf-8")).hexdigest()


def _load_mono_16k(audio_path: str) -> torch.Tensor:
    wav, sr = torchaudio.load(audio_path)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != SAMPLE_RATE:
        wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE)
    return wav.squeeze(0)


def _frame_mask_to_segments(mask: np.ndarray,
                            hop_sec: float = FRAME_HOP_SEC,
                            min_dur_sec: float = 0.1,
                            merge_gap_sec: float = 0.2) -> List:
    """Convert a 1-D binary frame mask to a list of (start, end) segment tuples.

    Contiguous active frames are grouped; gaps shorter than `merge_gap_sec`
    are filled to avoid over-fragmentation; final segments shorter than
    `min_dur_sec` are dropped.
    """
    if mask.size == 0 or not mask.any():
        return []

    diff = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    raw = list(zip(starts * hop_sec, ends * hop_sec))

    if not raw:
        return []

    merged = [raw[0]]
    for s, e in raw[1:]:
        last_s, last_e = merged[-1]
        if s - last_e <= merge_gap_sec:
            merged[-1] = (last_s, max(last_e, e))
        else:
            merged.append((s, e))

    return [(s, e) for s, e in merged if (e - s) >= min_dur_sec]


def _segments_to_rttm(segments: List, stem: str, label: str = "CHI") -> List[str]:
    lines = []
    for s, e in segments:
        dur = e - s
        if dur <= 0:
            continue
        lines.append(
            f"SPEAKER {stem} 1 {s:.3f} {dur:.3f} <NA> <NA> {label} <NA> <NA>"
        )
    return lines


def _load_model(checkpoint_path: str, device: torch.device):
    ckpt = torch.load(checkpoint_path, map_location=device)
    cfg = ckpt["cfg"]
    threshold = float(ckpt["val_threshold"])
    model = PseudoFrameModel(
        backbone_name=cfg.get("backbone", "microsoft/wavlm-base-plus"),
        backbone_layer=cfg.get("backbone_layer", -1),
        hidden_dim=cfg.get("hidden_dim", 256),
        dropout=cfg.get("dropout", 0.2),
    ).to(device)
    model.head.load_state_dict(ckpt["head_state"])
    model.eval()
    chunk_sec = float(cfg.get("crop_sec", 10.0))
    return model, threshold, chunk_sec


def main():
    ap = argparse.ArgumentParser(
        description="Write per-frame RTTMs for the WavLM pseudo-frame classifier."
    )
    ap.add_argument("--checkpoint", required=True,
                    help="Path to a pseudo-frame best_checkpoint.pt.")
    ap.add_argument("--audio-list", required=True,
                    help="Text file with one audio path per line.")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--threshold-override", type=float, default=None,
                    help="Override the val-tuned threshold from the checkpoint.")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--min-dur-sec", type=float, default=0.1,
                    help="Drop predicted segments shorter than this.")
    ap.add_argument("--merge-gap-sec", type=float, default=0.2,
                    help="Merge predicted segments separated by ≤ this gap.")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available()
                          and args.device == "cuda" else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.audio_list) as f:
        audio_paths = [ln.strip() for ln in f if ln.strip()]
    if not audio_paths:
        print("No audio files in list — nothing to do.")
        return

    model, default_thr, chunk_sec = _load_model(args.checkpoint, device)
    threshold = args.threshold_override if args.threshold_override is not None else default_thr
    print(f"Loaded {args.checkpoint}; threshold={threshold:.4f}, chunk_sec={chunk_sec}")
    print(f"Processing {len(audio_paths)} audio file(s)...")

    for i, audio_path in enumerate(audio_paths, 1):
        stem = Path(audio_path).stem
        cid = _audio_cache_id(audio_path)
        dst = os.path.join(args.output_dir, f"{stem}__{cid}.rttm")
        if os.path.exists(dst):
            continue
        try:
            wav = _load_mono_16k(audio_path)
            with torch.no_grad():
                probs = chunked_inference(model, wav, device, chunk_sec=chunk_sec)
            mask = (probs.numpy() >= threshold)
            segs = _frame_mask_to_segments(mask, FRAME_HOP_SEC,
                                            args.min_dur_sec, args.merge_gap_sec)
            lines = _segments_to_rttm(segs, stem)
            with open(dst, "w") as f:
                if lines:
                    f.write("\n".join(lines) + "\n")
            print(f"  [{i}/{len(audio_paths)}] {stem}: "
                  f"{int(mask.sum())} active frames → {len(segs)} segments")
        except Exception as exc:
            print(f"  WARNING [{i}]: {audio_path}: {exc}", file=sys.stderr)
            open(dst, "w").close()
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print("Pseudo-frame inference complete.")


if __name__ == "__main__":
    main()
