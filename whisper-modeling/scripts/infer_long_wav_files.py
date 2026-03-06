import argparse
import os
from pathlib import Path

import torch
import torchaudio

from models.whisper import WhisperWrapper
from scripts.convert_output import get_timestamps, majority_filter

SAMPLE_RATE = 16000

def merge_segments(segments, min_duration=0.05, merge_gap=0.2):
    """
    segments: list of (start, end)
    - removes zero / tiny segments
    - merges overlapping or near-touching segments
    """

    # Remove tiny segments
    segments = [(s, e) for s, e in segments if (e - s) >= min_duration]

    if not segments:
        return []

    # Sort by start time
    segments = sorted(segments, key=lambda x: x[0])

    merged = [segments[0]]

    for s, e in segments[1:]:
        last_s, last_e = merged[-1]

        # If overlapping or close enough → merge
        if s <= last_e + merge_gap:
            merged[-1] = (last_s, max(last_e, e))
        else:
            merged.append((s, e))

    return merged

def combine_results(intervals, gap=0.01, ndigits=2):
    """Merge adjacent intervals with small gaps."""
    new_intervals = []
    for start, end in intervals:
        if not new_intervals or start - new_intervals[-1][1] > gap:
            new_intervals.append((round(start, ndigits), round(end, ndigits)))
        else:
            new_intervals[-1] = (new_intervals[-1][0], round(end, ndigits))
    return new_intervals


def load_mono_resampled(wav_path: str):
    x, sr = torchaudio.load(wav_path)
    if x.size(0) > 1:
        x = x.mean(dim=0, keepdim=True)
    if sr != SAMPLE_RATE:
        x = torchaudio.transforms.Resample(orig_freq=sr, new_freq=SAMPLE_RATE)(x)
        sr = SAMPLE_RATE
    return x.float(), sr


def process_wav_file(
    audio_file: str,
    model: WhisperWrapper,
    window_size_s: float = 10.0,
    stride_s: float = 5.0,
    device: str = "cuda",
):
    x, sr = load_mono_resampled(audio_file)
    x = x.to(device)

    length_s = x.size(1) / sr
    win_n = int(window_size_s * sr)

    child_pred, adult_pred, overlap_pred = [], [], []

    start_s = 0.0
    while start_s < length_s:
        start_n = int(start_s * sr)
        end_n = start_n + win_n

        x_window = x[:, start_n:end_n]

        # Pad final chunk to exactly 10s (model was trained on 10s windows)
        if x_window.size(1) < win_n:
            pad = win_n - x_window.size(1)
            x_window = torch.nn.functional.pad(x_window, (0, pad))

        with torch.no_grad():
            pred = model.forward_eval(x_window)
        pred = majority_filter(pred)
        child, adult, overlap = get_timestamps(pred)

        # Shift window-local timestamps to global time
        child_pred += [(start_s + s, start_s + e) for s, e in child]
        adult_pred += [(start_s + s, start_s + e) for s, e in adult]
        overlap_pred += [(start_s + s, start_s + e) for s, e in overlap]

        start_s += stride_s

    return (
        merge_segments(child_pred),
        merge_segments(adult_pred),
        merge_segments(overlap_pred),
    )


def write_segments_txt(out_path: str, segments, label: str):
    # simple tab-separated: start  end  label
    with open(out_path, "w") as f:
        for s, e in segments:
            f.write(f"{s:.2f}\t{e:.2f}\t{label}\n")


def write_rttm(out_path: str, recording_id: str, segments, label: str, min_dur=0.02):
    with open(out_path, "a") as f:
        for s, e in segments:
            dur = e - s
            if dur < min_dur:
                continue
            f.write(
                f"SPEAKER {recording_id} 1 {s:.3f} {dur:.3f} <NA> <NA> {label} <NA> <NA>\n"
            )

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav_file", type=str, default="")
    ap.add_argument("--wav_dir", type=str, default="")
    ap.add_argument("--filelist", type=str, default="")  # text file: one wav path per line
    ap.add_argument("--out_dir", type=str, required=True)

    ap.add_argument("--model_path", type=str, default="whisper-base_rank8_pretrained_50k.pt")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--window_size", type=float, default=10.0)
    ap.add_argument("--stride", type=float, default=5.0)

    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Collect inputs
    wavs = []
    if args.wav_file:
        wavs = [args.wav_file]
    elif args.filelist:
        with open(args.filelist) as f:
            wavs = [line.strip() for line in f if line.strip()]
    elif args.wav_dir:
        wavs = sorted(str(p) for p in Path(args.wav_dir).glob("*.wav"))
    else:
        raise ValueError("Provide one of --wav_file, --wav_dir, or --filelist")

    # Load model
    device = args.device
    model = WhisperWrapper()
    # replace positional embedding for 10s input audio (as repo README shows)
    model.backbone_model.encoder.embed_positions = (
        model.backbone_model.encoder.embed_positions.from_pretrained(model.embed_positions[:500])
    )
    sd = torch.load(args.model_path, map_location="cpu")
    model.load_state_dict(sd)
    model.to(device)
    model.eval()

    # Run batch
    for wav in wavs:
        stem = Path(wav).stem
        child, adult, overlap = process_wav_file(
            wav,
            model,
            window_size_s=args.window_size,
            stride_s=args.stride,
            device=device,
        )

        # Write three files per recording (easy + robust)
        rttm_path = out_dir / f"{stem}.rttm"
        if rttm_path.exists():
            rttm_path.unlink()  # start clean

        write_rttm(str(rttm_path), stem, child, "CHI")
        write_rttm(str(rttm_path), stem, adult, "ADULT")
        write_rttm(str(rttm_path), stem, overlap, "OVL")

        print(f"[OK] {wav} -> {stem} ({len(child)} child, {len(adult)} adult, {len(overlap)} overlap)")

if __name__ == "__main__":
    main()