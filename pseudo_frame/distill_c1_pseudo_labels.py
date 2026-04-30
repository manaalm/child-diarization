"""Distill the C1 USC-SAIL synth-only frame classifier into pseudo-frame labels.

Spec-016 follow-up #8: replace the VTC+USC-SAIL averaged pseudo-labels
(pseudo_frame/pseudo_labels/index.csv, mean per-clip Pearson 0.566) with
predictions from the spec-016 C1 checkpoint
(whisper-modeling/checkpoints/whisper_base_synth/epoch=17-val_loss=0.235.ckpt,
test frame-level accuracy 0.922 on synth val).

Per-frame target: P(child speech) = softmax(C1)[child] + softmax(C1)[overlap]
                  at 50 Hz (matches WavLM-Base+ output rate).

For positive clips (label=1) only — negatives stay all-zero (clip-level supervision).

Cache layout (mirrors build_pseudo_labels.py):
  pseudo_frame/pseudo_labels_c1/{md5(audio_path)}.npy    float32 (T,)
  pseudo_frame/pseudo_labels_c1/index.csv                same schema as the baseline

Usage:
  PYTHONPATH=. python pseudo_frame/distill_c1_pseudo_labels.py
"""
import argparse
import hashlib
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchaudio

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, "whisper-modeling"))

from models.whisper import WhisperWrapper  # noqa: E402

SAMPLE_RATE = 16000
WINDOW_SEC = 30.0      # C1 was trained at window_size=30 (transformers >=4.57 mel-3000 enforcement)
STRIDE_SEC = 20.0      # overlap window so the boundary frames are predicted twice and averaged
FRAME_STEP_SEC = 0.02
FRAME_RATE = int(round(1.0 / FRAME_STEP_SEC))  # 50 Hz
FRAMES_PER_WINDOW = int(WINDOW_SEC * FRAME_RATE)  # 1500

OUT_DIR = os.path.join(_REPO, "pseudo_frame/pseudo_labels_c1")
SPLIT_CSV = os.path.join(_REPO, "whisper-modeling/seen_child_splits/master_with_split.csv")
CKPT_PATH = os.path.join(_REPO, "whisper-modeling/checkpoints/whisper_base_synth/epoch=17-val_loss=0.235.ckpt")


def audio_id(audio_path: str) -> str:
    return hashlib.md5(audio_path.encode("utf-8")).hexdigest()


def load_mono_resampled(wav_path: str) -> torch.Tensor:
    x, sr = torchaudio.load(wav_path)
    if x.size(0) > 1:
        x = x.mean(dim=0, keepdim=True)
    if sr != SAMPLE_RATE:
        x = torchaudio.functional.resample(x, sr, SAMPLE_RATE)
    return x.float().squeeze(0)  # (T_audio,)


def load_c1_model(ckpt_path: str, device: str) -> WhisperWrapper:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    hp = ckpt.get("hyper_parameters", {})
    embedding = hp.get("embedding", "whisper_base")
    lora_rank = hp.get("lora_rank", 8)

    model = WhisperWrapper(pretrained_model=embedding, lora_rank=lora_rank)

    state_dict = ckpt["state_dict"]
    cleaned = {k.replace("model.", "", 1) if k.startswith("model.") else k: v
               for k, v in state_dict.items()}
    embed_pos_w = ckpt.get("embed_pos_weight")
    if embed_pos_w is not None:
        model.backbone_model.encoder.embed_positions.weight = torch.nn.Parameter(embed_pos_w)
    msg = model.load_state_dict(cleaned, strict=False)
    if msg.missing_keys:
        print(f"  Missing keys (first 5): {msg.missing_keys[:5]}", flush=True)
    if msg.unexpected_keys:
        print(f"  Unexpected keys (first 5): {msg.unexpected_keys[:5]}", flush=True)

    model.to(device).eval()
    return model


@torch.no_grad()
def child_probs_for_window(wav_window: torch.Tensor, model: WhisperWrapper, device: str) -> np.ndarray:
    """wav_window shape: (T_audio,) -> child+overlap probs (T_frames=1500,)."""
    x = wav_window.unsqueeze(0).to(device)  # (1, T_audio)
    logits = model(x)  # (1, 4, T_frames)
    probs = torch.softmax(logits, dim=1)
    child_p = probs[:, 1, :] + probs[:, 3, :]  # child + overlap (silence=0, child=1, adult=2, overlap=3)
    return child_p.squeeze(0).detach().cpu().numpy().astype(np.float32)


def distill_one(audio_path: str, model: WhisperWrapper, device: str) -> np.ndarray:
    """Return per-frame child-prob mask at 50 Hz, length matching audio duration."""
    wav = load_mono_resampled(audio_path)
    n_audio = wav.shape[0]
    n_frames_total = max(1, n_audio // int(SAMPLE_RATE * FRAME_STEP_SEC))

    # Single-window fast path
    if n_audio <= int(WINDOW_SEC * SAMPLE_RATE):
        pad = int(WINDOW_SEC * SAMPLE_RATE) - n_audio
        wav_padded = torch.cat([wav, torch.zeros(pad, dtype=wav.dtype)]) if pad > 0 else wav
        probs = child_probs_for_window(wav_padded, model, device)
        return probs[:n_frames_total]

    # Long clip: overlapping windows, average the soft probs in overlap regions
    window_n = int(WINDOW_SEC * SAMPLE_RATE)
    stride_n = int(STRIDE_SEC * SAMPLE_RATE)
    accum = np.zeros(n_frames_total, dtype=np.float64)
    weight = np.zeros(n_frames_total, dtype=np.float64)

    s = 0
    while s < n_audio:
        e = s + window_n
        if e > n_audio:
            wav_chunk = torch.nn.functional.pad(wav[s:n_audio], (0, e - n_audio))
        else:
            wav_chunk = wav[s:e]
        win_probs = child_probs_for_window(wav_chunk, model, device)  # (1500,)

        f_start = s // int(SAMPLE_RATE * FRAME_STEP_SEC)
        f_end_full = f_start + FRAMES_PER_WINDOW
        f_end = min(f_end_full, n_frames_total)
        usable = f_end - f_start
        if usable > 0:
            accum[f_start:f_end] += win_probs[:usable]
            weight[f_start:f_end] += 1.0
        s += stride_n

    weight = np.where(weight > 0, weight, 1.0)
    return (accum / weight).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=CKPT_PATH)
    ap.add_argument("--out-dir", default=OUT_DIR)
    ap.add_argument("--split-csv", default=SPLIT_CSV)
    ap.add_argument("--limit", type=int, default=None,
                    help="If set, process only first N clips (smoke test)")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    if not os.path.exists(args.ckpt):
        raise FileNotFoundError(f"C1 checkpoint missing: {args.ckpt}")

    os.makedirs(args.out_dir, exist_ok=True)

    df = pd.read_csv(args.split_csv)
    df = df[df["audio_exists"] == True].reset_index(drop=True)
    if args.limit:
        df = df.head(args.limit)

    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"Loading C1 model on {device}: {args.ckpt}", flush=True)
    model = load_c1_model(args.ckpt, device)
    print(f"Model loaded. Distilling pseudo-labels for {len(df)} clips → {args.out_dir}", flush=True)
    print(f"Window={WINDOW_SEC}s stride={STRIDE_SEC}s frame_rate={FRAME_RATE}Hz", flush=True)

    rows = []
    n_pos = n_neg = 0
    t0 = time.time()
    for i, row in enumerate(df.itertuples(index=False)):
        audio_path = row.audio_path
        label = int(row.label)

        npy_path = os.path.join(args.out_dir, f"{audio_id(audio_path)}.npy")

        try:
            wav_info = torchaudio.info(audio_path)
            n_audio = int(round(wav_info.num_frames * (SAMPLE_RATE / wav_info.sample_rate)))
            n_frames_total = max(1, n_audio // int(SAMPLE_RATE * FRAME_STEP_SEC))
        except Exception as e:
            print(f"  ERROR audio info {audio_path}: {e}", flush=True)
            continue

        if label == 0:
            mask = np.zeros(n_frames_total, dtype=np.float32)
            sources = ""
            n_src = 0
            n_neg += 1
        else:
            try:
                mask = distill_one(audio_path, model, device)
            except Exception as e:
                print(f"  ERROR distill {audio_path}: {e}", flush=True)
                continue
            sources = "c1_usc_sail_synth"
            n_src = 1
            n_pos += 1

        np.save(npy_path, mask)
        rows.append({
            "audio_path": audio_path,
            "split": row.split,
            "label": label,
            "n_frames": int(len(mask)),
            "n_sources": n_src,
            "sources": sources,
            "clip_pos_rate": float(round(mask.mean(), 4)),
            "npy_path": npy_path,
        })

        if (i + 1) % 100 == 0 or (i + 1) == len(df):
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (len(df) - (i + 1)) / max(rate, 1e-6)
            print(f"  {i+1}/{len(df)}  rate={rate:.2f}/s  ETA={eta:.0f}s", flush=True)

    idx = pd.DataFrame(rows)
    idx_path = os.path.join(args.out_dir, "index.csv")
    idx.to_csv(idx_path, index=False)

    print("\n=== STATS ===", flush=True)
    print(f"Total clips written: {len(idx)}", flush=True)
    print(f"By split:    {idx['split'].value_counts().to_dict()}", flush=True)
    print(f"By label:    {idx['label'].value_counts().to_dict()}", flush=True)
    print(f"Positives with C1 distill: {n_pos} | Negatives (zero mask): {n_neg}", flush=True)
    pos = idx[idx['label'] == 1]
    if len(pos):
        print(f"Mean clip-level pos rate (positives): {pos['clip_pos_rate'].mean():.3f}", flush=True)
        print(f"  std={pos['clip_pos_rate'].std():.3f} min={pos['clip_pos_rate'].min():.3f} max={pos['clip_pos_rate'].max():.3f}", flush=True)
    print(f"Wrote index → {idx_path}", flush=True)


if __name__ == "__main__":
    main()
