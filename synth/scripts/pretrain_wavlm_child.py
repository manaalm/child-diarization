"""
Continued masked-speech-unit pretraining of WavLM-Base+ on child speech.

Loads microsoft/wavlm-base-plus, applies random span masking, and trains a
projection head to predict CNN features at masked positions (MSE loss). This
forces the transformer to learn child-adapted representations without altering
the architecture used by downstream MIL models.

Usage:
    python synth/scripts/pretrain_wavlm_child.py \
        --wav-list synth_results/child_wavs.txt \
        --output-dir synth_results/child_wavlm_checkpoint \
        --max-steps 50000 \
        [--resume-from-checkpoint synth_results/child_wavlm_checkpoint/step_45000]
"""

import argparse
import csv
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from transformers import WavLMModel, WavLMConfig

SAMPLE_RATE = 16000
CLIP_DURATION_SEC = 5.0
CLIP_SAMPLES = int(SAMPLE_RATE * CLIP_DURATION_SEC)

# Span masking params (match WavLM paper §4)
MASK_PROB = 0.065        # probability of starting a mask span
MASK_LENGTH = 10         # span length in CNN frames
MIN_MASKS = 1


def load_wavs(wav_list_path: str) -> list[str]:
    with open(wav_list_path) as f:
        paths = [line.strip() for line in f if line.strip()]
    return [p for p in paths if os.path.exists(p)]


def load_clip(path: str, rng: random.Random) -> torch.Tensor | None:
    try:
        waveform, sr = torchaudio.load(path)
    except Exception:
        return None
    if sr != SAMPLE_RATE:
        waveform = torchaudio.functional.resample(waveform, sr, SAMPLE_RATE)
    waveform = waveform.mean(0)  # mono
    if waveform.shape[0] >= CLIP_SAMPLES:
        start = rng.randint(0, waveform.shape[0] - CLIP_SAMPLES)
        waveform = waveform[start : start + CLIP_SAMPLES]
    else:
        pad = CLIP_SAMPLES - waveform.shape[0]
        waveform = F.pad(waveform, (0, pad))
    return waveform  # [CLIP_SAMPLES]


def compute_mask_indices(seq_len: int, batch_size: int, rng: random.Random) -> torch.BoolTensor:
    """Random span masking in CNN-feature time dimension."""
    mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
    num_spans = max(MIN_MASKS, int(seq_len * MASK_PROB))
    for b in range(batch_size):
        starts = rng.choices(range(seq_len - MASK_LENGTH), k=num_spans)
        for s in starts:
            mask[b, s : s + MASK_LENGTH] = True
    return mask


class WavLMPretrainer(nn.Module):
    def __init__(self, model_name: str = "microsoft/wavlm-base-plus"):
        super().__init__()
        self.wavlm = WavLMModel.from_pretrained(model_name)
        cfg: WavLMConfig = self.wavlm.config
        conv_out_dim = cfg.conv_dim[-1]   # 512 for base models
        hidden_size = cfg.hidden_size     # 768 for base models
        self.pred_head = nn.Linear(hidden_size, conv_out_dim)
        self._conv_out_dim = conv_out_dim

    def forward(self, input_values: torch.Tensor, mask_time_indices: torch.BoolTensor):
        # CNN features as prediction targets (no grad needed)
        with torch.no_grad():
            cnn_features = self.wavlm.feature_extractor(input_values)
            cnn_features = cnn_features.transpose(1, 2)  # [B, T, conv_out_dim]

        outputs = self.wavlm(
            input_values=input_values,
            mask_time_indices=mask_time_indices,
            output_hidden_states=False,
        )
        hidden = outputs.last_hidden_state  # [B, T, hidden_size]

        # Align lengths (CNN stride can differ by ±1 frame)
        T = min(hidden.shape[1], cnn_features.shape[1], mask_time_indices.shape[1])
        hidden = hidden[:, :T, :]
        cnn_features = cnn_features[:, :T, :]
        mask = mask_time_indices[:, :T]

        if not mask.any():
            return torch.tensor(0.0, device=hidden.device, requires_grad=True)

        pred = self.pred_head(hidden[mask])      # [N_masked, conv_out_dim]
        target = cnn_features[mask].detach()
        loss = F.mse_loss(pred, target)
        return loss


def save_checkpoint(model: WavLMPretrainer, optimizer, step: int, output_dir: Path):
    ckpt_dir = output_dir / f"step_{step}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    model.wavlm.save_pretrained(str(ckpt_dir))
    torch.save({
        "step": step,
        "pred_head": model.pred_head.state_dict(),
        "optimizer": optimizer.state_dict(),
    }, ckpt_dir / "trainer_state.pt")
    return ckpt_dir


def load_checkpoint(model: WavLMPretrainer, optimizer, ckpt_path: str) -> int:
    state = torch.load(ckpt_path + "/trainer_state.pt", map_location="cpu")
    model.pred_head.load_state_dict(state["pred_head"])
    optimizer.load_state_dict(state["optimizer"])
    # Reload transformer weights
    model.wavlm = WavLMModel.from_pretrained(ckpt_path)
    return state["step"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wav-list", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-name", default="microsoft/wavlm-base-plus")
    parser.add_argument("--max-steps", type=int, default=50000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--save-every", type=int, default=5000)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume-from-checkpoint", default=None)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading WAV list from {args.wav_list}")
    wav_paths = load_wavs(args.wav_list)
    print(f"Found {len(wav_paths)} WAV files")

    print(f"Loading model: {args.model_name}")
    model = WavLMPretrainer(args.model_name).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    start_step = 0
    if args.resume_from_checkpoint:
        print(f"Resuming from {args.resume_from_checkpoint}")
        start_step = load_checkpoint(model, optimizer, args.resume_from_checkpoint)
        model = model.to(device)
        print(f"Resumed at step {start_step}")

    log_path = output_dir / "training_log.csv"
    log_exists = log_path.exists() and start_step > 0
    log_file = open(log_path, "a" if log_exists else "w", newline="")
    log_writer = csv.writer(log_file)
    if not log_exists:
        log_writer.writerow(["step", "loss", "elapsed_sec"])

    model.train()
    step = start_step
    loss_accum = 0.0
    t0 = time.time()

    print(f"Starting training from step {step}, max steps {args.max_steps}")

    while step < args.max_steps:
        # Sample a batch
        batch_paths = rng.choices(wav_paths, k=args.batch_size)
        clips = []
        for p in batch_paths:
            c = load_clip(p, rng)
            if c is not None:
                clips.append(c)
        if not clips:
            continue
        input_values = torch.stack(clips).to(device)  # [B, CLIP_SAMPLES]

        # CNN sequence length: WavLM base uses 320-sample stride
        T_cnn = (CLIP_SAMPLES - 400) // 320 + 1
        mask = compute_mask_indices(T_cnn, len(clips), rng).to(device)

        optimizer.zero_grad()
        loss = model(input_values, mask)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        step += 1
        loss_accum += loss.item()

        if step % args.log_every == 0:
            avg_loss = loss_accum / args.log_every
            elapsed = time.time() - t0
            print(f"Step {step}/{args.max_steps}  loss={avg_loss:.4f}  elapsed={elapsed:.0f}s")
            log_writer.writerow([step, f"{avg_loss:.6f}", f"{elapsed:.1f}"])
            log_file.flush()
            loss_accum = 0.0

        if step % args.save_every == 0:
            ckpt_dir = save_checkpoint(model, optimizer, step, output_dir)
            print(f"Saved checkpoint: {ckpt_dir}")
            # Keep only last 2 checkpoints to save disk
            existing = sorted(output_dir.glob("step_*"), key=lambda p: int(p.name.split("_")[1]))
            for old in existing[:-2]:
                import shutil
                shutil.rmtree(old)

    # Final save
    ckpt_dir = save_checkpoint(model, optimizer, step, output_dir)
    print(f"Training complete. Final checkpoint: {ckpt_dir}")
    log_file.close()


if __name__ == "__main__":
    main()
