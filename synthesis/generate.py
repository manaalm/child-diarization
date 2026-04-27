"""
Generate synthetic child speech samples from a trained model.

Usage:
    python synthesis/generate.py \\
        --checkpoint synthesis/checkpoints/12_16m_vae_v1/best_checkpoint.pt \\
        --age-group 12_16m --n-samples 1000 --seed 42
    python synthesis/generate.py \\
        --checkpoint synthesis/checkpoints/34_38m_vits_v1/best_checkpoint.pt \\
        --age-group 34_38m --n-samples 1000 --seed 42

Exit codes:
    0 = success
    1 = checkpoint error
    2 = output error
"""

import argparse
import json
import os
import sys
import uuid
from pathlib import Path

import numpy as np
import torch
import torchaudio

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

SAMPLE_RATE = 16000


def detect_model_type(checkpoint_path: str) -> str:
    state = torch.load(checkpoint_path, map_location="cpu")
    keys = list(state.get("model", state).keys())
    if any("encoder.net" in k or "decoder.net" in k for k in keys):
        return "vae"
    return "vits"


def generate_vae(checkpoint_path: str, n_samples: int, duration_range: tuple,
                 device: str, seed: int) -> list:
    from synthesis.models.vae_model import VAEModel

    torch.manual_seed(seed)
    state = torch.load(checkpoint_path, map_location=device)
    model = VAEModel().to(device)
    model.load_state_dict(state["model"])
    model.eval()

    min_dur, max_dur = duration_range
    fps = SAMPLE_RATE / 256  # frames per second (hop_length=256)
    min_frames = max(8, int(min_dur * fps))
    max_frames = int(max_dur * fps)

    wavs = []
    for i in range(n_samples):
        np.random.seed(seed + i)
        n_frames = np.random.randint(min_frames, max_frames + 1)
        # Latent sequence length is downsampled by 4 in the encoder
        lat_frames = max(2, n_frames // 4)
        z = torch.randn(1, model.latent_dim, lat_frames, device=device)
        with torch.no_grad():
            log_mel = model.decoder(z)
            wav = model.mel_to_audio(log_mel[0])
        wavs.append(wav.cpu())

    return wavs


def generate_vits(checkpoint_path: str, n_samples: int, duration_range: tuple,
                  device: str, seed: int,
                  speaker_embedding: torch.Tensor = None) -> list:
    from synthesis.models.vits_model import VITSModel

    # Find config alongside checkpoint
    ckpt_dir = Path(checkpoint_path).parent
    config_path = str(ckpt_dir / "config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"VITS config not found at {config_path}")

    model = VITSModel(config_path=config_path, checkpoint_path=checkpoint_path, device=device)
    if not model.is_loaded():
        raise RuntimeError(f"Failed to load VITS checkpoint from {checkpoint_path}")

    # Simple English phoneme-like syllables for toddler speech stimulation
    text_stimuli = ["a", "ba", "da", "ga", "ma", "na", "pa", "ta", "wa", "ya",
                    "baba", "mama", "dada", "nana", "wawa"]

    torch.manual_seed(seed)
    wavs = []
    for i in range(n_samples):
        text = text_stimuli[i % len(text_stimuli)]
        wav_batch = model.generate(n_samples=1, text=text,
                                    speaker_embedding=speaker_embedding)
        wavs.append(wav_batch[0].cpu())

    return wavs


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic child speech samples.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--age-group", required=True, choices=["12_16m", "34_38m"])
    parser.add_argument("--n-samples", type=int, required=True)
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--duration-range", default="1.0,5.0")
    parser.add_argument("--prototype-path", default="",
                        help="Path to ECAPA age-group prototype .pt file for conditioning.")
    args = parser.parse_args()

    if not os.path.exists(args.checkpoint):
        print(f"ERROR: Checkpoint not found: {args.checkpoint}", file=sys.stderr)
        sys.exit(1)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    try:
        min_dur, max_dur = [float(x) for x in args.duration_range.split(",")]
    except ValueError:
        print(f"ERROR: Invalid --duration-range: {args.duration_range}", file=sys.stderr)
        sys.exit(1)

    ckpt_path = Path(args.checkpoint)
    model_name = ckpt_path.parent.name

    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        out_dir = REPO_ROOT / "synthesis" / "generated" / model_name / args.age_group
    out_dir.mkdir(parents=True, exist_ok=True)

    speaker_embedding = None
    if args.prototype_path and os.path.exists(args.prototype_path):
        speaker_embedding = torch.load(args.prototype_path, map_location="cpu")
        if isinstance(speaker_embedding, dict):
            speaker_embedding = speaker_embedding.get("embedding", next(iter(speaker_embedding.values())))
        if speaker_embedding.dim() == 1:
            speaker_embedding = speaker_embedding.unsqueeze(0)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_type = detect_model_type(str(ckpt_path))
    print(f"Generating {args.n_samples} samples [{args.age_group}] using {model_type} on {device}")

    try:
        if model_type == "vae":
            wavs = generate_vae(str(ckpt_path), args.n_samples, (min_dur, max_dur),
                                device, args.seed)
        else:
            wavs = generate_vits(str(ckpt_path), args.n_samples, (min_dur, max_dur),
                                  device, args.seed, speaker_embedding)
    except Exception as e:
        print(f"ERROR: Generation failed: {e}", file=sys.stderr)
        sys.exit(2)

    registry_path = out_dir / "registry.jsonl"
    with open(registry_path, "w") as reg_f:
        for i, wav in enumerate(wavs):
            sample_id = str(uuid.uuid4())
            out_name = f"sample_{i:06d}.wav"
            out_path = out_dir / out_name
            try:
                if wav.dim() == 1:
                    wav = wav.unsqueeze(0)
                torchaudio.save(str(out_path), wav, SAMPLE_RATE)
            except Exception as e:
                print(f"  WARNING: Failed to save sample {i}: {e}")
                continue

            record = {
                "sample_id": sample_id,
                "age_group": args.age_group,
                "model_name": model_name,
                "path": str(out_path.resolve()),
                "seed": args.seed + i,
                "duration_secs": round(wav.shape[-1] / SAMPLE_RATE, 4),
                "mcd_score": None,
                "speaker_similarity": None,
                "age_classifier_pred": None,
                "split_usage": "eval_only",
            }
            reg_f.write(json.dumps(record) + "\n")

    print(f"Generated {len(wavs)} samples → {out_dir}")
    print(f"Registry: {registry_path}")


if __name__ == "__main__":
    main()
