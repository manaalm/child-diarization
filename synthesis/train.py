"""
Train a child speech synthesis model (VITS or VAE) for a given age group.

Usage:
    python synthesis/train.py --config synthesis/configs/vae_12m.yaml --age-group 12_16m
    python synthesis/train.py --config synthesis/configs/vits_34m.yaml --age-group 34_38m

Exit codes:
    0 = success
    1 = config error
    2 = data error
    3 = training failure
"""

import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torchaudio
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def resolve_paths(cfg: dict, repo_root: Path) -> dict:
    for key in ("train_manifest", "val_manifest"):
        p = cfg["data"].get(key, "")
        if p and not os.path.isabs(p):
            cfg["data"][key] = str(repo_root / p)
    chk = cfg["output"].get("checkpoint_dir", "synthesis/checkpoints/")
    if not os.path.isabs(chk):
        cfg["output"]["checkpoint_dir"] = str(repo_root / chk)
    return cfg


class WavDataset(torch.utils.data.Dataset):
    """Dataset loading WAV files from a manifest CSV (path, age_group columns)."""

    def __init__(self, manifest_path: str, sample_rate: int = 16000,
                 min_dur: float = 0.5, max_dur: float = 5.0):
        self.sample_rate = sample_rate
        self.min_samples = int(min_dur * sample_rate)
        self.max_samples = int(max_dur * sample_rate)

        import pandas as pd
        df = pd.read_csv(manifest_path)
        if "path" not in df.columns:
            raise ValueError(f"Manifest {manifest_path} must have a 'path' column")
        self.paths = [p for p in df["path"].tolist() if os.path.exists(p)]
        if not self.paths:
            raise ValueError(f"No valid audio files found in {manifest_path}")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        wav, sr = torchaudio.load(self.paths[idx])
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        if sr != self.sample_rate:
            wav = torchaudio.functional.resample(wav, sr, self.sample_rate)
        wav = wav.squeeze(0)
        n = wav.numel()
        if n < self.min_samples:
            wav = torch.nn.functional.pad(wav, (0, self.min_samples - n))
        elif n > self.max_samples:
            start = np.random.randint(0, n - self.max_samples)
            wav = wav[start: start + self.max_samples]
        return wav


def collate_fn(batch):
    max_len = max(x.numel() for x in batch)
    padded = torch.zeros(len(batch), max_len)
    for i, x in enumerate(batch):
        padded[i, : x.numel()] = x
    return padded


def train_vae(cfg: dict, out_dir: Path, device: str) -> None:
    from synthesis.models.vae_model import VAEModel

    sample_rate = cfg["data"]["sample_rate"]
    n_mels = 80
    model = VAEModel(n_mels=n_mels, sample_rate=sample_rate).to(device)

    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate, n_fft=1024, hop_length=256, n_mels=n_mels, power=1.0,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["training"]["learning_rate"])

    train_ds = WavDataset(
        cfg["data"]["train_manifest"],
        sample_rate=sample_rate,
        min_dur=cfg["data"]["min_duration_secs"],
        max_dur=cfg["data"]["max_duration_secs"],
    )
    val_ds = WavDataset(
        cfg["data"]["val_manifest"],
        sample_rate=sample_rate,
        min_dur=cfg["data"]["min_duration_secs"],
        max_dur=cfg["data"]["max_duration_secs"],
    )
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=cfg["training"]["batch_size"],
        shuffle=True, collate_fn=collate_fn, num_workers=2,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=cfg["training"]["batch_size"],
        shuffle=False, collate_fn=collate_fn, num_workers=2,
    )

    log_rows = []
    best_val_loss = float("inf")
    patience_counter = 0
    patience = cfg["training"]["early_stopping_patience"]
    log_interval = cfg["output"]["log_interval"]

    for epoch in range(1, cfg["training"]["max_epochs"] + 1):
        model.train()
        train_loss = 0.0
        for step, wav_batch in enumerate(train_loader):
            wav_batch = wav_batch.to(device)
            mel = (mel_transform(wav_batch) + 1e-8).log()
            recon, mu, logvar = model(mel)
            loss = model.loss(mel, recon, mu, logvar)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
            if (step + 1) % log_interval == 0:
                print(f"  Epoch {epoch} step {step+1}/{len(train_loader)} loss={loss.item():.4f}")

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for wav_batch in val_loader:
                wav_batch = wav_batch.to(device)
                mel = (mel_transform(wav_batch) + 1e-8).log()
                recon, mu, logvar = model(mel)
                val_loss += model.loss(mel, recon, mu, logvar).item()

        avg_train = train_loss / len(train_loader)
        avg_val = val_loss / len(val_loader)
        log_rows.append({"epoch": epoch, "train_loss": avg_train, "val_loss": avg_val})
        print(f"Epoch {epoch}: train={avg_train:.4f}  val={avg_val:.4f}")

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save({"model": model.state_dict(), "epoch": epoch, "val_loss": avg_val},
                       out_dir / "best_checkpoint.pt")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    with open(out_dir / "training_log.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_loss"])
        writer.writeheader()
        writer.writerows(log_rows)


def train_vits(cfg: dict, out_dir: Path, device: str) -> None:
    # VITS fine-tuning uses Coqui TTS trainer infrastructure.
    # This function sets up the Coqui config and delegates to TTS trainer.
    try:
        from TTS.trainer import Trainer, TrainingArgs
        from TTS.tts.configs.vits_config import VitsConfig
        from TTS.tts.datasets import load_tts_samples
        from TTS.tts.models.vits import Vits, VitsArgs, VitsAudioConfig
    except ImportError:
        print("ERROR: Coqui TTS not installed. Run: cd synthesis && uv sync", file=sys.stderr)
        sys.exit(3)

    vits_audio_cfg = VitsAudioConfig(
        sample_rate=cfg["data"]["sample_rate"],
        win_length=1024,
        hop_length=256,
        mel_fmin=0,
        mel_fmax=None,
    )
    vits_cfg = VitsConfig(
        audio=vits_audio_cfg,
        batch_size=cfg["training"]["batch_size"],
        eval_batch_size=max(1, cfg["training"]["batch_size"] // 2),
        num_loader_workers=2,
        run_eval=True,
        test_delay_epochs=-1,
        epochs=cfg["training"]["max_epochs"],
        text_cleaner="phoneme_cleaners",
        use_phonemes=True,
        phoneme_language="en-us",
        phoneme_cache_path=str(out_dir / "phoneme_cache"),
        print_step=cfg["output"]["log_interval"],
        output_path=str(out_dir),
        datasets=[{
            "formatter": "ljspeech",
            "meta_file_train": cfg["data"]["train_manifest"],
            "meta_file_val": cfg["data"]["val_manifest"],
            "path": str(REPO_ROOT),
            "language": "en",
        }],
    )

    training_args = TrainingArgs(
        restore_path=cfg.get("training", {}).get("resume", None),
        skip_train_epoch=False,
    )
    model = Vits(vits_cfg)
    trainer = Trainer(
        args=training_args,
        config=vits_cfg,
        output_path=str(out_dir),
        model=model,
    )
    trainer.fit()


def main():
    parser = argparse.ArgumentParser(description="Train child speech synthesis model.")
    parser.add_argument("--config", required=True, help="Path to synthesis training YAML config.")
    parser.add_argument("--age-group", required=True, choices=["12_16m", "34_38m"])
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", default="", help="Path to checkpoint to resume from.")
    args = parser.parse_args()

    if not os.path.exists(args.config):
        print(f"ERROR: Config not found: {args.config}", file=sys.stderr)
        sys.exit(1)

    cfg = resolve_paths(load_config(args.config), REPO_ROOT)

    if cfg["model"].get("age_group") != args.age_group:
        print(f"WARNING: Config age_group={cfg['model'].get('age_group')} "
              f"but --age-group={args.age_group}. Using --age-group.")
        cfg["model"]["age_group"] = args.age_group

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    cfg["training"]["seed"] = args.seed
    if args.resume:
        cfg["training"]["resume"] = args.resume

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_type = cfg["model"]["type"]
    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        out_dir = Path(cfg["output"]["checkpoint_dir"]) / f"{args.age_group}_{model_type}_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save resolved config alongside results
    with open(out_dir / "config.json", "w") as f:
        json.dump({"config_path": args.config, "age_group": args.age_group,
                   "seed": args.seed, **cfg}, f, indent=2)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training {model_type} for {args.age_group} on {device}")
    print(f"Output: {out_dir}")

    try:
        if model_type == "vae":
            train_vae(cfg, out_dir, device)
        elif model_type == "vits":
            train_vits(cfg, out_dir, device)
        else:
            print(f"ERROR: Unknown model type: {model_type}", file=sys.stderr)
            sys.exit(1)
    except Exception as e:
        print(f"Training failed: {e}", file=sys.stderr)
        sys.exit(3)

    print(f"\nTraining complete. Checkpoint at {out_dir}/best_checkpoint.pt")


if __name__ == "__main__":
    main()
