"""
Convolutional VAE for 12-16 month infant vocalizations.

Operates on mel spectrograms (80 mel bins). The encoder maps spectrogram frames
to a latent distribution; the decoder reconstructs the spectrogram. Audio waveforms
are produced via Griffin-Lim vocoding.

Accepts config from synthesis/configs/vae_12m.yaml.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, 3, stride=stride, padding=1),
            nn.BatchNorm1d(out_ch),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        return self.net(x)


class ConvTransposeBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose1d(in_ch, out_ch, 4, stride=stride, padding=1),
            nn.BatchNorm1d(out_ch),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        return self.net(x)


class VAEEncoder(nn.Module):
    def __init__(self, n_mels: int = 80, latent_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            ConvBlock(n_mels, 256),
            ConvBlock(256, 256, stride=2),
            ConvBlock(256, 128, stride=2),
            ConvBlock(128, 128),
        )
        self.mu_proj = nn.Conv1d(128, latent_dim, 1)
        self.logvar_proj = nn.Conv1d(128, latent_dim, 1)

    def forward(self, x):
        h = self.net(x)
        return self.mu_proj(h), self.logvar_proj(h)


class VAEDecoder(nn.Module):
    def __init__(self, n_mels: int = 80, latent_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            ConvTransposeBlock(latent_dim, 128),
            ConvTransposeBlock(128, 256, stride=2),
            ConvTransposeBlock(256, 256, stride=2),
            nn.Conv1d(256, n_mels, 1),
        )

    def forward(self, z):
        return self.net(z)


class VAEModel(nn.Module):
    """
    Convolutional VAE for infant vocalization synthesis (12-16 month age group).

    Input/output: mel spectrograms, shape (batch, n_mels, time_frames).
    """

    def __init__(self, n_mels: int = 80, latent_dim: int = 64,
                 sample_rate: int = 16000, n_fft: int = 1024,
                 hop_length: int = 256, n_iter: int = 32):
        super().__init__()
        self.n_mels = n_mels
        self.latent_dim = latent_dim
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_iter = n_iter

        self.encoder = VAEEncoder(n_mels, latent_dim)
        self.decoder = VAEDecoder(n_mels, latent_dim)

        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            power=1.0,
        )
        self.griffinlim = torchaudio.transforms.GriffinLim(
            n_fft=n_fft,
            hop_length=hop_length,
            n_iter=n_iter,
        )
        self.inverse_mel = torchaudio.transforms.InverseMelScale(
            n_stft=n_fft // 2 + 1,
            n_mels=n_mels,
            sample_rate=sample_rate,
        )

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if self.training:
            std = (0.5 * logvar).exp()
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def forward(self, mel: torch.Tensor):
        mu, logvar = self.encoder(mel)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)
        return recon, mu, logvar

    def loss(self, mel: torch.Tensor, recon: torch.Tensor,
             mu: torch.Tensor, logvar: torch.Tensor,
             kl_weight: float = 0.01) -> torch.Tensor:
        min_len = min(recon.shape[2], mel.shape[2])
        recon_loss = F.l1_loss(recon[:, :, :min_len], mel[:, :, :min_len])
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + kl_weight * kl_loss

    def audio_to_mel(self, wav: torch.Tensor) -> torch.Tensor:
        mel = self.mel_transform(wav)
        return (mel + 1e-8).log()

    @torch.no_grad()
    def mel_to_audio(self, log_mel: torch.Tensor) -> torch.Tensor:
        mel = log_mel.exp()
        spec = self.inverse_mel(mel)
        wav = self.griffinlim(spec)
        return wav

    @torch.no_grad()
    def generate(self, n_samples: int = 1, n_frames: int = 200,
                 device: str = "cpu") -> torch.Tensor:
        z = torch.randn(n_samples, self.latent_dim, n_frames // 4, device=device)
        log_mel = self.decoder(z)
        wavs = []
        for i in range(n_samples):
            wav = self.mel_to_audio(log_mel[i])
            wavs.append(wav)
        return torch.stack(wavs)
