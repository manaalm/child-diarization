"""
Audio utility functions for the synthetic scene generator.

All functions operate on 1-D float32 numpy arrays at 16 kHz.
"""

import numpy as np
from scipy.signal import fftconvolve


def resample_to_16k(wav: np.ndarray, sr: int) -> np.ndarray:
    """Resample a waveform to 16 kHz.

    Parameters
    ----------
    wav : np.ndarray
        1-D float32 waveform array.
    sr : int
        Current sample rate of ``wav``.

    Returns
    -------
    np.ndarray
        1-D float32 waveform resampled to 16 000 Hz.  If ``sr`` already equals
        16 000, the input array is returned as-is (no copy).
    """
    if sr == 16000:
        return wav.astype(np.float32)

    import torch
    import torchaudio.functional as F

    tensor = torch.from_numpy(wav.astype(np.float32))
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)  # (1, T) required by torchaudio
    resampled = F.resample(tensor, orig_freq=sr, new_freq=16000)
    return resampled.squeeze(0).numpy().astype(np.float32)


def peak_normalize(wav: np.ndarray, target_peak: float = 0.95) -> np.ndarray:
    """Scale a waveform so that its peak absolute value equals ``target_peak``.

    Parameters
    ----------
    wav : np.ndarray
        1-D float32 waveform array.
    target_peak : float
        Desired peak amplitude (default 0.95 to leave headroom).

    Returns
    -------
    np.ndarray
        Peak-normalised float32 array.  Silent signals (peak < 1e-8) are
        returned unchanged.
    """
    peak = np.max(np.abs(wav))
    if peak < 1e-8:
        return wav.astype(np.float32)
    return (wav * (target_peak / peak)).astype(np.float32)


def apply_crossfade(wav: np.ndarray, crossfade_samples: int) -> np.ndarray:
    """Apply a half-Hann fade-in at the start and fade-out at the end.

    Parameters
    ----------
    wav : np.ndarray
        1-D float32 waveform array.
    crossfade_samples : int
        Number of samples over which to apply each fade.

    Returns
    -------
    np.ndarray
        Waveform with fade-in/out applied.  If ``wav`` is shorter than
        ``2 * crossfade_samples``, the original array is returned unchanged.
    """
    wav = wav.astype(np.float32)
    if len(wav) < 2 * crossfade_samples:
        return wav

    # Half-Hann window: values 0 → 1 over crossfade_samples points
    hann_half = np.hanning(2 * crossfade_samples)[:crossfade_samples].astype(np.float32)

    result = wav.copy()
    result[:crossfade_samples] *= hann_half               # fade-in
    result[-crossfade_samples:] *= hann_half[::-1]        # fade-out
    return result


def convolve_rir(wav: np.ndarray, rir_wav: np.ndarray) -> np.ndarray:
    """Convolve a waveform with a room impulse response.

    Uses FFT-based convolution (``scipy.signal.fftconvolve``) and trims the
    result to the same length as the input.  The output peak is normalised to
    match the input peak so that RIR application does not change loudness.

    Parameters
    ----------
    wav : np.ndarray
        1-D float32 input waveform.
    rir_wav : np.ndarray
        1-D float32 room impulse response.

    Returns
    -------
    np.ndarray
        Reverberated float32 waveform of the same length as ``wav``.
    """
    wav = wav.astype(np.float32)
    rir_wav = rir_wav.astype(np.float32)

    input_peak = np.max(np.abs(wav))

    convolved = fftconvolve(wav, rir_wav, mode="full")
    # Trim to original length
    convolved = convolved[: len(wav)].astype(np.float32)

    # Restore original peak level
    conv_peak = np.max(np.abs(convolved))
    if conv_peak > 1e-8 and input_peak > 1e-8:
        convolved *= input_peak / conv_peak

    return convolved


def mix_at_snr(
    speech_wav: np.ndarray,
    noise_wav: np.ndarray,
    snr_db: float,
) -> np.ndarray:
    """Mix speech and noise at a specified SNR (dB).

    The noise is scaled so that ``speech_rms / noise_scaled_rms = 10^(snr_db/20)``.
    After mixing, peak normalisation is applied.

    Parameters
    ----------
    speech_wav : np.ndarray
        1-D float32 speech waveform.
    noise_wav : np.ndarray
        1-D float32 noise waveform.  If shorter than ``speech_wav``, it is
        loop-tiled; if longer, it is trimmed to match.
    snr_db : float
        Desired signal-to-noise ratio in dB.

    Returns
    -------
    np.ndarray
        Mixed float32 waveform (same length as ``speech_wav``), peak-normalised
        to 0.95.  Returns ``speech_wav`` unchanged if either signal is silent.
    """
    speech_wav = speech_wav.astype(np.float32)
    noise_wav = noise_wav.astype(np.float32)

    speech_rms = np.sqrt(np.mean(speech_wav ** 2))
    noise_rms = np.sqrt(np.mean(noise_wav ** 2))

    if speech_rms < 1e-8 or noise_rms < 1e-8:
        return speech_wav

    # Tile or trim noise to match speech length
    n_speech = len(speech_wav)
    n_noise = len(noise_wav)
    if n_noise < n_speech:
        repeats = int(np.ceil(n_speech / n_noise))
        noise_wav = np.tile(noise_wav, repeats)
    noise_wav = noise_wav[:n_speech]

    # Recompute noise RMS on the trimmed/tiled version
    noise_rms = np.sqrt(np.mean(noise_wav ** 2))
    if noise_rms < 1e-8:
        return speech_wav

    # Scale factor: speech_rms / (scale * noise_rms) = 10^(snr/20)
    #   => scale = speech_rms / (noise_rms * 10^(snr/20))
    scale = speech_rms / (noise_rms * (10 ** (snr_db / 20.0)))
    mixed = speech_wav + scale * noise_wav

    return peak_normalize(mixed)
