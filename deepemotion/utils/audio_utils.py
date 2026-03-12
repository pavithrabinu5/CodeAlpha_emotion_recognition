"""Audio helper utilities for DeepEmotion.

These functions keep I/O and simple audio processing in one place so
other modules can stay focused on modeling logic.
"""
from __future__ import annotations

import numpy as np
import librosa
import soundfile as sf


def load_audio(path: str, target_sr: int = 16000) -> tuple[np.ndarray, int]:
    """Load an audio file and resample to `target_sr`.

    Returns the audio signal (float32) and the sampling rate.
    """
    audio, sr = librosa.load(path, sr=target_sr, mono=True)
    return audio.astype(np.float32), sr


def trim_silence(audio: np.ndarray, top_db: float = 20.0) -> np.ndarray:
    """Remove leading and trailing silence using an energy-based threshold."""
    non_silent, _ = librosa.effects.trim(audio, top_db=top_db)
    return non_silent


def reduce_noise(audio: np.ndarray, sr: int, n_fft: int = 2048, hop_length: int = 512) -> np.ndarray:
    """Lightweight spectral gating for noise reduction.

    This avoids extra dependencies while providing a decent SNR bump.
    """
    stft = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
    magnitude, phase = np.abs(stft), np.angle(stft)
    noise_profile = librosa.decompose.nn_filter(magnitude,
                                               aggregate=np.median,
                                               metric='cosine',
                                               width=int(librosa.time_to_frames(0.5, sr=sr)))
    mask = np.maximum(magnitude - noise_profile, 0)
    cleaned = librosa.istft(mask * np.exp(1j * phase), hop_length=hop_length)
    return cleaned.astype(np.float32)


def normalize_signal(audio: np.ndarray) -> np.ndarray:
    """Normalize signal to the range [-1, 1]."""
    peak = np.max(np.abs(audio)) + 1e-9
    return (audio / peak).astype(np.float32)


def pad_or_truncate(audio: np.ndarray, sr: int, max_duration: float = 4.0) -> np.ndarray:
    """Ensure consistent length for batching."""
    target_len = int(sr * max_duration)
    if len(audio) < target_len:
        pad = target_len - len(audio)
        audio = np.pad(audio, (0, pad))
    else:
        audio = audio[:target_len]
    return audio.astype(np.float32)


def save_audio(path: str, audio: np.ndarray, sr: int) -> None:
    """Persist audio to disk."""
    sf.write(path, audio, sr)


def seed_everything(seed: int = 42) -> None:
    """Set all relevant seeds for reproducibility."""
    import random
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
