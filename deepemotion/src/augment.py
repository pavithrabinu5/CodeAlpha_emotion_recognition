"""Audio data augmentation utilities."""
from __future__ import annotations

import numpy as np
import librosa


def time_stretch(audio: np.ndarray, rate: float = 1.1) -> np.ndarray:
    return librosa.effects.time_stretch(audio, rate=rate).astype(np.float32)


def pitch_shift(audio: np.ndarray, sr: int, n_steps: float = 2.0) -> np.ndarray:
    return librosa.effects.pitch_shift(audio, sr=sr, n_steps=n_steps).astype(np.float32)


def add_background_noise(audio: np.ndarray, noise_factor: float = 0.005) -> np.ndarray:
    noise = np.random.randn(len(audio)).astype(np.float32)
    return (audio + noise_factor * noise).astype(np.float32)


def random_shift(audio: np.ndarray, shift_max: int) -> np.ndarray:
    shift = np.random.randint(-shift_max, shift_max)
    return np.roll(audio, shift).astype(np.float32)


def spec_augment(mel_spectrogram: np.ndarray, num_masks: int = 2, freq_mask: int = 12, time_mask: int = 18) -> np.ndarray:
    """Apply SpecAugment-style masking on a Mel spectrogram (freq x time)."""
    augmented = mel_spectrogram.copy()
    num_mel_channels, num_frames = augmented.shape

    for _ in range(num_masks):
        f = np.random.randint(0, freq_mask)
        f0 = np.random.randint(0, max(1, num_mel_channels - f))
        augmented[f0:f0 + f, :] = 0

        t = np.random.randint(0, time_mask)
        t0 = np.random.randint(0, max(1, num_frames - t))
        augmented[:, t0:t0 + t] = 0

    return augmented.astype(np.float32)


def apply_augmentations(audio: np.ndarray, sr: int) -> list[np.ndarray]:
    """Generate a small batch of augmented variants."""
    variants = [audio]
    variants.append(time_stretch(audio, rate=np.random.uniform(0.9, 1.1)))
    variants.append(pitch_shift(audio, sr, n_steps=np.random.uniform(-2, 2)))
    variants.append(add_background_noise(audio, noise_factor=np.random.uniform(0.002, 0.01)))
    variants.append(random_shift(audio, shift_max=int(0.1 * len(audio))))
    return variants
