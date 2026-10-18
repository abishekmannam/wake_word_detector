"""
Audio data augmentation for the Wake-Word Detector.

Augmentations are applied in two stages:
    1. Waveform-level (before MFCC): time shift, noise injection, volume perturbation
    2. Feature-level (after MFCC): SpecAugment-style frequency/time masking

All augmentations are stochastic and controlled by config.AUG_PROBABILITY.
"""

import random
import torch
import torchaudio

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import config


class WaveformAugmentor:
    """Applies random augmentations to raw audio waveforms."""

    def __init__(self, noise_waveforms: list[torch.Tensor] | None = None):
        """
        Args:
            noise_waveforms: Optional list of background noise tensors.
                             Each should be [1, n_samples] or [n_samples].
                             If None, noise injection is skipped.
        """
        self.noise_waveforms = noise_waveforms or []
        self.p = config.AUG_PROBABILITY

    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Apply random augmentations to a waveform.

        Args:
            waveform: [1, n_samples] audio tensor

        Returns:
            Augmented [1, n_samples] tensor
        """
        if random.random() < self.p:
            waveform = self.time_shift(waveform)

        if random.random() < self.p:
            waveform = self.volume_perturbation(waveform)

        if random.random() < self.p and len(self.noise_waveforms) > 0:
            waveform = self.add_noise(waveform)

        return waveform

    @staticmethod
    def time_shift(waveform: torch.Tensor) -> torch.Tensor:
        """Randomly shift the waveform left or right by up to AUG_TIME_SHIFT_MS."""
        max_shift_samples = int(config.SAMPLE_RATE * config.AUG_TIME_SHIFT_MS / 1000)
        shift = random.randint(-max_shift_samples, max_shift_samples)
        return torch.roll(waveform, shifts=shift, dims=-1)

    @staticmethod
    def volume_perturbation(waveform: torch.Tensor) -> torch.Tensor:
        """Randomly scale the amplitude."""
        scale = random.uniform(config.AUG_VOLUME_MIN, config.AUG_VOLUME_MAX)
        return waveform * scale

    def add_noise(self, waveform: torch.Tensor) -> torch.Tensor:
        """Mix in a random background noise clip at a random SNR."""
        noise = random.choice(self.noise_waveforms)

        # Ensure noise is 2D
        if noise.dim() == 1:
            noise = noise.unsqueeze(0)

        # Extract a random segment of the same length
        n_samples = waveform.shape[-1]
        if noise.shape[-1] > n_samples:
            start = random.randint(0, noise.shape[-1] - n_samples)
            noise = noise[..., start:start + n_samples]
        elif noise.shape[-1] < n_samples:
            # Repeat noise if shorter
            repeats = (n_samples // noise.shape[-1]) + 1
            noise = noise.repeat(1, repeats)[..., :n_samples]

        # Calculate scaling factor for desired SNR
        snr_db = random.uniform(config.AUG_NOISE_SNR_MIN, config.AUG_NOISE_SNR_MAX)
        signal_power = waveform.pow(2).mean()
        noise_power = noise.pow(2).mean()

        if noise_power > 0:
            snr_linear = 10 ** (snr_db / 10)
            scale = torch.sqrt(signal_power / (snr_linear * noise_power))
            waveform = waveform + scale * noise

        return waveform


class SpecAugment:
    """
    Simplified SpecAugment: randomly mask frequency bands and time steps
    in the MFCC feature matrix.
    """

    def __init__(self):
        self.p = config.AUG_PROBABILITY
        self.freq_mask_param = config.AUG_SPEC_FREQ_MASK
        self.time_mask_param = config.AUG_SPEC_TIME_MASK

    def __call__(self, features: torch.Tensor) -> torch.Tensor:
        """
        Apply SpecAugment masking.

        Args:
            features: [C, T, F] feature tensor (channels, time, frequency)

        Returns:
            Masked feature tensor of same shape
        """
        if random.random() >= self.p:
            return features

        features = features.clone()
        _, n_time, n_freq = features.shape

        # Frequency masking: zero out random frequency bands
        if self.freq_mask_param > 0 and n_freq > 1:
            f = random.randint(0, min(self.freq_mask_param, n_freq - 1))
            f0 = random.randint(0, n_freq - f)
            features[:, :, f0:f0 + f] = 0

        # Time masking: zero out random time steps
        if self.time_mask_param > 0 and n_time > 1:
            t = random.randint(0, min(self.time_mask_param, n_time - 1))
            t0 = random.randint(0, n_time - t)
            features[:, t0:t0 + t, :] = 0

        return features


if __name__ == "__main__":
    """Quick test of augmentation pipeline."""
    print("Testing augmentation pipeline...")

    # Test waveform augmentation
    waveform = torch.randn(1, config.CLIP_LENGTH)
    augmentor = WaveformAugmentor()

    print(f"\nWaveform augmentation (no noise files loaded):")
    print(f"  Input shape:  {waveform.shape}")
    aug_waveform = augmentor(waveform)
    print(f"  Output shape: {aug_waveform.shape}")
    print(f"  Values changed: {not torch.allclose(waveform, aug_waveform)}")

    # Test SpecAugment
    features = torch.randn(3, 98, 13)
    spec_aug = SpecAugment()

    print(f"\nSpecAugment:")
    print(f"  Input shape:  {features.shape}")
    aug_features = spec_aug(features)
    print(f"  Output shape: {aug_features.shape}")
    n_zeros = (aug_features == 0).sum().item()
    print(f"  Zeros added:  {n_zeros}")

    print("\n✅ Augmentation pipeline works!")
