"""
MFCC feature extraction pipeline.

Converts raw audio waveforms into MFCC feature matrices with optional
delta and delta-delta (acceleration) features.

Output shape: [n_channels, n_frames, n_mfcc]
    - n_channels = 1 (static only) or 3 (static + delta + delta-delta)
    - n_frames ≈ 98 for 1s audio at 16kHz with 10ms hop
    - n_mfcc = 13
"""

import torch
import torchaudio
import torchaudio.transforms as T

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import config


def build_mfcc_transform() -> T.MFCC:
    """Build the torchaudio MFCC transform with our config settings."""
    return T.MFCC(
        sample_rate=config.SAMPLE_RATE,
        n_mfcc=config.N_MFCC,
        melkwargs={
            "n_fft": config.N_FFT,
            "win_length": config.WIN_LENGTH,
            "hop_length": config.HOP_LENGTH,
            "n_mels": config.N_MELS,
        },
    )


# Module-level singleton so we don't recreate the transform every call
_mfcc_transform = None


def _get_mfcc_transform() -> T.MFCC:
    """Lazy-init the MFCC transform (avoids issues at import time)."""
    global _mfcc_transform
    if _mfcc_transform is None:
        _mfcc_transform = build_mfcc_transform()
    return _mfcc_transform


def compute_deltas(mfcc: torch.Tensor) -> torch.Tensor:
    """
    Compute delta and delta-delta features from static MFCCs.

    Args:
        mfcc: [n_mfcc, n_frames] static MFCC tensor

    Returns:
        [3, n_frames, n_mfcc] tensor stacking static, delta, delta-delta
    """
    # torchaudio.functional.compute_deltas expects [..., n_frames]
    delta = torchaudio.functional.compute_deltas(mfcc)
    delta_delta = torchaudio.functional.compute_deltas(delta)

    # Stack as channels: [3, n_mfcc, n_frames] -> transpose to [3, n_frames, n_mfcc]
    stacked = torch.stack([mfcc, delta, delta_delta], dim=0)  # [3, n_mfcc, n_frames]
    stacked = stacked.permute(0, 2, 1)  # [3, n_frames, n_mfcc]
    return stacked


def extract_mfcc(waveform: torch.Tensor) -> torch.Tensor:
    """
    Extract MFCC features from a raw audio waveform.

    Args:
        waveform: [1, n_samples] or [n_samples] raw audio tensor.
                  Will be padded/truncated to config.CLIP_LENGTH.

    Returns:
        Feature tensor:
            - If USE_DELTAS: [3, n_frames, n_mfcc]  (channels, time, freq)
            - Else:          [1, n_frames, n_mfcc]
    """
    # Ensure 2D: [1, n_samples]
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)

    # Pad or truncate to exactly CLIP_LENGTH samples
    waveform = pad_or_truncate(waveform, config.CLIP_LENGTH)

    # Compute MFCCs: [1, n_mfcc, n_frames]
    mfcc_transform = _get_mfcc_transform()
    mfcc = mfcc_transform(waveform)  # [1, n_mfcc, n_frames]
    mfcc = mfcc.squeeze(0)  # [n_mfcc, n_frames]

    if config.USE_DELTAS:
        features = compute_deltas(mfcc)  # [3, n_frames, n_mfcc]
    else:
        features = mfcc.permute(1, 0).unsqueeze(0)  # [1, n_frames, n_mfcc]

    return features


def pad_or_truncate(waveform: torch.Tensor, target_length: int) -> torch.Tensor:
    """
    Pad (with zeros) or truncate a waveform to a fixed length.

    Args:
        waveform: [1, n_samples] audio tensor
        target_length: desired number of samples

    Returns:
        [1, target_length] tensor
    """
    n_samples = waveform.shape[-1]

    if n_samples < target_length:
        # Pad with zeros on the right
        padding = target_length - n_samples
        waveform = torch.nn.functional.pad(waveform, (0, padding))
    elif n_samples > target_length:
        # Truncate
        waveform = waveform[..., :target_length]

    return waveform


def normalize_features(features: torch.Tensor) -> torch.Tensor:
    """
    Per-instance mean/variance normalization (cepstral mean normalization).

    Args:
        features: [C, T, F] feature tensor

    Returns:
        Normalized tensor with zero mean and unit variance per channel.
    """
    # Normalize each channel independently
    for c in range(features.shape[0]):
        channel = features[c]
        mean = channel.mean()
        std = channel.std()
        if std > 0:
            features[c] = (channel - mean) / std
        else:
            features[c] = channel - mean
    return features


if __name__ == "__main__":
    """Quick test: generate a random waveform and extract features."""
    print("Testing MFCC extraction pipeline...")
    print(f"  Sample rate: {config.SAMPLE_RATE} Hz")
    print(f"  Clip length: {config.CLIP_LENGTH} samples ({config.CLIP_DURATION_S}s)")
    print(f"  N_MFCC: {config.N_MFCC}")
    print(f"  Use deltas: {config.USE_DELTAS}")
    print()

    # Create a dummy 1-second waveform
    dummy_waveform = torch.randn(1, config.CLIP_LENGTH)
    features = extract_mfcc(dummy_waveform)
    features = normalize_features(features)

    print(f"  Input shape:  {dummy_waveform.shape}")
    print(f"  Output shape: {features.shape}")
    print(f"    → Channels: {features.shape[0]} ({'MFCC + Δ + ΔΔ' if config.USE_DELTAS else 'MFCC only'})")
    print(f"    → Frames:   {features.shape[1]}")
    print(f"    → Coeffs:   {features.shape[2]}")
    print(f"  Feature range: [{features.min():.3f}, {features.max():.3f}]")
    print(f"  Feature mean:  {features.mean():.3f}")
    print(f"  Feature std:   {features.std():.3f}")
    print("\n✅ MFCC pipeline works!")
