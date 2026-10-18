"""
PyTorch Dataset for the Wake-Word Detector.

Loads audio files, applies augmentations (in training mode), extracts MFCC
features, and returns (features, label) pairs ready for the model.
"""

import random
from pathlib import Path
from typing import List, Tuple

import numpy as np
import soundfile as sf
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from features.mfcc import extract_mfcc, normalize_features, pad_or_truncate
from augment.audio_augment import WaveformAugmentor, SpecAugment
from data.splits import create_splits


class WakeWordDataset(Dataset):
    """
    PyTorch Dataset for wake-word detection.

    Each item is a (features, label) tuple:
        - features: [C, T, F] MFCC tensor (C=3 with deltas, T≈98, F=13)
        - label: 0 or 1 (float tensor for BCE loss)
    """

    def __init__(
        self,
        file_label_pairs: List[Tuple[str, int]],
        train: bool = False,
        noise_waveforms: list[torch.Tensor] | None = None,
    ):
        """
        Args:
            file_label_pairs: List of (wav_path, label) tuples from splits.py.
            train: If True, apply data augmentation.
            noise_waveforms: Background noise tensors for augmentation.
        """
        self.file_label_pairs = file_label_pairs
        self.train = train

        # Augmentation (only for training)
        if train:
            self.waveform_augmentor = WaveformAugmentor(noise_waveforms)
            self.spec_augment = SpecAugment()
        else:
            self.waveform_augmentor = None
            self.spec_augment = None

    def __len__(self) -> int:
        return len(self.file_label_pairs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        wav_path, label = self.file_label_pairs[idx]

        # Load audio using soundfile (handles WAV natively, no FFmpeg needed)
        audio_np, sample_rate = sf.read(wav_path, dtype="float32")

        # Convert to torch tensor: [1, n_samples]
        if audio_np.ndim > 1:
            # Stereo → mono by averaging channels
            audio_np = audio_np.mean(axis=1)
        waveform = torch.from_numpy(audio_np).unsqueeze(0)  # [1, n_samples]

        # Resample if needed (should already be 16kHz but just in case)
        if sample_rate != config.SAMPLE_RATE:
            resampler = torchaudio.transforms.Resample(sample_rate, config.SAMPLE_RATE)
            waveform = resampler(waveform)

        # Pad/truncate to fixed length
        waveform = pad_or_truncate(waveform, config.CLIP_LENGTH)

        # Apply waveform augmentations (training only)
        if self.train and self.waveform_augmentor is not None:
            waveform = self.waveform_augmentor(waveform)

        # Extract MFCC features
        features = extract_mfcc(waveform)  # [C, T, F]

        # Apply SpecAugment (training only)
        if self.train and self.spec_augment is not None:
            features = self.spec_augment(features)

        # Normalize
        features = normalize_features(features)

        label_tensor = torch.tensor(label, dtype=torch.float32)

        return features, label_tensor


def load_background_noise(data_dir: Path = None) -> list[torch.Tensor]:
    """
    Load background noise waveforms from the _background_noise_ directory.

    Returns:
        List of long noise waveform tensors (each is [1, n_samples]).
    """
    if data_dir is None:
        data_dir = config.DATA_DIR

    noise_dir = data_dir / config.BACKGROUND_NOISE_DIR
    if not noise_dir.exists():
        print(f"  ⚠ Background noise directory not found: {noise_dir}")
        return []

    noise_waveforms = []
    for wav_path in sorted(noise_dir.glob("*.wav")):
        audio_np, sr = sf.read(str(wav_path), dtype="float32")
        if audio_np.ndim > 1:
            audio_np = audio_np.mean(axis=1)
        waveform = torch.from_numpy(audio_np).unsqueeze(0)
        if sr != config.SAMPLE_RATE:
            resampler = torchaudio.transforms.Resample(sr, config.SAMPLE_RATE)
            waveform = resampler(waveform)
        noise_waveforms.append(waveform)
        print(f"  Loaded noise: {wav_path.name} ({waveform.shape[-1] / config.SAMPLE_RATE:.1f}s)")

    return noise_waveforms


def get_class_weight(file_label_pairs: List[Tuple[str, int]]) -> torch.Tensor:
    """
    Compute the positive class weight for BCEWithLogitsLoss.

    weight = n_negative / n_positive

    This compensates for the heavy class imbalance where negatives >> positives.
    """
    n_pos = sum(1 for _, label in file_label_pairs if label == 1)
    n_neg = sum(1 for _, label in file_label_pairs if label == 0)

    if n_pos == 0:
        raise ValueError("No positive samples found! Check your WAKE_WORD setting.")

    weight = n_neg / n_pos
    print(f"  Class weight: {weight:.1f} (n_pos={n_pos}, n_neg={n_neg})")
    return torch.tensor(weight, dtype=torch.float32)


def create_dataloaders(
    data_dir: Path = None,
) -> Tuple[DataLoader, DataLoader, DataLoader, torch.Tensor]:
    """
    Create train, validation, and test DataLoaders.

    Returns:
        (train_loader, val_loader, test_loader, pos_weight)
    """
    print("Creating dataset splits...")
    splits = create_splits(data_dir)

    print(f"\nLoading background noise for augmentation...")
    noise_waveforms = load_background_noise(data_dir)

    print(f"\nBuilding datasets...")
    train_dataset = WakeWordDataset(splits["train"], train=True, noise_waveforms=noise_waveforms)
    val_dataset = WakeWordDataset(splits["val"], train=False)
    test_dataset = WakeWordDataset(splits["test"], train=False)

    print(f"  Train: {len(train_dataset)} samples")
    print(f"  Val:   {len(val_dataset)} samples")
    print(f"  Test:  {len(test_dataset)} samples")

    # Compute class weight from training set
    print(f"\nComputing class weights...")
    pos_weight = get_class_weight(splits["train"])

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader, pos_weight


if __name__ == "__main__":
    """Quick test: check that the dataset loads and returns correct shapes."""
    print("=" * 60)
    print("Testing WakeWordDataset")
    print("=" * 60)

    if not config.DATA_DIR.exists():
        print(f"\n❌ Dataset not found at {config.DATA_DIR}")
        print("   Run: bash data/download.sh")
        print("\n   Testing with dummy data instead...")

        # Create a few dummy wav files for testing
        import tempfile, os
        import soundfile as sf
        import numpy as np
        tmp_dir = Path(tempfile.mkdtemp())
        dummy_pairs = []
        for i in range(5):
            label = 1 if i < 2 else 0
            wav_path = tmp_dir / f"dummy_{i}.wav"
            waveform_np = np.random.randn(config.CLIP_LENGTH).astype(np.float32) * 0.1
            sf.write(str(wav_path), waveform_np, config.SAMPLE_RATE)
            dummy_pairs.append((str(wav_path), label))

        dataset = WakeWordDataset(dummy_pairs, train=True)
        print(f"\n  Dataset length: {len(dataset)}")

        features, label = dataset[0]
        print(f"  Feature shape:  {features.shape}")
        print(f"  Label:          {label}")
        print(f"  Feature dtype:  {features.dtype}")
        print(f"  Label dtype:    {label.dtype}")

        # Test batching
        loader = DataLoader(dataset, batch_size=3)
        batch_features, batch_labels = next(iter(loader))
        print(f"\n  Batch features: {batch_features.shape}")
        print(f"  Batch labels:   {batch_labels.shape}")

        # Cleanup
        for f in tmp_dir.glob("*"):
            f.unlink()
        tmp_dir.rmdir()

        print("\n✅ Dataset pipeline works (dummy data)!")
    else:
        train_loader, val_loader, test_loader, pos_weight = create_dataloaders()
        batch_features, batch_labels = next(iter(train_loader))
        print(f"\n  Batch features: {batch_features.shape}")
        print(f"  Batch labels:   {batch_labels.shape}")
        print(f"  Pos weight:     {pos_weight:.1f}")
        print("\n✅ Dataset pipeline works!")
