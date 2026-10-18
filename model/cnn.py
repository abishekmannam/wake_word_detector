"""
Small CNN model for wake-word detection.

Architecture: 3 conv layers with BatchNorm + ReLU + MaxPool,
followed by Global Average Pooling and a single linear output.

~57K parameters — small enough for always-on inference.
"""

import torch
import torch.nn as nn

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import config


class WakeWordCNN(nn.Module):
    """
    Small CNN for binary keyword spotting.

    Input:  [batch, channels, time, freq]  e.g. [B, 3, 101, 13]
    Output: [batch, 1]                     logit (pre-sigmoid)

    The model outputs raw logits. Apply sigmoid for probabilities.
    Use BCEWithLogitsLoss for training (numerically stable).
    """

    def __init__(
        self,
        in_channels: int = config.N_CHANNELS,
        conv_channels: list[int] = None,
        kernel_size: int = config.CNN_KERNEL_SIZE,
        dropout: float = config.CNN_DROPOUT,
    ):
        super().__init__()

        if conv_channels is None:
            conv_channels = config.CNN_CHANNELS  # [32, 64, 64]

        # Build conv blocks
        layers = []
        ch_in = in_channels
        for ch_out in conv_channels:
            layers.extend([
                nn.Conv2d(ch_in, ch_out, kernel_size=kernel_size, stride=1, padding=1),
                nn.BatchNorm2d(ch_out),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
            ])
            ch_in = ch_out

        self.conv_blocks = nn.Sequential(*layers)

        # Global Average Pooling → flatten to [batch, last_ch]
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(conv_channels[-1], 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, channels, time, freq] MFCC feature tensor

        Returns:
            [batch, 1] raw logits (apply sigmoid for probabilities)
        """
        x = self.conv_blocks(x)           # [B, C, T', F']
        x = self.global_pool(x)           # [B, C, 1, 1]
        x = x.view(x.size(0), -1)        # [B, C]
        x = self.classifier(x)           # [B, 1]
        return x

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get probability scores (for inference / evaluation).

        Returns:
            [batch, 1] probabilities in [0, 1]
        """
        logits = self.forward(x)
        return torch.sigmoid(logits)

    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_model() -> WakeWordCNN:
    """Create and return the wake-word CNN model."""
    model = WakeWordCNN()
    return model


if __name__ == "__main__":
    """Test the model with a dummy input."""
    print("Testing WakeWordCNN...")
    print()

    model = create_model()

    # Print architecture
    print(model)
    print()

    # Count parameters
    n_params = model.count_parameters()
    print(f"Total trainable parameters: {n_params:,}")
    print(f"Model size (float32): {n_params * 4 / 1024:.1f} KB")
    print()

    # Forward pass with dummy data
    # Shape: [batch=4, channels=3, time=101, freq=13]
    dummy_input = torch.randn(4, config.N_CHANNELS, 101, config.N_MFCC)
    print(f"Input shape:  {dummy_input.shape}")

    logits = model(dummy_input)
    print(f"Logit shape:  {logits.shape}")
    print(f"Logit values: {logits.detach().squeeze().tolist()}")

    probs = model.predict_proba(dummy_input)
    print(f"Prob shape:   {probs.shape}")
    print(f"Prob values:  {[f'{p:.4f}' for p in probs.detach().squeeze().tolist()]}")

    print("\n✅ Model works!")
