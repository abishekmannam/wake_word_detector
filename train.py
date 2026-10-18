"""
Training loop for the Wake-Word Detector.

Handles:
    - Training with weighted BCE loss (class imbalance)
    - Validation with AUC-ROC tracking
    - Learning rate scheduling
    - Early stopping
    - Checkpoint saving
    - Training history logging
"""

import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from model.cnn import create_model
from data.dataset import create_dataloaders


class EarlyStopping:
    """Stop training when validation loss stops improving."""

    def __init__(self, patience: int = config.EARLY_STOPPING_PATIENCE, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float("inf")
        self.should_stop = False

    def check(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """Train for one epoch. Returns average loss."""
    model.train()
    total_loss = 0.0
    n_batches = 0

    for features, labels in tqdm(train_loader, desc="  Train", leave=False):
        features = features.to(device)
        labels = labels.to(device).unsqueeze(1)  # [B] → [B, 1]

        optimizer.zero_grad()
        logits = model(features)  # [B, 1]
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Evaluate on validation set. Returns (avg_loss, auc_roc)."""
    model.eval()
    total_loss = 0.0
    n_batches = 0
    all_labels = []
    all_probs = []

    for features, labels in tqdm(val_loader, desc="  Val  ", leave=False):
        features = features.to(device)
        labels = labels.to(device).unsqueeze(1)

        logits = model(features)
        loss = criterion(logits, labels)

        total_loss += loss.item()
        n_batches += 1

        probs = torch.sigmoid(logits).cpu()
        all_labels.append(labels.cpu())
        all_probs.append(probs)

    avg_loss = total_loss / max(n_batches, 1)

    # Compute AUC-ROC
    all_labels = torch.cat(all_labels).numpy().flatten()
    all_probs = torch.cat(all_probs).numpy().flatten()

    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc = 0.0  # Can happen if only one class in batch

    return avg_loss, auc


def train(
    data_dir: Path = None,
    resume_checkpoint: str = None,
) -> dict:
    """
    Full training pipeline.

    Returns:
        Dictionary with training history (losses, AUCs per epoch).
    """
    print("=" * 60)
    print("WAKE-WORD DETECTOR — TRAINING")
    print("=" * 60)
    print(f"  Wake word:    '{config.WAKE_WORD}'")
    print(f"  Device:       {config.DEVICE}")
    print(f"  Batch size:   {config.BATCH_SIZE}")
    print(f"  LR:           {config.LEARNING_RATE}")
    print(f"  Max epochs:   {config.EPOCHS}")
    print()

    # Create data loaders
    train_loader, val_loader, test_loader, pos_weight = create_dataloaders(data_dir)
    print()

    # Create model
    model = create_model().to(config.DEVICE)
    n_params = model.count_parameters()
    print(f"Model parameters: {n_params:,} ({n_params * 4 / 1024:.1f} KB)")
    print()

    # Loss function with class weight
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(config.DEVICE))

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    # LR scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=config.LR_SCHEDULER_FACTOR,
        patience=config.LR_SCHEDULER_PATIENCE,
    )

    # Early stopping
    early_stopping = EarlyStopping()

    # Resume from checkpoint if provided
    start_epoch = 0
    if resume_checkpoint:
        checkpoint = torch.load(resume_checkpoint, map_location=config.DEVICE)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        print(f"Resumed from epoch {start_epoch}")

    # Training history
    history = {
        "train_loss": [],
        "val_loss": [],
        "val_auc": [],
        "lr": [],
    }

    best_val_auc = 0.0
    best_epoch = 0

    print("Starting training...")
    print("-" * 60)

    for epoch in range(start_epoch, config.EPOCHS):
        t_start = time.time()

        # Train
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, config.DEVICE)

        # Evaluate
        val_loss, val_auc = evaluate(model, val_loader, criterion, config.DEVICE)

        # Get current LR
        current_lr = optimizer.param_groups[0]["lr"]

        # Update scheduler
        scheduler.step(val_loss)

        # Log
        t_elapsed = time.time() - t_start
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_auc"].append(val_auc)
        history["lr"].append(current_lr)

        # Print epoch summary
        improved = ""
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_epoch = epoch

            # Save best model
            checkpoint_path = config.CHECKPOINTS_DIR / "best_model.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_auc": val_auc,
                "val_loss": val_loss,
                "config": {
                    "wake_word": config.WAKE_WORD,
                    "n_mfcc": config.N_MFCC,
                    "n_channels": config.N_CHANNELS,
                },
            }, checkpoint_path)
            improved = " ★ saved"

        print(
            f"  Epoch {epoch+1:3d}/{config.EPOCHS} │ "
            f"Train Loss: {train_loss:.4f} │ "
            f"Val Loss: {val_loss:.4f} │ "
            f"Val AUC: {val_auc:.4f} │ "
            f"LR: {current_lr:.1e} │ "
            f"{t_elapsed:.1f}s{improved}"
        )

        # Early stopping check
        if early_stopping.check(val_loss):
            print(f"\n  Early stopping at epoch {epoch+1} (patience={config.EARLY_STOPPING_PATIENCE})")
            break

    print("-" * 60)
    print(f"\nBest model: epoch {best_epoch+1} with Val AUC = {best_val_auc:.4f}")
    print(f"Saved to: {config.CHECKPOINTS_DIR / 'best_model.pt'}")

    # Save training history
    history_path = config.CHECKPOINTS_DIR / "training_history.pt"
    torch.save(history, history_path)
    print(f"History saved to: {history_path}")

    return history


if __name__ == "__main__":
    history = train()
