"""
Evaluation script for the Wake-Word Detector.

Generates:
    1. DET curve (False Accept Rate vs False Reject Rate)
    2. ROC curve (TPR vs FPR)
    3. Equal Error Rate (EER)
    4. Key operating points (FAR @ 1% FRR, FRR @ 1% FAR)
    5. Confusion matrix at EER threshold
    6. Training loss/AUC curves
"""

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from model.cnn import WakeWordCNN
from data.dataset import create_dataloaders


@torch.no_grad()
def collect_scores(
    model: WakeWordCNN,
    data_loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run the model on all data and collect scores + labels.

    Returns:
        (labels, scores) — both as 1D numpy arrays
    """
    model.eval()
    all_labels = []
    all_scores = []

    for features, labels in tqdm(data_loader, desc="Scoring"):
        features = features.to(device)
        logits = model(features)
        probs = torch.sigmoid(logits).cpu().numpy().flatten()

        all_labels.append(labels.numpy().flatten())
        all_scores.append(probs)

    return np.concatenate(all_labels), np.concatenate(all_scores)


def compute_det_curve(labels: np.ndarray, scores: np.ndarray, n_thresholds: int = config.NUM_THRESHOLDS):
    """
    Compute the Detection Error Tradeoff (DET) curve.

    Returns:
        thresholds: array of threshold values
        far: False Accept Rate at each threshold
        frr: False Reject Rate at each threshold
    """
    thresholds = np.linspace(0, 1, n_thresholds)

    pos_scores = scores[labels == 1]
    neg_scores = scores[labels == 0]

    n_pos = len(pos_scores)
    n_neg = len(neg_scores)

    far = np.zeros(n_thresholds)
    frr = np.zeros(n_thresholds)

    for i, tau in enumerate(thresholds):
        # False Accept: negative samples with score >= threshold
        far[i] = np.sum(neg_scores >= tau) / max(n_neg, 1)
        # False Reject: positive samples with score < threshold
        frr[i] = np.sum(pos_scores < tau) / max(n_pos, 1)

    return thresholds, far, frr


def find_eer(far: np.ndarray, frr: np.ndarray, thresholds: np.ndarray) -> tuple[float, float]:
    """
    Find the Equal Error Rate (EER) — the point where FAR ≈ FRR.

    Returns:
        (eer, eer_threshold)
    """
    # Find the crossing point
    diff = np.abs(far - frr)
    idx = np.argmin(diff)
    eer = (far[idx] + frr[idx]) / 2  # Average of the two at crossing
    return eer, thresholds[idx]


def find_operating_point(far, frr, thresholds, target_metric="far", target_value=0.01):
    """
    Find the operating point where target_metric = target_value.

    E.g., find FRR when FAR = 1%, or find FAR when FRR = 1%.
    """
    if target_metric == "far":
        # Find where FAR is closest to target_value
        idx = np.argmin(np.abs(far - target_value))
        return {"threshold": thresholds[idx], "FAR": far[idx], "FRR": frr[idx]}
    else:
        idx = np.argmin(np.abs(frr - target_value))
        return {"threshold": thresholds[idx], "FAR": far[idx], "FRR": frr[idx]}


def plot_det_curve(far, frr, eer, eer_threshold, save_path: Path):
    """Plot the DET curve with EER point marked."""
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(far * 100, frr * 100, "b-", linewidth=2, label="DET Curve")
    ax.plot([0, 100], [0, 100], "k--", alpha=0.3, label="Random")

    # Mark EER point
    ax.plot(eer * 100, eer * 100, "ro", markersize=10, label=f"EER = {eer*100:.2f}% (τ={eer_threshold:.3f})")

    ax.set_xlabel("False Accept Rate (%)", fontsize=12)
    ax.set_ylabel("False Reject Rate (%)", fontsize=12)
    ax.set_title("Detection Error Tradeoff (DET) Curve", fontsize=14)
    ax.legend(fontsize=11)
    ax.set_xlim([0, 50])
    ax.set_ylim([0, 50])
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Saved DET curve → {save_path}")


def plot_roc_curve(labels, scores, save_path: Path):
    """Plot the ROC curve with AUC."""
    fpr, tpr, _ = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr * 100, tpr * 100, "b-", linewidth=2, label=f"ROC (AUC = {roc_auc:.4f})")
    ax.plot([0, 100], [0, 100], "k--", alpha=0.3, label="Random")

    ax.set_xlabel("False Positive Rate (%)", fontsize=12)
    ax.set_ylabel("True Positive Rate (%)", fontsize=12)
    ax.set_title("Receiver Operating Characteristic (ROC) Curve", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Saved ROC curve → {save_path}")
    return roc_auc


def plot_confusion_matrix(labels, scores, threshold, save_path: Path):
    """Plot confusion matrix at a given threshold."""
    predictions = (scores >= threshold).astype(int)
    cm = confusion_matrix(labels, predictions)

    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(cm, display_labels=["Not Wake Word", config.WAKE_WORD.upper()])
    disp.plot(ax=ax, cmap="Blues", values_format="d")
    ax.set_title(f"Confusion Matrix (threshold = {threshold:.3f})", fontsize=13)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Saved confusion matrix → {save_path}")


def plot_training_history(history: dict, save_path: Path):
    """Plot training loss and validation AUC over epochs."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    epochs = range(1, len(history["train_loss"]) + 1)

    # Loss
    ax1.plot(epochs, history["train_loss"], "b-", label="Train Loss")
    ax1.plot(epochs, history["val_loss"], "r-", label="Val Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training & Validation Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # AUC
    ax2.plot(epochs, history["val_auc"], "g-", linewidth=2)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("AUC-ROC")
    ax2.set_title("Validation AUC-ROC")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Saved training history → {save_path}")


def evaluate_model(checkpoint_path: str = None, data_dir: Path = None):
    """
    Full evaluation pipeline.

    Loads the best model, runs it on the test set, and generates all plots/metrics.
    """
    print("=" * 60)
    print("WAKE-WORD DETECTOR — EVALUATION")
    print("=" * 60)

    # Load checkpoint
    if checkpoint_path is None:
        checkpoint_path = config.CHECKPOINTS_DIR / "best_model.pt"

    print(f"\nLoading model from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=config.DEVICE, weights_only=False)

    model = WakeWordCNN().to(config.DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"  Model from epoch {checkpoint['epoch']+1} (Val AUC: {checkpoint['val_auc']:.4f})")

    # Load test data
    print("\nPreparing test data...")
    _, _, test_loader, _ = create_dataloaders(data_dir)

    # Collect all scores
    print("\nRunning inference on test set...")
    labels, scores = collect_scores(model, test_loader, config.DEVICE)
    n_pos = (labels == 1).sum()
    n_neg = (labels == 0).sum()
    print(f"  Test samples: {len(labels)} ({n_pos} positive, {n_neg} negative)")

    # Compute DET curve
    print("\nComputing DET curve...")
    thresholds, far, frr = compute_det_curve(labels, scores)

    # Find EER
    eer, eer_threshold = find_eer(far, frr, thresholds)
    print(f"\n  ★ Equal Error Rate (EER): {eer*100:.2f}%")
    print(f"    at threshold τ = {eer_threshold:.3f}")

    # Operating points
    op_far1 = find_operating_point(far, frr, thresholds, "far", 0.01)
    op_frr1 = find_operating_point(far, frr, thresholds, "frr", 0.01)

    print(f"\n  @ FAR = 1%:  FRR = {op_far1['FRR']*100:.2f}%  (τ = {op_far1['threshold']:.3f})")
    print(f"  @ FRR = 1%:  FAR = {op_frr1['FAR']*100:.2f}%  (τ = {op_frr1['threshold']:.3f})")

    # Generate plots
    print("\nGenerating plots...")
    plot_det_curve(far, frr, eer, eer_threshold, config.PLOTS_DIR / "det_curve.png")
    roc_auc = plot_roc_curve(labels, scores, config.PLOTS_DIR / "roc_curve.png")
    plot_confusion_matrix(labels, scores, eer_threshold, config.PLOTS_DIR / "confusion_matrix.png")

    # Plot training history if available
    history_path = config.CHECKPOINTS_DIR / "training_history.pt"
    if history_path.exists():
        history = torch.load(history_path, weights_only=False)
        plot_training_history(history, config.PLOTS_DIR / "training_history.png")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Wake Word:      '{config.WAKE_WORD}'")
    print(f"  Model Params:   {model.count_parameters():,}")
    print(f"  Test Samples:   {len(labels)} ({n_pos} pos, {n_neg} neg)")
    print(f"  AUC-ROC:        {roc_auc:.4f}")
    print(f"  EER:            {eer*100:.2f}%")
    print(f"  FAR @ 1% FRR:   {op_frr1['FAR']*100:.2f}%")
    print(f"  FRR @ 1% FAR:   {op_far1['FRR']*100:.2f}%")
    print(f"  Plots saved to: {config.PLOTS_DIR}")
    print("=" * 60)

    return {
        "auc_roc": roc_auc,
        "eer": eer,
        "eer_threshold": eer_threshold,
        "far_at_1pct_frr": op_frr1["FAR"],
        "frr_at_1pct_far": op_far1["FRR"],
    }


if __name__ == "__main__":
    evaluate_model()
