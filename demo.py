"""
Live microphone demo for the Wake-Word Detector.

Listens to your microphone in real-time and detects the wake word "sheila".
Uses a sliding window approach — continuously captures audio and runs
inference every 0.5 seconds on the most recent 1-second window.

Usage:
    python demo.py                     # Use default threshold (EER)
    python demo.py --threshold 0.6     # Custom threshold (higher = fewer false alarms)

Press Ctrl+C to stop.
"""

import argparse
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import sounddevice as sd
import torch

sys.path.insert(0, str(Path(__file__).parent))
import config
from features.mfcc import extract_mfcc, normalize_features
from model.cnn import WakeWordCNN


# ─── ANSI colors for terminal output ───
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
BOLD = "\033[1m"
RESET = "\033[0m"
DIM = "\033[2m"


def make_bar(score: float, width: int = 40) -> str:
    """Create a visual confidence bar."""
    filled = int(score * width)
    bar = "█" * filled + "░" * (width - filled)
    return bar


def load_model(checkpoint_path: str = None) -> WakeWordCNN:
    """Load the trained model from checkpoint."""
    if checkpoint_path is None:
        checkpoint_path = config.CHECKPOINTS_DIR / "best_model.pt"

    print(f"  Loading model from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    model = WakeWordCNN()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    epoch = checkpoint["epoch"] + 1
    val_auc = checkpoint["val_auc"]
    print(f"  Model from epoch {epoch} (Val AUC: {val_auc:.4f})")
    print(f"  Parameters: {model.count_parameters():,}")

    return model


def run_demo(threshold: float = 0.475, checkpoint_path: str = None):
    """
    Run the live microphone demo.

    Continuously records from the mic and runs wake-word detection
    on a sliding 1-second window.
    """
    print()
    print(f"{BOLD}{'=' * 60}{RESET}")
    print(f"{BOLD}  🎙️  WAKE-WORD DETECTOR — LIVE DEMO{RESET}")
    print(f"{BOLD}{'=' * 60}{RESET}")
    print()
    print(f"  Wake word:   {BOLD}'{config.WAKE_WORD}'{RESET}")
    print(f"  Threshold:   {threshold:.3f}")
    print()

    # Load model
    model = load_model(checkpoint_path)
    print()

    # Audio settings
    sample_rate = config.SAMPLE_RATE
    clip_length = config.CLIP_LENGTH  # 16000 samples = 1 second
    hop_samples = sample_rate // 2     # Run inference every 0.5 seconds
    block_size = 1600                  # 100ms blocks from mic

    # Ring buffer to store the last 1 second of audio
    buffer = deque(maxlen=clip_length)
    # Pre-fill with silence
    for _ in range(clip_length):
        buffer.append(0.0)

    # Track detections for cooldown
    last_detection_time = 0
    cooldown_seconds = 1.5  # Don't re-trigger for 1.5s after detection
    samples_since_inference = 0

    print(f"  {CYAN}Listening... Say '{config.WAKE_WORD}' into your microphone.{RESET}")
    print(f"  {DIM}Press Ctrl+C to stop.{RESET}")
    print()
    print(f"  {'Confidence':^50}  Result")
    print(f"  {'─' * 50}  ──────")

    def audio_callback(indata, frames, time_info, status):
        """Called by sounddevice for each audio block."""
        nonlocal samples_since_inference

        if status:
            pass  # Skip status messages (e.g., overflow)

        # Add samples to ring buffer
        audio = indata[:, 0]  # Mono (first channel)
        for sample in audio:
            buffer.append(sample)
        samples_since_inference += frames

    # Start mic stream
    stream = sd.InputStream(
        samplerate=sample_rate,
        channels=1,
        dtype="float32",
        blocksize=block_size,
        callback=audio_callback,
    )

    try:
        with stream:
            while True:
                # Wait for enough new audio before running inference
                if samples_since_inference < hop_samples:
                    time.sleep(0.05)
                    continue

                samples_since_inference = 0

                # Convert buffer to tensor
                waveform = torch.tensor(list(buffer), dtype=torch.float32).unsqueeze(0)

                # Extract features
                features = extract_mfcc(waveform)
                features = normalize_features(features)
                features = features.unsqueeze(0)  # Add batch dim: [1, C, T, F]

                # Run inference
                with torch.no_grad():
                    logits = model(features)
                    score = torch.sigmoid(logits).item()

                # Check detection with cooldown
                now = time.time()
                detected = score >= threshold and (now - last_detection_time) > cooldown_seconds

                # Display
                bar = make_bar(score)
                if detected:
                    last_detection_time = now
                    print(f"\r  {GREEN}{bar}{RESET} {score:.3f}  {GREEN}{BOLD}✅ DETECTED!{RESET}   ", flush=True)
                    print()  # New line after detection so it stays visible
                elif score >= threshold * 0.7:  # Getting warm
                    print(f"\r  {YELLOW}{bar}{RESET} {score:.3f}  {YELLOW}hmm...{RESET}      ", end="", flush=True)
                else:
                    print(f"\r  {DIM}{bar}{RESET} {score:.3f}  {DIM}·{RESET}             ", end="", flush=True)

    except KeyboardInterrupt:
        print(f"\n\n  {CYAN}Demo stopped.{RESET}")
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Wake-Word Detector Live Demo")
    parser.add_argument(
        "--threshold", "-t",
        type=float,
        default=0.475,
        help="Detection threshold (default: 0.475 = EER point). Higher = fewer false alarms.",
    )
    parser.add_argument(
        "--checkpoint", "-c",
        type=str,
        default=None,
        help="Path to model checkpoint (default: checkpoints/best_model.pt)",
    )
    args = parser.parse_args()
    run_demo(threshold=args.threshold, checkpoint_path=args.checkpoint)
