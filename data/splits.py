"""
Dataset split logic for the Wake-Word Detector.

Uses the official validation_list.txt and testing_list.txt from Google Speech
Commands v2 to create reproducible train/val/test splits.

Each sample is labeled:
    1 = wake word (e.g., "sheila")
    0 = not the wake word (all other words + silence)
"""

import os
import random
from pathlib import Path
from typing import Dict, List, Tuple

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
import config


def load_official_lists(data_dir: Path) -> Tuple[set, set]:
    """Load the official validation and test file lists shipped with the dataset."""
    val_list_path = data_dir / "validation_list.txt"
    test_list_path = data_dir / "testing_list.txt"

    if not val_list_path.exists() or not test_list_path.exists():
        raise FileNotFoundError(
            f"Expected validation_list.txt and testing_list.txt in {data_dir}. "
            "Make sure you've downloaded and extracted the dataset."
        )

    with open(val_list_path) as f:
        val_files = {line.strip() for line in f if line.strip()}

    with open(test_list_path) as f:
        test_files = {line.strip() for line in f if line.strip()}

    return val_files, test_files


def get_all_audio_files(data_dir: Path) -> List[Tuple[str, str]]:
    """
    Scan the dataset directory for all .wav files.

    Returns:
        List of (relative_path, word) tuples.
        relative_path is like "sheila/abc123.wav"
        word is the directory name (the spoken word).
    """
    audio_files = []
    for word_dir in sorted(data_dir.iterdir()):
        # Skip non-directories and the background noise dir
        if not word_dir.is_dir():
            continue
        if word_dir.name == config.BACKGROUND_NOISE_DIR:
            continue
        if word_dir.name.startswith("."):
            continue

        word = word_dir.name
        for wav_file in sorted(word_dir.glob("*.wav")):
            rel_path = f"{word}/{wav_file.name}"
            audio_files.append((rel_path, word))

    return audio_files


def create_splits(data_dir: Path = None) -> Dict[str, List[Tuple[str, int]]]:
    """
    Create train/val/test splits with binary labels.

    Returns:
        Dictionary with keys "train", "val", "test".
        Each value is a list of (absolute_wav_path, label) tuples.
        label: 1 = wake word, 0 = not wake word.
    """
    if data_dir is None:
        data_dir = config.DATA_DIR

    # Load official split lists
    val_files, test_files = load_official_lists(data_dir)

    # Get all audio files
    all_files = get_all_audio_files(data_dir)

    splits = {"train": [], "val": [], "test": []}

    for rel_path, word in all_files:
        # Binary label
        label = 1 if word == config.WAKE_WORD else 0
        abs_path = str(data_dir / rel_path)

        # Assign to split based on official lists
        if rel_path in test_files:
            splits["test"].append((abs_path, label))
        elif rel_path in val_files:
            splits["val"].append((abs_path, label))
        else:
            splits["train"].append((abs_path, label))

    return splits


def print_split_stats(splits: Dict[str, List[Tuple[str, int]]]) -> None:
    """Print statistics about the dataset splits."""
    print("=" * 60)
    print(f"Wake Word: '{config.WAKE_WORD}'")
    print("=" * 60)

    total_pos, total_neg, total = 0, 0, 0
    for split_name in ["train", "val", "test"]:
        data = splits[split_name]
        n_pos = sum(1 for _, label in data if label == 1)
        n_neg = sum(1 for _, label in data if label == 0)
        n_total = len(data)

        total_pos += n_pos
        total_neg += n_neg
        total += n_total

        ratio = n_neg / n_pos if n_pos > 0 else float("inf")
        print(f"\n  {split_name.upper():>5}:  {n_total:>6} samples")
        print(f"         Positive (wake word):  {n_pos:>5}  ({100*n_pos/n_total:.1f}%)")
        print(f"         Negative (other):      {n_neg:>5}  ({100*n_neg/n_total:.1f}%)")
        print(f"         Imbalance ratio:        1:{ratio:.1f}")

    print(f"\n  TOTAL: {total:>6} samples  "
          f"({total_pos} pos, {total_neg} neg, ratio 1:{total_neg/total_pos:.1f})")
    print("=" * 60)


if __name__ == "__main__":
    print(f"Data directory: {config.DATA_DIR}")
    print(f"Looking for dataset...")

    if not config.DATA_DIR.exists():
        print(f"\n❌ Dataset not found at {config.DATA_DIR}")
        print("   Run: bash data/download.sh")
    else:
        splits = create_splits()
        print_split_stats(splits)
