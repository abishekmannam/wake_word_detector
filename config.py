"""
Central configuration for the Wake-Word Detector project.
All hyperparameters and paths live here.
"""

import os
from pathlib import Path

# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.resolve()
DATA_DIR = PROJECT_ROOT / "data" / "speech_commands_v0.02"
FEATURES_DIR = PROJECT_ROOT / "features" / "cached"
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
PLOTS_DIR = PROJECT_ROOT / "plots"

# Create directories if they don't exist
for d in [FEATURES_DIR, CHECKPOINTS_DIR, PLOTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ──────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────
DATASET_URL = "http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz"
WAKE_WORD = "sheila"  # The keyword we'll detect (one of the 35 words in the dataset)
# All 35 words: yes, no, up, down, left, right, on, off, stop, go,
#   zero, one, two, three, four, five, six, seven, eight, nine,
#   bed, bird, cat, dog, happy, house, marvin, sheila, tree, wow,
#   backward, follow, forward, learn, visual
# "sheila" is a good stand-in for a multi-syllable wake word like "Hey Siri"

BACKGROUND_NOISE_DIR = "_background_noise_"
SILENCE_RATIO = 0.1  # fraction of negatives that are pure silence/noise

# ──────────────────────────────────────────────
# Audio
# ──────────────────────────────────────────────
SAMPLE_RATE = 16_000  # Hz
CLIP_DURATION_S = 1.0  # seconds
CLIP_LENGTH = int(SAMPLE_RATE * CLIP_DURATION_S)  # 16,000 samples

# ──────────────────────────────────────────────
# MFCC Features
# ──────────────────────────────────────────────
N_MFCC = 13           # number of MFCC coefficients
N_MELS = 40           # number of Mel filterbank channels
N_FFT = 512           # FFT window size
WIN_LENGTH = 400      # window length in samples (25ms at 16kHz)
HOP_LENGTH = 160      # hop length in samples (10ms at 16kHz)
USE_DELTAS = True     # include delta and delta-delta features
N_CHANNELS = 3 if USE_DELTAS else 1  # MFCC + delta + delta-delta

# ──────────────────────────────────────────────
# Data Augmentation
# ──────────────────────────────────────────────
AUG_TIME_SHIFT_MS = 100       # max time shift in milliseconds
AUG_NOISE_SNR_MIN = 5         # minimum SNR in dB for noise mixing
AUG_NOISE_SNR_MAX = 15        # maximum SNR in dB for noise mixing
AUG_VOLUME_MIN = 0.7          # minimum volume scale factor
AUG_VOLUME_MAX = 1.3          # maximum volume scale factor
AUG_SPEC_FREQ_MASK = 2        # max number of frequency bands to mask
AUG_SPEC_TIME_MASK = 10       # max number of time frames to mask
AUG_PROBABILITY = 0.8         # probability of applying each augmentation

# ──────────────────────────────────────────────
# Model (Small CNN)
# ──────────────────────────────────────────────
CNN_CHANNELS = [32, 64, 64]   # output channels for each conv layer
CNN_KERNEL_SIZE = 3            # kernel size for all conv layers
CNN_DROPOUT = 0.3              # dropout before final linear layer

# ──────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
EPOCHS = 50
EARLY_STOPPING_PATIENCE = 10
LR_SCHEDULER_PATIENCE = 5
LR_SCHEDULER_FACTOR = 0.5
NUM_WORKERS = 4  # DataLoader workers (set to 0 on Windows if issues)

# ──────────────────────────────────────────────
# Evaluation
# ──────────────────────────────────────────────
NUM_THRESHOLDS = 1000  # number of thresholds to sweep for DET curve

# ──────────────────────────────────────────────
# Device
# ──────────────────────────────────────────────
import torch
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
