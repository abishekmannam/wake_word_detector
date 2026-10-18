#!/bin/bash
# Download Google Speech Commands v2 dataset
# Run this from the project root: bash data/download.sh

set -e

DATA_DIR="data"
ARCHIVE="$DATA_DIR/speech_commands_v0.02.tar.gz"
EXTRACT_DIR="$DATA_DIR/speech_commands_v0.02"
URL="http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz"

if [ -d "$EXTRACT_DIR" ]; then
    echo "✅ Dataset already exists at $EXTRACT_DIR"
    echo "   To re-download, delete that directory first."
    exit 0
fi

echo "📥 Downloading Google Speech Commands v2 (~2.3 GB)..."
mkdir -p "$EXTRACT_DIR"

if command -v wget &> /dev/null; then
    wget -q --show-progress -O "$ARCHIVE" "$URL"
elif command -v curl &> /dev/null; then
    curl -L -o "$ARCHIVE" "$URL"
else
    echo "❌ Neither wget nor curl found. Please install one."
    exit 1
fi

echo "📦 Extracting to $EXTRACT_DIR..."
tar -xzf "$ARCHIVE" -C "$EXTRACT_DIR"

echo "🗑️  Removing archive..."
rm "$ARCHIVE"

echo "✅ Done! Dataset extracted to $EXTRACT_DIR"
echo ""
echo "Contents:"
ls "$EXTRACT_DIR" | head -20
echo "..."
echo "Total word directories: $(ls -d $EXTRACT_DIR/*/ 2>/dev/null | wc -l)"
