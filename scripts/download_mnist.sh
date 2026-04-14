#!/usr/bin/env bash
# Download and decompress the MNIST dataset (IDX format).
#
# Usage:
#   ./scripts/download_mnist.sh [DATA_DIR]
#
# Default data directory: data/mnist/

set -euo pipefail

DATA_DIR="${1:-data/mnist}"
BASE_URL="https://storage.googleapis.com/cvdf-datasets/mnist"

FILES=(
    "train-images-idx3-ubyte"
    "train-labels-idx1-ubyte"
    "t10k-images-idx3-ubyte"
    "t10k-labels-idx1-ubyte"
)

mkdir -p "$DATA_DIR"

# Detect download tool
if command -v curl &>/dev/null; then
    DOWNLOAD="curl -fSL --retry 3 -o"
elif command -v wget &>/dev/null; then
    DOWNLOAD="wget -q -O"
else
    echo "ERROR: Neither curl nor wget found. Please install one of them." >&2
    exit 1
fi

for NAME in "${FILES[@]}"; do
    DEST="$DATA_DIR/$NAME"
    if [ -f "$DEST" ]; then
        echo "[skip] $NAME already exists"
        continue
    fi

    GZ_NAME="${NAME}.gz"
    GZ_DEST="$DATA_DIR/$GZ_NAME"
    URL="${BASE_URL}/${GZ_NAME}"

    echo "[download] $URL"
    $DOWNLOAD "$GZ_DEST" "$URL"

    echo "[decompress] $GZ_NAME"
    gunzip -f "$GZ_DEST"

    if [ ! -f "$DEST" ]; then
        echo "ERROR: Expected file not found after decompression: $DEST" >&2
        exit 1
    fi
done

echo ""
echo "MNIST data ready in $DATA_DIR/"
ls -lh "$DATA_DIR/"
