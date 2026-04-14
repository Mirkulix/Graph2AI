#!/bin/bash
# QLANG GPU Training — 2x RTX 2070 Super
# 30M parameter Mamba LM on WikiText-2
#
# Usage: bash scripts/gpu_train.sh
# Expected: 6-12 hours training, result in data/mamba_30m_final.bin

set -e

echo "=== QLANG GPU Training ==="
echo "Target: 30M parameter ternary Mamba LM"
echo ""

# Check GPU
nvidia-smi 2>/dev/null && echo "" || echo "WARNING: No GPU found, running on CPU (much slower)"

# Download data if needed
mkdir -p data/wikitext2
if [ ! -f data/wikitext2/train.txt ]; then
    echo "Downloading WikiText-2..."
    curl -sL "https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2/train.txt" -o data/wikitext2/train.txt
    curl -sL "https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2/valid.txt" -o data/wikitext2/valid.txt
fi
echo "Data: $(wc -c data/wikitext2/train.txt | awk '{print $1/1024/1024}') MB"

# Build
echo ""
echo "Building..."
LLVM_SYS_180_PREFIX=/opt/llvm18 cargo build --release --bin gpu_train 2>&1 | tail -3

# Train
echo ""
echo "=== Training Started ==="
echo "Log: data/gpu_train.log"
echo ""
./target/release/gpu_train data/wikitext2/train.txt data/mamba_30m 2>&1 | tee data/gpu_train.log

echo ""
echo "=== Done ==="
ls -la data/mamba_30m* 2>/dev/null
