#!/bin/bash
# QLANG Setup Script
# Usage: curl -sSf https://raw.githubusercontent.com/Mirkulix/qland/main/qlang/setup.sh | bash

set -e

echo "=== QLANG Setup ==="
echo ""

# Check Rust
if ! command -v cargo &> /dev/null; then
    echo "Installing Rust..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    source "$HOME/.cargo/env"
fi

echo "Rust: $(rustc --version)"

# Check OS and install LLVM
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    echo "Linux detected."
    if ! command -v llvm-config-18 &> /dev/null && ! command -v llvm-config &> /dev/null; then
        echo "Installing LLVM 18..."
        sudo apt-get update
        sudo apt-get install -y llvm-18-dev libpolly-18-dev libzstd-dev
    fi
elif [[ "$OSTYPE" == "darwin"* ]]; then
    echo "macOS detected."
    if ! command -v llvm-config &> /dev/null; then
        echo "Installing LLVM via Homebrew..."
        brew install llvm@18
    fi
else
    echo "Windows detected. Please install LLVM manually from:"
    echo "  https://github.com/llvm/llvm-project/releases"
fi

# Build
echo ""
echo "Building QLANG..."
cargo build --release --workspace

# Test
echo ""
echo "Running tests..."
cargo test --workspace

# Done
echo ""
echo "=== QLANG installed successfully! ==="
echo ""
echo "Try these commands:"
echo "  cargo run --release --example train_autograd    # Train a neural network"
echo "  cargo run --release --example full_pipeline     # Complete demo"
echo "  cargo run --release --bin qlang-cli -p qlang-compile -- repl  # Interactive REPL"
echo ""
