#!/usr/bin/env bash
set -euo pipefail

LLVM_VERSION=18
RUST_MIN_VERSION="1.83.0"

info()  { printf "\033[1;34m[INFO]\033[0m  %s\n" "$*"; }
warn()  { printf "\033[1;33m[WARN]\033[0m  %s\n" "$*"; }
error() { printf "\033[1;31m[ERROR]\033[0m %s\n" "$*"; exit 1; }

detect_os() {
    case "$(uname -s)" in
        Linux*)  OS=linux ;;
        Darwin*) OS=macos ;;
        CYGWIN*|MINGW*|MSYS*) OS=windows ;;
        *) error "Unsupported operating system: $(uname -s)" ;;
    esac
    info "Detected OS: $OS"
}

check_rust() {
    if command -v rustc &>/dev/null; then
        local version
        version=$(rustc --version | awk '{print $2}')
        info "Rust $version found"
    else
        warn "Rust not found. Installing via rustup..."
        curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
        # shellcheck source=/dev/null
        source "$HOME/.cargo/env"
        info "Rust installed: $(rustc --version)"
    fi
}

install_llvm_linux() {
    if dpkg -l "llvm-${LLVM_VERSION}-dev" &>/dev/null; then
        info "LLVM $LLVM_VERSION already installed"
        return
    fi
    info "Installing LLVM $LLVM_VERSION..."
    sudo apt-get update
    sudo apt-get install -y \
        llvm-${LLVM_VERSION}-dev \
        libpolly-${LLVM_VERSION}-dev \
        libzstd-dev
}

install_llvm_macos() {
    if brew list llvm@${LLVM_VERSION} &>/dev/null; then
        info "LLVM $LLVM_VERSION already installed"
        return
    fi
    info "Installing LLVM $LLVM_VERSION via Homebrew..."
    if ! command -v brew &>/dev/null; then
        error "Homebrew is required on macOS. Install from https://brew.sh"
    fi
    brew install llvm@${LLVM_VERSION}
    export LLVM_SYS_180_PREFIX="$(brew --prefix llvm@${LLVM_VERSION})"
    info "Set LLVM_SYS_180_PREFIX=$LLVM_SYS_180_PREFIX"
}

install_llvm_windows() {
    warn "On Windows, install LLVM $LLVM_VERSION manually from https://releases.llvm.org"
    warn "Then set LLVM_SYS_180_PREFIX to the LLVM install directory."
    if [ -z "${LLVM_SYS_180_PREFIX:-}" ]; then
        error "LLVM_SYS_180_PREFIX is not set. Please install LLVM $LLVM_VERSION and set this variable."
    fi
    info "LLVM_SYS_180_PREFIX=$LLVM_SYS_180_PREFIX"
}

install_llvm() {
    case "$OS" in
        linux)   install_llvm_linux ;;
        macos)   install_llvm_macos ;;
        windows) install_llvm_windows ;;
    esac
}

build_qlang() {
    info "Building QLANG (release mode)..."
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
    cd "$PROJECT_DIR"
    cargo build --release
    info "Build complete: target/release/qlang-cli"
}

run_tests() {
    info "Running tests..."
    cargo test --workspace
    info "All tests passed"
}

main() {
    info "=== QLANG Installer ==="
    detect_os
    check_rust
    install_llvm
    build_qlang
    run_tests

    echo ""
    info "=== Installation Successful ==="
    echo ""
    echo "  Usage:"
    echo "    # Start interactive REPL"
    echo "    ./target/release/qlang-cli repl"
    echo ""
    echo "    # Start API server"
    echo "    ./target/release/qlang-cli serve --port 8080"
    echo ""
    echo "    # Run an example"
    echo "    cargo run --release --example full_pipeline"
    echo ""
    echo "    # Install system-wide"
    echo "    sudo cp target/release/qlang-cli /usr/local/bin/"
    echo ""
    echo "    # Or use Make"
    echo "    make repl    # REPL"
    echo "    make api     # API server"
    echo "    make docker  # Docker image"
    echo ""
}

main "$@"
