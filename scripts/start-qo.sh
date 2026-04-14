#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
QO_DIR="$(dirname "$SCRIPT_DIR")"

cd "$QO_DIR"

# Load environment
if [ -f ~/.openclaw/.env ]; then
    export $(grep -v '^#' ~/.openclaw/.env | xargs)
fi

# Set QO-specific defaults
export QO_PORT="${QO_PORT:-4747}"
export LLVM_SYS_180_PREFIX="${LLVM_SYS_180_PREFIX:-/opt/llvm18}"

# Build if needed
if [ ! -f target/release/qo ] || [ "$(find qo/ -name '*.rs' -newer target/release/qo 2>/dev/null | head -1)" ]; then
    echo "Building QO..."
    cargo build --bin qo --release
fi

echo ""
echo "  ██████╗  ██████╗ "
echo "  ██╔═══██╗██╔═══██╗"
echo "  ██║   ██║██║   ██║"
echo "  ██║▄▄ ██║██║   ██║"
echo "  ╚██████╔╝╚██████╔╝"
echo "   ╚══▀▀═╝  ╚═════╝ "
echo ""
echo "  QLANG + Orbit = QO"
echo "  Port: $QO_PORT"
echo "  http://localhost:$QO_PORT"
echo ""

exec ./target/release/qo
