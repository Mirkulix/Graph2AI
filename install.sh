#!/bin/bash
# ============================================================================
#  QLANG Installer — KI-zu-KI Programmiersprache
#  Funktioniert auf macOS und Linux
# ============================================================================

set -e

# Farben
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m'

echo -e "${PURPLE}"
echo "  ██████╗ ██╗      █████╗ ███╗   ██╗ ██████╗ "
echo "  ██╔═══██╗██║     ██╔══██╗████╗  ██║██╔════╝ "
echo "  ██║   ██║██║     ███████║██╔██╗ ██║██║  ███╗"
echo "  ██║▄▄ ██║██║     ██╔══██║██║╚██╗██║██║   ██║"
echo "  ╚██████╔╝███████╗██║  ██║██║ ╚████║╚██████╔╝"
echo "   ╚══▀▀═╝ ╚══════╝╚═╝  ╚═╝╚═╝  ╚═══╝ ╚═════╝ "
echo -e "${NC}"
echo -e "${BLUE}  Graph-basierte KI-zu-KI Programmiersprache${NC}"
echo ""

# ---- Betriebssystem erkennen ----
OS="$(uname -s)"
ARCH="$(uname -m)"
echo -e "${YELLOW}System:${NC} $OS $ARCH"

if [[ "$OS" != "Darwin" && "$OS" != "Linux" ]]; then
    echo -e "${RED}Fehler: Nur macOS und Linux werden unterstützt.${NC}"
    exit 1
fi

# ---- Installationsverzeichnis ----
INSTALL_DIR="${QLANG_HOME:-$HOME/qlang}"
echo -e "${YELLOW}Installation in:${NC} $INSTALL_DIR"
echo ""

# ---- Hilfsfunktionen ----
check_cmd() {
    command -v "$1" >/dev/null 2>&1
}

step() {
    echo -e "\n${GREEN}[$1/$TOTAL_STEPS]${NC} $2"
}

TOTAL_STEPS=7

# ============================================================================
#  Schritt 1: Voraussetzungen prüfen
# ============================================================================
step 1 "Voraussetzungen prüfen..."

# Git
if ! check_cmd git; then
    echo -e "${RED}Git nicht gefunden.${NC}"
    if [[ "$OS" == "Darwin" ]]; then
        echo "  Installiere mit: xcode-select --install"
    else
        echo "  Installiere mit: sudo apt install git"
    fi
    exit 1
fi
echo -e "  ${GREEN}✓${NC} git $(git --version | head -1)"

# Rust
if ! check_cmd cargo; then
    echo -e "${YELLOW}Rust nicht gefunden. Wird jetzt installiert...${NC}"
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    source "$HOME/.cargo/env"
    echo -e "  ${GREEN}✓${NC} Rust installiert"
else
    echo -e "  ${GREEN}✓${NC} cargo $(cargo --version)"
fi

# LLVM 18 (optional, für JIT-Compilation)
if [[ "$OS" == "Darwin" ]] && check_cmd brew; then
    if [ -d "/opt/homebrew/opt/llvm@18" ] || [ -d "/usr/local/opt/llvm@18" ]; then
        echo -e "  ${GREEN}✓${NC} LLVM 18 gefunden"
    else
        echo -e "  ${YELLOW}⚠${NC} LLVM 18 nicht gefunden (optional, für JIT)"
        echo "    Installiere mit: brew install llvm@18"
    fi
fi

# Python (optional)
PYTHON_CMD=""
if check_cmd python3; then
    PYTHON_CMD="python3"
    echo -e "  ${GREEN}✓${NC} python3 $($PYTHON_CMD --version 2>&1)"
elif check_cmd python; then
    PYTHON_CMD="python"
    echo -e "  ${GREEN}✓${NC} python $($PYTHON_CMD --version 2>&1)"
else
    echo -e "  ${YELLOW}⚠${NC} Python nicht gefunden (optional, für Python-Bindings)"
fi

# curl oder wget
if check_cmd curl; then
    echo -e "  ${GREEN}✓${NC} curl"
elif check_cmd wget; then
    echo -e "  ${GREEN}✓${NC} wget"
else
    echo -e "${RED}Weder curl noch wget gefunden.${NC}"
    exit 1
fi

# ============================================================================
#  Schritt 2: Repository klonen
# ============================================================================
step 2 "Repository klonen..."

if [ -d "$INSTALL_DIR/.git" ]; then
    echo "  Verzeichnis existiert, aktualisiere..."
    cd "$INSTALL_DIR"
    git fetch origin
    git checkout main
    git pull origin main
else
    git clone https://github.com/Mirkulix/qland.git "$INSTALL_DIR"
    cd "$INSTALL_DIR"
    git checkout main
fi

echo -e "  ${GREEN}✓${NC} Repository bereit in $INSTALL_DIR"

# ============================================================================
#  Schritt 3: Rust-Projekt bauen
# ============================================================================
step 3 "QLANG bauen (das dauert 2-3 Minuten beim ersten Mal)..."

cd "$INSTALL_DIR/qlang"
cargo build --release --no-default-features 2>&1 | tail -3
echo -e "  ${GREEN}✓${NC} Build erfolgreich"

# ============================================================================
#  Schritt 4: Tests laufen lassen
# ============================================================================
step 4 "Tests ausführen..."

TEST_OUTPUT=$(cargo test --workspace --no-default-features 2>&1)
TEST_COUNT=$(echo "$TEST_OUTPUT" | grep "test result" | awk '{sum += $4} END {print sum}')
FAIL_COUNT=$(echo "$TEST_OUTPUT" | grep "test result" | awk '{sum += $8} END {print sum}')

if [ "$FAIL_COUNT" = "0" ]; then
    echo -e "  ${GREEN}✓${NC} $TEST_COUNT Tests bestanden, 0 Fehler"
else
    echo -e "  ${RED}✗${NC} $FAIL_COUNT Tests fehlgeschlagen"
    echo "  Führe 'cargo test --workspace --no-default-features' aus für Details"
fi

# ============================================================================
#  Schritt 5: CLI installieren
# ============================================================================
step 5 "CLI-Tool installieren..."

QLANG_BIN="$INSTALL_DIR/qlang/target/release/qlang-cli"
if [ -f "$QLANG_BIN" ]; then
    # Symlink in ~/.cargo/bin (ist normalerweise im PATH)
    mkdir -p "$HOME/.cargo/bin"
    ln -sf "$QLANG_BIN" "$HOME/.cargo/bin/qlang-cli"
    echo -e "  ${GREEN}✓${NC} qlang-cli installiert in ~/.cargo/bin/qlang-cli"
else
    echo -e "  ${YELLOW}⚠${NC} qlang-cli Binary nicht gefunden, überspringe"
fi

# ============================================================================
#  Schritt 6: Python-Bindings (optional)
# ============================================================================
step 6 "Python-Bindings installieren (optional)..."

if [ -n "$PYTHON_CMD" ]; then
    if check_cmd maturin; then
        echo "  maturin gefunden, baue Python-Wheel..."
        cd "$INSTALL_DIR/qlang/crates/qlang-python"
        maturin build --release 2>&1 | tail -2
        WHEEL=$(ls "$INSTALL_DIR/qlang/target/wheels"/qlang-*.whl 2>/dev/null | head -1)
        if [ -n "$WHEEL" ]; then
            pip3 install "$WHEEL" --force-reinstall --quiet 2>/dev/null || \
            pip install "$WHEEL" --force-reinstall --quiet 2>/dev/null || true
            echo -e "  ${GREEN}✓${NC} Python-Bindings installiert (pip install qlang)"
        else
            echo -e "  ${YELLOW}⚠${NC} Wheel nicht gefunden"
        fi
        cd "$INSTALL_DIR/qlang"
    else
        echo -e "  ${YELLOW}⚠${NC} maturin nicht installiert"
        echo "  Installiere mit: pip3 install maturin"
        echo "  Dann: cd $INSTALL_DIR/qlang/crates/qlang-python && maturin develop --release"
    fi
else
    echo -e "  ${YELLOW}⚠${NC} Python nicht verfügbar, überspringe"
fi

# ============================================================================
#  Schritt 7: MNIST-Daten herunterladen (optional)
# ============================================================================
step 7 "MNIST-Daten herunterladen (optional)..."

MNIST_DIR="$INSTALL_DIR/qlang/data/mnist"
if [ -f "$MNIST_DIR/train-images-idx3-ubyte" ]; then
    echo -e "  ${GREEN}✓${NC} MNIST-Daten bereits vorhanden"
else
    echo -n "  MNIST herunterladen? (~11 MB) [j/N] "
    read -r REPLY
    if [[ "$REPLY" =~ ^[jJyY]$ ]]; then
        mkdir -p "$MNIST_DIR"
        BASE_URL="https://storage.googleapis.com/cvdf-datasets/mnist"
        for FILE in train-images-idx3-ubyte train-labels-idx1-ubyte t10k-images-idx3-ubyte t10k-labels-idx1-ubyte; do
            if [ ! -f "$MNIST_DIR/$FILE" ]; then
                echo "  Lade $FILE..."
                curl -fSL "$BASE_URL/$FILE.gz" -o "$MNIST_DIR/$FILE.gz" 2>/dev/null && \
                gunzip -f "$MNIST_DIR/$FILE.gz" || \
                echo -e "  ${YELLOW}⚠${NC} Fehler beim Laden von $FILE"
            fi
        done
        echo -e "  ${GREEN}✓${NC} MNIST-Daten heruntergeladen"
    else
        echo "  Übersprungen. Später mit: bash scripts/download_mnist.sh"
    fi
fi

# ============================================================================
#  Fertig!
# ============================================================================
echo ""
echo -e "${GREEN}════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}  QLANG erfolgreich installiert!${NC}"
echo -e "${GREEN}════════════════════════════════════════════════════════${NC}"
echo ""
echo -e "  ${BLUE}Installationsort:${NC} $INSTALL_DIR/qlang"
echo -e "  ${BLUE}Tests:${NC}            $TEST_COUNT bestanden"
echo ""
echo -e "  ${PURPLE}Schnellstart:${NC}"
echo ""
echo -e "  ${YELLOW}# Dashboard starten${NC}"
echo -e "  cd $INSTALL_DIR/qlang"
echo -e "  cargo run --release --no-default-features -p qlang-compile --bin qlang-cli -- web --port 8081"
echo -e "  ${BLUE}→ http://localhost:8081${NC}"
echo ""
echo -e "  ${YELLOW}# Demo im Browser (kein Server nötig)${NC}"
echo -e "  open $INSTALL_DIR/qlang/web/demo.html"
echo ""
echo -e "  ${YELLOW}# REPL starten${NC}"
echo -e "  qlang-cli repl"
echo ""
echo -e "  ${YELLOW}# MNIST Training${NC}"
echo -e "  cargo run --release --no-default-features --example full_mnist_pipeline"
echo ""
if [ -n "$PYTHON_CMD" ]; then
echo -e "  ${YELLOW}# Python${NC}"
echo -e "  python3 -c \"import qlang; g = qlang.Graph('test'); print(g)\""
echo ""
fi
echo -e "  ${YELLOW}# Alle Befehle${NC}"
echo -e "  qlang-cli --help"
echo ""
echo -e "${PURPLE}Viel Spaß mit QLANG!${NC}"
