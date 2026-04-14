#!/bin/bash
set -e

echo "=== QLANG IDE Setup (Eclipse Theia) ==="
echo ""

# Check prerequisites
command -v node >/dev/null 2>&1 || { echo "Node.js required. Install: brew install node"; exit 1; }
command -v yarn >/dev/null 2>&1 || { echo "Yarn required. Install: npm install -g yarn"; exit 1; }

NODE_VERSION=$(node -v | cut -d. -f1 | tr -d 'v')
if [ "$NODE_VERSION" -lt 18 ]; then
    echo "Node.js 18+ required. Current: $(node -v)"
    exit 1
fi

echo "Node.js: $(node -v)"
echo "Yarn: $(yarn --version)"
echo ""

# Build the VS Code extension first
echo "[1/4] Building QLANG VS Code extension..."
cd "$(dirname "$0")/../vscode"
npm install --no-optional 2>/dev/null || yarn install
npx tsc -p tsconfig.json
echo "  Extension built: out/extension.js"

# Copy extension to Theia plugins
echo "[2/4] Installing extension as Theia plugin..."
cd "$(dirname "$0")"
cd ../theia
mkdir -p plugins/qlang
cp -r ../vscode/package.json plugins/qlang/
cp -r ../vscode/out plugins/qlang/
cp -r ../vscode/syntaxes plugins/qlang/
cp -r ../vscode/language-configuration.json plugins/qlang/
echo "  Plugin installed in plugins/qlang/"

# Install Theia
echo "[3/4] Installing Theia packages (this takes a few minutes)..."
yarn install

# Build Theia
echo "[4/4] Building Theia IDE..."
yarn build

echo ""
echo "=== QLANG IDE Ready ==="
echo ""
echo "Start with:  cd editors/theia && yarn start"
echo "Open:        http://localhost:3000"
echo ""
echo "Make sure qlang-cli is in your PATH:"
echo "  cp target/release/qlang-cli /usr/local/bin/qlang-cli"
