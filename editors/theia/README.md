# QLANG IDE (Eclipse Theia)

A browser-based IDE for the QLANG programming language, built on Eclipse Theia.

## Features

- Full syntax highlighting for `.qlang` files
- LSP integration (diagnostics, completions, hover) via `qlang-cli lsp`
- Integrated terminal for running QLANG programs
- File explorer, search, and outline view
- Run and REPL commands

## Prerequisites

- **Node.js 18+** - `brew install node` (macOS) or [nodejs.org](https://nodejs.org)
- **Yarn** - `npm install -g yarn`
- **qlang-cli** - built from the project root:
  ```bash
  cargo build --release -p qlang-compile
  cp target/release/qlang-cli /usr/local/bin/qlang-cli
  ```

## Quick Start

```bash
# From the project root
cd editors/theia
bash setup.sh

# Start the IDE
yarn start
```

Open **http://localhost:3000** in your browser.

## Manual Setup

If you prefer to set things up step by step:

```bash
# 1. Build the VS Code extension
cd editors/vscode
npm install
npx tsc -p tsconfig.json

# 2. Copy extension to Theia plugins
cd ../theia
mkdir -p plugins/qlang
cp ../vscode/package.json plugins/qlang/
cp -r ../vscode/out plugins/qlang/
cp -r ../vscode/syntaxes plugins/qlang/
cp ../vscode/language-configuration.json plugins/qlang/

# 3. Install and build Theia
yarn install
yarn build

# 4. Start
yarn start
```

## Using the CLI

Once `qlang-cli` is in your PATH, you can also start the IDE with:

```bash
qlang-cli ide
qlang-cli ide --port 4000   # custom port
```

## Configuration

### Custom Port

```bash
yarn start --port 4000
```

Or via the CLI:

```bash
qlang-cli ide --port 4000
```

### LSP Settings

The QLANG language server is configured through the VS Code extension settings:

- `qlang.lsp.enabled` - Enable/disable the language server (default: `true`)
- `qlang.lsp.path` - Path to the `qlang-cli` binary (default: `qlang-cli`)

## Architecture

```
editors/theia/
  package.json        Theia app configuration with browser target
  setup.sh            One-command setup script
  plugins/qlang/      VS Code extension (copied during setup)
    package.json        Extension manifest
    out/extension.js    Compiled LSP client
    syntaxes/           TextMate grammar
    language-configuration.json

editors/vscode/
  package.json        VS Code extension manifest
  src/extension.ts    LSP client source (TypeScript)
  tsconfig.json       TypeScript compiler config
  syntaxes/           TextMate grammar for .qlang files
  language-configuration.json  Bracket matching, comments, etc.
```

## Troubleshooting

### "qlang-cli: command not found"

Make sure the CLI is built and in your PATH:

```bash
cargo build --release -p qlang-compile
export PATH="$PWD/target/release:$PATH"
```

### Theia build fails

Ensure you have the correct Node.js version (18+) and that Yarn is installed:

```bash
node -v   # should be v18.x or higher
yarn -v   # should be installed
```

### LSP not connecting

Check that `qlang-cli lsp` works from the terminal:

```bash
echo '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"capabilities":{}}}' | qlang-cli lsp
```

You should see a JSON-RPC response.
