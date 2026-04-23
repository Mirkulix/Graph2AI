# OrbitQLang IDE — Quickstart

Bring up the QO server, install the extension into every VS Code-family IDE on
your machine, and run a signed agent handover in under 5 minutes.

> Status: works on Windows. Linux/macOS not yet covered by the installer.

## One-shot install

From a fresh PowerShell in the repo root:

```powershell
pwsh -File scripts\install-qlang-ide.ps1
```

The script will:

1. `cargo build --release --bin qlang-cli --bin qo`
2. Copy both binaries into `%USERPROFILE%\.cargo\bin` (already on user PATH)
3. Compile the extension TypeScript and package it as `qlang-0.2.0.vsix`
4. Detect installed IDEs (VS Code, VS Code Insiders, Cursor, Windsurf, Kiro, Antigravity, Trae, VSCodium) and `--install-extension` the VSIX into each
5. (Optional, with `-StartServer`) launch `qo --offline` in a new window

### Useful flags

| Flag | Effect |
|------|--------|
| `-DebugBinaries` | Use `cargo build` (debug, ~5x faster, ~10x larger) instead of release |
| `-SkipBuild` | Don't rebuild — just (re)install the existing VSIX |
| `-NoInstall` | Build everything but skip IDE installation |
| `-StartServer` | Launch `qo --offline` in a detached window when done |

Examples:

```powershell
# Daily development: rebuild + reinstall + restart server
pwsh scripts\install-qlang-ide.ps1 -StartServer

# After tweaking only TS code: skip Rust rebuild
pwsh scripts\install-qlang-ide.ps1 -DebugBinaries

# Just (re)install in IDEs without rebuilding anything
pwsh scripts\install-qlang-ide.ps1 -SkipBuild
```

## After install

1. **Reload your IDE window** so the extension picks up new bindings:
   `Ctrl+Shift+P` → `Developer: Reload Window`
2. **Start the QO server** if not done by `-StartServer`:
   `qo --offline` (now globally available because it's in `.cargo\bin`)
3. **Open the cockpit** in a browser: <http://localhost:4646/supervisor>
4. **In the IDE**, the status bar shows:
   - `$(play) QLANG` — run current `.qlang` file
   - `$(shield) QLMS signed` — signed connection to QO is up
   - `$(export) Handover` — send current file as signed graph

## Verify it actually works

Open any source file in your IDE and:

- `Ctrl+Shift+P` → `QLMS: Handover Graph to Agent` → pick `developer`
- The notification should say **"Signed handover delivered to 'developer' (... msg). Verified."**
- The cockpit's message log should show: `vscode-assistant -> developer · Execute`
- If the agent replies, an inbox notification pops up — click `Open in editor` to inspect the reply graph

## What the extension contributes

| Command | Purpose |
|---------|---------|
| `QLMS: Check Connection` | Manual reconnect + settings shortcut |
| `QLMS: Handover Graph to Agent` | Sign + deliver active document as a QLMS graph |
| `QLMS: Show Inbox` | Browse the last 50 replies addressed to this IDE |
| `QLMS: Start Local QO Backend` | Spawn `qo --offline` in an integrated terminal |
| `QLANG: Open QO Dashboard` | Open `localhost:4646/supervisor` in browser |
| `QLANG: Run Current File` | `qlang-cli exec <file>` (Ctrl+Shift+R) |
| `QLANG: Open REPL` | `qlang-cli repl` |

## Settings (`Ctrl+,` → search "qlang")

| Setting | Default | Purpose |
|---------|---------|---------|
| `qlang.qlms.baseUrl` | `http://localhost:4646` | QO server URL |
| `qlang.qlms.authToken` | empty | Bearer token; falls back to `QO_AUTH_TOKEN` env |
| `qlang.qlms.seedHex` | empty | 64-hex signing seed; falls back to `QO_SEED_HEX` env, then to a generated workspace seed |
| `qlang.qlms.inbox.enabled` | `true` | Subscribe to the message bus and surface replies |
| `qlang.qlms.inbox.identity` | `vscode-assistant` | Agent name this IDE listens for |
| `qlang.lsp.enabled` | `true` | Spawn `qlang-cli lsp` for `.qlang` diagnostics |
| `qlang.lsp.path` | `qlang-cli` | Override the LSP server binary path |

## Troubleshooting

**Status bar shows `$(warning) QLMS offline`**
The QO server isn't reachable. Run `qo --offline` in a terminal, then click the badge.

**`command 'qlang.qlms.check' not found`**
Activation crashed. Reload the window. If it persists, view `Help > Toggle Developer Tools > Console` for the actual error — most likely a missing dependency in the VSIX. Rebuild with `pwsh scripts\install-qlang-ide.ps1`.

**`spawn qlang-cli ENOENT`**
LSP can't find the language server. Either:
- Run the installer (it copies `qlang-cli.exe` to `.cargo\bin`)
- Or set `qlang.lsp.path` to a full path
- Or set `qlang.lsp.enabled` to `false`

**Handover notification says "delivered unsigned"**
No seed configured. Set `qlang.qlms.seedHex` (64 hex chars) or `QO_SEED_HEX` env var, or trust the auto-generated workspace seed.
