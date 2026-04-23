# OrbitQO

> Graph-native AI-to-AI control plane: signed message graphs over a multi-LLM
> bus, with IDE extensions for Cursor, Antigravity, Trae, VS Code, ...

Status: early-alpha · Rust 2021 · React 19 · DeepSeek/OpenAI/Claude/Groq/Ollama

## Why use this

- **Multi-LLM, multi-IDE bus** — route the same task to DeepSeek, OpenAI, Claude, Groq or local Ollama from any of your IDEs, instead of being locked to one vendor's inline AI.
- **Signed message graphs (QLMS)** — every agent-to-agent envelope is HMAC-SHA256 signed and audit-logged, not a fire-and-forget chat call.
- **Closed-loop automation** — file save in your IDE can trigger an LLM review without a click, with rate-limits and per-rule cooldowns.

## Five killer workflows

| Workflow | Latency | Trigger |
|----------|---------|---------|
| Specialist Q&A | 1-3s | Ctrl+Shift+P → handover |
| Multi-LLM Consensus | 2-3s | Cockpit composer → "consensus" → 3-6 agents |
| Sequential Pipeline | 15-25s | Cockpit composer → "pipeline" → ordered chain |
| Auto-trigger on save | 2-5s | `.qlang/routing.json` + opt-in setting |
| Cross-IDE handover | 2-3s | Picker shows online IDE peers |

Bonus: server agents can call internal MCP-style tools (`read_file`, `write_file`, `web_fetch`, `exec_shell`) inside a sandbox while answering.

## Quickstart

Prerequisites: Rust toolchain (MSYS2 MINGW64 on Windows), Node 20+.

```bash
# 1. Build server + LSP binaries
cargo build --bin qo --no-default-features
cargo build --bin qlang-cli --no-default-features

# 2. Put binaries on PATH (Windows: copy to a directory in %PATH%)
cp target/debug/{qo,qlang-cli}.exe ~/.cargo/bin/

# 3. Build the IDE extension VSIX
cd editors/vscode && npx tsc -p . && npx -y @vscode/vsce package --allow-missing-repository
# Drag-drop the resulting qlang-0.2.0.vsix into Cursor / Antigravity / Trae / VS Code

# 4. Build the cockpit (qo serves frontend/dist/ on port 4646)
cd ../../frontend && npx tsc && npx vite build

# 5. Run
qo --offline
# open http://localhost:4646/
```

## Configure a provider

```
Cockpit (top-right) → Profile → Providers → DeepSeek → Add → API key → Save
```

Done. The 6 server agents (ceo, developer, designer, ...) are now LLM-backed.
Hot-reload, no restart needed.

## Opt into auto-trigger (optional)

```bash
mkdir -p .qlang
cp editors/vscode/src/example-routing.json .qlang/routing.json
# In your IDE: Settings → qlang.qlms.triggers.enabled = true → Reload Window
```

Defaults: 2s debounce, 3 concurrent, 60/hour rolling cap, per-rule cooldowns.

## Repo layout

```
crates/      Rust workspace — qlang-{core,compile,runtime,agent,sdk}
qo/          QO server crates — qo-{server,agents,llm,memory,values,...}
frontend/    React cockpit (Vite + TypeScript)
editors/     IDE integrations (vscode/, theia/)
spec/        QLMS protocol spec (v1.1)
scripts/     PowerShell installers (install-qlang-ide.ps1)
docs/        ARCHITECTURE.md, BUILD.md, QUICKSTART.md, vault/
```

## Read next

- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — full system topology, every module, honest limitations
- [editors/vscode/QUICKSTART.md](editors/vscode/QUICKSTART.md) — extension setup, commands, settings reference
- [QLANG-STATUS.md](QLANG-STATUS.md) — "what actually works" source of truth
- [CLAUDE.md](CLAUDE.md) — project-specific developer rules

## Honest status

Production-ready: no — needs CI, observability, and a security review before
anything mission-critical. Daily-driver-ready: yes — verified end-to-end with
Playwright against real DeepSeek calls (1.4s consensus across 3 agents,
21.7s 3-hop pipeline, 2-5s auto-trigger reviews on file save). Conversation
history is browser-localStorage only, the tool sandbox is `/workspace`-scoped,
and the internal `<tool/>` markers are not yet real MCP JSON-RPC. Use it,
file issues, don't bet your job on it.

## License

MIT — see `LICENSE` (TODO: add file if missing).
