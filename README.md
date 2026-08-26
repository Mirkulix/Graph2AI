# OrbitQO

> Graph-native AI-to-AI control plane: signed message graphs over a multi-LLM
> bus, with IDE extensions for Cursor, Antigravity, Trae, VS Code, ...

Status: early-alpha · Rust 2021 · React 19 · DeepSeek/OpenAI/Claude/Groq/Ollama

## Why use this

- **Multi-LLM, multi-IDE bus** — route the same task to DeepSeek, OpenAI, Claude, Groq or local Ollama from any of your IDEs, instead of being locked to one vendor's inline AI. (Caveat: the *IDE extension* has a native Anthropic client; the *server* does not — server-side "anthropic" goes through the generic OpenAI-shaped cloud slot, so it needs an OpenAI-compatible gateway, and that one slot is shared with OpenAI/Gemini/Mistral.)
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
# 1. Build server + CLI binaries
cargo build --bin qo --no-default-features
cargo build --bin qlang --no-default-features

# 2. Put binaries on PATH (Windows: copy to a directory in %PATH%)
cp target/debug/{qo,qlang}.exe ~/.cargo/bin/

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

## Share knowledge between agent sessions

Sessions exchange findings as signed OrbitQLang deltas instead of copied text.
The graph keeps provenance and evidence, so a later session can check a claim
rather than trust it.

```bash
# 1. Each worker gets a keypair. The seed is private; the command prints the
#    public half in the shape the trust store wants.
qlang graph keygen --out ~/.qlang/worker.key --key-id k1

# 2. The operator lists the key on the qo host. Without an entry here, that
#    producer's submissions are refused — a missing file trusts nobody.
cp .qlang/trusted_delta_producers.example.json .qlang/trusted_delta_producers.json
# ...paste in the public_key_hex from step 1, under your producer id.

# 3. Before a task: ask what is already known and backed by evidence.
qlang graph context --kind file --name src/auth.rs

# 4. After a task: hand findings back, signed.
qlang graph sign --seed ~/.qlang/worker.key --file findings.qlang \
  | qlang graph commit
```

A delta looks like this — one operation per line, no nesting:

```
DELTA|1|d-42
BY|worker-3|1700000000|abc123
SIG|ed25519|k1|<128 hex chars>
+E|file|src/auth.rs
+C|c1|file:src/auth.rs|auth hashes passwords with bcrypt
OK|c1|source|src/auth.rs|42:42|use bcrypt::hash;
```

About 4x smaller than the equivalent JSON — check it yourself with
`cargo run -p qo-knowledge --example orbitql_demo`.

What the server refuses: an unsigned or tampered delta (401), a replay of an
id that producer already used (409), and any write that would silently
overwrite another session's decision — that comes back as a conflict naming
both sides, visible in the cockpit's delta feed. See the whole thing, attacks
included, with `cargo run -p qo-knowledge --example signed_sync`.

Claude Code and other MCP clients get the same protocol as tools
(`orbit_graph_context`, `orbit_graph_commit_delta`, `orbit_graph_swarm_state`)
at `POST /mcp/v1`.

## The knowledge graph checks its own claims

Beyond the delta transport, `qo-knowledge` is a durable memory that *verifies*
what it is told rather than trusting it:

- **Deterministic source verification** — `orbit_graph_verify_source` reads a
  claim's source file and promotes it to `verified` only when every distinctive
  term appears verbatim. A partial match stays `proposed`; the graph never
  auto-refutes.
- **Proposals, not facts** — `orbit_graph_propose` runs model text through an
  admission gate (no `OK`/`NO`, self-contained references, length cap); claims
  land as `proposed` and are excluded from context until verified.
- **Proof receipts** — `orbit_graph_receipt` renders a claim's whole revision
  trail (who decided what, when, with which evidence), disagreements included.
- **Self-maintenance** — `orbit_graph_verify_all` harvests proposals,
  `orbit_graph_refresh_sources` marks stale facts when the code moves on, and
  `orbit_graph_heal_stale` re-verifies and heals the ones that still hold.
- **Divergence + health** — `orbit_graph_divergences` lists where agents settled
  in opposite directions; `orbit_graph_health` (or `qlang graph health`) is the
  one-block operator summary, and `qlang graph export|backup|import` cover
  portability. See it all in one run: `cargo run -p qo-knowledge --example
  lifecycle_demo`.

Multi-user: per-seat API keys with **enforced** roles (`qlang keys issue --role
viewer|member|admin`), per-IP rate limiting, a global body cap, and CORS
allow-listing — so a "$49 viewer seat" is genuinely read-only, and the server is
DoS-bounded before it ever binds to 0.0.0.0.

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
qo/          QO server crates — qo-{server,agents,knowledge,llm,memory,values,...}
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

Daily-driver-ready: yes — 1070 tests pass (`cargo test --workspace
--no-default-features`) with CI on Linux and Windows, role-enforced seats,
rate/body limits, structured tracing and an append-only audit log, all verified
end-to-end against a running server. Production-ready for a *public multi-tenant
SaaS*: not yet — it still lacks a formal threat model and an independent
security review, a release process and a redb migration story, and multi-tenancy
(one graph per instance today). Conversation history is browser-localStorage
only, the tool sandbox is `/workspace`-scoped, and the internal `<tool/>`
markers are not yet real MCP JSON-RPC. Use it, file issues, don't bet your job
on it yet.

## License

MIT — see `LICENSE` (TODO: add file if missing).
