# Codex Configuration — A-2A-qlang

Project-specific guidance for Codex when working in this repository.

## Project in one sentence

A-2A-qlang is a graph-native AI-to-AI control plane: a Rust workspace whose
goal is to let coding agents exchange **signed, executable graphs (QLMS)**
instead of loose text, coordinated by a **QO supervisor cockpit** with a
ternary/IGQK-compressed routing layer on top.

See `QLANG-STATUS.md` for the single source of truth on what actually works.
`README.md` and the vault docs describe the broader vision; when they disagree
with `QLANG-STATUS.md`, the status file wins.

## Behavioral rules (always enforced)

- Do what was asked — nothing more, nothing less. No speculative refactors.
- Prefer editing an existing file over creating a new one.
- Never create `.md` or README files proactively — only when asked.
- Never save working files, ad-hoc notes, or tests to the repo root.
- Always read a file before editing it.
- Never commit secrets, credentials, or `.env` files.
- When docs and code disagree, fix the docs — or the code, if the doc reflects
  the intended contract.

## File organization

- `/crates/` — Rust workspace members for the QLANG core (core, compile,
  runtime, agent, python, sdk).
- `/qo/` — QO supervisor and related service crates (server, agents, llm,
  memory, embed, consciousness, evolution, simulation, telegram, values).
- `/src/` — Top-level binaries (`qo`, `qlang`, routers, handover CLI).
- `/frontend/` — React UI (QO cockpit) and design system.
- `/spec/` — QLANG and QLMS wire-protocol specifications.
- `/docs/` — Human-facing docs (`BUILD.md`, `SUPERVISOR.md`, `QUICKSTART.md`,
  `vault/` for the deep design notes).
- `/tests/` — Integration tests (unit tests live beside their code).
- `/examples/` — Runnable examples and demos.
- `/scripts/` — Utility scripts (data download, training, swarm launchers).
- `/bindings/` — Non-Rust language bindings (Python/PyO3, stdlib QLMS parser).
- `/editors/` — VS Code and Theia integrations.
- `/k8s-manifests/` — Kubernetes manifests for distributed training.
- `/datasets/` — Local dataset cache (gitignored content).

Never put working files in the repo root.

## Architecture rules

- Rust 2021 edition, single Cargo workspace.
- Keep files under ~500 lines; split by concern when they grow.
- Public APIs have typed interfaces — no `serde_json::Value` at crate
  boundaries.
- Validate input at the system boundary (HTTP/WS handlers, protocol decoders,
  CLI argument parsers); trust it internally.
- QLMS uses pure-Rust crypto (SHA-256, HMAC-SHA-256) — no `openssl`/`ring`
  dependency. If a new primitive is needed, prefer implementing it in-tree
  under `qlang-core::crypto` over adding a dependency.
- Graphs are the primary data structure; prefer adding a typed op over adding
  a bespoke code path.

## Build & test

```bash
# Full build (requires LLVM 18, see docs/BUILD.md)
cargo build --release

# Build without LLVM (JIT disabled)
cargo build --release --no-default-features

# Run the QO server (port 4646 by default; override with QO_PORT)
cargo run --bin qo -- --offline

# Tests
cargo test --workspace

# Lint
cargo clippy --workspace --all-targets
```

- Run `cargo test` for the affected crate before committing.
- Run `cargo clippy` and fix or explicitly allow warnings in new code.
- Frontend: `cd frontend && npm run build` (served embedded by `qo`).

## Security rules

- No hardcoded API keys, tokens, or credentials in source or tests.
- `.env` is gitignored; use `.env.example` as the contract.
- HMAC/signature comparison MUST be constant-time (`subtle::ConstantTimeEq`
  or equivalent) — timing leaks break the QLMS threat model.
- All QLMS envelopes passing the wire must verify signatures before the
  payload is interpreted.
- Sanitize paths on any filesystem write exposed to the network (handover
  files, checkpoints, vault imports) to prevent traversal.

## Honesty rules (project-specific)

This project has a history of ambitious docs outrunning the code. To keep it
honest:

- `QLANG-STATUS.md` is the single source of truth for what works on real
  data. Update it when capabilities change.
- Never claim a metric without a runnable example that reproduces it.
- Experimental / broken ML paths must live under an `experimental/` module
  or be clearly tagged `#[doc = "experimental: <status>"]` — do not leave
  them looking production-ready.
- If a doc describes something that does not exist yet, mark it
  `> Status: planned` at the top.

## Git workflow

- Develop on the branch assigned for the current task; never push to `main`
  without explicit approval.
- Commit messages focus on the *why*, not the *what*.
- Do not amend pushed commits; create a new commit instead.
- Do not use `--no-verify`, `--force`, or skip signing unless the user asks.
