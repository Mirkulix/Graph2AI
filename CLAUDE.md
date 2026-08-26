# Claude Code Configuration — A-2A-qlang

Project-specific guidance for Claude Code when working in this repository.

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
- QLMS **hashing** is in-tree pure Rust (SHA-256, HMAC-SHA-256) — no
  `openssl`/`ring` dependency. For a new *hash* or MAC, prefer implementing it
  under `qlang-core::crypto` over adding a dependency.
- **Signature schemes are the exception: never hand-roll one.** Signing uses
  `ed25519-dalek`. The previous in-tree scheme was forgeable from public data
  alone — `verify` recomputed the signature from values an attacker already
  had — and it read as plausible until someone attacked it. See the note above
  `Keypair` in `qlang-core::crypto` for the full account.
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

## Active OrbitQLang / Graph2AI session handoff

Read this section before starting OrbitQLang implementation work. It records
the current product direction and prevents parallel agents from creating a
second graph system.

### Product decision

- **Rust is the production implementation language.** Do not create a Python
  `orbitqlang` core, a NetworkX source of truth, or a second graph store.
  Python may only become an adapter/SDK after the Rust protocol is complete.
- `qo-knowledge` is the durable knowledge graph. Claims retain provenance,
  evidence and append-only revisions. An LLM may propose claims; it must not
  establish facts without the existing evidence policy.
- QO already exposes a local MCP server and a knowledge cockpit. Extend these
  surfaces; do not replace or duplicate them.
- The authoritative capability statement is `QLANG-STATUS.md`; the planned
  delivery order is `docs/ORBITQLANG-COMPLETION-ROADMAP.md`.

### Current implementation state

The first uncommitted foundation for the multi-agent protocol is in
`qo/qo-knowledge`:

- `src/delta.rs` defines the typed, versioned `GraphDelta` contract:
  producer identity, source revision, optional signature metadata and
  append-only operations.
- `src/orbitql.rs` is the beginning of a compact, line-oriented, bracket-free
  OrbitQLang surface syntax using fixed `|` separators.

Before building new features, run `git status --short`. Other agents may be
editing these files. Do not overwrite another agent's uncommitted work; if a
file overlaps your assignment, report the collision and choose a separate
module.

### Immediate completion target

Finish the **formal delta schema and parser contract** before beginning CRDT
or UI work:

1. Fix any focused `qo-knowledge` compilation errors first. In particular,
   `GraphDelta::from_json` must parse explicitly as `GraphDelta`.
2. Export `orbitql` from `qo-knowledge` only after its parser compiles.
3. Complete deterministic `GraphDelta -> OrbitQLang -> GraphDelta` round-trip
   parsing and serialization. Parsing errors must include at least a source
   line and never silently create partial truth.
4. Keep worker-added claims at `proposed`; verify/refute operations must carry
   correctly signed evidence direction (`supports: true` for verify,
   `false` for refute).
5. Add focused unit tests for valid round trips, malformed lines, unsupported
   versions, invalid claim state and escaping of `|`, backslashes and newlines.
6. Run `cargo test -p qo-knowledge`. Do not claim completion until it passes.

### Delivery order after the parser is green

1. Deterministic merger in a separate `qo-knowledge` module: idempotency,
   append-only history, source-revision ordering and explicit conflict records.
2. Bounded graph-to-prompt context compiler that excludes unverified
   proposals from established context.
3. MCP/API tools: `query_subgraph`, `commit_graph_delta`, `get_swarm_state`.
4. Cockpit: live delta log, conflict viewer and agent/session monitor.
5. Optional adapters and grammar-constrained decoding for inference systems
   that actually control token sampling. Claude Code cannot mask tokens before
   sampling; it must submit output to deterministic validation instead.

### Non-negotiable integration safeguards

- Never capture or replace Claude's full prompt history. Request bounded graph
  context before a task and submit a validated proposed delta afterwards.
- Do not trust, merge or persist unvalidated LLM text.
- Do not stage or commit unrelated dirty files. Check the diff and run the
  focused tests before every commit.
