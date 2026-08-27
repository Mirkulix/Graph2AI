# QLANG Status

Single source of truth for what currently works in this repository.

Last updated: 2026-08-26

## Current Product Scope

The repository is currently focused on the OrbitQLang control-plane path:

- `qo` server and supervisor flow
- agent orchestration in `qo-agents`
- QLMS / GraphMessage transport
- graph storage and message-bus streaming
- knowledge graph with provenance and evidence (`qo-knowledge`)
- per-agent model selection via the hardcoded table in
  `qo-server/src/agent_models.rs`
- DeepSeek-first multi-agent orchestration via `Planner -> Worker -> Reviewer`
- simplified intent classification (deterministic keyword-based)

## Confirmed Working Surfaces

- `cargo check --workspace --all-targets` passes clean, and
  `cargo test --workspace --no-default-features` (JIT disabled) is **1082
  passing tests with 0 failures** (re-measured 2026-08-26). The full with-JIT
  suite (`cargo test --workspace`) additionally needs LLVM 18 — `llvm-sys`'s
  build script requires `gcc`, so it does not run on a machine without that
  toolchain. The QLMS §14.2 timing conformance test is `#[ignore]`d (it took
  ~12 minutes in a dev build, long enough that nobody ran the suite) and runs
  in CI in release mode instead — `cargo test -p qlang-core --release --test
  crypto_timing -- --ignored`.
- `/api/chat`, `/api/chat/history`, `/api/goals`, `/api/agents`, `/api/messages/*`, `/api/neo/*` are the active server surfaces.
- `/api/multi-agent/run`, `/api/multi-agent/runs`, `/api/multi-agent/runs/{id}`, and `/api/multi-agent/stream` are active and wired into the cockpit.
- **QLMS v1.1 Bridge**: `/qlms/v1.1/deliver` and `/qlms/v1.1/reply` are active and integrated with the internal `MessageBus`.
- **IDE Integration**: VSCode / Trae extensions are functional, providing signed GraphMessage handover to the backend.
- **Knowledge graph**: `qo-knowledge` persists claims with provenance, evidence
  and append-only revisions in the shared redb database. Exposed over MCP as
  `orbit_graph_{search,neighbors,impact,add_claim,verify_claim,context,commit_delta,swarm_state}`
  alongside the existing three tools at `POST /mcp/v1`. The local cockpit exposes
  `GET /api/knowledge/{stats,snapshot}` and a bounded `POST /api/knowledge/index`.
  The indexer is bound to QO's configured workspace, records deterministic file
  observations with source evidence, and preserves stale/reobserved revisions
  when file content changes. A live scan of this repository completed with 377
  indexed files and no scan errors.
- **OrbitQLang surface syntax**: `qo-knowledge::orbitql` renders and parses the
  line-oriented, bracket-free delta format. `GraphDelta -> text -> GraphDelta`
  round-trips losslessly across every entity kind, relation, evidence kind and
  optional-field combination (16 tests). Parse errors carry a source line, and
  `parse_recovering` reports every bad line rather than stopping at the first.
  Measured 4.2x smaller than the equivalent canonical JSON (191 vs 794 bytes);
  reproduce with `cargo run -p qo-knowledge --example orbitql_demo`.
- **Deterministic merger**: `qo-knowledge::merge` applies a delta with
  idempotency, append-only history, source-revision ordering and explicit
  conflict records. Two conflicting deltas produce the same graph and the same
  conflicts in either arrival order (14 tests, incl. `merge_is_order_independent`).
- **Context compiler**: `qo-knowledge::context` compiles a bounded, deterministic
  prompt block from a subgraph. Unverified proposals are excluded by default and
  labelled when explicitly requested; truncation is always stated (10 tests).
- **Proposal admission (text-to-graph)**: `qo-knowledge::extract` is the gate
  between "an LLM wrote something" and "the graph may consider it".
  `propose_from_text` parses recoveringly and applies the proposal policy:
  model text may never verify or refute (`OK`/`NO` are refused with their
  source line), every reference must resolve (subjects/objects declared in the
  document or known to the caller's context, relations only to claims in the
  same document), statements are length-capped, and admission is all-or-nothing
  with every violation reported at once. `proposal_system_prompt` renders the
  constrained prompt for integrations. A proposal merges but is not load-bearing
  until an authorised verifier promotes it (12 unit + 4 integration tests;
  `cargo run -p qo-knowledge --example extract_demo`). **Wired to the server** as
  `orbit_graph_propose` (MCP, write), `orbit_graph_proposal_prompt` (MCP, read)
  and `POST /api/knowledge/propose` — the text-to-graph pipeline is no longer
  library-only.
- **Deterministic source-evidence verification**: `qo-knowledge::sourcecheck`
  lets the graph check its own proposals against real source code.
  `verify_claim_against_source` resolves a claim's source file within the
  workspace root (canonicalised, escapes refused), reduces the statement to its
  distinctive terms (stopwords and short words dropped), and promotes the claim
  to `verified` via the single `verify_claim` path only when *every* term
  appears literally — capturing the exact matching line as evidence. The check
  is deliberately asymmetric and conservative: it confirms, never disproves
  (a partial match is inconclusive and leaves the claim untouched, because a
  paraphrase must not be refuted by absence), and it never re-promotes a
  settled claim. Fully offline and reproducible (6 unit + 6 integration tests;
  `cargo run -p qo-knowledge --example sourcecheck_demo`). Exposed to the
  running server as the `orbit_graph_verify_source` MCP tool and
  `POST /api/knowledge/verify-source` HTTP route; verified end to end against a
  live instance (propose → verify → re-promotion guard → load-bearing context).
- **Proof receipts**: `qo-knowledge::receipt` answers "why should I believe
  this?" with a deterministic, bounded block — a claim's current status, its
  full append-only revision history (who decided what, when), its evidence, and
  every other claim about the same subject including disagreements, which are
  kept and never overwritten. A `proposed` claim's receipt says so explicitly
  rather than reading like a fact (5 tests). Exposed as the
  `orbit_graph_receipt` MCP tool and `GET /api/knowledge/receipt?claim_id=…`;
  verified end to end against a live instance (verified claim + refuted
  counter-claim rendered side by side).
- **Verification sweep**: `qo-knowledge::sourcecheck::verify_all_proposals`
  checks every open (`proposed`) claim against its source in one deterministic
  pass — the harvest step after workers propose. It reports how many were
  `verified` (code substantiates every term), `inconclusive` (left proposed)
  and `unavailable` (missing source / unsafe path), and is incremental: a
  settled claim drops out of the next sweep. Exposed as the
  `orbit_graph_verify_all` MCP tool and `POST /api/knowledge/verify-all`;
  verified end to end against a live instance (3 proposals → 1 verified,
  1 inconclusive, 1 unavailable; a second sweep checks only the 2 still open).
- **Source refresh (staleness)**: `qo-knowledge::sourcecheck::refresh_sources`
  re-checks every settled (`verified`/`observed`) claim against its source and
  marks it `stale` when its recorded verbatim excerpt no longer appears in the
  file — the graph noticing its facts rot. Deterministic: it compares the exact
  captured excerpt, never a filesystem timestamp. Claims without a verbatim
  excerpt are skipped, not guessed at. Exposed as the
  `orbit_graph_refresh_sources` MCP tool and `POST /api/knowledge/refresh-sources`;
  verified end to end (rev 1 proposed → rev 2 verified → rev 3 stale when the
  source changed).
- **Divergence report**: `qo-knowledge::divergence` answers the aggregate
  question the merger never did — *where, across the whole graph, do we hold a
  settled fact and a settled counter-fact about the same thing?* A subject is
  divergent when it carries a load-bearing (`verified`/`observed`) claim and a
  `refuted` claim; the report lists both sides with statements, ordered
  deterministically, and never asserts a contradiction from prose — it surfaces
  the disagreement for a human to judge (4 tests). Exposed as the
  `orbit_graph_divergences` MCP tool and `GET /api/knowledge/divergences`;
  verified end to end against a live instance (verified "bcrypt" claim + refuted
  "md5" counter-claim rendered side by side).
- **Self-healing**: `qo-knowledge::sourcecheck::heal_stale` re-verifies every
  `stale` claim against its *current* source (whole file, not the stale line
  hint) and promotes it back to `verified` with fresh evidence when its
  statement is still literally substantiated — the code moved, the fact did
  not. A genuinely rotted fact stays stale. The store keeps the full
  `verified → stale → verified` trail, so rot and healing are both auditable
  (2 integration tests). Exposed as the `orbit_graph_heal_stale` MCP tool and
  `POST /api/knowledge/heal-stale`; verified end to end (rot → heal → receipt
  shows both the old and the fresh evidence).
- **Graph health**: `qo-knowledge::health` gathers the operator's one-block
  summary — load-bearing (verified/observed), open proposals, stale, refuted,
  divergence count and entity count — deterministically from the store.
  Exposed as the `orbit_graph_health` MCP tool, `GET /api/knowledge/health`
  and `qlang graph health`; verified over all three surfaces.
- **Worker sync path**: OrbitQLang document -> parse -> validate -> merge ->
  per-operation report is covered end to end (5 tests in `worker_sync_flow.rs`).
- **Lifecycle capstone**: `tests/lifecycle.rs` walks the whole story in one
  deterministic run — propose → sweep → verify → disagreement (divergence) →
  rot (refresh) → heal → receipt — so `cargo test -p qo-knowledge --test
  lifecycle` reproduces the project's central claim end to end. Its narrative
  twin, `cargo run -p qo-knowledge --example lifecycle_demo`, tells the same
  story in eight printed acts (the showpiece).
- **Server route coverage**: `qo-server` now tests the knowledge HTTP handlers
  directly against a real `AppState` built by `build_app` (arg parsing,
  serialization, status codes — including the 404 path). The full lifecycle
  walks through the routes themselves: verify-source → receipt → verify-all →
  divergences → refresh-sources → heal-stale. This closes the gap where the
  routes had only manual end-to-end verification.
- **MCP surface** now also exposes `orbit_graph_commit_delta` and
  `orbit_graph_swarm_state` alongside the previous six tools. HTTP equivalents:
  `POST /api/knowledge/delta` and `GET /api/knowledge/deltas`.
- **Claude Code plugin**: `plugins/orbitqlang-claude` now documents the full
  lifecycle and ships a **`sync` skill** — the closed agent loop: pull bounded
  `orbit_graph_context` before a non-trivial task, then submit validated
  proposals and evidence (`add_claim` → `verify_source`/`verify_claim` →
  `commit_delta`) after. The `orbit` skill documents all 14 `orbit_graph_*`
  tools (verified: every referenced tool exists in the server's tool set and
  vice versa). The plugin README/description were updated to match.
- **Cockpit**: `DeltaLogPanel` shows the live delta feed with a conflict filter
  and the submitted document per entry. `KnowledgeGraphPanel`'s inspector now
  exposes the trust loop to a human: a proposed claim has a
  **"check against source"** action (calls `POST /api/knowledge/verify-source`
  and refreshes the graph when the claim promotes) and a **"proof receipt"**
  action that renders `GET /api/knowledge/receipt` inline. The panel header also
  offers **"sweep proposals"** (`POST /api/knowledge/verify-all`),
  **"refresh sources"** (`POST /api/knowledge/refresh-sources`) and
  **"heal stale"** (`POST /api/knowledge/heal-stale`) with inline outcome
  counts, and a **divergent-subjects banner** (`GET /api/knowledge/divergences`)
  that lists where the graph holds both a load-bearing and a refuted claim. The
  whole harvest/staleness/healing cycle is now one click. Frontend production
  build passes (`npm run build`).
- **CLI adapter**: `qlang graph {context,commit,deltas,export,import,backup,backups,health,events}`
  drives the same protocol over HTTP for clients that do not speak MCP.
  `commit` exits 3 when the merge reported a conflict, so CI can gate on it.
  `export`/`import`/`backup`/`backups` cover portability; `health` is the
  operator summary; `events` is the recent knowledge-lifecycle stream (proposals,
  verifications, sweeps, refreshes, heals, imports, backups) with the actor.
  Verified end to end against a running server.
- **QLMS signatures are now enforced**: `/qlms/v1.1/deliver` requires a verified
  signature. Dropping the signed flag no longer bypasses verification; unsigned
  legacy frames need an explicit `QO_QLMS_ALLOW_UNSIGNED=1` opt-in (5 tests).
- **Delta signatures are enforced end to end**: `POST /api/knowledge/delta` and
  `orbit_graph_commit_delta` require an Ed25519 signature from a producer key
  listed in `.qlang/trusted_delta_producers.json`. Verification happens inside
  `merge_signed_delta`, not at each transport, so no entry point can skip it.
  Keys support rotation windows and revocation, judged by the receiver's clock.
  A `(producer, delta id)` pair is accepted once — replays are refused, not
  silently re-applied. 28 tests, written as attacks (foreign key, producer
  impersonation, field tampering, algorithm downgrade, backdating past a
  revocation). Verified against a running server: unsigned -> 401,
  replay -> 409, tampered -> 401, and the graph kept only the legitimate write.
- **Archive**: `qo-knowledge::archive` exports and restores the whole graph as
  JSON, including every superseded revision and counter-evidence. Import is
  additive and never overwrites; a colliding claim id is reported, not merged
  (9 tests). Exposed to the running server as `GET /api/knowledge/export`
  (read-only backup snapshot) and `POST /api/knowledge/import` (additive
  restore, provenance verbatim, behind the auth layer; unsupported archive
  versions are refused). Verified over the wire (export → import idempotent →
  version guard) and by a route-level round-trip test that rebuilds the audit
  trail into a fresh instance. **Backup policy**: `write_backup`/`list_backups`
  write timestamped snapshots to the backup directory and list them newest
  first; exposed as `POST /api/knowledge/backup`, `GET /api/knowledge/backups`
  and `qlang graph backup|backups`. **Recovery**: `POST /api/knowledge/restore`
  and `qlang graph restore [--exported-at <ts>]` recover the graph from the
  newest (or a specific) backup, additively — the operator path after a redb
  loss. The *schedule* stays an operator decision (cron).
- **Observability**: `qo-knowledge` emits structured `tracing` events for merge
  conflicts, rejected deltas, context truncation and every claim status change.
- **Per-seat access (SaaS foundation)**: `.qlang/api_keys.json` holds named
  keys with roles (member/admin/viewer), managed via `qlang keys issue|list|revoke`.
  Each is compared in constant time and individually revocable, so a team can be
  given seats without sharing one token. Verified end to end: no key → 401,
  valid seat → 200; an instance with issued seats binds to 0.0.0.0 rather than
  loopback (7 tests). The single `QO_AUTH_TOKEN` still works as the admin key.
- **Roles are enforced, not just stored**: write routes are behind a
  `require_write` middleware and admin routes behind `require_admin`, and the MCP
  dispatcher refuses write tools for a `viewer`. A viewer seat reads but gets
  403 on every write/admin route and on every write MCP tool; a member writes
  but gets 403 on admin routes. Verified by router-level attack tests and over
  the wire (viewer read 200 / write 403 / admin 403; member write 400 / admin
  403; admin 422; unauthenticated 401).
- **Auth fails closed**: without `QO_AUTH_TOKEN` and with no issued API keys the
  server binds to 127.0.0.1
  instead of 0.0.0.0 and logs a warning, so unauthenticated code-execution
  routes are not reachable from the network.
- **Rate + body limits (DoS bound)**: a per-IP token bucket (default 50 req/s,
  burst 200; `QO_RATE_PER_SEC`/`QO_RATE_BURST`) runs as the outermost middleware,
  before auth, so unauthenticated floods get 429; and a global body cap (default
  16 MiB, `QO_MAX_BODY_BYTES`) refuses oversized bodies with 413 before any
  handler runs. Verified by unit + router tests and over the wire.
- **Threat model**: `docs/THREAT-MODEL.md` enumerates assets, trust boundaries,
  eleven threat→mitigation pairs with code locations, and the residual risks
  (prompt-injection is a mitigation not a proof, no frame pubkey allow-list,
  seed-over-the-wire, best-effort per-IP limiting, single-tenant, no independent
  review).
- **CORS is no longer permissive**: `CorsLayer::permissive()` (any origin) was
  replaced by a configurable allow-list (`QO_CORS_ORIGINS`, comma-separated).
  An empty list emits no `Access-Control-Allow-Origin`, so the browser's
  same-origin policy applies and the embedded cockpit is unaffected. Verified
  behaviorally against a live instance (restrictive default + exact allow-list
  matching).
- **`web_fetch` output is framed as untrusted data**: fetched content is
  wrapped in an explicit `UNTRUSTED EXTERNAL CONTENT` boundary telling the
  model it is reference data, not instructions, and must not steer tool calls.
  This is a deterministic mitigation for prompt injection, not a proof — an LLM
  may still follow injected text.
- Frontend production build currently passes via `cd frontend && npm run build`.
- All non-deterministic ML training and evolution loops have been purged.

## Intentionally Removed From Active Scope

These areas were removed from the project to ensure a lean, deterministic core:

- legacy ML/GPU logic (candle, mamba-tokenizer, training loops)
- deleted `qo-embed` and legacy `qlang-python` bindings
- spiking neural network (SNN) / STDP logic
- evolution/consciousness/organism subsystems
- legacy ML binaries and training examples

## Honesty Notes

- The project has been radically simplified to a pure AI-to-AI control plane.
- All heavy ML dependencies have been removed from `Cargo.toml`.
- The multi-agent product path is real but still intentionally narrow: planning, generation, review, run history, and cockpit visibility are implemented; general tool autonomy is not.
