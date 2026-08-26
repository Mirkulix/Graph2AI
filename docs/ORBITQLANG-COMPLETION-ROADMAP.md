# OrbitQLang Completion Roadmap

> Status: partially delivered as of 2026-08-26. Sections 1-4 below are marked
> individually. `QLANG-STATUS.md` remains the authority on what runs; this file
> records the intent and what is still open.
>
> **Delivered:** the surface syntax and parser contract (§1), the proposal
> admission pipeline and the graph-to-prompt context compiler with the
> proposal/evidence policy in it (§2), the deterministic merger and conflict
> engine (§3), and the MCP/HTTP sync tools (§4, first half).
>
> **Still open:** the automatic Claude Code sync hook (§4, second half), the
> transport-neutral Gemini adapter, and everything under *Supporting
> production work*.

## Goal

OrbitQLang should let several coding-agent sessions exchange small, signed,
validated graph deltas instead of relying on copied conversation text. The
knowledge graph remains the durable evidence layer; the components below turn
it into a synchronised multi-agent state system.

## 1. Stable OrbitQLang surface syntax and parser contract

> **Delivered.** `qo-knowledge::orbitql` — grammar in the module docs, exact
> parse errors with line numbers, round-trip tests, and a version field on
> every delta (`DELTA|<version>|<id>`).

The repository already contains a QLANG text parser in `qlang-compile`. What
is still required is a small, versioned public contract aimed at LLM output:

- canonical grammar and examples for entities, relations, claims, evidence,
  graph deltas, and conflict declarations;
- deterministic parse errors with source spans;
- round-trip tests: text -> typed graph -> canonical text;
- compatibility/version field on every serialized delta.

**Done when:** a worker can emit a compact `.qlang` delta and the parser either
creates a typed, valid graph delta or returns an exact validation error.

## 2. Context translator: text <-> graph

### Text to graph

> **Delivered.** `qo-knowledge::extract` — `propose_from_text` is the
> admission gate between model text and the graph: it parses recoveringly and
> applies the proposal policy (no `OK`/`NO` from model text, self-contained
> entity/claim references, statement cap, all-or-nothing admission), and
> `proposal_system_prompt` renders the constrained prompt an integration hands
> the LLM. The LLM call itself lives in the integration layer (the crate has
> no LLM dependency); wiring it into QO's server surface is §4 work.

Introduce a constrained extraction pipeline. An LLM may suggest structured
claims and deltas, but the parser and policy layer validate them before storage.
Every LLM-derived item starts as `proposed`; only deterministic source evidence
or an explicit verification promotes it.

### Graph to prompt

> **Delivered.** `qo-knowledge::context` — bounded, deterministic, and it
> excludes unverified proposals unless the caller asks for them by name.

Build a bounded context compiler that selects a task-relevant subgraph and
renders a compact worker prompt containing only load-bearing claims, relations,
evidence locators, freshness status, and a token budget.

**Done when:** the same task and graph state produce a stable context payload,
and unverified proposals never appear as established facts.

## 3. Delta merger and conflict engine

> **Delivered.** `qo-knowledge::merge` — every rule below is enforced and
> covered by `tests/merge_determinism.rs`.

Define a typed `GraphDelta` model with adds, revisions, refutations, relation
updates, source revision, producer identity, and signature metadata. The merger
must validate every delta before it reaches the main graph.

Required conflict rules:

- append-only claim history; no silent overwrite;
- identical changes are idempotent;
- incompatible changes become explicit conflict records;
- verified evidence outranks proposals, but never deletes them;
- stale source revisions cannot overwrite newer observations;
- merge decisions are auditable and reproducible.

**Done when:** two workers can submit conflicting deltas in either order and
the resulting graph plus conflict record is deterministic.

## 4. CLI and agent integration

> **Partially delivered.** Steps 1-4 exist as MCP tools
> (`orbit_graph_context`, `orbit_graph_commit_delta`, `orbit_graph_swarm_state`),
> as HTTP routes, and as the `qlang graph` CLI adapter for non-MCP clients.
> Signing is now wired: `SIG` carries the signature through the text layer,
> `merge_signed_delta` is the gate, and `qlang graph keygen|sign` produces what
> the trust store accepts. Still open: the automatic Claude Code session hook.

Extend the Claude Code plugin from manual MCP use to an explicit sync workflow:

1. request compact graph context before a non-trivial task;
2. collect a worker's proposed delta after the task;
3. validate, sign, and merge it through QO;
4. return merge/conflict status to the worker.

Provide the same protocol as a transport-neutral CLI/API adapter for Gemini SDK
and future clients. No integration may transmit secrets, unrestricted paths,
or unvalidated LLM output.

**Done when:** two independently started worker sessions can read shared
context, submit deltas, and observe the same merged result through QO.

## Supporting production work

- incremental indexer coverage for symbols, imports, endpoints, and tests;
- source-change invalidation and re-observation;
- graph export/import and backup policy;
- authentication and per-agent write policies for non-local deployments;
- end-to-end tests for parser, merger, MCP, CLI and cockpit;
- observability for scan, merge, conflict and context-compilation outcomes.

## Delivery order

1. Formal delta schema and parser tests.
2. Deterministic merger and conflict records.
3. Graph-to-prompt context compiler.
4. Constrained text-to-graph proposal extraction.
5. Claude Code automatic sync, then Gemini adapter.
6. Production security, backup and operational hardening.
