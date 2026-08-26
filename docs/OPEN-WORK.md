# What is still missing

> Recorded 2026-08-26, after the OrbitQLang parser, merger, context compiler,
> MCP/CLI surface and cockpit delta view landed. Every item below was checked
> against the code, not carried over from a plan.
>
> `QLANG-STATUS.md` says what works. This says what does not.

## 1. The delta signature — DONE

> Resolved 2026-08-26. Kept here because the design decisions are worth
> knowing before touching this code.

Signing is wired end to end:

- **Transport.** `SIG|<algorithm>|<key_id>|<value>` carries the signature
  through the text layer. Without this line the feature was inert — the
  serializer dropped the field and the parser always set it to `None`, so no
  signature could ever reach a verifier.
- **What is signed.** `GraphDelta::signing_payload()` — the *unsigned* delta,
  domain-separated and version-tagged. Signing `to_canonical_json()` directly
  would be circular, since that output contains the signature field itself.
  Deriving the payload from the typed delta rather than the document text also
  means comments and whitespace cannot break verification.
- **Where it is checked.** `merge_signed_delta`, not the HTTP handler. There
  are three entry points (HTTP, MCP, CLI-via-HTTP); a check in one is a check
  the other two skip.
- **Trust store.** `.qlang/trusted_delta_producers.json`, operator-edited,
  producer -> keys with `active_from` / `accept_until` / `revoked_at`. A key is
  resolved *within* its producer, so one legitimate signer cannot claim
  another's provenance. Validity is judged by the receiver's clock — using the
  delta's own `emitted_at` would let a submitter backdate past a revocation.
  A missing or malformed file trusts nobody.
- **Replay.** `(producer, delta_id)` is recorded on acceptance. Re-submitting
  is refused with 409 rather than passed off as an idempotent retry.

One non-obvious constraint, found by a failing test: `sign_delta` normalises
claim provenance to match what the parser derives from the `BY` line.
Without that, a signed delta could fail verification after a clean round trip,
for no reason a reader could see.

`merge_delta` still exists for trusted local callers (the workspace indexer,
tests, examples) and does **not** check signatures. Anything reachable from a
network must call `merge_signed_delta`; a review confirmed no networked caller
uses the unsigned entry point.

Two residues, stated rather than implied:

- The replay record and the merge are separate redb commits. A crash between
  them consumes the delta id without applying anything; the producer must
  reissue under a new id. Closing it means threading one transaction through
  the merge — a larger change than the failure warrants.
- The trust store answers *whether* a producer may write, not *what about*.
  Any trusted producer can still make any claim about any entity.

## 2. Text-to-graph extraction

Roadmap §2, first half. The validating half is done: the parser rejects
malformed documents with line numbers, and the merger refuses to promote
anything without evidence. What does not exist is the pipeline that reads prose
and *proposes* structured claims from it. Today a worker must write OrbitQLang
by hand (or be prompted to).

## 3. Automatic Claude Code session hook

Roadmap §4. `qlang graph` and the MCP tools cover explicit, opt-in use. The
"request context before a task, submit a delta after" loop is still manual.

Note the constraint recorded in `CLAUDE.md`: the hook may never capture or
replace the session's prompt history. It requests bounded context and submits a
validated delta — nothing else.

## 4. One open security finding

Three of the four have been fixed (SSRF, symlink escape, shell allowlist) —
see `docs/SECURITY-FOLLOWUP.md` for what was done and the limits of each fix.

What remains:

- **`CorsLayer::permissive()`** (`qo-server/src/lib.rs`) sits above the auth
  layer. With a token set, any origin may still attempt authenticated requests;
  the loopback binding is what limits the untokened case. A deployment that
  serves anyone but localhost should narrow the allowed origins.
- **Prompt injection via `web_fetch`** is narrowed, not closed. Fetched content
  still enters the model's context verbatim, and `write_file` plus
  `/api/tools/exec_file` remain reachable from there.

## 5. `publish-pypi.yml` is dead

`.github/workflows/publish-pypi.yml` builds `crates/qlang-python`, which was
deleted (see *Intentionally Removed From Active Scope* in `QLANG-STATUS.md`).
It triggers on every `v*` tag and would start 15 matrix jobs that all fail at
checkout-time path resolution.

Left in place deliberately — deleting a workflow is the repository owner's
call. Either remove it, or repoint it at `bindings/python/qlms_parser` if that
is still meant to ship.

## 6. Operational gaps

None of these block the protocol; all of them block running it as a service.

- ~~No observability~~ — **done.** `qo-knowledge` now emits structured
  `tracing` events for merge conflicts, rejected deltas, context truncation and
  claim status changes.
- ~~No export/import~~ — **done.** `qo-knowledge::archive` round-trips the
  whole graph as JSON, every revision included. A backup *schedule* is still an
  operator decision; nothing runs it automatically.
- **Per-agent write policy is coarse.** The trust store decides *whether* a
  producer may write, not *what* it may write about. Any trusted producer can
  still make any claim about any entity. Per-entity or per-kind scoping is the
  next refinement, not a gap in the current guarantee.
- **Indexer covers files only** (`repository_indexer.rs:32` derives
  `EntityKind::File` and nothing else). Symbols, imports, endpoints and tests
  are modelled in `EntityKind` but never produced, so `orbit_graph_impact`
  answers at file granularity only.
- **No source-change invalidation.** `refresh_observed` exists in the store but
  nothing watches the workspace and calls it, so observations go stale silently
  between manual scans.

## Fixed in earlier passes

- CI ran `cargo test -p qo-server --lib`, which skipped `qo-knowledge` entirely
  and excluded every integration test. The protocol guarantees — round-trip
  losslessness, merge order-independence, the proposal/evidence rule — had no
  regression protection. Now `cargo test -p qo-knowledge -p qo-server`.
