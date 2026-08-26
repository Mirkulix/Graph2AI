# Threat Model — OrbitQO

> Scope: the **QO server** and the **knowledge-graph / QLMS / agent-tool** paths
> it exposes. This documents what the current code protects against and — just
> as important — what it does *not*. Grounded in `docs/SECURITY-FOLLOWUP.md` and
> the measures implemented and verified in this repository; where a claim is not
> re-verified, it says so rather than asserting.
>
> This is a living document, not a proof of security. An independent review is
> still outstanding.

## Assets

| Asset | Worth protecting because |
|---|---|
| Knowledge graph (claims, provenance, evidence, revisions) | the product's trust basis — a forged claim laundered as `verified` poisons every later session |
| Provider API keys | stored credentials for paid LLMs |
| Delta producer keys (Ed25519 seeds) | whoever holds one can write as that producer |
| Workspace files | the sandboxed working set agents read/write |
| Audit log (`action_history`) | the append-only record of who did what |

## Trust boundaries

```
client (IDE / MCP / CLI / cockpit)
        │  HTTP(S) + WS/SSE
        ▼
   QO server  ──► redb store (local, one file)
        │
        ├──► workspace filesystem (sandboxed)
        └──► LLM providers (outbound, untrusted *content* back)
```

Everything below the QO server is local; the server is the single trust
boundary between the network and the assets.

## Threats → mitigations

| # | Threat | Mitigation (where) | Status |
|---|---|---|---|
| T1 | Unauthenticated code execution over the network | Auth fails closed: no token **and** no keys ⇒ bind 127.0.0.1 only + warn | ✅ `src/main.rs` |
| T2 | Forged / signature-stripped QLMS envelope | Envelope must carry a *verified* signature; unsigned legacy frames need explicit opt-in | ✅ `mcp_qlms.rs` |
| T3 | Forged / tampered / replayed graph delta | Ed25519 + trust store (`active_from`/`accept_until`/`revoked_at`, receiver clock) + `(producer, delta_id)` accepted once | ✅ `qo-knowledge::trust`, `merge_signed_delta` |
| T4 | SSRF via `web_fetch` | Loopback/link-local/private/unique-local + `.internal`/`.local` refused; redirects not followed | ✅ `tools.rs::check_fetch_target` |
| T5 | Symlink / path escape from the tool sandbox | Canonicalise + require path under root | ✅ `workspace.rs::sandbox_resolve` |
| T6 | Shell injection / allow-list bypass | Exact program match, direct spawn (no `sh -c`), metacharacters refused | ✅ `qo-agents::tools` |
| T7 | Cross-origin request with a stolen token | CORS allow-list (`QO_CORS_ORIGINS`); empty = same-origin only | ✅ `lib.rs::cors_layer` |
| T8 | Privilege escalation (viewer seat writing) | `require_write` / `require_admin` middleware + per-tool MCP check | ✅ `auth.rs`, `knowledge_tools.rs` |
| T9 | DoS — oversized body / request flood | `DefaultBodyLimit` + per-IP token bucket (before auth) | ✅ `rate_limit.rs` |
| T10 | Timing side-channel on token/key comparison | Constant-time `ct_eq` | ✅ `api_keys.rs`, `auth.rs` |
| T11 | Prompt injection via fetched content | Content framed as *untrusted data* (`UNTRUSTED EXTERNAL CONTENT` boundary) | ⚠️ mitigation, not a proof (see R1) |

## Residual risks (stated, not implied)

- **R1 — prompt injection.** Framing fetched content reduces but does not
  eliminate the chance a model follows instructions embedded in it. The
  `write_file`/`exec_file` chain is still reachable; a deployment must weigh
  tool exposure against this.
- **R2 — no pubkey allow-list for QLMS *frames*.** A valid frame signature
  proves the sender holds *a* key, not an *authorised* one. (Graph *deltas* do
  have a trust store; frames do not.)
- **R3 — seed over the wire.** `POST /qlms/v1.1/reply` takes the private
  Ed25519 seed as hex in the request body and never zeroizes it.
- **R4 — per-IP rate limit is best-effort.** A token bucket keyed by peer IP
  bounds a single source; it does not stop a distributed or IP-spoofed flood.
- **R5 — single tenant.** One `qo.redb`, one workspace, one graph per instance;
  there is no cross-team isolation to reason about yet (multi-tenancy is
  deliberately deferred until a paying customer exists).
- **R6 — no independent review.** The security fixes are covered by
  attack-oriented tests written by the same authors; an outside review has not
  happened.
- **R7 — provider-key storage.** Keys are persisted in redb and not returned by
  `/api/providers`; the at-rest encryption reported in older docs was not
  re-verified this session.
