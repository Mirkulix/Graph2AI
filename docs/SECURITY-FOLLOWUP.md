# Security Follow-up

> Status as of 2026-08-26: the two critical findings below are **fixed**, and
> so are three of the four lower-severity ones (see *Also fixed since*). What
> remains is listed under *Still open* — nothing there is silently carried.
>
> Every finding was confirmed by reading the code, and every fix is covered by
> tests written as attacks rather than as happy paths.

The two critical defects let an unauthenticated caller reach code execution
and let a forged QLMS envelope onto the internal message bus. They were small,
local fixes; neither needed a redesign.

## 1. Authentication is fail-open — FIXED

> **Fix applied** in `src/main.rs`: without `QO_AUTH_TOKEN` the server now
> binds to `127.0.0.1` instead of `0.0.0.0` and logs a warning naming the
> consequence. The unauthenticated routes stay reachable for local
> development, but are no longer exposed to the network.
>
> Deliberately *not* done: refusing to start. That would break every existing
> local workflow for a threat that loopback binding already removes. If this
> is ever deployed as a shared service, the token must become mandatory —
> loopback is not authentication.

`qo/qo-server/src/auth.rs:35`

```rust
} else {
    Ok(next.run(request).await)   // QO_AUTH_TOKEN unset -> every request passes
}
```

An unset (or empty, `auth.rs:16-18`) `QO_AUTH_TOKEN` disables authentication
for the **whole** `api_router` — the layer is mounted once over all routes
(`lib.rs:901`) with no per-route exemptions. What that exposes:

- `POST /api/tools/exec_file` — runs arbitrary Python/Node/TS (`routes/workspace.rs:301`)
- `POST /api/git/merge`, `POST /api/git/discard` — write and delete branches
- `POST /api/providers/add` — reads and writes stored LLM API keys
- `POST /mcp/v1`, all of `/api/supervisor/*`

`CorsLayer::permissive()` (`lib.rs:914`) sits above the auth layer, so any web
page the user visits can reach these on `localhost:4646`.

**Fix.** Fail closed: refuse to start unless `QO_AUTH_TOKEN` is set, or bind to
loopback only and log a prominent warning when it is absent. Deciding between
those two is a product call — the current behaviour is neither.

A smaller companion fix: the query-parameter token fallback (`auth.rs:23-26`)
applies to every route, so tokens land in access logs and browser history.
Restrict it to the WebSocket handshake, which is the only caller that cannot
set a header.

## 2. QLMS signature verification is optional — FIXED

> **Fix applied** in `qo/qo-server/src/routes/mcp_qlms.rs`: the check is now
> `!decoded.signature_verified`, so an unsigned frame is refused rather than
> waved through. Legacy unsigned peers require an explicit
> `QO_QLMS_ALLOW_UNSIGNED=1`, which logs a warning on every frame it lets in.
> Covered by five tests in the same module, including the stripping attack
> itself (`unsigned_frame_is_rejected_by_default`).
>
> Also fixed here: the `eprintln!` that interpolated attacker-controlled agent
> names into stderr is now a structured `tracing::info!`, so a name containing
> newlines cannot forge log lines.

`qo/qo-server/src/routes/mcp_qlms.rs:121`

```rust
if decoded.signed && !decoded.signature_verified {
    return Err(unauthorized("QLMS signature verification failed"));
}
```

The check only fires when the frame *claims* to be signed. An attacker omits
the signed flag, `decoded.signed` is `false`, the condition short-circuits, and
the frame is pushed onto `state.message_bus` unverified (`:128-135`). A v1
frame hardcodes `signed: false` (`crates/qlang-agent/src/protocol.rs:496-513`)
and is therefore always waved through.

This is textbook signature stripping. Note the ordering: the Ed25519 migration
on this branch made the signature itself sound, which means the remaining gap
is entirely in whether the receiver *insists* on one.

**Fix.** Require a verified signature:

```rust
if !decoded.signed || !decoded.signature_verified {
    return Err(unauthorized("QLMS envelope must carry a verified signature"));
}
```

Then decide explicitly whether unsigned v1 frames stay supported. If they do,
they need their own opt-in flag and must not share a bus with signed traffic.

**Related, same area:**

- No pubkey allowlist. A valid signature only proves the sender holds *a* key,
  not an authorised one — there is no trust store (`protocol.rs:520-534`).
- `POST /qlms/v1.1/reply` takes the private Ed25519 seed as hex in the request
  body (`mcp_qlms.rs:160-164`). Clients send their signing key over the wire,
  and it is never zeroized.
- ~~`routes/qlms_demo.rs` (22 KB, HMAC-SHA256 + `ct_eq`) is not listed in
  `routes/mod.rs` and is unreachable.~~ — **DELETED** (2026-08-26): the dead
  module is gone, so nothing reads like an active protection that is not
  actually mounted.

## Also fixed since

- **SSRF in `web_fetch`** — `check_fetch_target` (`qo-server/src/tools.rs`)
  refuses loopback, link-local (including `169.254.169.254`), private and
  unique-local addresses, plus `localhost` / `.internal` / `.local` names, and
  redirects are no longer followed. 4 tests cover 15 internal targets and
  confirm public documentation still resolves. **Known limit, stated rather
  than implied:** this is a hostname check. A public DNS name that resolves to
  a private address still gets through — closing that needs resolve-then-pin,
  which `reqwest` does not expose.

- **Symlink escape from the tool sandbox** — `sandbox_resolve`
  (`routes/workspace.rs`) now canonicalises the resolved path and requires it
  to stay under the canonicalised root. For a file being created it checks the
  nearest existing ancestor instead, so first writes still work. A test
  actually creates an escaping symlink and asserts it is refused.

- **Allowlist bypass in `tool_shell`** (`qo-agents/src/tools.rs`) — the match
  is exact (`/tmp/evilcat` and `mygit` no longer pass), and the command is
  spawned directly instead of through `sh -c`. That removes the injection path
  `ls; rm -rf ~` entirely and makes the tool work on Windows, where `sh` does
  not exist. Shell operators are refused with an explanation rather than
  silently doing something else.

## Still open

- ~~**`CorsLayer::permissive()`**~~ — **FIXED** (2026-08-26).
  `cors_layer` now emits `Access-Control-Allow-Origin` only for origins in
  `QO_CORS_ORIGINS`; an empty list means the browser's same-origin policy
  applies and no cross-origin request is allowed. The embedded cockpit is
  same-origin and unaffected. Malformed entries are dropped, never widened to
  `*`.

- **Prompt injection via `web_fetch`** — narrowed further (2026-08-26):
  `exec_web_fetch` now wraps fetched content in an explicit
  `UNTRUSTED EXTERNAL CONTENT` boundary that tells the model the text is
  reference data, not instructions, and must not steer tool calls. This is a
  deterministic mitigation, not a proof — no wrapping makes an LLM immune to
  injected instructions, so the `write_file` / `/api/tools/exec_file` chain
  remains a risk a deployment should weigh against its tool exposure. The
  SSRF fix narrows where content can come from; the framing narrows what the
  model can read it as.
