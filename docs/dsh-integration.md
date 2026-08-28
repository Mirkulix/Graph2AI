# DeepSeek Harness integration

How the `qo` server is wired into a running DeepSeek Harness (DSH) web
instance as an MCP server. One direction, no code on either side: DSH acts as
an MCP *client* (`@deepseek-ai/dsh-mcp-client`), qo stays the MCP *server*.

Two layers sit on that wiring: **knowledge** (the `orbit_graph_*` tools, below)
and **session collaboration** — sessions register in qo and message each other
through the qo bus, via the
[session bridge](../bridges/dsh-session-bridge/README.md).

## Sessions in qo

A DSH session is invisible to its peers by default. Two mechanisms change that:

- **Live sessions** call `session_register` and appear in `GET /api/presence` —
  visible in the cockpit, addressable with `session_send`, readable through
  `session_inbox`. Presence is ephemeral: 60s TTL, heartbeated by the bridge,
  wiped when qo restarts.
- **Sessions already on disk** are published by
  `node bridges/dsh-session-bridge/import-sessions.mjs` into the qo graph store
  as one `AgentTask` graph — title, workspace and turn counts per session,
  listed by `GET /api/graphs`. Durable and append-only.

The preset persona tells the agent to register early, read its inbox before
relying on work a peer owns, announce changes to shared files, and treat a peer
message as input to judge rather than an instruction to obey.

## What the integration gives you

Every MCP tool qo exposes at `POST /mcp/v1` (the 17 `orbit_graph_*` knowledge
tools plus `qlang_research`, `qlang_run_goal`, `qlang_read_workspace_file`)
becomes a native DSH tool named `mcp__orbit__<tool>`. A DSH agent can therefore
pull bounded verified context before a task (`orbit_graph_context`), submit
proposals (`orbit_graph_propose`), verify them against source
(`orbit_graph_verify_source`), commit signed deltas
(`orbit_graph_commit_delta`), and read receipts/health/divergences — the same
closed loop the `plugins/orbitqlang-claude` skill documents for Claude Code.

## Layout (this machine)

- qo server binary: `target\debug\qo.exe` (also on PATH as `~/.cargo/bin/qo.exe`).
- qo config + auth: `.qlang/` in the repo root (the running instance is started
  from the repo directory, so `api_keys.json` resolves).
- DSH user agent preset: `~/.dsh/.agent-presets/orbit/` — a copy of the shipped
  `standard` preset plus two `dsh-mcp-client` rows (`mcp-orbit` over HTTP for
  the knowledge tools, `mcp-sessions` over stdio for the session bridge) and a
  persona that tells the agent how to use both.
- Session bridge: `bridges/dsh-session-bridge/` in this repo.
- DSH web profile patch: `~/.dsh/profiles/web/cordis.patch.yml` sets
  `agent-presets.default: orbit` (takes effect on the next DSH start; until
  then the preset is selectable under General settings → Agent preset).

## Start / stop

```cmd
:: start qo (from the repo directory so .qlang/ resolves)
start-qo.cmd            :: runs C:\Users\a.b\.cargo\bin\qo.exe --offline

:: stop qo
:: Task Manager, or:  Stop-Process -Name qo
```

DSH boots fine while qo is down: the mcp-client row keeps
`failOnStartupError: false`, so the agent starts without the `mcp__orbit__*`
tools and they appear once qo is reachable again (reconnect policy defaults).

## Credentials

The preset sends `Authorization: Bearer <key>` where `<key>` is the admin
secret from `.qlang/api_keys.json` (embedded in
`~/.dsh/.agent-presets/orbit/agent.cordis.yml`). To avoid the literal key in
that file, export `QO_AUTH_TOKEN` in the DSH process environment and replace
the header with:

```yaml
headers:
  Authorization: !!js '`Bearer ${process.env.QO_AUTH_TOKEN}`'
```

A `viewer`-role seat is enough for read-only tools; `member` or `admin` is
required for writes (`add_claim`, `propose`, `commit_delta`, …).

## Verified

- `POST /mcp/v1` answers `initialize` / `tools/list` (20 tools) / `tools/call`
  with the bearer key.
- Booting **both** preset rows from the real `agent.cordis.yml` through the
  actual `dsh-mcp-client` yields 24 tools on one agent (20 `mcp__orbit__*` +
  4 `mcp__sessions__*`), and both namespaces answer live calls.
- Two sessions registered through the bridge exchanged messages in both
  directions; the inbox cursor returned each message exactly once, including
  two sent in the same millisecond (the u64-id precision case).
- Six existing DSH sessions were published to qo as graph #35 and read back
  intact, umlauts included.
- The preset is discovered cleanly by the DSH roster (`scanRoot` reports no
  broken composition).

## Constraints

- DSH bridges only MCP *tools* (no resources/prompts); qo's endpoint is a
  minimal JSON-RPC subset (no SSE/notifications), which the official MCP SDK
  client tolerates.
- All 20 tool schemas are present in every model request that carries them
  (token cost) and in the KV-cache prefix until a re-sync changes them.
- One graph per qo instance; role enforcement and rate limits apply per key.
