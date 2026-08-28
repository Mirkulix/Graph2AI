# qo session bridge

An MCP stdio server that puts DeepSeek Harness (DSH) sessions **into** the qo
server and lets them **talk to each other** over the qo message bus.

Without it, two DSH sessions on the same machine are strangers: each has its own
context and neither can see or reach the other. With it, a session announces
itself once and can then address any other session by name.

## What it adds to an agent

Four tools, mounted as `mcp__sessions__<tool>`:

| Tool | What it does |
|---|---|
| `session_register` | Registers this session in qo presence under a short handle plus a one-line note. Makes it visible in the qo cockpit and addressable by peers. The bridge keeps the registration alive. |
| `session_directory` | Lists every session/agent currently registered, with what each said it is working on. |
| `session_send` | Sends a message to one handle, or to `*` for every other registered session. |
| `session_inbox` | Reads messages addressed to this session — only what arrived since the last read, unless `all=true`. |

Delivery is asynchronous: a message waits in qo until the recipient reads its
inbox. Nothing arrives mid-turn, and no session can interrupt another.

## How it maps onto qo

Nothing here is a new protocol — the bridge only uses endpoints qo already
serves:

- `POST /api/presence/register` + `/api/presence/heartbeat/{identity}` — the
  directory. Registering also creates the session's mailbox on qo's internal
  message bus, which is what makes it a legal broadcast target.
- `GET /api/presence` — who is online.
- `POST /api/broadcast` — one bus message per recipient, carrying the text.
- `GET /api/messages/recent` — the bus ring buffer (newest 200) the inbox reads.

Presence is **ephemeral by design**: a 60s TTL kept alive by this process's
heartbeat, wiped when qo restarts. That is correct for "who is live right now".
For the durable half, see `import-sessions.mjs` below.

## Durable session inventory

`node import-sessions.mjs` publishes the sessions that already exist on disk
into qo's graph store as one `AgentTask` graph — title, workspace and turn
counts per session, read from the harness projection cache (no session log is
decompressed). Listed by `GET /api/graphs`, visible in the cockpit, and
append-only: re-running stores a fresh snapshot instead of overwriting the
previous one.

## Wiring it into DSH

One `dsh-mcp-client` row in the agent preset
(`~/.dsh/.agent-presets/orbit/agent.cordis.yml`):

```yaml
- id: mcp-sessions
  name: '@deepseek-ai/dsh-mcp-client'
  config:
    serverName: sessions
    transport: stdio
    command: C:\Users\a.b\pprog\node\node.exe
    args:
      - C:\Users\a.b\Graph\OrbitQLang\bridges\dsh-session-bridge\server.mjs
    cwd: C:\Users\a.b\Graph\OrbitQLang
    toolCallTimeoutMs: 30000
```

The preset is mounted once per harness process, so **one** bridge process serves
every session on the host — which is what makes presence heartbeats and
per-handle inbox cursors shared rather than duplicated.

`command` is an absolute path on purpose: a `!!js` expression is only resolved
by the Loader, and a PATH lookup can select a different interpreter than the one
the composition was verified against.

## Configuration

All optional, read from the environment:

| Variable | Default | Meaning |
|---|---|---|
| `QO_URL` | `http://127.0.0.1:4646` | qo base URL |
| `QO_TOKEN` | first secret in `QO_API_KEYS` | bearer token |
| `QO_API_KEYS` | `<repo>/.qlang/api_keys.json` | key store to read the token from |
| `DSH_HOME` | `~/.dsh` | harness home (importer only) |

## Design notes

- **Zero dependencies.** Only `node:` builtins and `fetch`, so the file runs
  from whatever directory the harness spawns it in, with no install step.
- **Exact message ids.** qo's bus ids are u64 snowflakes past
  `Number.MAX_SAFE_INTEGER`; parsed as plain JSON numbers they round to
  multiples of 16, so two messages from the same millisecond compare equal and
  one silently disappears from an inbox. The bridge re-quotes `"id":<digits>`
  before parsing and compares with `BigInt`.
- **Handles are model-supplied.** One bridge process serves all sessions, so it
  cannot infer which session is calling; every tool takes the handle explicitly
  and the preset persona tells the agent to register one early.
- **Peer messages are input, not orders.** The persona states this: a message
  from another session is evidence to judge, never an instruction to obey.

## Limitations

- A handle is claimed by whoever registers it first; there is no authentication
  between sessions beyond the shared qo API key.
- The inbox window is qo's ring buffer (200 messages). A session that never
  reads its inbox while 200 newer messages pass loses the older ones.
- Cursors live in this process. Restarting the harness re-reads from the newest
  message, so messages that arrived while it was down are only visible with
  `all=true`.
- Presence disappears when qo restarts; sessions re-register on their next
  tool call (the heartbeat re-registers automatically).
