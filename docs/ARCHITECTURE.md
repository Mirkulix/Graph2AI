# OrbitQO — System Architecture

> Status: 2026-04-23 — early-alpha, end-to-end verified, merged into `NewWayLLMHandling`.

## What this is

A graph-native AI-to-AI control plane. IDEs (Cursor, Antigravity, Trae, Kiro,
VS Code, ...) and headless server agents talk to each other through a single
QO server using **cryptographically signed message graphs** (QLMS protocol).
Replaces "ask one chatbot one question" with "trigger, route, fan-out, chain,
audit" workflows across multiple LLMs and multiple machines.

## Topology

```
                 ┌─────────────────────────────────────┐
                 │  Cockpit (React, Swiss Modernism)   │
                 │  localhost:4646/                    │
                 │  3-pane: Agents · Conversation ·    │
                 │  Detail · plus secondary views      │
                 └────────────────┬────────────────────┘
                                  │ HTTP + SSE + WebSocket
                                  ▼
   ┌──────────────────────────────────────────────────────────────────┐
   │  QO Server (Rust, axum, port 4646)                              │
   │                                                                 │
   │  ┌──────────────────────┐    ┌──────────────────────────────┐   │
   │  │ QLMS Bus             │    │ LlmRouter (hot-reload)       │   │
   │  │ HMAC-SHA256 envelopes│    │ DeepSeek/OpenAI/Groq/Anthrop/│   │
   │  │ 6 named agents +     │    │ Ollama, install_provider live│   │
   │  │ ad-hoc IDE identities│    └──────────────────────────────┘   │
   │  └──────────────────────┘                                       │
   │                                                                 │
   │  ┌──────────────────────────────────────────────────────────┐   │
   │  │ Agent loop per server agent (ceo / developer / ...):     │   │
   │  │   1. recv from mailbox                                   │   │
   │  │   2. ReAct tool loop (max 3 iter):                       │   │
   │  │        DeepSeek → parse <tool/> markers → execute →      │   │
   │  │        DeepSeek again with results → final text          │   │
   │  │   3. if metadata.chain non-empty:                        │   │
   │  │        forward as Execute to chain[0] with chain[1..]    │   │
   │  │      else if metadata.pipeline_origin set:               │   │
   │  │        Result lands at pipeline_origin                   │   │
   │  │      else:                                               │   │
   │  │        Result lands at msg.from                          │   │
   │  └──────────────────────────────────────────────────────────┘   │
   │                                                                 │
   │  Routes:                                                        │
   │  /qlms/v1.1/{deliver,reply}    — bus bridge for IDEs            │
   │  /api/messages/{stream,...}    — SSE + stats + agents           │
   │  /api/consensus                — fan-out + dual scores          │
   │  /api/presence/{register,...}  — IDE registry, 60s TTL          │
   │  /api/providers/*              — hot-reload LLM providers       │
   │  /api/{health,chat,goals,...}  — supporting endpoints           │
   └──────────────────────────────────────────────────────────────────┘
                                  ▲
                                  │ /qlms/v1.1 + SSE inbox
                                  │
   ┌──────────────────────────────────────────────────────────────────┐
   │  IDE Extensions (one per IDE instance)                          │
   │    Identity   <ide>-<host>-<6hex> (auto-gen, persisted)         │
   │    Presence   register on activate, heartbeat 25s, deregister   │
   │    Inbox      SSE listen for to:identity, 4-action notification │
   │    Auto-respond  optional: incoming → call configured LLM →     │
   │                  signed Result back via /qlms                   │
   │    Triggers    on_save / on_change / on_open → dispatch per     │
   │                .qlang/routing.json with rate-limit + cooldown   │
   │    Handover    Strg+Shift+P → picks server agent OR online IDE  │
   └──────────────────────────────────────────────────────────────────┘
```

## Killer workflows

| Workflow | What you do | What happens |
|----------|------------|--------------|
| **Spezialist-Frage** | Composer → "single" → developer → text | DeepSeek answers in 1-3s |
| **Konsens** | Composer → "consensus" → 3-6 agents → text | All N answer in parallel, 2 scores (Jaccard + trigram) |
| **Pipeline** | Composer → "pipeline" → ordered chain → text | Sequential, each agent builds on prior, final back to cockpit |
| **Auto-Trigger** | Save a `.rs` file in IDE | `routing.json` matches → developer reviews via DeepSeek → "Insert as comment" |
| **Cross-IDE** | Strg+Shift+P → handover to peer-IDE identity | Peer's auto-respond fires its LLM → reply back to your inbox |
| **Tool-Use** | Ask developer "read README.md and summarize" | Agent emits `<tool name="read_file"/>`, server executes, agent re-prompts with result |

## Components

### Server (Rust)

| Crate / Module | Purpose |
|----------------|---------|
| `qo-server` | axum HTTP server, AppState, agent loop with ReAct + chain forwarding |
| `qo-server::routes::consensus` | `/api/consensus` fan-out + Jaccard + trigram cosine scoring |
| `qo-server::routes::presence` | `/api/presence` IDE registry, 60s TTL, 30s sweeper |
| `qo-server::tools` | 4 tools: read_file/write_file/web_fetch/exec_shell with sandbox |
| `qo-llm` | LlmRouter with RwLock<Option<X>> per tier, hot-reload via install_provider |
| `qo-llm::deepseek` | DeepSeek client (chat-completions API) |
| `qlang-runtime::deepseek_client` | In-tree TLS client used by ProviderRegistry |
| `qlang-agent::bus` | MessageBus, mailboxes, SSE listener fan-out |
| `qlang-agent::protocol` | QLMS frame encode/decode, HMAC sign/verify |

### Frontend (React/TypeScript)

| Module | Purpose |
|--------|---------|
| `cockpit/AgentsPane` | Left pane, agents + connected IDEs + live tail |
| `cockpit/ConversationPane` | Center, single/consensus/pipeline composer modes, message cards w/ AUTO badge |
| `cockpit/DetailPane` | Right pane, routes between GraphInspector / AgentStats / IdePresenceDetail / ProvidersDetail |
| `cockpit/secondary/*` | Federation / WerteRadar / Hardware / Knowledge3D / Settings |
| `lib/api.ts` | Typed fetch wrapper for all backend routes |
| `lib/sse.ts` | SSE subscription with auto-reconnect (mirror of IDE inbox pattern) |
| `lib/history.ts` | localStorage-backed liveTail persistence (cap 100) |

### IDE Extension (TypeScript, packaged as VSIX)

| Module | Purpose |
|--------|---------|
| `extension.ts` | Activation, identity resolution, command registration |
| `inbox.ts` | SSE inbox + 4-action notification + auto-respond loop |
| `triggers.ts` | File-event subscribers, routing.json matcher, rate-limit + cooldown |
| `routing-config.ts` | Schema + matcher for .qlang/routing.json rules |
| `llm.ts` | 5-provider LLM client (DeepSeek/OpenAI/Groq/Anthropic/Ollama) for auto-respond |
| `qlms-client.ts` | HMAC-signed envelope wrapper around /qlms/v1.1 |
| `lsp.ts` | Wires `qlang-cli lsp` as VS Code language server |

## Settings (extension)

```
qlang.qlms.baseUrl                    = http://localhost:4646
qlang.qlms.authToken                  = ""
qlang.qlms.seedHex                    = ""    (auto-gen + persist)
qlang.qlms.inbox.enabled              = true
qlang.qlms.inbox.identity             = ""    (auto-gen <ide>-<host>-<6hex>)
qlang.qlms.autoRespond.enabled        = false
qlang.qlms.autoRespond.providerType   = deepseek | openai | anthropic | ollama | groq
qlang.qlms.autoRespond.apiKey         = ""
qlang.qlms.autoRespond.baseUrl        = ""
qlang.qlms.autoRespond.model          = "deepseek-chat"
qlang.qlms.autoRespond.systemPrompt   = ""
qlang.qlms.triggers.enabled           = false
qlang.qlms.triggers.debounceMs        = 2000
qlang.qlms.triggers.maxConcurrent     = 3
qlang.qlms.triggers.maxPerHour        = 60
qlang.qlms.triggers.warnOnQuota       = true
qlang.lsp.enabled                     = true
qlang.lsp.path                        = "qlang-cli"
```

## Routing rule schema (`.qlang/routing.json`)

```json
{
  "version": 1,
  "rules": [{
    "id": "rust-on-save-review",
    "trigger": "on_save",
    "when": { "languageId": "rust", "minLines": 20 },
    "send_to": "developer",
    "prompt": "Review for clarity, error handling, unused code.",
    "intent": "Execute",
    "cooldownSec": 30
  }]
}
```

## Security model

- **Bus**: every QLMS frame is HMAC-SHA256 signed; `/deliver` rejects bad signatures.
- **Tools**:
  - read_file/write_file: paths sandboxed to `<QO_DATA_DIR>/workspace`, no `..`, no abs paths, no drive letters, 64KB cap
  - web_fetch: HTTPS only, 10s timeout, 8KB cap, loose HTML strip
  - exec_shell: hard whitelist (ls, pwd, git status, git log -5, version queries), 5s timeout, 4KB stdout cap
- **Triggers**: opt-in (default OFF), per-rule cooldowns, rolling 1h hourly cap with one-time warning.
- **Provider keys**: stored encrypted in redb on server, never returned via /api responses.
- **Auth**: optional QO_AUTH_TOKEN bearer token enforced by middleware.

## Honest limitations (what's NOT done)

| Gap | Severity | Notes |
|-----|----------|-------|
| Real semantic embeddings for consensus_score_semantic | low | Trigram is a meaningful approximation; Ollama/OpenAI embeddings can plug in later |
| Conversation history is browser-localStorage only | medium | No server-side replay endpoint; reload across machines = empty |
| Cockpit "live stream banner" copy is stale | cosmetic | Says "reload clears" but localStorage now persists |
| Tool sandbox is /workspace only | by design | Read_file can't access repo source unless mounted into the workspace dir |
| MCP tool-use is internal format, not actual MCP protocol | medium | `<tool/>` markers, not JSON-RPC. Real MCP integration is a future task |
| No automated test suite for end-to-end flows | medium | Verified via Playwright manually; need CI |
| Frontend production deployment | low | Currently served by qo via ServeDir from frontend/dist |
| Per-IDE provider override via cockpit | low | Today: edit IDE settings.json; cockpit shows but doesn't edit remotely |

## Branches consolidated into main (`NewWayLLMHandling`)

| Branch | Brought | Commit |
|--------|---------|--------|
| `orbitqlang` | DeepSeek + 6 agents + Cockpit redesign | `2aaf109` |
| `multi-ide-mesh` | Presence + Auto-respond + Cross-IDE | `1bbc3fa` |
| `auto-trigger` | File-triggers + cooldowns + 4-action inbox | `89c47fa` |
| `consensus-fanout` | Multi-agent fan-out + Jaccard score | `191d2a7` |
| `pipeline-chains` | Sequential chains + history + IDE detail | `c4a6b9d` |
| `final-mcp-and-merge` | MCP tools + trigram cosine score | `60842d5` |
| **Merge into main** | All of the above | `c506215` |

## Quickstart for a new developer

```bash
# 1. Build qo (Rust, MSYS2 MINGW64)
cd /c/Users/a.b/Graph/OrbitQLang
export PATH="/c/Users/a.b/.cargo/bin:$PATH"
cargo build --bin qo --no-default-features

# 2. Build qlang-cli (LSP)
cargo build --bin qlang-cli --no-default-features
cp target/debug/{qo,qlang-cli}.exe ~/.cargo/bin/

# 3. Build VSIX + install in your IDEs
cd editors/vscode
npx tsc -p .
npx -y @vscode/vsce package --allow-missing-repository
# install via your IDE: drag-drop the .vsix into the Extensions panel

# 4. Run qo
qo --offline       # no DEEPSEEK_API_KEY env needed if you UI-add it later

# 5. Open cockpit
# http://localhost:4646/

# 6. Add DeepSeek via cockpit Profile → Providers → DeepSeek → Add → API key

# 7. (Optional) opt into auto-trigger:
mkdir -p .qlang
cp editors/vscode/src/example-routing.json .qlang/routing.json
# in your IDE: Settings → qlang.qlms.triggers.enabled = true → Reload Window
```
