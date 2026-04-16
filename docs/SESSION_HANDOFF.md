# Session Handoff — A2A-qlang System

**Last updated:** 2026-04-16
**Target remote:** `a2a` → `Mirkulix/A-2A-qlang`
**Working branch:** `claude/sleepy-stonebraker` (worktree) → merged into `NewWayLLMHandling`
**Second remote:** `origin` → `Mirkulix/QlangNeo.git`
**Head commit at handoff:** `1730ba0` — *"Harden LLM provider error handling"*

This document brings a new Claude session (on any machine) fully up to speed on where we are, what was built, what works, and what comes next. Read it start-to-finish before touching code.

---

## 0. How to resume on a new machine

```bash
# 1. Clone
git clone https://github.com/Mirkulix/A-2A-qlang.git
cd A-2A-qlang
git checkout NewWayLLMHandling   # <- current working branch

# 2. Toolchain (Windows dev was primary)
# Rust stable, Node 20+, Python 3.11+, Docker Desktop (optional, only for swarm)
rustup install stable
rustup default stable

# 3. Secrets — NEVER commit this file
cp .env.example .env
# Paste your keys into .env:
#   GROQ_API_KEY=gsk_...
#   DEEPSEEK_API_KEY=sk-...
#   CLOUD_BASE_URL=https://api.deepseek.com/v1
#   CLOUD_MODEL=deepseek-chat
#   TAVILY_API_KEY=tvly-dev-...
#   QO_NODE_ADDR=localhost:4646
#   PEER_DISCOVERY_SEEDS=localhost:4646

# 4. Run
cargo build --workspace --release          # ~5-10min first time
cargo run -p qo-server                     # backend on :4646
cd frontend && npm install && npm run dev  # frontend on :5173
```

Dashboard lives at `http://localhost:5173`, backend at `http://localhost:4646`.

**Previous keys are in the local `.env` on the original Windows dev box (`C:\Users\a.b\Graph\qlang\.claude\worktrees\sleepy-stonebraker\.env`) — these are gitignored so the new machine needs fresh keys (Groq/DeepSeek/Tavily all have free tiers).**

---

## 1. What this project is

**A2A-qlang** = binary, cryptographically signed **agent-to-agent protocol** designed to replace JSON-over-HTTP in agent swarms. Key properties:

- QLMS envelope (magic `0x51 0x4C 0x4D 0x53`, HMAC-SHA256 signed, replay-guarded)
- IGQK ternary weights + federated majority-vote aggregation
- Native `.qlg` binary payloads (≥2.3× smaller, ≥2× faster than JSON)
- 5-Werte Guardian (Integrity / Autonomy / Care / Truth / Justice)
- Bidirectional MCP bridge (JSON-RPC 2.0 ↔ QLMS graphs)
- Multi-agent orchestration (CEO · Researcher · Developer · Guardian · Strategist · Artisan) with per-role LLM model binding
- Dashboard: React 19 + Vite 6, light-first "Clarity" design system (Indigo `#5B5BF0` accent)

The goal is **AAIF submission** (Linux Foundation Agentic AI Foundation) as the reference open-source A2A protocol.

---

## 2. State of the 5-phase PRD plan

Master plan: [.claude/plans/expressive-herding-globe.md](.claude/plans/expressive-herding-globe.md)

| Phase | Scope | Status |
|---|---|---|
| **M1 AAIF readiness** | Conformance-tests, constant-time HMAC, Python reference parser | ✅ done |
| **M2 MCP bridge** | Bidirectional MCP↔QLMS, base64 deliver/reply routes | ✅ done |
| **M3 Native binary + runtime gaps** | `.qlg` payload replaces JSON-in-envelope (FLAG_NATIVE_BINARY=0x0002), Scan/SubGraph opcodes | ✅ done |
| **M4 Swarm scale-out** | Peer discovery, docker-compose.swarm.yml, soak test | ⚠️ code-complete, not stress-tested |
| **M5 VSCode extension** | QLMS-signed agent calls from the IDE | ✅ status-bar + client done |
| **Epic 6 Dashboard** | Mission Control, Graph Inspector, Werte-Radar, Swarm Map + home, chat, workspace, goals | ✅ done (see §5) |

All 14 task codes (1.1–6.4) are landed. 23 commits on `claude/sleepy-stonebraker`, all pushed to both remotes.

---

## 3. Key components & where they live

### Rust workspace (`cargo` members)

| Crate / Path | Purpose |
|---|---|
| [crates/qlang-core](crates/qlang-core) | Primitives — crypto, replay-guard, graph, serial, tensor |
| [crates/qlang-runtime](crates/qlang-runtime) | Execution — ops, federation majority vote, `mcp_bridge.rs` |
| [crates/qlang-agent](crates/qlang-agent) | Protocol envelope — `protocol.rs` with `FLAG_NATIVE_BINARY` |
| [qo/qo-server](qo/qo-server) | Axum HTTP backend, all routes in `qo/qo-server/src/routes/` |
| [qo/qo-agents](qo/qo-agents) | Multi-agent orchestration, tools, MCP client |
| [qo/qo-llm](qo/qo-llm) | `router.rs` (3-tier + fallback), `cloud.rs` (DeepSeek-compatible), Groq client |
| [qo/qo-values](qo/qo-values) | 5-Werte scorer (Guardian) |
| [qo/qo-memory](qo/qo-memory) | Graph store + goal store |
| [bindings/python/qlms_parser](bindings/python/qlms_parser) | Stdlib-only Python reference parser + golden tests |
| [editors/vscode](editors/vscode) | VSCode extension w/ QLMS status-bar indicator |

### Frontend (`frontend/src/`)

Primary tabs: Home · Chat · Live (Mission Control) · Workspace · Ziele (Goals Browser) · Verlauf (Graph Inspector) · Netzwerk (Swarm Map) · Werte-Radar.

Advanced (collapsed): Neo Shell, Agenten, Goals, QLANG-Editor, Messages, Provider, Evolution, Training, GPU-Training, Spiking, Organismus, Bewusstsein, Knowledge-3D, Historie.

**Design system:** [frontend/src/styles.css](frontend/src/styles.css) — "Clarity" light-first, ~3000 lines, CSS custom properties for theme swap.

### Key backend routes

```
POST /qlms/v1.1/deliver       base64-framed envelope, replay+signature check, then graph execute
POST /qlms/v1.1/reply         reply handshake
POST /api/chat                chat with Decision-Trace + force_intent override
POST /api/goals               create goal
POST /api/goals/{id}/continue multi-turn continuation
GET  /api/values              current Werte snapshot
GET  /api/federation/peers    peer list (for Swarm Map)
GET  /api/federation/stats    gossip_rate, bandwidth_saved_pct
GET  /ws/graph-stream         WebSocket, broadcasts GraphMessage events live
POST /mcp/v1                  JSON-RPC MCP server (exposes qlang_research / qlang_run_goal / qlang_read_workspace_file)
POST /api/workspace/exec      sandbox code execution (Python/Node/Bash/TS, 10s timeout)
GET  /api/workspace/files     workspace tree
```

See [qo/qo-server/src/lib.rs](qo/qo-server/src/lib.rs) for the full router.

---

## 4. LLM routing (3-tier)

Router: [qo/qo-llm/src/router.rs](qo/qo-llm/src/router.rs)

| Tier | Provider | Use | Notes |
|---|---|---|---|
| **Fast** | Groq (Llama-3.3-70B) | Simple, high-throughput | Free tier, ~500ms |
| **Heavy** | DeepSeek-Chat | Reasoning, architecture, security | OpenAI-compat API |
| **Local** | Ollama (optional) | Offline fallback | Not required |

**Per-agent binding:** each `AgentRole` in [qo/qo-agents/src/agent.rs](qo/qo-agents/src/agent.rs) has a `preferred_provider()` — Researcher→Groq (fast), Developer→DeepSeek (quality), Guardian→Groq (speed on policy), CEO→DeepSeek (decomposition).

**Fallback logic:** `chat_preferring(tier, messages)` in router.rs tries the preferred tier, silently falls back to auto-routing on failure. This was added to fix the *"CEO ✗ Fehler bei Dekomposition: error decoding response body"* bug.

**Error hardening:** [qo/qo-llm/src/cloud.rs](qo/qo-llm/src/cloud.rs) reads body as text first, checks HTTP status, detects `{error:{message:...}}` JSON envelopes, emits clear tracing diagnostics.

---

## 5. Autonomy & transparency stack

### Decision-Trace
Every chat response carries a `DecisionTrace` object ([qo/qo-server/src/routes/chat.rs](qo/qo-server/src/routes/chat.rs:DecisionTrace)):

```rust
pub struct DecisionTrace {
    pub intent: String,            // "chat" | "goal" | "research"
    pub intent_confidence: f32,
    pub intent_forced: bool,       // true when UI toggle overrode classifier
    pub path: String,              // "direct_llm" | "research_then_llm" | "dispatch_to_swarm"
    pub tier: Option<String>,      // "fast" | "heavy" | "local"
    pub goal_id: Option<u64>,
    pub tokens_estimated: u64,
    pub duration_ms: u64,
    pub tools_used: Vec<String>,   // ["tavily", "wikipedia", "firecrawl", ...]
}
```

Frontend renders it below every assistant message ([frontend/src/ChatView.tsx](frontend/src/ChatView.tsx)).

### Goal-Toggle
Home hero and Chat input both expose "Als Ziel an den Schwarm senden" — when on, POST body sets `force_intent: "goal"` which bypasses the classifier and hands the prompt straight to CEO for decomposition. Required because the classifier was sometimes routing what the user *meant* as a goal to simple chat.

### Multi-turn goals
`POST /api/goals/{id}/continue` with `{prompt}` reuses the goal's existing graph + agent state. Up to `MAX_RETRIES=2` automatic rounds on failure. Browser UI: [frontend/src/GoalsBrowserView.tsx](frontend/src/GoalsBrowserView.tsx).

---

## 6. Tools & integrations

### Web search (parallel multi-source)
[qo/qo-agents/src/tools.rs](qo/qo-agents/src/tools.rs) `tool_web_search`:

1. **Tavily** (primary, needs `TAVILY_API_KEY`)
2. **Wikipedia** (always on, no key)
3. **SearXNG** (optional, self-hosted — set `SEARXNG_URL`)
4. **DuckDuckGo HTML** (fallback)

All four fire in parallel, results merged and deduped by URL.

### URL readers
`tool_fetch_url`:

1. **Firecrawl** (primary, needs `FIRECRAWL_API_KEY`)
2. **Jina Reader** (fallback, free — `https://r.jina.ai/<url>`)

Researcher auto-fetches the first HTTP URL in its reasoning output via `extract_first_http_url` in [qo/qo-agents/src/llm_node.rs](qo/qo-agents/src/llm_node.rs).

### MCP (Model Context Protocol)
- **As MCP server:** [qo/qo-server/src/routes/mcp_server.rs](qo/qo-server/src/routes/mcp_server.rs) → exposes `qlang_research`, `qlang_run_goal`, `qlang_read_workspace_file` at `POST /mcp/v1`
- **As MCP client:** [qo/qo-agents/src/mcp_client.rs](qo/qo-agents/src/mcp_client.rs) → `tool_mcp_call` lets agents call any external MCP server

### Workspace sandbox
Agents emit `<qo:file path="relative/path.ext">…content…</qo:file>` blocks; [qo/qo-agents/src/extract_artifacts.rs](qo/qo-agents/src/extract_artifacts.rs) parses and writes them under `data/workspace/` (gitignored). Frontend [frontend/src/WorkspaceView.tsx](frontend/src/WorkspaceView.tsx) shows the tree, file preview, Run button (executes via `/api/workspace/exec`, Python / Node / Bash / TypeScript supported, 10s wall-clock timeout), and a "VSCode öffnen" link.

---

## 7. Federation / Swarm Map

- **Current runtime:** single node at `localhost:4646` (set via `.env`'s `QO_NODE_ADDR` + `PEER_DISCOVERY_SEEDS`).
- `/api/federation/peers` returns `{total_nodes:1, local_nodes:1, peers:[{id:"localhost", addr:"localhost:4646", is_local:true}]}`.
- **To run a real multi-node swarm** (user asked for this right before handoff — pending action): either
  - **Option A (no Docker):** start 2 more qo-server processes on `:4647` + `:4648` with their own data dirs and `PEER_DISCOVERY_SEEDS=localhost:4646,localhost:4647,localhost:4648`.
  - **Option B (Docker):** `docker compose -f docker-compose.swarm.yml up --build` — brings up 3 services on `qo-1:4646`, `qo-2:4747`, `qo-3:4848` with shared `swarm-net`.
- Gossip loop in [qo/qo-server/src/peer_discovery.rs](qo/qo-server/src/peer_discovery.rs) pings seeds every 10s.

---

## 8. Dashboard UX cheatsheet

| Tab | File | What it shows |
|---|---|---|
| Home | [Home.tsx](frontend/src/Home.tsx) | Hero prompt with "Schwarm ausführen" toggle, live stats bento |
| Chat | [ChatView.tsx](frontend/src/ChatView.tsx) | Markdown-rendered conversation with Decision-Trace + attachments |
| Live | [MissionControl.tsx](frontend/src/MissionControl.tsx) | Live agent DAG (@xyflow/react) + roster with model chips |
| Workspace | [WorkspaceView.tsx](frontend/src/WorkspaceView.tsx) | File tree, preview, Run, open-in-VSCode |
| Ziele | [GoalsBrowserView.tsx](frontend/src/GoalsBrowserView.tsx) | Split-pane goals browser + "Weitermachen" textarea |
| Verlauf | [GraphInspectorView.tsx](frontend/src/GraphInspectorView.tsx) | Pagination through recent GraphMessages, replay button |
| Netzwerk | [SwarmMap.tsx](frontend/src/SwarmMap.tsx) | 3D force-graph of peers (react-force-graph-3d) |
| Werte-Radar | [ValuesRadar.tsx](frontend/src/ValuesRadar.tsx) | Live SVG radar of the 5 Werte, polls `/api/values` every 5s |

Header chips: Werte status · QLMS health · Cost badge (per-session $ + provider popover) · Auth token panel · Theme toggle.

---

## 9. Security & privacy

- `.env` is **gitignored**. The Windows dev box holds real keys — they are NEVER in any commit. If a key ever ends up in chat history or code, rotate it.
- Keys were rotated during this session chain — current Groq + DeepSeek + Tavily keys live only in the local `.env`.
- HMAC constant-time compare: [crates/qlang-core/src/crypto.rs:174,187](crates/qlang-core/src/crypto.rs:174) — verified by `crates/qlang-core/tests/crypto_timing.rs` timing-ratio test.
- Replay guard: [crates/qlang-core/src/replay_guard.rs](crates/qlang-core/src/replay_guard.rs) — nonce + timestamp window.
- Values-Guardian gates destructive actions; integrity/autonomy/care/truth/justice are scored per action, hits under threshold raise alerts on the Werte-Radar.

---

## 10. Known limits / open items

1. **Multi-node swarm not yet launched.** Code is ready, user said "ja mach das" to Option A (3 local qo processes) right before handoff. Next Claude should start 2 more qo-server instances on `:4647` + `:4648` with distinct data dirs and chained peer-seeds, then verify Swarm Map shows 3 balls.
2. **Task 3.2 opcodes** — Scan + SubGraph shipped, but no exhaustive numpy/torch ground-truth coverage yet. Good follow-up.
3. **Federation soak test** — [tests/federation_soak.rs](tests/federation_soak.rs) runs, but has not been executed for a full 10-min run under load on this branch.
4. **VSCode extension** — status-bar indicator and QLMS client work; the MultiAgentCoder flow needs end-to-end validation (capture a Wireshark dump to confirm QLMS frames on the wire).
5. **Graph Inspector replay** — POSTs to `/qlms/v1.1/deliver` but UI doesn't yet surface the resulting reply body nicely.
6. **i18n** — UI is deutsch-first with englisch fallbacks; no proper i18n framework.

---

## 11. Recent commit trail (most recent first)

```
1730ba0 Harden LLM provider error handling — "error decoding response body" bug fixed
3021005 5-agent parallel swarm: MCP (in+out) + multi-turn goals + Goals browser + Cost badge + Auth panel
008e572 Transparency + autonomy sprint — built via 3-agent parallel swarm
b238083 Firecrawl + SearXNG + Jina Reader — Researcher gets real tools
a292984 Wikipedia alongside Tavily — key-free second source
1018bd5 Researcher auto-runs web search before every LLM call
1eba963 Live activity feed in the chat while responses generate
be4b663 Code execution (▶ Run) + goal multi-round retry loop
b1e0ee4 Per-agent model binding — each role gets its best LLM
ef2efce Workspace bridge: agents write files → frontend → VSCode
d34b1cd Chat v2: Markdown rendering + attachments + cleaner layout
f01da1e Flesh out Mission Control + Swarm Map
a89fd99 Wire Home hero prompt → ChatView so submits actually run
bcb78dd Go live: Mission Control publisher + project-root .env loader
770398f Redesign dashboard: light-first 'Clarity' system (2026)
f552505 Close remaining tasks: Scan/SubGraph, soak, Swarm Map, VSCode QLMS
bf3e3ea Add Graph Inspector dashboard tab (Task 6.2)
832afa8 Add 3-node QO swarm: peer discovery + docker-compose.swarm.yml
1873d44 Add Mission Control + Werte-Radar dashboard tabs
878cb9a Add dashboard back-end prereqs: /api/values + /ws/graph-stream
```

Full log: `git log --oneline`.

---

## 12. Conversation arc (high-level narrative)

The human user was operating on Windows, German-speaking, preferring terse "ja mach das!" affirmations. Journey in order:

1. **PRD ingestion** — user dropped PRD/UX/tasks markdown from Downloads, asked for analysis + plan for all 14 tasks on `a2a` remote targeting `NewWayLLMHandling` branch.
2. **Phase 1-5 execution** — conformance tests, MCP bridge, native binary payload, peer discovery, docker swarm, VSCode extension all landed as a sequence of focused commits.
3. **Dashboard overhaul** — rejected the initial design, asked for "2026 design, modern und cool und hell". Full rewrite of `styles.css` into the light-first Clarity system (Indigo accent).
4. **Autonomy dissatisfaction** — noticed Home prompt didn't trigger agents; insisted on real LLM backing. Provided Groq + DeepSeek keys. Led to `.env` loader covering project-root + per-agent model binding.
5. **Tooling expansion** — asked for real web search. Added Tavily key, then demanded a second source (Wikipedia), then OSS alternative to Apify (Firecrawl + SearXNG + Jina Reader).
6. **Transparency sprint** — "ist das System nur demo?" → "wer entscheidet das?" → built Decision-Trace + Goal-Toggle so the user can see intent classification and override it.
7. **Agent-swarm parallelism** — "bau das alles mit agenten-swarm und werde schneller" → general-purpose agents split into strict non-overlapping file assignments to avoid merge conflicts.
8. **Bug fixes** — the final two commits fixed "CEO Fehler bei Dekomposition" (CloudClient error hardening) and the empty Swarm Map (local peer discovery).
9. **Handoff** — user wants the chat saved as MD for resumption on a different machine + everything pushed to GitHub.

---

## 13. Anti-patterns to avoid (learned the hard way)

- **Do NOT** add `skip_serializing_if` to `GraphMessage` fields — bincode breaks.
- **Do NOT** pass absolute paths to the workspace sandbox — `strip_workspace_prefix` expects prefixes like `workspace/`, `./workspace/`, `/workspace/` and normalises them; double-prefix produces `data\workspace\data\workspace\hello.py`.
- **Do NOT** use Unicode glyphs (✓ ✗) in Python scripts run from Windows — cp1252 crashes. Use `PASS` / `FAIL`.
- **Do NOT** commit `data/workspace/` — it's generated artefact, already in `.gitignore`.
- **Do NOT** use `Op::Custom` in the graph — it doesn't exist. `Manifold::Custom` is the enum that has Custom. For unknown ops use `Op::SubGraph { graph_id }`.
- **Do NOT** batch-kill qo-server with PowerShell `$_.Id` syntax inside bash — it's misparsed; kill by explicit PID.

---

## 14. What the next Claude should do first

1. Read this file end-to-end.
2. `git status` — should be clean, branch `NewWayLLMHandling` synced with `a2a/NewWayLLMHandling`.
3. Ensure `.env` has the three keys (Groq / DeepSeek / Tavily) and the two peer vars (`QO_NODE_ADDR`, `PEER_DISCOVERY_SEEDS`).
4. `cargo run -p qo-server` + `cd frontend && npm run dev`, then open `http://localhost:5173`.
5. Decide with the user: start 2 more qo-server processes on `:4647` + `:4648` to populate the Swarm Map with real peers, OR move on to the next feature the user asks for.

Good luck. The system is in a good, coherent state — build on it, don't tear it down.
