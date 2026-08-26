pub mod agent_models;
pub mod api_keys;
pub mod auth;
pub mod config;
pub mod git_ops;
pub mod mesh_history;
pub mod peer_discovery;
pub mod repository_indexer;
pub mod rate_limit;
pub mod routes;
pub mod tools;
use axum::{
    middleware,
    routing::{delete, get, post, put},
    Router,
};
use qlang_agent::bus::MessageBus;
use qo_agents::{AgentRegistry, AgentRole};

use qo_llm::LlmRouter;
use qo_memory::{GraphStore, MemoryContext, ObsidianBridge, Store};
use qo_values::ValueScores;
use tokio::sync::broadcast;

use crate::peer_discovery::FederationStatsHandle;
use crate::routes::dashboard::GraphEvent;
use serde::Serialize;
use std::collections::{HashMap, VecDeque};
use std::sync::atomic::AtomicBool;
use std::sync::Arc;
use tokio::sync::Mutex;
use tower_http::cors::CorsLayer;
use tower_http::services::{ServeDir, ServeFile};

/// Snapshot of a single bus message stored in the server-side ring buffer.
/// Mirrors the shape emitted by `/api/messages/stream` so the cockpit can
/// hydrate its liveTail from `/api/messages/recent` without a separate
/// adapter. `content` is capped at 4 KB (suffix-elided with `…`).
#[derive(Debug, Clone, Serialize)]
pub struct RecentMessage {
    pub id: u64,
    pub from: String,
    pub to: String,
    pub intent: String,
    pub graph_name: String,
    pub timestamp: u64,
    pub content: String,
    pub is_reply: bool,
    pub auto_triggered: bool,
    pub trigger_kind: String,
}

pub struct AppState {
    pub llm: Arc<LlmRouter>,
    pub store: Store,
    pub graph_store: GraphStore,
    /// Durable, checkable knowledge layer: claims with provenance and
    /// evidence. Shares the same redb database as `store`.
    pub knowledge: qo_knowledge::KnowledgeStore,
    /// Repository boundary for deterministic knowledge ingestion. Network
    /// callers never choose this path, preventing arbitrary filesystem scans.
    pub workspace_root: std::path::PathBuf,
    /// Where manual knowledge backups are written (`data/backups` by default).
    /// The schedule that triggers them is an operator decision, not the server.
    pub backup_dir: std::path::PathBuf,
    pub llm_routing: config::LlmRoutingConfig,

    pub configured_providers: Mutex<Vec<qo_llm::ProviderConfig>>,
    pub memory: Mutex<MemoryContext>,
    /// QLANG Message Bus — routes GraphMessages between AI agents.
    pub message_bus: Arc<MessageBus>,

    pub obsidian: ObsidianBridge,
    pub agents: Mutex<AgentRegistry>,
    pub supervisor_daemon: Mutex<routes::supervisor::SupervisorDaemonState>,
    pub live_supervisor_sessions: Mutex<HashMap<u64, Arc<routes::supervisor::LiveSessionHandle>>>,
    // --- Dashboard prerequisites (PRD Epic 6) ---
    /// Current 5-Werte scores. Mutated by Guardian agent decisions, read
    /// by `/api/values` for the frontend Werte-Radar (Task 6.3).
    pub values: Mutex<ValueScores>,
    /// Broadcast channel that fans out `GraphEvent`s to every
    /// `/ws/graph-stream` WebSocket subscriber (Task 6.1 Mission Control).
    pub graph_events_tx: broadcast::Sender<GraphEvent>,
    /// Peer-discovery gossip statistics. Populated by the background
    /// task (Task 4.2), read by `/api/federation/stats` (Task 6.4).
    pub gossip_stats: FederationStatsHandle,
    /// IDE/agent presence registry. Ephemeral, in-memory only — a `qo`
    /// restart wipes it so dead clients aren't resurrected from disk.
    /// Mutated by `/api/presence/*` handlers; swept by a background
    /// task that removes expired entries every 30s.
    pub presence: Mutex<HashMap<String, routes::presence::PresenceEntry>>,
    /// Bounded ring buffer of recent bus messages (cap 200) — populated by
    /// a background task that subscribes to `message_bus`. Read by
    /// `/api/messages/recent` for cross-machine cockpit hydration.
    pub recent_messages: Mutex<VecDeque<RecentMessage>>,
    /// Live swarm state, keyed by swarm id. Inserted by
    /// `POST /api/swarm/start`, mutated by the background orchestrator
    /// task, read by `/api/swarm/{id}` and `/api/swarm/active`. Bounded
    /// only by user behavior — no automatic eviction yet (each entry is
    /// ~a few KB so this is fine for the initial demo).
    pub swarms:
        Arc<tokio::sync::RwLock<std::collections::HashMap<u64, routes::swarm::SwarmState>>>,
    /// Autonomous swarm scheduler state. Mutated by `/api/autonomous/*`
    /// handlers and the single global scheduler task spawned at first
    /// `/api/autonomous/start`.
    pub autonomous: Arc<tokio::sync::RwLock<routes::autonomous::AutonomousState>>,
    /// Idempotency guard for the autonomous scheduler — flipped to `true`
    /// the first time `/api/autonomous/start` spawns the loop. Subsequent
    /// `/start` calls just update the config without spawning a second
    /// task.
    pub autonomous_loop_started: Arc<AtomicBool>,
    /// Multi-agent run snapshots, keyed by run id. Populated by
    /// `/api/multi-agent/run` and `/api/multi-agent/runs/start`,
    /// updated while runs execute, listed by `/api/multi-agent/runs`,
    /// read by `/api/multi-agent/runs/{id}`.
    pub multi_agent_runs: Arc<
        tokio::sync::RwLock<std::collections::HashMap<u64, routes::multi_agent::StoredMultiAgentRun>>,
    >,
    /// Broadcast channel for live multi-agent run snapshots. Consumed by
    /// `/api/multi-agent/stream` so the cockpit can update without polling.
    pub multi_agent_events_tx: broadcast::Sender<routes::multi_agent::MultiAgentRunEvent>,
    /// Bounded log of recent graph-delta merges, newest last. The merge
    /// itself is durable in redb; this keeps the *reports* — what applied,
    /// what conflicted and why — so the cockpit can show a delta feed and a
    /// conflict view without re-deriving them from claim history.
    pub delta_log: Arc<tokio::sync::RwLock<VecDeque<routes::knowledge_tools::DeltaLogEntry>>>,
    /// Producer keys allowed to write graph deltas, loaded from
    /// `.qlang/trusted_delta_producers.json`. Empty means nobody is trusted,
    /// which is why an untokened instance also refuses remote submissions
    /// rather than accepting anonymous ones.
    pub delta_trust: qo_knowledge::TrustStore,
    /// Per-seat API keys, loaded from `.qlang/api_keys.json`. The multi-user
    /// half of auth: an admin issues one key per person and can revoke them
    /// individually, instead of everyone sharing one `QO_AUTH_TOKEN`.
    pub api_keys: api_keys::ApiKeyStore,
}

pub struct QoConfig {
    pub port: u16,
    pub groq_api_key: Option<String>,
    /// (api_key, base_url, model) for a custom cloud LLM
    pub cloud_config: Option<(String, String, String)>,
    /// Ollama base URL for Tier 1 local inference (e.g. "http://localhost:11434")
    pub ollama_url: Option<String>,
    /// Ollama model name (e.g. "orbit-companion-ft-q4")
    pub ollama_model: Option<String>,
    pub data_dir: std::path::PathBuf,
    /// The only repository the local QO instance may index.
    pub workspace_root: std::path::PathBuf,
    /// Where the per-seat API-key store is read from at startup.
    pub api_keys_path: std::path::PathBuf,
    pub obsidian_vault: std::path::PathBuf,
    pub static_dir: Option<std::path::PathBuf>,
    /// Optional API token for bearer auth (reads QO_AUTH_TOKEN from env if None)
    pub auth_token: Option<String>,
    /// Origins allowed to make cross-origin requests (CORS). Empty means the
    /// browser's same-origin policy applies and no cross-origin requests are
    /// allowed — the cockpit is served same-origin by qo, so this is the safe
    /// default. Set to a comma-separated list of origins to open it up for an
    /// IDE webview or a hosted frontend.
    pub cors_origins: Vec<String>,
    /// Hard cap on a buffered request body, in bytes (DoS bound).
    pub max_body_bytes: usize,
    /// Per-IP request rate limit (requests per second, with `rate_burst_size`
    /// burst).
    pub rate_per_second: u32,
    pub rate_burst_size: u32,
    pub llm_routing: config::LlmRoutingConfig,
}

impl Default for QoConfig {
    fn default() -> Self {
        Self {
            port: 3000,
            groq_api_key: None,
            cloud_config: None,
            ollama_url: None,
            ollama_model: None,
            data_dir: std::path::PathBuf::from("data"),
            workspace_root: std::path::PathBuf::from("."),
            api_keys_path: std::path::PathBuf::from(".qlang/api_keys.json"),
            obsidian_vault: std::path::PathBuf::from("vault"),
            static_dir: None,
            auth_token: None,
            cors_origins: Vec::new(),
            max_body_bytes: 16 * 1024 * 1024,
            rate_per_second: 50,
            rate_burst_size: 200,
            llm_routing: config::LlmRoutingConfig::default(),
        }
    }
}

/// Load the producer keys allowed to submit graph deltas.
///
/// A missing or malformed file yields an empty store, which trusts nobody.
/// That is deliberate: the failure mode of a typo in this file must be
/// "submissions are refused", never "submissions are accepted unchecked".
fn load_delta_trust(path: &str) -> qo_knowledge::TrustStore {
    let Ok(contents) = std::fs::read_to_string(path) else {
        tracing::info!(
            path,
            "no delta trust store — remote delta submissions will be refused"
        );
        return qo_knowledge::TrustStore::new();
    };
    match qo_knowledge::TrustStore::from_json(&contents) {
        Ok(store) => {
            tracing::info!(path, producers = store.producers.len(), "delta trust store loaded");
            store
        }
        Err(error) => {
            tracing::error!(path, %error, "delta trust store is malformed — trusting nobody");
            qo_knowledge::TrustStore::new()
        }
    }
}

/// Load the per-seat API keys. A missing file means no seats — the server
/// then relies on `QO_AUTH_TOKEN` alone, exactly as before this existed. A
/// malformed file loads empty and logs, so a typo cannot silently grant
/// access.
fn load_api_keys(path: &std::path::Path) -> api_keys::ApiKeyStore {
    let Ok(contents) = std::fs::read_to_string(path) else {
        return api_keys::ApiKeyStore::new();
    };
    match api_keys::ApiKeyStore::from_json(&contents) {
        Ok(store) => {
            tracing::info!(path = %path.display(), seats = store.active_seats(), "API-key store loaded");
            store
        }
        Err(error) => {
            tracing::error!(path = %path.display(), %error, "API-key store is malformed — no seats loaded");
            api_keys::ApiKeyStore::new()
        }
    }
}

pub async fn build_app(
    config: QoConfig,
) -> Result<(Router, Arc<AppState>), Box<dyn std::error::Error + Send + Sync>> {
    let db_path = config.data_dir.join("qo.redb");
    // Ensure the data directory exists
    std::fs::create_dir_all(&config.data_dir)?;

    let store = Store::open(&db_path)?;
    let graph_store = GraphStore::new(store.db())?;
    let knowledge = qo_knowledge::KnowledgeStore::from_db(store.db())?;
    let ollama_config = match (config.ollama_url, config.ollama_model) {
        (Some(url), Some(model)) => Some((url, model)),
        _ => None,
    };
    let llm = Arc::new(LlmRouter::new(config.groq_api_key, config.cloud_config, ollama_config));
    let obsidian = ObsidianBridge::new(config.obsidian_vault);

    // Load persisted data BEFORE creating AppState (no async runtime yet)
    let mut agents_reg = AgentRegistry::new();

    // Restore goals
    if let Ok(goals) = store.list_goals() {
        for (_, json) in goals {
            if let Ok(goal) = serde_json::from_str::<qo_agents::Goal>(&json) {
                agents_reg.restore_goal(goal);
            }
        }
    }

    // Restore agent stats
    if let Ok(agent_stats) = store.load_agent_stats() {
        for (role_str, json) in agent_stats {
            let role = match role_str.as_str() {
                "Ceo" => Some(AgentRole::Ceo),
                "Researcher" => Some(AgentRole::Researcher),
                "Developer" => Some(AgentRole::Developer),
                "Guardian" => Some(AgentRole::Guardian),
                "Strategist" => Some(AgentRole::Strategist),
                "Artisan" => Some(AgentRole::Artisan),
                _ => None,
            };
            if let Some(role) = role {
                #[derive(serde::Deserialize)]
                struct Stats { tasks_completed: u32, tasks_failed: u32 }
                if let Ok(stats) = serde_json::from_str::<Stats>(&json) {
                    agents_reg.restore_agent_stats(role, stats.tasks_completed, stats.tasks_failed);
                }
            }
        }
    }

    // Load persisted embeddings into vector store for long-term memory
    let mut memory_ctx = MemoryContext::new(384);
    memory_ctx.load_from_store(&store);
    tracing::info!("Loaded {} memories from vector store", memory_ctx.count());

    // Load configured providers from redb so they are available for routing on startup
    let mut configured_providers = Vec::new();
    if let Ok(providers) = store.list_providers() {
        for (_, json) in providers {
            if let Ok(cfg) = serde_json::from_str::<qo_llm::ProviderConfig>(&json) {
                if cfg.enabled {
                    configured_providers.push(cfg);
                }
            }
        }
    }
    tracing::info!(
        "Loaded {} configured providers from store",
        configured_providers.len()
    );

    // Inject persisted providers into the live LlmRouter so a UI-added
    // key (e.g. DeepSeek) survives a restart of `qo --offline` even
    // when no DEEPSEEK_API_KEY env var is set.
    for cfg in &configured_providers {
        if let Err(e) = llm
            .install_provider(
                cfg.provider_type_str(),
                cfg.api_key.clone(),
                cfg.base_url.clone(),
                Some(cfg.model.clone()),
            )
            .await
        {
            tracing::warn!(
                "startup: provider {} (type {}) not hot-reloaded: {}",
                cfg.id,
                cfg.provider_type_str(),
                e
            );
        }
    }

    tracing::info!("Restored: {} goals",
        agents_reg.list_goals().len(),
    );

    // Initialize the QLANG Message Bus for AI-to-AI communication
    let message_bus = MessageBus::new();

    let state = Arc::new(AppState {
        llm,
        store,
        graph_store,
        knowledge,
        workspace_root: config.workspace_root,
        backup_dir: config.data_dir.join("backups"),
        llm_routing: config.llm_routing,
        obsidian,
        agents: Mutex::new(agents_reg),
        configured_providers: Mutex::new(configured_providers),
        memory: Mutex::new(memory_ctx),
        message_bus: message_bus.clone(),
        supervisor_daemon: Mutex::new(routes::supervisor::SupervisorDaemonState::default()),
        live_supervisor_sessions: Mutex::new(HashMap::new()),
        values: Mutex::new(ValueScores::default()),
        // Channel capacity 256: plenty of headroom for a single-agent
        // demo; subscribers that lag beyond this get a `Lagged` notice
        // so they can show a "catching up" indicator.
        graph_events_tx: broadcast::channel::<GraphEvent>(256).0,
        gossip_stats: peer_discovery::new_stats_handle(std::time::Duration::from_secs(10)),
        presence: Mutex::new(HashMap::new()),
        recent_messages: Mutex::new(VecDeque::new()),
        swarms: Arc::new(tokio::sync::RwLock::new(HashMap::new())),
        autonomous: Arc::new(tokio::sync::RwLock::new(
            routes::autonomous::AutonomousState::default(),
        )),
        autonomous_loop_started: Arc::new(AtomicBool::new(false)),
        multi_agent_runs: Arc::new(tokio::sync::RwLock::new(HashMap::new())),
        multi_agent_events_tx: broadcast::channel::<routes::multi_agent::MultiAgentRunEvent>(256).0,
        delta_log: Arc::new(tokio::sync::RwLock::new(VecDeque::new())),
        delta_trust: load_delta_trust(".qlang/trusted_delta_producers.json"),
        api_keys: load_api_keys(&config.api_keys_path),
    });

    // Background drain: subscribe to the bus and append every message to
    // the bounded ring (cap 200). Lets `/api/messages/recent` hydrate the
    // cockpit on a fresh machine where localStorage is empty. The task
    // exits only when the bus is dropped (i.e. process shutdown).
    {
        let recent_state = state.clone();
        tokio::spawn(async move {
            let mut rx = recent_state.message_bus.subscribe().await;
            while let Some(msg) = rx.recv().await {
                let intent = format!("{:?}", msg.intent);
                let is_reply = intent.starts_with("Result");
                let content = msg
                    .graph
                    .metadata
                    .get("content")
                    .map(|c| {
                        if c.len() > 4096 {
                            format!("{}…", &c[..4096])
                        } else {
                            c.clone()
                        }
                    })
                    .unwrap_or_default();
                let auto_triggered = msg
                    .graph
                    .metadata
                    .get("auto_triggered")
                    .map(|v| v == "true" || v == "1")
                    .unwrap_or(false);
                let trigger_kind = msg
                    .graph
                    .metadata
                    .get("trigger_kind")
                    .cloned()
                    .unwrap_or_default();
                let entry = RecentMessage {
                    id: msg.id,
                    from: msg.from.name.clone(),
                    to: msg.to.name.clone(),
                    intent,
                    graph_name: msg.graph.id.clone(),
                    timestamp: std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .unwrap_or_default()
                        .as_secs(),
                    content,
                    is_reply,
                    auto_triggered,
                    trigger_kind,
                };
                let mut buf = recent_state.recent_messages.lock().await;
                buf.push_back(entry);
                while buf.len() > 200 {
                    buf.pop_front();
                }
            }
        });
    }

    // Spawn the presence sweeper — evicts expired IDE/agent entries
    // every 30s. Runs for the life of the process.
    {
        let sweeper_state = state.clone();
        tokio::spawn(async move {
            routes::presence::sweeper_loop(sweeper_state).await;
        });
    }

    // Register all QO agents on the message bus and wire each one to an LLM.
    //
    // Each agent runs its own background task that:
    //   1. Drains its mailbox.
    //   2. Extracts the user prompt (file content / chat text) from the message.
    //   3. Calls the LLM router with a role-specific system prompt.
    //   4. Builds a reply GraphMessage with the LLM response in graph.metadata.
    //   5. Sends the reply back via the bus, addressed to the original sender.
    //
    // Dashboard fanout (graph_events_tx) is kept for the cockpit's edge animation.
    {
        use qlang_agent::protocol::{AgentId, Capability, MessageIntent};
        use qlang_core::graph::Graph;
        use std::collections::HashMap as StdHashMap;

        fn system_prompt_for(role: &str) -> &'static str {
            match role {
                "ceo"        => "You are CEO, a coordinator agent. Decompose the user's request into clear steps, suggest which specialist should handle each step (developer, researcher, guardian, strategist, artisan), and give a one-paragraph executive summary.",
                "developer"  => "You are Developer, a senior software engineer. Review code, suggest refactors, write functions, and explain trade-offs. Be precise. Use code blocks for any code you produce.",
                "researcher" => "You are Researcher, a knowledge synthesizer. Find relevant information, cite sources when possible, summarize concisely, and flag uncertainty.",
                "guardian"   => "You are Guardian, a security and safety reviewer. Find vulnerabilities, unsafe patterns, missing validation, and compliance gaps. Suggest concrete mitigations.",
                "strategist" => "You are Strategist, a planning advisor. Lay out multi-step strategies, trade-offs, and second-order effects. Prefer numbered plans.",
                "artisan"    => "You are Artisan, a creative implementer. Generate concrete artifacts (text, prose, examples, snippets) that match the user's intent.",
                _ => "You are an AI assistant. Help the user with their request.",
            }
        }

        fn extract_prompt(msg: &qlang_agent::protocol::GraphMessage) -> String {
            // Primary: graph.metadata.content (IDE handover, cockpit composer)
            if let Some(content) = msg.graph.metadata.get("content") {
                if !content.is_empty() {
                    let filename = msg.graph.metadata.get("filename").cloned().unwrap_or_default();
                    let language = msg.graph.metadata.get("language").cloned().unwrap_or_default();
                    if !filename.is_empty() {
                        return format!("File: {}\nLanguage: {}\n\n---\n{}", filename, language, content);
                    }
                    return content.clone();
                }
            }
            // Fallback: stringify the whole graph (last resort, won't be useful but keeps the agent talking)
            serde_json::to_string_pretty(&msg.graph)
                .unwrap_or_else(|_| "(empty graph)".to_string())
        }

        let agent_names = ["ceo", "researcher", "developer", "guardian", "strategist", "artisan"];
        for name in &agent_names {
            let agent_id = AgentId {
                name: name.to_string(),
                capabilities: vec![Capability::Execute],
            };
            let mut mailbox = message_bus.register(agent_id).await;
            let agent_name = name.to_string();
            let events_tx = state.graph_events_tx.clone();
            let llm = state.llm.clone();
            let bus = message_bus.clone();
            tokio::spawn(async move {
                loop {
                    match mailbox.recv().await {
                        Some(msg) => {
                            // Ignore Result-intent messages so we don't reply to our own replies.
                            if matches!(msg.intent, MessageIntent::Result { .. }) {
                                continue;
                            }

                            tracing::debug!(
                                "Agent '{}' received QLMS from '{}' (intent: {:?})",
                                agent_name, msg.from.name, msg.intent
                            );

                            // Dashboard fanout (existing behavior).
                            let size_bytes = serde_json::to_vec(&msg.graph)
                                .map(|v| v.len() as u32)
                                .unwrap_or(0);
                            let intent_label = format!("{:?}", msg.intent)
                                .split('{')
                                .next()
                                .unwrap_or("Unknown")
                                .trim()
                                .to_string();
                            let _ = events_tx.send(routes::dashboard::GraphEvent::now(
                                &msg.from.name,
                                &agent_name,
                                &intent_label,
                                size_bytes,
                            ));

                            // ─── LLM call (with MCP-style tool loop) ─────────
                            //
                            // For selected agents (developer, researcher) the
                            // system prompt advertises a small set of tools.
                            // After each LLM reply we scan for `<tool .../>`
                            // markers; if any are present, we execute them,
                            // append the results as a new user turn, and call
                            // the LLM again. Capped at 3 iterations so a
                            // misbehaving model can't loop forever.
                            let user_prompt = extract_prompt(&msg);
                            let tools_block = match agent_name.as_str() {
                                "developer" | "researcher" => {
                                    format!("\n\n{}", crate::tools::available_tools_help())
                                }
                                _ => String::new(),
                            };
                            let system =
                                format!("{}{}", system_prompt_for(&agent_name), tools_block);
                            let mut messages: Vec<(String, String)> = vec![
                                ("system".to_string(), system),
                                ("user".to_string(), user_prompt),
                            ];

                            // Per-agent (tier, model) mapping. Some agents
                            // run on local Ollama (guardian, artisan), others
                            // on DeepSeek with a role-specific model. The
                            // router falls back to auto-routing if the
                            // preferred tier is offline, so this is safe even
                            // when Ollama isn't running.
                            let (agent_tier, agent_model) =
                                agent_models::model_for_agent(&agent_name);
                            const MAX_ITERATIONS: usize = 3;
                            let mut reply_text = String::new();
                            let mut tools_used: Vec<String> = Vec::new();

                            for _iter in 0..MAX_ITERATIONS {
                                let response = match llm
                                    .chat_with_model(
                                        Some(agent_tier),
                                        agent_model.clone(),
                                        messages.clone(),
                                    )
                                    .await
                                {
                                    Ok((text, used)) => {
                                        tracing::debug!(?used, "agent '{}' got LLM reply", agent_name);
                                        text
                                    }
                                    Err(e) => {
                                        tracing::warn!(
                                            "Agent '{}' LLM call failed: {}",
                                            agent_name,
                                            e
                                        );
                                        reply_text = format!("[agent '{}' error: {}]", agent_name, e);
                                        break;
                                    }
                                };

                                let tool_calls = crate::tools::parse_tool_calls(&response);
                                if tool_calls.is_empty() {
                                    // No tool calls — this IS the final answer.
                                    reply_text = response;
                                    break;
                                }

                                // Execute every tool call in order, building
                                // up a single user-turn message that the LLM
                                // sees on the next round.
                                let mut tool_results_text = String::new();
                                for call in tool_calls.iter().cloned() {
                                    let tool_name = call.name.clone();
                                    let result = crate::tools::execute_tool(call).await;
                                    tool_results_text.push_str(&format!(
                                        "<tool_result name=\"{}\" ok=\"{}\">\n{}\n</tool_result>\n",
                                        tool_name,
                                        result.ok,
                                        if result.ok {
                                            result.output.as_str()
                                        } else {
                                            result.error.as_deref().unwrap_or("?")
                                        }
                                    ));
                                    tools_used.push(tool_name);
                                }

                                messages.push(("assistant".to_string(), response));
                                messages.push((
                                    "user".to_string(),
                                    format!(
                                        "Tool results:\n{}\n\nNow give your final answer.",
                                        tool_results_text
                                    ),
                                ));
                            }

                            // If we exited the loop with reply_text still empty
                            // (i.e. hit MAX_ITERATIONS while still emitting
                            // tool calls), fall back to a graceful note so the
                            // bus delivery path always has something to send.
                            if reply_text.is_empty() {
                                reply_text = format!(
                                    "[agent '{}' reached the {}-iteration tool loop cap]",
                                    agent_name, MAX_ITERATIONS
                                );
                            }

                            // ─── Pipeline-chain forwarding ────────────────────
                            // If the incoming graph carries a `chain` metadata key
                            // (comma-separated list of next agent names), forward
                            // this agent's reply as a fresh Execute to the first
                            // name in the chain instead of replying to the sender.
                            // The original sender is preserved via `pipeline_origin`
                            // so the LAST agent in the chain can route the final
                            // Result back to the true initiator.
                            let chain_str = msg
                                .graph
                                .metadata
                                .get("chain")
                                .cloned()
                                .unwrap_or_default();
                            let chain: Vec<String> = chain_str
                                .split(',')
                                .map(|s| s.trim().to_string())
                                .filter(|s| !s.is_empty())
                                .collect();

                            if !chain.is_empty() {
                                let next_target = chain[0].clone();
                                let remaining: Vec<String> = chain[1..].to_vec();
                                let next_chain_str = remaining.join(",");

                                let original_sender = msg
                                    .graph
                                    .metadata
                                    .get("pipeline_origin")
                                    .cloned()
                                    .unwrap_or_else(|| msg.from.name.clone());

                                let mut forward_metadata = StdHashMap::new();
                                forward_metadata
                                    .insert("source".to_string(), "pipeline-forward".to_string());
                                forward_metadata
                                    .insert("agent".to_string(), agent_name.clone());
                                forward_metadata
                                    .insert("content".to_string(), reply_text.clone());
                                forward_metadata
                                    .insert("chain".to_string(), next_chain_str);
                                forward_metadata.insert(
                                    "pipeline_origin".to_string(),
                                    original_sender.clone(),
                                );
                                forward_metadata.insert(
                                    "pipeline_step".to_string(),
                                    format!(
                                        "{}",
                                        msg.graph
                                            .metadata
                                            .get("pipeline_step")
                                            .and_then(|s| s.parse::<u32>().ok())
                                            .unwrap_or(0)
                                            + 1
                                    ),
                                );

                                let forward_id = (msg.id ^ 0x6A09E667F3BCC908u64).wrapping_add(
                                    std::time::SystemTime::now()
                                        .duration_since(std::time::UNIX_EPOCH)
                                        .map(|d| d.as_nanos() as u64)
                                        .unwrap_or(0),
                                );

                                let forward_graph = Graph {
                                    id: format!(
                                        "pipeline-{}-{}-{}",
                                        agent_name, next_target, forward_id
                                    ),
                                    version: "1.0".to_string(),
                                    nodes: vec![],
                                    edges: vec![],
                                    constraints: vec![],
                                    metadata: forward_metadata,
                                };

                                let forward_msg = qlang_agent::protocol::GraphMessage {
                                    id: forward_id,
                                    from: AgentId {
                                        name: agent_name.clone(),
                                        capabilities: vec![Capability::Execute],
                                    },
                                    to: AgentId {
                                        name: next_target.clone(),
                                        capabilities: vec![Capability::Execute],
                                    },
                                    graph: forward_graph,
                                    inputs: StdHashMap::new(),
                                    intent: MessageIntent::Execute,
                                    in_reply_to: Some(msg.id),
                                    signature: None,
                                    signer_pubkey: None,
                                    graph_hash: None,
                                };

                                let _ = bus.send(forward_msg.clone()).await;

                                // Dashboard fanout for the pipeline edge.
                                let forward_size = serde_json::to_vec(&forward_msg.graph)
                                    .map(|v| v.len() as u32)
                                    .unwrap_or(0);
                                let _ = events_tx.send(routes::dashboard::GraphEvent::now(
                                    &agent_name,
                                    &next_target,
                                    "PipelineForward",
                                    forward_size,
                                ));

                                // Skip the regular Result-reply: the LAST agent
                                // in the chain produces the final Result and
                                // routes it to pipeline_origin. Replying here
                                // would double-deliver to the sender.
                                continue;
                            }

                            // ─── Build reply GraphMessage ─────────────────────
                            let reply_id = (msg.id ^ 0x9E3779B97F4A7C15u64).wrapping_add(
                                std::time::SystemTime::now()
                                    .duration_since(std::time::UNIX_EPOCH)
                                    .map(|d| d.as_nanos() as u64)
                                    .unwrap_or(0),
                            );
                            let mut reply_metadata = StdHashMap::new();
                            reply_metadata.insert("source".to_string(), "agent".to_string());
                            reply_metadata.insert("agent".to_string(), agent_name.clone());
                            reply_metadata.insert("content".to_string(), reply_text.clone());
                            reply_metadata.insert("in_reply_to_graph".to_string(), msg.graph.id.clone());

                            // Surface tool usage so the cockpit can render a
                            // "tools used" badge next to the agent reply.
                            if !tools_used.is_empty() {
                                reply_metadata
                                    .insert("tools_used".to_string(), tools_used.join(","));
                            }

                            // Pipeline summary for the cockpit/IDE: copy origin
                            // and step counter through, and tag this final agent
                            // so the receiver can render the chain history.
                            if let Some(origin) = msg.graph.metadata.get("pipeline_origin") {
                                reply_metadata
                                    .insert("pipeline_origin".to_string(), origin.clone());
                            }
                            if let Some(step) = msg.graph.metadata.get("pipeline_step") {
                                reply_metadata
                                    .insert("pipeline_step".to_string(), step.clone());
                            }
                            if msg.graph.metadata.contains_key("pipeline_origin") {
                                reply_metadata.insert(
                                    "pipeline_chain_completed".to_string(),
                                    agent_name.clone(),
                                );
                            }

                            let reply_graph = Graph {
                                id: format!("reply-{}-{}", agent_name, reply_id),
                                version: "1.0".to_string(),
                                nodes: vec![],
                                edges: vec![],
                                constraints: vec![],
                                metadata: reply_metadata,
                            };

                            // If we are the LAST agent in a pipeline, route the
                            // Result back to the true original sender (carried in
                            // pipeline_origin) instead of the immediate `from`
                            // (which would be the previous pipeline agent).
                            let reply_to = if let Some(origin) =
                                msg.graph.metadata.get("pipeline_origin")
                            {
                                AgentId {
                                    name: origin.clone(),
                                    capabilities: vec![Capability::Execute],
                                }
                            } else {
                                msg.from.clone()
                            };

                            let reply = qlang_agent::protocol::GraphMessage {
                                id: reply_id,
                                from: AgentId {
                                    name: agent_name.clone(),
                                    capabilities: vec![Capability::Execute],
                                },
                                to: reply_to,
                                graph: reply_graph,
                                inputs: StdHashMap::new(),
                                intent: MessageIntent::Result { original_message_id: msg.id },
                                in_reply_to: Some(msg.id),
                                signature: None,
                                signer_pubkey: None,
                                graph_hash: None,
                            };

                            // Send reply. If the recipient has no mailbox (e.g., vscode-assistant),
                            // bus.send() still emits to the SSE subscribers — the IDE inbox listens there.
                            let _ = bus.send(reply.clone()).await;

                            // Dashboard fanout for the reply edge too.
                            let reply_size = serde_json::to_vec(&reply.graph)
                                .map(|v| v.len() as u32)
                                .unwrap_or(0);
                            let _ = events_tx.send(routes::dashboard::GraphEvent::now(
                                &agent_name,
                                &reply.to.name,
                                "Result",
                                reply_size,
                            ));
                        }
                        None => break, // Channel closed
                    }
                }
            });
        }
        tracing::info!(
            "Message bus: {} agents registered with LLM-backed mailboxes",
            agent_names.len()
        );
    }

    // ---- Read routes: any authenticated principal, including a viewer seat. ----
    let read_router = Router::new()
        .route("/api/health", get(routes::health::health))
        .route("/api/chat/history", get(routes::chat::chat_history))
        .route("/api/hardware", get(routes::hardware::hardware))
        .route("/api/multi-agent/runs", get(routes::multi_agent::list_runs))
        .route("/api/multi-agent/runs/{id}", get(routes::multi_agent::get_run))
        .route("/api/multi-agent/stream", get(routes::multi_agent::stream_runs))
        .route("/api/swarm/active", get(routes::swarm::list_active))
        .route("/api/swarm/{id}", get(routes::swarm::get_swarm))
        .route("/api/autonomous/status", get(routes::autonomous::get_status))
        .route("/api/git/branches", get(routes::git::list_auto_branches))
        .route("/api/git/diff/{branch}", get(routes::git::diff_branch))
        .route("/api/history", get(routes::history::get_history))
        .route("/api/goals/{id}/graph", get(routes::goals::get_goal_graph))
        .route("/api/graphs", get(routes::graphs::list_graphs))
        .route("/api/graphs/stats", get(routes::graphs::graph_stats))
        .route("/api/graphs/{id}", get(routes::graphs::get_graph))
        .route("/api/providers", get(routes::providers::list_providers))
        .route("/api/providers/costs", get(routes::providers::cost_summary))
        .route("/api/providers/templates", get(routes::providers::list_templates))
        .route("/api/providers/configured", get(routes::providers::list_configured))
        .route("/api/knowledge/stats", get(routes::knowledge_tools::knowledge_stats))
        .route("/api/knowledge/snapshot", get(routes::knowledge_tools::knowledge_snapshot))
        .route("/api/knowledge/deltas", get(routes::knowledge_tools::delta_log))
        .route("/api/knowledge/receipt", get(routes::knowledge_tools::knowledge_receipt))
        .route("/api/knowledge/divergences", get(routes::knowledge_tools::divergences))
        .route("/api/knowledge/export", get(routes::knowledge_tools::knowledge_export))
        .route("/api/knowledge/health", get(routes::knowledge_tools::knowledge_health))
        .route("/api/knowledge/backups", get(routes::knowledge_tools::knowledge_backups))
        .route("/api/memory/stats", get(routes::memory::memory_stats))
        .route("/api/memory/search", get(routes::memory::memory_search))
        .route("/api/messages/stats", get(routes::messages::bus_stats))
        .route("/api/messages/agents", get(routes::messages::bus_agents))
        .route("/api/messages/conversations", get(routes::messages::bus_conversations))
        .route("/api/messages/stream", get(routes::messages::bus_stream))
        .route("/api/messages/recent", get(routes::messages::recent_messages))
        .route("/api/history/unified", get(routes::messages::unified_history))
        .route("/api/supervisor/state", get(routes::supervisor::state))
        .route("/api/supervisor/logs", get(routes::supervisor::logs))
        .route("/api/supervisor/console", get(routes::supervisor::console))
        .route("/api/supervisor/presets", get(routes::supervisor::presets))
        .route("/api/supervisor/handover/show", get(routes::supervisor::show_handover))
        .route("/api/supervisor/stream", get(routes::supervisor::stream))
        .route("/api/supervisor/daemon/status", get(routes::supervisor::daemon_status))
        .route("/api/values", get(routes::dashboard::get_values))
        .route("/ws/graph-stream", get(routes::dashboard::graph_stream))
        .route("/api/federation/peers", get(routes::dashboard::get_peers))
        .route("/api/federation/stats", get(routes::dashboard::get_federation_stats))
        .route("/api/tools/web_search", get(routes::workspace::web_search))
        .route("/api/tools/fetch_url", get(routes::workspace::fetch_url))
        .route("/api/workspace/tree", get(routes::workspace::tree))
        .route("/api/workspace/file", get(routes::workspace::read_file))
        .route("/api/presence", get(routes::presence::list))
        .route("/api/neo/hardware", get(routes::neo::hardware))
        .route("/api/neo/memory", get(routes::neo::memory))
        .route("/api/neo/status", get(routes::neo::status))
        .route("/api/neo/agents", get(routes::neo::list_agents))
        .route("/api/neo/agents/{id}", get(routes::neo::get_agent))
        .route("/supervisor", get(routes::supervisor::cockpit))
        .route("/supervisor/legacy", get(routes::supervisor::cockpit_legacy))
        // MCP is one POST endpoint that dispatches read AND write tools; the
        // per-tool role check lives in the dispatcher, not here.
        .route("/mcp/v1", post(routes::mcp_server::handle_rpc));

    // ---- Write routes: a member (or admin) seat. A viewer is 403'd here. ----
    let write_router = Router::new()
        .route("/api/chat", post(routes::chat::chat))
        .route("/api/consensus", post(routes::consensus::consensus))
        // Mesh fan-out — push one prompt to N IDE identities at once.
        .route("/api/broadcast", post(routes::broadcast::broadcast))
        // IDE-side LLM delegation: extensions POST chat requests here so qo
        // can use its centrally-stored API keys.
        .route("/api/llm/proxy", post(routes::llm_proxy::proxy_chat))
        .route("/api/multi-agent/run", post(routes::multi_agent::run_multi_agent))
        .route("/api/multi-agent/runs/start", post(routes::multi_agent::start_run))
        .route("/api/swarm/start", post(routes::swarm::start_swarm))
        .route("/api/swarm/{id}/stop", post(routes::swarm::stop_swarm))
        .route("/api/graphs", post(routes::graphs::store_graph))
        .route("/api/knowledge/index", post(routes::knowledge_tools::index_workspace))
        .route("/api/knowledge/delta", post(routes::knowledge_tools::commit_delta))
        .route("/api/knowledge/propose", post(routes::knowledge_tools::propose))
        .route("/api/knowledge/verify-source", post(routes::knowledge_tools::verify_source))
        .route("/api/knowledge/verify-all", post(routes::knowledge_tools::verify_all))
        .route("/api/knowledge/refresh-sources", post(routes::knowledge_tools::refresh_sources))
        .route("/api/knowledge/heal-stale", post(routes::knowledge_tools::heal_stale))
        .route("/api/knowledge/import", post(routes::knowledge_tools::knowledge_import))
        .route("/api/knowledge/backup", post(routes::knowledge_tools::knowledge_backup))
        .route("/api/knowledge/restore", post(routes::knowledge_tools::knowledge_restore))
        .route("/api/values", post(routes::dashboard::update_values))
        .route("/api/supervisor/suggest-agent", post(routes::supervisor::suggest_agent))
        .route("/api/supervisor/dispatch", post(routes::supervisor::dispatch_preset))
        .route("/api/supervisor/task", post(routes::supervisor::add_task))
        .route("/api/supervisor/action", post(routes::supervisor::action))
        .route("/api/supervisor/task-action", post(routes::supervisor::task_action))
        .route("/api/supervisor/session-prompt", post(routes::supervisor::session_prompt))
        .route("/api/supervisor/handover/create", post(routes::supervisor::create_handover))
        .route("/api/supervisor/handover/reply", post(routes::supervisor::reply_handover))
        // MCP ↔ QLMS bridge (spec §15.2 / PRD Task 2.2)
        .route("/qlms/v1.1/deliver", post(routes::mcp_qlms::deliver))
        .route("/qlms/v1.1/reply", post(routes::mcp_qlms::reply))
        // Workspace — agent-writable sandbox + file browser + runner
        .route("/api/tools/write_file", post(routes::workspace::write_file))
        .route("/api/tools/exec_file", post(routes::workspace::exec_file))
        .route("/api/workspace/file", delete(routes::workspace::delete_file))
        // Presence registry — connected IDEs/agents register and discover.
        .route("/api/presence/register", post(routes::presence::register))
        .route("/api/presence/heartbeat/{identity}", post(routes::presence::heartbeat))
        .route("/api/presence/{identity}/eligibility", post(routes::presence::set_eligibility))
        .route("/api/presence/{identity}", delete(routes::presence::deregister))
        .layer(middleware::from_fn(auth::require_write));

    // ---- Admin routes: server configuration and destructive operations. ----
    let admin_router = Router::new()
        .route("/api/providers/add", post(routes::providers::add_provider))
        .route("/api/providers/test", post(routes::providers::test_provider))
        .route("/api/providers/{id}/toggle", put(routes::providers::toggle_provider))
        .route("/api/providers/{id}/edit", put(routes::providers::update_provider))
        .route("/api/providers/{id}", delete(routes::providers::delete_provider))
        .route("/api/git/merge", post(routes::git::merge_branch))
        .route("/api/git/discard", post(routes::git::discard_branch))
        // Autonomous swarm scheduler — runs swarms on a timer with a budget cap.
        .route("/api/autonomous/start", post(routes::autonomous::start_autonomous))
        .route("/api/autonomous/stop", post(routes::autonomous::stop_autonomous))
        .route("/api/autonomous/queue", put(routes::autonomous::set_queue))
        .route("/api/supervisor/agent", post(routes::supervisor::add_agent))
        .route("/api/supervisor/install-preset", post(routes::supervisor::install_preset))
        .route("/api/supervisor/daemon/start", post(routes::supervisor::daemon_start))
        .route("/api/supervisor/daemon/stop", post(routes::supervisor::daemon_stop))
        .layer(middleware::from_fn(auth::require_admin));

    // Rate limit before auth: an unauthenticated flood must be stopped before
    // it reaches any handler. Built once so every request shares one limiter.
    let rate_limiter = std::sync::Arc::new(rate_limit::RateLimiter::new(
        config.rate_per_second,
        config.rate_burst_size,
    ));

    let api_router = read_router
        .merge(write_router)
        .merge(admin_router)
        .layer(middleware::from_fn_with_state(
            state.clone(),
            auth::auth_middleware,
        ))
        .layer(middleware::from_fn_with_state(
            rate_limiter,
            rate_limit::middleware,
        ))
        // Bound every buffered body, so a 2 GB stream is refused up front.
        .layer(axum::extract::DefaultBodyLimit::max(config.max_body_bytes))
        .with_state(state.clone());

    let router = if let Some(static_dir) = config.static_dir {
        let index_html = static_dir.join("index.html");
        api_router
            .nest_service("/assets", ServeDir::new(static_dir.join("assets")))
            .route_service("/favicon.svg", ServeFile::new(static_dir.join("favicon.svg")))
            .fallback_service(ServeFile::new(index_html))
    } else {
        api_router
    };

    let router = router.layer(cors_layer(&config.cors_origins));

    Ok((router, state))
}

/// Build the CORS layer from the configured allow-list.
///
/// The old behaviour was [`CorsLayer::permissive`] — any origin could attempt
/// cross-origin requests. That sat *above* the auth layer, so a token did not
/// save a deployment from an attacker's web page. Now:
///
/// - An empty allow-list means no `Access-Control-Allow-Origin` is emitted, so
///   the browser's same-origin policy blocks cross-origin requests. The
///   embedded cockpit is same-origin and unaffected.
/// - A non-empty list allows exactly those origins (malformed entries are
///   dropped, never turned into `*`).
fn cors_layer(origins: &[String]) -> CorsLayer {
    use tower_http::cors::AllowOrigin;

    let allowed: Vec<axum::http::HeaderValue> = origins
        .iter()
        .map(|o| o.trim())
        .filter(|o| !o.is_empty())
        .filter_map(|o| o.parse().ok())
        .collect();

    if allowed.is_empty() {
        CorsLayer::new()
    } else {
        CorsLayer::new().allow_origin(AllowOrigin::list(allowed))
    }
}

#[cfg(test)]
mod knowledge_route_tests {
    use super::*;
    use crate::routes::knowledge_tools::{
        divergences, heal_stale, knowledge_backup, knowledge_export, knowledge_import,
        knowledge_receipt, knowledge_restore, propose, refresh_sources, verify_all, verify_source,
        HealStaleRequest, ProposeRequest, ReceiptQuery, RefreshSourcesRequest, RestoreRequest,
        VerifyAllRequest, VerifySourceRequest,
    };
    use axum::extract::{Query, State};
    use qo_knowledge::model::{EntityId, EntityKind, Evidence, EvidenceKind, Provenance};
    use qo_knowledge::{Claim, ClaimId};

    fn prov(producer: &str, at: u64) -> Provenance {
        Provenance {
            producer: producer.into(),
            observed_at: at,
            git_revision: None,
            run_id: None,
        }
    }

    /// The knowledge HTTP handlers, called directly against a real `AppState`
    /// built by `build_app`. This closes the last test gap: the routes' arg
    /// parsing, serialization and status codes had only manual verification
    /// until now. The full lifecycle walks through the routes themselves.
    #[tokio::test]
    async fn knowledge_routes_walk_the_lifecycle() {
        let dir = tempfile::tempdir().unwrap();
        let workspace = dir.path().join("ws");
        std::fs::create_dir_all(&workspace).unwrap();
        std::fs::write(
            workspace.join("auth.rs"),
            "// auth hashes passwords with bcrypt\nfn probe() {}\n",
        )
        .unwrap();

        let (_router, state) = crate::build_app(QoConfig {
            data_dir: dir.path().join("data"),
            workspace_root: workspace.clone(),
            ..Default::default()
        })
        .await
        .expect("build app");

        let subject = EntityId::derive(EntityKind::File, "auth.rs");

        // Seed a proposal directly on the store — the routes are what we test.
        state
            .knowledge
            .add_claim(&Claim::proposed(
                "c1",
                "auth hashes passwords with bcrypt",
                subject.clone(),
                prov("worker", 1),
            ))
            .unwrap();

        // verify-source route: promotes only on a literal match.
        let (status, axum::Json(body)) = verify_source(
            State(state.clone()),
            axum::Json(VerifySourceRequest {
                id: "c1".into(),
                by: "checker".into(),
            }),
        )
        .await;
        assert_eq!(status, axum::http::StatusCode::OK);
        assert_eq!(body["verdict"]["kind"], "verified");

        // receipt route: renders the trail; unknown claim is 404.
        let (status, axum::Json(body)) = knowledge_receipt(
            State(state.clone()),
            Query(ReceiptQuery {
                claim_id: "c1".into(),
            }),
        )
        .await;
        assert_eq!(status, axum::http::StatusCode::OK);
        assert!(body["rendered"].as_str().unwrap().contains("VERIFIED"));
        let (status, _) = knowledge_receipt(
            State(state.clone()),
            Query(ReceiptQuery {
                claim_id: "nope".into(),
            }),
        )
        .await;
        assert_eq!(status, axum::http::StatusCode::NOT_FOUND);

        // verify-all route: c1 is settled, c2 is inconclusive.
        state
            .knowledge
            .add_claim(&Claim::proposed(
                "c2",
                "auth validates tokens",
                subject.clone(),
                prov("worker", 2),
            ))
            .unwrap();
        let (status, axum::Json(body)) = verify_all(
            State(state.clone()),
            axum::Json(VerifyAllRequest { by: "sweeper".into() }),
        )
        .await;
        assert_eq!(status, axum::http::StatusCode::OK);
        assert_eq!(body["verified"], 0);
        assert_eq!(body["inconclusive"], 1);

        // divergences route: a refuted counter-claim shows up.
        state
            .knowledge
            .add_claim(&Claim::proposed(
                "c3",
                "auth uses md5",
                subject.clone(),
                prov("worker-9", 3),
            ))
            .unwrap();
        state
            .knowledge
            .refute_claim(
                &ClaimId("c3".into()),
                Evidence {
                    kind: EvidenceKind::Source,
                    locator: "auth.rs".into(),
                    lines: None,
                    excerpt: None,
                    supports: false,
                },
                prov("reviewer", 4),
            )
            .unwrap();
        let axum::Json(body) = divergences(State(state.clone())).await;
        assert_eq!(body["divergences"].as_array().unwrap().len(), 1);

        // refresh route: the source moved on -> c1 goes stale.
        std::fs::write(workspace.join("auth.rs"), "fn probe() { /* nothing relevant */ }\n").unwrap();
        let (status, axum::Json(body)) = refresh_sources(
            State(state.clone()),
            axum::Json(RefreshSourcesRequest {
                by: "refresher".into(),
            }),
        )
        .await;
        assert_eq!(status, axum::http::StatusCode::OK);
        assert_eq!(body["stale"], 1);

        // heal route: the fact is genuinely gone -> it stays stale.
        let (status, axum::Json(body)) = heal_stale(
            State(state.clone()),
            axum::Json(HealStaleRequest { by: "healer".into() }),
        )
        .await;
        assert_eq!(status, axum::http::StatusCode::OK);
        assert_eq!(body["healed"], 0);
        assert_eq!(body["remained_stale"], 1);
    }

    /// Backup then restore: export the graph from one instance, import it into
    /// a fresh one, and confirm the restored claim keeps its status, revision
    /// and provenance — the whole audit trail survives the round trip.
    #[tokio::test]
    async fn knowledge_export_import_round_trips() {
        let dir = tempfile::tempdir().unwrap();
        let workspace = dir.path().join("ws");
        std::fs::create_dir_all(&workspace).unwrap();
        std::fs::write(workspace.join("auth.rs"), "// auth hashes passwords with bcrypt\n").unwrap();

        // Source instance: one verified claim with evidence.
        let (_router_a, state_a) = crate::build_app(QoConfig {
            data_dir: dir.path().join("data-a"),
            workspace_root: workspace.clone(),
            ..Default::default()
        })
        .await
        .expect("build app A");
        let subject = EntityId::derive(EntityKind::File, "auth.rs");
        state_a
            .knowledge
            .add_claim(&Claim::proposed(
                "c1",
                "auth hashes passwords with bcrypt",
                subject,
                prov("worker", 1),
            ))
            .unwrap();
        state_a
            .knowledge
            .verify_claim(
                &ClaimId("c1".into()),
                Evidence {
                    kind: EvidenceKind::Source,
                    locator: "auth.rs".into(),
                    lines: Some((1, 1)),
                    excerpt: Some("// auth hashes passwords with bcrypt".into()),
                    supports: true,
                },
                prov("reviewer", 2),
            )
            .unwrap();

        // Export.
        let axum::Json(archive) = knowledge_export(State(state_a.clone()))
            .await
            .expect("export must succeed");
        assert!(!archive.claims.is_empty());

        // Fresh instance, empty graph.
        let (_router_b, state_b) = crate::build_app(QoConfig {
            data_dir: dir.path().join("data-b"),
            workspace_root: workspace,
            ..Default::default()
        })
        .await
        .expect("build app B");

        // Import and assert the audit trail is intact, not degraded.
        let (status, axum::Json(body)) = knowledge_import(State(state_b.clone()), axum::Json(archive)).await;
        assert_eq!(status, axum::http::StatusCode::OK);
        assert_eq!(body["claims_added"], 2, "{body}"); // rev 1 + rev 2

        let restored = state_b
            .knowledge
            .latest(&ClaimId("c1".into()))
            .unwrap()
            .unwrap();
        assert_eq!(restored.status, qo_knowledge::ClaimStatus::Verified);
        assert_eq!(restored.revision, 2);
        assert_eq!(restored.provenance.producer, "reviewer");
    }

    /// Role enforcement, exercised through the real router + middleware: a
    /// viewer seat reads but cannot write; a member writes but cannot
    /// administer; an admin administers. Written as attacks.
    #[tokio::test]
    async fn roles_are_enforced_at_the_router() {
        use axum::body::Body;
        use axum::http::{Request, StatusCode};
        use tower::ServiceExt;

        let dir = tempfile::tempdir().unwrap();

        // Per-seat keys are loaded from a file at startup; write one so the
        // middleware has viewer/member/admin seats to resolve.
        let keys_path = dir.path().join("api_keys.json");
        std::fs::write(
            &keys_path,
            r#"{"keys":[
                {"label":"viewer","secret":"viewer-secret","role":"viewer","revoked":false},
                {"label":"member","secret":"member-secret","role":"member","revoked":false},
                {"label":"admin","secret":"admin-secret","role":"admin","revoked":false}
            ]}"#,
        )
        .unwrap();

        let (router, _state) = crate::build_app(QoConfig {
            data_dir: dir.path().join("data"),
            workspace_root: dir.path().join("ws"),
            api_keys_path: keys_path,
            ..Default::default()
        })
        .await
        .expect("build app");

        async fn call(
            router: &axum::Router,
            method: &str,
            path: &str,
            token: &str,
            body: &str,
        ) -> StatusCode {
            let mut builder = Request::builder()
                .method(method)
                .uri(path)
                .header("authorization", format!("Bearer {token}"));
            let request = if body.is_empty() {
                builder.body(Body::empty()).unwrap()
            } else {
                builder
                    .header("content-type", "application/json")
                    .body(Body::from(body.to_string()))
                    .unwrap()
            };
            router.clone().oneshot(request).await.unwrap().status()
        }

        // A viewer reads fine…
        assert_eq!(
            call(&router, "GET", "/api/knowledge/stats", "viewer-secret", "").await,
            StatusCode::OK
        );
        // …but a write route and an admin route are both 403.
        assert_eq!(
            call(&router, "POST", "/api/knowledge/delta", "viewer-secret", "{\"document\":\"DELTA|1|x\\nBY|x|1\\n\"}").await,
            StatusCode::FORBIDDEN
        );
        assert_eq!(
            call(&router, "POST", "/api/providers/add", "viewer-secret", "{}").await,
            StatusCode::FORBIDDEN
        );

        // A member passes the write gate (the handler then runs and returns
        // its own status — here 401 for an untrusted delta, not 403).
        assert_ne!(
            call(&router, "POST", "/api/knowledge/delta", "member-secret", "{\"document\":\"DELTA|1|x\\nBY|x|1\\n\"}").await,
            StatusCode::FORBIDDEN
        );
        // …but is refused at the admin gate.
        assert_eq!(
            call(&router, "POST", "/api/providers/add", "member-secret", "{}").await,
            StatusCode::FORBIDDEN
        );

        // An admin passes the admin gate.
        assert_ne!(
            call(&router, "POST", "/api/providers/add", "admin-secret", "{}").await,
            StatusCode::FORBIDDEN
        );

        // With seats issued, an unauthenticated request is refused outright.
        let request = Request::builder()
            .uri("/api/knowledge/stats")
            .body(Body::empty())
            .unwrap();
        assert_eq!(
            router.clone().oneshot(request).await.unwrap().status(),
            StatusCode::UNAUTHORIZED
        );
    }

    /// MCP enforcement lives in the dispatcher, not the route: a viewer may
    /// call read tools, but a write tool is refused before any work happens.
    #[tokio::test]
    async fn mcp_write_tools_are_refused_for_viewers() {
        let dir = tempfile::tempdir().unwrap();
        let (_router, state) = crate::build_app(QoConfig {
            data_dir: dir.path().join("data"),
            workspace_root: dir.path().join("ws"),
            ..Default::default()
        })
        .await
        .expect("build app");

        let viewer = crate::api_keys::Principal {
            label: "viewer".into(),
            role: crate::api_keys::Role::Viewer,
        };
        let member = crate::api_keys::Principal {
            label: "member".into(),
            role: crate::api_keys::Role::Member,
        };

        // Viewer: read tool works, write tool is forbidden (-32001).
        let read = crate::routes::knowledge_tools::call(
            state.clone(),
            &viewer,
            "orbit_graph_health",
            serde_json::Value::Null,
        )
        .await;
        assert!(read.is_ok(), "{read:?}");

        let write = crate::routes::knowledge_tools::call(
            state.clone(),
            &viewer,
            "orbit_graph_add_claim",
            serde_json::json!({"id":"c1","statement":"x","subject_kind":"file","subject_name":"a.rs","by":"w"}),
        )
        .await;
        assert!(matches!(write, Err((code, _)) if code == -32001), "{write:?}");

        // Member: the same write tool is not forbidden at the gate (it runs
        // and returns its own result).
        let member_write = crate::routes::knowledge_tools::call(
            state.clone(),
            &member,
            "orbit_graph_add_claim",
            serde_json::json!({"id":"c1","statement":"x","subject_kind":"file","subject_name":"a.rs","by":"w"}),
        )
        .await;
        assert!(
            !matches!(member_write, Err((code, _)) if code == -32001),
            "{member_write:?}"
        );
    }

    /// A request body over the configured cap is refused before any handler
    /// runs — the 2 GB-delta DoS vector is bounded.
    #[tokio::test]
    async fn oversized_bodies_are_refused() {
        use axum::body::Body;
        use axum::http::{Request, StatusCode};
        use tower::ServiceExt;

        let dir = tempfile::tempdir().unwrap();
        let (router, _state) = crate::build_app(QoConfig {
            data_dir: dir.path().join("data"),
            workspace_root: dir.path().join("ws"),
            max_body_bytes: 1024,
            ..Default::default()
        })
        .await
        .expect("build app");

        let request = Request::builder()
            .method("POST")
            .uri("/api/knowledge/delta")
            .header("content-type", "application/json")
            .body(Body::from("x".repeat(2048)))
            .unwrap();
        assert_eq!(
            router.oneshot(request).await.unwrap().status(),
            StatusCode::PAYLOAD_TOO_LARGE
        );
    }

    /// A flood from one peer is refused once the burst bucket is empty.
    #[tokio::test]
    async fn floods_are_rate_limited() {
        use axum::body::Body;
        use axum::http::{Request, StatusCode};
        use tower::ServiceExt;

        let dir = tempfile::tempdir().unwrap();
        let (router, _state) = crate::build_app(QoConfig {
            data_dir: dir.path().join("data"),
            workspace_root: dir.path().join("ws"),
            rate_per_second: 1,
            rate_burst_size: 2,
            ..Default::default()
        })
        .await
        .expect("build app");

        let get = || Request::builder().uri("/api/health").body(Body::empty()).unwrap();
        assert_eq!(
            router.clone().oneshot(get()).await.unwrap().status(),
            StatusCode::OK
        );
        assert_eq!(
            router.clone().oneshot(get()).await.unwrap().status(),
            StatusCode::OK
        );
        assert_eq!(
            router.clone().oneshot(get()).await.unwrap().status(),
            StatusCode::TOO_MANY_REQUESTS
        );
    }

    /// The proposal admission gate, through the real route: a valid document is
    /// merged as proposals; a document that tries to verify (OK) is refused
    /// whole.
    #[tokio::test]
    async fn propose_admits_and_rejects_via_the_gate() {
        let dir = tempfile::tempdir().unwrap();
        let workspace = dir.path().join("ws");
        std::fs::create_dir_all(&workspace).unwrap();
        let (_router, state) = crate::build_app(QoConfig {
            data_dir: dir.path().join("data"),
            workspace_root: workspace,
            ..Default::default()
        })
        .await
        .expect("build app");

        let ok_doc = "DELTA|1|d-1\nBY|worker-3|1700000000\n+E|file|src/auth.rs\n+C|c1|file:src/auth.rs|auth uses bcrypt\n";
        let (status, axum::Json(body)) = propose(
            State(state.clone()),
            axum::Json(ProposeRequest {
                document: ok_doc.into(),
            }),
        )
        .await;
        assert_eq!(status, axum::http::StatusCode::OK, "{body}");
        assert_eq!(body["applied"], 2); // +E and +C

        // The admission gate refuses OK — a model may not verify.
        let bad_doc = "DELTA|1|d-2\nBY|worker-3|1700000000\n+E|file|src/auth.rs\n+C|c1|file:src/auth.rs|auth uses bcrypt\nOK|c1|source|src/auth.rs|1:1|x\n";
        let (status, axum::Json(body)) = propose(
            State(state.clone()),
            axum::Json(ProposeRequest {
                document: bad_doc.into(),
            }),
        )
        .await;
        assert_eq!(status, axum::http::StatusCode::BAD_REQUEST);
        assert!(!body["violations"].as_array().unwrap().is_empty());
    }

    /// The operator's recovery path: back up, then restore from that backup.
    /// Restore is additive, so re-importing reports the existing claim as
    /// skipped rather than duplicating it.
    #[tokio::test]
    async fn restore_recovers_from_a_backup() {
        let dir = tempfile::tempdir().unwrap();
        let workspace = dir.path().join("ws");
        std::fs::create_dir_all(&workspace).unwrap();
        let (_router, state) = crate::build_app(QoConfig {
            data_dir: dir.path().join("data"),
            workspace_root: workspace,
            ..Default::default()
        })
        .await
        .expect("build app");

        let subject = EntityId::derive(EntityKind::File, "auth.rs");
        state
            .knowledge
            .add_claim(&Claim::proposed(
                "c1",
                "auth uses bcrypt",
                subject,
                prov("w", 1),
            ))
            .unwrap();
        state
            .knowledge
            .verify_claim(
                &ClaimId("c1".into()),
                Evidence {
                    kind: EvidenceKind::Source,
                    locator: "auth.rs".into(),
                    lines: None,
                    excerpt: None,
                    supports: true,
                },
                prov("r", 2),
            )
            .unwrap();

        // Back up, then restore the newest backup.
        let (status, _) = knowledge_backup(State(state.clone())).await;
        assert_eq!(status, axum::http::StatusCode::OK);

        let (status, axum::Json(body)) = knowledge_restore(
            State(state.clone()),
            axum::Json(RestoreRequest { exported_at: None }),
        )
        .await;
        assert_eq!(status, axum::http::StatusCode::OK, "{body}");
        // Additive: the claim already exists, so it is skipped, not duplicated.
        assert!(
            body["claims_skipped"]
                .as_array()
                .unwrap()
                .contains(&serde_json::json!("c1")),
            "{body}"
        );

        // An unknown timestamp is a 404.
        let (status, _) = knowledge_restore(
            State(state.clone()),
            axum::Json(RestoreRequest {
                exported_at: Some(999_999),
            }),
        )
        .await;
        assert_eq!(status, axum::http::StatusCode::NOT_FOUND);
    }
}
