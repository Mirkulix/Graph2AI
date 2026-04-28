pub mod agent_models;
pub mod auth;
pub mod config;
pub mod git_ops;
pub mod mesh_history;
pub mod peer_discovery;
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
use tower_http::services::ServeDir;

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
    pub obsidian_vault: std::path::PathBuf,
    pub static_dir: Option<std::path::PathBuf>,
    /// Optional API token for bearer auth (reads QO_AUTH_TOKEN from env if None)
    pub auth_token: Option<String>,
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
            obsidian_vault: std::path::PathBuf::from("vault"),
            static_dir: None,
            auth_token: None,
            llm_routing: config::LlmRoutingConfig::default(),
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

    let api_router = Router::new()
        .route("/api/health", get(routes::health::health))
        .route("/api/chat", post(routes::chat::chat))
        .route("/api/chat/history", get(routes::chat::chat_history))
        .route("/api/consensus", post(routes::consensus::consensus))
        // Mesh fan-out — push one prompt to N IDE identities at once. Fire
        // and forget; replies trickle in through the existing SSE stream.
        .route("/api/broadcast", post(routes::broadcast::broadcast))
        // IDE-side LLM delegation: extensions POST chat requests here so
        // qo can use its centrally-stored API keys instead of every IDE
        // shipping its own credentials.
        .route("/api/llm/proxy", post(routes::llm_proxy::proxy_chat))
        // Host telemetry — CPU + RAM via sysinfo crate.
        .route("/api/hardware", get(routes::hardware::hardware))
        // Autonomous multi-agent swarm orchestrator.
        .route("/api/swarm/start", post(routes::swarm::start_swarm))
        .route("/api/swarm/active", get(routes::swarm::list_active))
        .route("/api/swarm/{id}", get(routes::swarm::get_swarm))
        .route("/api/swarm/{id}/stop", post(routes::swarm::stop_swarm))
        // Autonomous swarm scheduler — runs swarms on a timer with
        // a hard daily USD budget cap.
        .route("/api/autonomous/start", post(routes::autonomous::start_autonomous))
        .route("/api/autonomous/stop", post(routes::autonomous::stop_autonomous))
        .route("/api/autonomous/status", get(routes::autonomous::get_status))
        .route("/api/autonomous/queue", put(routes::autonomous::set_queue))
        // Git auto-improver branches produced by autonomous swarms.
        .route("/api/git/branches", get(routes::git::list_auto_branches))
        .route("/api/git/diff/{branch}", get(routes::git::diff_branch))
        .route("/api/git/merge", post(routes::git::merge_branch))
        .route("/api/git/discard", post(routes::git::discard_branch))

        .route("/api/history", get(routes::history::get_history))
        .route("/api/goals/{id}/graph", get(routes::goals::get_goal_graph))
        .route("/api/graphs", get(routes::graphs::list_graphs).post(routes::graphs::store_graph))
        .route("/api/graphs/stats", get(routes::graphs::graph_stats))
        .route("/api/graphs/{id}", get(routes::graphs::get_graph))
        .route("/api/providers", get(routes::providers::list_providers))
        .route("/api/providers/costs", get(routes::providers::cost_summary))
        .route("/api/providers/templates", get(routes::providers::list_templates))
        .route("/api/providers/configured", get(routes::providers::list_configured))
        .route("/api/providers/add", post(routes::providers::add_provider))
        .route("/api/providers/test", post(routes::providers::test_provider))
        .route("/api/providers/{id}/toggle", put(routes::providers::toggle_provider))
        .route("/api/providers/{id}/edit", put(routes::providers::update_provider))
        .route("/api/providers/{id}", delete(routes::providers::delete_provider))
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
        .route("/api/supervisor/agent", post(routes::supervisor::add_agent))
        .route("/api/supervisor/presets", get(routes::supervisor::presets))
        .route("/api/supervisor/install-preset", post(routes::supervisor::install_preset))
        .route("/api/supervisor/suggest-agent", post(routes::supervisor::suggest_agent))
        .route("/api/supervisor/dispatch", post(routes::supervisor::dispatch_preset))
        .route("/api/supervisor/task", post(routes::supervisor::add_task))
        .route("/api/supervisor/action", post(routes::supervisor::action))
        .route("/api/supervisor/task-action", post(routes::supervisor::task_action))
        .route("/api/supervisor/session-prompt", post(routes::supervisor::session_prompt))
        .route("/api/supervisor/handover/create", post(routes::supervisor::create_handover))
        .route("/api/supervisor/handover/reply", post(routes::supervisor::reply_handover))
        .route("/api/supervisor/handover/show", get(routes::supervisor::show_handover))
        .route("/api/supervisor/stream", get(routes::supervisor::stream))
        .route("/api/supervisor/daemon/status", get(routes::supervisor::daemon_status))
        .route("/api/supervisor/daemon/start", post(routes::supervisor::daemon_start))
        .route("/api/supervisor/daemon/stop", post(routes::supervisor::daemon_stop))
        // MCP ↔ QLMS bridge (spec §15.2 / PRD Task 2.2)
        .route("/qlms/v1.1/deliver", post(routes::mcp_qlms::deliver))
        .route("/qlms/v1.1/reply", post(routes::mcp_qlms::reply))
        // Dashboard prerequisites (PRD Epic 6)
        .route(
            "/api/values",
            get(routes::dashboard::get_values).post(routes::dashboard::update_values),
        )
        .route("/ws/graph-stream", get(routes::dashboard::graph_stream))
        // Swarm Map data (Task 6.4)
        .route("/api/federation/peers", get(routes::dashboard::get_peers))
        .route(
            "/api/federation/stats",
            get(routes::dashboard::get_federation_stats),
        )
        // Workspace — agent-writable sandbox + file browser + runner
        .route("/api/tools/write_file", post(routes::workspace::write_file))
        .route("/api/tools/exec_file", post(routes::workspace::exec_file))
        .route("/api/tools/web_search", get(routes::workspace::web_search))
        .route("/api/tools/fetch_url", get(routes::workspace::fetch_url))
        // Inbound MCP server — external clients can call QLANG as a tool
        .route("/mcp/v1", post(routes::mcp_server::handle_rpc))
        .route("/api/workspace/tree", get(routes::workspace::tree))
        .route(
            "/api/workspace/file",
            get(routes::workspace::read_file).delete(routes::workspace::delete_file),
        )
        // Presence registry — connected IDEs/agents register, heartbeat,
        // and discover each other for multi-IDE-mesh routing.
        .route("/api/presence", get(routes::presence::list))
        .route("/api/presence/register", post(routes::presence::register))
        .route(
            "/api/presence/heartbeat/{identity}",
            post(routes::presence::heartbeat),
        )
        .route(
            "/api/presence/{identity}/eligibility",
            post(routes::presence::set_eligibility),
        )
        .route(
            "/api/presence/{identity}",
            delete(routes::presence::deregister),
        )
        .route("/api/neo/hardware", get(routes::neo::hardware))
        .route("/api/neo/memory", get(routes::neo::memory))
        .route("/api/neo/status", get(routes::neo::status))
        .route("/api/neo/agents", get(routes::neo::list_agents))
        .route("/api/neo/agents/{id}", get(routes::neo::get_agent))
        .route("/supervisor", get(routes::supervisor::cockpit))
        .route("/supervisor/legacy", get(routes::supervisor::cockpit_legacy));

    let api_router = api_router
        .layer(middleware::from_fn(auth::auth_middleware))
        .with_state(state.clone());

    let router = if let Some(static_dir) = config.static_dir {
        api_router.fallback_service(ServeDir::new(static_dir))
    } else {
        api_router
    };

    let router = router.layer(CorsLayer::permissive());

    Ok((router, state))
}
