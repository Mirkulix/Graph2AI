pub mod auth;
pub mod config;
pub mod peer_discovery;
pub mod routes;
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
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::Mutex;
use tower_http::cors::CorsLayer;
use tower_http::services::ServeDir;

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
    });

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

                            // ─── LLM call ─────────────────────────────────────
                            let user_prompt = extract_prompt(&msg);
                            let messages = vec![
                                ("system".to_string(), system_prompt_for(&agent_name).to_string()),
                                ("user".to_string(), user_prompt),
                            ];

                            // Prefer DeepSeek explicitly — auto-router never selects it.
                            // Falls back to whatever tier is available if DeepSeek is offline.
                            let reply_text = match llm
                                .chat_preferring(Some(qo_llm::Tier::DeepSeek), messages)
                                .await
                            {
                                Ok((text, used)) => {
                                    tracing::debug!(?used, "agent '{}' got LLM reply", agent_name);
                                    text
                                }
                                Err(e) => {
                                    tracing::warn!("Agent '{}' LLM call failed: {}", agent_name, e);
                                    format!("[agent '{}' error: {}]", agent_name, e)
                                }
                            };

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

                            let reply_graph = Graph {
                                id: format!("reply-{}-{}", agent_name, reply_id),
                                version: "1.0".to_string(),
                                nodes: vec![],
                                edges: vec![],
                                constraints: vec![],
                                metadata: reply_metadata,
                            };

                            let reply = qlang_agent::protocol::GraphMessage {
                                id: reply_id,
                                from: AgentId {
                                    name: agent_name.clone(),
                                    capabilities: vec![Capability::Execute],
                                },
                                to: msg.from.clone(),
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
