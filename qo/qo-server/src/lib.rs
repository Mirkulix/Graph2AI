pub mod auth;
pub mod routes;

use axum::{
    middleware,
    routing::{delete, get, post, put},
    Router,
};
use qlang_agent::bus::MessageBus;
use qo_agents::{AgentRegistry, AgentRole};
use qo_consciousness::{ConsciousnessState, ConsciousnessStream};
use qo_evolution::{Pattern, PatternDetector, Proposal, ProposalEngine, QuantumState};
use qo_llm::LlmRouter;
use qo_memory::{GraphStore, MemoryContext, ObsidianBridge, Store};
use std::sync::Arc;
use tokio::sync::Mutex;
use tower_http::cors::CorsLayer;
use tower_http::services::ServeDir;

pub struct AppState {
    pub llm: Arc<LlmRouter>,
    pub store: Store,
    pub graph_store: GraphStore,
    pub consciousness: Mutex<ConsciousnessState>,
    pub stream: ConsciousnessStream,
    pub obsidian: ObsidianBridge,
    pub agents: Mutex<AgentRegistry>,
    pub patterns: Mutex<PatternDetector>,
    pub proposals: Mutex<ProposalEngine>,
    pub quantum: Mutex<QuantumState>,
    pub configured_providers: Mutex<Vec<qo_llm::ProviderConfig>>,
    pub memory: Mutex<MemoryContext>,
    /// QLANG Message Bus — routes GraphMessages between AI agents.
    pub message_bus: Arc<MessageBus>,
    /// GPU training state — tracks running training job for SSE streaming.
    pub gpu_training: Arc<routes::gpu_training::GpuTrainingState>,
    /// Evolution daemon (legacy stub — simulated fitness drift).
    /// Still kept for backwards compatibility with any CLI/tests that pin
    /// the old behavior via `use_real_daemon=false`.
    pub evolution_daemon: Arc<
        Mutex<Option<Arc<qlang_runtime::evolution::daemon::EvolutionDaemon>>>,
    >,
    /// REAL evolution daemon — trains ternary MNIST specialists. Default
    /// backend for `/api/evolution/*` endpoints.
    pub real_evolution_daemon: Arc<
        Mutex<Option<Arc<qlang_runtime::evolution::real_daemon::RealEvolutionDaemon>>>,
    >,
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
    let stream = ConsciousnessStream::new(64);
    let consciousness = Mutex::new(ConsciousnessState::default());

    // Load persisted data BEFORE creating AppState (no async runtime yet)
    let mut agents_reg = AgentRegistry::new();
    let mut pattern_det = PatternDetector::new();
    let mut proposal_eng = ProposalEngine::new();
    let mut quantum_st = QuantumState::new(vec![
        "Direkte Ausführung".into(),
        "Dekomposition + Delegation".into(),
        "Recherche zuerst".into(),
        "Kreative Lösung".into(),
    ]);

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

    // Restore patterns
    if let Ok(data) = store.list_patterns() {
        for (_, json) in data {
            if let Ok(p) = serde_json::from_str::<Pattern>(&json) {
                pattern_det.restore_pattern(p);
            }
        }
    }

    // Restore proposals
    if let Ok(data) = store.list_proposals() {
        for (_, json) in data {
            if let Ok(p) = serde_json::from_str::<Proposal>(&json) {
                proposal_eng.restore_proposal(p);
            }
        }
    }

    // Restore quantum state
    if let Ok(Some(json)) = store.load_quantum_state() {
        if let Ok(qs) = serde_json::from_str::<QuantumState>(&json) {
            quantum_st = qs;
        }
    }

    // Load persisted embeddings into vector store for long-term memory
    let mut memory_ctx = MemoryContext::new(384); // all-MiniLM-L6-v2 via candle: 384 dimensions
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

    tracing::info!("Restored: {} goals, {} patterns, {} proposals, gen {}",
        agents_reg.list_goals().len(),
        pattern_det.all_patterns().len(),
        proposal_eng.all().len(),
        quantum_st.generation,
    );

    // Initialize the QLANG Message Bus for AI-to-AI communication
    let message_bus = MessageBus::new();

    let state = Arc::new(AppState {
        llm,
        store,
        graph_store,
        consciousness,
        stream,
        obsidian,
        agents: Mutex::new(agents_reg),
        patterns: Mutex::new(pattern_det),
        proposals: Mutex::new(proposal_eng),
        quantum: Mutex::new(quantum_st),
        configured_providers: Mutex::new(configured_providers),
        memory: Mutex::new(memory_ctx),
        message_bus: message_bus.clone(),
        gpu_training: Arc::new(routes::gpu_training::GpuTrainingState::default()),
        evolution_daemon: Arc::new(Mutex::new(None)),
        real_evolution_daemon: Arc::new(Mutex::new(None)),
    });

    // Register all QO agents on the message bus.
    // Mailboxes are kept alive in background tasks that drain messages.
    {
        use qlang_agent::protocol::{AgentId, Capability};
        let agent_names = ["ceo", "researcher", "developer", "guardian", "strategist", "artisan"];
        for name in &agent_names {
            let agent_id = AgentId {
                name: name.to_string(),
                capabilities: vec![Capability::Execute],
            };
            let mut mailbox = message_bus.register(agent_id).await;
            // Keep mailbox alive in a background task that logs received messages
            let agent_name = name.to_string();
            tokio::spawn(async move {
                loop {
                    match mailbox.recv().await {
                        Some(msg) => {
                            tracing::debug!(
                                "Agent '{}' received QLMS message from '{}' (intent: {:?})",
                                agent_name, msg.from.name, msg.intent
                            );
                        }
                        None => break, // Channel closed
                    }
                }
            });
        }
        tracing::info!("Message bus: {} agents registered with active mailboxes", agent_names.len());
    }

    let api_router = Router::new()
        .route("/api/health", get(routes::health::health))
        .route("/api/chat", post(routes::chat::chat))
        .route("/api/chat/history", get(routes::chat::chat_history))
        .route(
            "/api/consciousness/stream",
            get(routes::consciousness::stream),
        )
        .route(
            "/api/consciousness/state",
            get(routes::consciousness::current_state),
        )
        .route("/api/agents", get(routes::agents::list_agents))
        .route("/api/agents/{role}", get(routes::agents::get_agent))
        .route("/api/goals", get(routes::goals::list_goals))
        .route("/api/goals", post(routes::goals::create_goal))
        .route("/api/goals/{id}", get(routes::goals::get_goal))
        .route("/api/evolution/state", get(routes::evolution::quantum_state))
        .route("/api/evolution/patterns", get(routes::evolution::list_patterns))
        .route("/api/evolution/proposals", get(routes::evolution::list_proposals))
        .route("/api/evolution/proposals/{id}/approve", post(routes::evolution::approve_proposal))
        .route("/api/evolution/proposals/{id}/reject", post(routes::evolution::reject_proposal))
        .route("/api/evolution/analyze", post(routes::evolution::analyze))
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
        .route("/api/simulation/run", post(routes::simulation::run_simulation))
        .route("/api/simulation/strategies", get(routes::simulation::list_strategies))
        .route("/api/memory/stats", get(routes::memory::memory_stats))
        .route("/api/memory/search", get(routes::memory::memory_search))
        .route("/api/messages/stats", get(routes::messages::bus_stats))
        .route("/api/messages/agents", get(routes::messages::bus_agents))
        .route("/api/messages/conversations", get(routes::messages::bus_conversations))
        .route("/api/messages/stream", get(routes::messages::bus_stream))
        .route("/api/proof/tensor-exchange", post(routes::proof::tensor_exchange))
        .route("/api/organism/chat", post(routes::organism::chat))
        .route("/api/organism/evolve", post(routes::organism::evolve))
        .route("/api/organism/status", get(routes::organism::status))
        .route("/api/organism/load-model", post(routes::organism::load_model))
        .route("/api/training/qlang", post(routes::training::train_qlang))
        .route("/api/training/monitor", get(routes::train_monitor::monitor))
        .route("/api/training/gpu", post(routes::gpu_training::start_gpu_training))
        .route("/api/training/gpu/status", get(routes::gpu_training::gpu_training_status))
        .route("/api/training/gpu/stop", post(routes::gpu_training::stop_gpu_training))
        .route("/api/training/gpu/stream", get(routes::gpu_training::gpu_training_stream))
        .route("/api/spiking/run", post(routes::spiking::run_spiking))
        .route("/api/spiking/train", post(routes::spiking::train_spiking))
        .route("/api/spiking/status", get(routes::spiking::spiking_status))
        .route("/api/demo/mnist-igqk", post(routes::demo::start_mnist_igqk_demo))
        .route("/api/qlms/send-model", post(routes::qlms_demo::send_model))
        .route("/api/qlms/receive", post(routes::qlms_demo::receive_model))
        .route("/api/qlms/demo-log", get(routes::qlms_demo::demo_log))
        .route("/api/qlms/federation/gossip", post(routes::qlms_federation::gossip))
        .route("/api/qlms/federation/weights", get(routes::qlms_federation::get_weights))
        .route("/api/qlms/federation/eval", get(routes::qlms_federation::eval_local))
        .route("/api/evolution/start", post(routes::evolution_daemon::start))
        .route("/api/evolution/stop", post(routes::evolution_daemon::stop))
        .route("/api/evolution/status", get(routes::evolution_daemon::status))
        .route("/api/evolution/history", get(routes::evolution_daemon::history))
        .route("/api/evolution/specialists", get(routes::evolution_daemon::specialists))
        .route("/api/evolution/lineage/{id}", get(routes::evolution_daemon::lineage))
        .route("/api/evolution/stream", get(routes::evolution_daemon::stream))
        .route("/api/neo/hardware", get(routes::neo::hardware))
        .route("/api/neo/memory", get(routes::neo::memory))
        .route("/api/neo/status", get(routes::neo::status))
        .route("/api/neo/agents", get(routes::neo::list_agents))
        .route("/api/neo/agents/{id}", get(routes::neo::get_agent))
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
