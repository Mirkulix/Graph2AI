use std::path::PathBuf;
use std::sync::Arc;

async fn run_evolution_cycle(state: &Arc<qo_server::AppState>) {
    use qo_evolution::SystemStats;

    // 1. Gather system stats
    let (stats, _cs_energy) = {
        let agents = state.agents.lock().await;
        let cs = state.consciousness.lock().await;
        let agent_list = agents.list_agents();
        let active = agent_list.iter().filter(|a| a.status == qo_agents::AgentStatus::Active).count() as u8;
        let idle = agent_list.iter().filter(|a| a.status == qo_agents::AgentStatus::Idle).count() as u8;
        let completed: u32 = agent_list.iter().map(|a| a.tasks_completed).sum();
        let failed: u32 = agent_list.iter().map(|a| a.tasks_failed).sum();
        (SystemStats {
            total_tasks: completed + failed,
            tasks_completed: completed,
            tasks_failed: failed,
            agents_active: active,
            agents_idle: idle,
            avg_energy: cs.energy,
            completed_streak: 0, // simplified
        }, cs.energy)
    };

    // 2. Run pattern analysis
    let new_patterns = {
        let mut patterns = state.patterns.lock().await;
        patterns.analyze(&stats)
            .iter()
            .map(|p| p.name.clone())
            .collect::<Vec<_>>()
    };

    // 3. Generate proposals
    let new_proposals = {
        let patterns = state.patterns.lock().await;
        let active = patterns.active_patterns();
        let mut proposals = state.proposals.lock().await;
        proposals.generate_from_patterns(&active)
            .iter()
            .map(|p| p.title.clone())
            .collect::<Vec<_>>()
    };

    // 4. Evolve quantum state
    {
        let mut quantum = state.quantum.lock().await;
        if stats.total_tasks > 0 {
            let success_rate = stats.tasks_completed as f32 / stats.total_tasks as f32;
            if success_rate > 0.7 {
                quantum.evolve(1, true, 0.05);
            } else {
                quantum.evolve(1, false, 0.05);
            }
        }
    }

    // 5. Publish activity events
    if !new_patterns.is_empty() || !new_proposals.is_empty() {
        state.stream.publish_activity(
            format!("Evolution: {} Patterns, {} Vorschläge", new_patterns.len(), new_proposals.len()),
            None,
            "info",
        );
    }

    // 6. Log to consciousness
    {
        let mut cs = state.consciousness.lock().await;
        cs.mood = qo_consciousness::Mood::Reflecting;
        state.stream.publish(cs.clone());
    }

    tracing::info!("Evolution cycle: {} patterns, {} proposals", new_patterns.len(), new_proposals.len());
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt::init();

    // Load env vars from ~/.openclaw/.env if it exists
    let env_path = dirs_home().join(".openclaw/.env");
    if env_path.exists() {
        for line in std::fs::read_to_string(&env_path)?.lines() {
            if let Some((key, value)) = line.split_once('=') {
                let key = key.trim();
                let value = value.trim().trim_matches('"');
                if !key.is_empty() && !key.starts_with('#') {
                    std::env::set_var(key, value);
                }
            }
        }
    }

    let cloud_config = {
        let api_key = std::env::var("ANTHROPIC_API_KEY")
            .or_else(|_| std::env::var("DEEPSEEK_API_KEY"))
            .ok();
        let base_url = std::env::var("CLOUD_BASE_URL").ok();
        let model = std::env::var("CLOUD_MODEL").ok();
        match (api_key, base_url, model) {
            (Some(k), Some(u), Some(m)) => Some((k, u, m)),
            _ => None,
        }
    };

    // Resolve paths relative to the binary location
    let exe_dir = std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(|p| p.to_path_buf()))
        .unwrap_or_else(|| PathBuf::from("."));
    // Binary is in target/release/ — project root is 2 levels up
    let project_root = exe_dir.join("../../").canonicalize().unwrap_or_else(|_| PathBuf::from("."));

    let static_dir = project_root.join("frontend/dist");
    let static_dir = if static_dir.exists() {
        Some(static_dir)
    } else {
        // Fallback: try relative to CWD
        let cwd_static = PathBuf::from("frontend/dist");
        if cwd_static.exists() { Some(cwd_static) } else { None }
    };

    let config = qo_server::QoConfig {
        port: std::env::var("QO_PORT")
            .ok()
            .and_then(|p| p.parse().ok())
            .unwrap_or(4646),
        groq_api_key: std::env::var("GROQ_API_KEY").ok(),
        cloud_config,
        ollama_url: std::env::var("OLLAMA_URL")
            .ok()
            .or_else(|| Some("http://localhost:11434".to_string())),
        ollama_model: std::env::var("OLLAMA_MODEL")
            .ok()
            .or_else(|| Some("orbit-companion-ft-q4".to_string())),
        data_dir: std::env::var("QO_DATA_DIR")
            .ok()
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from("data")),
        obsidian_vault: dirs_home().join("Dokumente/Obsidian Vault/QO"),
        static_dir,
        auth_token: std::env::var("QO_AUTH_TOKEN").ok(),
    };

    if config.static_dir.is_some() {
        tracing::info!("Frontend: {:?}", config.static_dir.as_ref().unwrap());
    } else {
        tracing::warn!("Frontend not found! Build with: cd frontend && npm run build");
    }

    let port = config.port;
    let (app, state) = qo_server::build_app(config).await.map_err(|e| format!("{e}"))?;

    // Import Orbit data if available
    let import_dir = std::path::PathBuf::from("data/orbit-import");
    if import_dir.exists() {
        match qo_memory::import_orbit_data(&state.store, &import_dir) {
            Ok(result) => {
                if result.messages + result.goals + result.patterns + result.proposals > 0 {
                    tracing::info!(
                        "Orbit import: {} messages, {} goals, {} patterns, {} proposals",
                        result.messages, result.goals, result.patterns, result.proposals
                    );
                    // Rename dir to prevent re-import
                    let done_dir = std::path::PathBuf::from("data/orbit-imported");
                    let _ = std::fs::rename(&import_dir, &done_dir);
                }
            }
            Err(e) => tracing::warn!("Orbit import failed: {e}"),
        }
    }

    // Idle detection — check every 5 minutes, regen energy if idle
    let idle_state = state.clone();
    tokio::spawn(async move {
        let mut interval = tokio::time::interval(std::time::Duration::from_secs(300));
        loop {
            interval.tick().await;
            let mut cs = idle_state.consciousness.lock().await;
            // Regen energy when idle
            let agents = idle_state.agents.lock().await;
            if agents.active_count() == 0 {
                cs.energy = (cs.energy + 5.0).min(100.0);
            }
            cs.process_event(&qo_consciousness::StateEvent::Idle);
            idle_state.stream.publish(cs.clone());
            drop(agents);
            drop(cs);
        }
    });

    // Auto-evolution cycle (every 10 minutes)
    let evo_state = state.clone();
    tokio::spawn(async move {
        // Wait 60 seconds before first run
        tokio::time::sleep(std::time::Duration::from_secs(60)).await;
        let mut interval = tokio::time::interval(std::time::Duration::from_secs(600));
        loop {
            interval.tick().await;
            run_evolution_cycle(&evo_state).await;
        }
    });

    // Telegram bot (if configured)
    if let Ok(telegram_token) = std::env::var("QO_TELEGRAM_TOKEN") {
        let telegram_chat_id = std::env::var("QO_TELEGRAM_CHAT_ID")
            .ok()
            .and_then(|s| s.parse::<i64>().ok());
        let qo_url = format!("http://127.0.0.1:{}", port);
        let bot = qo_telegram::TelegramBot::new(telegram_token, qo_url, telegram_chat_id);
        tokio::spawn(async move {
            bot.run().await;
        });
        tracing::info!("Telegram bot started");
    }

    let addr = format!("0.0.0.0:{port}");
    tracing::info!("QO starting on {addr}");
    let listener = tokio::net::TcpListener::bind(&addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}

fn dirs_home() -> PathBuf {
    std::env::var("HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("/tmp"))
}
