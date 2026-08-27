use std::path::PathBuf;


#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt::init();

    // Load env vars from (in order): project-root .env, ~/.openclaw/.env.
    // Project-root .env wins (gitignored; preferred per-repo place for
    // provider API keys). Both files are tolerated-missing.
    for env_path in [
        PathBuf::from(".env"),
        dirs_home().join(".openclaw/.env"),
    ] {
        if !env_path.exists() {
            continue;
        }
        tracing::info!("Loading env from {}", env_path.display());
        if let Ok(contents) = std::fs::read_to_string(&env_path) {
            for line in contents.lines() {
                let line = line.trim();
                if line.is_empty() || line.starts_with('#') {
                    continue;
                }
                if let Some((key, value)) = line.split_once('=') {
                    let key = key.trim();
                    let value = value.trim().trim_matches('"').trim_matches('\'');
                    // Respect existing env: shell > file, so operators can
                    // override on the command line without editing .env.
                    if !key.is_empty() && std::env::var(key).is_err() {
                        std::env::set_var(key, value);
                    }
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

    // The workspace boundary the server may index. Defaults to the repo the
    // binary lives in; an operator can point it at any directory with
    // QO_WORKSPACE (e.g. the project an agent team is actually working on).
    let workspace_root = std::env::var("QO_WORKSPACE")
        .map(PathBuf::from)
        .unwrap_or_else(|_| project_root.clone());

    let static_dir = workspace_root.join("frontend/dist");
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
        workspace_root: workspace_root,
        obsidian_vault: dirs_home().join("Dokumente/Obsidian Vault/QO"),
        static_dir: static_dir.clone(),
        auth_token: std::env::var("QO_AUTH_TOKEN").ok(),
        cors_origins: std::env::var("QO_CORS_ORIGINS")
            .map(|s| s.split(',').map(|o| o.trim().to_string()).collect())
            .unwrap_or_default(),
        api_keys_path: std::env::var("QO_API_KEYS_PATH")
            .ok()
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from(".qlang/api_keys.json")),
        max_body_bytes: std::env::var("QO_MAX_BODY_BYTES")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(16 * 1024 * 1024),
        rate_per_second: std::env::var("QO_RATE_PER_SEC")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(50),
        rate_burst_size: std::env::var("QO_RATE_BURST")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(200),
        llm_routing: qo_server::config::LlmRoutingConfig::load_or_default(".qlang/llm_routing.toml"),
    };

    if config.static_dir.is_some() {
        tracing::info!("Frontend: {:?}", config.static_dir.as_ref().unwrap());
    } else {
        tracing::warn!("Frontend not found! Build with: cd frontend && npm run build");
    }

    let port = config.port;
    let auth_token_for_discovery = config.auth_token.clone();
    let has_auth_token = config.auth_token.is_some();
    let (app, state) = qo_server::build_app(config).await.map_err(|e| format!("{e}"))?;

    // PRD Task 4.2: opt-in multi-node gossip loop. No-op unless
    // `PEER_DISCOVERY_SEEDS` is set, so single-node dev usage is unaffected.
    qo_server::peer_discovery::spawn_if_enabled(
        port,
        auth_token_for_discovery,
        state.gossip_stats.clone(),
    );

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

    // Without a token every route is unauthenticated, including ones that
    // execute code (`/api/tools/exec_file`), rewrite git state and read back
    // stored provider keys. Binding those to 0.0.0.0 would publish them to
    // the network, so an instance with *no* way to authenticate is confined
    // to loopback. Either a token or at least one issued API key counts as a
    // way to authenticate.
    let has_seats = !state.api_keys.read().await.keys.is_empty();
    let bind_host = if has_auth_token || has_seats {
        "0.0.0.0"
    } else {
        tracing::warn!(
            "No QO_AUTH_TOKEN and no API keys — every route is unauthenticated. \
             Binding to 127.0.0.1 only. Set QO_AUTH_TOKEN or issue an API key \
             (.qlang/api_keys.json) to accept remote clients."
        );
        "127.0.0.1"
    };

    let addr = format!("{bind_host}:{port}");
    tracing::info!("QO starting on {addr}");
    let listener = tokio::net::TcpListener::bind(&addr).await?;
    // `into_make_service_with_connect_info` populates the peer socket address,
    // which the per-IP rate limiter keys on.
    axum::serve(
        listener,
        app.into_make_service_with_connect_info::<std::net::SocketAddr>(),
    )
    .await?;

    Ok(())
}

fn dirs_home() -> PathBuf {
    std::env::var("HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("/tmp"))
}
