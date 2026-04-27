//! POST /api/llm/proxy — IDE-side LLM delegation endpoint.
//!
//! IDE extensions (VS Code, Theia, …) call this endpoint instead of
//! holding their own provider API keys. The qo server holds the keys
//! and selects the appropriate `LlmRouter` tier on the IDE's behalf.
//!
//! Auth: this endpoint sits inside the `/api/*` namespace, which is
//! already wrapped by the optional `QO_AUTH_TOKEN` bearer middleware.
//! No additional QLMS HMAC verification for this MVP.
//!
//! Provider mapping (see `qo_llm::Tier`):
//!   - "anthropic" / "openai"        → Tier::Cloud   (OpenAI-compatible
//!                                       CloudClient — Anthropic uses
//!                                       the cloud tier with base_url
//!                                       set to api.anthropic.com via
//!                                       `install_provider`).
//!   - "deepseek"                    → Tier::DeepSeek
//!   - "groq"                        → Tier::Groq
//!   - "ollama" / "local"            → Tier::Local
//!   - "cloud"                       → Tier::Cloud (raw escape hatch)

use axum::{extract::State, http::StatusCode, Json};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::process::Command;

use crate::AppState;

/// Spawn `claude` as a FULL Claude Code agent — Read/Edit/Write/Bash/Grep/Glob
/// tools enabled, scoped to the OrbitQLang repo. This turns each call into a
/// real agentic investigation: the model can grep the codebase, read files,
/// run commands, and answer with grounded facts instead of hallucinating.
///
/// Bounded by `--max-budget-usd 0.50` (per call) and a 180-second hard timeout
/// so a stuck session can't block the swarm orchestrator forever.
pub(crate) async fn call_claude_cli_agent(
    model: &str,
    system_prompt: Option<&str>,
    user_content: &str,
) -> Result<String, String> {
    let bin = std::env::var("CLAUDE_CLI_PATH").unwrap_or_else(|_| "claude".to_string());
    let workspace = std::env::var("ORBITQ_REPO")
        .unwrap_or_else(|_| "c:/Users/a.b/Graph/OrbitQLang".to_string());

    let mut cmd = Command::new(&bin);
    cmd.arg("-p")
        .arg(user_content)
        .arg("--model").arg(model)
        .arg("--output-format").arg("text")
        .arg("--no-session-persistence")
        .arg("--allow-dangerously-skip-permissions")
        .arg("--max-budget-usd").arg("0.50")
        .arg("--add-dir").arg(&workspace)
        .arg("--tools").arg("Read,Edit,Write,Bash,Grep,Glob")
        .current_dir(&workspace);

    if let Some(sys) = system_prompt {
        if !sys.trim().is_empty() {
            cmd.arg("--system-prompt").arg(sys);
        }
    }
    cmd.env_remove("CLAUDECODE");

    let fut = cmd.output();
    let output = tokio::time::timeout(std::time::Duration::from_secs(180), fut)
        .await
        .map_err(|_| "claude-cli-agent: timeout after 180s".to_string())?
        .map_err(|e| format!("claude-cli-agent spawn: {} (binary: '{}')", e, bin))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        let stdout = String::from_utf8_lossy(&output.stdout);
        return Err(format!(
            "claude-cli-agent exit={:?}: {} {}",
            output.status.code(),
            stderr.chars().take(400).collect::<String>(),
            stdout.chars().take(400).collect::<String>(),
        ));
    }
    let text = String::from_utf8_lossy(&output.stdout).trim().to_string();
    if text.is_empty() {
        return Err("claude-cli-agent returned empty output".to_string());
    }
    Ok(text)
}

/// Spawn the `claude` Code CLI in non-interactive print mode and return
/// the model's stdout. Uses the user's logged-in Max subscription, so
/// there are no per-call API costs. Requires the `claude` binary on PATH
/// (or set CLAUDE_CLI_PATH env var). Tools are disabled — this is a
/// pure chat-completion call, not an agentic Claude Code session.
async fn call_claude_cli(
    model: &str,
    system_prompt: Option<&str>,
    user_content: &str,
) -> Result<String, String> {
    let bin = std::env::var("CLAUDE_CLI_PATH").unwrap_or_else(|_| "claude".to_string());

    let mut cmd = Command::new(&bin);
    cmd.arg("-p")
        .arg(user_content)
        .arg("--model").arg(model)
        .arg("--output-format").arg("text")
        .arg("--no-session-persistence")
        .arg("--tools").arg("");

    if let Some(sys) = system_prompt {
        if !sys.trim().is_empty() {
            cmd.arg("--system-prompt").arg(sys);
        }
    }

    // Strip CLAUDECODE so a Claude-Code-spawned qo can still invoke `claude`.
    cmd.env_remove("CLAUDECODE");

    let output = cmd
        .output()
        .await
        .map_err(|e| format!("claude-cli spawn failed: {} (binary: '{}')", e, bin))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        let stdout = String::from_utf8_lossy(&output.stdout);
        return Err(format!(
            "claude-cli exit={:?}: {} {}",
            output.status.code(),
            stderr.chars().take(300).collect::<String>(),
            stdout.chars().take(300).collect::<String>(),
        ));
    }

    let text = String::from_utf8_lossy(&output.stdout).trim().to_string();
    if text.is_empty() {
        return Err("claude-cli returned empty output".to_string());
    }
    Ok(text)
}

/// Direct Anthropic /v1/messages call. The qo LlmRouter routes Anthropic
/// through the OpenAI-shaped CloudClient which doesn't actually work
/// against api.anthropic.com (no /chat/completions there). Until qo-llm
/// gets a dedicated Anthropic client, the proxy bypasses the router for
/// this provider and hits Anthropic's native endpoint directly using the
/// api_key from the stored provider config.
async fn call_anthropic_direct(
    api_key: &str,
    base_url: &str,
    model: &str,
    messages: Vec<(String, String)>,
) -> Result<String, String> {
    let url = format!("{}/v1/messages", base_url.trim_end_matches('/').trim_end_matches("/v1"));

    // Anthropic does not accept role="system" in messages[]; extract it.
    let (system_prompt, user_msgs): (Vec<_>, Vec<_>) = messages
        .into_iter()
        .partition(|(role, _)| role == "system");
    let system = system_prompt
        .into_iter()
        .map(|(_, c)| c)
        .collect::<Vec<_>>()
        .join("\n\n");

    let msgs_json: Vec<serde_json::Value> = user_msgs
        .into_iter()
        .map(|(role, content)| serde_json::json!({
            "role": if role == "assistant" { "assistant" } else { "user" },
            "content": content,
        }))
        .collect();

    if msgs_json.is_empty() {
        return Err("anthropic needs at least one user/assistant message".to_string());
    }

    let mut body = serde_json::json!({
        "model": model,
        "max_tokens": 4096,
        "messages": msgs_json,
    });
    if !system.is_empty() {
        body["system"] = serde_json::Value::String(system);
    }

    let client = reqwest::Client::new();
    let resp = client
        .post(&url)
        .header("Content-Type", "application/json")
        .header("x-api-key", api_key)
        .header("anthropic-version", "2023-06-01")
        .json(&body)
        .send()
        .await
        .map_err(|e| format!("anthropic request failed: {}", e))?;

    let status = resp.status();
    let text = resp.text().await.map_err(|e| format!("anthropic body read: {}", e))?;
    if !status.is_success() {
        return Err(format!("anthropic HTTP {}: {}", status.as_u16(), text.chars().take(400).collect::<String>()));
    }

    let parsed: serde_json::Value = serde_json::from_str(&text)
        .map_err(|e| format!("anthropic JSON parse: {} (body: {})", e, text.chars().take(200).collect::<String>()))?;
    let content = parsed["content"]
        .as_array()
        .and_then(|arr| arr.iter().find(|b| b["type"] == "text"))
        .and_then(|b| b["text"].as_str())
        .ok_or("anthropic: no text block in content[]")?;
    Ok(content.to_string())
}

#[derive(Debug, Deserialize)]
pub struct LlmProxyRequest {
    /// Provider tier hint:
    /// "anthropic" | "deepseek" | "groq" | "ollama" | "openai" | "cloud" | "local"
    pub provider: String,
    /// Optional model override. If `None`, the provider's default is used.
    pub model: Option<String>,
    /// OpenAI-style messages: `[{role: "system"|"user"|"assistant", content: "..."}]`
    pub messages: Vec<ProxyMessage>,
}

#[derive(Debug, Deserialize)]
pub struct ProxyMessage {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Serialize)]
pub struct LlmProxyResponse {
    pub provider: String,
    pub model: String,
    pub content: String,
    pub latency_ms: u64,
}

#[derive(Debug, Serialize)]
pub struct LlmProxyError {
    pub error: String,
}

/// POST /api/llm/proxy
pub async fn proxy_chat(
    State(state): State<Arc<AppState>>,
    Json(req): Json<LlmProxyRequest>,
) -> Result<Json<LlmProxyResponse>, (StatusCode, Json<LlmProxyError>)> {
    if req.messages.is_empty() {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(LlmProxyError {
                error: "messages empty".into(),
            }),
        ));
    }

    let msgs: Vec<(String, String)> = req
        .messages
        .into_iter()
        .map(|m| (m.role, m.content))
        .collect();

    let model = req.model.clone();
    let model_str = model.clone().unwrap_or_else(|| "default".to_string());

    let start = std::time::Instant::now();
    let provider_lc = req.provider.to_lowercase();

    // claude-cli-agent: spawn a FULL Claude Code session with Read/Edit/
    // Write/Bash/Grep/Glob tools scoped to the OrbitQLang repo. Each call
    // is a real agentic investigation, not a pure chat. ~30s+ latency,
    // bounded by --max-budget-usd 0.50 and a 180s timeout.
    let result: Result<String, String> = if provider_lc == "claude-cli-agent" || provider_lc == "claude-agent" {
        let model = model.clone().unwrap_or_else(|| "sonnet".to_string());
        let mut sys_parts = Vec::new();
        let mut user_parts = Vec::new();
        for (role, content) in &msgs {
            if role == "system" { sys_parts.push(content.clone()); }
            else { user_parts.push(format!("{}: {}", role, content)); }
        }
        let sys = if sys_parts.is_empty() { None } else { Some(sys_parts.join("\n\n")) };
        let user = user_parts.join("\n\n");
        call_claude_cli_agent(&model, sys.as_deref(), &user).await
    }
    // claude-cli: shell out to the user's logged-in Claude Code CLI.
    // No API credits needed — uses the Max subscription.
    else if provider_lc == "claude-cli" || provider_lc == "claude" {
        let model = model.clone().unwrap_or_else(|| "sonnet".to_string());
        // Pull system from messages[role=system], collapse user/assistant
        // turns into one user prompt (claude -p takes a single positional).
        let mut sys_parts = Vec::new();
        let mut user_parts = Vec::new();
        for (role, content) in &msgs {
            if role == "system" { sys_parts.push(content.clone()); }
            else { user_parts.push(format!("{}: {}", role, content)); }
        }
        let sys = if sys_parts.is_empty() { None } else { Some(sys_parts.join("\n\n")) };
        let user = user_parts.join("\n\n");
        call_claude_cli(&model, sys.as_deref(), &user).await
    }
    // Anthropic bypasses the LlmRouter and hits /v1/messages directly,
    // because the OpenAI-shaped CloudClient cannot speak Anthropic's
    // native shape. Pull the api_key from the stored provider entry.
    else if provider_lc == "anthropic" {
        let stored = state
            .store
            .get_provider("anthropic")
            .ok()
            .flatten()
            .and_then(|j| serde_json::from_str::<qo_llm::ProviderConfig>(&j).ok());
        if let Some(cfg) = stored {
            if !cfg.enabled {
                Err("anthropic provider is disabled in cockpit — toggle it on".to_string())
            } else if cfg.api_key.is_empty() {
                Err("anthropic provider has empty api_key — re-add via cockpit".to_string())
            } else {
                let base = cfg.base_url.unwrap_or_else(|| "https://api.anthropic.com".to_string());
                let mdl = model.unwrap_or(cfg.model);
                call_anthropic_direct(&cfg.api_key, &base, &mdl, msgs).await
            }
        } else {
            Err("anthropic provider not found in store — add it in cockpit first".to_string())
        }
    } else {
        let tier = match provider_lc.as_str() {
            "deepseek" => qo_llm::Tier::DeepSeek,
            "groq" => qo_llm::Tier::Groq,
            "ollama" | "local" => qo_llm::Tier::Local,
            "openai" | "cloud" => qo_llm::Tier::Cloud,
            other => {
                return Err((
                    StatusCode::BAD_REQUEST,
                    Json(LlmProxyError {
                        error: format!("unknown provider '{}'", other),
                    }),
                ))
            }
        };
        state
            .llm
            .chat_with_model(Some(tier), model, msgs)
            .await
            .map(|(c, _t)| c)
            .map_err(|e| e.to_string())
    };
    let latency_ms = start.elapsed().as_millis() as u64;

    match result {
        Ok(content) => {
            tracing::info!(
                "llm_proxy: provider={} model={} latency={}ms len={}",
                req.provider,
                model_str,
                latency_ms,
                content.len()
            );
            Ok(Json(LlmProxyResponse {
                provider: req.provider,
                model: model_str,
                content,
                latency_ms,
            }))
        }
        Err(e) => {
            tracing::warn!("llm_proxy failed: {}", e);
            Err((
                StatusCode::BAD_GATEWAY,
                Json(LlmProxyError {
                    error: format!("llm error: {}", e),
                }),
            ))
        }
    }
}
