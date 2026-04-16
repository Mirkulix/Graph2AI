use axum::{
    extract::{Path, State},
    Json,
};
use qo_agents::{Agent, AgentRole, AgentSummary};
use serde::Serialize;
use std::sync::Arc;

use crate::AppState;

#[derive(Serialize)]
pub struct AgentListResponse {
    pub agents: Vec<AgentSummary>,
    pub active_count: u8,
    pub idle_count: u8,
}

pub async fn list_agents(
    State(state): State<Arc<AppState>>,
) -> Result<Json<AgentListResponse>, (axum::http::StatusCode, String)> {
    let registry = state.agents.lock().await;
    let agents = registry.list_agents();
    let active_count = registry.active_count();
    let idle_count = registry.idle_count();
    Ok(Json(AgentListResponse {
        agents,
        active_count,
        idle_count,
    }))
}

#[derive(Serialize)]
pub struct AgentModelEntry {
    pub role: String,
    pub preferred_tier: String,
    /// One of "active" / "fallback" / "auto".
    pub effective: &'static str,
    pub model: String,
}

#[derive(Serialize)]
pub struct AgentModelsResponse {
    pub bindings: Vec<AgentModelEntry>,
}

/// GET /api/agents/models — per-role tier + resolved model name.
/// Feeds the Mission-Control Agent-Roster so operators see which LLM
/// each agent is currently hitting (without guessing).
pub async fn list_agent_models(
    State(state): State<Arc<AppState>>,
) -> Json<AgentModelsResponse> {
    let stats = state.llm.provider_stats();
    // Map provider names to their live model string.
    let lookup_model = |tier: &str| -> (String, &'static str) {
        let (name_match, fallback_name) = match tier {
            "cloud" | "deepseek" | "anthropic" | "claude" | "openai" | "gemini" => {
                ("Cloud", tier)
            }
            "groq" => ("Groq", "groq"),
            "ollama" | "local" => ("Ollama (local)", "ollama"),
            _ => ("Groq", "auto"),
        };
        let hit = stats
            .iter()
            .find(|s| s.name.starts_with(name_match) && s.status == "active");
        if let Some(h) = hit {
            (h.model.clone(), "active")
        } else {
            // Fall back to the first active provider so the UI stays
            // honest about what will actually be used.
            let fb = stats.iter().find(|s| s.status == "active");
            match fb {
                Some(f) => (f.model.clone(), "fallback"),
                None => (format!("unavailable ({fallback_name})"), "auto"),
            }
        }
    };

    let bindings = qo_agents::AgentRole::ALL
        .iter()
        .map(|role| {
            let hint = role.preferred_provider();
            let (model, effective) = lookup_model(hint);
            AgentModelEntry {
                role: role.name().to_string(),
                preferred_tier: hint.to_string(),
                effective,
                model,
            }
        })
        .collect();

    Json(AgentModelsResponse { bindings })
}

pub async fn get_agent(
    State(state): State<Arc<AppState>>,
    Path(role_str): Path<String>,
) -> Result<Json<Agent>, (axum::http::StatusCode, String)> {
    let role = parse_role(&role_str).ok_or_else(|| {
        (
            axum::http::StatusCode::NOT_FOUND,
            format!("Unknown agent role: {}", role_str),
        )
    })?;

    let registry = state.agents.lock().await;
    let agent = registry.get_agent(role).ok_or_else(|| {
        (
            axum::http::StatusCode::NOT_FOUND,
            format!("Agent not found: {}", role_str),
        )
    })?;

    Ok(Json(agent.clone()))
}

fn parse_role(s: &str) -> Option<AgentRole> {
    match s.to_lowercase().as_str() {
        "ceo" => Some(AgentRole::Ceo),
        "researcher" => Some(AgentRole::Researcher),
        "developer" => Some(AgentRole::Developer),
        "guardian" => Some(AgentRole::Guardian),
        "strategist" => Some(AgentRole::Strategist),
        "artisan" => Some(AgentRole::Artisan),
        _ => None,
    }
}
