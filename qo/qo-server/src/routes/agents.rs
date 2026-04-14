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
