use axum::{
    extract::{Path, Query, State},
    http::StatusCode,
    Json,
};
use qo_memory::{GraphType, StoredGraph};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;

use crate::AppState;

#[derive(Deserialize)]
pub struct ListQuery {
    pub limit: Option<usize>,
}

#[derive(Serialize)]
pub struct GraphStats {
    pub total_graphs: u64,
    pub by_type: HashMap<String, u64>,
}

/// GET /api/graphs?limit=50
pub async fn list_graphs(
    State(state): State<Arc<AppState>>,
    Query(params): Query<ListQuery>,
) -> Result<Json<Vec<StoredGraph>>, (StatusCode, String)> {
    let limit = params.limit.unwrap_or(50);
    let graphs = state
        .graph_store
        .list_recent(limit)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    Ok(Json(graphs))
}

/// GET /api/graphs/stats
pub async fn graph_stats(
    State(state): State<Arc<AppState>>,
) -> Result<Json<GraphStats>, (StatusCode, String)> {
    let total_graphs = state
        .graph_store
        .count()
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    let all = state
        .graph_store
        .list_recent(10_000)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    let mut by_type: HashMap<String, u64> = HashMap::new();
    for g in &all {
        let key = match g.graph_type {
            GraphType::Chat => "chat",
            GraphType::GoalExecution => "goal_execution",
            GraphType::AgentTask => "agent_task",
            GraphType::Evolution => "evolution",
            GraphType::ValueCheck => "value_check",
        };
        *by_type.entry(key.to_string()).or_insert(0) += 1;
    }

    Ok(Json(GraphStats {
        total_graphs,
        by_type,
    }))
}

/// GET /api/graphs/:id
pub async fn get_graph(
    State(state): State<Arc<AppState>>,
    Path(id): Path<u64>,
) -> Result<Json<StoredGraph>, (StatusCode, String)> {
    let graph = state
        .graph_store
        .get(id)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?
        .ok_or_else(|| (StatusCode::NOT_FOUND, format!("Graph {} not found", id)))?;
    Ok(Json(graph))
}

#[derive(Serialize)]
pub struct StoreGraphResponse {
    pub id: u64,
    pub message: String,
}

/// POST /api/graphs — store a new graph
pub async fn store_graph(
    State(state): State<Arc<AppState>>,
    Json(mut graph): Json<StoredGraph>,
) -> Result<Json<StoreGraphResponse>, (StatusCode, String)> {
    // Ensure timestamp is set
    if graph.timestamp == 0 {
        graph.timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
    }
    let id = state
        .graph_store
        .store(&graph)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    Ok(Json(StoreGraphResponse {
        id,
        message: format!("Graph '{}' gespeichert ({} nodes, {} edges)", graph.title, graph.nodes.len(), graph.edges.len()),
    }))
}
