use axum::{
    extract::{Query, State},
    Json,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

use crate::AppState;

#[derive(Serialize)]
pub struct MemoryStatsResponse {
    pub count: usize,
    pub dimension: usize,
}

#[derive(Deserialize)]
pub struct MemorySearchQuery {
    pub q: String,
    #[serde(default = "default_k")]
    pub k: usize,
}

fn default_k() -> usize {
    5
}

#[derive(Serialize)]
pub struct MemorySearchResult {
    pub key: String,
    pub score: f32,
}

pub async fn memory_stats(
    State(state): State<Arc<AppState>>,
) -> Json<MemoryStatsResponse> {
    let mem = state.memory.lock().await;
    Json(MemoryStatsResponse {
        count: mem.count(),
        dimension: 64,
    })
}

pub async fn memory_search(
    State(state): State<Arc<AppState>>,
    Query(params): Query<MemorySearchQuery>,
) -> Json<Vec<MemorySearchResult>> {
    let mem = state.memory.lock().await;
    let results = mem.recall(&params.q, params.k);
    Json(
        results
            .into_iter()
            .map(|(key, score)| MemorySearchResult { key, score })
            .collect(),
    )
}
