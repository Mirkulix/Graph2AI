use axum::extract::State;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::Json;
use futures::stream::Stream;
use serde::Serialize;
use std::convert::Infallible;
use std::sync::Arc;
use tokio_stream::wrappers::ReceiverStream;
use tokio_stream::StreamExt;

use crate::AppState;

#[derive(Serialize)]
pub struct BusStatsResponse {
    pub total_messages: u64,
    pub delivered: u64,
    pub failed: u64,
    pub active_agents: usize,
    pub active_conversations: usize,
}

#[derive(Serialize)]
pub struct ConversationEntry {
    pub key: String,
    pub participants: Vec<String>,
}

#[derive(Serialize)]
struct MessageEvent {
    id: u64,
    from: String,
    to: String,
    intent: String,
    graph_name: String,
    timestamp: u64,
}

/// GET /api/messages/stats — Message bus statistics.
pub async fn bus_stats(State(state): State<Arc<AppState>>) -> Json<BusStatsResponse> {
    let stats = state.message_bus.stats().await;
    Json(BusStatsResponse {
        total_messages: stats.total_messages,
        delivered: stats.delivered,
        failed: stats.failed,
        active_agents: stats.active_agents,
        active_conversations: stats.active_conversations,
    })
}

/// GET /api/messages/agents — Registered agents on the bus.
pub async fn bus_agents(State(state): State<Arc<AppState>>) -> Json<Vec<String>> {
    let agents = state.message_bus.registered_agents().await;
    Json(agents)
}

/// GET /api/messages/conversations — Active conversations.
pub async fn bus_conversations(
    State(state): State<Arc<AppState>>,
) -> Json<Vec<ConversationEntry>> {
    let convs = state.message_bus.active_conversations().await;
    Json(
        convs
            .into_iter()
            .map(|(key, participants)| ConversationEntry { key, participants })
            .collect(),
    )
}

/// GET /api/messages/stream — SSE stream of all messages flowing through the bus.
pub async fn bus_stream(
    State(state): State<Arc<AppState>>,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    let rx = state.message_bus.subscribe().await;
    let stream = ReceiverStream::new(rx).map(|msg| {
        let intent = format!("{:?}", msg.intent);
        let ev = MessageEvent {
            id: msg.id,
            from: msg.from.name.clone(),
            to: msg.to.name.clone(),
            intent,
            graph_name: msg.graph.id.clone(),
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
        };
        let json = serde_json::to_string(&ev).unwrap_or_default();
        Ok(Event::default().data(json))
    });

    Sse::new(stream).keep_alive(KeepAlive::default())
}
