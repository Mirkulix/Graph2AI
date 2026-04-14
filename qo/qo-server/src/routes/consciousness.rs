use axum::{
    extract::State,
    response::{
        sse::{Event, Sse},
        IntoResponse, Json,
    },
};
use std::sync::Arc;
use tokio_stream::wrappers::BroadcastStream;
use tokio_stream::StreamExt;

use crate::AppState;

pub async fn stream(
    State(state): State<Arc<AppState>>,
) -> impl IntoResponse {
    let rx = state.stream.subscribe();
    let broadcast_stream = BroadcastStream::new(rx);

    // Filter out lagged errors, map to SSE events
    // BroadcastEvent is tagged with "type" so the frontend can distinguish
    // "state" events from "activity" events
    let sse_stream = broadcast_stream.filter_map(|result| match result {
        Ok(event) => {
            let data = serde_json::to_string(&event).unwrap_or_default();
            Some(Ok::<Event, std::convert::Infallible>(
                Event::default().data(data),
            ))
        }
        Err(_lagged) => None, // drop lagged messages
    });

    Sse::new(sse_stream)
}

pub async fn current_state(
    State(state): State<Arc<AppState>>,
) -> impl IntoResponse {
    let cs = state.consciousness.lock().await;
    Json(cs.clone())
}
