//! QO Dashboard back-end prerequisites (PRD Epic 6).
//!
//! Two routes that unblock the frontend Mission Control + Values Radar
//! without committing to the full DAG visualisation yet:
//!
//!   GET  /api/values
//!       → current `ValueScores` snapshot (the 5 Werte). Read by
//!         `ValuesRadar.tsx` (Task 6.3).
//!
//!   POST /api/values
//!       → setter for the 5 Werte, clamped to `[0.0, 1.0]` per field.
//!         Used by the Guardian agent to push updated scores and by
//!         integration tests to stage radar snapshots.
//!
//!   GET  /ws/graph-stream
//!       → WebSocket that streams lightweight `GraphEvent` frames each
//!         time something flows through the QLANG message bus. The
//!         events are JSON — small, easy to render in `MissionControl.tsx`
//!         (Task 6.1) without paying the cost of the full `GraphMessage`.
//!
//! The values store and the broadcast channel live on [`AppState`]; this
//! module only owns the handler glue.

use axum::{
    extract::{
        ws::{Message, WebSocket, WebSocketUpgrade},
        State,
    },
    response::IntoResponse,
    Json,
};
use qo_values::{Value, ValueScores};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

use crate::AppState;

// ---------------------------------------------------------------------------
// GraphEvent — tiny payload broadcast over /ws/graph-stream
// ---------------------------------------------------------------------------

/// A node in the live DAG visualisation.
///
/// Deliberately small and flat — the dashboard needs shape + color,
/// not the full `GraphMessage`. A fat payload would turn the
/// `/ws/graph-stream` into a network hog the moment traffic picks up.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphEvent {
    /// UNIX milliseconds, for ordering and relative timing in the UI.
    pub timestamp_ms: u64,
    pub from: String,
    pub to: String,
    /// Serialised [`qlang_agent::protocol::MessageIntent`] variant name
    /// (e.g. "Execute", "Compress", "Result"). Kept as a string so the
    /// UI can display it without pulling the full intent schema.
    pub intent: String,
    /// Envelope size on the wire. Feeds the "edge thickness" animation.
    pub size_bytes: u32,
    /// Optional status label: "ok", "error", "pending". Allows the UI to
    /// colour edges red when an agent returned an error without
    /// requiring a schema of error codes.
    pub status: Option<String>,
}

impl GraphEvent {
    /// Build a fresh event with the current wall-clock timestamp.
    pub fn now(from: &str, to: &str, intent: &str, size_bytes: u32) -> Self {
        let ts = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);
        Self {
            timestamp_ms: ts,
            from: from.to_string(),
            to: to.to_string(),
            intent: intent.to_string(),
            size_bytes,
            status: None,
        }
    }
}

// ---------------------------------------------------------------------------
// /api/values  (GET + POST)
// ---------------------------------------------------------------------------

#[derive(Debug, Serialize)]
pub struct ValuesResponse {
    pub scores: ValueScores,
    pub average: f32,
}

impl ValuesResponse {
    fn from_scores(scores: ValueScores) -> Self {
        let average = scores.average();
        Self { scores, average }
    }
}

/// Body for `POST /api/values`. Any omitted field keeps its current value.
#[derive(Debug, Deserialize, Default)]
pub struct UpdateValuesRequest {
    pub achtsamkeit: Option<f32>,
    pub anerkennung: Option<f32>,
    pub aufmerksamkeit: Option<f32>,
    pub entwicklung: Option<f32>,
    pub sinn: Option<f32>,
}

pub async fn get_values(State(state): State<Arc<AppState>>) -> Json<ValuesResponse> {
    let scores = state.values.lock().await.clone();
    Json(ValuesResponse::from_scores(scores))
}

pub async fn update_values(
    State(state): State<Arc<AppState>>,
    Json(req): Json<UpdateValuesRequest>,
) -> Json<ValuesResponse> {
    let mut guard = state.values.lock().await;
    if let Some(v) = req.achtsamkeit {
        guard.set(Value::Achtsamkeit, v);
    }
    if let Some(v) = req.anerkennung {
        guard.set(Value::Anerkennung, v);
    }
    if let Some(v) = req.aufmerksamkeit {
        guard.set(Value::Aufmerksamkeit, v);
    }
    if let Some(v) = req.entwicklung {
        guard.set(Value::Entwicklung, v);
    }
    if let Some(v) = req.sinn {
        guard.set(Value::Sinn, v);
    }
    let scores = guard.clone();
    drop(guard);
    Json(ValuesResponse::from_scores(scores))
}

// ---------------------------------------------------------------------------
// /ws/graph-stream  (WebSocket)
// ---------------------------------------------------------------------------

/// Upgrade handler: splits the socket and pipes broadcast events to the
/// client as JSON text frames until the client disconnects.
pub async fn graph_stream(
    ws: WebSocketUpgrade,
    State(state): State<Arc<AppState>>,
) -> impl IntoResponse {
    let rx = state.graph_events_tx.subscribe();
    ws.on_upgrade(move |socket| handle_ws(socket, rx))
}

async fn handle_ws(
    mut socket: WebSocket,
    mut rx: tokio::sync::broadcast::Receiver<GraphEvent>,
) {
    // Greet the client so the UI can flip from "connecting" to "live"
    // without waiting for the first real event.
    let hello = serde_json::json!({ "type": "hello", "protocol": "qlms/graph-stream/v1" });
    if socket
        .send(Message::Text(hello.to_string().into()))
        .await
        .is_err()
    {
        return;
    }

    loop {
        tokio::select! {
            // Any client message (ping / close / text). We drain them so
            // the socket stays half-duplex-clean and detect disconnects.
            msg = socket.recv() => {
                match msg {
                    Some(Ok(Message::Close(_))) | None => break,
                    Some(Err(_)) => break,
                    _ => continue,
                }
            }
            // Broadcast event → forward as a JSON text frame.
            evt = rx.recv() => {
                match evt {
                    Ok(event) => {
                        let line = match serde_json::to_string(&event) {
                            Ok(s) => s,
                            Err(_) => continue,
                        };
                        if socket.send(Message::Text(line.into())).await.is_err() {
                            break;
                        }
                    }
                    // Lagged — we missed some events. Tell the client so
                    // the UI can flash a "stream catching up" indicator.
                    Err(tokio::sync::broadcast::error::RecvError::Lagged(n)) => {
                        let lag = serde_json::json!({ "type": "lagged", "missed": n });
                        if socket
                            .send(Message::Text(lag.to_string().into()))
                            .await
                            .is_err()
                        {
                            break;
                        }
                    }
                    Err(tokio::sync::broadcast::error::RecvError::Closed) => break,
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
//
// The Axum handlers here are thin glue over `ValueScores` (already unit-
// tested in the `qo-values` crate) and a `broadcast::Sender<GraphEvent>`.
// Building the full `AppState` for a handler test pulls in MNIST, the
// Store, the evolution daemon, etc. — too heavy for the signal.
//
// We therefore unit-test the *values-update pure function* and the
// broadcast channel plumbing directly. End-to-end handler behaviour is
// covered by the server's integration tests (qo/qo-server/tests/).
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// The inner logic of [`update_values`] without the Axum/AppState wrapper.
    /// Keeping this as its own helper lets us unit-test clamping + partial
    /// update semantics without spinning up the full server state.
    fn apply_update(scores: &mut ValueScores, req: &UpdateValuesRequest) {
        if let Some(v) = req.achtsamkeit {
            scores.set(Value::Achtsamkeit, v);
        }
        if let Some(v) = req.anerkennung {
            scores.set(Value::Anerkennung, v);
        }
        if let Some(v) = req.aufmerksamkeit {
            scores.set(Value::Aufmerksamkeit, v);
        }
        if let Some(v) = req.entwicklung {
            scores.set(Value::Entwicklung, v);
        }
        if let Some(v) = req.sinn {
            scores.set(Value::Sinn, v);
        }
    }

    #[test]
    fn update_clamps_and_preserves_untouched_fields() {
        let mut scores = ValueScores::default();
        apply_update(
            &mut scores,
            &UpdateValuesRequest {
                achtsamkeit: Some(1.5),   // clamp to 1.0
                anerkennung: Some(-0.2),  // clamp to 0.0
                aufmerksamkeit: Some(0.7),
                entwicklung: None, // unchanged
                sinn: None,        // unchanged
            },
        );
        assert!((scores.achtsamkeit - 1.0).abs() < f32::EPSILON);
        assert!((scores.anerkennung - 0.0).abs() < f32::EPSILON);
        assert!((scores.aufmerksamkeit - 0.7).abs() < f32::EPSILON);
        assert!((scores.entwicklung - 0.5).abs() < f32::EPSILON);
        assert!((scores.sinn - 0.5).abs() < f32::EPSILON);

        let resp = ValuesResponse::from_scores(scores.clone());
        // Average: (1.0 + 0.0 + 0.7 + 0.5 + 0.5) / 5 = 0.54
        assert!((resp.average - 0.54).abs() < 1e-5);
    }

    #[test]
    fn partial_update_touches_only_specified_fields() {
        let mut scores = ValueScores::default();
        apply_update(
            &mut scores,
            &UpdateValuesRequest {
                achtsamkeit: Some(0.9),
                anerkennung: Some(0.9),
                aufmerksamkeit: Some(0.9),
                entwicklung: Some(0.9),
                sinn: Some(0.9),
            },
        );
        apply_update(
            &mut scores,
            &UpdateValuesRequest {
                sinn: Some(0.1),
                ..Default::default()
            },
        );
        assert!((scores.achtsamkeit - 0.9).abs() < f32::EPSILON);
        assert!((scores.anerkennung - 0.9).abs() < f32::EPSILON);
        assert!((scores.sinn - 0.1).abs() < f32::EPSILON);
    }

    #[test]
    fn values_response_returns_precomputed_average() {
        let mut scores = ValueScores::default();
        scores.set(Value::Achtsamkeit, 1.0);
        scores.set(Value::Sinn, 0.0);
        let resp = ValuesResponse::from_scores(scores);
        // Average of {1.0, 0.5, 0.5, 0.5, 0.0} = 0.5
        assert!((resp.average - 0.5).abs() < f32::EPSILON);
    }

    #[tokio::test]
    async fn graph_event_broadcast_delivers_to_subscribers() {
        let (tx, _rx0) = tokio::sync::broadcast::channel::<GraphEvent>(16);
        let mut rx = tx.subscribe();

        let event = GraphEvent::now("ceo", "researcher", "Execute", 1536);
        tx.send(event.clone()).expect("broadcast must have subscriber");

        let received = rx.recv().await.expect("subscriber must receive");
        assert_eq!(received.from, "ceo");
        assert_eq!(received.to, "researcher");
        assert_eq!(received.intent, "Execute");
        assert_eq!(received.size_bytes, 1536);
    }

    #[tokio::test]
    async fn graph_event_broadcast_reports_lag() {
        // Capacity 2, send 4 before receiving → receiver sees Lagged(2).
        let (tx, mut rx) = tokio::sync::broadcast::channel::<GraphEvent>(2);
        for i in 0..4 {
            tx.send(GraphEvent::now("a", "b", "Execute", i as u32))
                .unwrap();
        }
        match rx.recv().await {
            Err(tokio::sync::broadcast::error::RecvError::Lagged(n)) => {
                assert_eq!(n, 2, "must report exactly the number of dropped events");
            }
            other => panic!("expected Lagged, got {other:?}"),
        }
    }

    #[test]
    fn graph_event_serialises_compactly() {
        let evt = GraphEvent::now("a", "b", "Execute", 100);
        let json = serde_json::to_string(&evt).unwrap();
        assert!(json.contains("\"from\":\"a\""));
        assert!(json.contains("\"to\":\"b\""));
        assert!(json.contains("\"intent\":\"Execute\""));
        assert!(json.contains("\"size_bytes\":100"));
        // No status set → serialises as null (or omitted). Either is fine,
        // but the field must be decodable back into an Option<String>.
        let back: GraphEvent = serde_json::from_str(&json).unwrap();
        assert_eq!(back.status, None);
    }
}
