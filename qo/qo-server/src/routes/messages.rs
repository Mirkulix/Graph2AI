use axum::extract::{Query, State};
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::Json;
use futures::stream::Stream;
use serde::{Deserialize, Serialize};
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
    /// Excerpt of the message content (first 4 KB of graph.metadata.content).
    /// Lets IDE inboxes show the actual reply without a separate fetch.
    #[serde(skip_serializing_if = "String::is_empty")]
    content: String,
    /// Convenience flag for IDE inboxes — true when this is an agent reply.
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    is_reply: bool,
    /// Whether this message was sent by an automatic trigger (e.g. file save)
    /// rather than a manual user action. Read from graph.metadata["auto_triggered"].
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    auto_triggered: bool,
    /// What kind of trigger fired this message. Empty string when not auto-triggered.
    /// Examples: "on_save", "on_change", "on_test_fail", "on_commit", "scheduled".
    #[serde(skip_serializing_if = "String::is_empty")]
    trigger_kind: String,
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
        let is_reply = intent.starts_with("Result");
        let content = msg.graph.metadata.get("content")
            .map(|c| {
                if c.len() > 4096 { format!("{}…", &c[..4096]) } else { c.clone() }
            })
            .unwrap_or_default();
        let auto_triggered = msg.graph.metadata.get("auto_triggered")
            .map(|v| v == "true" || v == "1")
            .unwrap_or(false);
        let trigger_kind = msg.graph.metadata.get("trigger_kind")
            .cloned()
            .unwrap_or_default();
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
            content,
            is_reply,
            auto_triggered,
            trigger_kind,
        };
        let json = serde_json::to_string(&ev).unwrap_or_default();
        Ok(Event::default().data(json))
    });

    Sse::new(stream).keep_alive(KeepAlive::default())
}

#[derive(Deserialize, Default)]
pub struct RecentQuery {
    #[serde(default = "default_limit")]
    pub n: usize,
}

fn default_limit() -> usize {
    50
}

/// GET /api/messages/recent?n=50 — returns the last N bus messages from
/// the server-side ring buffer (cross-machine cockpit hydration). The
/// per-request cap matches the ring capacity (200) to bound payload size;
/// `n` is also clamped to a minimum of 1.
pub async fn recent_messages(
    State(state): State<Arc<AppState>>,
    Query(q): Query<RecentQuery>,
) -> Json<Vec<crate::RecentMessage>> {
    let buf = state.recent_messages.lock().await;
    let n = q.n.clamp(1, 200);
    let len = buf.len();
    let start = if len > n { len - n } else { 0 };
    let slice: Vec<_> = buf.iter().skip(start).cloned().collect();
    Json(slice)
}

/// One row in the cross-source unified-history feed. The cockpit reads
/// this from `/api/history/unified` to populate the Conversation Pane
/// with everything that happened across IDE sidebars, /api/chat, and
/// swarm subtasks in a single chronological list.
#[derive(Serialize)]
pub struct UnifiedEvent {
    /// Source bucket — "bus" | "chat" | "swarm".
    pub kind: String,
    /// Unix seconds.
    pub ts: u64,
    pub from: String,
    pub to: String,
    pub content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub provider: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub duration_ms: Option<u64>,
}

/// GET /api/history/unified?n=50 — merges three history sources
/// (recent bus messages, persisted chat history, live swarm subtask
/// responses) into one chronological stream, newest first. Single
/// endpoint that the cockpit can hit to see "everything anywhere".
pub async fn unified_history(
    State(state): State<Arc<AppState>>,
    Query(q): Query<RecentQuery>,
) -> Json<Vec<UnifiedEvent>> {
    let n = q.n.clamp(1, 500);
    let mut events: Vec<UnifiedEvent> = Vec::new();

    // Source 1 — bus ring buffer (covers /api/llm/proxy + native bus traffic).
    {
        let buf = state.recent_messages.lock().await;
        for m in buf.iter() {
            events.push(UnifiedEvent {
                kind: "bus".to_string(),
                ts: m.timestamp,
                from: m.from.clone(),
                to: m.to.clone(),
                content: m.content.clone(),
                provider: None,
                duration_ms: None,
            });
        }
    }

    // Source 2 — persisted /api/chat history. Each redb row holds a JSON
    // {id, user, assistant, timestamp} blob; we expand it into two
    // logical events (user prompt, QO reply) so the timeline reads
    // naturally next to bus traffic.
    if let Ok(rows) = state.store.chat_history(n.min(200)) {
        for (_id, json) in rows {
            if let Ok(v) = serde_json::from_str::<serde_json::Value>(&json) {
                let ts = v.get("timestamp").and_then(|x| x.as_u64()).unwrap_or(0);
                let user = v
                    .get("user")
                    .and_then(|x| x.as_str())
                    .unwrap_or("")
                    .to_string();
                let assistant = v
                    .get("assistant")
                    .and_then(|x| x.as_str())
                    .unwrap_or("")
                    .to_string();
                if !user.is_empty() {
                    events.push(UnifiedEvent {
                        kind: "chat".to_string(),
                        ts,
                        from: "user".to_string(),
                        to: "qo".to_string(),
                        content: user,
                        provider: None,
                        duration_ms: None,
                    });
                }
                if !assistant.is_empty() {
                    events.push(UnifiedEvent {
                        kind: "chat".to_string(),
                        ts,
                        from: "qo".to_string(),
                        to: "user".to_string(),
                        content: assistant,
                        provider: None,
                        duration_ms: None,
                    });
                }
            }
        }
    }

    // Source 3 — live swarm subtasks. We surface each completed subtask
    // as a single event labeled with the swarm id, so the cockpit can
    // group them or filter them out.
    {
        let swarms = state.swarms.read().await;
        for s in swarms.values() {
            for round in &s.rounds {
                for st in &round.subtasks {
                    let Some(resp) = st.response.as_deref() else {
                        continue;
                    };
                    let ts = st.finished_at.or(st.started_at).unwrap_or(s.started_at);
                    events.push(UnifiedEvent {
                        kind: "swarm".to_string(),
                        ts,
                        from: format!("swarm-{}", s.id),
                        to: st.assigned_to.clone(),
                        content: resp.to_string(),
                        provider: Some(st.status.clone()),
                        duration_ms: None,
                    });
                }
            }
        }
    }

    // Newest first, then trim to `n`.
    events.sort_by(|a, b| b.ts.cmp(&a.ts));
    events.truncate(n);
    Json(events)
}
