//! POST /api/broadcast — fan-out one prompt to N IDE identities.
//!
//! Sister endpoint to `/api/consensus`, but fire-and-forget: we publish
//! one GraphMessage per target onto the bus and return immediately. Each
//! receiving IDE/agent decides on its own whether to act (auto-respond
//! flow in the VS Code extension picks Execute envelopes up and answers
//! via the same bus). Replies flow back through the existing SSE stream
//! that the cockpit already consumes.
//!
//! Why not reuse `/api/consensus`? Consensus blocks until every reply
//! lands (or times out) and aggregates a Jaccard score. Broadcast is for
//! "kick everyone off, see results trickle in" — the cockpit returns
//! instantly, the user keeps working.

use axum::{extract::State, http::StatusCode, Json};
use qlang_agent::protocol::{
    next_msg_id, AgentId, Capability, GraphMessage, MessageIntent,
};
use qlang_core::graph::Graph;
use serde::{Deserialize, Serialize};
use std::collections::HashMap as StdHashMap;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use crate::AppState;

#[derive(Debug, Deserialize)]
pub struct BroadcastRequest {
    pub prompt: String,
    pub targets: Vec<String>,
    /// Optional sender label (defaults to "cockpit-broadcast"). Useful
    /// when the cockpit wants to attribute the fan-out to a specific
    /// human/operator name in the conversation history.
    #[serde(default)]
    pub from: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct BroadcastResponse {
    pub sent: usize,
    pub from: String,
    pub targets: Vec<String>,
    pub message_ids: Vec<u64>,
    pub failures: Vec<BroadcastFailure>,
}

#[derive(Debug, Serialize)]
pub struct BroadcastFailure {
    pub target: String,
    pub error: String,
}

/// POST /api/broadcast
pub async fn broadcast(
    State(state): State<Arc<AppState>>,
    Json(req): Json<BroadcastRequest>,
) -> Result<Json<BroadcastResponse>, (StatusCode, String)> {
    if req.prompt.trim().is_empty() {
        return Err((StatusCode::BAD_REQUEST, "prompt empty".to_string()));
    }
    if req.targets.is_empty() {
        return Err((StatusCode::BAD_REQUEST, "targets empty".to_string()));
    }

    let from_name = req
        .from
        .clone()
        .filter(|s| !s.trim().is_empty())
        .unwrap_or_else(|| {
            let stamp = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.as_millis() as u64)
                .unwrap_or(0);
            format!("cockpit-broadcast-{}", stamp)
        });
    let from_agent = AgentId {
        name: from_name.clone(),
        capabilities: vec![Capability::Execute],
    };

    let bus = state.message_bus.clone();
    let mut message_ids: Vec<u64> = Vec::with_capacity(req.targets.len());
    let mut failures: Vec<BroadcastFailure> = Vec::new();

    // De-dupe targets in-place (preserve order). Sending the same prompt
    // twice to one identity is never what the user meant.
    let mut seen: std::collections::HashSet<&str> = std::collections::HashSet::new();
    let unique_targets: Vec<&String> = req
        .targets
        .iter()
        .filter(|t| seen.insert(t.as_str()))
        .collect();

    // Snapshot the eligibility state of every requested target up front,
    // so we don't have to lock the presence map inside the send loop.
    // Targets that are unknown OR explicitly opted out via
    // `eligible_for_swarms=false` get rejected — broadcasts override the
    // per-IDE auto-respond gate, so eligibility is the only consent
    // surface left. The user controls this from the IDE Presence detail
    // pane in the cockpit.
    let eligibility: std::collections::HashMap<String, bool> = {
        let presence = state.presence.lock().await;
        unique_targets
            .iter()
            .map(|t| {
                let identity = t.as_str();
                let allowed = presence
                    .get(identity)
                    .map(|entry| entry.eligible_for_swarms)
                    .unwrap_or(false);
                (identity.to_string(), allowed)
            })
            .collect()
    };

    for target in &unique_targets {
        let target_str = target.as_str();
        if target_str.trim().is_empty() {
            failures.push(BroadcastFailure {
                target: target_str.to_string(),
                error: "empty target".to_string(),
            });
            continue;
        }
        if !eligibility.get(target_str).copied().unwrap_or(false) {
            failures.push(BroadcastFailure {
                target: target_str.to_string(),
                error: "not eligible (unknown identity or eligible_for_swarms=false)"
                    .to_string(),
            });
            continue;
        }

        let msg_id = next_msg_id();
        let mut metadata = StdHashMap::new();
        metadata.insert("source".to_string(), "broadcast".to_string());
        metadata.insert("content".to_string(), req.prompt.clone());
        metadata.insert("broadcast_size".to_string(), unique_targets.len().to_string());
        // Tag this envelope as an autonomous trigger so receiving IDE
        // inboxes process it via their configured LLM without requiring
        // each IDE to have flipped `qlang.qlms.autoRespond.enabled`. The
        // user already consented by clicking BROADCAST in the cockpit.
        metadata.insert("auto_triggered".to_string(), "true".to_string());
        metadata.insert("trigger_kind".to_string(), "broadcast".to_string());

        let msg = GraphMessage {
            id: msg_id,
            from: from_agent.clone(),
            to: AgentId {
                name: target_str.to_string(),
                capabilities: vec![Capability::Execute],
            },
            graph: Graph {
                id: format!("broadcast-{}", msg_id),
                version: "1.0".to_string(),
                nodes: vec![],
                edges: vec![],
                constraints: vec![],
                metadata,
            },
            inputs: StdHashMap::new(),
            intent: MessageIntent::Execute,
            in_reply_to: None,
            signature: None,
            signer_pubkey: None,
            graph_hash: None,
        };

        // Best-effort delivery. AgentNotFound is benign — IDE identities
        // are not registered as bus mailboxes; the SSE stream still
        // forwards the envelope to listening cockpits and IDE inboxes
        // (same trick `mesh_history.rs` relies on).
        use qlang_agent::bus::DeliveryStatus;
        match bus.send(msg).await {
            DeliveryStatus::Delivered | DeliveryStatus::AgentNotFound(_) => {
                message_ids.push(msg_id);
            }
            DeliveryStatus::MailboxFull(name) => {
                failures.push(BroadcastFailure {
                    target: target_str.to_string(),
                    error: format!("mailbox full for {}", name),
                });
            }
        }
    }

    Ok(Json(BroadcastResponse {
        sent: message_ids.len(),
        from: from_name,
        targets: unique_targets.iter().map(|t| t.to_string()).collect(),
        message_ids,
        failures,
    }))
}
