//! Unified mesh-history helper. Any handler that produces a (request, reply) chat
//! exchange can call `record_chat_pair` to:
//!   1. Emit two synthetic GraphMessages on the bus (Execute + Result) so the
//!      cockpit Conversation Pane (SSE), recent_messages ring buffer, and any
//!      IDE inbox subscribers see the activity — same way native bus traffic does.
//!   2. Persist a chat-shaped graph to graph_store so /api/graphs lists it and
//!      it shows up in the Knowledge Ledger.
//!   3. Remember the (prompt, response) pair in memory_ctx so future chat-mode
//!      requests can recall it (chat.rs already calls mem.recall, this feeds
//!      into the same store).
//!
//! NEVER let any of these fail in a way that breaks the caller. Each step is
//! best-effort with tracing::warn on failure.

use crate::AppState;
use std::sync::Arc;

pub struct ChatPair<'a> {
    /// Who initiated (e.g. "cursor-01-b983f2" or "cockpit-user").
    pub requester: &'a str,
    /// Which agent/identity produced the reply (e.g. "developer" or "claude-cli-agent").
    pub responder: &'a str,
    /// Original user prompt (or subtask prompt).
    pub prompt: &'a str,
    /// Reply text.
    pub response: &'a str,
    /// Provider that served the call ("claude-cli", "deepseek", "claude-cli-agent", etc.)
    pub provider: &'a str,
    pub duration_ms: u64,
}

/// Best-effort write of a chat exchange into all three mesh-history sinks.
/// Each sink is independent: a failure in one is logged but does not block
/// the others, and never propagates back to the caller.
pub async fn record_chat_pair(state: &Arc<AppState>, pair: ChatPair<'_>) {
    // 1. Emit synthetic bus messages so cockpit + IDE inboxes see the activity.
    if let Err(e) = emit_bus_pair(state, &pair).await {
        tracing::warn!("mesh_history: emit_bus_pair failed: {}", e);
    }
    // 2. Store as graph for /api/graphs + Knowledge Ledger.
    if let Err(e) = store_graph(state, &pair) {
        tracing::warn!("mesh_history: store_graph failed: {}", e);
    }
    // 3. Memory recall for future /api/chat sessions.
    store_memory(state, &pair).await;
}

async fn emit_bus_pair(state: &Arc<AppState>, p: &ChatPair<'_>) -> Result<(), String> {
    use qlang_agent::protocol::{
        next_msg_id, AgentId, Capability, GraphMessage, MessageIntent,
    };
    use qlang_core::graph::Graph;
    use std::collections::HashMap as StdHashMap;

    let req_id = next_msg_id();
    let reply_id = next_msg_id();

    let mut req_meta = StdHashMap::new();
    req_meta.insert("source".into(), "mesh_history".into());
    req_meta.insert("content".into(), p.prompt.to_string());
    let req_msg = GraphMessage {
        id: req_id,
        from: AgentId {
            name: p.requester.into(),
            capabilities: vec![Capability::Execute],
        },
        to: AgentId {
            name: p.responder.into(),
            capabilities: vec![Capability::Execute],
        },
        graph: Graph {
            id: format!("mesh-req-{}", req_id),
            version: "1.0".into(),
            nodes: vec![],
            edges: vec![],
            constraints: vec![],
            metadata: req_meta,
        },
        inputs: StdHashMap::new(),
        intent: MessageIntent::Execute,
        in_reply_to: None,
        signature: None,
        signer_pubkey: None,
        graph_hash: None,
    };

    let mut reply_meta = StdHashMap::new();
    reply_meta.insert("source".into(), "mesh_history".into());
    reply_meta.insert("content".into(), p.response.to_string());
    reply_meta.insert("provider".into(), p.provider.to_string());
    reply_meta.insert("duration_ms".into(), p.duration_ms.to_string());
    let reply_msg = GraphMessage {
        id: reply_id,
        from: AgentId {
            name: p.responder.into(),
            capabilities: vec![Capability::Execute],
        },
        to: AgentId {
            name: p.requester.into(),
            capabilities: vec![Capability::Execute],
        },
        graph: Graph {
            id: format!("mesh-rep-{}", reply_id),
            version: "1.0".into(),
            nodes: vec![],
            edges: vec![],
            constraints: vec![],
            metadata: reply_meta,
        },
        inputs: StdHashMap::new(),
        intent: MessageIntent::Result {
            original_message_id: req_id,
        },
        in_reply_to: Some(req_id),
        signature: None,
        signer_pubkey: None,
        graph_hash: None,
    };

    // `MessageBus::send` always forwards to listeners (cockpit SSE +
    // recent_messages drain) regardless of whether the target agent has a
    // mailbox. AgentNotFound here is expected and benign for synthetic
    // history events whose `from`/`to` are IDE identities or providers
    // that aren't registered as bus agents — only forward the failure when
    // it's something more interesting (e.g. mailbox full).
    use qlang_agent::bus::DeliveryStatus;
    match state.message_bus.send(req_msg).await {
        DeliveryStatus::Delivered | DeliveryStatus::AgentNotFound(_) => {}
        DeliveryStatus::MailboxFull(name) => {
            return Err(format!("publish req: mailbox full for {}", name));
        }
    }
    match state.message_bus.send(reply_msg).await {
        DeliveryStatus::Delivered | DeliveryStatus::AgentNotFound(_) => {}
        DeliveryStatus::MailboxFull(name) => {
            return Err(format!("publish reply: mailbox full for {}", name));
        }
    }
    Ok(())
}

fn store_graph(state: &Arc<AppState>, p: &ChatPair<'_>) -> Result<(), String> {
    use qo_memory::graph_builders;
    let graph = graph_builders::build_chat_graph(p.prompt, p.response, p.provider, p.duration_ms);
    state
        .graph_store
        .store(&graph)
        .map(|_| ())
        .map_err(|e| e.to_string())
}

async fn store_memory(state: &Arc<AppState>, p: &ChatPair<'_>) {
    let id = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0);
    let key = format!("mesh_chat_{}", id);
    let snippet_end = p.response.len().min(800);
    let text = format!(
        "{}: {}\n→ {}: {}",
        p.requester,
        p.prompt,
        p.responder,
        &p.response[..snippet_end],
    );
    {
        let mut mem = state.memory.lock().await;
        mem.remember(key.clone(), &text, &state.store);
    }
    let _ = state.store.set(&key, &text);
}
