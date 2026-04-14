//! QLANG Message Bus — async routing layer for AI-to-AI communication.
//!
//! The MessageBus is the central nervous system of QLANG agent communication:
//!
//!   Agent A ──► MessageBus ──► Agent B
//!              (routes by AgentId)
//!
//! Each registered agent gets a Mailbox (tokio mpsc channel).
//! Messages are routed by the `to` field in GraphMessage.
//! Conversations are tracked for request/reply correlation.

use crate::protocol::{AgentConversation, AgentId, GraphMessage, MessageIntent};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::{mpsc, Mutex, RwLock};

/// Configuration for the message bus.
#[derive(Debug, Clone)]
pub struct BusConfig {
    /// Per-agent mailbox capacity (default: 256).
    pub mailbox_capacity: usize,
    /// Whether to log all routed messages.
    pub trace_messages: bool,
}

impl Default for BusConfig {
    fn default() -> Self {
        Self {
            mailbox_capacity: 256,
            trace_messages: false,
        }
    }
}

/// A handle to receive messages for a specific agent.
pub struct Mailbox {
    pub agent_id: AgentId,
    rx: mpsc::Receiver<GraphMessage>,
}

impl Mailbox {
    /// Receive the next message (blocks until one arrives).
    pub async fn recv(&mut self) -> Option<GraphMessage> {
        self.rx.recv().await
    }

    /// Try to receive without blocking.
    pub fn try_recv(&mut self) -> Option<GraphMessage> {
        self.rx.try_recv().ok()
    }
}

/// Delivery result for a sent message.
#[derive(Debug, Clone, PartialEq)]
pub enum DeliveryStatus {
    /// Message was delivered to the agent's mailbox.
    Delivered,
    /// Target agent is not registered on this bus.
    AgentNotFound(String),
    /// Agent's mailbox is full.
    MailboxFull(String),
}

/// Statistics for the message bus.
#[derive(Debug, Clone, Default)]
pub struct BusStats {
    pub total_messages: u64,
    pub delivered: u64,
    pub failed: u64,
    pub active_agents: usize,
    pub active_conversations: usize,
}

/// Entry tracking a registered agent.
struct AgentEntry {
    #[allow(dead_code)]
    id: AgentId,
    tx: mpsc::Sender<GraphMessage>,
}

/// A conversation tracked by the bus.
struct TrackedConversation {
    /// The agents involved.
    participants: Vec<String>,
    /// All message IDs in order.
    message_ids: Vec<u64>,
    /// The full conversation (for serialization).
    conversation: AgentConversation,
}

/// The central message bus for QLANG agent communication.
///
/// Agents register to get a Mailbox. Messages sent via `send()` are
/// routed to the target agent's mailbox based on `GraphMessage.to`.
pub struct MessageBus {
    agents: RwLock<HashMap<String, AgentEntry>>,
    conversations: Mutex<HashMap<String, TrackedConversation>>,
    config: BusConfig,
    stats: Mutex<BusStats>,
    /// Listeners that receive a copy of every message (for monitoring/logging).
    listeners: RwLock<Vec<mpsc::Sender<GraphMessage>>>,
}

impl MessageBus {
    /// Create a new message bus with default configuration.
    pub fn new() -> Arc<Self> {
        Self::with_config(BusConfig::default())
    }

    /// Create a new message bus with custom configuration.
    pub fn with_config(config: BusConfig) -> Arc<Self> {
        Arc::new(Self {
            agents: RwLock::new(HashMap::new()),
            conversations: Mutex::new(HashMap::new()),
            config,
            stats: Mutex::new(BusStats::default()),
            listeners: RwLock::new(Vec::new()),
        })
    }

    /// Register an agent and get its Mailbox for receiving messages.
    pub async fn register(&self, agent_id: AgentId) -> Mailbox {
        let (tx, rx) = mpsc::channel(self.config.mailbox_capacity);
        let name = agent_id.name.clone();
        let entry = AgentEntry {
            id: agent_id.clone(),
            tx,
        };
        self.agents.write().await.insert(name, entry);
        Mailbox { agent_id, rx }
    }

    /// Unregister an agent (its mailbox will be dropped).
    pub async fn unregister(&self, name: &str) {
        self.agents.write().await.remove(name);
    }

    /// List all registered agent names.
    pub async fn registered_agents(&self) -> Vec<String> {
        self.agents.read().await.keys().cloned().collect()
    }

    /// Send a GraphMessage to its target agent.
    ///
    /// The message is routed based on `msg.to.name`. If the target is
    /// registered, the message is placed in their mailbox. The message
    /// is also tracked in the conversation log and forwarded to listeners.
    pub async fn send(&self, msg: GraphMessage) -> DeliveryStatus {
        let target = msg.to.name.clone();
        let from = msg.from.name.clone();
        let msg_id = msg.id;

        // Track in conversation
        {
            let conv_key = conversation_key(&from, &target);
            let mut convs = self.conversations.lock().await;
            let tracked = convs.entry(conv_key).or_insert_with(|| TrackedConversation {
                participants: vec![from.clone(), target.clone()],
                message_ids: Vec::new(),
                conversation: AgentConversation::new(),
            });
            tracked.message_ids.push(msg_id);
            tracked.conversation.send(
                msg.from.clone(),
                msg.to.clone(),
                msg.graph.clone(),
                msg.inputs.clone(),
                msg.intent.clone(),
                msg.in_reply_to,
            );
        }

        // Forward to listeners
        {
            let listeners = self.listeners.read().await;
            for listener in listeners.iter() {
                let _ = listener.try_send(msg.clone());
            }
        }

        // Route to target
        let agents = self.agents.read().await;
        let status = match agents.get(&target) {
            Some(entry) => match entry.tx.try_send(msg) {
                Ok(()) => DeliveryStatus::Delivered,
                Err(mpsc::error::TrySendError::Full(_)) => {
                    DeliveryStatus::MailboxFull(target.clone())
                }
                Err(mpsc::error::TrySendError::Closed(_)) => {
                    DeliveryStatus::AgentNotFound(target.clone())
                }
            },
            None => DeliveryStatus::AgentNotFound(target.clone()),
        };

        // Update stats
        {
            let mut stats = self.stats.lock().await;
            stats.total_messages += 1;
            match &status {
                DeliveryStatus::Delivered => stats.delivered += 1,
                _ => stats.failed += 1,
            }
        }

        status
    }

    /// Send a message and wait for a reply (with timeout).
    ///
    /// The caller must have a Mailbox to receive the reply on.
    pub async fn send_and_wait(
        &self,
        msg: GraphMessage,
        reply_mailbox: &mut Mailbox,
        timeout: std::time::Duration,
    ) -> Result<GraphMessage, BusError> {
        let msg_id = msg.id;
        let status = self.send(msg).await;
        match status {
            DeliveryStatus::Delivered => {}
            DeliveryStatus::AgentNotFound(name) => {
                return Err(BusError::AgentNotFound(name));
            }
            DeliveryStatus::MailboxFull(name) => {
                return Err(BusError::MailboxFull(name));
            }
        }

        // Wait for reply
        match tokio::time::timeout(timeout, async {
            loop {
                if let Some(reply) = reply_mailbox.recv().await {
                    if reply.in_reply_to == Some(msg_id) {
                        return Some(reply);
                    }
                    // Not our reply — skip (in a real system, push back or buffer)
                } else {
                    return None;
                }
            }
        })
        .await
        {
            Ok(Some(reply)) => Ok(reply),
            Ok(None) => Err(BusError::ChannelClosed),
            Err(_) => Err(BusError::Timeout),
        }
    }

    /// Subscribe to all messages passing through the bus (for monitoring).
    pub async fn subscribe(&self) -> mpsc::Receiver<GraphMessage> {
        let (tx, rx) = mpsc::channel(self.config.mailbox_capacity);
        self.listeners.write().await.push(tx);
        rx
    }

    /// Get current bus statistics.
    pub async fn stats(&self) -> BusStats {
        let mut stats = self.stats.lock().await.clone();
        stats.active_agents = self.agents.read().await.len();
        stats.active_conversations = self.conversations.lock().await.len();
        stats
    }

    /// Get the message history for a conversation between two agents.
    pub async fn conversation_history(
        &self,
        agent_a: &str,
        agent_b: &str,
    ) -> Option<Vec<u64>> {
        let key = conversation_key(agent_a, agent_b);
        let convs = self.conversations.lock().await;
        convs.get(&key).map(|c| c.message_ids.clone())
    }

    /// Get all active conversation keys.
    pub async fn active_conversations(&self) -> Vec<(String, Vec<String>)> {
        let convs = self.conversations.lock().await;
        convs
            .iter()
            .map(|(key, tracked)| (key.clone(), tracked.participants.clone()))
            .collect()
    }

    /// Serialize a conversation to QLMS binary format.
    pub async fn conversation_to_binary(
        &self,
        agent_a: &str,
        agent_b: &str,
    ) -> Option<Vec<u8>> {
        let key = conversation_key(agent_a, agent_b);
        let convs = self.conversations.lock().await;
        convs
            .get(&key)
            .and_then(|c| c.conversation.to_binary().ok())
    }
}

/// Generate a stable key for a conversation between two agents.
/// Alphabetically sorted so A↔B and B↔A use the same key.
fn conversation_key(a: &str, b: &str) -> String {
    if a <= b {
        format!("{a}↔{b}")
    } else {
        format!("{b}↔{a}")
    }
}

/// Errors from bus operations.
#[derive(Debug, thiserror::Error)]
pub enum BusError {
    #[error("agent not found: {0}")]
    AgentNotFound(String),
    #[error("mailbox full: {0}")]
    MailboxFull(String),
    #[error("reply timeout")]
    Timeout,
    #[error("channel closed")]
    ChannelClosed,
}

// ============================================================
// Helper: Build a reply GraphMessage
// ============================================================

/// Build a reply message from agent B back to agent A.
pub fn build_reply(
    original: &GraphMessage,
    from: AgentId,
    graph: qlang_core::graph::Graph,
    inputs: HashMap<String, qlang_core::tensor::TensorData>,
) -> GraphMessage {
    GraphMessage {
        id: crate::protocol::next_msg_id(),
        from,
        to: original.from.clone(),
        graph,
        inputs,
        intent: MessageIntent::Result {
            original_message_id: original.id,
        },
        in_reply_to: Some(original.id),
        signature: None,
        signer_pubkey: None,
        graph_hash: None,
    }
}

// ============================================================
// Tests
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::{AgentId, Capability, MessageIntent};
    use qlang_core::graph::Graph;

    fn agent(name: &str) -> AgentId {
        AgentId {
            name: name.into(),
            capabilities: vec![Capability::Execute],
        }
    }

    fn test_message(from: &str, to: &str, id: u64) -> GraphMessage {
        GraphMessage {
            id,
            from: agent(from),
            to: agent(to),
            graph: Graph::new("test"),
            inputs: HashMap::new(),
            intent: MessageIntent::Execute,
            in_reply_to: None,
            signature: None,
            signer_pubkey: None,
            graph_hash: None,
        }
    }

    #[tokio::test]
    async fn register_and_send() {
        let bus = MessageBus::new();
        let mut mailbox = bus.register(agent("receiver")).await;

        let msg = test_message("sender", "receiver", 1);
        let status = bus.send(msg).await;
        assert_eq!(status, DeliveryStatus::Delivered);

        let received = mailbox.try_recv().unwrap();
        assert_eq!(received.id, 1);
        assert_eq!(received.from.name, "sender");
    }

    #[tokio::test]
    async fn send_to_unknown_agent() {
        let bus = MessageBus::new();
        let msg = test_message("sender", "nobody", 1);
        let status = bus.send(msg).await;
        assert!(matches!(status, DeliveryStatus::AgentNotFound(_)));
    }

    #[tokio::test]
    async fn conversation_tracking() {
        let bus = MessageBus::new();
        let _mb_a = bus.register(agent("alice")).await;
        let _mb_b = bus.register(agent("bob")).await;

        bus.send(test_message("alice", "bob", 1)).await;
        bus.send(test_message("bob", "alice", 2)).await;
        bus.send(test_message("alice", "bob", 3)).await;

        let history = bus.conversation_history("alice", "bob").await.unwrap();
        assert_eq!(history, vec![1, 2, 3]);

        // Same conversation from the other side
        let history2 = bus.conversation_history("bob", "alice").await.unwrap();
        assert_eq!(history2, vec![1, 2, 3]);
    }

    #[tokio::test]
    async fn stats_tracking() {
        let bus = MessageBus::new();
        let _mb = bus.register(agent("agent1")).await;

        bus.send(test_message("x", "agent1", 1)).await;
        bus.send(test_message("x", "agent1", 2)).await;
        bus.send(test_message("x", "nobody", 3)).await;

        let stats = bus.stats().await;
        assert_eq!(stats.total_messages, 3);
        assert_eq!(stats.delivered, 2);
        assert_eq!(stats.failed, 1);
        assert_eq!(stats.active_agents, 1);
    }

    #[tokio::test]
    async fn subscribe_receives_all_messages() {
        let bus = MessageBus::new();
        let _mb = bus.register(agent("target")).await;
        let mut listener = bus.subscribe().await;

        bus.send(test_message("a", "target", 1)).await;
        bus.send(test_message("b", "target", 2)).await;

        let m1 = listener.try_recv().unwrap();
        let m2 = listener.try_recv().unwrap();
        assert_eq!(m1.id, 1);
        assert_eq!(m2.id, 2);
    }

    #[tokio::test]
    async fn send_and_wait_reply() {
        let bus = MessageBus::new();
        let mut mb_a = bus.register(agent("alice")).await;
        let mut mb_b = bus.register(agent("bob")).await;

        let bus_clone = bus.clone();

        // Bob auto-replies
        tokio::spawn(async move {
            if let Some(msg) = mb_b.recv().await {
                let reply = GraphMessage {
                    id: 99,
                    from: agent("bob"),
                    to: msg.from.clone(),
                    graph: Graph::new("reply"),
                    inputs: HashMap::new(),
                    intent: MessageIntent::Result {
                        original_message_id: msg.id,
                    },
                    in_reply_to: Some(msg.id),
                    signature: None,
                    signer_pubkey: None,
                    graph_hash: None,
                };
                bus_clone.send(reply).await;
            }
        });

        let msg = test_message("alice", "bob", 42);
        let reply = bus
            .send_and_wait(msg, &mut mb_a, std::time::Duration::from_secs(2))
            .await
            .unwrap();

        assert_eq!(reply.in_reply_to, Some(42));
        assert_eq!(reply.from.name, "bob");
    }

    #[tokio::test]
    async fn conversation_to_binary() {
        let bus = MessageBus::new();
        let _mb_a = bus.register(agent("alice")).await;
        let _mb_b = bus.register(agent("bob")).await;

        bus.send(test_message("alice", "bob", 1)).await;

        let binary = bus.conversation_to_binary("alice", "bob").await.unwrap();
        assert_eq!(&binary[0..4], &[0x51, 0x4C, 0x4D, 0x53]); // QLMS magic
    }

    #[tokio::test]
    async fn unregister_agent() {
        let bus = MessageBus::new();
        let _mb = bus.register(agent("temp")).await;
        assert_eq!(bus.registered_agents().await.len(), 1);

        bus.unregister("temp").await;
        assert_eq!(bus.registered_agents().await.len(), 0);

        let status = bus.send(test_message("x", "temp", 1)).await;
        assert!(matches!(status, DeliveryStatus::AgentNotFound(_)));
    }

    #[tokio::test]
    async fn active_conversations_list() {
        let bus = MessageBus::new();
        let _mb_a = bus.register(agent("a")).await;
        let _mb_b = bus.register(agent("b")).await;
        let _mb_c = bus.register(agent("c")).await;

        bus.send(test_message("a", "b", 1)).await;
        bus.send(test_message("b", "c", 2)).await;

        let convs = bus.active_conversations().await;
        assert_eq!(convs.len(), 2);
    }
}
