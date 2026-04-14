//! QLANG Network Bridge — connects MessageBus to TCP for remote AI agents.
//!
//! Architecture:
//!
//!   Local Agent ──► MessageBus ──► Bridge ──► TCP ──► Remote Agent
//!                                    ◄──────────────────────┘
//!
//! The bridge watches for messages addressed to remote agents and forwards
//! them over TCP. Incoming TCP messages are injected into the local bus.

use crate::bus::MessageBus;
use crate::protocol::{AgentId, Capability, GraphMessage, MessageIntent};
use crate::negotiate::{AgentCapabilities, NegotiatedProtocol};
use crate::server::{self, Client, Request, Response};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

/// A remote agent reachable over TCP.
#[derive(Debug, Clone)]
pub struct RemoteAgent {
    /// The agent's identity.
    pub id: AgentId,
    /// TCP address (e.g., "192.168.1.5:9100").
    pub addr: String,
    /// Negotiated protocol (agreed after handshake).
    pub protocol: Option<NegotiatedProtocol>,
}

/// The Bridge connects a local MessageBus to remote agents over TCP.
pub struct NetworkBridge {
    bus: Arc<MessageBus>,
    remotes: RwLock<HashMap<String, RemoteAgent>>,
    local_capabilities: AgentCapabilities,
}

impl NetworkBridge {
    /// Create a new bridge connected to the given MessageBus.
    pub fn new(bus: Arc<MessageBus>) -> Arc<Self> {
        Arc::new(Self {
            bus,
            remotes: RwLock::new(HashMap::new()),
            local_capabilities: AgentCapabilities::local(),
        })
    }

    /// Register a remote agent reachable at the given TCP address.
    ///
    /// Optionally performs a capability handshake to negotiate the protocol.
    pub async fn add_remote(&self, agent_id: AgentId, addr: String) -> RemoteAgent {
        let remote = RemoteAgent {
            id: agent_id.clone(),
            addr,
            protocol: None,
        };
        self.remotes.write().await.insert(agent_id.name.clone(), remote.clone());
        remote
    }

    /// Remove a remote agent.
    pub async fn remove_remote(&self, name: &str) {
        self.remotes.write().await.remove(name);
    }

    /// List all known remote agents.
    pub async fn list_remotes(&self) -> Vec<RemoteAgent> {
        self.remotes.read().await.values().cloned().collect()
    }

    /// Negotiate protocol with a remote agent.
    ///
    /// Sends our capabilities and computes the agreed protocol.
    pub async fn negotiate(&self, name: &str, remote_caps: AgentCapabilities) -> Option<NegotiatedProtocol> {
        let protocol = self.local_capabilities.negotiate(&remote_caps);
        let mut remotes = self.remotes.write().await;
        if let Some(remote) = remotes.get_mut(name) {
            remote.protocol = Some(protocol.clone());
        }
        Some(protocol)
    }

    /// Forward a GraphMessage to a remote agent over TCP.
    ///
    /// The graph is submitted to the remote server, then executed with
    /// the message inputs. The result is returned as a new GraphMessage.
    pub async fn forward_to_remote(
        &self,
        msg: &GraphMessage,
    ) -> Result<Option<GraphMessage>, BridgeError> {
        let remotes = self.remotes.read().await;
        let remote = remotes
            .get(&msg.to.name)
            .ok_or_else(|| BridgeError::RemoteNotFound(msg.to.name.clone()))?;

        let client = Client::new(&remote.addr);

        // Submit the graph
        let graph_id = client
            .submit_graph(msg.graph.clone())
            .await
            .map_err(|e| BridgeError::Network(e.to_string()))?;

        // Execute with inputs
        let outputs = client
            .execute_graph(graph_id, msg.inputs.clone())
            .await
            .map_err(|e| BridgeError::Network(e.to_string()))?;

        // Build reply message
        let reply_graph = qlang_core::graph::Graph::new(format!("reply_from_{}", msg.to.name));
        let reply = GraphMessage {
            id: crate::protocol::next_msg_id(),
            from: msg.to.clone(),
            to: msg.from.clone(),
            graph: reply_graph,
            inputs: outputs,
            intent: MessageIntent::Result {
                original_message_id: msg.id,
            },
            in_reply_to: Some(msg.id),
            signature: None,
            signer_pubkey: None,
            graph_hash: None,
        };

        Ok(Some(reply))
    }

    /// Start the bridge loop: subscribe to the local bus and forward
    /// messages addressed to remote agents over TCP.
    ///
    /// Replies from remote agents are injected back into the local bus.
    pub async fn run(self: Arc<Self>) {
        let mut subscription = self.bus.subscribe().await;

        loop {
            match subscription.recv().await {
                Some(msg) => {
                    let target = msg.to.name.clone();
                    let is_remote = {
                        let remotes = self.remotes.read().await;
                        remotes.contains_key(&target)
                    };

                    if is_remote {
                        match self.forward_to_remote(&msg).await {
                            Ok(Some(reply)) => {
                                // Inject reply into local bus
                                let status = self.bus.send(reply).await;
                                tracing::debug!("Bridge: remote reply injected: {:?}", status);
                            }
                            Ok(None) => {
                                tracing::debug!("Bridge: no reply from remote {}", target);
                            }
                            Err(e) => {
                                tracing::warn!("Bridge: forward to {} failed: {}", target, e);
                            }
                        }
                    }
                }
                None => {
                    tracing::info!("Bridge: subscription closed, stopping");
                    break;
                }
            }
        }
    }

    /// Start a TCP listener that accepts incoming messages from remote agents
    /// and injects them into the local MessageBus.
    pub async fn listen(self: Arc<Self>, addr: &str) -> Result<(), BridgeError> {
        let listener = tokio::net::TcpListener::bind(addr)
            .await
            .map_err(|e| BridgeError::Network(e.to_string()))?;

        tracing::info!("Bridge listening on {}", addr);

        loop {
            let (mut stream, peer_addr) = listener
                .accept()
                .await
                .map_err(|e| BridgeError::Network(e.to_string()))?;

            let bus = self.bus.clone();
            tokio::spawn(async move {
                match server::read_message::<_, Request>(&mut stream).await {
                    Ok(Request::SubmitGraph(graph)) => {
                        // Treat incoming graph as a message from the remote peer
                        let msg = GraphMessage {
                            id: crate::protocol::next_msg_id(),
                            from: AgentId {
                                name: format!("remote_{}", peer_addr),
                                capabilities: vec![Capability::Execute],
                            },
                            to: AgentId {
                                name: "ceo".into(),
                                capabilities: vec![Capability::Execute],
                            },
                            graph,
                            inputs: HashMap::new(),
                            intent: MessageIntent::Execute,
                            in_reply_to: None,
                            signature: None,
                            signer_pubkey: None,
                            graph_hash: None,
                        };
                        let _ = bus.send(msg).await;
                        let resp = Response::GraphSubmitted(0);
                        let _ = server::write_message(&mut stream, &resp).await;
                    }
                    Ok(_) => {
                        let resp = Response::Error("Bridge only accepts SubmitGraph".into());
                        let _ = server::write_message(&mut stream, &resp).await;
                    }
                    Err(e) => {
                        tracing::warn!("Bridge: read from {} failed: {}", peer_addr, e);
                    }
                }
            });
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum BridgeError {
    #[error("remote agent not found: {0}")]
    RemoteNotFound(String),
    #[error("network error: {0}")]
    Network(String),
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bus::MessageBus;
    use crate::protocol::{AgentId, Capability};

    fn agent(name: &str) -> AgentId {
        AgentId {
            name: name.into(),
            capabilities: vec![Capability::Execute],
        }
    }

    #[tokio::test]
    async fn add_and_list_remotes() {
        let bus = MessageBus::new();
        let bridge = NetworkBridge::new(bus);

        bridge.add_remote(agent("gpu-worker"), "10.0.0.5:9100".into()).await;
        bridge.add_remote(agent("edge-device"), "10.0.0.10:9100".into()).await;

        let remotes = bridge.list_remotes().await;
        assert_eq!(remotes.len(), 2);
    }

    #[tokio::test]
    async fn remove_remote() {
        let bus = MessageBus::new();
        let bridge = NetworkBridge::new(bus);

        bridge.add_remote(agent("temp"), "localhost:9100".into()).await;
        assert_eq!(bridge.list_remotes().await.len(), 1);

        bridge.remove_remote("temp").await;
        assert_eq!(bridge.list_remotes().await.len(), 0);
    }

    #[tokio::test]
    async fn negotiate_protocol() {
        let bus = MessageBus::new();
        let bridge = NetworkBridge::new(bus);

        bridge.add_remote(agent("worker"), "localhost:9100".into()).await;

        let remote_caps = AgentCapabilities::local();
        let protocol = bridge.negotiate("worker", remote_caps).await.unwrap();

        assert_eq!(protocol.dtype, "f16");
        assert!(protocol.streaming);
    }

    #[tokio::test]
    async fn forward_to_unknown_remote_fails() {
        let bus = MessageBus::new();
        let bridge = NetworkBridge::new(bus);

        let msg = GraphMessage {
            id: 1,
            from: agent("local"),
            to: agent("unknown"),
            graph: qlang_core::graph::Graph::new("test"),
            inputs: HashMap::new(),
            intent: MessageIntent::Execute,
            in_reply_to: None,
            signature: None,
            signer_pubkey: None,
            graph_hash: None,
        };

        let result = bridge.forward_to_remote(&msg).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn bridge_end_to_end_with_tcp_server() {
        // Start a real TCP server
        let server = crate::server::Server::bind("127.0.0.1:0").await.unwrap();
        let addr = server.local_addr().unwrap().to_string();

        // Handle one request (submit + execute would need 2, just test submit)
        let handle = tokio::spawn(async move {
            server.handle_one().await.unwrap();
            server.handle_one().await.unwrap();
        });

        let bus = MessageBus::new();
        let _mb = bus.register(agent("local")).await;
        let bridge = NetworkBridge::new(bus.clone());

        bridge.add_remote(agent("remote-worker"), addr).await;

        // Create a simple graph message
        let mut g = qlang_core::graph::Graph::new("bridge_test");
        let str_type = qlang_core::tensor::TensorType::new(
            qlang_core::tensor::Dtype::Utf8,
            qlang_core::tensor::Shape::scalar(),
        );
        let input = g.add_node(
            qlang_core::ops::Op::Input { name: "x".into() },
            vec![], vec![str_type.clone()],
        );
        let output = g.add_node(
            qlang_core::ops::Op::Output { name: "y".into() },
            vec![str_type.clone()], vec![],
        );
        g.add_edge(input, 0, output, 0, str_type);

        let msg = GraphMessage {
            id: 42,
            from: agent("local"),
            to: agent("remote-worker"),
            graph: g,
            inputs: {
                let mut m = HashMap::new();
                m.insert("x".to_string(), qlang_core::tensor::TensorData::from_string("hello"));
                m
            },
            intent: MessageIntent::Execute,
            in_reply_to: None,
            signature: None,
            signer_pubkey: None,
            graph_hash: None,
        };

        let reply = bridge.forward_to_remote(&msg).await.unwrap();
        assert!(reply.is_some());
        let reply = reply.unwrap();
        assert_eq!(reply.to.name, "local");
        assert_eq!(reply.in_reply_to, Some(42));

        handle.await.unwrap();
    }
}
