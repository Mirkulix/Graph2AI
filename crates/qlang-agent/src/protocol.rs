//! QLANG Agent Protocol — Binary KI-to-KI communication.
//!
//! This defines how two AI agents exchange QLANG graphs:
//!
//!   Agent A ──[GraphMessage]──► Agent B
//!            ◄──[GraphMessage]──
//!
//! No JSON. No text. Binary graph exchange with typed metadata.
//! Each message is a complete, verifiable computation graph.

use serde::{Deserialize, Serialize};
use qlang_core::crypto::{self, Keypair};
use qlang_core::graph::Graph;
use qlang_core::tensor::TensorData;
use std::collections::HashMap;

/// Serde helper for `Option<[u8; 32]>`: serialize as byte array, skip if None.
mod opt_bytes_32 {
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    pub fn serialize<S: Serializer>(val: &Option<[u8; 32]>, ser: S) -> Result<S::Ok, S::Error> {
        match val {
            Some(arr) => arr.as_slice().serialize(ser),
            None => ser.serialize_none(),
        }
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(de: D) -> Result<Option<[u8; 32]>, D::Error> {
        let opt: Option<Vec<u8>> = Option::deserialize(de)?;
        match opt {
            Some(v) => {
                let arr: [u8; 32] = v
                    .try_into()
                    .map_err(|_| serde::de::Error::custom("expected 32 bytes"))?;
                Ok(Some(arr))
            }
            None => Ok(None),
        }
    }
}

/// Serde helper for `Option<[u8; 64]>`: serialize as byte array, skip if None.
mod opt_bytes_64 {
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    pub fn serialize<S: Serializer>(val: &Option<[u8; 64]>, ser: S) -> Result<S::Ok, S::Error> {
        match val {
            Some(arr) => arr.as_slice().serialize(ser),
            None => ser.serialize_none(),
        }
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(de: D) -> Result<Option<[u8; 64]>, D::Error> {
        let opt: Option<Vec<u8>> = Option::deserialize(de)?;
        match opt {
            Some(v) => {
                let arr: [u8; 64] = v
                    .try_into()
                    .map_err(|_| serde::de::Error::custom("expected 64 bytes"))?;
                Ok(Some(arr))
            }
            None => Ok(None),
        }
    }
}

/// A message exchanged between two AI agents.
///
/// This replaces text-based prompts/responses with structured graph data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphMessage {
    /// Unique message identifier
    pub id: u64,
    /// Sender agent identifier
    pub from: AgentId,
    /// Receiver agent identifier
    pub to: AgentId,
    /// The computation graph (the actual "program")
    pub graph: Graph,
    /// Input data (pre-filled tensors, if any)
    pub inputs: HashMap<String, TensorData>,
    /// What the sender expects the receiver to do
    pub intent: MessageIntent,
    /// Response to a previous message (if applicable)
    pub in_reply_to: Option<u64>,
    /// Optional cryptographic signature (64 bytes) over the graph hash
    #[serde(default, skip_serializing_if = "Option::is_none", with = "opt_bytes_64")]
    pub signature: Option<[u8; 64]>,
    /// Optional signer public key (32 bytes)
    #[serde(default, skip_serializing_if = "Option::is_none", with = "opt_bytes_32")]
    pub signer_pubkey: Option<[u8; 32]>,
    /// Optional SHA-256 hash of the graph at signing time
    #[serde(default, skip_serializing_if = "Option::is_none", with = "opt_bytes_32")]
    pub graph_hash: Option<[u8; 32]>,
}

impl GraphMessage {
    /// Sign this message's graph with a keypair.
    /// Returns a new GraphMessage with signature fields populated.
    pub fn sign(mut self, keypair: &Keypair) -> Self {
        let hash = crypto::hash_graph(&self.graph);
        let sig = keypair.sign(&hash);
        self.signature = Some(sig);
        self.signer_pubkey = Some(*keypair.public_key());
        self.graph_hash = Some(hash);
        self
    }

    /// Verify the signature on this message.
    ///
    /// Returns `true` if:
    /// - The message is unsigned (backward compatible), OR
    /// - The signature is valid for the current graph content.
    pub fn verify_signature(&self) -> bool {
        match (&self.signature, &self.signer_pubkey, &self.graph_hash) {
            (Some(sig), Some(pubkey), Some(stored_hash)) => {
                // Check the stored hash matches the current graph
                let current_hash = crypto::hash_graph(&self.graph);
                if current_hash != *stored_hash {
                    return false;
                }
                // Check signature
                Keypair::verify(pubkey, &current_hash, sig)
            }
            (None, None, None) => true, // Unsigned message is valid
            _ => false, // Partial signature fields = invalid
        }
    }
}

/// Identifies an AI agent in the protocol.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct AgentId {
    pub name: String,
    pub capabilities: Vec<Capability>,
}

// ================================================================
// Agent Coordination — Roles, Status, and Teams
// ================================================================

/// The role an agent plays in a coordinated team.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum AgentRole {
    /// Designs model architectures
    Architect,
    /// Trains models
    Trainer,
    /// Compresses models (IGQK)
    Compressor,
    /// Evaluates results
    Evaluator,
    /// Finds/prepares data
    DataCollector,
    /// Deploys models (WASM, native)
    Deployer,
    /// Monitors system health
    Monitor,
    /// Orchestrates other agents
    Coordinator,
}

/// Tracks the current status of an agent.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum AgentStatus {
    Idle,
    Thinking,
    Working { progress: f32, description: String },
    WaitingForInput,
    Done { result: String },
    Error { message: String },
}

/// A coordinated team of agents working on a shared task.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Team {
    pub name: String,
    pub agents: Vec<TeamMember>,
    pub task: String,
    pub status: AgentStatus,
}

/// A single member within a [`Team`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TeamMember {
    pub agent: AgentId,
    pub role: AgentRole,
    pub status: AgentStatus,
    /// Optional Ollama model name assigned to this member.
    pub model: Option<String>,
}

impl Team {
    /// Create a new team with the given name and task.
    pub fn new(name: &str, task: &str) -> Self {
        Team {
            name: name.to_string(),
            agents: Vec::new(),
            task: task.to_string(),
            status: AgentStatus::Idle,
        }
    }

    /// Add a member to the team.
    pub fn add_member(&mut self, agent: AgentId, role: AgentRole, model: Option<String>) {
        self.agents.push(TeamMember {
            agent,
            role,
            status: AgentStatus::Idle,
            model,
        });
    }

    /// Find the first team member with a matching role.
    pub fn get_by_role(&self, role: &AgentRole) -> Option<&TeamMember> {
        self.agents
            .iter()
            .find(|m| std::mem::discriminant(&m.role) == std::mem::discriminant(role))
    }
}

/// What an agent can do.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Capability {
    /// Can execute graphs (has a runtime)
    Execute,
    /// Can compile graphs to native code (has LLVM)
    Compile,
    /// Can optimize graphs
    Optimize,
    /// Can perform IGQK compression
    Compress,
    /// Can train models (has data access)
    Train,
    /// Can verify proofs
    Verify,
}

/// What the sender wants the receiver to do with the graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MessageIntent {
    /// "Execute this graph and return the results"
    Execute,
    /// "Optimize this graph and return the optimized version"
    Optimize,
    /// "Compress the weights in this graph using IGQK"
    Compress { method: String },
    /// "Verify the proofs in this graph"
    Verify,
    /// "Here are the results you requested"
    Result { original_message_id: u64 },
    /// "Compose this graph with yours"
    Compose,
    /// "Train this model on your data"
    Train { epochs: usize },
}

/// Generate a globally unique message ID (timestamp + counter).
pub fn next_msg_id() -> u64 {
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let ts = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64;
    let seq = COUNTER.fetch_add(1, Ordering::Relaxed);
    (ts << 16) | (seq & 0xFFFF)
}

/// A conversation between agents: sequence of graph messages.
#[derive(Debug)]
pub struct AgentConversation {
    messages: Vec<GraphMessage>,
    next_id: u64,
}

impl AgentConversation {
    pub fn new() -> Self {
        Self {
            messages: Vec::new(),
            next_id: 0,
        }
    }

    /// Send a graph from one agent to another.
    pub fn send(
        &mut self,
        from: AgentId,
        to: AgentId,
        graph: Graph,
        inputs: HashMap<String, TensorData>,
        intent: MessageIntent,
        in_reply_to: Option<u64>,
    ) -> u64 {
        let id = self.next_id;
        self.next_id += 1;

        self.messages.push(GraphMessage {
            id,
            from,
            to,
            graph,
            inputs,
            intent,
            in_reply_to,
            signature: None,
            signer_pubkey: None,
            graph_hash: None,
        });

        id
    }

    /// Get all messages in the conversation.
    pub fn messages(&self) -> &[GraphMessage] {
        &self.messages
    }

    /// Get a specific message by ID.
    pub fn get_message(&self, id: u64) -> Option<&GraphMessage> {
        self.messages.iter().find(|m| m.id == id)
    }

    /// Serialize the entire conversation to binary (unsigned, v1 format).
    pub fn to_binary(&self) -> Result<Vec<u8>, serde_json::Error> {
        // Use JSON-in-binary envelope (same as graph serial format)
        let json = serde_json::to_vec(&self.messages)?;
        let mut buf = Vec::with_capacity(8 + json.len());
        buf.extend_from_slice(&[0x51, 0x4C, 0x4D, 0x53]); // "QLMS" = QLANG Message Stream
        buf.extend_from_slice(&(self.messages.len() as u32).to_le_bytes());
        buf.extend_from_slice(&json);
        Ok(buf)
    }

    /// Serialize the conversation to binary with signatures.
    ///
    /// Wire format v2:
    ///   [0x51,0x4C,0x4D,0x53]  -- "QLMS" magic
    ///   [version: u16 LE]       -- protocol version (2)
    ///   [flags: u16 LE]         -- bit 0: signed
    ///   [signature: 64 bytes]   -- if signed flag set
    ///   [pubkey: 32 bytes]      -- if signed flag set
    ///   [payload_hash: 32 bytes] -- SHA-256 of payload
    ///   [msg_count: u32 LE]     -- number of messages
    ///   [payload...]            -- JSON-encoded messages
    pub fn to_signed_binary(&self, keypair: &Keypair) -> Result<Vec<u8>, serde_json::Error> {
        let json = serde_json::to_vec(&self.messages)?;
        let payload_hash = crypto::sha256(&json);
        let signature = keypair.sign(&payload_hash);

        let mut buf = Vec::with_capacity(8 + 64 + 32 + 32 + 4 + json.len());
        buf.extend_from_slice(&[0x51, 0x4C, 0x4D, 0x53]); // "QLMS" magic
        buf.extend_from_slice(&2u16.to_le_bytes()); // version 2
        buf.extend_from_slice(&1u16.to_le_bytes()); // flags: SIGNED=0x01
        buf.extend_from_slice(&signature); // 64 bytes
        buf.extend_from_slice(keypair.public_key()); // 32 bytes
        buf.extend_from_slice(&payload_hash); // 32 bytes
        buf.extend_from_slice(&(self.messages.len() as u32).to_le_bytes());
        buf.extend_from_slice(&json);
        Ok(buf)
    }
}

impl Default for AgentConversation {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use qlang_core::graph::Graph;
    use qlang_core::ops::Op;
    use qlang_core::tensor::{Shape, TensorData, TensorType};

    fn trainer_agent() -> AgentId {
        AgentId {
            name: "trainer".into(),
            capabilities: vec![Capability::Execute, Capability::Train],
        }
    }

    fn compressor_agent() -> AgentId {
        AgentId {
            name: "compressor".into(),
            capabilities: vec![Capability::Compress, Capability::Verify],
        }
    }

    #[test]
    fn agent_conversation() {
        let mut conv = AgentConversation::new();

        // Trainer builds a model graph and sends it to compressor
        let mut graph = Graph::new("model_weights");
        graph.add_node(
            Op::Input { name: "weights".into() },
            vec![],
            vec![TensorType::f32_matrix(128, 64)],
        );

        let mut inputs = HashMap::new();
        inputs.insert(
            "weights".into(),
            TensorData::from_f32(
                Shape::matrix(2, 2),
                &[0.5, -0.3, 0.8, -0.1],
            ),
        );

        // Message 1: Trainer → Compressor: "compress these weights"
        let msg1 = conv.send(
            trainer_agent(),
            compressor_agent(),
            graph,
            inputs,
            MessageIntent::Compress { method: "ternary".into() },
            None,
        );
        assert_eq!(msg1, 0);

        // Message 2: Compressor → Trainer: "here are the compressed weights"
        let compressed_graph = Graph::new("compressed_weights");
        let msg2 = conv.send(
            compressor_agent(),
            trainer_agent(),
            compressed_graph,
            HashMap::new(),
            MessageIntent::Result { original_message_id: msg1 },
            Some(msg1),
        );
        assert_eq!(msg2, 1);

        assert_eq!(conv.messages().len(), 2);
        assert_eq!(conv.get_message(0).unwrap().from.name, "trainer");
        assert_eq!(conv.get_message(1).unwrap().in_reply_to, Some(0));
    }

    #[test]
    fn serialize_conversation() {
        let mut conv = AgentConversation::new();
        let graph = Graph::new("test");
        conv.send(
            trainer_agent(),
            compressor_agent(),
            graph,
            HashMap::new(),
            MessageIntent::Verify,
            None,
        );

        let binary = conv.to_binary().unwrap();
        assert_eq!(&binary[0..4], &[0x51, 0x4C, 0x4D, 0x53]); // "QLMS"
    }

    #[test]
    fn graph_message_with_signature() {
        let kp = qlang_core::crypto::Keypair::from_seed(&[10u8; 32]);
        let graph = Graph::new("signed");

        let mut conv = AgentConversation::new();
        let msg_id = conv.send(
            trainer_agent(),
            compressor_agent(),
            graph,
            HashMap::new(),
            MessageIntent::Execute,
            None,
        );

        let msg = conv.get_message(msg_id).unwrap();
        assert!(msg.signature.is_none());

        // Sign the message
        let signed_msg = msg.clone().sign(&kp);
        assert!(signed_msg.signature.is_some());
        assert!(signed_msg.verify_signature());
    }

    #[test]
    fn unsigned_message_verify_returns_true() {
        let graph = Graph::new("unsigned");
        let mut conv = AgentConversation::new();
        conv.send(
            trainer_agent(),
            compressor_agent(),
            graph,
            HashMap::new(),
            MessageIntent::Execute,
            None,
        );
        let msg = conv.get_message(0).unwrap();
        // Unsigned messages are valid (backward compatible)
        assert!(msg.verify_signature());
    }

    #[test]
    fn signed_binary_roundtrip() {
        let kp = qlang_core::crypto::Keypair::from_seed(&[11u8; 32]);
        let mut conv = AgentConversation::new();
        let graph = Graph::new("binary_signed");
        conv.send(
            trainer_agent(),
            compressor_agent(),
            graph,
            HashMap::new(),
            MessageIntent::Verify,
            None,
        );

        let binary = conv.to_signed_binary(&kp).unwrap();
        // Check QLMS magic
        assert_eq!(&binary[0..4], &[0x51, 0x4C, 0x4D, 0x53]);
        // Check version (byte 4-5)
        let version = u16::from_le_bytes([binary[4], binary[5]]);
        assert_eq!(version, 2);
        // Check flags (byte 6-7) -- signed flag should be set
        let flags = u16::from_le_bytes([binary[6], binary[7]]);
        assert!(flags & 0x01 != 0); // SIGNED flag
    }

    #[test]
    fn signed_message_serializes_deserializes() {
        let kp = qlang_core::crypto::Keypair::from_seed(&[12u8; 32]);
        let graph = Graph::new("serde_test");

        let mut conv = AgentConversation::new();
        conv.send(
            trainer_agent(),
            compressor_agent(),
            graph,
            HashMap::new(),
            MessageIntent::Execute,
            None,
        );
        let msg = conv.get_message(0).unwrap().clone().sign(&kp);

        // Roundtrip via JSON
        let json = serde_json::to_string(&msg).unwrap();
        let deserialized: GraphMessage = serde_json::from_str(&json).unwrap();
        assert!(deserialized.verify_signature());
        assert_eq!(deserialized.signature, msg.signature);
        assert_eq!(deserialized.signer_pubkey, msg.signer_pubkey);
        assert_eq!(deserialized.graph_hash, msg.graph_hash);
    }

    #[test]
    fn unsigned_message_omits_signature_in_json() {
        let graph = Graph::new("omit_test");
        let mut conv = AgentConversation::new();
        conv.send(
            trainer_agent(),
            compressor_agent(),
            graph,
            HashMap::new(),
            MessageIntent::Execute,
            None,
        );
        let msg = conv.get_message(0).unwrap();
        let json = serde_json::to_string(msg).unwrap();
        // skip_serializing_if = "Option::is_none" should omit these fields
        assert!(!json.contains("signature"));
        assert!(!json.contains("signer_pubkey"));
        assert!(!json.contains("graph_hash"));
    }

    // ================================================================
    // Agent coordination tests
    // ================================================================

    #[test]
    fn team_creation_and_members() {
        let mut team = Team::new("mnist-squad", "Train a digit classifier");
        assert_eq!(team.name, "mnist-squad");
        assert_eq!(team.task, "Train a digit classifier");
        assert_eq!(team.status, AgentStatus::Idle);
        assert!(team.agents.is_empty());

        team.add_member(
            trainer_agent(),
            AgentRole::Trainer,
            Some("deepseek-r1:1.5b".into()),
        );
        team.add_member(
            compressor_agent(),
            AgentRole::Compressor,
            None,
        );

        assert_eq!(team.agents.len(), 2);
        assert_eq!(team.agents[0].role, AgentRole::Trainer);
        assert_eq!(team.agents[0].model, Some("deepseek-r1:1.5b".into()));
        assert_eq!(team.agents[1].role, AgentRole::Compressor);
        assert_eq!(team.agents[1].model, None);
    }

    #[test]
    fn agent_status_transitions() {
        let mut member = TeamMember {
            agent: trainer_agent(),
            role: AgentRole::Trainer,
            status: AgentStatus::Idle,
            model: None,
        };

        assert_eq!(member.status, AgentStatus::Idle);

        member.status = AgentStatus::Thinking;
        assert_eq!(member.status, AgentStatus::Thinking);

        member.status = AgentStatus::Working {
            progress: 0.42,
            description: "Training epoch 5/10".into(),
        };
        if let AgentStatus::Working { progress, description } = &member.status {
            assert!((progress - 0.42).abs() < f32::EPSILON);
            assert_eq!(description, "Training epoch 5/10");
        } else {
            panic!("expected Working status");
        }

        member.status = AgentStatus::Done { result: "accuracy=0.97".into() };
        assert_eq!(member.status, AgentStatus::Done { result: "accuracy=0.97".into() });

        member.status = AgentStatus::Error { message: "OOM".into() };
        assert_eq!(member.status, AgentStatus::Error { message: "OOM".into() });
    }

    #[test]
    fn team_get_by_role() {
        let mut team = Team::new("test-team", "test task");

        team.add_member(trainer_agent(), AgentRole::Trainer, None);
        team.add_member(compressor_agent(), AgentRole::Compressor, None);
        team.add_member(
            AgentId {
                name: "architect".into(),
                capabilities: vec![Capability::Optimize],
            },
            AgentRole::Architect,
            Some("llama3:8b".into()),
        );

        let trainer = team.get_by_role(&AgentRole::Trainer);
        assert!(trainer.is_some());
        assert_eq!(trainer.unwrap().agent.name, "trainer");

        let compressor = team.get_by_role(&AgentRole::Compressor);
        assert!(compressor.is_some());
        assert_eq!(compressor.unwrap().agent.name, "compressor");

        let architect = team.get_by_role(&AgentRole::Architect);
        assert!(architect.is_some());
        assert_eq!(architect.unwrap().model, Some("llama3:8b".into()));

        // Role not in team returns None
        let deployer = team.get_by_role(&AgentRole::Deployer);
        assert!(deployer.is_none());
    }

    #[test]
    fn agent_coordination_serde_roundtrip() {
        let mut team = Team::new("serde-team", "roundtrip test");
        team.add_member(trainer_agent(), AgentRole::Trainer, Some("phi3:mini".into()));
        team.status = AgentStatus::Working {
            progress: 0.5,
            description: "half done".into(),
        };

        let json = serde_json::to_string(&team).unwrap();
        let deserialized: Team = serde_json::from_str(&json).unwrap();

        assert_eq!(deserialized.name, "serde-team");
        assert_eq!(deserialized.task, "roundtrip test");
        assert_eq!(deserialized.agents.len(), 1);
        assert_eq!(deserialized.agents[0].role, AgentRole::Trainer);
        assert_eq!(deserialized.agents[0].model, Some("phi3:mini".into()));
        if let AgentStatus::Working { progress, description } = &deserialized.status {
            assert!((progress - 0.5).abs() < f32::EPSILON);
            assert_eq!(description, "half done");
        } else {
            panic!("expected Working status after roundtrip");
        }

        // Also roundtrip individual types
        let role_json = serde_json::to_string(&AgentRole::Coordinator).unwrap();
        let role_back: AgentRole = serde_json::from_str(&role_json).unwrap();
        assert_eq!(role_back, AgentRole::Coordinator);

        let status = AgentStatus::Error { message: "test error".into() };
        let status_json = serde_json::to_string(&status).unwrap();
        let status_back: AgentStatus = serde_json::from_str(&status_json).unwrap();
        assert_eq!(status_back, AgentStatus::Error { message: "test error".into() });
    }
}
