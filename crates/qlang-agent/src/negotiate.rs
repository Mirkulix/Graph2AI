//! Auto-Negotiation Protocol for AI Agent Communication.
//!
//! When two AI agents connect, they automatically negotiate capabilities:
//! - Data types (f32, f16, int8, ternary, utf8)
//! - Operations (matmul, relu, softmax, ...)
//! - Hardware (GPU type, max tensor size)
//! - Protocol features (streaming, compression, signing, merkle proofs)
//!
//! The result is a `NegotiatedProtocol` that both sides agree on --
//! the best common denominator of their capabilities.

use serde::{Deserialize, Serialize};

/// Full capability description of an AI agent.
///
/// Exchanged during the handshake phase. Each agent sends its
/// capabilities, then both compute the negotiated protocol.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentCapabilities {
    /// Human-readable agent name.
    pub agent_name: String,
    /// Protocol version (must match for communication).
    pub protocol_version: u16,
    /// Supported data types for tensor exchange.
    pub supported_dtypes: Vec<String>,
    /// Supported operations (from the Op catalog).
    pub supported_ops: Vec<String>,
    /// Whether the agent has GPU acceleration.
    pub has_gpu: bool,
    /// GPU type, if available.
    pub gpu_type: Option<String>,
    /// Maximum tensor size this agent can handle (bytes).
    pub max_tensor_size: u64,
    /// Whether the agent supports streaming results.
    pub supports_streaming: bool,
    /// Whether the agent supports tensor compression.
    pub supports_compression: bool,
    /// Supported compression methods.
    pub compression_methods: Vec<String>,
    /// Whether the agent supports cryptographic signing.
    pub supports_signing: bool,
    /// Whether the agent supports Merkle tree proofs.
    pub supports_merkle: bool,
    /// Estimated bandwidth in Mbps (None = unknown).
    pub bandwidth_estimate_mbps: Option<f64>,
}

/// The agreed protocol between two agents after negotiation.
///
/// Both sides compute this identically from their capabilities,
/// so no further round-trip is needed.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct NegotiatedProtocol {
    /// Agreed data type for tensor communication.
    pub dtype: String,
    /// Agreed compression method (None = no compression).
    pub compression: Option<String>,
    /// Agreed batch size for streaming.
    pub batch_size: usize,
    /// Whether to stream results incrementally.
    pub streaming: bool,
    /// Whether to sign messages.
    pub signing: bool,
    /// Whether to include Merkle proofs.
    pub merkle: bool,
}

impl AgentCapabilities {
    /// Create capabilities for a local QLANG agent.
    ///
    /// Reflects the current crate's built-in support:
    /// all dtypes, all ops, streaming, compression, signing, merkle.
    pub fn local() -> Self {
        AgentCapabilities {
            agent_name: "qlang-local".to_string(),
            protocol_version: 2,
            supported_dtypes: vec![
                "f32".into(),
                "f16".into(),
                "int8".into(),
                "ternary".into(),
                "utf8".into(),
            ],
            supported_ops: vec![
                "add".into(),
                "sub".into(),
                "mul".into(),
                "div".into(),
                "matmul".into(),
                "relu".into(),
                "sigmoid".into(),
                "softmax".into(),
                "to_ternary".into(),
                "ollama_generate".into(),
                "ollama_chat".into(),
            ],
            has_gpu: cfg!(target_os = "macos"),
            gpu_type: if cfg!(target_os = "macos") {
                Some("apple_mlx".into())
            } else {
                None
            },
            max_tensor_size: 1024 * 1024 * 1024, // 1 GB
            supports_streaming: true,
            supports_compression: true,
            compression_methods: vec!["ternary".into()],
            supports_signing: true,
            supports_merkle: true,
            bandwidth_estimate_mbps: None,
        }
    }

    /// Negotiate a common protocol between this agent and another.
    ///
    /// The negotiation is deterministic and symmetric:
    /// `a.negotiate(&b)` produces the same result as `b.negotiate(&a)`.
    ///
    /// Strategy:
    /// - dtype: pick the highest-priority type both support
    /// - compression: use the first method both support (if any)
    /// - streaming: enabled if both support it
    /// - signing: enabled if both support it
    /// - merkle: enabled if both support it
    /// - batch_size: fixed at 64 (future: based on bandwidth/memory)
    pub fn negotiate(&self, other: &AgentCapabilities) -> NegotiatedProtocol {
        // Find best common dtype (ordered by preference)
        let dtype_priority = ["f16", "f32", "int8", "ternary"];
        let dtype = dtype_priority
            .iter()
            .find(|d| {
                self.supported_dtypes.contains(&d.to_string())
                    && other.supported_dtypes.contains(&d.to_string())
            })
            .unwrap_or(&"f32")
            .to_string();

        // Compression: use if both support it, pick first common method
        let compression = if self.supports_compression && other.supports_compression {
            self.compression_methods
                .iter()
                .find(|m| other.compression_methods.contains(m))
                .cloned()
        } else {
            None
        };

        // Feature flags: enabled only if both support it
        let streaming = self.supports_streaming && other.supports_streaming;
        let signing = self.supports_signing && other.supports_signing;
        let merkle = self.supports_merkle && other.supports_merkle;

        // Batch size: fixed for now, could be computed from bandwidth/memory
        let batch_size = 64;

        NegotiatedProtocol {
            dtype,
            compression,
            batch_size,
            streaming,
            signing,
            merkle,
        }
    }

    /// Find operations supported by both agents.
    pub fn common_ops(&self, other: &AgentCapabilities) -> Vec<String> {
        self.supported_ops
            .iter()
            .filter(|op| other.supported_ops.contains(op))
            .cloned()
            .collect()
    }

    /// Find data types supported by both agents.
    pub fn common_dtypes(&self, other: &AgentCapabilities) -> Vec<String> {
        self.supported_dtypes
            .iter()
            .filter(|dt| other.supported_dtypes.contains(dt))
            .cloned()
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- Helpers ----

    fn local_agent() -> AgentCapabilities {
        AgentCapabilities::local()
    }

    fn minimal_agent() -> AgentCapabilities {
        AgentCapabilities {
            agent_name: "minimal".to_string(),
            protocol_version: 2,
            supported_dtypes: vec!["f32".into()],
            supported_ops: vec!["add".into(), "mul".into()],
            has_gpu: false,
            gpu_type: None,
            max_tensor_size: 1024 * 1024, // 1 MB
            supports_streaming: false,
            supports_compression: false,
            compression_methods: vec![],
            supports_signing: false,
            supports_merkle: false,
            bandwidth_estimate_mbps: Some(10.0),
        }
    }

    fn gpu_agent() -> AgentCapabilities {
        AgentCapabilities {
            agent_name: "gpu-worker".to_string(),
            protocol_version: 2,
            supported_dtypes: vec!["f32".into(), "f16".into(), "int8".into()],
            supported_ops: vec![
                "add".into(),
                "mul".into(),
                "matmul".into(),
                "relu".into(),
                "softmax".into(),
            ],
            has_gpu: true,
            gpu_type: Some("cuda".into()),
            max_tensor_size: 8 * 1024 * 1024 * 1024, // 8 GB
            supports_streaming: true,
            supports_compression: true,
            compression_methods: vec!["ternary".into(), "int8".into()],
            supports_signing: true,
            supports_merkle: true,
            bandwidth_estimate_mbps: Some(1000.0),
        }
    }

    fn edge_agent() -> AgentCapabilities {
        AgentCapabilities {
            agent_name: "edge-device".to_string(),
            protocol_version: 2,
            supported_dtypes: vec!["int8".into(), "ternary".into()],
            supported_ops: vec!["add".into(), "relu".into()],
            has_gpu: false,
            gpu_type: None,
            max_tensor_size: 64 * 1024, // 64 KB
            supports_streaming: true,
            supports_compression: true,
            compression_methods: vec!["ternary".into()],
            supports_signing: false,
            supports_merkle: true,
            bandwidth_estimate_mbps: Some(1.0),
        }
    }

    // ---- Two identical agents ----

    #[test]
    fn negotiate_local_with_local() {
        let a = local_agent();
        let b = local_agent();
        let proto = a.negotiate(&b);

        // Both support f16, which is highest priority
        assert_eq!(proto.dtype, "f16");
        assert_eq!(proto.compression, Some("ternary".into()));
        assert!(proto.streaming);
        assert!(proto.signing);
        assert!(proto.merkle);
        assert_eq!(proto.batch_size, 64);
    }

    // ---- Local + minimal ----

    #[test]
    fn negotiate_local_with_minimal() {
        let a = local_agent();
        let b = minimal_agent();
        let proto = a.negotiate(&b);

        // Only f32 is common
        assert_eq!(proto.dtype, "f32");
        // Minimal doesn't support compression
        assert_eq!(proto.compression, None);
        // Minimal doesn't support streaming, signing, or merkle
        assert!(!proto.streaming);
        assert!(!proto.signing);
        assert!(!proto.merkle);
    }

    // ---- GPU + edge ----

    #[test]
    fn negotiate_gpu_with_edge() {
        let a = gpu_agent();
        let b = edge_agent();
        let proto = a.negotiate(&b);

        // GPU supports [f32, f16, int8], edge supports [int8, ternary].
        // Common dtype in priority list: "int8".
        assert_eq!(proto.dtype, "int8");
        // Both support ternary compression
        assert_eq!(proto.compression, Some("ternary".into()));
        // Both support streaming
        assert!(proto.streaming);
        // Edge doesn't support signing
        assert!(!proto.signing);
        // Both support merkle
        assert!(proto.merkle);
    }

    // ---- Symmetry: a.negotiate(b) == b.negotiate(a) ----

    #[test]
    fn negotiation_is_symmetric() {
        let a = local_agent();
        let b = gpu_agent();
        let proto_ab = a.negotiate(&b);
        let proto_ba = b.negotiate(&a);
        assert_eq!(proto_ab, proto_ba);
    }

    #[test]
    fn negotiation_is_symmetric_edge_case() {
        let a = minimal_agent();
        let b = edge_agent();
        let proto_ab = a.negotiate(&b);
        let proto_ba = b.negotiate(&a);
        assert_eq!(proto_ab, proto_ba);
    }

    // ---- No common dtype defaults to f32 ----

    #[test]
    fn no_common_dtype_defaults_to_f32() {
        // Edge only supports int8 + ternary, which are in the priority list
        // but let's make an agent that only supports an exotic dtype
        let exotic = AgentCapabilities {
            agent_name: "exotic".into(),
            protocol_version: 2,
            supported_dtypes: vec!["bfloat16".into()], // not in priority list
            supported_ops: vec![],
            has_gpu: false,
            gpu_type: None,
            max_tensor_size: 0,
            supports_streaming: false,
            supports_compression: false,
            compression_methods: vec![],
            supports_signing: false,
            supports_merkle: false,
            bandwidth_estimate_mbps: None,
        };
        let local = local_agent();
        let proto = local.negotiate(&exotic);
        // No common dtype in priority list -> defaults to "f32"
        assert_eq!(proto.dtype, "f32");
    }

    // ---- No common compression method ----

    #[test]
    fn no_common_compression_method() {
        let a = AgentCapabilities {
            agent_name: "a".into(),
            protocol_version: 2,
            supported_dtypes: vec!["f32".into()],
            supported_ops: vec![],
            has_gpu: false,
            gpu_type: None,
            max_tensor_size: 0,
            supports_streaming: false,
            supports_compression: true,
            compression_methods: vec!["zstd".into()],
            supports_signing: false,
            supports_merkle: false,
            bandwidth_estimate_mbps: None,
        };
        let b = AgentCapabilities {
            agent_name: "b".into(),
            protocol_version: 2,
            supported_dtypes: vec!["f32".into()],
            supported_ops: vec![],
            has_gpu: false,
            gpu_type: None,
            max_tensor_size: 0,
            supports_streaming: false,
            supports_compression: true,
            compression_methods: vec!["lz4".into()],
            supports_signing: false,
            supports_merkle: false,
            bandwidth_estimate_mbps: None,
        };
        let proto = a.negotiate(&b);
        assert_eq!(proto.compression, None);
    }

    // ---- Common ops ----

    #[test]
    fn common_ops_between_agents() {
        let a = local_agent();
        let b = gpu_agent();
        let common = a.common_ops(&b);
        assert!(common.contains(&"add".to_string()));
        assert!(common.contains(&"mul".to_string()));
        assert!(common.contains(&"matmul".to_string()));
        assert!(common.contains(&"relu".to_string()));
        assert!(common.contains(&"softmax".to_string()));
        // Only local has these
        assert!(!common.contains(&"sigmoid".to_string()));
        assert!(!common.contains(&"to_ternary".to_string()));
    }

    #[test]
    fn common_ops_minimal_with_edge() {
        let a = minimal_agent();
        let b = edge_agent();
        let common = a.common_ops(&b);
        assert!(common.contains(&"add".to_string()));
        // minimal has "mul", edge does not
        assert!(!common.contains(&"mul".to_string()));
        // edge has "relu", minimal does not
        assert!(!common.contains(&"relu".to_string()));
    }

    // ---- Common dtypes ----

    #[test]
    fn common_dtypes_between_agents() {
        let a = local_agent();
        let b = gpu_agent();
        let common = a.common_dtypes(&b);
        assert!(common.contains(&"f32".to_string()));
        assert!(common.contains(&"f16".to_string()));
    }

    // ---- Serialization roundtrip ----

    #[test]
    fn capabilities_serde_roundtrip() {
        let cap = local_agent();
        let json = serde_json::to_string(&cap).unwrap();
        let decoded: AgentCapabilities = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.agent_name, cap.agent_name);
        assert_eq!(decoded.protocol_version, cap.protocol_version);
        assert_eq!(decoded.supported_dtypes, cap.supported_dtypes);
        assert_eq!(decoded.supported_ops, cap.supported_ops);
        assert_eq!(decoded.has_gpu, cap.has_gpu);
        assert_eq!(decoded.supports_streaming, cap.supports_streaming);
    }

    #[test]
    fn negotiated_protocol_serde_roundtrip() {
        let a = local_agent();
        let b = gpu_agent();
        let proto = a.negotiate(&b);
        let json = serde_json::to_string(&proto).unwrap();
        let decoded: NegotiatedProtocol = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, proto);
    }

    // ---- Self-negotiation is stable ----

    #[test]
    fn self_negotiation_is_stable() {
        let a = local_agent();
        let p1 = a.negotiate(&a);
        let p2 = a.negotiate(&a);
        assert_eq!(p1, p2);
    }
}
