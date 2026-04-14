//! QLANG SDK - Secure AI-to-AI Communication
//!
//! # Quick Start
//! ```rust,no_run
//! use qlang_sdk::prelude::*;
//!
//! // Generate a keypair for signing
//! let keypair = Keypair::generate();
//!
//! // Build a graph using GraphEmitter
//! let mut emitter = GraphEmitter::new("my_model");
//! let x = emitter.input("x", Dtype::F32, Shape::vector(4));
//!
//! // Sign it
//! let graph = emitter.build();
//! let signed = sign(&graph, &keypair);
//! assert!(signed.verify());
//! ```

pub mod prelude;

// Re-export core types
pub use qlang_core::graph::Graph;
pub use qlang_core::tensor::{TensorData, Dtype, Shape, Dim, TensorType};
pub use qlang_core::ops::Op;

// Re-export crypto from qlang-core
pub use qlang_core::crypto::{Keypair, SignedGraph, sha256, hash_graph};

// Re-export agent types
pub use qlang_agent::protocol::{GraphMessage, AgentId, MessageIntent, Capability};
pub use qlang_agent::emitter::GraphEmitter;

// Top-level convenience functions

/// Sign a graph with a keypair, producing a `SignedGraph`.
pub fn sign(graph: &Graph, keypair: &Keypair) -> SignedGraph {
    SignedGraph::sign(graph.clone(), keypair)
}

/// Verify a signed graph's signature.
pub fn verify(signed: &SignedGraph) -> bool {
    signed.verify()
}

/// Ternary compression: quantize f32 weights to {-1, 0, +1}.
///
/// Values with absolute magnitude below 0.5 map to 0,
/// positive values map to 1, negative values map to -1.
pub fn compress_ternary(weights: &[f32]) -> Vec<i8> {
    weights.iter().map(|&w| {
        if w.abs() < 0.5 { 0i8 }
        else if w > 0.0 { 1i8 }
        else { -1i8 }
    }).collect()
}

/// SDK version string.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Protocol version for AI-to-AI communication.
pub const PROTOCOL_VERSION: u16 = 2;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sign_verify_roundtrip() {
        let keypair = Keypair::generate();

        let mut emitter = GraphEmitter::new("test_graph");
        let _x = emitter.input("x", Dtype::F32, Shape::vector(4));
        let graph = emitter.build();

        let signed = sign(&graph, &keypair);
        assert!(verify(&signed));
    }

    #[test]
    fn test_sign_verify_detects_tampering() {
        let keypair = Keypair::generate();

        let mut emitter = GraphEmitter::new("test_graph");
        emitter.input("x", Dtype::F32, Shape::vector(4));
        let graph = emitter.build();

        let mut signed = sign(&graph, &keypair);
        // Tamper with the graph
        signed.graph.id = "tampered".to_string();
        assert!(!verify(&signed));
    }

    #[test]
    fn test_compress_ternary_basic() {
        let weights = vec![0.9, -0.8, 0.1, -0.2, 0.0, 1.5, -1.5];
        let compressed = compress_ternary(&weights);
        assert_eq!(compressed, vec![1, -1, 0, 0, 0, 1, -1]);
    }

    #[test]
    fn test_compress_ternary_empty() {
        let compressed = compress_ternary(&[]);
        assert!(compressed.is_empty());
    }

    #[test]
    fn test_compress_ternary_boundary() {
        // Exactly at 0.5 boundary
        let weights = vec![0.5, -0.5, 0.49, -0.49];
        let compressed = compress_ternary(&weights);
        assert_eq!(compressed, vec![1, -1, 0, 0]);
    }

    #[test]
    fn test_reexports_accessible() {
        // Verify that all re-exported types are accessible
        let _shape = Shape::vector(10);
        let _dtype = Dtype::F32;
        let _dim = Dim::Fixed(5);
        let _tt = TensorType::new(Dtype::F32, Shape::scalar());

        // Agent types
        let _agent = AgentId {
            name: "test".to_string(),
            capabilities: vec![Capability::Execute],
        };

        // Crypto types
        let _kp = Keypair::generate();
        let _hash = sha256(b"test");
        let graph = Graph::new("test");
        let _ghash = hash_graph(&graph);

        // Constants
        assert!(!VERSION.is_empty());
        assert_eq!(PROTOCOL_VERSION, 2);
    }

    #[test]
    fn test_prelude_imports() {
        // Ensure the prelude re-exports work
        use crate::prelude::*;
        let kp = Keypair::generate();
        let _ = kp;
    }
}
