//! Convenience re-exports for common QLANG SDK usage.
//!
//! ```rust
//! use qlang_sdk::prelude::*;
//! ```

// SDK top-level functions
pub use crate::{sign, verify, compress_ternary};

// Core types
pub use crate::{Graph, TensorData, Dtype, Shape, Dim, TensorType, Op};

// Crypto
pub use crate::{Keypair, SignedGraph, sha256, hash_graph};

// Agent types
pub use crate::GraphEmitter;
pub use crate::{GraphMessage, AgentId, MessageIntent, Capability};
