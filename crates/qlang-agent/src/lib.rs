//! QLANG Agent Interface — AI-to-AI graph communication
//!
//! This crate provides the interface for AI agents to:
//! - Emit QLANG graphs (structured decisions, not text)
//! - Receive and interpret QLANG graphs
//! - Compose graphs from sub-graphs
//!
//! The key insight: instead of generating code as text (token by token),
//! an AI agent makes structured decisions:
//!   1. Choose a node type (from catalog)
//!   2. Choose an operation (from catalog)
//!   3. Wire inputs (select edges)
//!   4. Set constraints (from catalog)
//!
//! 4 decisions instead of 47 tokens. Each decision is valid by construction.

pub mod bridge;
pub mod bus;
pub mod compose;
pub mod diff;
pub mod distributed;
pub mod kubernetes;
pub mod emitter;
pub mod negotiate;
pub mod packages;
pub mod protocol;
pub mod server;
