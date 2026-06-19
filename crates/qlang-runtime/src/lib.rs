//! QLANG Runtime — Graph executor
//!
//! Executes QLANG graphs by:
//! 1. Topologically sorting nodes
//! 2. Executing each node in order
//! 3. Flowing tensor data along edges
//!
//! This is the interpreter backend (Phase 1).
//! Phase 2 will add LLVM JIT compilation.

pub mod accel;
pub mod checkpoint;
pub mod diagnostics;
pub mod executor;
pub mod profiler;
pub mod scheduler;
pub mod bench;
pub mod stdlib;
pub mod vm;
pub mod bytecode;
pub mod debugger;

pub mod mcp_bridge;
pub mod config;
pub mod types;
pub mod unified;
pub mod concurrency;
pub mod graph_ops;
pub mod federation;
pub mod registry;
pub mod parallel;
pub mod hub;
pub mod ollama;
pub mod providers;
pub mod cloud_http;
pub mod openai_client;
pub mod anthropic_client;
pub mod gemini_client;
pub mod groq_client;
pub mod deepseek_client;
pub mod orchestrator;
pub mod tokenizer;
pub mod web_server;
pub mod hybrid_router;
