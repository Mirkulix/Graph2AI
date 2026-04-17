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
pub mod autograd;
pub mod checkpoint;
pub mod optimizers;
pub mod diagnostics;
pub mod executor;
pub mod graph_train;
pub mod mnist;
pub mod profiler;
pub mod scheduler;
pub mod training;
pub mod transformer;
pub mod conv;
pub mod bench;
pub mod stdlib;
pub mod vm;
pub mod bytecode;
pub mod debugger;
pub mod linalg;
pub mod bitnet_math;
// --- Experimental ML modules ---------------------------------------------
// The following paths are known-broken on real data (see QLANG-STATUS.md)
// and are gated behind the `experimental-ml` feature so the default build
// only exposes stable code. Enable with `--features experimental-ml`.
#[cfg(feature = "experimental-ml")]
pub mod cifar10;
#[cfg(feature = "experimental-ml")]
pub mod cifar10_features;
pub mod forward_forward;
pub mod forward_forward_gpu;
// `candle_train` / `gpu_train` consume `mamba_train` + `qlang_lm` types,
// so they follow the experimental gate.
#[cfg(feature = "experimental-ml")]
pub mod candle_train;
#[cfg(feature = "experimental-ml")]
pub mod gpu_train;
pub mod graph_ff_train;
pub mod hdc;
#[cfg(feature = "experimental-ml")]
pub mod lm_export;
#[cfg(feature = "experimental-ml")]
pub mod mamba;
#[cfg(feature = "experimental-ml")]
pub mod mamba_train;
pub mod neurosymbolic;
pub mod noprop;
// `organism` embeds `SpikingNetwork` + `TrainableLM` directly, so it moves
// with the experimental gate.
#[cfg(feature = "experimental-ml")]
pub mod organism;
pub mod evolution;
#[cfg(feature = "experimental-ml")]
pub mod spiking;
#[cfg(feature = "experimental-ml")]
pub mod hybrid_spiking;
pub mod ttt;
#[cfg(feature = "experimental-ml")]
pub mod qlang_lm;
pub mod quantum_flow;
pub mod tci;
pub mod ternary_net;
pub mod ternary_ops;
pub mod ternary_brain;
pub mod ternary_ensemble;
pub mod ternary_matrix;
pub mod ternary_vote;
pub mod federation;
pub mod mcp_bridge;
pub mod diffusion;
pub mod theorems;
pub mod igqk;
pub mod igqk_compress;
pub mod config;
pub mod types;
pub mod unified;
pub mod concurrency;
pub mod graph_ops;
pub mod gpu_compute;
#[cfg(feature = "experimental-ml")]
pub mod gpu_mamba;
pub mod gpu_runtime;
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
pub mod orchestrator;
pub mod distributed_train;
pub mod tokenizer;
pub mod transformer_train;
pub mod web_server;
pub mod hybrid_router;
