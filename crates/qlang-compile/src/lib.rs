//! QLANG Compiler — Graph to machine code via LLVM
//!
//! Compilation pipeline:
//!   QLANG Graph → Optimize → Schedule → LLVM IR → Machine Code
//!
//! Uses inkwell (Rust LLVM bindings) for native code generation.
//! The compiled code runs at the same speed as hand-written C/Rust —
//! because it IS the same LLVM backend that C and Rust use.

#[cfg(feature = "llvm")]
pub mod aligned;
#[cfg(feature = "llvm")]
pub mod aot;
#[cfg(feature = "llvm")]
pub mod codegen;
pub mod gpu;
#[cfg(feature = "llvm")]
pub mod matmul_jit;
pub mod parser;
pub mod repl;
pub mod wasm;
pub mod optimize;
#[cfg(feature = "llvm")]
pub mod script_jit;
#[cfg(feature = "llvm")]
pub mod simd;
pub mod visualize;
pub mod selfhost;
pub mod lsp;
pub mod onnx;
pub mod api;
pub mod modules;
