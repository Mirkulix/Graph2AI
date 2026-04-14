//! Ahead-of-Time Compilation — Compile QLANG graphs to object files.
//!
//! Pipeline:
//!   QLANG Graph → LLVM IR → LLVM Machine Code → .o object file
//!
//! The object file can then be linked into any C/Rust/C++ program,
//! or directly into an executable.
//!
//! This is the path to self-hosting: QLANG compiles itself to native code.

use inkwell::context::Context;
use inkwell::targets::{
    CodeModel, FileType, InitializationConfig, RelocMode, Target, TargetMachine,
};
use inkwell::OptimizationLevel;

use qlang_core::graph::Graph;

use crate::codegen::{CodegenError, compile_graph};

/// Compile a QLANG graph to an object file (.o).
///
/// The resulting object file exports a function:
///   void qlang_graph(float* input_a, float* input_b, float* output, uint64_t n)
///
/// This can be linked with any C/Rust program:
///   extern void qlang_graph(float*, float*, float*, uint64_t);
pub fn compile_to_object(
    graph: &Graph,
    output_path: &str,
    opt_level: OptimizationLevel,
) -> Result<AotResult, CodegenError> {
    // Initialize LLVM targets
    Target::initialize_native(&InitializationConfig::default())
        .map_err(|e| CodegenError::LlvmError(e.to_string()))?;

    let context = Context::create();
    let compiled = compile_graph(&context, graph, opt_level)?;

    // Get the native target
    let triple = TargetMachine::get_default_triple();
    let cpu = TargetMachine::get_host_cpu_name();
    let features = TargetMachine::get_host_cpu_features();

    let target = Target::from_triple(&triple)
        .map_err(|e| CodegenError::LlvmError(e.to_string()))?;

    let machine = target
        .create_target_machine(
            &triple,
            cpu.to_str().unwrap_or("generic"),
            features.to_str().unwrap_or(""),
            opt_level,
            RelocMode::Default,
            CodeModel::Default,
        )
        .ok_or_else(|| CodegenError::LlvmError("Failed to create target machine".into()))?;

    // Emit object file
    let path = std::path::Path::new(output_path);
    machine
        .write_to_file(&compiled.module, FileType::Object, path)
        .map_err(|e| CodegenError::LlvmError(e.to_string()))?;

    // Also get assembly for inspection
    let asm = machine
        .write_to_memory_buffer(&compiled.module, FileType::Assembly)
        .map_err(|e| CodegenError::LlvmError(e.to_string()))?;

    let asm_str = std::str::from_utf8(asm.as_slice())
        .unwrap_or("<binary>")
        .to_string();

    let file_size = std::fs::metadata(output_path)
        .map(|m| m.len())
        .unwrap_or(0);

    Ok(AotResult {
        object_path: output_path.to_string(),
        file_size,
        target_triple: triple.as_str().to_string_lossy().to_string(),
        cpu: cpu.to_str().unwrap_or("unknown").to_string(),
        llvm_ir: compiled.llvm_ir,
        assembly: asm_str,
    })
}

/// Result of AOT compilation.
pub struct AotResult {
    /// Path to the generated object file.
    pub object_path: String,
    /// Size of the object file in bytes.
    pub file_size: u64,
    /// Target triple (e.g., "x86_64-unknown-linux-gnu").
    pub target_triple: String,
    /// Target CPU (e.g., "skylake").
    pub cpu: String,
    /// The LLVM IR that was compiled.
    pub llvm_ir: String,
    /// The native assembly code.
    pub assembly: String,
}

#[cfg(test)]
mod tests {
    use super::*;
    use qlang_core::graph::Graph;
    use qlang_core::ops::Op;
    use qlang_core::tensor::TensorType;

    #[test]
    fn compile_to_object_file() {
        let mut g = Graph::new("aot_test");
        let a = g.add_node(Op::Input { name: "a".into() }, vec![], vec![TensorType::f32_vector(4)]);
        let b = g.add_node(Op::Input { name: "b".into() }, vec![], vec![TensorType::f32_vector(4)]);
        let add = g.add_node(Op::Add, vec![TensorType::f32_vector(4); 2], vec![TensorType::f32_vector(4)]);
        let out = g.add_node(Op::Output { name: "y".into() }, vec![TensorType::f32_vector(4)], vec![]);
        g.add_edge(a, 0, add, 0, TensorType::f32_vector(4));
        g.add_edge(b, 0, add, 1, TensorType::f32_vector(4));
        g.add_edge(add, 0, out, 0, TensorType::f32_vector(4));

        let result = compile_to_object(&g, "/tmp/qlang_test.o", OptimizationLevel::Aggressive).unwrap();

        assert!(result.file_size > 0);
        assert!(result.assembly.contains("qlang_graph"));
        assert!(result.llvm_ir.contains("fadd float"));

        // Clean up
        let _ = std::fs::remove_file("/tmp/qlang_test.o");
    }

    #[test]
    fn aot_relu_generates_native_code() {
        let mut g = Graph::new("aot_relu");
        let a = g.add_node(Op::Input { name: "a".into() }, vec![], vec![TensorType::f32_vector(8)]);
        let _b = g.add_node(Op::Input { name: "b".into() }, vec![], vec![TensorType::f32_vector(8)]);
        let relu = g.add_node(Op::Relu, vec![TensorType::f32_vector(8)], vec![TensorType::f32_vector(8)]);
        let out = g.add_node(Op::Output { name: "y".into() }, vec![TensorType::f32_vector(8)], vec![]);
        g.add_edge(a, 0, relu, 0, TensorType::f32_vector(8));
        g.add_edge(relu, 0, out, 0, TensorType::f32_vector(8));

        let result = compile_to_object(&g, "/tmp/qlang_relu.o", OptimizationLevel::Aggressive).unwrap();

        // The assembly should contain x86 instructions
        assert!(result.assembly.len() > 100);

        println!("Target: {} ({})", result.target_triple, result.cpu);
        println!("Object size: {} bytes", result.file_size);
        println!("Assembly excerpt:\n{}", &result.assembly[..result.assembly.len().min(500)]);

        let _ = std::fs::remove_file("/tmp/qlang_relu.o");
    }
}
