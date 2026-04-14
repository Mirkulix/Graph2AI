//! Example: JIT-compile a QLANG graph to native machine code via LLVM.
//!
//! This demonstrates the core innovation:
//! 1. AI agent builds a graph (no text, no syntax)
//! 2. Graph is compiled to LLVM IR
//! 3. LLVM IR is JIT-compiled to native x86-64 machine code
//! 4. Native code executes at C/Rust speed
//!
//! The compiled code uses the SAME LLVM backend as Clang (C) and Rustc (Rust).
//! Same performance. Zero interpreter overhead.


fn main() {
    #[cfg(not(feature = "llvm"))]
    {
        eprintln!("LLVM not available. Build with: cargo run --example jit_compile --features llvm");
        return;
    }

    #[cfg(feature = "llvm")]
    run();
}

#[cfg(feature = "llvm")]
fn run() {
    use std::time::Instant;
    println!("=== QLANG JIT Compilation Demo ===\n");

    use inkwell::context::Context;
    use inkwell::OptimizationLevel;
    use qlang_core::tensor::{Dtype, Shape, TensorType as TT};

    // ─── 1. Build graph: element-wise operations ───
    println!("[1] Building computation graph...");
    let mut e = qlang_agent::emitter::GraphEmitter::new("jit_demo");

    let a = e.input("a", Dtype::F32, Shape::vector(1024));
    let b = e.input("b", Dtype::F32, Shape::vector(1024));
    let sum = e.add(a, b, TT::f32_vector(1024));
    let activated = e.relu(sum, TT::f32_vector(1024));
    e.output("y", activated, TT::f32_vector(1024));

    let graph = e.build();
    println!("  Graph: {} nodes, {} edges\n", graph.nodes.len(), graph.edges.len());

    // ─── 2. Show ASCII visualization ───
    println!("[2] Graph visualization:");
    println!("{}", qlang_compile::visualize::to_ascii(&graph));

    // ─── 3. Compile to LLVM IR ───
    println!("[3] Compiling to LLVM IR...");
    let context = Context::create();
    let compiled = qlang_compile::codegen::compile_graph(
        &context,
        &graph,
        OptimizationLevel::Aggressive,
    )
    .expect("JIT compilation failed");

    println!("  LLVM IR generated ({} bytes)", compiled.llvm_ir.len());
    println!("\n  --- LLVM IR (excerpt) ---");
    for line in compiled.llvm_ir.lines().take(30) {
        println!("  {line}");
    }
    println!("  --- end ---\n");

    // ─── 4. Execute: interpreter vs JIT ───
    let n = 1024;
    let input_a: Vec<f32> = (0..n).map(|i| (i as f32) * 0.01 - 5.0).collect();
    let input_b: Vec<f32> = (0..n).map(|i| (i as f32) * 0.005).collect();

    // JIT execution
    println!("[4] Executing compiled native code...");
    let start_jit = Instant::now();
    let jit_result = qlang_compile::codegen::execute_compiled(
        &compiled,
        &input_a,
        &input_b,
    )
    .expect("JIT execution failed");
    let jit_time = start_jit.elapsed();

    println!("  JIT time:    {:?}", jit_time);
    println!("  First 10 values: {:?}", &jit_result[..10]);
    println!("  Last 5 values:   {:?}", &jit_result[n - 5..]);

    // Interpreter execution for comparison
    println!("\n[5] Executing via interpreter (for comparison)...");
    let mut inputs = std::collections::HashMap::new();
    inputs.insert(
        "a".to_string(),
        qlang_core::tensor::TensorData::from_f32(Shape::vector(n), &input_a),
    );
    inputs.insert(
        "b".to_string(),
        qlang_core::tensor::TensorData::from_f32(Shape::vector(n), &input_b),
    );

    let start_interp = Instant::now();
    let interp_result = qlang_runtime::executor::execute(&graph, inputs).unwrap();
    let interp_time = start_interp.elapsed();

    let interp_values = interp_result.outputs.get("y").unwrap().as_f32_slice().unwrap();
    println!("  Interpreter: {:?}", interp_time);
    println!("  First 10 values: {:?}", &interp_values[..10]);

    // Verify results match
    let max_diff: f32 = jit_result
        .iter()
        .zip(interp_values.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f32::max);
    println!("\n  Max difference: {max_diff:.2e}");
    if max_diff < 1e-6 {
        println!("  Results MATCH — JIT produces identical output to interpreter.");
    }

    // ─── 6. IGQK Ternary Compression via JIT ───
    println!("\n[6] IGQK Ternary Compression (JIT)...");

    let mut ternary_emitter = qlang_agent::emitter::GraphEmitter::new("jit_ternary");
    let w = ternary_emitter.input("weights", Dtype::F32, Shape::vector(1024));
    let _dummy = ternary_emitter.input("dummy", Dtype::F32, Shape::vector(1024));
    let compressed = ternary_emitter.to_ternary(w, TT::f32_vector(1024));
    ternary_emitter.output("ternary", compressed, TT::f32_vector(1024));
    let ternary_graph = ternary_emitter.build();

    let ternary_compiled = qlang_compile::codegen::compile_graph(
        &context,
        &ternary_graph,
        OptimizationLevel::Aggressive,
    )
    .expect("Ternary JIT failed");

    let weights: Vec<f32> = (0..1024).map(|i| (i as f32 * 0.01).sin()).collect();
    let dummy = vec![0.0f32; 1024];
    let ternary_result =
        qlang_compile::codegen::execute_compiled(&ternary_compiled, &weights, &dummy).unwrap();

    let count_pos = ternary_result.iter().filter(|&&v| v == 1.0).count();
    let count_neg = ternary_result.iter().filter(|&&v| v == -1.0).count();
    let count_zero = ternary_result.iter().filter(|&&v| v == 0.0).count();

    println!("  Ternary distribution:");
    println!("    +1: {} ({:.1}%)", count_pos, count_pos as f64 / 1024.0 * 100.0);
    println!("     0: {} ({:.1}%)", count_zero, count_zero as f64 / 1024.0 * 100.0);
    println!("    -1: {} ({:.1}%)", count_neg, count_neg as f64 / 1024.0 * 100.0);
    println!("  Compression: f32 → ternary = {:.0}x (with 2-bit packing: {:.0}x)",
        4.0, 4.0 * 8.0 / 2.0);

    // ─── 7. Show DOT output ───
    println!("\n[7] Graphviz DOT output (pipe to 'dot -Tpng > graph.png'):");
    println!("{}", qlang_compile::visualize::to_dot(&graph));

    println!("=== Done. QLANG compiles graphs to native machine code. ===");
}
