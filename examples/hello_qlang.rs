//! First QLANG example: Build and execute a simple computation graph.
//!
//! This demonstrates:
//! 1. Building a graph programmatically (as an AI agent would)
//! 2. Executing the graph on concrete tensor data
//! 3. IGQK ternary compression
//!
//! The graph computes: output = relu(A * B) → ternary compression
//!
//! In the future, an AI agent would emit this graph directly —
//! no text parsing, no syntax, just structured decisions.

use std::collections::HashMap;

fn main() {
    println!("=== QLANG v0.1 — First Graph Execution ===\n");

    // ─── Step 1: AI agent builds the graph ───
    println!("[1] Building computation graph...");

    let mut emitter = qlang_agent::emitter::GraphEmitter::new("hello_qlang");

    // Input: a 2×3 matrix (e.g., mini-batch of 2 samples, 3 features)
    let a = emitter.input("A", qlang_core::tensor::Dtype::F32, qlang_core::tensor::Shape::matrix(2, 3));

    // Input: a 3×2 weight matrix
    let b = emitter.input("B", qlang_core::tensor::Dtype::F32, qlang_core::tensor::Shape::matrix(3, 2));

    // MatMul: A × B → [2, 2]
    let matmul = emitter.matmul(
        a, b,
        qlang_core::tensor::TensorType::f32_matrix(2, 3),
        qlang_core::tensor::TensorType::f32_matrix(3, 2),
        qlang_core::tensor::TensorType::f32_matrix(2, 2),
    );

    // ReLU activation
    let activated = emitter.relu(matmul, qlang_core::tensor::TensorType::f32_matrix(2, 2));

    // IGQK: Compress to ternary weights
    let compressed = emitter.to_ternary(activated, qlang_core::tensor::TensorType::f32_matrix(2, 2));

    // Output
    emitter.output(
        "result",
        compressed,
        qlang_core::tensor::TensorType::ternary_matrix(2, 2),
    );

    let graph = emitter.build();

    println!("  Graph built: {} nodes, {} edges", graph.nodes.len(), graph.edges.len());
    println!("\n{graph}");

    // ─── Step 2: Verify the graph ───
    println!("[2] Verifying graph...");
    let verification = qlang_core::verify::verify_graph(&graph);
    println!("  {verification}");

    // ─── Step 3: Serialize to binary and back ───
    println!("[3] Serializing to QLANG binary format (.qlg)...");
    let binary = qlang_core::serial::to_binary(&graph).expect("serialization failed");
    println!("  Binary size: {} bytes", binary.len());
    println!("  Magic: {:?}", &binary[0..4]);

    let graph_back = qlang_core::serial::from_binary(&binary).expect("deserialization failed");
    println!("  Roundtrip OK: {} nodes recovered", graph_back.nodes.len());

    // ─── Step 4: Execute with concrete data ───
    println!("\n[4] Executing graph...");

    let mut inputs = HashMap::new();

    // A = [[1, 2, 3], [4, 5, 6]]
    inputs.insert(
        "A".to_string(),
        qlang_core::tensor::TensorData::from_f32(
            qlang_core::tensor::Shape::matrix(2, 3),
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        ),
    );

    // B = [[0.5, -0.5], [-0.3, 0.8], [0.1, -0.2]]
    inputs.insert(
        "B".to_string(),
        qlang_core::tensor::TensorData::from_f32(
            qlang_core::tensor::Shape::matrix(3, 2),
            &[0.5, -0.5, -0.3, 0.8, 0.1, -0.2],
        ),
    );

    match qlang_runtime::executor::execute(&graph, inputs) {
        Ok(result) => {
            println!("  Execution complete!");
            println!("  Nodes executed: {}", result.stats.nodes_executed);
            println!("  Quantum ops:    {}", result.stats.quantum_ops);
            println!("  Total FLOPs:    {}", result.stats.total_flops);

            if let Some(output) = result.outputs.get("result") {
                println!("\n  Output tensor:");
                println!("    dtype: {}", output.dtype);
                println!("    shape: {}", output.shape);
                println!("    data (raw bytes): {:?}", &output.data);
                println!("\n  Ternary values: -1=0xFF, 0=0x00, +1=0x01");
            }
        }
        Err(e) => {
            println!("  Execution failed: {e}");
        }
    }

    // ─── Step 5: Show JSON representation (human view) ───
    println!("\n[5] JSON view (for human inspection):");
    let json = qlang_core::serial::to_json(&graph).unwrap();
    // Just show first 500 chars
    if json.len() > 500 {
        println!("  {}...", &json[..500]);
    } else {
        println!("  {json}");
    }

    println!("\n=== Done. QLANG Phase 1 operational. ===");
}
