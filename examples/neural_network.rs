//! Example: A simple neural network (MLP) expressed as a QLANG graph.
//!
//! Architecture: Input(784) → Dense(128) → ReLU → Dense(10) → Softmax
//! Then: IGQK ternary compression of weights
//!
//! This is what an AI agent would emit to describe a neural network —
//! no Python, no PyTorch, just a graph.

use std::collections::HashMap;

fn main() {
    println!("=== QLANG Neural Network Example ===\n");

    // ─── Build the MLP graph ───
    let mut e = qlang_agent::emitter::GraphEmitter::new("mnist_mlp");

    use qlang_core::tensor::{Dtype, Shape, TensorType as TT};

    // Inputs
    let x = e.input("x", Dtype::F32, Shape::matrix(1, 784)); // 28×28 image, flattened
    let w1 = e.input("W1", Dtype::F32, Shape::matrix(784, 128)); // Layer 1 weights
    let b1 = e.input("b1", Dtype::F32, Shape::matrix(1, 128)); // Layer 1 bias
    let w2 = e.input("W2", Dtype::F32, Shape::matrix(128, 10)); // Layer 2 weights
    let b2 = e.input("b2", Dtype::F32, Shape::matrix(1, 10)); // Layer 2 bias

    // Layer 1: x @ W1 + b1
    let h1 = e.matmul(x, w1, TT::f32_matrix(1, 784), TT::f32_matrix(784, 128), TT::f32_matrix(1, 128));
    let h1_biased = e.add(h1, b1, TT::f32_matrix(1, 128));
    let h1_activated = e.relu(h1_biased, TT::f32_matrix(1, 128));

    // Layer 2: h1 @ W2 + b2
    let h2 = e.matmul(h1_activated, w2, TT::f32_matrix(1, 128), TT::f32_matrix(128, 10), TT::f32_matrix(1, 10));
    let logits = e.add(h2, b2, TT::f32_matrix(1, 10));

    // Output: softmax probabilities
    e.output("probabilities", logits, TT::f32_matrix(1, 10));

    let graph = e.build();

    println!("MLP Graph:");
    println!("{graph}");

    // ─── Verify ───
    let verification = qlang_core::verify::verify_graph(&graph);
    println!("{verification}");

    // ─── Execute with random-ish weights ───
    println!("Executing with sample data...\n");

    let mut inputs = HashMap::new();

    // Simulate a digit "1": mostly zeros with a vertical line
    let mut image = vec![0.0f32; 784];
    for row in 4..24 {
        image[row * 28 + 14] = 1.0; // vertical line in center
        image[row * 28 + 15] = 0.8;
    }
    inputs.insert("x".to_string(), qlang_core::tensor::TensorData::from_f32(Shape::matrix(1, 784), &image));

    // Random weights (small values)
    let w1_data: Vec<f32> = (0..784 * 128).map(|i| ((i as f32 * 0.001).sin() * 0.1)).collect();
    inputs.insert("W1".to_string(), qlang_core::tensor::TensorData::from_f32(Shape::matrix(784, 128), &w1_data));

    let b1_data = vec![0.01f32; 128];
    inputs.insert("b1".to_string(), qlang_core::tensor::TensorData::from_f32(Shape::matrix(1, 128), &b1_data));

    let w2_data: Vec<f32> = (0..128 * 10).map(|i| ((i as f32 * 0.01).cos() * 0.1)).collect();
    inputs.insert("W2".to_string(), qlang_core::tensor::TensorData::from_f32(Shape::matrix(128, 10), &w2_data));

    let b2_data = vec![0.0f32; 10];
    inputs.insert("b2".to_string(), qlang_core::tensor::TensorData::from_f32(Shape::matrix(1, 10), &b2_data));

    match qlang_runtime::executor::execute(&graph, inputs) {
        Ok(result) => {
            println!("Execution Stats:");
            println!("  Nodes executed: {}", result.stats.nodes_executed);
            println!("  Total FLOPs:    {}", result.stats.total_flops);

            if let Some(probs) = result.outputs.get("probabilities") {
                let values = probs.as_f32_slice().unwrap();
                println!("\n  Raw logits for digits 0-9:");
                for (i, &v) in values.iter().enumerate() {
                    let bar = "█".repeat((v.abs() * 100.0).min(50.0) as usize);
                    println!("    [{i}] {v:>8.4} {bar}");
                }

                // Find predicted class
                let (predicted, &confidence) = values
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .unwrap();
                println!("\n  Predicted digit: {predicted} (logit: {confidence:.4})");
                println!("  (Untrained network — prediction is random)");
            }
        }
        Err(e) => {
            eprintln!("Execution failed: {e}");
        }
    }

    // ─── Now compress the weights with IGQK ───
    println!("\n\n=== IGQK Ternary Compression ===\n");

    let mut compress_emitter = qlang_agent::emitter::GraphEmitter::new("compress_weights");

    let w_in = compress_emitter.input("weights", Dtype::F32, Shape::matrix(784, 128));
    let w_compressed = compress_emitter.to_ternary(w_in, TT::f32_matrix(784, 128));
    compress_emitter.output("compressed", w_compressed, TT::ternary_matrix(784, 128));

    let compress_graph = compress_emitter.build();

    let mut compress_inputs = HashMap::new();
    let w1_data: Vec<f32> = (0..784 * 128).map(|i| ((i as f32 * 0.001).sin() * 0.1)).collect();
    compress_inputs.insert(
        "weights".to_string(),
        qlang_core::tensor::TensorData::from_f32(Shape::matrix(784, 128), &w1_data),
    );

    match qlang_runtime::executor::execute(&compress_graph, compress_inputs) {
        Ok(result) => {
            if let Some(compressed) = result.outputs.get("compressed") {
                let original_bytes = 784 * 128 * 4; // f32 = 4 bytes
                let compressed_bytes = 784 * 128 * 1; // ternary = 1 byte (could be 2 bits)

                println!("  Original:   {} bytes ({:.1} KB)", original_bytes, original_bytes as f64 / 1024.0);
                println!("  Compressed: {} bytes ({:.1} KB)", compressed_bytes, compressed_bytes as f64 / 1024.0);
                println!("  Ratio:      {:.1}x compression", original_bytes as f64 / compressed_bytes as f64);
                println!("  (With 2-bit packing: {:.1}x compression)", original_bytes as f64 / (784.0 * 128.0 * 0.25));

                // Count ternary distribution
                let mut counts = [0usize; 3]; // -1, 0, +1
                for &byte in &compressed.data {
                    match byte {
                        0 => counts[1] += 1,   // 0
                        1 => counts[2] += 1,   // +1
                        255 => counts[0] += 1, // -1
                        _ => {}
                    }
                }
                let total = compressed.data.len();
                println!("\n  Ternary distribution:");
                println!("    -1: {} ({:.1}%)", counts[0], counts[0] as f64 / total as f64 * 100.0);
                println!("     0: {} ({:.1}%)", counts[1], counts[1] as f64 / total as f64 * 100.0);
                println!("    +1: {} ({:.1}%)", counts[2], counts[2] as f64 / total as f64 * 100.0);
            }
        }
        Err(e) => {
            eprintln!("Compression failed: {e}");
        }
    }

    // ─── Save graph to JSON ───
    println!("\n=== Saving graph ===");
    let json = qlang_core::serial::to_json(&graph).unwrap();
    let binary = qlang_core::serial::to_binary(&graph).unwrap();
    println!("  JSON size:   {} bytes", json.len());
    println!("  Binary size: {} bytes", binary.len());
    println!("\n=== Done. ===");
}
