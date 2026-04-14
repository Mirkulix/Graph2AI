//! Train a neural network entirely in QLANG.
//!
//! No Python. No PyTorch. Pure Rust + QLANG.
//!
//! Architecture: 64-input → 32-hidden → 4-output
//! Dataset: Toy pattern recognition (lines, diagonals, crosses)
//! Training: Numerical gradient descent
//! Compression: IGQK ternary (post-training)

fn main() {
    println!("=== QLANG Neural Network Training ===\n");

    use qlang_runtime::training::{MlpWeights, TrainConfig, generate_toy_dataset};
    use std::time::Instant;

    let input_dim = 64; // 8x8 images
    let hidden_dim = 32;
    let output_dim = 4;  // 4 pattern classes

    // ─── Generate dataset ───
    println!("[1] Generating toy dataset (8×8 patterns)...");
    let n_train = 200;
    let n_test = 40;
    let (train_images, train_labels) = generate_toy_dataset(n_train, input_dim);
    let (test_images, test_labels) = generate_toy_dataset(n_test, input_dim);

    println!("  Training: {} samples", n_train);
    println!("  Test:     {} samples", n_test);
    println!("  Classes:  horizontal, vertical, diagonal, cross\n");

    // ─── Initialize model ───
    println!("[2] Initializing MLP ({} → {} → {})...", input_dim, hidden_dim, output_dim);
    let mut mlp = MlpWeights::new(input_dim, hidden_dim, output_dim);
    println!("  Parameters: {}", mlp.param_count());

    // Initial accuracy
    let probs = mlp.forward(&test_images);
    let initial_acc = mlp.accuracy(&probs, &test_labels);
    let initial_loss = mlp.loss(&probs, &test_labels);
    println!("  Initial loss:     {:.4}", initial_loss);
    println!("  Initial accuracy: {:.1}% (random=25%)\n", initial_acc * 100.0);

    // ─── Train ───
    println!("[3] Training with gradient descent...");
    let config = TrainConfig {
        learning_rate: 0.05,
        epochs: 5,
        batch_size: 8,
        log_interval: 1,
    };

    let start = Instant::now();

    for epoch in 0..config.epochs {
        let mut epoch_loss = 0.0f32;
        let n_batches = n_train / config.batch_size;

        for batch in 0..n_batches {
            let offset = batch * config.batch_size;
            let x = &train_images[offset * input_dim..(offset + config.batch_size) * input_dim];
            let y = &train_labels[offset..offset + config.batch_size];

            let loss = mlp.train_step(x, y, config.learning_rate);
            epoch_loss += loss;
        }

        let avg_loss = epoch_loss / n_batches as f32;

        // Test accuracy
        let test_probs = mlp.forward(&test_images);
        let test_acc = mlp.accuracy(&test_probs, &test_labels);
        let test_loss = mlp.loss(&test_probs, &test_labels);

        println!("  Epoch {}/{}: train_loss={:.4}, test_loss={:.4}, test_acc={:.1}%",
            epoch + 1, config.epochs, avg_loss, test_loss, test_acc * 100.0);
    }

    let train_time = start.elapsed();
    println!("\n  Training time: {:?}", train_time);

    // ─── Final evaluation ───
    println!("\n[4] Final evaluation...");
    let final_probs = mlp.forward(&test_images);
    let final_acc = mlp.accuracy(&final_probs, &test_labels);
    let final_loss = mlp.loss(&final_probs, &test_labels);
    println!("  Final loss:     {:.4} (was {:.4})", final_loss, initial_loss);
    println!("  Final accuracy: {:.1}% (was {:.1}%)", final_acc * 100.0, initial_acc * 100.0);

    // Show some predictions
    println!("\n  Sample predictions:");
    let class_names = ["horizontal", "vertical", "diagonal", "cross"];
    for i in 0..8.min(n_test) {
        let offset = i * output_dim;
        let predicted = final_probs[offset..offset + output_dim]
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(idx, _)| idx)
            .unwrap();
        let label = test_labels[i] as usize;
        let marker = if predicted == label { "✓" } else { "✗" };
        println!("    [{marker}] Sample {i}: predicted={} ({}), actual={} ({})",
            predicted, class_names[predicted],
            label, class_names[label]);
    }

    // ─── IGQK Compression ───
    println!("\n[5] IGQK Ternary Compression...");
    let compressed = mlp.compress_ternary();

    // Test compressed model
    let compressed_probs = compressed.forward(&test_images);
    let compressed_acc = compressed.accuracy(&compressed_probs, &test_labels);
    let compressed_loss = compressed.loss(&compressed_probs, &test_labels);

    println!("  Compressed loss:     {:.4} (was {:.4})", compressed_loss, final_loss);
    println!("  Compressed accuracy: {:.1}% (was {:.1}%)", compressed_acc * 100.0, final_acc * 100.0);

    let orig_bytes = mlp.param_count() * 4; // f32
    let compressed_bytes = (mlp.w1.len() + mlp.w2.len()) * 1 + (mlp.b1.len() + mlp.b2.len()) * 4;
    println!("  Original size:   {} bytes", orig_bytes);
    println!("  Compressed size: {} bytes", compressed_bytes);
    println!("  Compression:     {:.1}x", orig_bytes as f64 / compressed_bytes as f64);

    // Weight distribution
    let w1_pos = compressed.w1.iter().filter(|&&w| w == 1.0).count();
    let w1_neg = compressed.w1.iter().filter(|&&w| w == -1.0).count();
    let w1_zero = compressed.w1.iter().filter(|&&w| w == 0.0).count();
    println!("\n  W1 ternary distribution:");
    println!("    +1: {} ({:.1}%)", w1_pos, w1_pos as f64 / compressed.w1.len() as f64 * 100.0);
    println!("     0: {} ({:.1}%)", w1_zero, w1_zero as f64 / compressed.w1.len() as f64 * 100.0);
    println!("    -1: {} ({:.1}%)", w1_neg, w1_neg as f64 / compressed.w1.len() as f64 * 100.0);

    // ─── Emit QLANG text ───
    println!("\n[6] QLANG text representation of the model:");
    let qlang_text = format!(r#"graph trained_mlp {{
  input x: f32[1, {input_dim}]
  input W1: f32[{input_dim}, {hidden_dim}]
  input W2: f32[{hidden_dim}, {output_dim}]

  node h = matmul(x, W1)
  node a = relu(h)
  node logits = matmul(a, W2)
  node probs = softmax(logits)

  // IGQK compression
  node compressed_W1 = to_ternary(W1) @proof theorem_5_2
  node compressed_W2 = to_ternary(W2) @proof theorem_5_2

  output predictions = probs
  output ternary_W1 = compressed_W1
  output ternary_W2 = compressed_W2
}}"#);

    println!("{qlang_text}");

    // Parse it back to verify
    match qlang_compile::parser::parse(&qlang_text) {
        Ok(graph) => {
            println!("\n  Parsed successfully: {} nodes, {} edges", graph.nodes.len(), graph.edges.len());
            let verify = qlang_core::verify::verify_graph(&graph);
            println!("  Verification: {} passed, {} failed", verify.passed.len(), verify.failed.len());
        }
        Err(e) => println!("\n  Parse error: {e}"),
    }

    println!("\n=== Training complete. Model trained, compressed, and verified in QLANG. ===");
}
