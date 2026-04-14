//! QLANG Full MNIST Pipeline — Download, Train, Compress, Export.
//!
//! End-to-end demonstration of the entire QLANG pipeline:
//!   1. Download MNIST (or fall back to synthetic data)
//!   2. Load data
//!   3. Build 784->256->128->10 MLP (3 layers)
//!   4. Train with backpropagation (mini-batch, lr decay)
//!   5. IGQK ternary compression
//!   6. Export as ONNX JSON
//!   7. Export as .qlang text
//!   8. Print summary
//!
//! Run:
//!   cargo run --release --no-default-features --example full_mnist_pipeline

fn main() {
    println!("================================================================");
    println!("  QLANG Full MNIST Pipeline");
    println!("  Download -> Train -> Compress -> ONNX Export");
    println!("================================================================\n");

    use qlang_runtime::mnist::MnistData;
    use qlang_runtime::training::MlpWeights3;
    use std::time::Instant;

    let total_start = Instant::now();

    // ================================================================
    // 1. Download / Load MNIST
    // ================================================================
    println!("--- [1/8] DOWNLOAD MNIST ---");
    let mnist_dir = std::env::var("MNIST_DIR").unwrap_or_else(|_| "data/mnist".to_string());

    let data = match MnistData::download_and_load(&mnist_dir) {
        Ok(d) => {
            println!("  Loaded REAL MNIST from '{}'", mnist_dir);
            d
        }
        Err(e) => {
            println!("  Could not load real MNIST: {}", e);
            println!("  Falling back to synthetic MNIST data.");
            MnistData::synthetic(2000, 500)
        }
    };

    // ================================================================
    // 2. Data summary
    // ================================================================
    println!("\n--- [2/8] DATA LOADED ---");
    println!("  Train samples: {}", data.n_train);
    println!("  Test samples:  {}", data.n_test);
    println!("  Image size:    {}px (28x28)", data.image_size);
    println!("  Classes:       {}", data.n_classes);

    // ================================================================
    // 3. Build model: 784 -> 256 -> 128 -> 10
    // ================================================================
    println!("\n--- [3/8] BUILD MODEL ---");
    let input_dim = 784;
    let hidden1_dim = 256;
    let hidden2_dim = 128;
    let output_dim = 10;

    let mut model = MlpWeights3::new(input_dim, hidden1_dim, hidden2_dim, output_dim);
    let total_params = model.param_count();

    println!("  Architecture: {}->{}->{}->{}  (3-layer MLP)", input_dim, hidden1_dim, hidden2_dim, output_dim);
    println!("  Parameters:   {} ({:.1} KB as f32)", total_params, total_params as f64 * 4.0 / 1024.0);

    // ================================================================
    // 4. Train with backpropagation
    // ================================================================
    println!("\n--- [4/8] TRAIN ---");
    let epochs = 50;
    let batch_size = 64;
    let mut lr = 0.1f32;
    let lr_decay = 0.95f32;
    let lr_decay_every = 10;

    println!("  Epochs:        {}", epochs);
    println!("  Batch size:    {}", batch_size);
    println!("  Learning rate: {} (decay {}x every {} epochs)", lr, lr_decay, lr_decay_every);
    println!();

    let train_start = Instant::now();

    for epoch in 0..epochs {
        // Learning rate decay
        if epoch > 0 && epoch % lr_decay_every == 0 {
            lr *= lr_decay;
        }

        let n_batches = data.n_train / batch_size;
        let mut epoch_loss = 0.0f32;

        for batch_idx in 0..n_batches {
            let (x, y) = data.train_batch(batch_idx * batch_size, batch_size);
            let loss = model.train_step_backprop(x, y, lr);
            epoch_loss += loss;
        }
        epoch_loss /= n_batches.max(1) as f32;

        // Print every 5 epochs
        if epoch % 5 == 0 || epoch == epochs - 1 {
            let train_probs = model.forward(&data.train_images);
            let train_acc = model.accuracy(&train_probs, &data.train_labels);
            println!("  Epoch {:>3}/{}: loss={:.4}  train_acc={:.1}%  lr={:.4}",
                epoch + 1, epochs, epoch_loss, train_acc * 100.0, lr);
        }
    }

    let train_time = train_start.elapsed();

    // Test accuracy
    let test_probs = model.forward(&data.test_images);
    let test_acc = model.accuracy(&test_probs, &data.test_labels);
    let test_loss = model.loss(&test_probs, &data.test_labels);
    println!();
    println!("  Training time:  {:?}", train_time);
    println!("  Test accuracy:  {:.1}%", test_acc * 100.0);
    println!("  Test loss:      {:.4}", test_loss);

    // ================================================================
    // 5. IGQK Ternary Compression
    // ================================================================
    println!("\n--- [5/8] IGQK TERNARY COMPRESSION ---");
    let compressed = model.compress_ternary();

    let comp_probs = compressed.forward(&data.test_images);
    let comp_acc = compressed.accuracy(&comp_probs, &data.test_labels);
    let comp_loss = compressed.loss(&comp_probs, &data.test_labels);

    // Compression ratio: original uses 32-bit floats for weights,
    // ternary only needs 2 bits per weight (values: -1, 0, +1).
    let weight_count = model.w1.len() + model.w2.len() + model.w3.len();
    let original_bytes = total_params * 4; // all params as f32
    let ternary_weight_bytes = (weight_count * 2 + 7) / 8; // 2 bits per weight
    let bias_bytes = (model.b1.len() + model.b2.len() + model.b3.len()) * 4;
    let compressed_bytes = ternary_weight_bytes + bias_bytes;
    let compression_ratio = original_bytes as f64 / compressed_bytes as f64;

    println!("  Before: {:.1}% accuracy, {:.1} KB", test_acc * 100.0, original_bytes as f64 / 1024.0);
    println!("  After:  {:.1}% accuracy, {:.1} KB", comp_acc * 100.0, compressed_bytes as f64 / 1024.0);
    println!("  Accuracy drop: {:.1}%", (test_acc - comp_acc) * 100.0);
    println!("  Compression ratio: {:.1}x", compression_ratio);

    // ================================================================
    // 6. Export as ONNX JSON
    // ================================================================
    println!("\n--- [6/8] ONNX JSON EXPORT ---");

    // Build a QLANG graph representing the model
    let qlang_source = format!(r#"graph mnist_3layer {{
  input x: f32[1, 784]
  input W1: f32[784, {hidden1_dim}]
  input b1: f32[1, {hidden1_dim}]
  input W2: f32[{hidden1_dim}, {hidden2_dim}]
  input b2: f32[1, {hidden2_dim}]
  input W3: f32[{hidden2_dim}, 10]
  input b3: f32[1, 10]

  node h1 = matmul(x, W1)
  node a1 = relu(h1)
  node h2 = matmul(a1, W2)
  node a2 = relu(h2)
  node logits = matmul(a2, W3)
  node probs = softmax(logits)
  node comp = to_ternary(W1) @proof theorem_5_2

  output predictions = probs
  output compressed = comp
}}"#);

    match qlang_compile::parser::parse(&qlang_source) {
        Ok(graph) => {
            let onnx_json = qlang_compile::onnx::to_onnx_json(&graph);
            println!("  ONNX JSON size: {} bytes", onnx_json.len());

            // Save to file
            let onnx_path = "/tmp/qlang_mnist_3layer.onnx.json";
            match std::fs::write(onnx_path, &onnx_json) {
                Ok(_) => println!("  Saved to: {}", onnx_path),
                Err(e) => println!("  Could not save: {}", e),
            }

            // Print first few lines
            println!("  Preview:");
            for line in onnx_json.lines().take(12) {
                println!("    {}", line);
            }
            println!("    ...");

            // ================================================================
            // 7. Export as .qlang text
            // ================================================================
            println!("\n--- [7/8] .QLANG TEXT EXPORT ---");
            let qlang_text = qlang_compile::parser::to_qlang_text(&graph);
            println!("  .qlang text size: {} bytes", qlang_text.len());
            println!("  Graph definition:");
            for line in qlang_text.lines() {
                println!("    {}", line);
            }
        }
        Err(e) => {
            println!("  Parse error (non-fatal): {}", e);
            println!("\n--- [7/8] .QLANG TEXT EXPORT ---");
            println!("  Printing source directly:");
            for line in qlang_source.lines() {
                println!("    {}", line);
            }
        }
    }

    // ================================================================
    // 8. Summary
    // ================================================================
    let total_time = total_start.elapsed();
    println!("\n================================================================");
    println!("  PIPELINE SUMMARY");
    println!("================================================================");
    println!("  Total time:         {:?}", total_time);
    println!("  Training time:      {:?}", train_time);
    println!("  Architecture:       {}->{}->{}->{}  (3-layer MLP)", input_dim, hidden1_dim, hidden2_dim, output_dim);
    println!("  Parameters:         {} ({:.1} KB)", total_params, total_params as f64 * 4.0 / 1024.0);
    println!("  Training:           {} epochs, batch_size={}", epochs, batch_size);
    println!("  Test accuracy:      {:.1}%", test_acc * 100.0);
    println!("  Compressed acc:     {:.1}%", comp_acc * 100.0);
    println!("  Compression ratio:  {:.1}x", compression_ratio);
    println!("  Exports:            ONNX JSON + .qlang text");
    println!("================================================================");
    println!("  No Python. No PyTorch. Pure Rust + QLANG.");
    println!("================================================================");

    // Cleanup
    let _ = std::fs::remove_file("/tmp/qlang_mnist_3layer.onnx.json");
}
