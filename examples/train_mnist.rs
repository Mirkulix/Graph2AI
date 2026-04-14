//! Train on MNIST (synthetic) with autograd + IGQK compression + GPU shader export.
//!
//! Full QLANG pipeline:
//! 1. Load data (synthetic MNIST-like, 10 digit classes)
//! 2. Train 784→256→10 MLP with autograd (backpropagation)
//! 3. Evaluate accuracy
//! 4. IGQK ternary compression
//! 5. Export as .qlang text, GPU shader, and LLVM IR

fn main() {
    println!("=== QLANG MNIST Training Pipeline ===\n");

    use qlang_runtime::autograd::train_mlp_autograd;
    use qlang_runtime::mnist::MnistData;
    use qlang_runtime::training::MlpWeights;
    use std::time::Instant;

    // ─── 1. Load data ───
    // Try real MNIST first (from ./data/mnist/), fall back to synthetic
    let mnist_dir = std::env::var("MNIST_DIR").unwrap_or_else(|_| "data/mnist".to_string());
    let data = match MnistData::load_from_dir(&mnist_dir) {
        Ok(d) => {
            println!("[1] Loaded REAL MNIST from '{}'", mnist_dir);
            d
        }
        Err(_) => {
            println!("[1] Real MNIST not found in '{}', using synthetic data.", mnist_dir);
            println!("    To use real MNIST, download IDX files and set MNIST_DIR env var.");
            MnistData::synthetic(2000, 400)
        }
    };
    println!("  Train: {} samples, Test: {} samples", data.n_train, data.n_test);
    println!("  Image: {}px (28×28), Classes: {}", data.image_size, data.n_classes);

    // ─── 2. Initialize model ───
    let input_dim = 784;
    let hidden_dim = 128;
    let output_dim = 10;
    let epochs = 30;
    let batch_size = 50;
    let lr = 0.02;

    println!("\n[2] Model: {}→{}→{} MLP", input_dim, hidden_dim, output_dim);

    let scale1 = (2.0 / (input_dim + hidden_dim) as f64).sqrt() as f32;
    let scale2 = (2.0 / (hidden_dim + output_dim) as f64).sqrt() as f32;
    let mut w1: Vec<f32> = (0..input_dim * hidden_dim).map(|i| (i as f32 * 0.371).sin() * scale1).collect();
    let mut b1 = vec![0.0f32; hidden_dim];
    let mut w2: Vec<f32> = (0..hidden_dim * output_dim).map(|i| (i as f32 * 0.529).sin() * scale2).collect();
    let mut b2 = vec![0.0f32; output_dim];

    let params = w1.len() + b1.len() + w2.len() + b2.len();
    println!("  Parameters: {} ({:.1} KB as f32)", params, params as f64 * 4.0 / 1024.0);

    // ─── 3. Train ───
    println!("\n[3] Training ({} epochs, lr={}, batch={})...", epochs, lr, batch_size);
    let start = Instant::now();

    for epoch in 0..epochs {
        let n_batches = data.n_train / batch_size;
        let mut epoch_loss = 0.0f32;
        let mut epoch_acc = 0.0f32;

        for batch in 0..n_batches {
            let (x, y) = data.train_batch(batch * batch_size, batch_size);

            let (loss, acc) = train_mlp_autograd(
                &mut w1, &mut b1, &mut w2, &mut b2,
                x, y,
                input_dim, hidden_dim, output_dim,
                lr,
            );
            epoch_loss += loss;
            epoch_acc += acc;
        }

        epoch_loss /= n_batches as f32;
        epoch_acc /= n_batches as f32;

        if epoch % 5 == 0 || epoch == epochs - 1 {
            // Test
            let (test_loss, test_acc) = train_mlp_autograd(
                &mut w1.clone(), &mut b1.clone(), &mut w2.clone(), &mut b2.clone(),
                &data.test_images, &data.test_labels,
                input_dim, hidden_dim, output_dim,
                0.0,
            );
            println!("  Epoch {:>2}/{}: loss={:.4} acc={:.1}%  |  test: loss={:.4} acc={:.1}%",
                epoch + 1, epochs, epoch_loss, epoch_acc * 100.0, test_loss, test_acc * 100.0);
        }
    }

    let train_time = start.elapsed();
    println!("  Training time: {:?}", train_time);

    // ─── 4. IGQK Compression ───
    println!("\n[4] IGQK Ternary Compression...");
    let mlp = MlpWeights { w1: w1.clone(), b1: b1.clone(), w2: w2.clone(), b2: b2.clone(),
        input_dim, hidden_dim, output_dim };
    let compressed = mlp.compress_ternary();

    let comp_probs = compressed.forward(&data.test_images);
    let comp_acc = compressed.accuracy(&comp_probs, &data.test_labels);

    let (test_loss, test_acc) = train_mlp_autograd(
        &mut w1.clone(), &mut b1.clone(), &mut w2.clone(), &mut b2.clone(),
        &data.test_images, &data.test_labels,
        input_dim, hidden_dim, output_dim, 0.0,
    );

    println!("  Original:   {:.1}% accuracy, {} KB", test_acc * 100.0, params * 4 / 1024);
    println!("  Compressed: {:.1}% accuracy, {} KB", comp_acc * 100.0,
        (w1.len() + w2.len()) / 1024 + (b1.len() + b2.len()) * 4 / 1024);
    println!("  Compression ratio: {:.1}x", params as f64 * 4.0 / ((w1.len() + w2.len()) as f64 + (b1.len() + b2.len()) as f64 * 4.0));

    // ─── 5. Export QLANG text ───
    println!("\n[5] QLANG exports:");

    let qlang_text = format!(r#"graph mnist_mlp {{
  input x: f32[1, 784]
  input W1: f32[784, {hidden_dim}]
  input W2: f32[{hidden_dim}, 10]

  node h = matmul(x, W1)
  node a = relu(h)
  node logits = matmul(a, W2)
  node probs = softmax(logits)
  node comp = to_ternary(W1) @proof theorem_5_2

  output predictions = probs
  output compressed = comp
}}"#);

    println!("  .qlang text ({} bytes):", qlang_text.len());
    println!("{}", qlang_text.lines().map(|l| format!("    {l}")).collect::<Vec<_>>().join("\n"));

    // Parse to verify
    match qlang_compile::parser::parse(&qlang_text) {
        Ok(graph) => {
            println!("\n  Parsed: {} nodes, {} edges ✓", graph.nodes.len(), graph.edges.len());

            // Export GPU shader
            let wgsl = qlang_compile::gpu::to_wgsl(&graph);
            println!("\n  WGSL GPU shader ({} bytes):", wgsl.len());
            for line in wgsl.lines().take(10) {
                println!("    {line}");
            }
            println!("    ...");

            // Export matmul GPU shader
            let matmul_wgsl = qlang_compile::gpu::matmul_wgsl(1, 784, hidden_dim);
            println!("\n  MatMul GPU shader: {} bytes (tiled, 16×16 workgroups)", matmul_wgsl.len());

            // Export visualization
            println!("\n  ASCII visualization:");
            println!("{}", qlang_compile::visualize::to_ascii(&graph));

            // Binary size
            if let Ok(binary) = qlang_core::serial::to_binary(&graph) {
                println!("  Binary .qlg size: {} bytes", binary.len());
            }
        }
        Err(e) => println!("  Parse error: {e}"),
    }

    println!("=== Complete: Trained, compressed, and exported as QLANG. ===");
}
