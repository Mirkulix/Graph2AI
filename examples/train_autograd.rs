//! Train a neural network with automatic differentiation.
//!
//! Unlike train_model.rs (numerical gradients), this uses proper
//! reverse-mode AD (backpropagation) — the same algorithm as PyTorch.
//!
//! This is 100x+ faster and actually converges on the toy dataset.

fn main() {
    println!("=== QLANG Autograd Training ===\n");

    use qlang_runtime::autograd::train_mlp_autograd;
    use qlang_runtime::training::{generate_toy_dataset, MlpWeights};
    use std::time::Instant;

    let input_dim = 64;
    let hidden_dim = 32;
    let output_dim = 4;
    let n_train = 200;
    let n_test = 80;
    let epochs = 50;
    let lr = 0.05;
    let batch_size = 20;

    // Generate data
    println!("[1] Dataset: 8×8 patterns, {} train / {} test", n_train, n_test);
    let (train_images, train_labels) = generate_toy_dataset(n_train, input_dim);
    let (test_images, test_labels) = generate_toy_dataset(n_test, input_dim);

    // Initialize weights (Xavier)
    let scale1 = (2.0 / (input_dim + hidden_dim) as f64).sqrt() as f32;
    let scale2 = (2.0 / (hidden_dim + output_dim) as f64).sqrt() as f32;
    let mut w1: Vec<f32> = (0..input_dim * hidden_dim).map(|i| (i as f32 * 0.4871).sin() * scale1).collect();
    let mut b1 = vec![0.0f32; hidden_dim];
    let mut w2: Vec<f32> = (0..hidden_dim * output_dim).map(|i| (i as f32 * 0.7291).sin() * scale2).collect();
    let mut b2 = vec![0.0f32; output_dim];

    let params = w1.len() + b1.len() + w2.len() + b2.len();
    println!("[2] MLP: {} → {} → {} ({} params)", input_dim, hidden_dim, output_dim, params);

    // Initial eval
    let (init_loss, init_acc) = train_mlp_autograd(
        &mut w1.clone(), &mut b1.clone(), &mut w2.clone(), &mut b2.clone(),
        &test_images, &test_labels,
        input_dim, hidden_dim, output_dim, 0.0, // lr=0 = no update
    );
    println!("[3] Initial: loss={:.4}, accuracy={:.1}%\n", init_loss, init_acc * 100.0);

    // Train
    println!("[4] Training ({} epochs, lr={}, batch={})...", epochs, lr, batch_size);
    let start = Instant::now();

    for epoch in 0..epochs {
        let n_batches = n_train / batch_size;
        let mut epoch_loss = 0.0f32;
        let mut epoch_acc = 0.0f32;

        for batch in 0..n_batches {
            let offset = batch * batch_size;
            let x = &train_images[offset * input_dim..(offset + batch_size) * input_dim];
            let y = &train_labels[offset..offset + batch_size];

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

        if epoch % 10 == 0 || epoch == epochs - 1 {
            // Test eval
            let (test_loss, test_acc) = train_mlp_autograd(
                &mut w1.clone(), &mut b1.clone(), &mut w2.clone(), &mut b2.clone(),
                &test_images, &test_labels,
                input_dim, hidden_dim, output_dim, 0.0,
            );
            println!("  Epoch {:>3}/{}: train_loss={:.4} train_acc={:.1}%  |  test_loss={:.4} test_acc={:.1}%",
                epoch + 1, epochs, epoch_loss, epoch_acc * 100.0, test_loss, test_acc * 100.0);
        }
    }

    let train_time = start.elapsed();
    println!("\n  Training time: {:?}", train_time);

    // Final eval
    let (final_loss, final_acc) = train_mlp_autograd(
        &mut w1.clone(), &mut b1.clone(), &mut w2.clone(), &mut b2.clone(),
        &test_images, &test_labels,
        input_dim, hidden_dim, output_dim, 0.0,
    );

    println!("\n[5] Results:");
    println!("  Loss:     {:.4} → {:.4}", init_loss, final_loss);
    println!("  Accuracy: {:.1}% → {:.1}%", init_acc * 100.0, final_acc * 100.0);

    // IGQK Compression
    println!("\n[6] IGQK Ternary Compression...");
    let compressed_mlp = MlpWeights {
        w1: w1.clone(), b1: b1.clone(),
        w2: w2.clone(), b2: b2.clone(),
        input_dim, hidden_dim, output_dim,
    };
    let compressed = compressed_mlp.compress_ternary();

    let test_probs = compressed.forward(&test_images);
    let compressed_acc = compressed.accuracy(&test_probs, &test_labels);
    let orig_bytes = params * 4;
    let comp_bytes = (w1.len() + w2.len()) + (b1.len() + b2.len()) * 4;

    println!("  Compressed accuracy: {:.1}% (was {:.1}%)", compressed_acc * 100.0, final_acc * 100.0);
    println!("  Size: {} → {} bytes ({:.1}x compression)", orig_bytes, comp_bytes, orig_bytes as f64 / comp_bytes as f64);

    // Ternary distribution
    let ternary_w1 = compressed.w1.clone();
    let pos = ternary_w1.iter().filter(|&&w| w == 1.0).count();
    let neg = ternary_w1.iter().filter(|&&w| w == -1.0).count();
    let zero = ternary_w1.iter().filter(|&&w| w == 0.0).count();
    println!("  W1: +1={pos} ({:.0}%), 0={zero} ({:.0}%), -1={neg} ({:.0}%)",
        pos as f64 / ternary_w1.len() as f64 * 100.0,
        zero as f64 / ternary_w1.len() as f64 * 100.0,
        neg as f64 / ternary_w1.len() as f64 * 100.0);

    println!("\n=== Done. Neural network trained with autograd in pure QLANG/Rust. ===");
}
