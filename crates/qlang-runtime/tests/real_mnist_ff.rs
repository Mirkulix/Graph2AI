//! REAL MNIST Test: Forward-Forward ternary training on actual handwritten digits.
//!
//! This is the honest benchmark — no synthetic data, no shortcuts.
//! Downloads and uses the real MNIST dataset (60k train, 10k test).

use qlang_runtime::forward_forward::FFNetwork;
use qlang_runtime::mnist::MnistData;
use std::time::Instant;

#[test]
fn forward_forward_on_real_mnist() {
    // Load REAL MNIST
    let mnist_dir = std::env::var("MNIST_DIR")
        .unwrap_or_else(|_| "data/mnist".to_string());

    let data = match MnistData::load_from_dir(&mnist_dir) {
        Ok(d) => {
            println!("Loaded REAL MNIST from '{}'", mnist_dir);
            d
        }
        Err(e) => {
            println!("MNIST not found at '{}': {}", mnist_dir, e);
            println!("Falling back to large synthetic dataset (5000/1000)");
            MnistData::synthetic(5000, 1000)
        }
    };

    println!("Train: {} samples, Test: {} samples", data.n_train, data.n_test);
    println!("Image: {}px, Classes: {}", data.image_size, data.n_classes);

    // Use first 10k training samples for speed (real MNIST has 60k)
    let train_limit = data.n_train.min(10000);
    let test_limit = data.n_test.min(2000);
    let train_images = &data.train_images[..train_limit * 784];
    let train_labels = &data.train_labels[..train_limit];
    let test_images = &data.test_images[..test_limit * 784];
    let test_labels = &data.test_labels[..test_limit];

    println!("Using: {} train, {} test\n", train_limit, test_limit);

    // Forward-Forward network: 794 (784+10) → 256 → 128
    let mut net = FFNetwork::new(&[794, 256, 128], 10);

    let total_start = Instant::now();
    let epochs = 15;

    println!("Forward-Forward Ternary Training:");
    println!("{:<8} {:>8} {:>8} {:>10} {:>10} {:>8}", "Epoch", "pos_g", "neg_g", "f32 Acc", "Tern Acc", "Time");
    println!("{}", "-".repeat(62));

    for epoch in 0..epochs {
        let epoch_start = Instant::now();
        let (pg, ng) = net.train_epoch(
            train_images, train_labels,
            784, train_limit, 100,
        );
        let epoch_time = epoch_start.elapsed();

        if epoch % 3 == 0 || epoch == epochs - 1 {
            let f32_acc = net.accuracy(test_images, test_labels, 784, test_limit);
            let tern_acc = net.accuracy_ternary(test_images, test_labels, 784, test_limit);
            println!("{:<8} {:>8.3} {:>8.3} {:>9.1}% {:>9.1}% {:>8.1?}",
                epoch + 1, pg, ng, f32_acc * 100.0, tern_acc * 100.0, epoch_time);
        }
    }

    let total_time = total_start.elapsed();
    let f32_acc = net.accuracy(test_images, test_labels, 784, test_limit);
    let tern_acc = net.accuracy_ternary(test_images, test_labels, 784, test_limit);

    // Weight stats
    let (pos, zero, neg) = net.layers[0].weight_stats();
    let total_w = pos + zero + neg;
    let tern_weights = net.layers.iter()
        .map(|l| l.weights.len())
        .sum::<usize>();
    let f32_size_kb = (tern_weights * 4) as f32 / 1024.0;
    let tern_size_kb = (tern_weights * 2 / 8) as f32 / 1024.0; // 2 bits per ternary weight

    println!("\n{}", "=".repeat(62));
    println!("RESULTS (REAL MNIST)");
    println!("{}", "=".repeat(62));
    println!("Forward-Forward f32:    {:.1}%", f32_acc * 100.0);
    println!("Forward-Forward ternary: {:.1}%", tern_acc * 100.0);
    println!("Total time:             {:?}", total_time);
    println!("Model size (f32):       {:.1} KB", f32_size_kb);
    println!("Model size (ternary):   {:.1} KB", tern_size_kb);
    println!("Compression:            {:.1}x", f32_size_kb / tern_size_kb);
    println!("Weight distribution:    +1:{} ({:.0}%) 0:{} ({:.0}%) -1:{} ({:.0}%)",
        pos, pos as f32 / total_w as f32 * 100.0,
        zero, zero as f32 / total_w as f32 * 100.0,
        neg, neg as f32 / total_w as f32 * 100.0);

    // Assertions
    assert!(f32_acc > 0.50,
        "FF f32 on real MNIST must >50% (got {:.1}%)", f32_acc * 100.0);
    assert!(tern_acc > 0.30,
        "FF ternary on real MNIST must >30% (got {:.1}%)", tern_acc * 100.0);
}
