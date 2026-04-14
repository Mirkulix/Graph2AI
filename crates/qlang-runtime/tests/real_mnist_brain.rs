//! Real MNIST: Ternary Brain — 100% ternary, zero f32 weights.

use qlang_runtime::ternary_brain::TernaryBrain;
use qlang_runtime::mnist::MnistData;
use std::time::Instant;

#[test]
fn ternary_brain_real_mnist() {
    let mnist_dir = std::env::var("MNIST_DIR")
        .unwrap_or_else(|_| "data/mnist".to_string());

    let data = match MnistData::load_from_dir(&mnist_dir) {
        Ok(d) => { println!("REAL MNIST from '{}'", mnist_dir); d }
        Err(_) => { println!("Synthetic fallback (5000/1000)"); MnistData::synthetic(5000, 1000) }
    };

    let train_n = data.n_train.min(60000);
    let test_n = data.n_test.min(10000);

    println!("Train: {}, Test: {}\n", train_n, test_n);

    let total = Instant::now();

    // Phase 1: Statistical init with 100 neurons per class (subset diversity)
    let mut brain = TernaryBrain::init(
        &data.train_images[..train_n * 784],
        &data.train_labels[..train_n],
        784, train_n, 10, 100,
    );
    let acc1 = brain.accuracy(
        &data.test_images[..test_n * 784],
        &data.test_labels[..test_n], test_n,
    );
    println!("Phase 1: {:.1}% ({:?})", acc1 * 100.0, total.elapsed());

    // Phase 2: Competitive refinement
    for r in 0..10 {
        brain.refine(
            &data.train_images[..train_n * 784],
            &data.train_labels[..train_n],
            train_n, 1,
        );
        if r % 3 == 0 || r == 9 {
            let acc = brain.accuracy(
                &data.test_images[..test_n * 784],
                &data.test_labels[..test_n], test_n,
            );
            println!("Phase 2 round {:>2}: {:.1}% ({:?})", r + 1, acc * 100.0, total.elapsed());
        }
    }

    let final_acc = brain.accuracy(
        &data.test_images[..test_n * 784],
        &data.test_labels[..test_n], test_n,
    );

    println!("\nRESULT: {:.1}%", final_acc * 100.0);
    println!("Time:   {:?}", total.elapsed());
    println!("Weights: {} (100% ternary i8)", brain.total_weights());
    println!("Verified: {}", brain.verify_ternary());

    assert!(brain.verify_ternary());
    assert!(final_acc > 0.70, "Must >70% on real MNIST (got {:.1}%)", final_acc * 100.0);
}
