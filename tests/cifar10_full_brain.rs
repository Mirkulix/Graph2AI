//! Full CIFAR-10: 50K train, 10K test with pre-extracted ResNet-18 features.
//!
//! Loads features from disk (extracted by cifar10_extract_save.rs).
//! TernaryBrain with many neurons, Phase 1 only (Hebbian hurts with good features).
//!
//! Target: >70% with 50K samples and 100 neurons per class.

use qlang_runtime::ternary_brain::TernaryBrain;
use std::time::Instant;

fn load_f32_bin(path: &str) -> Option<Vec<f32>> {
    let bytes = std::fs::read(path).ok()?;
    Some(bytes.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect())
}

#[test]
fn cifar10_full_50k() {
    let train_feat = match load_f32_bin("data/cifar10_resnet_train.bin")
        .or_else(|| load_f32_bin("/home/mirkulix/neoqlang/qlang/data/cifar10_resnet_train.bin"))
    {
        Some(f) => f,
        None => { println!("Pre-extracted features not found. Run cifar10_extract_save first."); return; }
    };
    let test_feat = load_f32_bin("data/cifar10_resnet_test.bin")
        .or_else(|| load_f32_bin("/home/mirkulix/neoqlang/qlang/data/cifar10_resnet_test.bin"))
        .unwrap();
    let train_labels = std::fs::read("data/cifar10_train_labels.bin")
        .or_else(|_| std::fs::read("/home/mirkulix/neoqlang/qlang/data/cifar10_train_labels.bin"))
        .unwrap();
    let test_labels = std::fs::read("data/cifar10_test_labels.bin")
        .or_else(|_| std::fs::read("/home/mirkulix/neoqlang/qlang/data/cifar10_test_labels.bin"))
        .unwrap();

    let feat_dim = 512;
    let n_train = train_feat.len() / feat_dim;
    let n_test = test_feat.len() / feat_dim;

    println!("\n{}", "=".repeat(60));
    println!("CIFAR-10 Full: {} train, {} test, {} features", n_train, n_test, feat_dim);
    println!("{}\n", "=".repeat(60));

    // Test with different neuron counts
    for neurons_per_class in [50, 100, 200] {
        let start = Instant::now();
        let brain = TernaryBrain::init(
            &train_feat, &train_labels,
            feat_dim, n_train, 10, neurons_per_class,
        );
        let acc = brain.accuracy(&test_feat, &test_labels, n_test);
        let total_w = brain.total_weights();
        println!("  {} neurons/class: {:.1}% ({} weights, {:?})",
            neurons_per_class, acc * 100.0, total_w, start.elapsed());
    }

    // Best config: most neurons
    let start = Instant::now();
    let brain = TernaryBrain::init(
        &train_feat, &train_labels,
        feat_dim, n_train, 10, 200,
    );
    let final_acc = brain.accuracy(&test_feat, &test_labels, n_test);
    println!("\n{}", "=".repeat(60));
    println!("BEST: {:.1}% (200 neurons/class, {:?})", final_acc * 100.0, start.elapsed());
    println!("All ternary: {}", brain.verify_ternary());
    println!("{}", "=".repeat(60));

    assert!(final_acc > 0.50,
        "Full CIFAR-10 must beat 50% (got {:.1}%)", final_acc * 100.0);
}
