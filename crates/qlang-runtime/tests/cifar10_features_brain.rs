//! CIFAR-10 with Feature Extraction + TernaryBrain.
//!
//! Pipeline: Raw pixels → Feature extraction (parallel) → TernaryBrain (parallel)
//! All ternary, all parallel, all Rust.

use qlang_runtime::cifar10::Cifar10Data;
use qlang_runtime::cifar10_features;
use qlang_runtime::ternary_brain::TernaryBrain;
use std::time::Instant;

#[test]
fn cifar10_with_features() {
    let paths = [
        "data/cifar10",
        "../data/cifar10",
        "/home/mirkulix/neoqlang/qlang/data/cifar10",
    ];

    let mut data = None;
    for p in &paths {
        if let Ok(d) = Cifar10Data::load(p) { data = Some(d); break; }
    }
    let data = match data {
        Some(d) => d,
        None => { println!("CIFAR-10 not found, skipping"); return; }
    };

    let train_limit = data.n_train.min(20000);
    let test_limit = data.n_test.min(5000);

    println!("\n{}", "=".repeat(60));
    println!("CIFAR-10 + Features + TernaryBrain");
    println!("Train: {}, Test: {}", train_limit, test_limit);
    println!("{}\n", "=".repeat(60));

    // Step 1: Extract features (parallel)
    println!("=== Step 1: Feature Extraction (parallel) ===");
    let start = Instant::now();
    let train_features = cifar10_features::extract_batch(
        &data.train_images[..train_limit * 3072], train_limit
    );
    let test_features = cifar10_features::extract_batch(
        &data.test_images[..test_limit * 3072], test_limit
    );
    let feat_time = start.elapsed();
    let feat_dim = cifar10_features::feature_dim();
    println!("  {} dims → {} features in {:?}", 3072, feat_dim, feat_time);

    let train_labels = &data.train_labels[..train_limit];
    let test_labels = &data.test_labels[..test_limit];

    // Step 2: TernaryBrain Phase 1 (parallel)
    println!("\n=== Step 2: TernaryBrain Phase 1 (Statistical Init) ===");
    let start = Instant::now();
    let mut brain = TernaryBrain::init(
        &train_features, train_labels,
        feat_dim,
        train_limit,
        10,
        50, // 50 neurons per class for richer representation
    );
    let phase1_time = start.elapsed();
    let phase1_acc = brain.accuracy(&test_features, test_labels, test_limit);
    println!("  Accuracy: {:.1}% in {:?}", phase1_acc * 100.0, phase1_time);
    println!("  Weights: {}, all ternary: {}", brain.total_weights(), brain.verify_ternary());

    // Step 3: TernaryBrain Phase 2 (Competitive Hebbian)
    println!("\n=== Step 3: Competitive Hebbian Refinement ===");
    let total_start = Instant::now();
    for round in 0..15 {
        brain.refine(&train_features, train_labels, train_limit, 1);
        if round % 3 == 0 || round == 14 {
            let acc = brain.accuracy(&test_features, test_labels, test_limit);
            println!("  Round {:>2}: {:.1}% ({:.1?})", round + 1, acc * 100.0, total_start.elapsed());
        }
    }

    let final_acc = brain.accuracy(&test_features, test_labels, test_limit);
    let total_time = total_start.elapsed();

    println!("\n{}", "=".repeat(60));
    println!("RESULT: CIFAR-10 + Features + TernaryBrain");
    println!("{}", "=".repeat(60));
    println!("  Feature extraction: {:?}", feat_time);
    println!("  Phase 1 (init):     {:.1}%", phase1_acc * 100.0);
    println!("  Final accuracy:     {:.1}%", final_acc * 100.0);
    println!("  Total weights:      {}", brain.total_weights());
    println!("  All ternary:        {}", brain.verify_ternary());
    println!("  Total time:         {:?}", total_time);
    println!("  Random chance:      10.0%");
    println!();

    assert!(final_acc > 0.20,
        "CIFAR-10 with features must beat 20% (got {:.1}%)", final_acc * 100.0);

    if final_acc > 0.40 {
        println!("  >>> 40%+ : Feature extraction helps significantly!");
    }
    if final_acc > 0.50 {
        println!("  >>> 50%+ : Genuinely useful ternary classifier!");
    }
}
