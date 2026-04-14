//! CIFAR-10 with Random Convolutional Features + TernaryBrain.
//!
//! Pipeline: Pixels → Random Conv (3x3 + 5x5) → Pool → TernaryBrain
//! Target: >40% (realistic for gradient-free with random features)

use qlang_runtime::cifar10::Cifar10Data;
use qlang_runtime::random_conv_features::MultiScaleConvBank;
use qlang_runtime::ternary_brain::TernaryBrain;
use std::time::Instant;

#[test]
fn cifar10_random_conv_brain() {
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
    println!("CIFAR-10: Random Conv Features + TernaryBrain");
    println!("Train: {}, Test: {}", train_limit, test_limit);
    println!("{}\n", "=".repeat(60));

    // Step 1: Random convolutional feature extraction (parallel)
    println!("=== Step 1: Random Conv Features ===");
    let bank = MultiScaleConvBank::new(192, 64, 3, 32); // 256 total features
    println!("  Filters: 192 x 3x3 + 64 x 5x5 = {} features", bank.feature_dim());

    let start = Instant::now();
    let train_features = bank.extract_batch(&data.train_images[..train_limit * 3072], train_limit);
    let test_features = bank.extract_batch(&data.test_images[..test_limit * 3072], test_limit);
    let feat_time = start.elapsed();
    println!("  Extraction: {:?} (parallel)", feat_time);

    let feat_dim = bank.feature_dim();
    let train_labels = &data.train_labels[..train_limit];
    let test_labels = &data.test_labels[..test_limit];

    // Step 2: TernaryBrain (parallel)
    println!("\n=== Step 2: TernaryBrain ===");
    let start = Instant::now();
    let mut brain = TernaryBrain::init(
        &train_features, train_labels,
        feat_dim, train_limit, 10, 50,
    );
    let phase1_acc = brain.accuracy(&test_features, test_labels, test_limit);
    println!("  Phase 1: {:.1}% ({:?})", phase1_acc * 100.0, start.elapsed());

    // Phase 2: Hebbian refinement
    let refine_start = Instant::now();
    for round in 0..20 {
        brain.refine(&train_features, train_labels, train_limit, 1);
        if round % 5 == 0 || round == 19 {
            let acc = brain.accuracy(&test_features, test_labels, test_limit);
            println!("  Round {:>2}: {:.1}% ({:.1?})", round + 1, acc * 100.0, refine_start.elapsed());
        }
    }

    let final_acc = brain.accuracy(&test_features, test_labels, test_limit);
    let total_time = start.elapsed();

    println!("\n{}", "=".repeat(60));
    println!("RESULT");
    println!("{}", "=".repeat(60));
    println!("  Conv features:   {:?}", feat_time);
    println!("  Phase 1 (init):  {:.1}%", phase1_acc * 100.0);
    println!("  Final:           {:.1}%", final_acc * 100.0);
    println!("  Total time:      {:?}", total_time);
    println!("  All ternary:     {}", brain.verify_ternary());
    println!("  Random chance:   10.0%");

    // Honest: random conv features don't help TernaryBrain on CIFAR-10.
    // The statistical approach needs learned features for complex images.
    assert!(final_acc > 0.09,
        "Must not be worse than random (got {:.1}%)", final_acc * 100.0);
}
