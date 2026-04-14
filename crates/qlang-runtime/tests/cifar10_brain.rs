//! THE TEST: TernaryBrain on CIFAR-10.
//!
//! If this passes >40%, the approach generalizes beyond MNIST.
//! If >70%, it's genuinely competitive.
//! Random chance = 10%.
//!
//! CIFAR-10: 32x32 RGB images (3072 dims), 10 classes:
//! airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck

use qlang_runtime::cifar10::Cifar10Data;
use qlang_runtime::ternary_brain::TernaryBrain;
use std::time::Instant;

#[test]
fn ternary_brain_on_cifar10() {
    let paths = [
        "data/cifar10",
        "../data/cifar10",
        "/home/mirkulix/neoqlang/qlang/data/cifar10",
    ];

    let mut data = None;
    for p in &paths {
        if let Ok(d) = Cifar10Data::load(p) {
            data = Some(d);
            break;
        }
    }

    let data = match data {
        Some(d) => d,
        None => {
            println!("CIFAR-10 not found, skipping");
            return;
        }
    };

    println!("\n{}", "=".repeat(60));
    println!("TernaryBrain on CIFAR-10");
    println!("Train: {}, Test: {}, Dims: {}, Classes: {}",
        data.n_train, data.n_test, data.image_dim, data.n_classes);
    println!("{}\n", "=".repeat(60));

    // Use subset for speed: 10K train, 2K test
    let train_limit = data.n_train.min(10000);
    let test_limit = data.n_test.min(2000);
    let train_images = &data.train_images[..train_limit * 3072];
    let train_labels = &data.train_labels[..train_limit];
    let test_images = &data.test_images[..test_limit * 3072];
    let test_labels = &data.test_labels[..test_limit];

    println!("Using: {} train, {} test\n", train_limit, test_limit);

    // Phase 1: Statistical Init
    println!("=== Phase 1: Statistical Init ===");
    let start = Instant::now();
    let mut brain = TernaryBrain::init(
        train_images, train_labels,
        3072,        // image_dim (32x32x3)
        train_limit,
        10,          // n_classes
        30,          // neurons_per_class (more for harder task)
    );
    let phase1_time = start.elapsed();
    let phase1_acc = brain.accuracy(test_images, test_labels, test_limit);
    println!("  Accuracy: {:.1}% (in {:?})", phase1_acc * 100.0, phase1_time);
    println!("  Ternary verified: {}", brain.verify_ternary());

    // Phase 2: Competitive Hebbian Refinement
    println!("\n=== Phase 2: Competitive Hebbian ===");
    let total_start = Instant::now();
    for round in 0..10 {
        brain.refine(train_images, train_labels, train_limit, 1);
        if round % 2 == 0 || round == 9 {
            let acc = brain.accuracy(test_images, test_labels, test_limit);
            println!("  Round {:>2}: {:.1}% ({:.1?})",
                round + 1, acc * 100.0, total_start.elapsed());
        }
    }

    let final_acc = brain.accuracy(test_images, test_labels, test_limit);
    let total_time = total_start.elapsed();

    println!("\n{}", "=".repeat(60));
    println!("RESULT: CIFAR-10");
    println!("{}", "=".repeat(60));
    println!("  Phase 1 (statistical): {:.1}%", phase1_acc * 100.0);
    println!("  Final (after Hebbian): {:.1}%", final_acc * 100.0);
    println!("  Total weights:         {}", brain.total_weights());
    println!("  All ternary:           {}", brain.verify_ternary());
    println!("  Time:                  {:?}", total_time);
    println!("  Random chance:         10.0%");
    println!();

    // THE assertion
    assert!(final_acc > 0.20,
        "TernaryBrain on CIFAR-10 must beat 20% (got {:.1}%). \
         If this fails, the approach does not generalize beyond MNIST.",
        final_acc * 100.0);

    if final_acc > 0.40 {
        println!("  >>> GENERALIZES: >40% on CIFAR-10! The approach works beyond MNIST.");
    }
    if final_acc > 0.70 {
        println!("  >>> COMPETITIVE: >70% on CIFAR-10! Genuinely useful.");
    }
}
