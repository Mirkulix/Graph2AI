//! CIFAR-10: Pretrained ResNet-18 features + TernaryBrain.
//!
//! Pipeline: CIFAR-10 → ResNet-18 (pretrained, frozen) → 512-dim features → TernaryBrain
//! This is transfer learning: pretrained deep features + ternary classification.

use qlang_runtime::cifar10::Cifar10Data;
use qlang_runtime::ternary_brain::TernaryBrain;
use std::time::Instant;

#[test]
fn cifar10_resnet_ternary() {
    let paths = [
        "data/cifar10",
        "../data/cifar10",
        "/home/mirkulix/neoqlang/qlang/data/cifar10",
    ];
    let mut data = None;
    for p in &paths { if let Ok(d) = Cifar10Data::load(p) { data = Some(d); break; } }
    let data = match data {
        Some(d) => d,
        None => { println!("CIFAR-10 not found"); return; }
    };

    println!("\n{}", "=".repeat(60));
    println!("CIFAR-10: Pretrained ResNet-18 + TernaryBrain");
    println!("{}", "=".repeat(60));

    // Load pretrained ResNet-18
    println!("\n=== Loading ResNet-18 ===");
    let resnet = match qo_embed::vision_resnet::ResNetExtractor::load() {
        Ok(r) => r,
        Err(e) => { println!("ResNet load failed: {}. Skipping test.", e); return; }
    };
    println!("  Feature dim: {}", resnet.feature_dim());

    // Extract features
    let train_limit = 10000; // ResNet on CPU is slow, start small
    let test_limit = 2000;

    println!("\n=== Feature Extraction ===");
    let start = Instant::now();
    let train_feat = match resnet.extract_batch(&data.train_images[..train_limit * 3072], train_limit) {
        Ok(f) => f,
        Err(e) => { println!("Feature extraction failed: {}. Skipping.", e); return; }
    };
    println!("  Train: {} images in {:?}", train_limit, start.elapsed());

    let start = Instant::now();
    let test_feat = match resnet.extract_batch(&data.test_images[..test_limit * 3072], test_limit) {
        Ok(f) => f,
        Err(e) => { println!("Test extraction failed: {}. Skipping.", e); return; }
    };
    println!("  Test:  {} images in {:?}", test_limit, start.elapsed());

    // TernaryBrain on ResNet features
    println!("\n=== TernaryBrain ===");
    let start = Instant::now();
    let mut brain = TernaryBrain::init(
        &train_feat, &data.train_labels[..train_limit],
        resnet.feature_dim(), train_limit, 10, 50,
    );
    let phase1 = brain.accuracy(&test_feat, &data.test_labels[..test_limit], test_limit);
    println!("  Phase 1: {:.1}% ({:?})", phase1 * 100.0, start.elapsed());

    let refine_start = Instant::now();
    for round in 0..15 {
        brain.refine(&train_feat, &data.train_labels[..train_limit], train_limit, 1);
        if round % 3 == 0 || round == 14 {
            let acc = brain.accuracy(&test_feat, &data.test_labels[..test_limit], test_limit);
            println!("  Round {:>2}: {:.1}% ({:.1?})", round + 1, acc * 100.0, refine_start.elapsed());
        }
    }

    let final_acc = brain.accuracy(&test_feat, &data.test_labels[..test_limit], test_limit);

    println!("\n{}", "=".repeat(60));
    println!("RESULT: ResNet-18 + TernaryBrain on CIFAR-10");
    println!("{}", "=".repeat(60));
    println!("  Phase 1:    {:.1}%", phase1 * 100.0);
    println!("  Final:      {:.1}%", final_acc * 100.0);
    println!("  All ternary: {}", brain.verify_ternary());
    println!("  Pipeline:   ResNet-18 (pretrained) → 512 features → TernaryBrain (ternary)");

    if final_acc > 0.70 {
        println!("\n  >>> 70%+ : PRETRAINED FEATURES + TERNARY BRAIN WORKS ON CIFAR-10!");
    }
}
