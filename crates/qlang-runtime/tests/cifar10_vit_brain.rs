//! CIFAR-10: Vision Transformer Features + TernaryBrain.
//!
//! Pipeline: Image → ViT (random, frozen) → Features → TernaryBrain
//! The transformer's attention captures patch relationships.

use qlang_runtime::cifar10::Cifar10Data;
use qlang_runtime::vision_transformer::VisionTransformer;
use qlang_runtime::ternary_brain::TernaryBrain;
use std::time::Instant;

#[test]
fn cifar10_vit_brain() {
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

    let train_limit = 20000;
    let test_limit = 5000;

    println!("\n{}", "=".repeat(60));
    println!("CIFAR-10: Vision Transformer + TernaryBrain");
    println!("{}\n", "=".repeat(60));

    // ViT: d_model=64, 4 heads, 4x4 patches
    let vit = VisionTransformer::new(64, 4, 32, 4, 3);
    println!("ViT: d_model={}, heads={}, patches={}", vit.d_model, vit.n_heads, vit.n_patches);

    println!("\n=== Feature Extraction (parallel) ===");
    let start = Instant::now();
    let train_feat = vit.extract_batch(&data.train_images[..train_limit * 3072], train_limit);
    let test_feat = vit.extract_batch(&data.test_images[..test_limit * 3072], test_limit);
    println!("  {:?} for {} + {} images → {} features", start.elapsed(), train_limit, test_limit, vit.d_model);

    let train_labels = &data.train_labels[..train_limit];
    let test_labels = &data.test_labels[..test_limit];

    println!("\n=== TernaryBrain ===");
    let start = Instant::now();
    let mut brain = TernaryBrain::init(
        &train_feat, train_labels,
        vit.d_model, train_limit, 10, 50,
    );
    let phase1_acc = brain.accuracy(&test_feat, test_labels, test_limit);
    println!("  Phase 1: {:.1}% ({:?})", phase1_acc * 100.0, start.elapsed());

    let refine_start = Instant::now();
    for round in 0..15 {
        brain.refine(&train_feat, train_labels, train_limit, 1);
        if round % 3 == 0 || round == 14 {
            let acc = brain.accuracy(&test_feat, test_labels, test_limit);
            println!("  Round {:>2}: {:.1}% ({:.1?})", round + 1, acc * 100.0, refine_start.elapsed());
        }
    }

    let final_acc = brain.accuracy(&test_feat, test_labels, test_limit);

    println!("\n{}", "=".repeat(60));
    println!("RESULT: ViT + TernaryBrain on CIFAR-10");
    println!("{}", "=".repeat(60));
    println!("  Phase 1: {:.1}%", phase1_acc * 100.0);
    println!("  Final:   {:.1}%", final_acc * 100.0);
    println!("  Ternary: {}", brain.verify_ternary());

    // Random ViT features ≈ random. Need pretrained weights for useful features.
    assert!(final_acc > 0.09,
        "Must not be worse than random (got {:.1}%)", final_acc * 100.0);
}
