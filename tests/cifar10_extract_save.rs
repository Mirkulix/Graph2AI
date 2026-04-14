//! Extract ResNet-18 features for all CIFAR-10 and save to disk.
//! Run once, then cifar10_full_brain.rs loads instantly.

use qlang_runtime::cifar10::Cifar10Data;
use std::time::Instant;

#[test]
fn extract_and_save_features() {
    let paths = [
        "data/cifar10",
        "/home/mirkulix/neoqlang/qlang/data/cifar10",
    ];
    let mut data = None;
    for p in &paths { if let Ok(d) = Cifar10Data::load(p) { data = Some(d); break; } }
    let data = match data {
        Some(d) => d,
        None => { println!("CIFAR-10 not found"); return; }
    };

    let resnet = match qo_embed::vision_resnet::ResNetExtractor::load() {
        Ok(r) => r,
        Err(e) => { println!("ResNet failed: {e}"); return; }
    };

    println!("Extracting features for {} train + {} test images...", data.n_train, data.n_test);

    // Train features
    let start = Instant::now();
    let train_feat = resnet.extract_batch(&data.train_images, data.n_train).unwrap();
    println!("Train: {:?} ({} features)", start.elapsed(), train_feat.len());

    // Test features
    let start = Instant::now();
    let test_feat = resnet.extract_batch(&data.test_images, data.n_test).unwrap();
    println!("Test: {:?} ({} features)", start.elapsed(), test_feat.len());

    // Save as raw f32 binary
    let train_bytes: Vec<u8> = train_feat.iter().flat_map(|f| f.to_le_bytes()).collect();
    let test_bytes: Vec<u8> = test_feat.iter().flat_map(|f| f.to_le_bytes()).collect();

    std::fs::write("data/cifar10_resnet_train.bin", &train_bytes).unwrap();
    std::fs::write("data/cifar10_resnet_test.bin", &test_bytes).unwrap();
    std::fs::write("data/cifar10_train_labels.bin", &data.train_labels).unwrap();
    std::fs::write("data/cifar10_test_labels.bin", &data.test_labels).unwrap();

    println!("Saved: train={} bytes, test={} bytes",
        train_bytes.len(), test_bytes.len());
}
