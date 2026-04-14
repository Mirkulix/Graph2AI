//! CIFAR-10: NoProp with pre-extracted ResNet features.
//! Target: >70% with ternary weights.

use qlang_runtime::noprop::NoPropNet;
use std::time::Instant;

fn load_f32_bin(path: &str) -> Option<Vec<f32>> {
    let bytes = std::fs::read(path).ok()?;
    Some(bytes.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect())
}

#[test]
fn cifar10_noprop() {
    let train_feat = match load_f32_bin("data/cifar10_resnet_train.bin")
        .or_else(|| load_f32_bin("/home/mirkulix/neoqlang/qlang/data/cifar10_resnet_train.bin"))
    {
        Some(f) => f,
        None => { println!("Features not found"); return; }
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
    let train_limit = n_train.min(30000);
    let test_limit = n_test.min(10000);

    println!("\n{}", "=".repeat(60));
    println!("CIFAR-10: NoProp (Denoising, no backprop between blocks)");
    println!("Train: {}, Test: {}, Features: {}", train_limit, test_limit, feat_dim);
    println!("{}\n", "=".repeat(60));

    // NoProp: 10 denoising steps, label_dim=10 (one-hot)
    let mut net = NoPropNet::new(feat_dim, 10, 10, 10);
    println!("  Blocks: {}, label_dim: {}", net.n_steps, net.label_dim);

    let start = Instant::now();
    for epoch in 0..20 {
        let loss = net.train_epoch(
            &train_feat[..train_limit * feat_dim],
            &train_labels[..train_limit],
            train_limit, 100,
        );
        if epoch % 4 == 0 || epoch == 19 {
            let f32_acc = net.accuracy(&test_feat[..test_limit * feat_dim], &test_labels[..test_limit], test_limit, false);
            let tern_acc = net.accuracy(&test_feat[..test_limit * feat_dim], &test_labels[..test_limit], test_limit, true);
            println!("  Epoch {:>2}: loss={:.4} f32={:.1}% tern={:.1}% ({:.1?})",
                epoch + 1, loss, f32_acc * 100.0, tern_acc * 100.0, start.elapsed());
        }
    }

    let f32_acc = net.accuracy(&test_feat[..test_limit * feat_dim], &test_labels[..test_limit], test_limit, false);
    let tern_acc = net.accuracy(&test_feat[..test_limit * feat_dim], &test_labels[..test_limit], test_limit, true);

    println!("\n{}", "=".repeat(60));
    println!("RESULT: CIFAR-10 NoProp");
    println!("{}", "=".repeat(60));
    println!("  f32:     {:.1}%", f32_acc * 100.0);
    println!("  Ternary: {:.1}%", tern_acc * 100.0);
    println!("  Time:    {:?}", start.elapsed());

    if tern_acc > 0.70 {
        println!("  >>> 70%+ TERNARY ON CIFAR-10!");
    }
}
