//! CIFAR-10: Forward-Forward on QLANG Graph with pre-extracted ResNet features.
//!
//! Pipeline: ResNet features (512-dim) → QLANG Graph (3 layers, FF trained) → Classification
//! This is the full pipeline: deep features + QLANG training + ternary inference.

use qlang_runtime::graph_ff_train::TrainableGraph;
use std::time::Instant;

fn load_f32_bin(path: &str) -> Option<Vec<f32>> {
    let bytes = std::fs::read(path).ok()?;
    Some(bytes.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect())
}

#[test]
fn cifar10_graph_ff_training() {
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
    let train_limit = n_train.min(20000); // Use 20K for speed
    let test_limit = n_test.min(5000);

    println!("\n{}", "=".repeat(60));
    println!("CIFAR-10: Forward-Forward on QLANG Graph");
    println!("Train: {}, Test: {}, Features: {}", train_limit, test_limit, feat_dim);
    println!("{}\n", "=".repeat(60));

    // 3-layer graph: 522 (512+10) → 256 → 128 → 64
    let mut graph = TrainableGraph::new(&[522, 256, 128, 64], 10);

    let total_start = Instant::now();
    for epoch in 0..20 {
        let (pg, ng) = graph.train_epoch(
            &train_feat[..train_limit * feat_dim],
            &train_labels[..train_limit],
            feat_dim, train_limit, 100,
        );
        if epoch % 4 == 0 || epoch == 19 {
            let f32_acc = graph.accuracy(&test_feat[..test_limit * feat_dim], &test_labels[..test_limit], feat_dim, test_limit);
            let tern_acc = graph.accuracy_ternary(&test_feat[..test_limit * feat_dim], &test_labels[..test_limit], feat_dim, test_limit);
            println!("  Epoch {:>2}: f32={:.1}% tern={:.1}% pg={:.2} ng={:.2} ({:.1?})",
                epoch + 1, f32_acc * 100.0, tern_acc * 100.0, pg, ng, total_start.elapsed());
        }
    }

    let f32_acc = graph.accuracy(&test_feat[..test_limit * feat_dim], &test_labels[..test_limit], feat_dim, test_limit);
    let tern_acc = graph.accuracy_ternary(&test_feat[..test_limit * feat_dim], &test_labels[..test_limit], feat_dim, test_limit);

    println!("\n{}", "=".repeat(60));
    println!("RESULT: CIFAR-10 Graph FF");
    println!("{}", "=".repeat(60));
    println!("  f32:     {:.1}%", f32_acc * 100.0);
    println!("  Ternary: {:.1}%", tern_acc * 100.0);
    println!("  Time:    {:?}", total_start.elapsed());
    println!("  Layers:  3 (522→256→128→64)");
    println!("  Method:  Forward-Forward (no backprop)");
}
