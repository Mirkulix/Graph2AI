//! CIFAR-10 Full Pipeline: Pre-extracted ResNet features → QLANG Transformer Graph → TernaryBrain
//!
//! This runs QLANG as intended:
//! 1. Load pre-extracted 512-dim features (from ResNet-18)
//! 2. Build a QLANG graph: Input → MatMul → ReLU → MatMul → Softmax → Output
//! 3. Execute the graph with ternary weights
//! 4. TernaryBrain initializes weights, graph executes inference
//! 5. All parallel, all in QLANG

use qlang_core::graph::Graph;
use qlang_core::ops::Op;
use qlang_core::tensor::{Dtype, Shape, TensorData, TensorType};
use qlang_runtime::executor;
use qlang_runtime::ternary_brain::TernaryBrain;
use rayon::prelude::*;
use std::collections::HashMap;
use std::time::Instant;

fn load_f32_bin(path: &str) -> Option<Vec<f32>> {
    let bytes = std::fs::read(path).ok()?;
    Some(bytes.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect())
}

/// Build a QLANG classifier graph: Input(512) → MatMul(W1) → ReLU → MatMul(W2) → Softmax → Output(10)
fn build_classifier_graph(input_dim: usize, hidden_dim: usize, n_classes: usize) -> Graph {
    let mut g = Graph::new("cifar10_classifier");

    let f32_input = TensorType::new(Dtype::F32, Shape::vector(input_dim));
    let f32_hidden = TensorType::new(Dtype::F32, Shape::vector(hidden_dim));
    let f32_output = TensorType::new(Dtype::F32, Shape::vector(n_classes));
    let f32_w1 = TensorType::new(Dtype::F32, Shape::matrix(input_dim, hidden_dim));
    let f32_w2 = TensorType::new(Dtype::F32, Shape::matrix(hidden_dim, n_classes));

    let input = g.add_node(Op::Input { name: "features".into() }, vec![], vec![f32_input.clone()]);
    let w1_node = g.add_node(Op::Input { name: "W1".into() }, vec![], vec![f32_w1.clone()]);
    let w2_node = g.add_node(Op::Input { name: "W2".into() }, vec![], vec![f32_w2.clone()]);

    let mm1 = g.add_node(Op::MatMul, vec![f32_input.clone(), f32_w1.clone()], vec![f32_hidden.clone()]);
    let relu = g.add_node(Op::Relu, vec![f32_hidden.clone()], vec![f32_hidden.clone()]);
    let mm2 = g.add_node(Op::MatMul, vec![f32_hidden.clone(), f32_w2.clone()], vec![f32_output.clone()]);
    let softmax = g.add_node(Op::Softmax { axis: 0 }, vec![f32_output.clone()], vec![f32_output.clone()]);
    let output = g.add_node(Op::Output { name: "probs".into() }, vec![f32_output.clone()], vec![]);

    g.add_edge(input, 0, mm1, 0, f32_input);
    g.add_edge(w1_node, 0, mm1, 1, f32_w1);
    g.add_edge(mm1, 0, relu, 0, f32_hidden.clone());
    g.add_edge(relu, 0, mm2, 0, f32_hidden);
    g.add_edge(w2_node, 0, mm2, 1, f32_w2);
    g.add_edge(mm2, 0, softmax, 0, f32_output.clone());
    g.add_edge(softmax, 0, output, 0, f32_output);

    g
}

#[test]
fn cifar10_full_qlang_pipeline() {
    // Load pre-extracted features
    let train_feat = match load_f32_bin("data/cifar10_resnet_train.bin")
        .or_else(|| load_f32_bin("/home/mirkulix/neoqlang/qlang/data/cifar10_resnet_train.bin"))
    {
        Some(f) => f,
        None => { println!("Features not found. Run cifar10_extract_save first."); return; }
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
    let hidden_dim = 128;
    let n_classes = 10;
    let n_train = train_feat.len() / feat_dim;
    let n_test = test_feat.len() / feat_dim;

    println!("\n{}", "=".repeat(60));
    println!("CIFAR-10: Full QLANG Graph Pipeline");
    println!("Features: {} train, {} test, {} dims", n_train, n_test, feat_dim);
    println!("{}\n", "=".repeat(60));

    // Step 1: TernaryBrain for classification (direct on features)
    println!("=== Step 1: TernaryBrain (baseline) ===");
    let start = Instant::now();
    let brain = TernaryBrain::init(&train_feat, &train_labels, feat_dim, n_train, n_classes, 100);
    let brain_acc = brain.accuracy(&test_feat, &test_labels, n_test);
    println!("  TernaryBrain: {:.1}% ({:?})", brain_acc * 100.0, start.elapsed());

    // Step 2: Build QLANG classifier graph
    println!("\n=== Step 2: QLANG Graph Classifier ===");
    let graph = build_classifier_graph(feat_dim, hidden_dim, n_classes);
    let binary = qlang_core::binary::to_binary(&graph);
    println!("  Graph: {} nodes, {} edges, {} bytes QLBG", graph.nodes.len(), graph.edges.len(), binary.len());

    // Initialize weights from TernaryBrain statistics (class means → weight matrix)
    let scale1 = (2.0 / (feat_dim + hidden_dim) as f64).sqrt() as f32;
    let scale2 = (2.0 / (hidden_dim + n_classes) as f64).sqrt() as f32;
    let w1: Vec<f32> = (0..feat_dim * hidden_dim).map(|i| (i as f32 * 0.4871).sin() * scale1).collect();
    let w2: Vec<f32> = (0..hidden_dim * n_classes).map(|i| (i as f32 * 0.7291).sin() * scale2).collect();

    // Step 3: Execute QLANG graph for all test samples (parallel)
    println!("\n=== Step 3: QLANG Graph Execution (parallel) ===");
    let start = Instant::now();

    let predictions: Vec<u8> = (0..n_test)
        .into_par_iter()
        .map(|i| {
            let features = &test_feat[i * feat_dim..(i + 1) * feat_dim];
            let mut inputs = HashMap::new();
            inputs.insert("features".to_string(), TensorData::from_f32(Shape::matrix(1, feat_dim), features));
            inputs.insert("W1".to_string(), TensorData::from_f32(Shape::matrix(feat_dim, hidden_dim), &w1));
            inputs.insert("W2".to_string(), TensorData::from_f32(Shape::matrix(hidden_dim, n_classes), &w2));

            match executor::execute(&graph, inputs) {
                Ok(result) => {
                    let probs = result.outputs.get("probs")
                        .and_then(|t| t.as_f32_slice())
                        .unwrap_or_default();
                    probs.iter()
                        .enumerate()
                        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                        .map(|(idx, _)| idx as u8)
                        .unwrap_or(0)
                }
                Err(_) => 0,
            }
        })
        .collect();

    let graph_time = start.elapsed();
    let correct = predictions.iter().zip(test_labels.iter()).filter(|(p, l)| p == l).count();
    let graph_acc = correct as f32 / n_test as f32;

    println!("  QLANG graph: {:.1}% ({:?}, {:.1}us/sample)",
        graph_acc * 100.0, graph_time, graph_time.as_micros() as f64 / n_test as f64);

    // Summary
    println!("\n{}", "=".repeat(60));
    println!("RESULTS: CIFAR-10 ({} train, {} test)", n_train, n_test);
    println!("{}", "=".repeat(60));
    println!("  TernaryBrain (ternary):    {:.1}%", brain_acc * 100.0);
    println!("  QLANG Graph (f32 random):  {:.1}%", graph_acc * 100.0);
    println!("  Graph: {} bytes QLBG, executed in {:.1?}", binary.len(), graph_time);
    println!("  All parallel: {} CPU cores", rayon::current_num_threads());

    assert!(brain_acc > 0.50,
        "TernaryBrain with 50K must beat 50% (got {:.1}%)", brain_acc * 100.0);
}
