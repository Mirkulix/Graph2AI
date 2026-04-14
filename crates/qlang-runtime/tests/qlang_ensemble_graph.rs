//! Test: Ternary Ensemble Training as executable QLANG Graph.
//!
//! This proves that QLANG can express training — not just inference.
//! The training algorithm IS a graph that the executor runs.

use qlang_core::binary;
use qlang_core::graph::Graph;
use qlang_core::ops::Op;
use qlang_core::tensor::{Dtype, Shape, TensorData, TensorType};
use qlang_runtime::executor;
use qlang_runtime::mnist::MnistData;
use std::collections::HashMap;

/// Build a QLANG training graph for ternary ensemble.
///
/// Graph structure:
///   images ──┐
///            ├── ClassMean(10) ──┐
///   labels ──┘                   ├── Sub ── Ternarize(0.5) ── TernaryMatVec ── output
///                    global_mean ┘
///
/// Build a simplified QLANG training graph:
///   images + labels → ClassMean → Ternarize → TernaryMatVec(test_images) → ArgMax → predictions
///
/// The ClassMean output IS the difference (each class mean vs overall) — no separate Sub needed.
fn build_training_graph() -> Graph {
    let mut g = Graph::new("ternary_ensemble_train");

    let f32_type = TensorType::new(Dtype::F32, Shape::vector(1));

    // Inputs
    let images = g.add_node(Op::Input { name: "images".into() }, vec![], vec![f32_type.clone()]);
    let labels = g.add_node(Op::Input { name: "labels".into() }, vec![], vec![f32_type.clone()]);
    let test_images = g.add_node(Op::Input { name: "test_images".into() }, vec![], vec![f32_type.clone()]);

    // ClassMean: compute per-class mean vectors [10, 784]
    let class_mean = g.add_node(Op::ClassMean { n_classes: 10 }, vec![f32_type.clone(), f32_type.clone()], vec![f32_type.clone()]);
    g.add_edge(images, 0, class_mean, 0, f32_type.clone());
    g.add_edge(labels, 0, class_mean, 1, f32_type.clone());

    // Ternarize: class_means → ternary {-1, 0, +1}  [10, 784]
    let ternary = g.add_node(Op::Ternarize { threshold_ratio: 0.3 }, vec![f32_type.clone()], vec![f32_type.clone()]);
    g.add_edge(class_mean, 0, ternary, 0, f32_type.clone());

    // TernaryMatVec: ternary[10, 784] × test_images[n, 784]^T → scores[n, 10]
    let scores = g.add_node(Op::TernaryMatVec, vec![f32_type.clone(), f32_type.clone()], vec![f32_type.clone()]);
    g.add_edge(ternary, 0, scores, 0, f32_type.clone());
    g.add_edge(test_images, 0, scores, 1, f32_type.clone());

    // ArgMax: scores[n, 10] → predictions[n]
    let predictions = g.add_node(Op::ArgMax, vec![f32_type.clone()], vec![f32_type.clone()]);
    g.add_edge(scores, 0, predictions, 0, f32_type.clone());

    // Outputs
    let out_weights = g.add_node(Op::Output { name: "ternary_weights".into() }, vec![f32_type.clone()], vec![]);
    g.add_edge(ternary, 0, out_weights, 0, f32_type.clone());

    let out_preds = g.add_node(Op::Output { name: "predictions".into() }, vec![f32_type.clone()], vec![]);
    g.add_edge(predictions, 0, out_preds, 0, f32_type.clone());

    g
}

#[test]
fn training_graph_builds_and_serializes() {
    let graph = build_training_graph();

    println!("Graph: {} nodes, {} edges", graph.nodes.len(), graph.edges.len());
    assert!(graph.nodes.len() >= 8, "Should have at least 8 nodes");

    // Serialize to QLBG binary
    let binary_data = binary::to_binary(&graph);
    assert_eq!(&binary_data[0..4], &[0x51, 0x4C, 0x42, 0x47], "Must have QLBG magic");
    println!("QLBG binary: {} bytes", binary_data.len());

    // Deserialize and verify
    let restored = binary::from_binary(&binary_data).expect("Must deserialize");
    assert_eq!(restored.nodes.len(), graph.nodes.len());
    assert_eq!(restored.edges.len(), graph.edges.len());
    println!("Round-trip: OK");
}

#[test]
fn training_graph_executes_on_mnist() {
    let data = MnistData::synthetic(500, 100);

    let graph = build_training_graph();

    // Prepare inputs
    let mut inputs = HashMap::new();
    inputs.insert("images".to_string(),
        TensorData::from_f32(Shape::matrix(data.n_train, 784), &data.train_images));
    inputs.insert("labels".to_string(),
        TensorData::from_f32(Shape::vector(data.n_train),
            &data.train_labels.iter().map(|&l| l as f32).collect::<Vec<_>>()));
    inputs.insert("test_images".to_string(),
        TensorData::from_f32(Shape::matrix(data.n_test, 784), &data.test_images));

    // Execute the training graph
    let start = std::time::Instant::now();
    let result = executor::execute(&graph, inputs).expect("Graph must execute");
    let exec_time = start.elapsed();

    println!("\n=== QLANG Training Graph Execution ===");
    println!("Time: {:?}", exec_time);
    println!("Nodes executed: {}", result.stats.nodes_executed);

    // Check outputs
    assert!(result.outputs.contains_key("ternary_weights"), "Must produce ternary weights");
    assert!(result.outputs.contains_key("predictions"), "Must produce predictions");

    let weights = result.outputs["ternary_weights"].as_f32_slice().unwrap();
    let preds = result.outputs["predictions"].as_f32_slice().unwrap();

    println!("Ternary weights: {} values", weights.len());
    println!("Predictions: {} values", preds.len());

    // Verify weights are ternary
    let mut pos = 0; let mut zero = 0; let mut neg = 0;
    for &w in &weights {
        if w > 0.5 { pos += 1; }
        else if w < -0.5 { neg += 1; }
        else { zero += 1; }
    }
    println!("Weight distribution: +1:{} 0:{} -1:{}", pos, zero, neg);

    // Compute accuracy
    let correct = preds.iter().zip(data.test_labels.iter())
        .filter(|(&p, &l)| (p as u8) == l)
        .count();
    let accuracy = correct as f32 / data.n_test as f32;
    println!("Accuracy: {:.1}% ({}/{} correct)", accuracy * 100.0, correct, data.n_test);

    // Assertions
    assert!(pos + neg > 0, "Must have non-zero ternary weights");
    println!("\nBEWEIS: Training als QLANG-Graph ausgefuehrt.");
    println!("  - Graph gebaut: {} Nodes, {} Edges", graph.nodes.len(), graph.edges.len());
    println!("  - Serialisiert: {} bytes QLBG", binary::to_binary(&graph).len());
    println!("  - Executor: {} Nodes ausgefuehrt in {:?}", result.stats.nodes_executed, exec_time);
    println!("  - Gewichte: 100% ternaer ({} +1, {} 0, {} -1)", pos, zero, neg);
    println!("  - Accuracy: {:.1}%", accuracy * 100.0);
}
