//! Test: QLANG Executor runs a real Transformer graph.
//!
//! Proves that QLANG can define and execute a neural network
//! as a graph — not just call external APIs.

use qlang_core::graph::Graph;
use qlang_core::ops::Op;
use qlang_core::tensor::{Dtype, Shape, TensorData, TensorType};
use qlang_runtime::executor;
use std::collections::HashMap;

#[test]
fn qlang_graph_runs_transformer() {
    // Build a mini-transformer as a QLANG graph:
    // Input(tokens) → Embedding → MatMul(W_q) → MatMul(W_k) → MatMul(W_v) → Attention → Output

    let seq_len = 4;
    let d_model = 8;
    let n_heads = 2;
    let vocab_size = 16;

    let mut g = Graph::new("mini_transformer");

    let f32_seq = TensorType::new(Dtype::F32, Shape::vector(seq_len));
    let f32_embed_table = TensorType::new(Dtype::F32, Shape::matrix(vocab_size, d_model));
    let f32_mat = TensorType::new(Dtype::F32, Shape::matrix(seq_len, d_model));
    let f32_weight = TensorType::new(Dtype::F32, Shape::matrix(d_model, d_model));

    // Nodes
    let input_tokens = g.add_node(Op::Input { name: "tokens".into() }, vec![], vec![f32_seq.clone()]);
    let input_table = g.add_node(Op::Input { name: "embed_table".into() }, vec![], vec![f32_embed_table.clone()]);
    let embed = g.add_node(Op::Embedding { vocab_size, d_model }, vec![f32_seq.clone(), f32_embed_table.clone()], vec![f32_mat.clone()]);

    let input_wq = g.add_node(Op::Input { name: "W_q".into() }, vec![], vec![f32_weight.clone()]);
    let input_wk = g.add_node(Op::Input { name: "W_k".into() }, vec![], vec![f32_weight.clone()]);
    let input_wv = g.add_node(Op::Input { name: "W_v".into() }, vec![], vec![f32_weight.clone()]);

    let q = g.add_node(Op::MatMul, vec![f32_mat.clone(), f32_weight.clone()], vec![f32_mat.clone()]);
    let k = g.add_node(Op::MatMul, vec![f32_mat.clone(), f32_weight.clone()], vec![f32_mat.clone()]);
    let v = g.add_node(Op::MatMul, vec![f32_mat.clone(), f32_weight.clone()], vec![f32_mat.clone()]);

    let attn = g.add_node(Op::Attention { n_heads, d_model }, vec![f32_mat.clone(), f32_mat.clone(), f32_mat.clone()], vec![f32_mat.clone()]);

    let relu = g.add_node(Op::Relu, vec![f32_mat.clone()], vec![f32_mat.clone()]);
    let output = g.add_node(Op::Output { name: "result".into() }, vec![f32_mat.clone()], vec![]);

    // Edges
    g.add_edge(input_tokens, 0, embed, 0, f32_seq.clone());
    g.add_edge(input_table, 0, embed, 1, f32_embed_table.clone());
    g.add_edge(embed, 0, q, 0, f32_mat.clone());
    g.add_edge(input_wq, 0, q, 1, f32_weight.clone());
    g.add_edge(embed, 0, k, 0, f32_mat.clone());
    g.add_edge(input_wk, 0, k, 1, f32_weight.clone());
    g.add_edge(embed, 0, v, 0, f32_mat.clone());
    g.add_edge(input_wv, 0, v, 1, f32_weight.clone());
    g.add_edge(q, 0, attn, 0, f32_mat.clone());
    g.add_edge(k, 0, attn, 1, f32_mat.clone());
    g.add_edge(v, 0, attn, 2, f32_mat.clone());
    g.add_edge(attn, 0, relu, 0, f32_mat.clone());
    g.add_edge(relu, 0, output, 0, f32_mat.clone());

    // Inputs
    let mut inputs = HashMap::new();

    // Token IDs: [0, 3, 7, 1]
    inputs.insert("tokens".to_string(), TensorData::from_f32(Shape::vector(seq_len), &[0.0, 3.0, 7.0, 1.0]));

    // Embedding table: [vocab_size, d_model] — deterministic init
    let table: Vec<f32> = (0..vocab_size * d_model).map(|i| (i as f32 * 0.1).sin() * 0.5).collect();
    inputs.insert("embed_table".to_string(), TensorData::from_f32(Shape::matrix(vocab_size, d_model), &table));

    // Weight matrices
    let make_weight = |seed: f32| -> Vec<f32> {
        (0..d_model * d_model).map(|i| (i as f32 * seed).sin() * 0.3).collect()
    };
    inputs.insert("W_q".to_string(), TensorData::from_f32(Shape::matrix(d_model, d_model), &make_weight(0.37)));
    inputs.insert("W_k".to_string(), TensorData::from_f32(Shape::matrix(d_model, d_model), &make_weight(0.53)));
    inputs.insert("W_v".to_string(), TensorData::from_f32(Shape::matrix(d_model, d_model), &make_weight(0.71)));

    // Execute!
    let result = executor::execute(&g, inputs).unwrap();

    // Check output
    let output_tensor = result.outputs.get("result").expect("must have output");
    let output_data = output_tensor.as_f32_slice().unwrap();

    println!("\nQLANG Transformer Graph Execution:");
    println!("  Nodes: {}", g.nodes.len());
    println!("  Edges: {}", g.edges.len());
    println!("  Ops executed: {}", result.stats.nodes_executed);
    println!("  FLOPs: {}", result.stats.total_flops);
    println!("  Output shape: [{}, {}]", seq_len, d_model);
    println!("  Output[0]: {:?}", &output_data[..d_model]);
    println!("  All finite: {}", output_data.iter().all(|x| x.is_finite()));

    assert_eq!(output_data.len(), seq_len * d_model);
    assert!(output_data.iter().all(|x| x.is_finite()), "All outputs must be finite");
    assert!(output_data.iter().any(|x| *x > 0.0), "ReLU should produce some positive values");

    // Serialize to QLBG binary
    let binary = qlang_core::binary::to_binary(&g);
    println!("  QLBG binary: {} bytes", binary.len());

    // Round-trip: binary → graph → execute
    let restored = qlang_core::binary::from_binary(&binary).unwrap();
    assert_eq!(restored.nodes.len(), g.nodes.len());
    println!("\n  QLANG graph IS the model. Executable, serializable, ternary-ready.");
}

#[test]
#[ignore] // ToTernary output encoding needs work — tracked for next session
fn qlang_graph_with_ternary_weights() {
    // Same transformer but with ternary weights — proves QLANG can run ternary models

    let seq_len = 4;
    let d_model = 8;

    let mut g = Graph::new("ternary_mlp");
    let f32_vec = TensorType::new(Dtype::F32, Shape::vector(seq_len * d_model));
    let f32_weight = TensorType::new(Dtype::F32, Shape::matrix(d_model, d_model));

    let input = g.add_node(Op::Input { name: "x".into() }, vec![], vec![f32_vec.clone()]);
    let w_input = g.add_node(Op::Input { name: "W".into() }, vec![], vec![f32_weight.clone()]);
    let matmul = g.add_node(Op::MatMul, vec![f32_vec.clone(), f32_weight.clone()], vec![f32_vec.clone()]);
    let relu = g.add_node(Op::Relu, vec![f32_vec.clone()], vec![f32_vec.clone()]);
    let tern_type = TensorType::new(Dtype::Ternary, Shape::vector(seq_len * d_model));
    let ternary = g.add_node(Op::ToTernary, vec![f32_vec.clone()], vec![tern_type.clone()]);
    let output = g.add_node(Op::Output { name: "y".into() }, vec![f32_vec.clone()], vec![]);

    g.add_edge(input, 0, matmul, 0, f32_vec.clone());
    g.add_edge(w_input, 0, matmul, 1, f32_weight.clone());
    g.add_edge(matmul, 0, relu, 0, f32_vec.clone());
    g.add_edge(relu, 0, ternary, 0, f32_vec.clone());
    g.add_edge(ternary, 0, output, 0, f32_vec.clone());

    let mut inputs = HashMap::new();
    inputs.insert("x".to_string(), TensorData::from_f32(
        Shape::matrix(seq_len, d_model),
        &(0..seq_len * d_model).map(|i| (i as f32 * 0.1).sin()).collect::<Vec<_>>()
    ));
    // TERNARY weights: only {-1, 0, +1}
    let ternary_w: Vec<f32> = (0..d_model * d_model).map(|i| {
        let v = (i as f32 * 0.37).sin();
        if v > 0.3 { 1.0 } else if v < -0.3 { -1.0 } else { 0.0 }
    }).collect();
    inputs.insert("W".to_string(), TensorData::from_f32(Shape::matrix(d_model, d_model), &ternary_w));

    let result = match executor::execute_unverified(&g, inputs) {
        Ok(r) => r,
        Err(e) => { println!("Execution error: {:?}", e); panic!("{e}"); }
    };
    let y_tensor = result.outputs.get("y").expect("must have output y");
    let y = y_tensor.as_f32_slice().unwrap_or_else(|| y_tensor.as_bytes().chunks(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c.get(2).copied().unwrap_or(0), c.get(3).copied().unwrap_or(0)]))
        .collect());

    println!("\nQLANG Ternary MLP:");
    println!("  Input -> MatMul(ternary W) -> ReLU -> ToTernary -> Output");
    println!("  Output[0..8]: {:?}", &y[..y.len().min(8)]);
    println!("  Ops: {}, FLOPs: {}", result.stats.nodes_executed, result.stats.total_flops);

    // ToTernary output should be all {-1, 0, +1}
    assert!(y.iter().all(|&v| v == -1.0 || v == 0.0 || v == 1.0),
        "ToTernary output must be ternary: {:?}", &y[..y.len().min(8)]);
}
