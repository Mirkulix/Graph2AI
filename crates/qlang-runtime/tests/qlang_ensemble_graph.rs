//! Integration test: executable QLANG graph roundtrip using only currently
//! supported core ops.

use qlang_core::binary;
use qlang_core::graph::Graph;
use qlang_core::ops::Op;
use qlang_core::tensor::{Shape, TensorData, TensorType};
use qlang_runtime::executor;
use std::collections::HashMap;

fn build_inference_graph() -> Graph {
    let mut g = Graph::new("linear_relu");
    let x_ty = TensorType::f32_matrix(2, 2);
    let w_ty = TensorType::f32_matrix(2, 2);
    let y_ty = TensorType::f32_matrix(2, 2);

    let x = g.add_node(Op::Input { name: "x".into() }, vec![], vec![x_ty.clone()]);
    let w = g.add_node(Op::Input { name: "w".into() }, vec![], vec![w_ty.clone()]);
    let mm = g.add_node(Op::MatMul, vec![x_ty.clone(), w_ty.clone()], vec![y_ty.clone()]);
    let relu = g.add_node(Op::Relu, vec![y_ty.clone()], vec![y_ty.clone()]);
    let out = g.add_node(Op::Output { name: "y".into() }, vec![y_ty.clone()], vec![]);

    g.add_edge(x, 0, mm, 0, x_ty.clone());
    g.add_edge(w, 0, mm, 1, w_ty.clone());
    g.add_edge(mm, 0, relu, 0, y_ty.clone());
    g.add_edge(relu, 0, out, 0, y_ty);

    g
}

#[test]
fn executable_graph_builds_and_serializes() {
    let graph = build_inference_graph();
    assert_eq!(graph.nodes.len(), 5);
    assert_eq!(graph.edges.len(), 4);

    let binary_data = binary::to_binary(&graph);
    assert_eq!(&binary_data[0..4], &[0x51, 0x4C, 0x42, 0x47], "must have QLBG magic");

    let restored = binary::from_binary(&binary_data).expect("graph must deserialize");
    assert_eq!(restored.nodes.len(), graph.nodes.len());
    assert_eq!(restored.edges.len(), graph.edges.len());
}

#[test]
fn executable_graph_runs_end_to_end() {
    let graph = build_inference_graph();

    let mut inputs = HashMap::new();
    inputs.insert(
        "x".to_string(),
        TensorData::from_f32(Shape::matrix(2, 2), &[1.0, -2.0, 3.0, 4.0]),
    );
    inputs.insert(
        "w".to_string(),
        TensorData::from_f32(Shape::matrix(2, 2), &[2.0, -1.0, 0.5, 1.5]),
    );

    let result = executor::execute(&graph, inputs).expect("graph must execute");
    let output = result.outputs["y"].as_f32_slice().expect("output must be f32");
    assert_eq!(output.len(), 4);
    assert_eq!(output, vec![1.0, 0.0, 8.0, 3.0]);
    assert_eq!(result.stats.nodes_executed, 5);
}
