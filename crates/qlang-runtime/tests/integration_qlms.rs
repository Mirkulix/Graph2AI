//! Integration test: build a graph → encode (QLMS/QLAN binary) → decode →
//! verify structural equality. Uses qlang_core::serial as the QLMS codec.

use qlang_core::graph::Graph;
use qlang_core::ops::Op;
use qlang_core::serial::{from_binary, to_binary, MAGIC};
use qlang_core::tensor::TensorType;
use qlang_runtime::mnist::MnistData;

fn f32v(n: usize) -> TensorType {
    TensorType::f32_vector(n)
}
fn f32m(r: usize, c: usize) -> TensorType {
    TensorType::f32_matrix(r, c)
}

fn build_pipeline_graph() -> Graph {
    // A small realistic pipeline: Input -> MatMul -> Relu -> Output
    let mut g = Graph::new("qlms_pipeline");
    let x = g.add_node(Op::Input { name: "x".into() }, vec![], vec![f32v(16)]);
    let w = g.add_node(Op::Input { name: "w".into() }, vec![], vec![f32m(16, 8)]);
    let mm = g.add_node(Op::MatMul, vec![f32v(16), f32m(16, 8)], vec![f32v(8)]);
    let relu = g.add_node(Op::Relu, vec![f32v(8)], vec![f32v(8)]);
    let y = g.add_node(Op::Output { name: "y".into() }, vec![f32v(8)], vec![]);
    g.add_edge(x, 0, mm, 0, f32v(16));
    g.add_edge(w, 0, mm, 1, f32m(16, 8));
    g.add_edge(mm, 0, relu, 0, f32v(8));
    g.add_edge(relu, 0, y, 0, f32v(8));
    g
}

#[test]
fn integration_qlms_encode_decode_structural_equality() {
    // Ensure MNIST helper still works (pipeline sanity).
    let data = MnistData::synthetic(1000, 200);
    assert_eq!(data.image_size, 784);

    let graph = build_pipeline_graph();
    assert_eq!(graph.nodes.len(), 5);
    assert_eq!(graph.edges.len(), 4);

    // --- encode ---
    let bytes = to_binary(&graph).expect("encode_qlms failed");
    assert!(bytes.len() > 8);
    assert_eq!(&bytes[0..4], &MAGIC, "missing QLMS/QLAN magic bytes");
    let version = u16::from_le_bytes([bytes[4], bytes[5]]);
    assert_eq!(version, 1, "unexpected wire version");

    // --- decode ---
    let round = from_binary(&bytes).expect("decode_qlms failed");

    // --- structural equality ---
    assert_eq!(round.id, graph.id);
    assert_eq!(round.nodes.len(), graph.nodes.len());
    assert_eq!(round.edges.len(), graph.edges.len());
    // Graph derives PartialEq for complete structural comparison.
    assert_eq!(round, graph, "round-tripped graph differs structurally");

    // --- stability: encode(decode(encode(g))) == encode(g) ---
    let bytes2 = to_binary(&round).expect("re-encode failed");
    assert_eq!(bytes, bytes2, "encoding is not stable across roundtrips");

    // --- corrupt detection ---
    let mut bad = bytes.clone();
    bad[0] ^= 0xFF;
    assert!(from_binary(&bad).is_err(), "corrupted magic not rejected");

    println!(
        "QLMS roundtrip OK: {} bytes, {} nodes, {} edges",
        bytes.len(), round.nodes.len(), round.edges.len()
    );
}
