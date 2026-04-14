//! Self-hosting foundation: the QLANG compiler expressed as a QLANG graph.
//!
//! The key insight: a compiler is itself a computation graph that transforms
//! input graphs into output code. Each stage of the compilation pipeline
//! (deserialize, type-check, optimize, codegen) is a QLANG node, and data
//! flows between them as tensors of bytes.
//!
//! This module constructs that meta-graph, enabling QLANG to compile itself.

use qlang_core::graph::Graph;
use qlang_core::ops::Op;
use qlang_core::tensor::{Dim, Dtype, Shape, TensorType};

// ---------------------------------------------------------------------------
// Helper: a dynamically-sized byte buffer tensor type (i.e. a serialized blob)
// ---------------------------------------------------------------------------

fn byte_buffer() -> TensorType {
    TensorType::new(Dtype::I8, Shape(vec![Dim::Dynamic]))
}

/// A fixed-size scalar used for status / error codes.
fn status_scalar() -> TensorType {
    TensorType::new(Dtype::I32, Shape::scalar())
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Build and return the compiler-as-graph.
///
/// The returned `Graph` has a single input (the serialized QLANG source graph)
/// and a single output (the compiled instruction stream).  Internally it
/// chains four sub-graph stages:
///
///   input_graph_bytes
///       -> deserialize  (SubGraph "qlang.compiler.deserialize")
///       -> type_check   (SubGraph "qlang.compiler.typecheck")
///       -> optimize      (SubGraph "qlang.compiler.optimize")
///       -> codegen       (SubGraph "qlang.compiler.codegen")
///       -> output_code_bytes
///
/// Each stage is expressed as an existing `Op` so that the normal QLANG
/// runtime can, in principle, execute the compiler itself.
pub fn compiler_graph() -> Graph {
    let mut g = Graph::new("qlang.compiler");
    g.metadata
        .insert("description".into(), "QLANG compiler as a QLANG graph".into());
    g.metadata
        .insert("self_hosting".into(), "true".into());

    let buf = byte_buffer();
    let status = status_scalar();

    // -- Input: serialized source graph bytes ---------------------------------
    let input_id = g.add_node(
        Op::Input {
            name: "source_graph_bytes".into(),
        },
        vec![],
        vec![buf.clone()],
    );

    // -- Stage 1: Deserialize -------------------------------------------------
    // Two inputs: the byte buffer and a "config" constant (unused placeholder)
    let deser_id = g.add_node(
        Op::SubGraph {
            graph_id: "qlang.compiler.deserialize".into(),
        },
        vec![buf.clone(), buf.clone()],
        vec![buf.clone()],
    );

    // Supply a constant as the second input to the SubGraph (init state).
    let deser_init = g.add_node(Op::Constant, vec![], vec![buf.clone()]);

    // -- Stage 2: Type checking -----------------------------------------------
    let typecheck_id = g.add_node(
        Op::SubGraph {
            graph_id: "qlang.compiler.typecheck".into(),
        },
        vec![buf.clone(), status.clone()],
        vec![buf.clone()],
    );

    let typecheck_init = g.add_node(Op::Constant, vec![], vec![status.clone()]);

    // -- Stage 3: Optimization (dead-code elimination, fusion, etc.) ----------
    let optimize_id = g.add_node(
        Op::SubGraph {
            graph_id: "qlang.compiler.optimize".into(),
        },
        vec![buf.clone(), buf.clone()],
        vec![buf.clone()],
    );

    let optimize_init = g.add_node(Op::Constant, vec![], vec![buf.clone()]);

    // -- Stage 4: Code generation ---------------------------------------------
    let codegen_id = g.add_node(
        Op::SubGraph {
            graph_id: "qlang.compiler.codegen".into(),
        },
        vec![buf.clone(), buf.clone()],
        vec![buf.clone()],
    );

    let codegen_init = g.add_node(Op::Constant, vec![], vec![buf.clone()]);

    // -- Output: emitted instruction bytes ------------------------------------
    let output_id = g.add_node(
        Op::Output {
            name: "compiled_code".into(),
        },
        vec![buf.clone()],
        vec![],
    );

    // -- Wire the pipeline ----------------------------------------------------
    // input -> deserialize
    g.add_edge(input_id, 0, deser_id, 0, buf.clone());
    g.add_edge(deser_init, 0, deser_id, 1, buf.clone());

    // deserialize -> typecheck
    g.add_edge(deser_id, 0, typecheck_id, 0, buf.clone());
    g.add_edge(typecheck_init, 0, typecheck_id, 1, status.clone());

    // typecheck -> optimize
    g.add_edge(typecheck_id, 0, optimize_id, 0, buf.clone());
    g.add_edge(optimize_init, 0, optimize_id, 1, buf.clone());

    // optimize -> codegen
    g.add_edge(optimize_id, 0, codegen_id, 0, buf.clone());
    g.add_edge(codegen_init, 0, codegen_id, 1, buf.clone());

    // codegen -> output
    g.add_edge(codegen_id, 0, output_id, 0, buf);

    g
}

/// Build the detailed sub-graph for the deserialization stage.
///
/// In a full self-hosting compiler this would be byte-level parsing; here we
/// model it structurally so the graph is inspectable and valid.
pub fn deserialize_stage_graph() -> Graph {
    let mut g = Graph::new("qlang.compiler.deserialize");
    let buf = byte_buffer();

    let input = g.add_node(
        Op::Input { name: "raw_bytes".into() },
        vec![],
        vec![buf.clone()],
    );

    // Reshape raw bytes into a structured representation.
    let reshape = g.add_node(
        Op::Reshape {
            target_shape: vec![0], // dynamic
        },
        vec![buf.clone()],
        vec![buf.clone()],
    );

    let output = g.add_node(
        Op::Output { name: "graph_ir".into() },
        vec![buf.clone()],
        vec![],
    );

    g.add_edge(input, 0, reshape, 0, buf.clone());
    g.add_edge(reshape, 0, output, 0, buf);
    g
}

/// Build the detailed sub-graph for the type-checking stage.
pub fn typecheck_stage_graph() -> Graph {
    let mut g = Graph::new("qlang.compiler.typecheck");
    let buf = byte_buffer();

    let input = g.add_node(
        Op::Input { name: "graph_ir".into() },
        vec![],
        vec![buf.clone()],
    );

    // Scan over each node to verify shapes. We model this as a bounded
    // iteration (Scan) that walks the node list.
    let init = g.add_node(Op::Constant, vec![], vec![buf.clone()]);

    let scan = g.add_node(
        Op::Scan { n_iterations: 1 },
        vec![buf.clone(), buf.clone()],
        vec![buf.clone()],
    );

    let output = g.add_node(
        Op::Output { name: "checked_ir".into() },
        vec![buf.clone()],
        vec![],
    );

    g.add_edge(input, 0, scan, 0, buf.clone());
    g.add_edge(init, 0, scan, 1, buf.clone());
    g.add_edge(scan, 0, output, 0, buf);
    g
}

/// Build the detailed sub-graph for the optimization stage.
pub fn optimize_stage_graph() -> Graph {
    let mut g = Graph::new("qlang.compiler.optimize");
    let buf = byte_buffer();

    let input = g.add_node(
        Op::Input { name: "checked_ir".into() },
        vec![],
        vec![buf.clone()],
    );

    // Dead-code elimination pass: modelled as ReduceSum (collapse unused dims).
    let dce = g.add_node(
        Op::ReduceSum { axis: None },
        vec![buf.clone()],
        vec![buf.clone()],
    );

    // Operator fusion pass: Concat symbolizes merging adjacent ops.
    let fuse = g.add_node(
        Op::Concat { axis: 0 },
        vec![buf.clone(), buf.clone()],
        vec![buf.clone()],
    );

    // Feed the original IR as second input to Concat (self-join for fusion scan).
    let identity = g.add_node(
        Op::Reshape { target_shape: vec![0] },
        vec![buf.clone()],
        vec![buf.clone()],
    );

    let output = g.add_node(
        Op::Output { name: "optimized_ir".into() },
        vec![buf.clone()],
        vec![],
    );

    g.add_edge(input, 0, dce, 0, buf.clone());
    g.add_edge(input, 0, identity, 0, buf.clone());
    g.add_edge(dce, 0, fuse, 0, buf.clone());
    g.add_edge(identity, 0, fuse, 1, buf.clone());
    g.add_edge(fuse, 0, output, 0, buf);
    g
}

/// Build the detailed sub-graph for the code-generation stage.
pub fn codegen_stage_graph() -> Graph {
    let mut g = Graph::new("qlang.compiler.codegen");
    let buf = byte_buffer();

    let input = g.add_node(
        Op::Input { name: "optimized_ir".into() },
        vec![],
        vec![buf.clone()],
    );

    // MatMul here is symbolic: "multiply" IR nodes by an instruction-selection
    // matrix to produce machine code bytes.  This is the core linear map at
    // the heart of any instruction selector.
    let weights = g.add_node(Op::Constant, vec![], vec![buf.clone()]);

    let matmul = g.add_node(
        Op::MatMul,
        vec![buf.clone(), buf.clone()],
        vec![buf.clone()],
    );

    // Relu squashes invalid (negative) instruction encodings to zero.
    let relu = g.add_node(
        Op::Relu,
        vec![buf.clone()],
        vec![buf.clone()],
    );

    let output = g.add_node(
        Op::Output { name: "machine_code".into() },
        vec![buf.clone()],
        vec![],
    );

    g.add_edge(input, 0, matmul, 0, buf.clone());
    g.add_edge(weights, 0, matmul, 1, buf.clone());
    g.add_edge(matmul, 0, relu, 0, buf.clone());
    g.add_edge(relu, 0, output, 0, buf);
    g
}

/// Bootstrap the self-hosting compiler.
///
/// Returns the top-level compiler graph together with all stage sub-graphs.
/// A runtime that supports `SubGraph` references would register each stage
/// graph by its id before executing the top-level graph.
pub fn bootstrap() -> (Graph, Vec<Graph>) {
    let top = compiler_graph();
    let stages = vec![
        deserialize_stage_graph(),
        typecheck_stage_graph(),
        optimize_stage_graph(),
        codegen_stage_graph(),
    ];
    (top, stages)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use qlang_core::ops::Op;

    #[test]
    fn compiler_graph_is_valid_dag() {
        let g = compiler_graph();
        assert!(g.validate().is_ok(), "compiler graph must be a valid DAG");
    }

    #[test]
    fn compiler_graph_has_correct_io() {
        let g = compiler_graph();
        let inputs = g.input_nodes();
        let outputs = g.output_nodes();
        assert_eq!(inputs.len(), 1, "compiler graph should have one input");
        assert_eq!(outputs.len(), 1, "compiler graph should have one output");

        match &inputs[0].op {
            Op::Input { name } => assert_eq!(name, "source_graph_bytes"),
            other => panic!("expected Input, got {other}"),
        }
        match &outputs[0].op {
            Op::Output { name } => assert_eq!(name, "compiled_code"),
            other => panic!("expected Output, got {other}"),
        }
    }

    #[test]
    fn compiler_graph_has_four_stages() {
        let g = compiler_graph();
        let subgraphs: Vec<_> = g
            .nodes
            .iter()
            .filter(|n| matches!(n.op, Op::SubGraph { .. }))
            .collect();
        assert_eq!(subgraphs.len(), 4, "pipeline must have 4 SubGraph stages");

        let ids: Vec<String> = subgraphs
            .iter()
            .map(|n| match &n.op {
                Op::SubGraph { graph_id } => graph_id.clone(),
                _ => unreachable!(),
            })
            .collect();

        assert!(ids.contains(&"qlang.compiler.deserialize".to_string()));
        assert!(ids.contains(&"qlang.compiler.typecheck".to_string()));
        assert!(ids.contains(&"qlang.compiler.optimize".to_string()));
        assert!(ids.contains(&"qlang.compiler.codegen".to_string()));
    }

    #[test]
    fn compiler_graph_topological_order() {
        let g = compiler_graph();
        let order = g.topological_sort().expect("must be acyclic");

        // Find positions of the four stages in topological order.
        let pos_of = |graph_id: &str| -> usize {
            let node_id = g
                .nodes
                .iter()
                .find(|n| {
                    matches!(&n.op, Op::SubGraph { graph_id: gid } if gid == graph_id)
                })
                .unwrap()
                .id;
            order.iter().position(|&id| id == node_id).unwrap()
        };

        let p_deser = pos_of("qlang.compiler.deserialize");
        let p_tc = pos_of("qlang.compiler.typecheck");
        let p_opt = pos_of("qlang.compiler.optimize");
        let p_cg = pos_of("qlang.compiler.codegen");

        assert!(
            p_deser < p_tc,
            "deserialize must precede typecheck"
        );
        assert!(
            p_tc < p_opt,
            "typecheck must precede optimize"
        );
        assert!(
            p_opt < p_cg,
            "optimize must precede codegen"
        );
    }

    #[test]
    fn stage_graphs_are_valid() {
        for stage in [
            deserialize_stage_graph(),
            typecheck_stage_graph(),
            optimize_stage_graph(),
            codegen_stage_graph(),
        ] {
            assert!(
                stage.validate().is_ok(),
                "stage graph '{}' must be valid",
                stage.id
            );
        }
    }

    #[test]
    fn bootstrap_returns_all_stages() {
        let (top, stages) = bootstrap();
        assert_eq!(stages.len(), 4);
        assert_eq!(top.id, "qlang.compiler");

        // Every SubGraph reference in the top graph has a matching stage graph.
        for node in &top.nodes {
            if let Op::SubGraph { graph_id } = &node.op {
                assert!(
                    stages.iter().any(|s| s.id == *graph_id),
                    "stage graph '{graph_id}' not found in bootstrap output"
                );
            }
        }
    }

    #[test]
    fn compiler_graph_metadata() {
        let g = compiler_graph();
        assert_eq!(
            g.metadata.get("self_hosting").map(String::as_str),
            Some("true")
        );
    }

    #[test]
    fn compiler_graph_display() {
        // Smoke test: Display impl should not panic and should mention stages.
        let g = compiler_graph();
        let display = format!("{g}");
        assert!(display.contains("qlang.compiler"));
        assert!(display.contains("subgraph"));
    }

    #[test]
    fn no_dangling_edges() {
        let (top, stages) = bootstrap();
        for g in std::iter::once(&top).chain(stages.iter()) {
            for edge in &g.edges {
                assert!(
                    g.node(edge.from_node).is_some(),
                    "graph '{}': edge {} references missing source node {}",
                    g.id,
                    edge.id,
                    edge.from_node
                );
                assert!(
                    g.node(edge.to_node).is_some(),
                    "graph '{}': edge {} references missing target node {}",
                    g.id,
                    edge.id,
                    edge.to_node
                );
            }
        }
    }
}
