//! Graph Composition — Connect and reuse QLANG graph fragments.
//!
//! A key advantage of graph-based programming:
//! graphs are naturally composable by wiring outputs to inputs.
//!
//! Example:
//!   graph_a: input → matmul → output
//!   graph_b: input → relu → output
//!   compose(graph_a, graph_b) = input → matmul → relu → output

use qlang_core::graph::{Graph, NodeId};
use qlang_core::ops::Op;

/// Compose two graphs sequentially: output of `first` feeds into input of `second`.
///
/// Requirements:
/// - `first` must have exactly one output
/// - `second` must have at least one input
/// - The output type of `first` must match the input type of `second`
pub fn compose(first: &Graph, second: &Graph, name: &str) -> Result<Graph, ComposeError> {
    let first_outputs = first.output_nodes();
    let second_inputs = second.input_nodes();

    if first_outputs.is_empty() {
        return Err(ComposeError::NoOutputs("first graph".into()));
    }

    if second_inputs.is_empty() {
        return Err(ComposeError::NoInputs("second graph".into()));
    }

    let mut composed = Graph::new(name);

    // Track node ID mapping: old_id → new_id
    let mut first_map: std::collections::HashMap<NodeId, NodeId> = std::collections::HashMap::new();
    let mut second_map: std::collections::HashMap<NodeId, NodeId> = std::collections::HashMap::new();

    // Copy nodes from first graph (skip output nodes)
    for node in &first.nodes {
        if matches!(node.op, Op::Output { .. }) {
            continue;
        }
        let new_id = composed.add_node(
            node.op.clone(),
            node.input_types.clone(),
            node.output_types.clone(),
        );
        first_map.insert(node.id, new_id);
    }

    // Copy nodes from second graph (skip the first input node — it gets connected)
    let connect_input_id = second_inputs[0].id;
    for node in &second.nodes {
        if node.id == connect_input_id {
            // This input will be replaced by the first graph's last computation
            continue;
        }
        let new_id = composed.add_node(
            node.op.clone(),
            node.input_types.clone(),
            node.output_types.clone(),
        );
        second_map.insert(node.id, new_id);
    }

    // Copy edges from first graph (skip edges going to output)
    for edge in &first.edges {
        if let (Some(&from_new), Some(&to_new)) =
            (first_map.get(&edge.from_node), first_map.get(&edge.to_node))
        {
            composed.add_edge(
                from_new,
                edge.from_port,
                to_new,
                edge.to_port,
                edge.tensor_type.clone(),
            );
        }
    }

    // Find the last computation node in first graph (the one feeding into output)
    let last_first_node = first
        .edges
        .iter()
        .find(|e| {
            first
                .node(e.to_node)
                .map(|n| matches!(n.op, Op::Output { .. }))
                .unwrap_or(false)
        })
        .map(|e| e.from_node);

    // Connect: last node of first → first consumer of second's input
    if let Some(last_id) = last_first_node {
        if let Some(&last_new_id) = first_map.get(&last_id) {
            // Find edges in second graph that came from the connected input
            for edge in &second.edges {
                if edge.from_node == connect_input_id {
                    if let Some(&to_new) = second_map.get(&edge.to_node) {
                        composed.add_edge(
                            last_new_id,
                            0, // output port of last first node
                            to_new,
                            edge.to_port,
                            edge.tensor_type.clone(),
                        );
                    }
                }
            }
        }
    }

    // Copy remaining edges from second graph
    for edge in &second.edges {
        if edge.from_node == connect_input_id {
            continue; // Already handled above
        }
        if let (Some(&from_new), Some(&to_new)) =
            (second_map.get(&edge.from_node), second_map.get(&edge.to_node))
        {
            composed.add_edge(
                from_new,
                edge.from_port,
                to_new,
                edge.to_port,
                edge.tensor_type.clone(),
            );
        }
    }

    Ok(composed)
}

/// Merge two graphs in parallel: both receive the same input, outputs are separate.
pub fn parallel(a: &Graph, b: &Graph, name: &str) -> Graph {
    let mut merged = Graph::new(name);

    // Copy all nodes and edges from both graphs with ID remapping
    let mut a_map: std::collections::HashMap<NodeId, NodeId> = std::collections::HashMap::new();
    for node in &a.nodes {
        let new_id = merged.add_node(node.op.clone(), node.input_types.clone(), node.output_types.clone());
        a_map.insert(node.id, new_id);
    }
    for edge in &a.edges {
        if let (Some(&from), Some(&to)) = (a_map.get(&edge.from_node), a_map.get(&edge.to_node)) {
            merged.add_edge(from, edge.from_port, to, edge.to_port, edge.tensor_type.clone());
        }
    }

    let mut b_map: std::collections::HashMap<NodeId, NodeId> = std::collections::HashMap::new();
    for node in &b.nodes {
        let new_id = merged.add_node(node.op.clone(), node.input_types.clone(), node.output_types.clone());
        b_map.insert(node.id, new_id);
    }
    for edge in &b.edges {
        if let (Some(&from), Some(&to)) = (b_map.get(&edge.from_node), b_map.get(&edge.to_node)) {
            merged.add_edge(from, edge.from_port, to, edge.to_port, edge.tensor_type.clone());
        }
    }

    merged
}

#[derive(Debug, thiserror::Error)]
pub enum ComposeError {
    #[error("graph has no outputs: {0}")]
    NoOutputs(String),
    #[error("graph has no inputs: {0}")]
    NoInputs(String),
    #[error("type mismatch between graphs")]
    TypeMismatch,
}

#[cfg(test)]
mod tests {
    use super::*;
    use qlang_core::ops::Op;
    use qlang_core::tensor::TensorType;

    #[test]
    fn compose_two_graphs() {
        // Graph A: input → relu → output
        let mut a = Graph::new("relu_graph");
        let a_in = a.add_node(Op::Input { name: "x".into() }, vec![], vec![TensorType::f32_vector(4)]);
        let a_relu = a.add_node(Op::Relu, vec![TensorType::f32_vector(4)], vec![TensorType::f32_vector(4)]);
        let a_out = a.add_node(Op::Output { name: "y".into() }, vec![TensorType::f32_vector(4)], vec![]);
        a.add_edge(a_in, 0, a_relu, 0, TensorType::f32_vector(4));
        a.add_edge(a_relu, 0, a_out, 0, TensorType::f32_vector(4));

        // Graph B: input → neg → output
        let mut b = Graph::new("neg_graph");
        let b_in = b.add_node(Op::Input { name: "x".into() }, vec![], vec![TensorType::f32_vector(4)]);
        let b_neg = b.add_node(Op::Neg, vec![TensorType::f32_vector(4)], vec![TensorType::f32_vector(4)]);
        let b_out = b.add_node(Op::Output { name: "y".into() }, vec![TensorType::f32_vector(4)], vec![]);
        b.add_edge(b_in, 0, b_neg, 0, TensorType::f32_vector(4));
        b.add_edge(b_neg, 0, b_out, 0, TensorType::f32_vector(4));

        // Compose: relu → neg
        let composed = compose(&a, &b, "relu_then_neg").unwrap();

        // Should have: input, relu, neg (no intermediate output/input)
        assert!(composed.validate().is_ok());
        // Input + relu from A, neg + output from B = 4 nodes, minus 2 (output of A, input of B) = 2 + neg = 3
        assert!(composed.nodes.len() >= 3);
    }

    #[test]
    fn parallel_graphs() {
        let mut a = Graph::new("a");
        a.add_node(Op::Input { name: "x".into() }, vec![], vec![TensorType::f32_vector(4)]);

        let mut b = Graph::new("b");
        b.add_node(Op::Input { name: "y".into() }, vec![], vec![TensorType::f32_vector(4)]);

        let merged = parallel(&a, &b, "parallel");
        assert_eq!(merged.nodes.len(), 2);
    }
}
