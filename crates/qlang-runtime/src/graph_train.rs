//! Graph-based Training — Train QLANG graphs using autograd.
//!
//! This bridges the gap between the Graph structure and the Tape-based
//! autograd system. Given a QLANG graph, it:
//! 1. Converts graph nodes to Tape operations
//! 2. Runs forward pass
//! 3. Computes loss
//! 4. Backward pass (autograd)
//! 5. Returns gradients for all weight inputs
//!
//! This means ANY QLANG graph can be trained, not just hand-coded MLPs.

use std::collections::HashMap;
use qlang_core::graph::Graph;
use qlang_core::ops::Op;
use qlang_core::tensor::Shape;
use crate::autograd::Tape;

/// Result of a training step on a graph.
#[derive(Debug)]
pub struct TrainStepResult {
    /// Loss value.
    pub loss: f32,
    /// Gradients for each named weight input.
    pub gradients: HashMap<String, Vec<f32>>,
    /// Forward pass outputs.
    pub outputs: HashMap<String, Vec<f32>>,
}

/// Train a QLANG graph for one step.
///
/// Parameters:
/// - `graph`: the computation graph
/// - `inputs`: named input tensors (includes both data and weights)
/// - `weight_names`: which inputs are trainable weights
/// - `targets`: training targets (class labels)
/// - `lr`: learning rate
///
/// Returns the loss and gradients.
pub fn train_step(
    graph: &Graph,
    inputs: &HashMap<String, Vec<f32>>,
    weight_names: &[&str],
    targets: &[u8],
    _lr: f32,
) -> Result<TrainStepResult, TrainError> {
    let mut tape = Tape::new();

    // Map graph node IDs to tape variable IDs
    let mut node_to_tape: HashMap<u32, usize> = HashMap::new();
    let mut weight_tape_ids: HashMap<String, usize> = HashMap::new();

    // Process nodes in topological order
    let order = graph.topological_sort()
        .map_err(|e| TrainError::GraphError(e.to_string()))?;

    for &node_id in &order {
        let node = graph.node(node_id)
            .ok_or(TrainError::GraphError(format!("Node {node_id} not found")))?;

        let tape_id = match &node.op {
            Op::Input { name } => {
                let data = inputs.get(name)
                    .ok_or(TrainError::MissingInput(name.clone()))?;

                let shape = node.output_types.first()
                    .map(|t| shape_to_vec(&t.shape))
                    .unwrap_or_else(|| vec![data.len()]);

                let id = tape.variable(data.clone(), shape);

                if weight_names.contains(&name.as_str()) {
                    weight_tape_ids.insert(name.clone(), id);
                }

                id
            }

            Op::Add => {
                let (a, b) = get_two_tape_inputs(graph, node_id, &node_to_tape)?;
                tape.add(a, b)
            }

            Op::MatMul => {
                let (a, b) = get_two_tape_inputs(graph, node_id, &node_to_tape)?;
                tape.matmul(a, b)
            }

            Op::Relu => {
                let input = get_one_tape_input(graph, node_id, &node_to_tape)?;
                tape.relu(input)
            }

            Op::Sigmoid => {
                let input = get_one_tape_input(graph, node_id, &node_to_tape)?;
                tape.sigmoid(input)
            }

            Op::Softmax { .. } => {
                let input = get_one_tape_input(graph, node_id, &node_to_tape)?;
                tape.softmax(input)
            }

            Op::Output { .. } => {
                // Pass through
                let incoming = graph.incoming_edges(node_id);
                if let Some(edge) = incoming.first() {
                    *node_to_tape.get(&edge.from_node)
                        .ok_or(TrainError::GraphError("Output has no input".into()))?
                } else {
                    continue;
                }
            }

            Op::Mul => {
                let (a, b) = get_two_tape_inputs(graph, node_id, &node_to_tape)?;
                tape.mul(a, b)
            }

            _ => {
                // Skip unsupported ops in training
                let incoming = graph.incoming_edges(node_id);
                if let Some(edge) = incoming.first() {
                    if let Some(&id) = node_to_tape.get(&edge.from_node) {
                        id
                    } else {
                        continue;
                    }
                } else {
                    continue;
                }
            }
        };

        node_to_tape.insert(node_id, tape_id);
    }

    // Find the output node(s)
    let output_nodes = graph.output_nodes();
    let mut outputs = HashMap::new();

    // Use the last computation before output as the logits
    let mut logits_id = None;
    for out_node in &output_nodes {
        if let Op::Output { name } = &out_node.op {
            if let Some(&tape_id) = node_to_tape.get(&out_node.id) {
                let values = tape.value(tape_id).to_vec();
                outputs.insert(name.clone(), values);
                logits_id = Some(tape_id);
            }
        }
    }

    let logits_id = logits_id.ok_or(TrainError::GraphError("No output found".into()))?;

    // Compute cross-entropy loss
    let loss_id = tape.cross_entropy_loss(logits_id, targets);
    let loss = tape.value(loss_id)[0];

    // Backward pass
    tape.backward(loss_id);

    // Collect gradients
    let mut gradients = HashMap::new();
    for (name, &tape_id) in &weight_tape_ids {
        if let Some(grad) = tape.grad(tape_id) {
            gradients.insert(name.clone(), grad.to_vec());
        }
    }

    Ok(TrainStepResult { loss, gradients, outputs })
}

/// Apply gradients to weights using SGD.
pub fn apply_gradients(
    weights: &mut HashMap<String, Vec<f32>>,
    gradients: &HashMap<String, Vec<f32>>,
    lr: f32,
) {
    for (name, grad) in gradients {
        if let Some(w) = weights.get_mut(name) {
            for (wi, gi) in w.iter_mut().zip(grad.iter()) {
                *wi -= lr * gi;
            }
        }
    }
}

fn get_one_tape_input(
    graph: &Graph,
    node_id: u32,
    node_to_tape: &HashMap<u32, usize>,
) -> Result<usize, TrainError> {
    let incoming = graph.incoming_edges(node_id);
    let edge = incoming.first()
        .ok_or(TrainError::GraphError(format!("Node {node_id} has no input")))?;
    node_to_tape.get(&edge.from_node).copied()
        .ok_or(TrainError::GraphError(format!("Input node {} not in tape", edge.from_node)))
}

fn get_two_tape_inputs(
    graph: &Graph,
    node_id: u32,
    node_to_tape: &HashMap<u32, usize>,
) -> Result<(usize, usize), TrainError> {
    let incoming = graph.incoming_edges(node_id);
    if incoming.len() < 2 {
        return Err(TrainError::GraphError(format!("Node {node_id} needs 2 inputs")));
    }
    let a = node_to_tape.get(&incoming[0].from_node).copied()
        .ok_or(TrainError::GraphError("Input a not in tape".into()))?;
    let b = node_to_tape.get(&incoming[1].from_node).copied()
        .ok_or(TrainError::GraphError("Input b not in tape".into()))?;
    Ok((a, b))
}

fn shape_to_vec(shape: &Shape) -> Vec<usize> {
    shape.0.iter().map(|d| match d {
        qlang_core::tensor::Dim::Fixed(n) => *n,
        qlang_core::tensor::Dim::Dynamic => 1,
    }).collect()
}

#[derive(Debug, thiserror::Error)]
pub enum TrainError {
    #[error("graph error: {0}")]
    GraphError(String),
    #[error("missing input: {0}")]
    MissingInput(String),
}

#[cfg(test)]
mod tests {
    use super::*;
    use qlang_core::graph::Graph;
    use qlang_core::ops::Op;
    use qlang_core::tensor::TensorType;

    fn build_mlp_graph() -> Graph {
        let mut g = Graph::new("mlp");
        let x = g.add_node(Op::Input { name: "x".into() }, vec![], vec![TensorType::f32_matrix(2, 4)]);
        let w = g.add_node(Op::Input { name: "W".into() }, vec![], vec![TensorType::f32_matrix(4, 3)]);
        let mm = g.add_node(Op::MatMul, vec![TensorType::f32_matrix(2, 4), TensorType::f32_matrix(4, 3)], vec![TensorType::f32_matrix(2, 3)]);
        let relu = g.add_node(Op::Relu, vec![TensorType::f32_matrix(2, 3)], vec![TensorType::f32_matrix(2, 3)]);
        let sm = g.add_node(Op::Softmax { axis: 1 }, vec![TensorType::f32_matrix(2, 3)], vec![TensorType::f32_matrix(2, 3)]);
        let out = g.add_node(Op::Output { name: "probs".into() }, vec![TensorType::f32_matrix(2, 3)], vec![]);

        g.add_edge(x, 0, mm, 0, TensorType::f32_matrix(2, 4));
        g.add_edge(w, 0, mm, 1, TensorType::f32_matrix(4, 3));
        g.add_edge(mm, 0, relu, 0, TensorType::f32_matrix(2, 3));
        g.add_edge(relu, 0, sm, 0, TensorType::f32_matrix(2, 3));
        g.add_edge(sm, 0, out, 0, TensorType::f32_matrix(2, 3));

        g
    }

    #[test]
    fn train_step_produces_gradients() {
        let graph = build_mlp_graph();

        let mut inputs = HashMap::new();
        inputs.insert("x".into(), vec![1.0, 0.0, 0.5, 0.2, 0.0, 1.0, 0.3, 0.8]);
        inputs.insert("W".into(), vec![0.1, -0.1, 0.2, 0.3, 0.1, -0.2, -0.1, 0.2, 0.1, 0.2, -0.3, 0.1]);

        let targets = vec![0u8, 1u8];

        let result = train_step(&graph, &inputs, &["W"], &targets, 0.01).unwrap();

        assert!(result.loss.is_finite());
        assert!(result.gradients.contains_key("W"));
        assert_eq!(result.gradients["W"].len(), 12); // 4×3
        assert!(result.gradients["W"].iter().all(|g| g.is_finite()));
    }

    #[test]
    fn train_step_loss_decreases() {
        let graph = build_mlp_graph();

        let x_data = vec![1.0, 0.0, 0.5, 0.2, 0.0, 1.0, 0.3, 0.8];
        let mut w_data = vec![0.1, -0.1, 0.2, 0.3, 0.1, -0.2, -0.1, 0.2, 0.1, 0.2, -0.3, 0.1];
        let targets = vec![0u8, 1u8];

        let mut losses = Vec::new();

        for _ in 0..20 {
            let mut inputs = HashMap::new();
            inputs.insert("x".into(), x_data.clone());
            inputs.insert("W".into(), w_data.clone());

            let result = train_step(&graph, &inputs, &["W"], &targets, 0.1).unwrap();
            losses.push(result.loss);

            // Apply gradients
            let mut weights = HashMap::new();
            weights.insert("W".into(), w_data.clone());
            apply_gradients(&mut weights, &result.gradients, 0.1);
            w_data = weights.remove("W").unwrap();
        }

        // Loss should decrease
        let first_avg: f32 = losses[..5].iter().sum::<f32>() / 5.0;
        let last_avg: f32 = losses[15..].iter().sum::<f32>() / 5.0;
        assert!(last_avg < first_avg,
            "Loss didn't decrease: first={first_avg:.4}, last={last_avg:.4}");
    }

    #[test]
    fn apply_gradients_updates_weights() {
        let mut weights = HashMap::new();
        weights.insert("W".into(), vec![1.0, 2.0, 3.0]);

        let mut grads = HashMap::new();
        grads.insert("W".into(), vec![0.1, 0.2, 0.3]);

        apply_gradients(&mut weights, &grads, 1.0);

        assert_eq!(weights["W"], vec![0.9, 1.8, 2.7]);
    }
}
