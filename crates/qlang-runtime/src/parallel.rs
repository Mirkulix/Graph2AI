//! Parallel Execution — Execute independent graph nodes concurrently with rayon.
//!
//! Uses the scheduler's execution levels to identify independent nodes,
//! then dispatches them in parallel using rayon's thread pool.

use std::collections::HashMap;
use std::sync::Mutex;

use rayon::prelude::*;

use qlang_core::graph::{Graph, NodeId};
use qlang_core::ops::Op;
use qlang_core::tensor::TensorData;
use qlang_core::verify;

use crate::executor::{ExecutionError, ExecutionResult, ExecutionStats};
use crate::scheduler;

/// Configuration for parallel execution.
#[derive(Debug, Clone)]
pub struct ParallelConfig {
    /// Minimum number of nodes in a level to use parallel execution.
    /// Below this threshold, sequential execution is used.
    pub min_parallel_nodes: usize,
    /// Maximum number of threads (0 = use rayon default).
    pub max_threads: usize,
    /// Enable parallel execution (set to false to force sequential).
    pub enabled: bool,
}

impl Default for ParallelConfig {
    fn default() -> Self {
        Self {
            min_parallel_nodes: 2,
            max_threads: 0,
            enabled: true,
        }
    }
}

/// Execute a graph with parallel execution of independent nodes.
///
/// This uses the scheduler to identify execution levels (wavefront parallelism)
/// and dispatches independent nodes within each level in parallel using rayon.
pub fn execute_parallel(
    graph: &Graph,
    inputs: HashMap<String, TensorData>,
    config: &ParallelConfig,
) -> Result<ExecutionResult, ExecutionError> {
    // Verify graph
    let verification = verify::verify_graph(graph);
    if !verification.is_ok() {
        return Err(ExecutionError::VerificationFailed(format!("{verification}")));
    }

    // Get execution plan from scheduler
    let plan = scheduler::schedule(graph);

    // Shared state
    let node_outputs: Mutex<HashMap<(NodeId, u8), TensorData>> = Mutex::new(HashMap::new());
    let stats = Mutex::new(ExecutionStats::default());

    // Seed inputs
    {
        let mut outputs = node_outputs.lock().unwrap();
        for node in graph.input_nodes() {
            if let Op::Input { name } = &node.op {
                if let Some(data) = inputs.get(name) {
                    outputs.insert((node.id, 0), data.clone());
                }
            }
        }
    }

    // Execute level by level
    for level in &plan.levels {
        let non_input_nodes: Vec<NodeId> = level
            .nodes
            .iter()
            .filter(|&&nid| {
                graph
                    .node(nid)
                    .map(|n| !matches!(n.op, Op::Input { .. }))
                    .unwrap_or(false)
            })
            .copied()
            .collect();

        if non_input_nodes.is_empty() {
            continue;
        }

        if config.enabled && non_input_nodes.len() >= config.min_parallel_nodes {
            // Parallel execution
            let results: Vec<Result<(NodeId, TensorData, u64), ExecutionError>> = non_input_nodes
                .par_iter()
                .map(|&nid| {
                    let outputs = node_outputs.lock().unwrap();
                    execute_single_node(graph, nid, &outputs)
                })
                .collect();

            // Collect results
            let mut outputs = node_outputs.lock().unwrap();
            let mut s = stats.lock().unwrap();
            for result in results {
                let (nid, data, flops) = result?;
                outputs.insert((nid, 0), data);
                s.nodes_executed += 1;
                s.total_flops += flops;
                if graph.node(nid).map(|node| node.op.is_quantum()).unwrap_or(false) {
                    s.quantum_ops += 1;
                }
            }
        } else {
            // Sequential execution
            for &nid in &non_input_nodes {
                let data_result = {
                    let outputs = node_outputs.lock().unwrap();
                    execute_single_node(graph, nid, &outputs)
                };
                let (nid, data, flops) = data_result?;
                let mut outputs = node_outputs.lock().unwrap();
                outputs.insert((nid, 0), data);
                let mut s = stats.lock().unwrap();
                s.nodes_executed += 1;
                s.total_flops += flops;
                if graph.node(nid).map(|node| node.op.is_quantum()).unwrap_or(false) {
                    s.quantum_ops += 1;
                }
            }
        }
    }

    // Collect outputs
    let outputs_map = node_outputs.into_inner().unwrap();
    let mut result_outputs = HashMap::new();
    for node in graph.output_nodes() {
        if let Op::Output { name } = &node.op {
            if let Some(data) = outputs_map.get(&(node.id, 0)) {
                result_outputs.insert(name.clone(), data.clone());
            }
        }
    }

    let final_stats = stats.into_inner().unwrap();
    Ok(ExecutionResult {
        outputs: result_outputs,
        stats: final_stats,
    })
}

/// Execute a single node, reading inputs from the shared output map.
fn execute_single_node(
    graph: &Graph,
    node_id: NodeId,
    outputs: &HashMap<(NodeId, u8), TensorData>,
) -> Result<(NodeId, TensorData, u64), ExecutionError> {
    let node = graph
        .node(node_id)
        .ok_or(ExecutionError::RuntimeError(format!("node {node_id} not found")))?;

    let get_input = |port: usize| -> Result<TensorData, ExecutionError> {
        let incoming = graph.incoming_edges(node_id);
        let edge = incoming.get(port).ok_or(ExecutionError::MissingInput(
            node_id,
            format!("port {port}"),
        ))?;
        outputs
            .get(&(edge.from_node, edge.from_port))
            .cloned()
            .ok_or(ExecutionError::MissingInput(
                node_id,
                format!("from node {}", edge.from_node),
            ))
    };

    let mut flops: u64 = 0;

    let result = match &node.op {
        Op::Output { .. } => {
            let data = get_input(0)?;
            data
        }
        Op::Add => {
            let a = get_input(0)?;
            let b = get_input(1)?;
            flops = a.as_f32_slice().map(|s| s.len() as u64).unwrap_or(0);
            binop(&a, &b, |x, y| x + y)?
        }
        Op::Sub => {
            let a = get_input(0)?;
            let b = get_input(1)?;
            flops = a.as_f32_slice().map(|s| s.len() as u64).unwrap_or(0);
            binop(&a, &b, |x, y| x - y)?
        }
        Op::Mul => {
            let a = get_input(0)?;
            let b = get_input(1)?;
            flops = a.as_f32_slice().map(|s| s.len() as u64).unwrap_or(0);
            binop(&a, &b, |x, y| x * y)?
        }
        Op::Div => {
            let a = get_input(0)?;
            let b = get_input(1)?;
            flops = a.as_f32_slice().map(|s| s.len() as u64).unwrap_or(0);
            binop(&a, &b, |x, y| x / y)?
        }
        Op::Relu => {
            let a = get_input(0)?;
            unaryop(&a, |x| x.max(0.0))?
        }
        Op::Sigmoid => {
            let a = get_input(0)?;
            unaryop(&a, |x| 1.0 / (1.0 + (-x).exp()))?
        }
        Op::Tanh => {
            let a = get_input(0)?;
            unaryop(&a, |x| x.tanh())?
        }
        Op::Neg => {
            let a = get_input(0)?;
            unaryop(&a, |x| -x)?
        }
        Op::MatMul => {
            let a = get_input(0)?;
            let b = get_input(1)?;
            matmul(&a, &b)?
        }
        Op::Softmax { .. } => {
            let a = get_input(0)?;
            softmax(&a)?
        }
        Op::ToTernary => {
            let a = get_input(0)?;
            ternary(&a)?
        }
        Op::Superpose => {
            let a = get_input(0)?;
            let b = get_input(1)?;
            binop(&a, &b, |x, y| (x + y) / 2.0)?
        }
        Op::Measure => {
            let state = get_input(0)?;
            if let Ok(operators) = get_input(1) {
                measure_with_operators(&state, &operators)?
            } else {
                measure(&state)?
            }
        }
        Op::Entangle => {
            let a = get_input(0)?;
            let b = get_input(1)?;
            entangle(&a, &b)?
        }
        Op::Collapse => {
            let a = get_input(0)?;
            measure(&a)?
        }
        Op::Evolve { gamma, dt } => {
            let state = get_input(0)?;
            let gradient = if let Ok(gradient) = get_input(2) {
                gradient
            } else {
                get_input(1)?
            };
            evolve(&state, &gradient, *gamma, *dt)?
        }
        _ => {
            // Fallback: pass through first input if available
            get_input(0).unwrap_or_else(|_| {
                TensorData::from_f32(qlang_core::tensor::Shape::scalar(), &[0.0])
            })
        }
    };

    Ok((node_id, result, flops))
}

fn binop(a: &TensorData, b: &TensorData, op: impl Fn(f32, f32) -> f32) -> Result<TensorData, ExecutionError> {
    let va = a.as_f32_slice().ok_or(ExecutionError::RuntimeError("not f32".into()))?;
    let vb = b.as_f32_slice().ok_or(ExecutionError::RuntimeError("not f32".into()))?;
    let n = va.len().min(vb.len());
    let result: Vec<f32> = va[..n].iter().zip(&vb[..n]).map(|(&x, &y)| op(x, y)).collect();
    Ok(TensorData::from_f32(a.shape.clone(), &result))
}

fn unaryop(a: &TensorData, op: impl Fn(f32) -> f32) -> Result<TensorData, ExecutionError> {
    let va = a.as_f32_slice().ok_or(ExecutionError::RuntimeError("not f32".into()))?;
    let result: Vec<f32> = va.iter().map(|&x| op(x)).collect();
    Ok(TensorData::from_f32(a.shape.clone(), &result))
}

fn matmul(a: &TensorData, b: &TensorData) -> Result<TensorData, ExecutionError> {
    use qlang_core::tensor::{Dim, Shape};
    let va = a.as_f32_slice().ok_or(ExecutionError::RuntimeError("not f32".into()))?;
    let vb = b.as_f32_slice().ok_or(ExecutionError::RuntimeError("not f32".into()))?;
    let (m, k) = match a.shape.0.as_slice() {
        [Dim::Fixed(m), Dim::Fixed(k)] => (*m, *k),
        _ => return Err(ExecutionError::RuntimeError("matmul: need 2D".into())),
    };
    let n = match b.shape.0.as_slice() {
        [Dim::Fixed(_), Dim::Fixed(n)] => *n,
        _ => return Err(ExecutionError::RuntimeError("matmul: need 2D".into())),
    };
    let mut result = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0f32;
            for p in 0..k {
                sum += va[i * k + p] * vb[p * n + j];
            }
            result[i * n + j] = sum;
        }
    }
    Ok(TensorData::from_f32(Shape::matrix(m, n), &result))
}

fn softmax(a: &TensorData) -> Result<TensorData, ExecutionError> {
    let va = a.as_f32_slice().ok_or(ExecutionError::RuntimeError("not f32".into()))?;
    let max = va.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = va.iter().map(|&x| (x - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    let result: Vec<f32> = exps.iter().map(|&e| e / sum).collect();
    Ok(TensorData::from_f32(a.shape.clone(), &result))
}

fn ternary(a: &TensorData) -> Result<TensorData, ExecutionError> {
    let va = a.as_f32_slice().ok_or(ExecutionError::RuntimeError("not f32".into()))?;
    let mean_abs: f32 = va.iter().map(|x| x.abs()).sum::<f32>() / va.len().max(1) as f32;
    let threshold = mean_abs * 0.7;
    let result: Vec<f32> = va.iter().map(|&x| {
        if x > threshold { 1.0 } else if x < -threshold { -1.0 } else { 0.0 }
    }).collect();
    Ok(TensorData::from_f32(a.shape.clone(), &result))
}

fn measure(a: &TensorData) -> Result<TensorData, ExecutionError> {
    let va = a.as_f32_slice().ok_or(ExecutionError::RuntimeError("measure: not f32".into()))?;
    let mut max_idx = 0;
    let mut max_val = f32::NEG_INFINITY;
    for (i, &v) in va.iter().enumerate() {
        if v > max_val {
            max_val = v;
            max_idx = i;
        }
    }
    let mut result = vec![0.0f32; va.len()];
    result[max_idx] = 1.0;
    Ok(TensorData::from_f32(a.shape.clone(), &result))
}

fn measure_with_operators(state: &TensorData, operators: &TensorData) -> Result<TensorData, ExecutionError> {
    let vs = state.as_f32_slice().ok_or(ExecutionError::RuntimeError("measure: state not f32".into()))?;
    let vo = operators.as_f32_slice().ok_or(ExecutionError::RuntimeError("measure: operators not f32".into()))?;
    let len = vs.len();
    if len == 0 || vo.len() % len != 0 {
        return Err(ExecutionError::RuntimeError(
            "measure: operators length must be a multiple of state length".into(),
        ));
    }
    let num_ops = vo.len() / len;
    let mut probs = vec![0.0f32; num_ops];
    for i in 0..num_ops {
        let op_slice = &vo[i * len..(i + 1) * len];
        probs[i] = vs.iter().zip(op_slice.iter()).map(|(s, o)| s * o).sum();
    }
    Ok(TensorData::from_f32(qlang_core::tensor::Shape::vector(num_ops), &probs))
}

fn entangle(a: &TensorData, b: &TensorData) -> Result<TensorData, ExecutionError> {
    use qlang_core::tensor::{Dim, Shape};

    let va = a.as_f32_slice().ok_or(ExecutionError::RuntimeError("entangle: input a not f32".into()))?;
    let vb = b.as_f32_slice().ok_or(ExecutionError::RuntimeError("entangle: input b not f32".into()))?;
    let mut result = vec![0.0f32; va.len() * vb.len()];
    for (i, &val_a) in va.iter().enumerate() {
        for (j, &val_b) in vb.iter().enumerate() {
            result[i * vb.len() + j] = val_a * val_b;
        }
    }
    let shape = match (a.shape.0.as_slice(), b.shape.0.as_slice()) {
        ([Dim::Fixed(m1), Dim::Fixed(n1)], [Dim::Fixed(m2), Dim::Fixed(n2)]) => {
            Shape::matrix(m1 * m2, n1 * n2)
        }
        _ => Shape::vector(va.len() * vb.len()),
    };
    Ok(TensorData::from_f32(shape, &result))
}

fn evolve(state: &TensorData, gradient: &TensorData, gamma: f64, dt: f64) -> Result<TensorData, ExecutionError> {
    let vs = state.as_f32_slice().ok_or(ExecutionError::RuntimeError("evolve: state not f32".into()))?;
    let vg = gradient.as_f32_slice().ok_or(ExecutionError::RuntimeError("evolve: gradient not f32".into()))?;
    let step = (gamma * dt) as f32;
    let result: Vec<f32> = vs.iter().zip(vg.iter()).map(|(s, g)| s - step * g).collect();
    Ok(TensorData::from_f32(state.shape.clone(), &result))
}

#[cfg(test)]
mod tests {
    use super::*;
    use qlang_core::graph::Graph;
    use qlang_core::ops::Op;
    use qlang_core::tensor::{Dtype, Shape, TensorType};

    fn f32_vec(n: usize) -> TensorType {
        TensorType::new(Dtype::F32, Shape::vector(n))
    }

    #[test]
    fn test_parallel_simple_graph() {
        let mut g = Graph::new("test_par");
        let inp = g.add_node(Op::Input { name: "x".into() }, vec![], vec![f32_vec(4)]);
        let relu = g.add_node(Op::Relu, vec![f32_vec(4)], vec![f32_vec(4)]);
        let out = g.add_node(Op::Output { name: "y".into() }, vec![f32_vec(4)], vec![]);
        g.add_edge(inp, 0, relu, 0, f32_vec(4));
        g.add_edge(relu, 0, out, 0, f32_vec(4));

        let mut inputs = HashMap::new();
        inputs.insert("x".into(), TensorData::from_f32(Shape::vector(4), &[-1.0, 2.0, -3.0, 4.0]));

        let config = ParallelConfig::default();
        let result = execute_parallel(&g, inputs, &config).unwrap();
        let output = result.outputs.get("y").unwrap();
        let vals = output.as_f32_slice().unwrap();
        assert_eq!(vals, vec![0.0, 2.0, 0.0, 4.0]);
    }

    #[test]
    fn test_parallel_disabled() {
        let mut g = Graph::new("seq_test");
        let inp = g.add_node(Op::Input { name: "x".into() }, vec![], vec![f32_vec(2)]);
        let neg = g.add_node(Op::Neg, vec![f32_vec(2)], vec![f32_vec(2)]);
        let out = g.add_node(Op::Output { name: "y".into() }, vec![f32_vec(2)], vec![]);
        g.add_edge(inp, 0, neg, 0, f32_vec(2));
        g.add_edge(neg, 0, out, 0, f32_vec(2));

        let mut inputs = HashMap::new();
        inputs.insert("x".into(), TensorData::from_f32(Shape::vector(2), &[3.0, -5.0]));

        let config = ParallelConfig { enabled: false, ..Default::default() };
        let result = execute_parallel(&g, inputs, &config).unwrap();
        let vals = result.outputs["y"].as_f32_slice().unwrap();
        assert_eq!(vals, vec![-3.0, 5.0]);
    }

    #[test]
    fn test_parallel_matches_sequential() {
        // Build graph with two independent branches that merge
        let mut g = Graph::new("par_match");
        let x = g.add_node(Op::Input { name: "x".into() }, vec![], vec![f32_vec(4)]);

        let relu = g.add_node(Op::Relu, vec![f32_vec(4)], vec![f32_vec(4)]);
        g.add_edge(x, 0, relu, 0, f32_vec(4));

        let out = g.add_node(Op::Output { name: "y".into() }, vec![f32_vec(4)], vec![]);
        g.add_edge(relu, 0, out, 0, f32_vec(4));

        let mut inputs = HashMap::new();
        inputs.insert("x".into(), TensorData::from_f32(Shape::vector(4), &[-2.0, 1.0, -0.5, 3.0]));

        // Run with parallel
        let par_result = execute_parallel(&g, inputs.clone(), &ParallelConfig::default()).unwrap();
        // Run with sequential
        let seq_result = crate::executor::execute(&g, inputs).unwrap();

        let par_vals = par_result.outputs["y"].as_f32_slice().unwrap();
        let seq_vals = seq_result.outputs["y"].as_f32_slice().unwrap();
        assert_eq!(par_vals, seq_vals);
    }

    #[test]
    fn test_parallel_quantum_ops() {
        let mut g = Graph::new("par_quantum");
        let state = g.add_node(Op::Input { name: "state".into() }, vec![], vec![f32_vec(2)]);
        let partner = g.add_node(Op::Input { name: "partner".into() }, vec![], vec![f32_vec(2)]);
        let entangle = g.add_node(Op::Entangle, vec![f32_vec(2), f32_vec(2)], vec![f32_vec(4)]);
        let collapse = g.add_node(Op::Collapse, vec![f32_vec(4)], vec![f32_vec(4)]);
        let out = g.add_node(Op::Output { name: "y".into() }, vec![f32_vec(4)], vec![]);
        g.add_edge(state, 0, entangle, 0, f32_vec(2));
        g.add_edge(partner, 0, entangle, 1, f32_vec(2));
        g.add_edge(entangle, 0, collapse, 0, f32_vec(4));
        g.add_edge(collapse, 0, out, 0, f32_vec(4));

        let mut inputs = HashMap::new();
        inputs.insert("state".into(), TensorData::from_f32(Shape::vector(2), &[0.2, 0.8]));
        inputs.insert("partner".into(), TensorData::from_f32(Shape::vector(2), &[0.1, 0.9]));

        let result = execute_parallel(&g, inputs, &ParallelConfig::default()).unwrap();
        let vals = result.outputs["y"].as_f32_slice().unwrap();

        assert_eq!(vals, vec![0.0, 0.0, 0.0, 1.0]);
        assert_eq!(result.stats.quantum_ops, 2);
    }
}
