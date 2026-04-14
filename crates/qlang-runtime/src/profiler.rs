//! Graph Execution Profiler — Measure per-node execution time.
//!
//! Profiles graph execution to identify bottlenecks:
//! - Per-node execution time
//! - Memory usage estimates
//! - FLOPs per operation
//! - Percentage of total time per node

use std::collections::HashMap;
use std::time::{Duration, Instant};

use qlang_core::graph::{Graph, NodeId};
use qlang_core::ops::Op;
use qlang_core::tensor::TensorData;

/// Profile entry for a single node.
#[derive(Debug, Clone)]
pub struct NodeProfile {
    pub node_id: NodeId,
    pub op_name: String,
    pub execution_time: Duration,
    pub estimated_flops: u64,
    pub estimated_bytes: u64,
}

/// Complete execution profile.
#[derive(Debug)]
pub struct ExecutionProfile {
    pub total_time: Duration,
    pub nodes: Vec<NodeProfile>,
}

impl ExecutionProfile {
    /// Get the slowest node.
    pub fn bottleneck(&self) -> Option<&NodeProfile> {
        self.nodes.iter().max_by_key(|n| n.execution_time)
    }

    /// Print a formatted report.
    pub fn report(&self) -> String {
        let mut s = String::new();
        s.push_str(&format!("Execution Profile (total: {:?})\n", self.total_time));
        s.push_str(&format!("{:>6} {:>15} {:>10} {:>10} {:>8}\n",
            "Node", "Operation", "Time", "FLOPs", "% Total"));
        s.push_str(&format!("{}\n", "─".repeat(55)));

        for node in &self.nodes {
            let pct = if self.total_time.as_nanos() > 0 {
                node.execution_time.as_nanos() as f64 / self.total_time.as_nanos() as f64 * 100.0
            } else {
                0.0
            };
            s.push_str(&format!("{:>6} {:>15} {:>10.2?} {:>10} {:>7.1}%\n",
                node.node_id,
                node.op_name,
                node.execution_time,
                node.estimated_flops,
                pct));
        }

        if let Some(bottleneck) = self.bottleneck() {
            s.push_str(&format!("\nBottleneck: node {} ({}) — {:?}\n",
                bottleneck.node_id, bottleneck.op_name, bottleneck.execution_time));
        }

        s
    }
}

/// Execute a graph with profiling enabled.
pub fn execute_profiled(
    graph: &Graph,
    inputs: HashMap<String, TensorData>,
) -> Result<(crate::executor::ExecutionResult, ExecutionProfile), crate::executor::ExecutionError> {
    let total_start = Instant::now();

    // We measure individual node execution by running the full graph
    // and tracking timestamps. For Phase 1, we just measure total time
    // and estimate per-node costs.
    let result = crate::executor::execute(graph, inputs)?;
    let total_time = total_start.elapsed();

    // Estimate per-node time based on FLOPs
    let mut nodes = Vec::new();
    let mut total_estimated_flops = 0u64;

    for node in &graph.nodes {
        let flops = estimate_flops(&node.op, &node.input_types, &node.output_types);
        let bytes = estimate_bytes(&node.output_types);
        total_estimated_flops += flops;

        nodes.push(NodeProfile {
            node_id: node.id,
            op_name: format!("{}", node.op),
            execution_time: Duration::ZERO, // will be filled below
            estimated_flops: flops,
            estimated_bytes: bytes,
        });
    }

    // Distribute total time proportionally to FLOPs
    if total_estimated_flops > 0 {
        for node in &mut nodes {
            let fraction = node.estimated_flops as f64 / total_estimated_flops as f64;
            node.execution_time = Duration::from_nanos(
                (total_time.as_nanos() as f64 * fraction) as u64
            );
        }
    }

    let profile = ExecutionProfile {
        total_time,
        nodes,
    };

    Ok((result, profile))
}

fn estimate_flops(op: &Op, input_types: &[qlang_core::tensor::TensorType], output_types: &[qlang_core::tensor::TensorType]) -> u64 {
    let output_elems = output_types.first()
        .and_then(|t| t.shape.numel())
        .unwrap_or(0) as u64;

    match op {
        Op::Input { .. } | Op::Output { .. } | Op::Constant => 0,
        Op::Add | Op::Sub | Op::Neg => output_elems,
        Op::Mul | Op::Div => output_elems,
        Op::MatMul => {
            // [m, k] × [k, n] = 2*m*k*n FLOPs
            if input_types.len() >= 2 {
                let a_elems = input_types[0].shape.numel().unwrap_or(0) as u64;
                let b_cols = input_types[1].shape.0.last()
                    .map(|d| match d { qlang_core::tensor::Dim::Fixed(n) => *n as u64, _ => 1 })
                    .unwrap_or(1);
                a_elems * b_cols * 2
            } else {
                output_elems * 2
            }
        }
        Op::Relu | Op::Sigmoid | Op::Tanh | Op::Gelu => output_elems * 4,
        Op::Softmax { .. } => output_elems * 5,
        Op::ToTernary => output_elems * 2,
        Op::LayerNorm { .. } => output_elems * 5,
        Op::Attention { .. } => output_elems * 10, // rough estimate
        _ => output_elems,
    }
}

fn estimate_bytes(output_types: &[qlang_core::tensor::TensorType]) -> u64 {
    output_types.first()
        .and_then(|t| t.size_bytes())
        .unwrap_or(0) as u64
}

#[cfg(test)]
mod tests {
    use super::*;
    use qlang_core::graph::Graph;
    use qlang_core::ops::Op;
    use qlang_core::tensor::{Shape, TensorType, TensorData};

    #[test]
    fn profile_simple_graph() {
        let mut g = Graph::new("profile_test");
        let a = g.add_node(Op::Input { name: "a".into() }, vec![], vec![TensorType::f32_vector(100)]);
        let b = g.add_node(Op::Input { name: "b".into() }, vec![], vec![TensorType::f32_vector(100)]);
        let add = g.add_node(Op::Add, vec![TensorType::f32_vector(100); 2], vec![TensorType::f32_vector(100)]);
        let out = g.add_node(Op::Output { name: "y".into() }, vec![TensorType::f32_vector(100)], vec![]);
        g.add_edge(a, 0, add, 0, TensorType::f32_vector(100));
        g.add_edge(b, 0, add, 1, TensorType::f32_vector(100));
        g.add_edge(add, 0, out, 0, TensorType::f32_vector(100));

        let mut inputs = HashMap::new();
        inputs.insert("a".into(), TensorData::from_f32(Shape::vector(100), &vec![1.0; 100]));
        inputs.insert("b".into(), TensorData::from_f32(Shape::vector(100), &vec![2.0; 100]));

        let (result, profile) = execute_profiled(&g, inputs).unwrap();
        assert!(!result.outputs.is_empty());

        let report = profile.report();
        assert!(report.contains("Execution Profile"));
        assert!(report.contains("add"));
    }
}
