//! Graph Scheduler — Optimal execution ordering and parallelism detection.
//!
//! Given a QLANG graph, determines:
//! 1. Which nodes can execute in parallel (independent subgraphs)
//! 2. Optimal memory allocation (when to allocate/free tensors)
//! 3. Execution levels (wavefront parallelism)
//!
//! This is the foundation for multi-threaded and GPU execution.

use std::collections::{HashMap, HashSet, VecDeque};

use qlang_core::graph::{Graph, NodeId};
use qlang_core::ops::Op;

/// A scheduled execution plan.
#[derive(Debug)]
pub struct ExecutionPlan {
    /// Execution levels — nodes within a level can run in parallel.
    pub levels: Vec<ExecutionLevel>,
    /// Memory plan — when each tensor is allocated and freed.
    pub memory_plan: MemoryPlan,
    /// Total estimated FLOPs.
    pub total_flops: u64,
}

/// A set of nodes that can execute in parallel.
#[derive(Debug, Clone)]
pub struct ExecutionLevel {
    pub level: usize,
    pub nodes: Vec<NodeId>,
    pub estimated_flops: u64,
}

/// Memory allocation plan.
#[derive(Debug)]
pub struct MemoryPlan {
    /// For each edge: when its tensor is first needed and last used.
    pub tensor_lifetimes: Vec<TensorLifetime>,
    /// Peak memory usage in bytes.
    pub peak_memory_bytes: u64,
    /// Total memory allocated.
    pub total_allocated_bytes: u64,
}

/// Lifetime of a tensor in the execution plan.
#[derive(Debug, Clone)]
pub struct TensorLifetime {
    pub producer_node: NodeId,
    pub first_use_level: usize,
    pub last_use_level: usize,
    pub size_bytes: u64,
}

/// Create an execution plan for a graph.
pub fn schedule(graph: &Graph) -> ExecutionPlan {
    let levels = compute_levels(graph);
    let memory_plan = plan_memory(graph, &levels);

    let total_flops = levels.iter().map(|l| l.estimated_flops).sum();

    ExecutionPlan {
        levels,
        memory_plan,
        total_flops,
    }
}

/// Compute execution levels using wavefront parallelism.
/// Level 0: all input nodes (no dependencies)
/// Level N: nodes whose inputs are all in levels < N
fn compute_levels(graph: &Graph) -> Vec<ExecutionLevel> {
    let order = match graph.topological_sort() {
        Ok(o) => o,
        Err(_) => return Vec::new(),
    };

    // Compute the level of each node
    let mut node_level: HashMap<NodeId, usize> = HashMap::new();

    for &node_id in &order {
        let incoming = graph.incoming_edges(node_id);
        let level = if incoming.is_empty() {
            0
        } else {
            incoming.iter()
                .map(|e| node_level.get(&e.from_node).copied().unwrap_or(0) + 1)
                .max()
                .unwrap_or(0)
        };
        node_level.insert(node_id, level);
    }

    // Group by level
    let max_level = node_level.values().copied().max().unwrap_or(0);
    let mut levels = Vec::new();

    for level in 0..=max_level {
        let nodes: Vec<NodeId> = order.iter()
            .filter(|&&id| node_level.get(&id) == Some(&level))
            .copied()
            .collect();

        let estimated_flops: u64 = nodes.iter()
            .filter_map(|&id| graph.node(id))
            .map(|n| estimate_node_flops(&n.op, &n.output_types))
            .sum();

        if !nodes.is_empty() {
            levels.push(ExecutionLevel {
                level,
                nodes,
                estimated_flops,
            });
        }
    }

    levels
}

/// Plan memory allocation and deallocation.
fn plan_memory(graph: &Graph, levels: &[ExecutionLevel]) -> MemoryPlan {
    let mut lifetimes = Vec::new();
    let node_to_level: HashMap<NodeId, usize> = levels.iter()
        .flat_map(|l| l.nodes.iter().map(move |&n| (n, l.level)))
        .collect();

    for edge in &graph.edges {
        let producer_level = node_to_level.get(&edge.from_node).copied().unwrap_or(0);
        let consumer_level = node_to_level.get(&edge.to_node).copied().unwrap_or(0);

        let size_bytes = edge.tensor_type.size_bytes().unwrap_or(0) as u64;

        lifetimes.push(TensorLifetime {
            producer_node: edge.from_node,
            first_use_level: producer_level,
            last_use_level: consumer_level,
            size_bytes,
        });
    }

    // Compute peak memory: at each level, sum all live tensors
    let max_level = levels.len();
    let mut peak = 0u64;
    let total;

    for level in 0..max_level {
        let live_bytes: u64 = lifetimes.iter()
            .filter(|t| t.first_use_level <= level && t.last_use_level >= level)
            .map(|t| t.size_bytes)
            .sum();
        peak = peak.max(live_bytes);
    }

    total = lifetimes.iter().map(|t| t.size_bytes).sum();

    MemoryPlan {
        tensor_lifetimes: lifetimes,
        peak_memory_bytes: peak,
        total_allocated_bytes: total,
    }
}

fn estimate_node_flops(op: &Op, output_types: &[qlang_core::tensor::TensorType]) -> u64 {
    let n = output_types.first()
        .and_then(|t| t.shape.numel())
        .unwrap_or(0) as u64;

    match op {
        Op::Input { .. } | Op::Output { .. } | Op::Constant => 0,
        Op::Add | Op::Sub | Op::Neg => n,
        Op::Mul | Op::Div => n,
        Op::MatMul => n * 2, // rough
        Op::Relu => n,
        Op::Sigmoid | Op::Tanh | Op::Gelu => n * 4,
        Op::Softmax { .. } => n * 5,
        Op::ToTernary => n * 2,
        Op::Attention { .. } => n * 10,
        Op::LayerNorm { .. } => n * 5,
        _ => n,
    }
}

/// Find all independent subgraphs (connected components ignoring direction).
pub fn find_independent_subgraphs(graph: &Graph) -> Vec<Vec<NodeId>> {
    let mut visited = HashSet::new();
    let mut components = Vec::new();

    for node in &graph.nodes {
        if visited.contains(&node.id) {
            continue;
        }

        let mut component = Vec::new();
        let mut queue = VecDeque::new();
        queue.push_back(node.id);

        while let Some(id) = queue.pop_front() {
            if !visited.insert(id) {
                continue;
            }
            component.push(id);

            // Add neighbors (both directions)
            for edge in &graph.edges {
                if edge.from_node == id && !visited.contains(&edge.to_node) {
                    queue.push_back(edge.to_node);
                }
                if edge.to_node == id && !visited.contains(&edge.from_node) {
                    queue.push_back(edge.from_node);
                }
            }
        }

        component.sort();
        components.push(component);
    }

    components
}

impl ExecutionPlan {
    /// Format as a readable report.
    pub fn report(&self) -> String {
        let mut s = String::new();
        s.push_str("Execution Plan:\n");
        s.push_str(&format!("  Levels: {}\n", self.levels.len()));
        s.push_str(&format!("  Total FLOPs: {}\n", self.total_flops));
        s.push_str(&format!("  Peak memory: {:.1} KB\n",
            self.memory_plan.peak_memory_bytes as f64 / 1024.0));

        let max_parallel = self.levels.iter().map(|l| l.nodes.len()).max().unwrap_or(0);
        s.push_str(&format!("  Max parallelism: {} nodes\n\n", max_parallel));

        for level in &self.levels {
            s.push_str(&format!("  Level {} ({} nodes, {} FLOPs):\n",
                level.level, level.nodes.len(), level.estimated_flops));
            for &node_id in &level.nodes {
                s.push_str(&format!("    - node {}\n", node_id));
            }
        }

        s
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use qlang_core::graph::Graph;
    use qlang_core::ops::Op;
    use qlang_core::tensor::TensorType;

    #[test]
    fn schedule_linear_graph() {
        // a → relu → output (3 levels: 0, 1, 2)
        let mut g = Graph::new("linear");
        let a = g.add_node(Op::Input { name: "a".into() }, vec![], vec![TensorType::f32_vector(4)]);
        let relu = g.add_node(Op::Relu, vec![TensorType::f32_vector(4)], vec![TensorType::f32_vector(4)]);
        let out = g.add_node(Op::Output { name: "y".into() }, vec![TensorType::f32_vector(4)], vec![]);
        g.add_edge(a, 0, relu, 0, TensorType::f32_vector(4));
        g.add_edge(relu, 0, out, 0, TensorType::f32_vector(4));

        let plan = schedule(&g);
        assert_eq!(plan.levels.len(), 3);
        assert_eq!(plan.levels[0].nodes, vec![a]);
        assert_eq!(plan.levels[1].nodes, vec![relu]);
        assert_eq!(plan.levels[2].nodes, vec![out]);
    }

    #[test]
    fn schedule_parallel_inputs() {
        // a, b → add → out
        // a and b should be in the same level (parallel)
        let mut g = Graph::new("parallel");
        let a = g.add_node(Op::Input { name: "a".into() }, vec![], vec![TensorType::f32_vector(4)]);
        let b = g.add_node(Op::Input { name: "b".into() }, vec![], vec![TensorType::f32_vector(4)]);
        let add = g.add_node(Op::Add, vec![TensorType::f32_vector(4); 2], vec![TensorType::f32_vector(4)]);
        let out = g.add_node(Op::Output { name: "y".into() }, vec![TensorType::f32_vector(4)], vec![]);
        g.add_edge(a, 0, add, 0, TensorType::f32_vector(4));
        g.add_edge(b, 0, add, 1, TensorType::f32_vector(4));
        g.add_edge(add, 0, out, 0, TensorType::f32_vector(4));

        let plan = schedule(&g);
        // Level 0: a, b (parallel), Level 1: add, Level 2: out
        assert_eq!(plan.levels[0].nodes.len(), 2);
    }

    #[test]
    fn memory_plan_tracks_peak() {
        let mut g = Graph::new("mem_test");
        let a = g.add_node(Op::Input { name: "a".into() }, vec![], vec![TensorType::f32_vector(1000)]);
        let b = g.add_node(Op::Input { name: "b".into() }, vec![], vec![TensorType::f32_vector(1000)]);
        let add = g.add_node(Op::Add, vec![TensorType::f32_vector(1000); 2], vec![TensorType::f32_vector(1000)]);
        let out = g.add_node(Op::Output { name: "y".into() }, vec![TensorType::f32_vector(1000)], vec![]);
        g.add_edge(a, 0, add, 0, TensorType::f32_vector(1000));
        g.add_edge(b, 0, add, 1, TensorType::f32_vector(1000));
        g.add_edge(add, 0, out, 0, TensorType::f32_vector(1000));

        let plan = schedule(&g);
        assert!(plan.memory_plan.peak_memory_bytes > 0);
        assert!(plan.memory_plan.total_allocated_bytes > 0);
    }

    #[test]
    fn find_independent_subgraphs_single() {
        let mut g = Graph::new("single");
        let a = g.add_node(Op::Input { name: "a".into() }, vec![], vec![TensorType::f32_vector(4)]);
        let relu = g.add_node(Op::Relu, vec![TensorType::f32_vector(4)], vec![TensorType::f32_vector(4)]);
        g.add_edge(a, 0, relu, 0, TensorType::f32_vector(4));

        let components = find_independent_subgraphs(&g);
        assert_eq!(components.len(), 1);
    }

    #[test]
    fn find_independent_subgraphs_two() {
        let mut g = Graph::new("two_components");
        // Component 1: a → relu
        let a = g.add_node(Op::Input { name: "a".into() }, vec![], vec![TensorType::f32_vector(4)]);
        let relu = g.add_node(Op::Relu, vec![TensorType::f32_vector(4)], vec![TensorType::f32_vector(4)]);
        g.add_edge(a, 0, relu, 0, TensorType::f32_vector(4));

        // Component 2: b → neg (disconnected)
        let b = g.add_node(Op::Input { name: "b".into() }, vec![], vec![TensorType::f32_vector(4)]);
        let neg = g.add_node(Op::Neg, vec![TensorType::f32_vector(4)], vec![TensorType::f32_vector(4)]);
        g.add_edge(b, 0, neg, 0, TensorType::f32_vector(4));

        let components = find_independent_subgraphs(&g);
        assert_eq!(components.len(), 2);
    }

    #[test]
    fn execution_plan_report() {
        let mut g = Graph::new("report_test");
        let a = g.add_node(Op::Input { name: "a".into() }, vec![], vec![TensorType::f32_vector(100)]);
        let relu = g.add_node(Op::Relu, vec![TensorType::f32_vector(100)], vec![TensorType::f32_vector(100)]);
        let out = g.add_node(Op::Output { name: "y".into() }, vec![TensorType::f32_vector(100)], vec![]);
        g.add_edge(a, 0, relu, 0, TensorType::f32_vector(100));
        g.add_edge(relu, 0, out, 0, TensorType::f32_vector(100));

        let plan = schedule(&g);
        let report = plan.report();
        assert!(report.contains("Execution Plan"));
        assert!(report.contains("Level"));
    }
}
