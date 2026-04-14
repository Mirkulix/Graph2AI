//! Graph Statistics — Analyze structural properties of QLANG graphs.

use crate::graph::{Graph, NodeId};
use crate::ops::Op;

/// Comprehensive statistics about a graph.
#[derive(Debug, Clone)]
pub struct GraphStats {
    pub name: String,
    pub total_nodes: usize,
    pub total_edges: usize,
    pub input_nodes: usize,
    pub output_nodes: usize,
    pub compute_nodes: usize,
    pub quantum_nodes: usize,
    pub compression_nodes: usize,
    pub transformer_nodes: usize,
    pub max_depth: usize,
    pub max_width: usize,
    pub estimated_params: u64,
    pub estimated_memory_bytes: u64,
    pub ops_histogram: Vec<(String, usize)>,
}

/// Compute comprehensive statistics for a graph.
pub fn compute_stats(graph: &Graph) -> GraphStats {
    let mut quantum = 0;
    let mut compression = 0;
    let mut transformer = 0;
    let mut ops_count: std::collections::HashMap<String, usize> = std::collections::HashMap::new();

    for node in &graph.nodes {
        let op_name = format!("{}", node.op);
        *ops_count.entry(op_name).or_default() += 1;

        if node.op.is_quantum() { quantum += 1; }
        if matches!(node.op, Op::ToTernary | Op::ToLowRank { .. } | Op::ToSparse { .. }) {
            compression += 1;
        }
        if matches!(node.op, Op::Attention { .. } | Op::LayerNorm { .. } | Op::Gelu) {
            transformer += 1;
        }
    }

    let mut ops_histogram: Vec<(String, usize)> = ops_count.into_iter().collect();
    ops_histogram.sort_by(|a, b| b.1.cmp(&a.1));

    // Compute depth (longest path from any input to any output)
    let max_depth = compute_depth(graph);

    // Compute width (max nodes at any level)
    let max_width = compute_max_width(graph);

    // Estimate parameters (sum of all weight tensor sizes)
    let estimated_params: u64 = graph.nodes.iter()
        .filter(|n| matches!(n.op, Op::Input { .. }))
        .filter_map(|n| n.output_types.first())
        .filter_map(|t| t.shape.numel())
        .map(|n| n as u64)
        .sum();

    let estimated_memory: u64 = graph.edges.iter()
        .filter_map(|e| e.tensor_type.size_bytes())
        .map(|b| b as u64)
        .sum();

    GraphStats {
        name: graph.id.clone(),
        total_nodes: graph.nodes.len(),
        total_edges: graph.edges.len(),
        input_nodes: graph.input_nodes().len(),
        output_nodes: graph.output_nodes().len(),
        compute_nodes: graph.nodes.len() - graph.input_nodes().len() - graph.output_nodes().len(),
        quantum_nodes: quantum,
        compression_nodes: compression,
        transformer_nodes: transformer,
        max_depth,
        max_width,
        estimated_params,
        estimated_memory_bytes: estimated_memory,
        ops_histogram,
    }
}

fn compute_depth(graph: &Graph) -> usize {
    let order = match graph.topological_sort() {
        Ok(o) => o,
        Err(_) => return 0,
    };

    let mut depth: std::collections::HashMap<NodeId, usize> = std::collections::HashMap::new();
    for &id in &order {
        let incoming = graph.incoming_edges(id);
        let d = if incoming.is_empty() {
            0
        } else {
            incoming.iter()
                .map(|e| depth.get(&e.from_node).copied().unwrap_or(0) + 1)
                .max()
                .unwrap_or(0)
        };
        depth.insert(id, d);
    }

    depth.values().copied().max().unwrap_or(0)
}

fn compute_max_width(graph: &Graph) -> usize {
    let order = match graph.topological_sort() {
        Ok(o) => o,
        Err(_) => return 0,
    };

    let mut level: std::collections::HashMap<NodeId, usize> = std::collections::HashMap::new();
    for &id in &order {
        let incoming = graph.incoming_edges(id);
        let l = if incoming.is_empty() {
            0
        } else {
            incoming.iter()
                .map(|e| level.get(&e.from_node).copied().unwrap_or(0) + 1)
                .max()
                .unwrap_or(0)
        };
        level.insert(id, l);
    }

    let max_level = level.values().copied().max().unwrap_or(0);
    let mut max_width = 0;
    for l in 0..=max_level {
        let width = level.values().filter(|&&v| v == l).count();
        max_width = max_width.max(width);
    }
    max_width
}

impl std::fmt::Display for GraphStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Graph Statistics: {}", self.name)?;
        writeln!(f, "  Nodes:        {} ({} compute, {} I/O)",
            self.total_nodes, self.compute_nodes, self.input_nodes + self.output_nodes)?;
        writeln!(f, "  Edges:        {}", self.total_edges)?;
        writeln!(f, "  Depth:        {} levels", self.max_depth)?;
        writeln!(f, "  Width:        {} (max parallel)", self.max_width)?;
        writeln!(f, "  Params:       ~{}", self.estimated_params)?;
        writeln!(f, "  Memory:       ~{:.1} KB", self.estimated_memory_bytes as f64 / 1024.0)?;
        if self.quantum_nodes > 0 {
            writeln!(f, "  Quantum ops:  {}", self.quantum_nodes)?;
        }
        if self.compression_nodes > 0 {
            writeln!(f, "  Compression:  {}", self.compression_nodes)?;
        }
        if self.transformer_nodes > 0 {
            writeln!(f, "  Transformer:  {}", self.transformer_nodes)?;
        }
        writeln!(f, "  Op histogram:")?;
        for (op, count) in &self.ops_histogram {
            writeln!(f, "    {op}: {count}")?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::Graph;
    use crate::ops::Op;
    use crate::tensor::TensorType;

    #[test]
    fn stats_simple_graph() {
        let mut g = Graph::new("stats_test");
        let a = g.add_node(Op::Input { name: "a".into() }, vec![], vec![TensorType::f32_vector(100)]);
        let b = g.add_node(Op::Input { name: "b".into() }, vec![], vec![TensorType::f32_vector(100)]);
        let add = g.add_node(Op::Add, vec![TensorType::f32_vector(100); 2], vec![TensorType::f32_vector(100)]);
        let out = g.add_node(Op::Output { name: "y".into() }, vec![TensorType::f32_vector(100)], vec![]);
        g.add_edge(a, 0, add, 0, TensorType::f32_vector(100));
        g.add_edge(b, 0, add, 1, TensorType::f32_vector(100));
        g.add_edge(add, 0, out, 0, TensorType::f32_vector(100));

        let stats = compute_stats(&g);
        assert_eq!(stats.total_nodes, 4);
        assert_eq!(stats.input_nodes, 2);
        assert_eq!(stats.output_nodes, 1);
        assert_eq!(stats.compute_nodes, 1);
        assert_eq!(stats.max_depth, 2);
        assert_eq!(stats.max_width, 2); // a and b are parallel
    }

    #[test]
    fn stats_display() {
        let g = Graph::new("display_test");
        let stats = compute_stats(&g);
        let display = format!("{stats}");
        assert!(display.contains("Graph Statistics"));
    }
}
