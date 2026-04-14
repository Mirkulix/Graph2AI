use serde::{Deserialize, Serialize};
use std::collections::{BinaryHeap, HashMap};
use std::cmp::Reverse;
use std::fmt;

use crate::ops::Op;
use crate::tensor::TensorType;
use crate::verify::Constraint;

/// Unique identifier for a node within a graph.
pub type NodeId = u32;

/// Unique identifier for an edge within a graph.
pub type EdgeId = u32;

/// A QLANG program: a directed acyclic graph of computations.
///
/// This IS the program. Not text. Not syntax. A graph.
/// AI agents emit this directly. Compilers consume it directly.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Graph {
    pub id: String,
    pub version: String,
    pub nodes: Vec<Node>,
    pub edges: Vec<Edge>,
    pub constraints: Vec<Constraint>,
    pub metadata: HashMap<String, String>,
}

/// A single computation node.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Node {
    pub id: NodeId,
    pub op: Op,
    pub input_types: Vec<TensorType>,
    pub output_types: Vec<TensorType>,
    pub constraints: Vec<Constraint>,
    pub metadata: HashMap<String, String>,
}

/// A directed edge: data flows from one node's output to another's input.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Edge {
    pub id: EdgeId,
    pub from_node: NodeId,
    pub from_port: u8,
    pub to_node: NodeId,
    pub to_port: u8,
    pub tensor_type: TensorType,
}

/// Errors that can occur when building or validating a graph.
#[derive(Debug, thiserror::Error)]
pub enum GraphError {
    #[error("node {0} not found")]
    NodeNotFound(NodeId),

    #[error("duplicate node id {0}")]
    DuplicateNodeId(NodeId),

    #[error("edge {edge_id}: source node {from} port {from_port} not found")]
    InvalidEdgeSource {
        edge_id: EdgeId,
        from: NodeId,
        from_port: u8,
    },

    #[error("edge {edge_id}: target node {to} port {to_port} not found")]
    InvalidEdgeTarget {
        edge_id: EdgeId,
        to: NodeId,
        to_port: u8,
    },

    #[error("type mismatch on edge {edge_id}: expected {expected}, got {got}")]
    TypeMismatch {
        edge_id: EdgeId,
        expected: TensorType,
        got: TensorType,
    },

    #[error("cycle detected in graph")]
    CycleDetected,

    #[error("node {0} has unconnected input port {1}")]
    UnconnectedInput(NodeId, u8),
}

impl Graph {
    /// Create a new empty graph.
    pub fn new(id: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            version: "0.1".to_string(),
            nodes: Vec::new(),
            edges: Vec::new(),
            constraints: Vec::new(),
            metadata: HashMap::new(),
        }
    }

    /// Add a node to the graph, returning its ID.
    pub fn add_node(&mut self, op: Op, input_types: Vec<TensorType>, output_types: Vec<TensorType>) -> NodeId {
        let id = self.nodes.len() as NodeId;
        self.nodes.push(Node {
            id,
            op,
            input_types,
            output_types,
            constraints: Vec::new(),
            metadata: HashMap::new(),
        });
        id
    }

    /// Connect two nodes: from_node:from_port → to_node:to_port.
    pub fn add_edge(
        &mut self,
        from_node: NodeId,
        from_port: u8,
        to_node: NodeId,
        to_port: u8,
        tensor_type: TensorType,
    ) -> EdgeId {
        let id = self.edges.len() as EdgeId;
        self.edges.push(Edge {
            id,
            from_node,
            from_port,
            to_node,
            to_port,
            tensor_type,
        });
        id
    }

    /// Get a node by ID (O(1) — node IDs are sequential indices).
    pub fn node(&self, id: NodeId) -> Option<&Node> {
        self.nodes.get(id as usize).filter(|n| n.id == id)
    }

    /// Get all edges going INTO a node.
    pub fn incoming_edges(&self, node_id: NodeId) -> Vec<&Edge> {
        self.edges.iter().filter(|e| e.to_node == node_id).collect()
    }

    /// Get all edges going OUT of a node.
    pub fn outgoing_edges(&self, node_id: NodeId) -> Vec<&Edge> {
        self.edges.iter().filter(|e| e.from_node == node_id).collect()
    }

    /// Topological sort of nodes (execution order).
    /// Returns error if the graph contains a cycle.
    pub fn topological_sort(&self) -> Result<Vec<NodeId>, GraphError> {
        let n = self.nodes.len();
        let mut in_degree: HashMap<NodeId, usize> = HashMap::new();
        let mut adj: HashMap<NodeId, Vec<NodeId>> = HashMap::new();

        for node in &self.nodes {
            in_degree.entry(node.id).or_insert(0);
            adj.entry(node.id).or_insert_with(Vec::new);
        }

        for edge in &self.edges {
            *in_degree.entry(edge.to_node).or_insert(0) += 1;
            adj.entry(edge.from_node)
                .or_insert_with(Vec::new)
                .push(edge.to_node);
        }

        let mut heap: BinaryHeap<Reverse<NodeId>> = in_degree
            .iter()
            .filter(|&(_, deg)| *deg == 0)
            .map(|(&id, _)| Reverse(id))
            .collect();

        let mut order = Vec::with_capacity(n);

        while let Some(Reverse(node)) = heap.pop() {
            order.push(node);
            if let Some(neighbors) = adj.get(&node) {
                for &neighbor in neighbors {
                    let deg = in_degree.get_mut(&neighbor).unwrap();
                    *deg -= 1;
                    if *deg == 0 {
                        heap.push(Reverse(neighbor));
                    }
                }
            }
        }

        if order.len() != n {
            return Err(GraphError::CycleDetected);
        }

        Ok(order)
    }

    /// Validate the graph: check types, connections, acyclicity.
    pub fn validate(&self) -> Result<(), Vec<GraphError>> {
        let mut errors = Vec::new();

        // Check for duplicate IDs
        let mut seen_ids = std::collections::HashSet::new();
        for node in &self.nodes {
            if !seen_ids.insert(node.id) {
                errors.push(GraphError::DuplicateNodeId(node.id));
            }
        }

        // Check edges reference valid nodes
        for edge in &self.edges {
            if self.node(edge.from_node).is_none() {
                errors.push(GraphError::InvalidEdgeSource {
                    edge_id: edge.id,
                    from: edge.from_node,
                    from_port: edge.from_port,
                });
            }
            if self.node(edge.to_node).is_none() {
                errors.push(GraphError::InvalidEdgeTarget {
                    edge_id: edge.id,
                    to: edge.to_node,
                    to_port: edge.to_port,
                });
            }
        }

        // Check for cycles
        if let Err(e) = self.topological_sort() {
            errors.push(e);
        }

        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }

    /// Count of input nodes.
    pub fn input_nodes(&self) -> Vec<&Node> {
        self.nodes
            .iter()
            .filter(|n| matches!(n.op, Op::Input { .. }))
            .collect()
    }

    /// Count of output nodes.
    pub fn output_nodes(&self) -> Vec<&Node> {
        self.nodes
            .iter()
            .filter(|n| matches!(n.op, Op::Output { .. }))
            .collect()
    }
}

impl fmt::Display for Graph {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Graph '{}' (v{}):", self.id, self.version)?;
        writeln!(f, "  Nodes: {}", self.nodes.len())?;
        for node in &self.nodes {
            writeln!(f, "    [{:>3}] {}", node.id, node.op)?;
        }
        writeln!(f, "  Edges: {}", self.edges.len())?;
        for edge in &self.edges {
            writeln!(
                f,
                "    {}:{} → {}:{} ({})",
                edge.from_node, edge.from_port, edge.to_node, edge.to_port, edge.tensor_type
            )?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::{Dtype, Shape, TensorType};

    fn f32_vec(n: usize) -> TensorType {
        TensorType::new(Dtype::F32, Shape::vector(n))
    }

    #[test]
    fn build_simple_graph() {
        let mut g = Graph::new("test");

        let input = g.add_node(
            Op::Input { name: "x".into() },
            vec![],
            vec![f32_vec(4)],
        );

        let relu = g.add_node(Op::Relu, vec![f32_vec(4)], vec![f32_vec(4)]);

        let output = g.add_node(
            Op::Output { name: "y".into() },
            vec![f32_vec(4)],
            vec![],
        );

        g.add_edge(input, 0, relu, 0, f32_vec(4));
        g.add_edge(relu, 0, output, 0, f32_vec(4));

        assert_eq!(g.nodes.len(), 3);
        assert_eq!(g.edges.len(), 2);
        assert!(g.validate().is_ok());
    }

    #[test]
    fn topological_sort_works() {
        let mut g = Graph::new("topo_test");

        let a = g.add_node(Op::Input { name: "a".into() }, vec![], vec![f32_vec(4)]);
        let b = g.add_node(Op::Input { name: "b".into() }, vec![], vec![f32_vec(4)]);
        let add = g.add_node(Op::Add, vec![f32_vec(4), f32_vec(4)], vec![f32_vec(4)]);
        let out = g.add_node(Op::Output { name: "y".into() }, vec![f32_vec(4)], vec![]);

        g.add_edge(a, 0, add, 0, f32_vec(4));
        g.add_edge(b, 0, add, 1, f32_vec(4));
        g.add_edge(add, 0, out, 0, f32_vec(4));

        let order = g.topological_sort().unwrap();
        // add must come after a and b, out must come after add
        let pos_a = order.iter().position(|&x| x == a).unwrap();
        let pos_b = order.iter().position(|&x| x == b).unwrap();
        let pos_add = order.iter().position(|&x| x == add).unwrap();
        let pos_out = order.iter().position(|&x| x == out).unwrap();

        assert!(pos_a < pos_add);
        assert!(pos_b < pos_add);
        assert!(pos_add < pos_out);
    }

    #[test]
    fn display_graph() {
        let mut g = Graph::new("display_test");
        g.add_node(Op::Input { name: "x".into() }, vec![], vec![f32_vec(784)]);
        g.add_node(Op::ToTernary, vec![f32_vec(784)], vec![TensorType::new(Dtype::Ternary, Shape::vector(784))]);
        g.add_edge(0, 0, 1, 0, f32_vec(784));

        let display = format!("{g}");
        assert!(display.contains("input(x)"));
        assert!(display.contains("to_ternary"));
    }
}
