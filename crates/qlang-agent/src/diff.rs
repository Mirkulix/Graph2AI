//! Graph Diff & Merge — Compare and merge QLANG graphs.
//!
//! When multiple AI agents work on the same model, they need to:
//! 1. Compute what changed between graph versions
//! 2. Merge changes from different agents
//!
//! This is the graph-level equivalent of `git diff` and `git merge`.

use qlang_core::graph::Graph;

/// A single change between two graph versions.
#[derive(Debug, Clone)]
pub enum GraphChange {
    /// Node was added in the new version.
    NodeAdded { node_id: u32, op: String },
    /// Node was removed in the new version.
    NodeRemoved { node_id: u32, op: String },
    /// Node operation changed.
    NodeModified { node_id: u32, old_op: String, new_op: String },
    /// Edge was added.
    EdgeAdded { from: u32, to: u32 },
    /// Edge was removed.
    EdgeRemoved { from: u32, to: u32 },
}

/// Result of diffing two graphs.
#[derive(Debug)]
pub struct GraphDiff {
    pub changes: Vec<GraphChange>,
    pub nodes_added: usize,
    pub nodes_removed: usize,
    pub nodes_modified: usize,
    pub edges_added: usize,
    pub edges_removed: usize,
}

impl GraphDiff {
    pub fn is_empty(&self) -> bool {
        self.changes.is_empty()
    }

    pub fn summary(&self) -> String {
        if self.is_empty() {
            return "No changes.".to_string();
        }
        format!(
            "{} changes: +{} nodes, -{} nodes, ~{} modified, +{} edges, -{} edges",
            self.changes.len(),
            self.nodes_added,
            self.nodes_removed,
            self.nodes_modified,
            self.edges_added,
            self.edges_removed,
        )
    }
}

impl std::fmt::Display for GraphDiff {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Graph Diff ({}):", self.summary())?;
        for change in &self.changes {
            match change {
                GraphChange::NodeAdded { node_id, op } =>
                    writeln!(f, "  + node [{node_id}] {op}")?,
                GraphChange::NodeRemoved { node_id, op } =>
                    writeln!(f, "  - node [{node_id}] {op}")?,
                GraphChange::NodeModified { node_id, old_op, new_op } =>
                    writeln!(f, "  ~ node [{node_id}] {old_op} → {new_op}")?,
                GraphChange::EdgeAdded { from, to } =>
                    writeln!(f, "  + edge {from} → {to}")?,
                GraphChange::EdgeRemoved { from, to } =>
                    writeln!(f, "  - edge {from} → {to}")?,
            }
        }
        Ok(())
    }
}

/// Compute the diff between two graph versions.
pub fn diff(old: &Graph, new: &Graph) -> GraphDiff {
    let mut changes = Vec::new();
    let mut nodes_added = 0;
    let mut nodes_removed = 0;
    let mut nodes_modified = 0;
    let mut edges_added = 0;
    let mut edges_removed = 0;

    // Compare nodes
    let old_ids: std::collections::HashSet<u32> = old.nodes.iter().map(|n| n.id).collect();
    let new_ids: std::collections::HashSet<u32> = new.nodes.iter().map(|n| n.id).collect();

    // Added nodes
    for &id in new_ids.difference(&old_ids) {
        if let Some(node) = new.node(id) {
            changes.push(GraphChange::NodeAdded {
                node_id: id,
                op: format!("{}", node.op),
            });
            nodes_added += 1;
        }
    }

    // Removed nodes
    for &id in old_ids.difference(&new_ids) {
        if let Some(node) = old.node(id) {
            changes.push(GraphChange::NodeRemoved {
                node_id: id,
                op: format!("{}", node.op),
            });
            nodes_removed += 1;
        }
    }

    // Modified nodes (same ID, different op)
    for &id in old_ids.intersection(&new_ids) {
        let old_node = old.node(id).unwrap();
        let new_node = new.node(id).unwrap();
        let old_op = format!("{}", old_node.op);
        let new_op = format!("{}", new_node.op);
        if old_op != new_op {
            changes.push(GraphChange::NodeModified {
                node_id: id,
                old_op,
                new_op,
            });
            nodes_modified += 1;
        }
    }

    // Compare edges
    let old_edges: std::collections::HashSet<(u32, u32)> =
        old.edges.iter().map(|e| (e.from_node, e.to_node)).collect();
    let new_edges: std::collections::HashSet<(u32, u32)> =
        new.edges.iter().map(|e| (e.from_node, e.to_node)).collect();

    for &(from, to) in new_edges.difference(&old_edges) {
        changes.push(GraphChange::EdgeAdded { from, to });
        edges_added += 1;
    }

    for &(from, to) in old_edges.difference(&new_edges) {
        changes.push(GraphChange::EdgeRemoved { from, to });
        edges_removed += 1;
    }

    GraphDiff {
        changes,
        nodes_added,
        nodes_removed,
        nodes_modified,
        edges_added,
        edges_removed,
    }
}

/// Apply a diff to a graph, producing a new version.
pub fn apply_diff(base: &Graph, diff: &GraphDiff) -> Graph {
    let mut result = base.clone();

    for change in &diff.changes {
        match change {
            GraphChange::NodeRemoved { node_id, .. } => {
                result.nodes.retain(|n| n.id != *node_id);
                result.edges.retain(|e| e.from_node != *node_id && e.to_node != *node_id);
            }
            _ => {
                // NodeAdded and EdgeAdded would need the full node/edge data
                // For now, we only support removal (safe operation)
            }
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use qlang_core::ops::Op;
    use qlang_core::tensor::TensorType;

    #[test]
    fn diff_identical_graphs() {
        let g = Graph::new("same");
        let d = diff(&g, &g);
        assert!(d.is_empty());
    }

    #[test]
    fn diff_added_node() {
        let old = Graph::new("v1");
        let mut new = Graph::new("v2");
        new.add_node(Op::Input { name: "x".into() }, vec![], vec![TensorType::f32_vector(4)]);

        let d = diff(&old, &new);
        assert_eq!(d.nodes_added, 1);
        assert_eq!(d.nodes_removed, 0);

        let summary = d.summary();
        assert!(summary.contains("+1 nodes"));
    }

    #[test]
    fn diff_removed_node() {
        let mut old = Graph::new("v1");
        old.add_node(Op::Input { name: "x".into() }, vec![], vec![TensorType::f32_vector(4)]);
        let new = Graph::new("v2");

        let d = diff(&old, &new);
        assert_eq!(d.nodes_removed, 1);
    }

    #[test]
    fn diff_display() {
        let old = Graph::new("v1");
        let mut new = Graph::new("v2");
        new.add_node(Op::Relu, vec![TensorType::f32_vector(4)], vec![TensorType::f32_vector(4)]);

        let d = diff(&old, &new);
        let display = format!("{d}");
        assert!(display.contains("+ node"));
        assert!(display.contains("relu"));
    }

    #[test]
    fn diff_edges() {
        let mut old = Graph::new("v1");
        let a = old.add_node(Op::Input { name: "a".into() }, vec![], vec![TensorType::f32_vector(4)]);
        let b = old.add_node(Op::Relu, vec![TensorType::f32_vector(4)], vec![TensorType::f32_vector(4)]);

        let mut new = old.clone();
        new.add_edge(a, 0, b, 0, TensorType::f32_vector(4));

        let d = diff(&old, &new);
        assert_eq!(d.edges_added, 1);
    }
}
