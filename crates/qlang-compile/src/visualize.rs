//! Graph visualization — Generate human-readable views of QLANG graphs.
//!
//! Supports:
//! - DOT format (for Graphviz rendering)
//! - ASCII art (for terminal display)
//! - Text summary

use qlang_core::graph::Graph;
use qlang_core::ops::Op;

/// Generate DOT format (Graphviz) for a QLANG graph.
///
/// Render with: dot -Tpng graph.dot -o graph.png
/// Or view online: https://dreampuf.github.io/GraphvizOnline/
pub fn to_dot(graph: &Graph) -> String {
    let mut dot = String::new();
    dot.push_str(&format!("digraph \"{}\" {{\n", graph.id));
    dot.push_str("  rankdir=TB;\n");
    dot.push_str("  node [shape=box, style=rounded, fontname=\"Courier\"];\n");
    dot.push_str("  edge [fontname=\"Courier\", fontsize=10];\n\n");

    // Nodes
    for node in &graph.nodes {
        let (color, shape) = match &node.op {
            Op::Input { .. } => ("#d4edda", "ellipse"),   // green
            Op::Output { .. } => ("#d1ecf1", "ellipse"),  // blue
            Op::ToTernary | Op::ToLowRank { .. } | Op::ToSparse { .. } | Op::Project { .. } => {
                ("#f8d7da", "box") // red - compression
            }
            op if op.is_quantum() => ("#e2d5f1", "box"),  // purple - quantum
            _ => ("#fff3cd", "box"),                       // yellow - compute
        };

        let label = match &node.op {
            Op::Input { name } => format!("INPUT\\n{name}"),
            Op::Output { name } => format!("OUTPUT\\n{name}"),
            _ => format!("{}", node.op),
        };

        // Add type info
        let type_info = if let Some(t) = node.output_types.first() {
            format!("\\n{t}")
        } else if let Some(t) = node.input_types.first() {
            format!("\\n→ {t}")
        } else {
            String::new()
        };

        let proof_badge = if !node.constraints.is_empty() { "\\n✓ proven" } else { "" };

        dot.push_str(&format!(
            "  n{} [label=\"{}{}{}\", shape={}, style=\"filled,rounded\", fillcolor=\"{}\"];\n",
            node.id, label, type_info, proof_badge, shape, color
        ));
    }

    dot.push_str("\n");

    // Edges
    for edge in &graph.edges {
        let label = format!("{}", edge.tensor_type);
        dot.push_str(&format!(
            "  n{} -> n{} [label=\"{}\"];\n",
            edge.from_node, edge.to_node, label
        ));
    }

    dot.push_str("}\n");
    dot
}

/// Generate ASCII art visualization of a graph (for terminal).
pub fn to_ascii(graph: &Graph) -> String {
    let mut out = String::new();

    // Sort nodes topologically for display
    let order = graph.topological_sort().unwrap_or_else(|_| {
        graph.nodes.iter().map(|n| n.id).collect()
    });

    out.push_str(&format!("╔══ Graph: {} ══╗\n", graph.id));
    out.push_str("║\n");

    for &node_id in &order {
        if let Some(node) = graph.node(node_id) {
            let icon = match &node.op {
                Op::Input { .. } => "▶",
                Op::Output { .. } => "◀",
                op if op.is_quantum() => "◈",
                Op::ToTernary | Op::ToLowRank { .. } | Op::ToSparse { .. } => "▼",
                Op::Relu | Op::Sigmoid | Op::Tanh | Op::Softmax { .. } => "σ",
                Op::MatMul => "×",
                Op::Add => "+",
                Op::Mul => "·",
                _ => "■",
            };

            let type_str = node.output_types.first()
                .map(|t| format!(" → {t}"))
                .unwrap_or_default();

            let proof_str = if !node.constraints.is_empty() { " ✓" } else { "" };

            out.push_str(&format!("║  [{:>2}] {icon} {}{}{}\n",
                node.id, node.op, type_str, proof_str));

            // Show edges going out
            let outgoing = graph.outgoing_edges(node_id);
            for edge in outgoing {
                out.push_str(&format!("║       └──▶ node {}\n", edge.to_node));
            }
        }
    }

    out.push_str("║\n");
    out.push_str(&format!("╚══ {} nodes, {} edges ══╝\n", graph.nodes.len(), graph.edges.len()));

    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use qlang_core::graph::Graph;
    use qlang_core::ops::Op;
    use qlang_core::tensor::TensorType;

    fn make_test_graph() -> Graph {
        let mut g = Graph::new("vis_test");
        let a = g.add_node(Op::Input { name: "x".into() }, vec![], vec![TensorType::f32_vector(4)]);
        let relu = g.add_node(Op::Relu, vec![TensorType::f32_vector(4)], vec![TensorType::f32_vector(4)]);
        let out = g.add_node(Op::Output { name: "y".into() }, vec![TensorType::f32_vector(4)], vec![]);
        g.add_edge(a, 0, relu, 0, TensorType::f32_vector(4));
        g.add_edge(relu, 0, out, 0, TensorType::f32_vector(4));
        g
    }

    #[test]
    fn dot_output() {
        let g = make_test_graph();
        let dot = to_dot(&g);
        assert!(dot.contains("digraph"));
        assert!(dot.contains("INPUT"));
        assert!(dot.contains("relu"));
        assert!(dot.contains("OUTPUT"));
        assert!(dot.contains("n0 -> n1"));
    }

    #[test]
    fn ascii_output() {
        let g = make_test_graph();
        let ascii = to_ascii(&g);
        assert!(ascii.contains("Graph: vis_test"));
        assert!(ascii.contains("input(x)"));
        assert!(ascii.contains("relu"));
    }
}
