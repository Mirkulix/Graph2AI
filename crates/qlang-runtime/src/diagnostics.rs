//! Runtime diagnostics and error recovery for QLANG graph execution.
//!
//! This module provides comprehensive graph validation that collects
//! all issues without stopping at the first error, along with
//! human-readable diagnostic formatting.

use std::collections::{HashMap, HashSet};
use std::fmt;

use qlang_core::graph::{Graph, NodeId};
use qlang_core::ops::Op;
use qlang_core::tensor::{Dim, Dtype};

// ---------------------------------------------------------------------------
// Severity & Diagnostic
// ---------------------------------------------------------------------------

/// Severity level for a diagnostic message.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Severity {
    Info,
    Warning,
    Error,
}

impl fmt::Display for Severity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Severity::Info => write!(f, "info"),
            Severity::Warning => write!(f, "warning"),
            Severity::Error => write!(f, "error"),
        }
    }
}

/// A single diagnostic produced during graph validation or execution.
#[derive(Debug, Clone)]
pub struct RuntimeDiagnostic {
    pub severity: Severity,
    pub message: String,
    pub node_id: Option<NodeId>,
    pub suggestion: Option<String>,
}

impl RuntimeDiagnostic {
    pub fn error(message: impl Into<String>) -> Self {
        Self {
            severity: Severity::Error,
            message: message.into(),
            node_id: None,
            suggestion: None,
        }
    }

    pub fn warning(message: impl Into<String>) -> Self {
        Self {
            severity: Severity::Warning,
            message: message.into(),
            node_id: None,
            suggestion: None,
        }
    }

    pub fn info(message: impl Into<String>) -> Self {
        Self {
            severity: Severity::Info,
            message: message.into(),
            node_id: None,
            suggestion: None,
        }
    }

    pub fn with_node(mut self, id: NodeId) -> Self {
        self.node_id = Some(id);
        self
    }

    pub fn with_suggestion(mut self, suggestion: impl Into<String>) -> Self {
        self.suggestion = Some(suggestion.into());
        self
    }
}

impl fmt::Display for RuntimeDiagnostic {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{}]", self.severity)?;
        if let Some(nid) = self.node_id {
            write!(f, " node {nid}")?;
        }
        write!(f, ": {}", self.message)?;
        if let Some(ref s) = self.suggestion {
            write!(f, " (suggestion: {s})")?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// DiagnosticCollector
// ---------------------------------------------------------------------------

/// Collects diagnostics during graph execution without halting.
///
/// Use this to accumulate warnings and errors across an entire run,
/// then inspect or format them at the end.
#[derive(Debug, Clone, Default)]
pub struct DiagnosticCollector {
    diagnostics: Vec<RuntimeDiagnostic>,
}

impl DiagnosticCollector {
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a diagnostic.
    pub fn push(&mut self, diag: RuntimeDiagnostic) {
        self.diagnostics.push(diag);
    }

    /// Record an error diagnostic.
    pub fn error(&mut self, message: impl Into<String>, node_id: Option<NodeId>) {
        let mut d = RuntimeDiagnostic::error(message);
        d.node_id = node_id;
        self.diagnostics.push(d);
    }

    /// Record a warning diagnostic.
    pub fn warning(&mut self, message: impl Into<String>, node_id: Option<NodeId>) {
        let mut d = RuntimeDiagnostic::warning(message);
        d.node_id = node_id;
        self.diagnostics.push(d);
    }

    /// Record an info diagnostic.
    pub fn info(&mut self, message: impl Into<String>, node_id: Option<NodeId>) {
        let mut d = RuntimeDiagnostic::info(message);
        d.node_id = node_id;
        self.diagnostics.push(d);
    }

    /// All collected diagnostics.
    pub fn diagnostics(&self) -> &[RuntimeDiagnostic] {
        &self.diagnostics
    }

    /// Move diagnostics out of the collector.
    pub fn into_diagnostics(self) -> Vec<RuntimeDiagnostic> {
        self.diagnostics
    }

    /// True if any error-severity diagnostic was recorded.
    pub fn has_errors(&self) -> bool {
        self.diagnostics.iter().any(|d| d.severity == Severity::Error)
    }

    /// Number of diagnostics at a given severity.
    pub fn count(&self, severity: Severity) -> usize {
        self.diagnostics.iter().filter(|d| d.severity == severity).count()
    }

    /// Total number of diagnostics.
    pub fn len(&self) -> usize {
        self.diagnostics.len()
    }

    /// True if no diagnostics have been collected.
    pub fn is_empty(&self) -> bool {
        self.diagnostics.is_empty()
    }
}

// ---------------------------------------------------------------------------
// Deep graph validation
// ---------------------------------------------------------------------------

/// Maximum dimension size before we warn about potential numerical issues.
const LARGE_DIM_THRESHOLD: usize = 1_000_000;

/// Minimum dimension size that is suspiciously small (zero).
const ZERO_DIM: usize = 0;

/// Perform comprehensive validation of a graph, returning all diagnostics.
///
/// Checks performed:
/// 1. All edges connect valid nodes and valid ports
/// 2. Tensor shape compatibility across edges
/// 3. Unused nodes (not reachable from inputs to outputs)
/// 4. Type mismatches (e.g. ternary input to matmul)
/// 5. Potential numerical issues (very large / zero-sized dimensions)
/// 6. Suggested fixes for common errors
pub fn validate_graph_deep(graph: &Graph) -> Vec<RuntimeDiagnostic> {
    let mut collector = DiagnosticCollector::new();

    check_edge_validity(graph, &mut collector);
    check_shape_compatibility(graph, &mut collector);
    check_unused_nodes(graph, &mut collector);
    check_type_mismatches(graph, &mut collector);
    check_numerical_issues(graph, &mut collector);

    collector.into_diagnostics()
}

// -- helpers ----------------------------------------------------------------

/// Build a set of all node IDs for fast lookup.
fn node_id_set(graph: &Graph) -> HashSet<NodeId> {
    graph.nodes.iter().map(|n| n.id).collect()
}

/// 1. Check every edge references valid nodes and valid ports.
fn check_edge_validity(graph: &Graph, collector: &mut DiagnosticCollector) {
    let ids = node_id_set(graph);

    for edge in &graph.edges {
        // Source node exists?
        let from_node = graph.node(edge.from_node);
        if from_node.is_none() {
            collector.push(
                RuntimeDiagnostic::error(format!(
                    "Edge {}: source node {} does not exist",
                    edge.id, edge.from_node
                ))
                .with_suggestion("Remove the edge or add the missing source node."),
            );
        }

        // Target node exists?
        let to_node = graph.node(edge.to_node);
        if to_node.is_none() {
            if !ids.contains(&edge.to_node) {
                collector.push(
                    RuntimeDiagnostic::error(format!(
                        "Edge {}: target node {} does not exist",
                        edge.id, edge.to_node
                    ))
                    .with_suggestion("Remove the edge or add the missing target node."),
                );
            }
        }

        // Source port in range?
        if let Some(src) = from_node {
            let n_out = src.op.n_outputs();
            if (edge.from_port as usize) >= n_out {
                collector.push(
                    RuntimeDiagnostic::error(format!(
                        "Edge {}: source port {} on node {} ({}) exceeds output count ({n_out})",
                        edge.id, edge.from_port, src.id, src.op
                    ))
                    .with_node(src.id)
                    .with_suggestion(format!(
                        "Valid output ports for {} are 0..{n_out}.", src.op
                    )),
                );
            }
        }

        // Target port in range?
        if let Some(tgt) = to_node {
            let n_in = tgt.op.n_inputs();
            if (edge.to_port as usize) >= n_in {
                collector.push(
                    RuntimeDiagnostic::error(format!(
                        "Edge {}: target port {} on node {} ({}) exceeds input count ({n_in})",
                        edge.id, edge.to_port, tgt.id, tgt.op
                    ))
                    .with_node(tgt.id)
                    .with_suggestion(format!(
                        "Valid input ports for {} are 0..{n_in}.", tgt.op
                    )),
                );
            }
        }
    }
}

/// 2. Check tensor shape compatibility across edges.
fn check_shape_compatibility(graph: &Graph, collector: &mut DiagnosticCollector) {
    for edge in &graph.edges {
        let from_node = match graph.node(edge.from_node) {
            Some(n) => n,
            None => continue, // already reported in edge validity
        };
        let to_node = match graph.node(edge.to_node) {
            Some(n) => n,
            None => continue,
        };

        // Compare edge tensor type against source output type.
        if let Some(src_type) = from_node.output_types.get(edge.from_port as usize) {
            if !src_type.shape.is_compatible_with(&edge.tensor_type.shape) {
                collector.push(
                    RuntimeDiagnostic::error(format!(
                        "Edge {}: shape mismatch at source — node {} output port {} has shape {}, \
                         but edge declares {}",
                        edge.id, from_node.id, edge.from_port, src_type.shape, edge.tensor_type.shape
                    ))
                    .with_node(from_node.id)
                    .with_suggestion("Update the edge tensor type to match the source node output."),
                );
            }
        }

        // Compare edge tensor type against target input type.
        if let Some(tgt_type) = to_node.input_types.get(edge.to_port as usize) {
            if !tgt_type.shape.is_compatible_with(&edge.tensor_type.shape) {
                collector.push(
                    RuntimeDiagnostic::error(format!(
                        "Edge {}: shape mismatch at target — node {} input port {} expects shape {}, \
                         but edge carries {}",
                        edge.id, to_node.id, edge.to_port, tgt_type.shape, edge.tensor_type.shape
                    ))
                    .with_node(to_node.id)
                    .with_suggestion(
                        "Insert a Reshape node or correct the shapes to make them compatible.",
                    ),
                );
            }
        }
    }
}

/// 3. Detect unused nodes — those not on any path from an input to an output.
fn check_unused_nodes(graph: &Graph, collector: &mut DiagnosticCollector) {
    if graph.nodes.is_empty() {
        return;
    }

    // Build adjacency lists.
    let mut forward: HashMap<NodeId, Vec<NodeId>> = HashMap::new();
    let mut backward: HashMap<NodeId, Vec<NodeId>> = HashMap::new();
    for node in &graph.nodes {
        forward.entry(node.id).or_default();
        backward.entry(node.id).or_default();
    }
    for edge in &graph.edges {
        forward.entry(edge.from_node).or_default().push(edge.to_node);
        backward.entry(edge.to_node).or_default().push(edge.from_node);
    }

    // Forward reachable from inputs.
    let mut reachable_from_input: HashSet<NodeId> = HashSet::new();
    let mut stack: Vec<NodeId> = graph
        .nodes
        .iter()
        .filter(|n| matches!(n.op, Op::Input { .. } | Op::Constant))
        .map(|n| n.id)
        .collect();
    while let Some(nid) = stack.pop() {
        if reachable_from_input.insert(nid) {
            if let Some(neighbors) = forward.get(&nid) {
                stack.extend(neighbors);
            }
        }
    }

    // Backward reachable from outputs.
    let mut reachable_from_output: HashSet<NodeId> = HashSet::new();
    let mut stack: Vec<NodeId> = graph
        .nodes
        .iter()
        .filter(|n| matches!(n.op, Op::Output { .. }))
        .map(|n| n.id)
        .collect();
    while let Some(nid) = stack.pop() {
        if reachable_from_output.insert(nid) {
            if let Some(neighbors) = backward.get(&nid) {
                stack.extend(neighbors);
            }
        }
    }

    // A node is "used" if it is on a path from input to output.
    let used: HashSet<NodeId> = reachable_from_input
        .intersection(&reachable_from_output)
        .copied()
        .collect();

    for node in &graph.nodes {
        if !used.contains(&node.id) {
            // Inputs/outputs that are disconnected get a warning, not just info.
            let severity = if matches!(node.op, Op::Input { .. } | Op::Output { .. }) {
                Severity::Warning
            } else {
                Severity::Info
            };
            collector.push(
                RuntimeDiagnostic {
                    severity,
                    message: format!(
                        "Node {} ({}) is unused — not on any path from input to output",
                        node.id, node.op
                    ),
                    node_id: Some(node.id),
                    suggestion: Some(
                        "Remove the node or connect it to the rest of the graph.".into(),
                    ),
                },
            );
        }
    }
}

/// 4. Detect type mismatches — e.g., ternary dtype flowing into ops that
///    require floating-point.
fn check_type_mismatches(graph: &Graph, collector: &mut DiagnosticCollector) {
    // Ops that fundamentally require floating-point inputs.
    let requires_float = |op: &Op| -> bool {
        matches!(
            op,
            Op::MatMul
                | Op::Sigmoid
                | Op::Tanh
                | Op::Softmax { .. }
                | Op::Gelu
                | Op::LayerNorm { .. }
                | Op::Attention { .. }
                | Op::Evolve { .. }
                | Op::FisherMetric
                | Op::Entropy
        )
    };

    let is_float = |dtype: &Dtype| -> bool {
        matches!(dtype, Dtype::F16 | Dtype::F32 | Dtype::F64)
    };

    for edge in &graph.edges {
        let to_node = match graph.node(edge.to_node) {
            Some(n) => n,
            None => continue,
        };

        if requires_float(&to_node.op) && !is_float(&edge.tensor_type.dtype) {
            collector.push(
                RuntimeDiagnostic::error(format!(
                    "Edge {}: dtype {} fed into {} (node {}), which requires a floating-point type",
                    edge.id, edge.tensor_type.dtype, to_node.op, to_node.id
                ))
                .with_node(to_node.id)
                .with_suggestion(format!(
                    "Insert a type-cast node before {} to convert {} to f32.",
                    to_node.op, edge.tensor_type.dtype
                )),
            );
        }
    }
}

/// 5. Check for potential numerical issues: zero-sized or very large dimensions.
fn check_numerical_issues(graph: &Graph, collector: &mut DiagnosticCollector) {
    for node in &graph.nodes {
        // Check output types for suspicious dimensions.
        for (port_idx, tt) in node.output_types.iter().enumerate() {
            for (dim_idx, dim) in tt.shape.0.iter().enumerate() {
                if let Dim::Fixed(size) = dim {
                    if *size == ZERO_DIM {
                        collector.push(
                            RuntimeDiagnostic::error(format!(
                                "Node {} ({}) output port {}: dimension {} is zero",
                                node.id, node.op, port_idx, dim_idx
                            ))
                            .with_node(node.id)
                            .with_suggestion(
                                "A zero-sized dimension produces an empty tensor. \
                                 This is almost certainly unintentional.",
                            ),
                        );
                    } else if *size >= LARGE_DIM_THRESHOLD {
                        collector.push(
                            RuntimeDiagnostic::warning(format!(
                                "Node {} ({}) output port {}: dimension {} has size {} (>= {})",
                                node.id, node.op, port_idx, dim_idx, size, LARGE_DIM_THRESHOLD
                            ))
                            .with_node(node.id)
                            .with_suggestion(
                                "Very large dimensions may cause excessive memory usage or \
                                 numerical instability. Consider using a smaller size or \
                                 low-rank / sparse compression.",
                            ),
                        );
                    }
                }
            }
        }

        // Same check for input types.
        for (port_idx, tt) in node.input_types.iter().enumerate() {
            for (dim_idx, dim) in tt.shape.0.iter().enumerate() {
                if let Dim::Fixed(size) = dim {
                    if *size == ZERO_DIM {
                        collector.push(
                            RuntimeDiagnostic::error(format!(
                                "Node {} ({}) input port {}: dimension {} is zero",
                                node.id, node.op, port_idx, dim_idx
                            ))
                            .with_node(node.id)
                            .with_suggestion(
                                "A zero-sized dimension produces an empty tensor. \
                                 This is almost certainly unintentional.",
                            ),
                        );
                    }
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Pretty-print formatting (ANSI colors)
// ---------------------------------------------------------------------------

/// Format diagnostics as a human-readable string with ANSI color codes.
pub fn format_diagnostics(diagnostics: &[RuntimeDiagnostic]) -> String {
    if diagnostics.is_empty() {
        return "\x1b[32mNo diagnostics. Graph looks good.\x1b[0m\n".to_string();
    }

    let mut out = String::new();

    let n_err = diagnostics.iter().filter(|d| d.severity == Severity::Error).count();
    let n_warn = diagnostics.iter().filter(|d| d.severity == Severity::Warning).count();
    let n_info = diagnostics.iter().filter(|d| d.severity == Severity::Info).count();

    out.push_str(&format!(
        "Found {} diagnostic(s): {} error(s), {} warning(s), {} info(s)\n\n",
        diagnostics.len(),
        n_err,
        n_warn,
        n_info
    ));

    for (i, d) in diagnostics.iter().enumerate() {
        let (color, label) = match d.severity {
            Severity::Error => ("\x1b[1;31m", "ERROR"),
            Severity::Warning => ("\x1b[1;33m", "WARNING"),
            Severity::Info => ("\x1b[1;34m", "INFO"),
        };
        let reset = "\x1b[0m";

        // Severity and index
        out.push_str(&format!("{color}[{label}]{reset} #{}\n", i + 1));

        // Node reference
        if let Some(nid) = d.node_id {
            out.push_str(&format!("  Node: {nid}\n"));
        }

        // Message
        out.push_str(&format!("  {}\n", d.message));

        // Suggestion
        if let Some(ref suggestion) = d.suggestion {
            out.push_str(&format!("  \x1b[36mSuggestion: {suggestion}{reset}\n"));
        }

        out.push('\n');
    }

    out
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use qlang_core::graph::Graph;
    use qlang_core::ops::Op;
    use qlang_core::tensor::{Dtype, Shape, TensorType};

    fn f32_vec(n: usize) -> TensorType {
        TensorType::new(Dtype::F32, Shape::vector(n))
    }

    fn f32_mat(m: usize, n: usize) -> TensorType {
        TensorType::new(Dtype::F32, Shape::matrix(m, n))
    }

    fn ternary_vec(n: usize) -> TensorType {
        TensorType::new(Dtype::Ternary, Shape::vector(n))
    }

    // -- DiagnosticCollector -------------------------------------------------

    #[test]
    fn collector_basics() {
        let mut c = DiagnosticCollector::new();
        assert!(c.is_empty());

        c.error("bad thing", Some(0));
        c.warning("suspicious thing", None);
        c.info("fyi", Some(1));

        assert_eq!(c.len(), 3);
        assert!(c.has_errors());
        assert_eq!(c.count(Severity::Error), 1);
        assert_eq!(c.count(Severity::Warning), 1);
        assert_eq!(c.count(Severity::Info), 1);
    }

    #[test]
    fn collector_into_diagnostics() {
        let mut c = DiagnosticCollector::new();
        c.push(RuntimeDiagnostic::error("oops").with_node(5));
        let diags = c.into_diagnostics();
        assert_eq!(diags.len(), 1);
        assert_eq!(diags[0].node_id, Some(5));
    }

    // -- Edge validity -------------------------------------------------------

    #[test]
    fn detects_invalid_source_node() {
        let mut g = Graph::new("test");
        let out = g.add_node(
            Op::Output { name: "y".into() },
            vec![f32_vec(4)],
            vec![],
        );
        // Edge from non-existent node 99 to the output.
        g.add_edge(99, 0, out, 0, f32_vec(4));

        let diags = validate_graph_deep(&g);
        let errors: Vec<_> = diags.iter().filter(|d| d.severity == Severity::Error).collect();
        assert!(
            errors.iter().any(|d| d.message.contains("source node 99")),
            "Expected error about missing source node, got: {errors:?}"
        );
    }

    #[test]
    fn detects_invalid_target_node() {
        let mut g = Graph::new("test");
        let inp = g.add_node(
            Op::Input { name: "x".into() },
            vec![],
            vec![f32_vec(4)],
        );
        // Edge to non-existent node 99.
        g.add_edge(inp, 0, 99, 0, f32_vec(4));

        let diags = validate_graph_deep(&g);
        let errors: Vec<_> = diags.iter().filter(|d| d.severity == Severity::Error).collect();
        assert!(
            errors.iter().any(|d| d.message.contains("target node 99")),
            "Expected error about missing target node, got: {errors:?}"
        );
    }

    #[test]
    fn detects_invalid_port() {
        let mut g = Graph::new("test");
        let inp = g.add_node(
            Op::Input { name: "x".into() },
            vec![],
            vec![f32_vec(4)],
        );
        let relu = g.add_node(Op::Relu, vec![f32_vec(4)], vec![f32_vec(4)]);
        // Relu has 1 input (port 0). Port 5 is invalid.
        g.add_edge(inp, 0, relu, 5, f32_vec(4));

        let diags = validate_graph_deep(&g);
        let errors: Vec<_> = diags.iter().filter(|d| d.severity == Severity::Error).collect();
        assert!(
            errors.iter().any(|d| d.message.contains("target port 5")),
            "Expected error about invalid port, got: {errors:?}"
        );
    }

    // -- Shape compatibility ------------------------------------------------

    #[test]
    fn detects_shape_mismatch_on_edge() {
        let mut g = Graph::new("test");
        let inp = g.add_node(
            Op::Input { name: "x".into() },
            vec![],
            vec![f32_vec(4)],
        );
        let relu = g.add_node(Op::Relu, vec![f32_vec(8)], vec![f32_vec(8)]);
        let out = g.add_node(
            Op::Output { name: "y".into() },
            vec![f32_vec(8)],
            vec![],
        );
        // Edge carries shape [4] but relu expects [8].
        g.add_edge(inp, 0, relu, 0, f32_vec(4));
        g.add_edge(relu, 0, out, 0, f32_vec(8));

        let diags = validate_graph_deep(&g);
        let errors: Vec<_> = diags.iter().filter(|d| d.severity == Severity::Error).collect();
        assert!(
            errors.iter().any(|d| d.message.contains("shape mismatch")),
            "Expected shape mismatch error, got: {errors:?}"
        );
    }

    // -- Unused nodes -------------------------------------------------------

    #[test]
    fn detects_unused_node() {
        let mut g = Graph::new("test");
        let inp = g.add_node(
            Op::Input { name: "x".into() },
            vec![],
            vec![f32_vec(4)],
        );
        let out = g.add_node(
            Op::Output { name: "y".into() },
            vec![f32_vec(4)],
            vec![],
        );
        // Add an orphan node not connected to anything.
        let _orphan = g.add_node(Op::Relu, vec![f32_vec(4)], vec![f32_vec(4)]);

        g.add_edge(inp, 0, out, 0, f32_vec(4));

        let diags = validate_graph_deep(&g);
        let unused: Vec<_> = diags.iter().filter(|d| d.message.contains("unused")).collect();
        assert!(
            !unused.is_empty(),
            "Expected warning or info about unused node, got: {diags:?}"
        );
    }

    #[test]
    fn no_unused_warning_for_connected_graph() {
        let mut g = Graph::new("test");
        let inp = g.add_node(
            Op::Input { name: "x".into() },
            vec![],
            vec![f32_vec(4)],
        );
        let relu = g.add_node(Op::Relu, vec![f32_vec(4)], vec![f32_vec(4)]);
        let out = g.add_node(
            Op::Output { name: "y".into() },
            vec![f32_vec(4)],
            vec![],
        );
        g.add_edge(inp, 0, relu, 0, f32_vec(4));
        g.add_edge(relu, 0, out, 0, f32_vec(4));

        let diags = validate_graph_deep(&g);
        let unused_warnings: Vec<_> = diags
            .iter()
            .filter(|d| d.message.contains("unused"))
            .collect();
        assert!(
            unused_warnings.is_empty(),
            "No unused warnings expected, got: {unused_warnings:?}"
        );
    }

    // -- Type mismatches ----------------------------------------------------

    #[test]
    fn detects_ternary_input_to_matmul() {
        let mut g = Graph::new("test");
        let inp = g.add_node(
            Op::Input { name: "x".into() },
            vec![],
            vec![ternary_vec(4)],
        );
        let matmul = g.add_node(
            Op::MatMul,
            vec![f32_mat(4, 4), f32_mat(4, 4)],
            vec![f32_mat(4, 4)],
        );
        // Feed ternary data into matmul.
        g.add_edge(inp, 0, matmul, 0, ternary_vec(4));

        let diags = validate_graph_deep(&g);
        let errors: Vec<_> = diags
            .iter()
            .filter(|d| d.severity == Severity::Error && d.message.contains("ternary"))
            .collect();
        assert!(
            !errors.is_empty(),
            "Expected type mismatch error for ternary into matmul, got: {:?}",
            diags
        );
        // Should suggest a cast.
        assert!(
            errors[0].suggestion.as_ref().unwrap().contains("cast"),
            "Expected suggestion about type cast"
        );
    }

    #[test]
    fn no_type_mismatch_for_float_to_matmul() {
        let mut g = Graph::new("test");
        let a = g.add_node(
            Op::Input { name: "a".into() },
            vec![],
            vec![f32_mat(4, 4)],
        );
        let b = g.add_node(
            Op::Input { name: "b".into() },
            vec![],
            vec![f32_mat(4, 4)],
        );
        let mm = g.add_node(
            Op::MatMul,
            vec![f32_mat(4, 4), f32_mat(4, 4)],
            vec![f32_mat(4, 4)],
        );
        let out = g.add_node(
            Op::Output { name: "y".into() },
            vec![f32_mat(4, 4)],
            vec![],
        );
        g.add_edge(a, 0, mm, 0, f32_mat(4, 4));
        g.add_edge(b, 0, mm, 1, f32_mat(4, 4));
        g.add_edge(mm, 0, out, 0, f32_mat(4, 4));

        let diags = validate_graph_deep(&g);
        let type_errors: Vec<_> = diags
            .iter()
            .filter(|d| d.severity == Severity::Error && d.message.contains("requires a floating"))
            .collect();
        assert!(
            type_errors.is_empty(),
            "No type mismatch expected for f32 matmul, got: {type_errors:?}"
        );
    }

    // -- Numerical issues ---------------------------------------------------

    #[test]
    fn detects_zero_dimension() {
        let mut g = Graph::new("test");
        let zero_type = TensorType::new(Dtype::F32, Shape::vector(0));
        g.add_node(
            Op::Input { name: "x".into() },
            vec![],
            vec![zero_type],
        );

        let diags = validate_graph_deep(&g);
        let errors: Vec<_> = diags
            .iter()
            .filter(|d| d.severity == Severity::Error && d.message.contains("zero"))
            .collect();
        assert!(
            !errors.is_empty(),
            "Expected error about zero-sized dimension, got: {diags:?}"
        );
    }

    #[test]
    fn warns_large_dimension() {
        let mut g = Graph::new("test");
        let big_type = TensorType::new(Dtype::F32, Shape::vector(2_000_000));
        let inp = g.add_node(
            Op::Input { name: "x".into() },
            vec![],
            vec![big_type.clone()],
        );
        let out = g.add_node(
            Op::Output { name: "y".into() },
            vec![big_type.clone()],
            vec![],
        );
        g.add_edge(inp, 0, out, 0, big_type);

        let diags = validate_graph_deep(&g);
        let warnings: Vec<_> = diags
            .iter()
            .filter(|d| d.severity == Severity::Warning && d.message.contains("2000000"))
            .collect();
        assert!(
            !warnings.is_empty(),
            "Expected warning about large dimension, got: {diags:?}"
        );
    }

    // -- format_diagnostics -------------------------------------------------

    #[test]
    fn format_empty_diagnostics() {
        let out = format_diagnostics(&[]);
        assert!(out.contains("No diagnostics"));
    }

    #[test]
    fn format_contains_severity_labels() {
        let diags = vec![
            RuntimeDiagnostic::error("bad").with_node(0),
            RuntimeDiagnostic::warning("hmm").with_node(1).with_suggestion("try this"),
            RuntimeDiagnostic::info("fyi"),
        ];
        let out = format_diagnostics(&diags);
        assert!(out.contains("ERROR"), "output should contain ERROR label");
        assert!(out.contains("WARNING"), "output should contain WARNING label");
        assert!(out.contains("INFO"), "output should contain INFO label");
        assert!(out.contains("3 diagnostic(s)"));
        assert!(out.contains("1 error(s)"));
        assert!(out.contains("Suggestion: try this"));
    }

    #[test]
    fn format_ansi_colors() {
        let diags = vec![RuntimeDiagnostic::error("oops")];
        let out = format_diagnostics(&diags);
        // Check for red ANSI escape (error color).
        assert!(out.contains("\x1b[1;31m"), "expected red ANSI code for error");
        assert!(out.contains("\x1b[0m"), "expected ANSI reset code");
    }

    // -- RuntimeDiagnostic Display ------------------------------------------

    #[test]
    fn diagnostic_display() {
        let d = RuntimeDiagnostic::error("broken")
            .with_node(42)
            .with_suggestion("fix it");
        let s = d.to_string();
        assert!(s.contains("[error]"));
        assert!(s.contains("node 42"));
        assert!(s.contains("broken"));
        assert!(s.contains("fix it"));
    }

    // -- Full graph validation (integration) --------------------------------

    #[test]
    fn valid_graph_has_no_errors() {
        let mut g = Graph::new("good");
        let inp = g.add_node(
            Op::Input { name: "x".into() },
            vec![],
            vec![f32_vec(4)],
        );
        let relu = g.add_node(Op::Relu, vec![f32_vec(4)], vec![f32_vec(4)]);
        let out = g.add_node(
            Op::Output { name: "y".into() },
            vec![f32_vec(4)],
            vec![],
        );
        g.add_edge(inp, 0, relu, 0, f32_vec(4));
        g.add_edge(relu, 0, out, 0, f32_vec(4));

        let diags = validate_graph_deep(&g);
        let errors: Vec<_> = diags.iter().filter(|d| d.severity == Severity::Error).collect();
        assert!(errors.is_empty(), "Valid graph should have no errors, got: {errors:?}");
    }
}
