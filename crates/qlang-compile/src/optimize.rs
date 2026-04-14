use std::collections::{HashMap, HashSet};

use qlang_core::graph::{Graph, NodeId};
use qlang_core::ops::Op;
use qlang_core::tensor::TensorType;

/// Report summarizing what each optimization pass accomplished.
#[derive(Debug, Clone, Default)]
pub struct OptimizationReport {
    /// Number of dead nodes removed.
    pub dead_nodes_removed: usize,
    /// Number of constant-folded operations.
    pub constants_folded: usize,
    /// Number of fused operation pairs.
    pub ops_fused: usize,
    /// Descriptions of fused pairs (e.g. "MatMul+Add -> FusedMatMulAdd").
    pub fused_descriptions: Vec<String>,
    /// Number of identity operations removed.
    pub identity_ops_removed: usize,
    /// Number of common subexpressions eliminated.
    pub common_subexpressions_eliminated: usize,
}

/// Run all optimization passes on a graph and return a report.
pub fn optimize(graph: &mut Graph) -> OptimizationReport {
    let mut report = OptimizationReport::default();

    report.dead_nodes_removed = eliminate_dead_nodes(graph);
    report.constants_folded = constant_folding(graph);
    report.identity_ops_removed = eliminate_identity_ops(graph);
    report.common_subexpressions_eliminated = common_subexpression_elimination(graph);
    let (fused, descriptions) = fuse_operations(graph);
    report.ops_fused = fused;
    report.fused_descriptions = descriptions;
    // Final dead-node pass to clean up anything left behind.
    report.dead_nodes_removed += eliminate_dead_nodes(graph);

    report
}

// ---------------------------------------------------------------------------
// Pass 1: Dead node elimination
// ---------------------------------------------------------------------------

/// Remove nodes not reachable from any Output node.
/// Returns the number of nodes removed.
pub fn eliminate_dead_nodes(graph: &mut Graph) -> usize {
    let mut total_removed = 0;
    loop {
        let dead: Vec<NodeId> = graph
            .nodes
            .iter()
            .filter(|n| {
                !matches!(n.op, Op::Output { .. })
                    && graph.outgoing_edges(n.id).is_empty()
            })
            .map(|n| n.id)
            .collect();

        if dead.is_empty() {
            break;
        }

        total_removed += dead.len();
        for &id in &dead {
            graph.edges.retain(|e| e.to_node != id && e.from_node != id);
        }
        graph.nodes.retain(|n| !dead.contains(&n.id));
    }
    total_removed
}

// ---------------------------------------------------------------------------
// Pass 2: Constant folding
// ---------------------------------------------------------------------------

/// If both inputs to a binary op (Add, Sub, Mul) are Constant nodes, replace
/// the op node with a single Constant node. We do not evaluate actual data
/// here (the graph carries only types, not values), but we mark the result as
/// a Constant and remove the upstream constant pair when they become dead.
///
/// Returns the number of operations folded.
pub fn constant_folding(graph: &mut Graph) -> usize {
    let mut folded = 0;

    // Identify binary ops whose inputs are ALL Constant nodes.
    let candidates: Vec<NodeId> = graph
        .nodes
        .iter()
        .filter(|n| matches!(n.op, Op::Add | Op::Sub | Op::Mul | Op::Div))
        .filter(|n| {
            let incoming = graph.incoming_edges(n.id);
            if incoming.is_empty() {
                return false;
            }
            incoming.iter().all(|e| {
                graph
                    .node(e.from_node)
                    .map(|src| matches!(src.op, Op::Constant))
                    .unwrap_or(false)
            })
        })
        .map(|n| n.id)
        .collect();

    for node_id in candidates {
        // Turn this op into a Constant.
        if let Some(node) = graph.nodes.iter_mut().find(|n| n.id == node_id) {
            node.op = Op::Constant;
            node.input_types.clear();
            node.metadata
                .insert("folded".to_string(), "true".to_string());
        }
        // Remove incoming edges (the constant inputs).
        graph.edges.retain(|e| e.to_node != node_id);
        folded += 1;
    }

    folded
}

// ---------------------------------------------------------------------------
// Pass 3: Operation fusion
// ---------------------------------------------------------------------------

/// Fuse common operation pairs into fused operations.
///   - MatMul followed by Add  -> metadata "fused: FusedMatMulAdd"
///   - MatMul followed by Relu -> metadata "fused: FusedMatMulRelu"
///
/// The graph structure is left intact (no new Op variants needed); fusion is
/// recorded as metadata on the first node of the pair. Code generation can
/// inspect this metadata to emit a single fused instruction.
///
/// Returns (count, descriptions).
pub fn fuse_operations(graph: &mut Graph) -> (usize, Vec<String>) {
    let mut fused_count = 0;
    let mut descriptions = Vec::new();

    // Collect fusable pairs first to avoid borrow issues.
    let pairs: Vec<(NodeId, NodeId, String)> = graph
        .nodes
        .iter()
        .filter(|n| matches!(n.op, Op::MatMul))
        .flat_map(|matmul_node| {
            let outgoing = graph.outgoing_edges(matmul_node.id);
            outgoing
                .into_iter()
                .filter_map(|edge| {
                    let target = graph.node(edge.to_node)?;
                    // Only fuse when the target has a single input from this matmul.
                    let target_incoming = graph.incoming_edges(target.id);
                    if target_incoming.len() == 1
                        || (matches!(target.op, Op::Add) && target_incoming.len() == 2)
                    {
                        match &target.op {
                            Op::Add => Some((
                                matmul_node.id,
                                target.id,
                                "FusedMatMulAdd".to_string(),
                            )),
                            Op::Relu => Some((
                                matmul_node.id,
                                target.id,
                                "FusedMatMulRelu".to_string(),
                            )),
                            _ => None,
                        }
                    } else {
                        None
                    }
                })
                .collect::<Vec<_>>()
        })
        .collect();

    // Already-fused nodes (don't fuse the same node twice).
    let mut fused_targets: HashSet<NodeId> = HashSet::new();

    for (matmul_id, target_id, fusion_name) in pairs {
        if fused_targets.contains(&target_id) {
            continue;
        }
        if let Some(node) = graph.nodes.iter_mut().find(|n| n.id == matmul_id) {
            node.metadata
                .insert("fused".to_string(), fusion_name.clone());
        }
        if let Some(node) = graph.nodes.iter_mut().find(|n| n.id == target_id) {
            node.metadata
                .insert("fused_into".to_string(), matmul_id.to_string());
        }
        descriptions.push(format!(
            "MatMul({}) + target({}) -> {}",
            matmul_id, target_id, fusion_name
        ));
        fused_targets.insert(target_id);
        fused_count += 1;
    }

    (fused_count, descriptions)
}

// ---------------------------------------------------------------------------
// Pass 4: Identity operation elimination
// ---------------------------------------------------------------------------

/// Remove identity operations:
///   - Add(x, 0): when one input is a zero constant, bypass the Add
///   - Mul(x, 1): when one input is a one constant (via metadata), bypass Mul
///   - Neg(Neg(x)): double negation cancels out
///
/// Returns the number of identity ops removed.
pub fn eliminate_identity_ops(graph: &mut Graph) -> usize {
    let mut removed = 0;

    // --- Neg(Neg(x)) ---
    // Find Neg nodes whose single input comes from another Neg node.
    let double_negs: Vec<(NodeId, NodeId, NodeId)> = graph
        .nodes
        .iter()
        .filter(|n| matches!(n.op, Op::Neg))
        .filter_map(|outer_neg| {
            let incoming = graph.incoming_edges(outer_neg.id);
            if incoming.len() != 1 {
                return None;
            }
            let inner_id = incoming[0].from_node;
            let inner = graph.node(inner_id)?;
            if !matches!(inner.op, Op::Neg) {
                return None;
            }
            // inner_neg's input is the original value.
            let inner_incoming = graph.incoming_edges(inner_id);
            if inner_incoming.len() != 1 {
                return None;
            }
            let original_source = inner_incoming[0].from_node;
            let _original_port = inner_incoming[0].from_port;
            Some((outer_neg.id, inner_id, original_source))
        })
        .collect();

    for (outer_id, _inner_id, source_id) in double_negs {
        // Rewire: anything consuming outer_neg now consumes source_id instead.
        let outgoing: Vec<(NodeId, u8, u8, TensorType)> = graph
            .outgoing_edges(outer_id)
            .iter()
            .map(|e| (e.to_node, e.to_port, e.from_port, e.tensor_type.clone()))
            .collect();

        for (to_node, to_port, _from_port, tensor_type) in outgoing {
            graph.edges.retain(|e| !(e.from_node == outer_id && e.to_node == to_node && e.to_port == to_port));
            graph.add_edge(source_id, 0, to_node, to_port, tensor_type);
        }
        removed += 1;
    }

    // --- Add(x, 0) and Mul(x, 1) ---
    // A Constant node with metadata "value" = "0" or "1" serves as the
    // identity element.  We look for binary ops where one operand is such a
    // constant.
    let identity_ops: Vec<(NodeId, NodeId)> = graph
        .nodes
        .iter()
        .filter(|n| matches!(n.op, Op::Add | Op::Mul))
        .filter_map(|op_node| {
            let incoming = graph.incoming_edges(op_node.id);
            if incoming.len() != 2 {
                return None;
            }

            let src0 = graph.node(incoming[0].from_node)?;
            let src1 = graph.node(incoming[1].from_node)?;

            let is_zero = |n: &qlang_core::graph::Node| -> bool {
                matches!(n.op, Op::Constant)
                    && n.metadata.get("value").map(|v| v == "0").unwrap_or(false)
            };
            let is_one = |n: &qlang_core::graph::Node| -> bool {
                matches!(n.op, Op::Constant)
                    && n.metadata.get("value").map(|v| v == "1").unwrap_or(false)
            };

            match &op_node.op {
                Op::Add => {
                    if is_zero(src0) {
                        Some((op_node.id, incoming[1].from_node))
                    } else if is_zero(src1) {
                        Some((op_node.id, incoming[0].from_node))
                    } else {
                        None
                    }
                }
                Op::Mul => {
                    if is_one(src0) {
                        Some((op_node.id, incoming[1].from_node))
                    } else if is_one(src1) {
                        Some((op_node.id, incoming[0].from_node))
                    } else {
                        None
                    }
                }
                _ => None,
            }
        })
        .collect();

    for (op_id, passthrough_id) in identity_ops {
        let outgoing: Vec<(NodeId, u8, TensorType)> = graph
            .outgoing_edges(op_id)
            .iter()
            .map(|e| (e.to_node, e.to_port, e.tensor_type.clone()))
            .collect();

        for (to_node, to_port, tensor_type) in outgoing {
            graph.edges.retain(|e| !(e.from_node == op_id && e.to_node == to_node && e.to_port == to_port));
            graph.add_edge(passthrough_id, 0, to_node, to_port, tensor_type);
        }
        removed += 1;
    }

    removed
}

// ---------------------------------------------------------------------------
// Pass 5: Common subexpression elimination
// ---------------------------------------------------------------------------

/// If two nodes have the same operation and the same inputs (same source
/// nodes and ports, in the same order), keep one and rewire consumers of the
/// duplicate to the kept node.
///
/// Returns the number of duplicate nodes eliminated.
pub fn common_subexpression_elimination(graph: &mut Graph) -> usize {
    let mut eliminated = 0;

    // Build a signature for each node: (op_debug_string, sorted_input_edges).
    // We use the Debug representation of Op for a hashable key since Op does
    // not implement Hash (contains f64 fields).
    let mut signatures: HashMap<String, NodeId> = HashMap::new();
    let mut rewrites: Vec<(NodeId, NodeId)> = Vec::new(); // (duplicate, canonical)

    // Process in topological order so earlier nodes become canonical.
    let order = match graph.topological_sort() {
        Ok(o) => o,
        Err(_) => return 0,
    };

    for &node_id in &order {
        let node = match graph.node(node_id) {
            Some(n) => n,
            None => continue,
        };

        // Skip I/O and non-deterministic ops.
        if matches!(node.op, Op::Input { .. } | Op::Output { .. } | Op::Constant) {
            continue;
        }
        if !node.op.is_deterministic() {
            continue;
        }

        let mut incoming: Vec<(NodeId, u8, u8)> = graph
            .incoming_edges(node_id)
            .iter()
            .map(|e| (e.from_node, e.from_port, e.to_port))
            .collect();
        incoming.sort();

        let sig = format!("{:?}|{:?}", node.op, incoming);

        if let Some(&canonical_id) = signatures.get(&sig) {
            rewrites.push((node_id, canonical_id));
        } else {
            signatures.insert(sig, node_id);
        }
    }

    for (dup_id, canonical_id) in rewrites {
        // Rewire all consumers of dup_id to canonical_id.
        let outgoing: Vec<(NodeId, u8, u8, TensorType)> = graph
            .outgoing_edges(dup_id)
            .iter()
            .map(|e| (e.to_node, e.to_port, e.from_port, e.tensor_type.clone()))
            .collect();

        for (to_node, to_port, from_port, tensor_type) in outgoing {
            graph.edges.retain(|e| {
                !(e.from_node == dup_id && e.to_node == to_node && e.to_port == to_port)
            });
            graph.add_edge(canonical_id, from_port, to_node, to_port, tensor_type);
        }
        eliminated += 1;
    }

    eliminated
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use qlang_core::graph::Graph;
    use qlang_core::ops::Op;
    use qlang_core::tensor::TensorType;

    fn f32v4() -> TensorType {
        TensorType::f32_vector(4)
    }

    // -----------------------------------------------------------------------
    // Dead node elimination
    // -----------------------------------------------------------------------

    #[test]
    fn dead_node_elimination_removes_unreachable() {
        let mut g = Graph::new("dce_test");

        let input = g.add_node(
            Op::Input { name: "x".into() },
            vec![],
            vec![f32v4()],
        );
        let relu = g.add_node(Op::Relu, vec![f32v4()], vec![f32v4()]);
        // Dead node: not connected to anything downstream.
        let _dead = g.add_node(Op::Neg, vec![f32v4()], vec![f32v4()]);
        let out = g.add_node(
            Op::Output { name: "y".into() },
            vec![f32v4()],
            vec![],
        );

        g.add_edge(input, 0, relu, 0, f32v4());
        g.add_edge(relu, 0, out, 0, f32v4());

        assert_eq!(g.nodes.len(), 4);
        let removed = eliminate_dead_nodes(&mut g);
        assert_eq!(removed, 1);
        assert_eq!(g.nodes.len(), 3);
        // The dead Neg node should be gone.
        assert!(g.nodes.iter().all(|n| !matches!(n.op, Op::Neg)));
    }

    #[test]
    fn dead_node_chain_removed() {
        let mut g = Graph::new("dce_chain");

        let input = g.add_node(Op::Input { name: "x".into() }, vec![], vec![f32v4()]);
        let out = g.add_node(Op::Output { name: "y".into() }, vec![f32v4()], vec![]);
        g.add_edge(input, 0, out, 0, f32v4());

        // Dead chain: A -> B -> (nowhere)
        let dead_a = g.add_node(Op::Relu, vec![f32v4()], vec![f32v4()]);
        let dead_b = g.add_node(Op::Neg, vec![f32v4()], vec![f32v4()]);
        g.add_edge(dead_a, 0, dead_b, 0, f32v4());

        assert_eq!(g.nodes.len(), 4);
        let removed = eliminate_dead_nodes(&mut g);
        assert_eq!(removed, 2);
        assert_eq!(g.nodes.len(), 2);
    }

    // -----------------------------------------------------------------------
    // Constant folding
    // -----------------------------------------------------------------------

    #[test]
    fn constant_folding_folds_add_of_two_constants() {
        let mut g = Graph::new("cf_test");

        let c1 = g.add_node(Op::Constant, vec![], vec![f32v4()]);
        let c2 = g.add_node(Op::Constant, vec![], vec![f32v4()]);
        let add = g.add_node(Op::Add, vec![f32v4(), f32v4()], vec![f32v4()]);
        let out = g.add_node(Op::Output { name: "y".into() }, vec![f32v4()], vec![]);

        g.add_edge(c1, 0, add, 0, f32v4());
        g.add_edge(c2, 0, add, 1, f32v4());
        g.add_edge(add, 0, out, 0, f32v4());

        let folded = constant_folding(&mut g);
        assert_eq!(folded, 1);
        // The add node should now be a Constant.
        let add_node = g.node(add).unwrap();
        assert!(matches!(add_node.op, Op::Constant));
        assert_eq!(add_node.metadata.get("folded").unwrap(), "true");
    }

    #[test]
    fn constant_folding_does_not_fold_non_constant_input() {
        let mut g = Graph::new("cf_no_fold");

        let inp = g.add_node(Op::Input { name: "x".into() }, vec![], vec![f32v4()]);
        let c = g.add_node(Op::Constant, vec![], vec![f32v4()]);
        let add = g.add_node(Op::Add, vec![f32v4(), f32v4()], vec![f32v4()]);
        let out = g.add_node(Op::Output { name: "y".into() }, vec![f32v4()], vec![]);

        g.add_edge(inp, 0, add, 0, f32v4());
        g.add_edge(c, 0, add, 1, f32v4());
        g.add_edge(add, 0, out, 0, f32v4());

        let folded = constant_folding(&mut g);
        assert_eq!(folded, 0);
        assert!(matches!(g.node(add).unwrap().op, Op::Add));
    }

    // -----------------------------------------------------------------------
    // Operation fusion
    // -----------------------------------------------------------------------

    #[test]
    fn fuse_matmul_add() {
        let mut g = Graph::new("fuse_test");

        let a = g.add_node(Op::Input { name: "a".into() }, vec![], vec![f32v4()]);
        let b = g.add_node(Op::Input { name: "b".into() }, vec![], vec![f32v4()]);
        let bias = g.add_node(Op::Input { name: "bias".into() }, vec![], vec![f32v4()]);
        let mm = g.add_node(Op::MatMul, vec![f32v4(), f32v4()], vec![f32v4()]);
        let add = g.add_node(Op::Add, vec![f32v4(), f32v4()], vec![f32v4()]);
        let out = g.add_node(Op::Output { name: "y".into() }, vec![f32v4()], vec![]);

        g.add_edge(a, 0, mm, 0, f32v4());
        g.add_edge(b, 0, mm, 1, f32v4());
        g.add_edge(mm, 0, add, 0, f32v4());
        g.add_edge(bias, 0, add, 1, f32v4());
        g.add_edge(add, 0, out, 0, f32v4());

        let (count, descs) = fuse_operations(&mut g);
        assert_eq!(count, 1);
        assert!(descs[0].contains("FusedMatMulAdd"));

        let mm_node = g.node(mm).unwrap();
        assert_eq!(mm_node.metadata.get("fused").unwrap(), "FusedMatMulAdd");
    }

    #[test]
    fn fuse_matmul_relu() {
        let mut g = Graph::new("fuse_relu");

        let a = g.add_node(Op::Input { name: "a".into() }, vec![], vec![f32v4()]);
        let b = g.add_node(Op::Input { name: "b".into() }, vec![], vec![f32v4()]);
        let mm = g.add_node(Op::MatMul, vec![f32v4(), f32v4()], vec![f32v4()]);
        let relu = g.add_node(Op::Relu, vec![f32v4()], vec![f32v4()]);
        let out = g.add_node(Op::Output { name: "y".into() }, vec![f32v4()], vec![]);

        g.add_edge(a, 0, mm, 0, f32v4());
        g.add_edge(b, 0, mm, 1, f32v4());
        g.add_edge(mm, 0, relu, 0, f32v4());
        g.add_edge(relu, 0, out, 0, f32v4());

        let (count, descs) = fuse_operations(&mut g);
        assert_eq!(count, 1);
        assert!(descs[0].contains("FusedMatMulRelu"));
    }

    // -----------------------------------------------------------------------
    // Identity operation elimination
    // -----------------------------------------------------------------------

    #[test]
    fn eliminate_add_zero() {
        let mut g = Graph::new("id_add_zero");

        let inp = g.add_node(Op::Input { name: "x".into() }, vec![], vec![f32v4()]);
        let zero = g.add_node(Op::Constant, vec![], vec![f32v4()]);
        // Mark constant as zero.
        g.nodes.iter_mut().find(|n| n.id == zero).unwrap()
            .metadata.insert("value".into(), "0".into());
        let add = g.add_node(Op::Add, vec![f32v4(), f32v4()], vec![f32v4()]);
        let out = g.add_node(Op::Output { name: "y".into() }, vec![f32v4()], vec![]);

        g.add_edge(inp, 0, add, 0, f32v4());
        g.add_edge(zero, 0, add, 1, f32v4());
        g.add_edge(add, 0, out, 0, f32v4());

        let removed = eliminate_identity_ops(&mut g);
        assert_eq!(removed, 1);
        // Output should now be fed directly from input.
        let out_incoming = g.incoming_edges(out);
        assert_eq!(out_incoming.len(), 1);
        assert_eq!(out_incoming[0].from_node, inp);
    }

    #[test]
    fn eliminate_mul_one() {
        let mut g = Graph::new("id_mul_one");

        let inp = g.add_node(Op::Input { name: "x".into() }, vec![], vec![f32v4()]);
        let one = g.add_node(Op::Constant, vec![], vec![f32v4()]);
        g.nodes.iter_mut().find(|n| n.id == one).unwrap()
            .metadata.insert("value".into(), "1".into());
        let mul = g.add_node(Op::Mul, vec![f32v4(), f32v4()], vec![f32v4()]);
        let out = g.add_node(Op::Output { name: "y".into() }, vec![f32v4()], vec![]);

        g.add_edge(inp, 0, mul, 0, f32v4());
        g.add_edge(one, 0, mul, 1, f32v4());
        g.add_edge(mul, 0, out, 0, f32v4());

        let removed = eliminate_identity_ops(&mut g);
        assert_eq!(removed, 1);
        let out_incoming = g.incoming_edges(out);
        assert_eq!(out_incoming.len(), 1);
        assert_eq!(out_incoming[0].from_node, inp);
    }

    #[test]
    fn eliminate_double_neg() {
        let mut g = Graph::new("id_neg_neg");

        let inp = g.add_node(Op::Input { name: "x".into() }, vec![], vec![f32v4()]);
        let neg1 = g.add_node(Op::Neg, vec![f32v4()], vec![f32v4()]);
        let neg2 = g.add_node(Op::Neg, vec![f32v4()], vec![f32v4()]);
        let out = g.add_node(Op::Output { name: "y".into() }, vec![f32v4()], vec![]);

        g.add_edge(inp, 0, neg1, 0, f32v4());
        g.add_edge(neg1, 0, neg2, 0, f32v4());
        g.add_edge(neg2, 0, out, 0, f32v4());

        let removed = eliminate_identity_ops(&mut g);
        assert_eq!(removed, 1);
        // Output should now be fed directly from input.
        let out_incoming = g.incoming_edges(out);
        assert_eq!(out_incoming.len(), 1);
        assert_eq!(out_incoming[0].from_node, inp);
    }

    // -----------------------------------------------------------------------
    // Common subexpression elimination
    // -----------------------------------------------------------------------

    #[test]
    fn cse_eliminates_duplicate_relu() {
        let mut g = Graph::new("cse_test");

        let inp = g.add_node(Op::Input { name: "x".into() }, vec![], vec![f32v4()]);
        let relu1 = g.add_node(Op::Relu, vec![f32v4()], vec![f32v4()]);
        let relu2 = g.add_node(Op::Relu, vec![f32v4()], vec![f32v4()]);
        let add = g.add_node(Op::Add, vec![f32v4(), f32v4()], vec![f32v4()]);
        let out = g.add_node(Op::Output { name: "y".into() }, vec![f32v4()], vec![]);

        g.add_edge(inp, 0, relu1, 0, f32v4());
        g.add_edge(inp, 0, relu2, 0, f32v4());
        g.add_edge(relu1, 0, add, 0, f32v4());
        g.add_edge(relu2, 0, add, 1, f32v4());
        g.add_edge(add, 0, out, 0, f32v4());

        let eliminated = common_subexpression_elimination(&mut g);
        assert_eq!(eliminated, 1);
        // Both inputs to Add should now come from the same relu node.
        let add_incoming = g.incoming_edges(add);
        assert_eq!(add_incoming.len(), 2);
        assert_eq!(add_incoming[0].from_node, add_incoming[1].from_node);
    }

    #[test]
    fn cse_does_not_eliminate_different_ops() {
        let mut g = Graph::new("cse_no_elim");

        let inp = g.add_node(Op::Input { name: "x".into() }, vec![], vec![f32v4()]);
        let relu = g.add_node(Op::Relu, vec![f32v4()], vec![f32v4()]);
        let neg = g.add_node(Op::Neg, vec![f32v4()], vec![f32v4()]);
        let add = g.add_node(Op::Add, vec![f32v4(), f32v4()], vec![f32v4()]);
        let out = g.add_node(Op::Output { name: "y".into() }, vec![f32v4()], vec![]);

        g.add_edge(inp, 0, relu, 0, f32v4());
        g.add_edge(inp, 0, neg, 0, f32v4());
        g.add_edge(relu, 0, add, 0, f32v4());
        g.add_edge(neg, 0, add, 1, f32v4());
        g.add_edge(add, 0, out, 0, f32v4());

        let eliminated = common_subexpression_elimination(&mut g);
        assert_eq!(eliminated, 0);
    }

    // -----------------------------------------------------------------------
    // Full pipeline
    // -----------------------------------------------------------------------

    #[test]
    fn full_optimize_pipeline() {
        let mut g = Graph::new("full_test");

        let input = g.add_node(
            Op::Input { name: "x".into() },
            vec![],
            vec![f32v4()],
        );
        let relu = g.add_node(Op::Relu, vec![f32v4()], vec![f32v4()]);
        let _dead = g.add_node(Op::Neg, vec![f32v4()], vec![f32v4()]);
        let out = g.add_node(
            Op::Output { name: "y".into() },
            vec![f32v4()],
            vec![],
        );

        g.add_edge(input, 0, relu, 0, f32v4());
        g.add_edge(relu, 0, out, 0, f32v4());

        assert_eq!(g.nodes.len(), 4);
        let report = optimize(&mut g);
        assert_eq!(g.nodes.len(), 3);
        assert!(report.dead_nodes_removed >= 1);
    }

    #[test]
    fn optimization_report_fields() {
        let report = OptimizationReport::default();
        assert_eq!(report.dead_nodes_removed, 0);
        assert_eq!(report.constants_folded, 0);
        assert_eq!(report.ops_fused, 0);
        assert_eq!(report.identity_ops_removed, 0);
        assert_eq!(report.common_subexpressions_eliminated, 0);
        assert!(report.fused_descriptions.is_empty());
    }
}
