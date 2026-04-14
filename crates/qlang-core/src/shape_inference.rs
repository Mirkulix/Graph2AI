//! Shape inference — automatically propagate tensor shapes through a QLANG graph.
//!
//! Given a graph with typed inputs, `infer_shapes` walks nodes in topological
//! order and computes the output `TensorType` for every node. `validate_shapes`
//! collects all shape mismatches without stopping at the first one.

use std::collections::HashMap;

use crate::graph::{Graph, NodeId};
use crate::ops::Op;
use crate::tensor::{Dim, Dtype, Shape, TensorType};

// ---------------------------------------------------------------------------
// Error types
// ---------------------------------------------------------------------------

/// Errors that can occur during shape inference or validation.
#[derive(Debug, Clone, thiserror::Error)]
pub enum ShapeError {
    #[error("node {node}: dimension mismatch on axis {axis}: expected {expected}, got {got}")]
    DimensionMismatch {
        node: NodeId,
        axis: usize,
        expected: usize,
        got: usize,
    },

    #[error("node {node}: incompatible shapes {a} and {b}")]
    IncompatibleShapes {
        node: NodeId,
        a: Shape,
        b: Shape,
    },

    #[error("node {node}: unknown shape — no input edges found")]
    UnknownShape { node: NodeId },

    #[error("node {node}: rank mismatch — expected {expected}, got {got}")]
    RankMismatch {
        node: NodeId,
        expected: usize,
        got: usize,
    },

    #[error("node {node}: axis {axis} out of bounds for rank {rank}")]
    AxisOutOfBounds {
        node: NodeId,
        axis: usize,
        rank: usize,
    },

    #[error("node {node}: {msg}")]
    Other { node: NodeId, msg: String },

    #[error("graph contains a cycle")]
    CycleDetected,
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Propagate shapes through every node in topological order.
///
/// Returns a map from `NodeId` to the list of output `TensorType`s produced by
/// that node. Input/Constant nodes use the types already declared in
/// `node.output_types`.
pub fn infer_shapes(graph: &Graph) -> Result<HashMap<NodeId, Vec<TensorType>>, ShapeError> {
    let order = graph
        .topological_sort()
        .map_err(|_| ShapeError::CycleDetected)?;

    // node_id → vec of output TensorTypes
    let mut shapes: HashMap<NodeId, Vec<TensorType>> = HashMap::new();

    for &nid in &order {
        let node = graph.node(nid).expect("topo sort returned unknown node");

        // Collect input types from predecessor edges (sorted by to_port).
        let mut incoming: Vec<(u8, TensorType)> = graph
            .incoming_edges(nid)
            .iter()
            .map(|e| {
                // Prefer the inferred output of the source node if available.
                let src_types = shapes.get(&e.from_node);
                let tt = src_types
                    .and_then(|v| v.get(e.from_port as usize))
                    .cloned()
                    .unwrap_or_else(|| e.tensor_type.clone());
                (e.to_port, tt)
            })
            .collect();
        incoming.sort_by_key(|(port, _)| *port);
        let inputs: Vec<TensorType> = incoming.into_iter().map(|(_, t)| t).collect();

        let outputs = infer_node(nid, &node.op, &inputs, &node.output_types)?;
        shapes.insert(nid, outputs);
    }

    Ok(shapes)
}

/// Validate every node's shapes and return all errors found.
pub fn validate_shapes(graph: &Graph) -> Vec<ShapeError> {
    let mut errors = Vec::new();

    let order = match graph.topological_sort() {
        Ok(o) => o,
        Err(_) => {
            errors.push(ShapeError::CycleDetected);
            return errors;
        }
    };

    let mut shapes: HashMap<NodeId, Vec<TensorType>> = HashMap::new();

    for &nid in &order {
        let node = graph.node(nid).expect("topo sort returned unknown node");

        let mut incoming: Vec<(u8, TensorType)> = graph
            .incoming_edges(nid)
            .iter()
            .map(|e| {
                let src_types = shapes.get(&e.from_node);
                let tt = src_types
                    .and_then(|v| v.get(e.from_port as usize))
                    .cloned()
                    .unwrap_or_else(|| e.tensor_type.clone());
                (e.to_port, tt)
            })
            .collect();
        incoming.sort_by_key(|(port, _)| *port);
        let inputs: Vec<TensorType> = incoming.into_iter().map(|(_, t)| t).collect();

        match infer_node(nid, &node.op, &inputs, &node.output_types) {
            Ok(out) => {
                shapes.insert(nid, out);
            }
            Err(e) => {
                errors.push(e);
            }
        }
    }

    errors
}

// ---------------------------------------------------------------------------
// Per-op inference
// ---------------------------------------------------------------------------

fn infer_node(
    nid: NodeId,
    op: &Op,
    inputs: &[TensorType],
    declared_outputs: &[TensorType],
) -> Result<Vec<TensorType>, ShapeError> {
    match op {
        // --- I/O & constants: trust declared types ---
        Op::Input { .. } | Op::Constant => Ok(declared_outputs.to_vec()),

        Op::Output { .. } => {
            // Output nodes produce nothing; just pass through for bookkeeping.
            Ok(vec![])
        }

        // --- Element-wise binary: shapes must match ---
        Op::Add | Op::Sub | Op::Mul | Op::Div => {
            let (a, b) = binary_inputs(nid, inputs)?;
            check_compatible(nid, &a.shape, &b.shape)?;
            Ok(vec![TensorType::new(a.dtype, a.shape.clone())])
        }

        // --- Element-wise unary: shape preserved ---
        Op::Neg | Op::Relu | Op::Sigmoid | Op::Tanh | Op::Gelu | Op::Dropout { .. } => {
            let a = unary_input(nid, inputs)?;
            Ok(vec![a.clone()])
        }

        Op::Softmax { axis } => {
            let a = unary_input(nid, inputs)?;
            check_axis(nid, *axis, a.shape.rank())?;
            Ok(vec![a.clone()])
        }

        // --- MatMul: [m,k] x [k,n] -> [m,n] ---
        Op::MatMul => {
            let (a, b) = binary_inputs(nid, inputs)?;
            if a.shape.rank() != 2 {
                return Err(ShapeError::RankMismatch { node: nid, expected: 2, got: a.shape.rank() });
            }
            if b.shape.rank() != 2 {
                return Err(ShapeError::RankMismatch { node: nid, expected: 2, got: b.shape.rank() });
            }
            // Check inner dims match
            check_dim_match(nid, 1, &a.shape.0[1], &b.shape.0[0])?;
            let m = a.shape.0[0];
            let n = b.shape.0[1];
            Ok(vec![TensorType::new(a.dtype, Shape(vec![m, n]))])
        }

        // --- Transpose: [m,n] -> [n,m] ---
        Op::Transpose => {
            let a = unary_input(nid, inputs)?;
            if a.shape.rank() != 2 {
                return Err(ShapeError::RankMismatch { node: nid, expected: 2, got: a.shape.rank() });
            }
            let reversed: Vec<Dim> = a.shape.0.iter().copied().rev().collect();
            Ok(vec![TensorType::new(a.dtype, Shape(reversed))])
        }

        // --- Reshape ---
        Op::Reshape { target_shape } => {
            let a = unary_input(nid, inputs)?;
            let new_shape = Shape(target_shape.iter().map(|&d| Dim::Fixed(d)).collect());
            Ok(vec![TensorType::new(a.dtype, new_shape)])
        }

        // --- Reduce operations ---
        Op::ReduceSum { axis } | Op::ReduceMean { axis } | Op::ReduceMax { axis } => {
            let a = unary_input(nid, inputs)?;
            match axis {
                None => {
                    // Reduce to scalar
                    Ok(vec![TensorType::new(a.dtype, Shape::scalar())])
                }
                Some(ax) => {
                    check_axis(nid, *ax, a.shape.rank())?;
                    let new_dims: Vec<Dim> = a
                        .shape
                        .0
                        .iter()
                        .enumerate()
                        .filter(|(i, _)| *i != *ax)
                        .map(|(_, d)| *d)
                        .collect();
                    Ok(vec![TensorType::new(a.dtype, Shape(new_dims))])
                }
            }
        }

        // --- ToTernary: same shape, dtype changes ---
        Op::ToTernary => {
            let a = unary_input(nid, inputs)?;
            Ok(vec![TensorType::new(Dtype::Ternary, a.shape.clone())])
        }

        // --- LayerNorm: same shape ---
        Op::LayerNorm { .. } => {
            let a = unary_input(nid, inputs)?;
            Ok(vec![a.clone()])
        }

        // --- Concat along axis ---
        Op::Concat { axis } => {
            let (a, b) = binary_inputs(nid, inputs)?;
            check_axis(nid, *axis, a.shape.rank())?;
            if a.shape.rank() != b.shape.rank() {
                return Err(ShapeError::IncompatibleShapes {
                    node: nid,
                    a: a.shape.clone(),
                    b: b.shape.clone(),
                });
            }
            // All dims except the concat axis must match.
            for (i, (da, db)) in a.shape.0.iter().zip(b.shape.0.iter()).enumerate() {
                if i != *axis {
                    check_dim_match(nid, i, da, db)?;
                }
            }
            let mut new_dims = a.shape.0.clone();
            new_dims[*axis] = add_dims(&a.shape.0[*axis], &b.shape.0[*axis]);
            Ok(vec![TensorType::new(a.dtype, Shape(new_dims))])
        }

        // --- Residual: x + f(x), shapes must match ---
        Op::Residual => {
            let (a, b) = binary_inputs(nid, inputs)?;
            check_compatible(nid, &a.shape, &b.shape)?;
            Ok(vec![TensorType::new(a.dtype, a.shape.clone())])
        }

        // --- Slice ---
        Op::Slice { start, end } => {
            let a = unary_input(nid, inputs)?;
            let new_dims: Vec<Dim> = start
                .iter()
                .zip(end.iter())
                .map(|(&s, &e)| Dim::Fixed(e - s))
                .collect();
            Ok(vec![TensorType::new(a.dtype, Shape(new_dims))])
        }

        // --- Fallback: trust declared output types ---
        _ => Ok(declared_outputs.to_vec()),
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn unary_input(nid: NodeId, inputs: &[TensorType]) -> Result<&TensorType, ShapeError> {
    inputs
        .first()
        .ok_or(ShapeError::UnknownShape { node: nid })
}

fn binary_inputs(nid: NodeId, inputs: &[TensorType]) -> Result<(&TensorType, &TensorType), ShapeError> {
    if inputs.len() < 2 {
        return Err(ShapeError::UnknownShape { node: nid });
    }
    Ok((&inputs[0], &inputs[1]))
}

fn check_compatible(nid: NodeId, a: &Shape, b: &Shape) -> Result<(), ShapeError> {
    if !a.is_compatible_with(b) {
        return Err(ShapeError::IncompatibleShapes {
            node: nid,
            a: a.clone(),
            b: b.clone(),
        });
    }
    Ok(())
}

fn check_axis(nid: NodeId, axis: usize, rank: usize) -> Result<(), ShapeError> {
    if axis >= rank {
        return Err(ShapeError::AxisOutOfBounds { node: nid, axis, rank });
    }
    Ok(())
}

fn check_dim_match(nid: NodeId, axis: usize, a: &Dim, b: &Dim) -> Result<(), ShapeError> {
    match (a, b) {
        (Dim::Fixed(x), Dim::Fixed(y)) if x != y => Err(ShapeError::DimensionMismatch {
            node: nid,
            axis,
            expected: *x,
            got: *y,
        }),
        _ => Ok(()),
    }
}

fn add_dims(a: &Dim, b: &Dim) -> Dim {
    match (a, b) {
        (Dim::Fixed(x), Dim::Fixed(y)) => Dim::Fixed(x + y),
        _ => Dim::Dynamic,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::Graph;
    use crate::ops::Op;
    use crate::tensor::{Dtype, Shape, TensorType};

    fn f32_vec(n: usize) -> TensorType {
        TensorType::f32_vector(n)
    }

    fn f32_mat(m: usize, n: usize) -> TensorType {
        TensorType::f32_matrix(m, n)
    }

    /// Helper: build a unary graph (input -> op -> output) and infer.
    fn infer_unary(op: Op, input_tt: TensorType) -> HashMap<NodeId, Vec<TensorType>> {
        let mut g = Graph::new("test");
        let inp = g.add_node(Op::Input { name: "x".into() }, vec![], vec![input_tt.clone()]);
        let mid = g.add_node(op, vec![input_tt.clone()], vec![input_tt.clone()]);
        let out = g.add_node(Op::Output { name: "y".into() }, vec![input_tt.clone()], vec![]);
        g.add_edge(inp, 0, mid, 0, input_tt.clone());
        g.add_edge(mid, 0, out, 0, input_tt);
        infer_shapes(&g).unwrap()
    }

    /// Helper: build a binary graph (two inputs -> op -> output) and infer.
    fn infer_binary(op: Op, a: TensorType, b: TensorType, out_tt: TensorType) -> HashMap<NodeId, Vec<TensorType>> {
        let mut g = Graph::new("test");
        let i0 = g.add_node(Op::Input { name: "a".into() }, vec![], vec![a.clone()]);
        let i1 = g.add_node(Op::Input { name: "b".into() }, vec![], vec![b.clone()]);
        let mid = g.add_node(op, vec![a.clone(), b.clone()], vec![out_tt.clone()]);
        let out = g.add_node(Op::Output { name: "y".into() }, vec![out_tt.clone()], vec![]);
        g.add_edge(i0, 0, mid, 0, a);
        g.add_edge(i1, 0, mid, 1, b);
        g.add_edge(mid, 0, out, 0, out_tt);
        infer_shapes(&g).unwrap()
    }

    // --- 1. Add: same-shape element-wise ---
    #[test]
    fn test_add_shapes() {
        let tt = f32_vec(4);
        let shapes = infer_binary(Op::Add, tt.clone(), tt.clone(), tt.clone());
        assert_eq!(shapes[&2], vec![tt]);
    }

    // --- 2. Add: incompatible shapes produces error ---
    #[test]
    fn test_add_incompatible() {
        let mut g = Graph::new("test");
        let i0 = g.add_node(Op::Input { name: "a".into() }, vec![], vec![f32_vec(4)]);
        let i1 = g.add_node(Op::Input { name: "b".into() }, vec![], vec![f32_vec(5)]);
        let add = g.add_node(Op::Add, vec![f32_vec(4), f32_vec(5)], vec![f32_vec(4)]);
        g.add_edge(i0, 0, add, 0, f32_vec(4));
        g.add_edge(i1, 0, add, 1, f32_vec(5));
        let result = infer_shapes(&g);
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), ShapeError::IncompatibleShapes { .. }));
    }

    // --- 3. MatMul: [m,k] x [k,n] -> [m,n] ---
    #[test]
    fn test_matmul_shapes() {
        let a = f32_mat(3, 4);
        let b = f32_mat(4, 5);
        let expected = f32_mat(3, 5);
        let shapes = infer_binary(Op::MatMul, a, b, expected.clone());
        assert_eq!(shapes[&2], vec![expected]);
    }

    // --- 4. MatMul: inner dimension mismatch ---
    #[test]
    fn test_matmul_dim_mismatch() {
        let mut g = Graph::new("test");
        let i0 = g.add_node(Op::Input { name: "a".into() }, vec![], vec![f32_mat(3, 4)]);
        let i1 = g.add_node(Op::Input { name: "b".into() }, vec![], vec![f32_mat(5, 6)]);
        let mm = g.add_node(Op::MatMul, vec![f32_mat(3, 4), f32_mat(5, 6)], vec![f32_mat(3, 6)]);
        g.add_edge(i0, 0, mm, 0, f32_mat(3, 4));
        g.add_edge(i1, 0, mm, 1, f32_mat(5, 6));
        let result = infer_shapes(&g);
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), ShapeError::DimensionMismatch { .. }));
    }

    // --- 5. Relu: shape preserved ---
    #[test]
    fn test_relu_shapes() {
        let tt = f32_mat(8, 16);
        let shapes = infer_unary(Op::Relu, tt.clone());
        assert_eq!(shapes[&1], vec![tt]);
    }

    // --- 6. Transpose: [m,n] -> [n,m] ---
    #[test]
    fn test_transpose_shapes() {
        let input = f32_mat(3, 7);
        let shapes = infer_unary(Op::Transpose, input);
        let out = &shapes[&1];
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].shape, Shape::matrix(7, 3));
    }

    // --- 7. ReduceSum with axis ---
    #[test]
    fn test_reduce_sum_axis() {
        let input = f32_mat(4, 6);
        let mut g = Graph::new("test");
        let inp = g.add_node(Op::Input { name: "x".into() }, vec![], vec![input.clone()]);
        let red = g.add_node(Op::ReduceSum { axis: Some(1) }, vec![input.clone()], vec![f32_vec(4)]);
        g.add_edge(inp, 0, red, 0, input);
        let shapes = infer_shapes(&g).unwrap();
        assert_eq!(shapes[&1][0].shape, Shape::vector(4));
    }

    // --- 8. ReduceSum without axis -> scalar ---
    #[test]
    fn test_reduce_sum_scalar() {
        let input = f32_mat(4, 6);
        let mut g = Graph::new("test");
        let inp = g.add_node(Op::Input { name: "x".into() }, vec![], vec![input.clone()]);
        let red = g.add_node(Op::ReduceSum { axis: None }, vec![input.clone()], vec![TensorType::f32_scalar()]);
        g.add_edge(inp, 0, red, 0, input);
        let shapes = infer_shapes(&g).unwrap();
        assert_eq!(shapes[&1][0].shape, Shape::scalar());
    }

    // --- 9. ToTernary: dtype changes to Ternary ---
    #[test]
    fn test_to_ternary_dtype() {
        let input = f32_mat(16, 32);
        let mut g = Graph::new("test");
        let inp = g.add_node(Op::Input { name: "x".into() }, vec![], vec![input.clone()]);
        let tt_node = g.add_node(
            Op::ToTernary,
            vec![input.clone()],
            vec![TensorType::ternary_matrix(16, 32)],
        );
        g.add_edge(inp, 0, tt_node, 0, input);
        let shapes = infer_shapes(&g).unwrap();
        let out = &shapes[&1][0];
        assert_eq!(out.dtype, Dtype::Ternary);
        assert_eq!(out.shape, Shape::matrix(16, 32));
    }

    // --- 10. Concat along axis ---
    #[test]
    fn test_concat_shapes() {
        let a = f32_mat(3, 4);
        let b = f32_mat(3, 6);
        let expected = f32_mat(3, 10);
        let shapes = infer_binary(Op::Concat { axis: 1 }, a, b, expected.clone());
        assert_eq!(shapes[&2], vec![expected]);
    }

    // --- 11. Softmax: shape preserved, axis checked ---
    #[test]
    fn test_softmax_shapes() {
        let tt = f32_mat(8, 10);
        let shapes = infer_unary(Op::Softmax { axis: 1 }, tt.clone());
        assert_eq!(shapes[&1], vec![tt]);
    }

    // --- 12. Softmax: bad axis ---
    #[test]
    fn test_softmax_bad_axis() {
        let tt = f32_mat(8, 10);
        let mut g = Graph::new("test");
        let inp = g.add_node(Op::Input { name: "x".into() }, vec![], vec![tt.clone()]);
        let sm = g.add_node(Op::Softmax { axis: 5 }, vec![tt.clone()], vec![tt.clone()]);
        g.add_edge(inp, 0, sm, 0, tt);
        let result = infer_shapes(&g);
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), ShapeError::AxisOutOfBounds { .. }));
    }

    // --- 13. LayerNorm: shape preserved ---
    #[test]
    fn test_layer_norm_shapes() {
        let tt = f32_mat(4, 64);
        let shapes = infer_unary(Op::LayerNorm { eps: 1e-5 }, tt.clone());
        assert_eq!(shapes[&1], vec![tt]);
    }

    // --- 14. validate_shapes returns multiple errors ---
    #[test]
    fn test_validate_shapes_collects_errors() {
        let mut g = Graph::new("test");
        let i0 = g.add_node(Op::Input { name: "a".into() }, vec![], vec![f32_vec(4)]);
        let i1 = g.add_node(Op::Input { name: "b".into() }, vec![], vec![f32_vec(5)]);
        // Bad add
        let add = g.add_node(Op::Add, vec![f32_vec(4), f32_vec(5)], vec![f32_vec(4)]);
        g.add_edge(i0, 0, add, 0, f32_vec(4));
        g.add_edge(i1, 0, add, 1, f32_vec(5));
        let errors = validate_shapes(&g);
        assert!(!errors.is_empty());
    }

    // --- 15. Gelu: shape preserved ---
    #[test]
    fn test_gelu_shapes() {
        let tt = f32_vec(128);
        let shapes = infer_unary(Op::Gelu, tt.clone());
        assert_eq!(shapes[&1], vec![tt]);
    }

    // --- 16. Residual: shapes must match ---
    #[test]
    fn test_residual_shapes() {
        let tt = f32_mat(4, 8);
        let shapes = infer_binary(Op::Residual, tt.clone(), tt.clone(), tt.clone());
        assert_eq!(shapes[&2], vec![tt]);
    }
}
