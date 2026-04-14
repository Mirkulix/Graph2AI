//! Type System — Compile-time type checking for QLANG graphs.
//!
//! Verifies tensor type compatibility across all edges in a graph
//! and provides detailed error messages for type mismatches.

use std::collections::HashMap;

use crate::graph::{Graph, NodeId};
use crate::ops::Op;
use crate::tensor::{Dim, Dtype, Shape, TensorType};

/// Type error in a graph.
#[derive(Debug, Clone)]
pub struct TypeError {
    pub node_id: NodeId,
    pub message: String,
    pub expected: Option<String>,
    pub got: Option<String>,
    pub suggestion: Option<String>,
}

impl std::fmt::Display for TypeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "node {}: {}", self.node_id, self.message)?;
        if let Some(exp) = &self.expected {
            write!(f, " (expected: {exp}")?;
            if let Some(got) = &self.got {
                write!(f, ", got: {got}")?;
            }
            write!(f, ")")?;
        }
        if let Some(sug) = &self.suggestion {
            write!(f, " [hint: {sug}]")?;
        }
        Ok(())
    }
}

/// Run comprehensive type checking on a graph.
pub fn type_check(graph: &Graph) -> Vec<TypeError> {
    let mut errors = Vec::new();

    let order = match graph.topological_sort() {
        Ok(o) => o,
        Err(_) => {
            errors.push(TypeError {
                node_id: 0,
                message: "Graph contains a cycle".into(),
                expected: None, got: None, suggestion: None,
            });
            return errors;
        }
    };

    // Infer output type for each node
    let mut node_output_types: HashMap<NodeId, TensorType> = HashMap::new();

    for &node_id in &order {
        let node = match graph.node(node_id) {
            Some(n) => n,
            None => continue,
        };

        // Collect input types from incoming edges
        let incoming = graph.incoming_edges(node_id);
        let input_types: Vec<TensorType> = incoming.iter()
            .filter_map(|e| node_output_types.get(&e.from_node).cloned())
            .collect();

        match &node.op {
            Op::Input { .. } => {
                if let Some(tt) = node.output_types.first() {
                    node_output_types.insert(node_id, tt.clone());
                }
            }

            Op::Output { .. } => {
                // Just check input exists
                if incoming.is_empty() {
                    errors.push(TypeError {
                        node_id,
                        message: "Output node has no input".into(),
                        expected: None, got: None,
                        suggestion: Some("Connect a node to this output".into()),
                    });
                }
            }

            Op::Add | Op::Sub | Op::Mul | Op::Div => {
                if input_types.len() < 2 {
                    errors.push(TypeError {
                        node_id,
                        message: format!("{} requires 2 inputs, got {}", node.op, input_types.len()),
                        expected: Some("2 inputs".into()),
                        got: Some(format!("{} inputs", input_types.len())),
                        suggestion: None,
                    });
                } else if !input_types[0].shape.is_compatible_with(&input_types[1].shape) {
                    errors.push(TypeError {
                        node_id,
                        message: format!("Shape mismatch in {}", node.op),
                        expected: Some(format!("{}", input_types[0])),
                        got: Some(format!("{}", input_types[1])),
                        suggestion: Some("Shapes must be identical or broadcastable".into()),
                    });
                } else {
                    node_output_types.insert(node_id, input_types[0].clone());
                }
            }

            Op::MatMul => {
                if input_types.len() < 2 {
                    errors.push(TypeError {
                        node_id,
                        message: "MatMul requires 2 inputs".into(),
                        expected: Some("2 matrices".into()),
                        got: Some(format!("{} inputs", input_types.len())),
                        suggestion: None,
                    });
                } else {
                    let a = &input_types[0];
                    let b = &input_types[1];

                    if a.shape.rank() != 2 || b.shape.rank() != 2 {
                        errors.push(TypeError {
                            node_id,
                            message: "MatMul inputs must be 2D matrices".into(),
                            expected: Some("rank 2".into()),
                            got: Some(format!("rank {}, {}", a.shape.rank(), b.shape.rank())),
                            suggestion: Some("Use reshape to make inputs 2D".into()),
                        });
                    } else {
                        // Check inner dimensions match
                        let a_cols = &a.shape.0[1];
                        let b_rows = &b.shape.0[0];
                        match (a_cols, b_rows) {
                            (Dim::Fixed(ac), Dim::Fixed(br)) if ac != br => {
                                errors.push(TypeError {
                                    node_id,
                                    message: format!("MatMul inner dimension mismatch: {} vs {}", ac, br),
                                    expected: Some(format!("[?, {}] × [{}, ?]", ac, ac)),
                                    got: Some(format!("{} × {}", a, b)),
                                    suggestion: Some("Inner dimensions must match".into()),
                                });
                            }
                            _ => {
                                let m = a.shape.0[0].clone();
                                let n = b.shape.0[1].clone();
                                node_output_types.insert(node_id,
                                    TensorType::new(Dtype::F32, Shape(vec![m, n])));
                            }
                        }
                    }
                }
            }

            Op::Relu | Op::Sigmoid | Op::Tanh | Op::Gelu | Op::Neg => {
                if let Some(tt) = input_types.first() {
                    node_output_types.insert(node_id, tt.clone());
                }
            }

            Op::Softmax { .. } | Op::LayerNorm { .. } => {
                if let Some(tt) = input_types.first() {
                    node_output_types.insert(node_id, tt.clone());
                }
            }

            Op::Transpose => {
                if let Some(tt) = input_types.first() {
                    if tt.shape.rank() != 2 {
                        errors.push(TypeError {
                            node_id,
                            message: "Transpose requires 2D input".into(),
                            expected: Some("rank 2".into()),
                            got: Some(format!("rank {}", tt.shape.rank())),
                            suggestion: None,
                        });
                    } else {
                        let mut new_dims = tt.shape.0.clone();
                        new_dims.reverse();
                        node_output_types.insert(node_id,
                            TensorType::new(tt.dtype, Shape(new_dims)));
                    }
                }
            }

            Op::ToTernary => {
                if let Some(tt) = input_types.first() {
                    if tt.dtype != Dtype::F32 && tt.dtype != Dtype::F64 {
                        errors.push(TypeError {
                            node_id,
                            message: format!("ToTernary requires float input, got {}", tt.dtype),
                            expected: Some("f32 or f64".into()),
                            got: Some(format!("{}", tt.dtype)),
                            suggestion: Some("Cast input to f32 first".into()),
                        });
                    }
                    node_output_types.insert(node_id,
                        TensorType::new(Dtype::Ternary, tt.shape.clone()));
                }
            }

            Op::Constant => {
                if let Some(tt) = node.output_types.first() {
                    node_output_types.insert(node_id, tt.clone());
                }
            }

            _ => {
                // For other ops, use declared output types or pass through
                if let Some(tt) = node.output_types.first() {
                    node_output_types.insert(node_id, tt.clone());
                } else if let Some(tt) = input_types.first() {
                    node_output_types.insert(node_id, tt.clone());
                }
            }
        }
    }

    errors
}

impl std::error::Error for TypeError {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::Graph;
    use crate::ops::Op;

    #[test]
    fn valid_graph_no_errors() {
        let mut g = Graph::new("valid");
        let a = g.add_node(Op::Input { name: "a".into() }, vec![], vec![TensorType::f32_matrix(2, 3)]);
        let b = g.add_node(Op::Input { name: "b".into() }, vec![], vec![TensorType::f32_matrix(3, 4)]);
        let mm = g.add_node(Op::MatMul, vec![], vec![TensorType::f32_matrix(2, 4)]);
        let out = g.add_node(Op::Output { name: "y".into() }, vec![], vec![]);
        g.add_edge(a, 0, mm, 0, TensorType::f32_matrix(2, 3));
        g.add_edge(b, 0, mm, 1, TensorType::f32_matrix(3, 4));
        g.add_edge(mm, 0, out, 0, TensorType::f32_matrix(2, 4));

        let errors = type_check(&g);
        assert!(errors.is_empty(), "Errors: {:?}", errors);
    }

    #[test]
    fn matmul_dimension_mismatch() {
        let mut g = Graph::new("mismatch");
        let a = g.add_node(Op::Input { name: "a".into() }, vec![], vec![TensorType::f32_matrix(2, 3)]);
        let b = g.add_node(Op::Input { name: "b".into() }, vec![], vec![TensorType::f32_matrix(5, 4)]); // 5 != 3
        let mm = g.add_node(Op::MatMul, vec![], vec![]);
        g.add_edge(a, 0, mm, 0, TensorType::f32_matrix(2, 3));
        g.add_edge(b, 0, mm, 1, TensorType::f32_matrix(5, 4));

        let errors = type_check(&g);
        assert!(!errors.is_empty());
        assert!(errors[0].message.contains("dimension mismatch"));
    }

    #[test]
    fn add_shape_mismatch() {
        let mut g = Graph::new("add_mismatch");
        let a = g.add_node(Op::Input { name: "a".into() }, vec![], vec![TensorType::f32_vector(3)]);
        let b = g.add_node(Op::Input { name: "b".into() }, vec![], vec![TensorType::f32_vector(5)]);
        let add = g.add_node(Op::Add, vec![], vec![]);
        g.add_edge(a, 0, add, 0, TensorType::f32_vector(3));
        g.add_edge(b, 0, add, 1, TensorType::f32_vector(5));

        let errors = type_check(&g);
        assert!(!errors.is_empty());
        assert!(errors[0].message.contains("Shape mismatch"));
    }

    #[test]
    fn ternary_requires_float() {
        let mut g = Graph::new("ternary_type");
        let a = g.add_node(Op::Input { name: "a".into() }, vec![], vec![TensorType::new(Dtype::I32, Shape::vector(4))]);
        let t = g.add_node(Op::ToTernary, vec![], vec![]);
        g.add_edge(a, 0, t, 0, TensorType::new(Dtype::I32, Shape::vector(4)));

        let errors = type_check(&g);
        assert!(!errors.is_empty());
        assert!(errors[0].message.contains("float"));
    }

    #[test]
    fn output_without_input() {
        let mut g = Graph::new("dangling");
        g.add_node(Op::Output { name: "y".into() }, vec![], vec![]);

        let errors = type_check(&g);
        assert!(!errors.is_empty());
        assert!(errors[0].message.contains("no input"));
    }

    #[test]
    fn error_display() {
        let err = TypeError {
            node_id: 5,
            message: "Shape mismatch".into(),
            expected: Some("[2, 3]".into()),
            got: Some("[2, 5]".into()),
            suggestion: Some("Use reshape".into()),
        };
        let s = format!("{err}");
        assert!(s.contains("node 5"));
        assert!(s.contains("Shape mismatch"));
        assert!(s.contains("reshape"));
    }
}
