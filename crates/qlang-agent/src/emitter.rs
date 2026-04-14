use qlang_core::graph::Graph;
use qlang_core::ops::{Manifold, Op};
use qlang_core::tensor::{Dtype, Shape, TensorType};
use qlang_core::verify::{Constraint, ConstraintKind, Proof, ProofStatus, TheoremRef};

/// A graph builder that provides the structured interface for AI agents.
///
/// Instead of emitting text, an AI agent calls methods like:
///   emitter.input("x", f32, [784])
///   emitter.matmul(input_node, weight_node)
///   emitter.to_ternary(matmul_node)
///
/// Each call is a structured decision. No syntax errors possible.
pub struct GraphEmitter {
    graph: Graph,
}

impl GraphEmitter {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            graph: Graph::new(name),
        }
    }

    /// Add an input node.
    pub fn input(&mut self, name: &str, dtype: Dtype, shape: Shape) -> u32 {
        let tt = TensorType::new(dtype, shape);
        self.graph
            .add_node(Op::Input { name: name.into() }, vec![], vec![tt])
    }

    /// Add an output node.
    pub fn output(&mut self, name: &str, source: u32, tensor_type: TensorType) -> u32 {
        let out = self.graph.add_node(
            Op::Output { name: name.into() },
            vec![tensor_type.clone()],
            vec![],
        );
        self.graph.add_edge(source, 0, out, 0, tensor_type);
        out
    }

    /// Add two tensors element-wise.
    pub fn add(&mut self, a: u32, b: u32, tensor_type: TensorType) -> u32 {
        let node = self.graph.add_node(
            Op::Add,
            vec![tensor_type.clone(), tensor_type.clone()],
            vec![tensor_type.clone()],
        );
        self.graph
            .add_edge(a, 0, node, 0, tensor_type.clone());
        self.graph.add_edge(b, 0, node, 1, tensor_type);
        node
    }

    /// Multiply two tensors element-wise.
    pub fn mul(&mut self, a: u32, b: u32, tensor_type: TensorType) -> u32 {
        let node = self.graph.add_node(
            Op::Mul,
            vec![tensor_type.clone(), tensor_type.clone()],
            vec![tensor_type.clone()],
        );
        self.graph
            .add_edge(a, 0, node, 0, tensor_type.clone());
        self.graph.add_edge(b, 0, node, 1, tensor_type);
        node
    }

    /// Matrix multiplication.
    pub fn matmul(
        &mut self,
        a: u32,
        b: u32,
        a_type: TensorType,
        b_type: TensorType,
        out_type: TensorType,
    ) -> u32 {
        let node = self.graph.add_node(
            Op::MatMul,
            vec![a_type.clone(), b_type.clone()],
            vec![out_type.clone()],
        );
        self.graph.add_edge(a, 0, node, 0, a_type);
        self.graph.add_edge(b, 0, node, 1, b_type);
        node
    }

    /// Apply ReLU activation.
    pub fn relu(&mut self, input: u32, tensor_type: TensorType) -> u32 {
        let node = self.graph.add_node(
            Op::Relu,
            vec![tensor_type.clone()],
            vec![tensor_type.clone()],
        );
        self.graph.add_edge(input, 0, node, 0, tensor_type);
        node
    }

    /// IGQK: Compress to ternary weights {-1, 0, +1}.
    /// Automatically attaches Theorem 5.2 proof (assumed).
    pub fn to_ternary(&mut self, input: u32, input_type: TensorType) -> u32 {
        let output_type = TensorType::new(Dtype::Ternary, input_type.shape.clone());

        let node = self.graph.add_node(
            Op::ToTernary,
            vec![input_type.clone()],
            vec![output_type.clone()],
        );

        // Attach IGQK compression bound proof
        self.graph.nodes.last_mut().unwrap().constraints.push(Constraint {
            kind: ConstraintKind::DistortionBound {
                max_distortion: 0.01,
            },
            proof: Some(Proof {
                theorem: TheoremRef::IgqkCompressionBound,
                status: ProofStatus::Assumed,
                parameters: vec![
                    ("n".to_string(), input_type.shape.numel().unwrap_or(0) as f64),
                    ("beta".to_string(), 1.0),
                ],
            }),
        });

        self.graph.add_edge(input, 0, node, 0, input_type);
        node
    }

    /// IGQK: Project onto a submanifold.
    pub fn project(&mut self, input: u32, tensor_type: TensorType, manifold: Manifold) -> u32 {
        let node = self.graph.add_node(
            Op::Project { manifold },
            vec![tensor_type.clone()],
            vec![tensor_type.clone()],
        );
        self.graph.add_edge(input, 0, node, 0, tensor_type);
        node
    }

    /// Finalize and return the graph.
    pub fn build(self) -> Graph {
        self.graph
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn emit_simple_graph() {
        let mut e = GraphEmitter::new("test_emit");

        let x = e.input("x", Dtype::F32, Shape::vector(4));
        let y = e.input("y", Dtype::F32, Shape::vector(4));
        let sum = e.add(x, y, TensorType::f32_vector(4));
        e.output("result", sum, TensorType::f32_vector(4));

        let graph = e.build();
        assert_eq!(graph.nodes.len(), 4);
        assert_eq!(graph.edges.len(), 3);
        assert!(graph.validate().is_ok());
    }

    #[test]
    fn emit_igqk_compression_pipeline() {
        let mut e = GraphEmitter::new("igqk_compress");

        // Neural network weights as input
        let weights = e.input("weights", Dtype::F32, Shape::matrix(768, 768));

        // IGQK ternary compression
        let compressed = e.to_ternary(weights, TensorType::f32_matrix(768, 768));

        // Output
        e.output(
            "compressed_weights",
            compressed,
            TensorType::ternary_matrix(768, 768),
        );

        let graph = e.build();
        assert!(graph.validate().is_ok());

        // Check that the compression node has a proof
        let compress_node = &graph.nodes[1];
        assert!(!compress_node.constraints.is_empty());

        if let Some(constraint) = compress_node.constraints.first() {
            assert!(constraint.proof.is_some());
        }
    }

    #[test]
    fn emit_matmul_relu_pipeline() {
        let mut e = GraphEmitter::new("mlp_layer");

        let input = e.input("x", Dtype::F32, Shape::matrix(1, 784));
        let weights = e.input("W", Dtype::F32, Shape::matrix(784, 128));

        let hidden = e.matmul(
            input,
            weights,
            TensorType::f32_matrix(1, 784),
            TensorType::f32_matrix(784, 128),
            TensorType::f32_matrix(1, 128),
        );

        let activated = e.relu(hidden, TensorType::f32_matrix(1, 128));

        e.output("y", activated, TensorType::f32_matrix(1, 128));

        let graph = e.build();
        assert_eq!(graph.nodes.len(), 5); // input, weights, matmul, relu, output
        assert!(graph.validate().is_ok());
    }
}
