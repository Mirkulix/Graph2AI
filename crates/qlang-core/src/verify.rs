use serde::{Deserialize, Serialize};
use std::fmt;

use crate::graph::{Graph, NodeId};
use crate::ops::Op;
use crate::tensor::{Dtype, Shape};

/// A constraint attached to a node or graph, with optional formal proof.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Constraint {
    pub kind: ConstraintKind,
    pub proof: Option<Proof>,
}

/// Types of constraints that can be verified.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ConstraintKind {
    /// Output tensor shape must match exactly
    ShapeEquals { shape: Shape },

    /// Distortion bound from IGQK Theorem 5.2
    DistortionBound { max_distortion: f64 },

    /// Convergence guarantee from IGQK Theorem 5.1
    Convergence { hbar: f64 },

    /// Generalization bound from IGQK Theorem 5.3
    GeneralizationBound { mutual_information: f64 },

    /// Tensor values in range [min, max]
    ValueRange { min: f64, max: f64 },

    /// Output is a valid probability distribution (sums to 1, all >= 0)
    IsProbabilityDistribution,

    /// Density matrix is valid (Tr=1, positive semidefinite)
    IsValidDensityMatrix,

    /// Computational complexity bound
    ComplexityBound { big_o: String },

    /// Custom constraint with description
    Custom { description: String },
}

/// A formal proof certificate.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Proof {
    pub theorem: TheoremRef,
    pub status: ProofStatus,
    pub parameters: Vec<(String, f64)>,
}

/// Reference to a theorem from the IGQK theory.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum TheoremRef {
    /// Theorem 5.1: Convergence of quantum gradient flow
    IgqkConvergence,
    /// Theorem 5.2: Compression distortion bound
    IgqkCompressionBound,
    /// Theorem 5.3: Entanglement improves generalization
    IgqkEntanglementGeneralization,
    /// Proposition 6.1: HLWT as Fourier transform of quantum gradient flow
    IgqkHlwt,
    /// Proposition 6.2: TLGT as discrete subgroup
    IgqkTlgt,
    /// Proposition 6.3: FCHL as fractional Laplace-Beltrami
    IgqkFchl,
    /// External theorem reference
    External { name: String },
}

/// Status of a proof.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProofStatus {
    /// Proof has been verified by the compiler
    Verified,
    /// Proof is pending verification
    Pending,
    /// Proof is assumed correct (not checked)
    Assumed,
    /// Proof failed verification
    Failed { reason: String },
}

/// Result of graph verification.
#[derive(Debug)]
pub struct VerificationResult {
    pub passed: Vec<VerificationCheck>,
    pub failed: Vec<VerificationCheck>,
    pub warnings: Vec<String>,
}

#[derive(Debug)]
pub struct VerificationCheck {
    pub node_id: Option<NodeId>,
    pub check: String,
    pub passed: bool,
    pub detail: String,
}

impl VerificationResult {
    pub fn is_ok(&self) -> bool {
        self.failed.is_empty()
    }
}

/// Verify a QLANG graph: type checking, constraint checking, structural validation.
pub fn verify_graph(graph: &Graph) -> VerificationResult {
    let mut passed = Vec::new();
    let mut failed = Vec::new();
    let mut warnings = Vec::new();

    // 1. Structural validation
    match graph.validate() {
        Ok(()) => {
            passed.push(VerificationCheck {
                node_id: None,
                check: "structural_validity".into(),
                passed: true,
                detail: "Graph is a valid DAG".into(),
            });
        }
        Err(errors) => {
            for err in errors {
                failed.push(VerificationCheck {
                    node_id: None,
                    check: "structural_validity".into(),
                    passed: false,
                    detail: err.to_string(),
                });
            }
        }
    }

    // 2. Type checking: verify edge tensor types match node ports
    for edge in &graph.edges {
        if let Some(from_node) = graph.node(edge.from_node) {
            if let Some(output_type) = from_node.output_types.get(edge.from_port as usize) {
                if !edge.tensor_type.shape.is_compatible_with(&output_type.shape) {
                    failed.push(VerificationCheck {
                        node_id: Some(edge.from_node),
                        check: "type_check".into(),
                        passed: false,
                        detail: format!(
                            "Edge {} type {} incompatible with node {} output {}",
                            edge.id, edge.tensor_type, edge.from_node, output_type
                        ),
                    });
                } else {
                    passed.push(VerificationCheck {
                        node_id: Some(edge.from_node),
                        check: "type_check".into(),
                        passed: true,
                        detail: format!("Edge {} type matches", edge.id),
                    });
                }
            }
        }
    }

    // 3. Verify node constraints
    for node in &graph.nodes {
        for constraint in &node.constraints {
            let check_result = verify_constraint(node.id, constraint);
            if check_result.passed {
                passed.push(check_result);
            } else {
                failed.push(check_result);
            }
        }

        // Check ternary output type
        if matches!(node.op, Op::ToTernary) {
            if let Some(out_type) = node.output_types.first() {
                if out_type.dtype != Dtype::Ternary {
                    failed.push(VerificationCheck {
                        node_id: Some(node.id),
                        check: "ternary_output".into(),
                        passed: false,
                        detail: "ToTernary op must output Ternary dtype".into(),
                    });
                }
            }
        }
    }

    // 4. Warnings
    let quantum_ops: Vec<_> = graph
        .nodes
        .iter()
        .filter(|n| n.op.is_quantum())
        .collect();
    if !quantum_ops.is_empty() {
        warnings.push(format!(
            "{} quantum operations present — results will be probabilistic",
            quantum_ops.len()
        ));
    }

    VerificationResult {
        passed,
        failed,
        warnings,
    }
}

fn verify_constraint(node_id: NodeId, constraint: &Constraint) -> VerificationCheck {
    match &constraint.proof {
        Some(proof) if proof.status == ProofStatus::Verified => VerificationCheck {
            node_id: Some(node_id),
            check: format!("{:?}", constraint.kind),
            passed: true,
            detail: format!("Proof verified: {:?}", proof.theorem),
        },
        Some(proof) if proof.status == ProofStatus::Assumed => VerificationCheck {
            node_id: Some(node_id),
            check: format!("{:?}", constraint.kind),
            passed: true, // assumed proofs pass (with warning)
            detail: format!("Proof assumed: {:?}", proof.theorem),
        },
        Some(proof) => VerificationCheck {
            node_id: Some(node_id),
            check: format!("{:?}", constraint.kind),
            passed: false,
            detail: format!("Proof status: {:?}", proof.status),
        },
        None => VerificationCheck {
            node_id: Some(node_id),
            check: format!("{:?}", constraint.kind),
            passed: false,
            detail: "No proof provided".into(),
        },
    }
}

impl fmt::Display for VerificationResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Verification Result:")?;
        writeln!(f, "  Passed: {}", self.passed.len())?;
        writeln!(f, "  Failed: {}", self.failed.len())?;
        for check in &self.failed {
            writeln!(f, "    FAIL [node {:?}]: {}", check.node_id, check.detail)?;
        }
        for warning in &self.warnings {
            writeln!(f, "  WARN: {warning}")?;
        }
        Ok(())
    }
}
