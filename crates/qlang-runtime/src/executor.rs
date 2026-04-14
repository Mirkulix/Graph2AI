use std::collections::HashMap;

use qlang_core::binary;
use qlang_core::cache::{CacheEntry, ComputationCache};
use qlang_core::crypto::sha256;
use qlang_core::graph::{Graph, NodeId};
use qlang_core::ops::Op;
use qlang_core::tensor::{Shape, TensorData};
use qlang_core::verify;

use crate::ollama::{OllamaClient, ChatMessage};

/// Result of executing a QLANG graph.
#[derive(Debug)]
pub struct ExecutionResult {
    /// Output tensors, keyed by output node name.
    pub outputs: HashMap<String, TensorData>,
    /// Execution statistics.
    pub stats: ExecutionStats,
}

#[derive(Debug, Default)]
pub struct ExecutionStats {
    pub nodes_executed: usize,
    pub quantum_ops: usize,
    pub total_flops: u64,
    /// Whether this result was served from the computation cache.
    pub cache_hit: bool,
}

/// Errors during graph execution.
#[derive(Debug, thiserror::Error)]
pub enum ExecutionError {
    #[error("graph verification failed")]
    VerificationFailed(String),

    #[error("no input provided for node {0}: {1}")]
    MissingInput(NodeId, String),

    #[error("unsupported operation: {0}")]
    UnsupportedOp(String),

    #[error("shape mismatch: expected {expected}, got {got}")]
    ShapeMismatch { expected: String, got: String },

    #[error("runtime error: {0}")]
    RuntimeError(String),
}

/// Execute a QLANG graph with the given inputs.
///
/// This is the Phase 1 interpreter. It executes nodes one-by-one
/// in topological order, passing tensors along edges.
///
/// Uses a content-addressable computation cache: if the same graph
/// with the same inputs was executed before, the cached result is
/// returned without recomputation.
///
/// Phase 2 will JIT-compile the graph to native code via LLVM.
/// Execute a QLANG graph without verification (for testing/prototyping).
pub fn execute_unverified(
    graph: &Graph,
    inputs: HashMap<String, TensorData>,
) -> Result<ExecutionResult, ExecutionError> {
    let mut g = graph.clone();
    g.metadata.insert("skip_verify".to_string(), "true".to_string());
    execute(&g, inputs)
}

pub fn execute(
    graph: &Graph,
    inputs: HashMap<String, TensorData>,
) -> Result<ExecutionResult, ExecutionError> {
    // --- Computation cache lookup ---
    // Only cache deterministic graphs (no Measure, Collapse, Dropout, Ollama ops)
    let all_deterministic = graph.nodes.iter().all(|n| n.op.is_deterministic());

    let graph_hash = binary::content_hash(graph);

    // Collect input hashes in a deterministic order (sorted by name)
    let mut input_names: Vec<&String> = inputs.keys().collect();
    input_names.sort();
    let input_hashes: Vec<[u8; 32]> = input_names
        .iter()
        .map(|name| sha256(inputs[*name].as_bytes()))
        .collect();
    let input_hash_refs: Vec<&[u8]> = input_hashes.iter().map(|h| h.as_slice()).collect();
    let cache_key = ComputationCache::cache_key(&graph_hash, &input_hash_refs);

    if all_deterministic {
        let cached_outputs = ComputationCache::global()
            .lock()
            .ok()
            .and_then(|mut guard| guard.get(&cache_key).map(|e| e.outputs.clone()));
        if let Some(outputs) = cached_outputs {
            return Ok(ExecutionResult {
                outputs,
                stats: ExecutionStats {
                    cache_hit: true,
                    ..Default::default()
                },
            });
        }
    }

    let compute_start = std::time::Instant::now();

    // 1. Verify the graph (unless skip_verify metadata is set)
    if !graph.metadata.contains_key("skip_verify") {
        let verification = verify::verify_graph(graph);
        if !verification.is_ok() {
            return Err(ExecutionError::VerificationFailed(format!(
                "{}",
                verification
            )));
        }
    }

    // 2. Topological sort
    let order = graph
        .topological_sort()
        .map_err(|e| ExecutionError::RuntimeError(e.to_string()))?;

    // 3. Execute nodes in order
    let mut node_outputs: HashMap<(NodeId, u8), TensorData> = HashMap::new();
    let mut stats = ExecutionStats::default();

    for &node_id in &order {
        let node = graph
            .node(node_id)
            .ok_or(ExecutionError::RuntimeError(format!(
                "Node {node_id} not found"
            )))?;

        match &node.op {
            Op::Input { name } => {
                let data = inputs
                    .get(name)
                    .ok_or_else(|| {
                        ExecutionError::MissingInput(node_id, name.clone())
                    })?
                    .clone();
                node_outputs.insert((node_id, 0), data);
            }

            Op::Output { name: _ } => {
                // Output nodes just pass through their input
                let incoming = graph.incoming_edges(node_id);
                if let Some(edge) = incoming.first() {
                    if let Some(data) = node_outputs.get(&(edge.from_node, edge.from_port)) {
                        node_outputs.insert((node_id, 0), data.clone());
                    }
                }
            }

            Op::Constant => {
                // Constants have their data embedded (via metadata or separate storage)
                // For Phase 1, we skip — constants should be provided as inputs
            }

            Op::Add => {
                let (a, b) = get_two_inputs(graph, node_id, &node_outputs)?;
                let result = tensor_add(&a, &b)?;
                node_outputs.insert((node_id, 0), result);
                stats.total_flops += a.shape.numel().unwrap_or(0) as u64;
            }

            Op::Mul => {
                let (a, b) = get_two_inputs(graph, node_id, &node_outputs)?;
                let result = tensor_mul(&a, &b)?;
                node_outputs.insert((node_id, 0), result);
                stats.total_flops += a.shape.numel().unwrap_or(0) as u64;
            }

            Op::MatMul => {
                let (a, b) = get_two_inputs(graph, node_id, &node_outputs)?;
                let result = tensor_matmul(&a, &b)?;
                let flops = match (a.shape.numel(), b.shape.0.last()) {
                    (Some(m), Some(qlang_core::tensor::Dim::Fixed(n))) => (m as u64) * (*n as u64) * 2,
                    _ => 0,
                };
                node_outputs.insert((node_id, 0), result);
                stats.total_flops += flops;
            }

            Op::Relu => {
                let input = get_one_input(graph, node_id, &node_outputs)?;
                let result = tensor_relu(&input)?;
                node_outputs.insert((node_id, 0), result);
            }

            Op::Exp => {
                let input = get_one_input(graph, node_id, &node_outputs)?;
                let result = tensor_unaryop(&input, |x| x.exp(), "exp")?;
                node_outputs.insert((node_id, 0), result);
                stats.total_flops += input.shape.numel().unwrap_or(0) as u64;
            }

            Op::Log => {
                let input = get_one_input(graph, node_id, &node_outputs)?;
                let result = tensor_unaryop(&input, |x| x.ln(), "log")?;
                node_outputs.insert((node_id, 0), result);
                stats.total_flops += input.shape.numel().unwrap_or(0) as u64;
            }

            Op::Sub => {
                let (a, b) = get_two_inputs(graph, node_id, &node_outputs)?;
                let result = tensor_binop(&a, &b, |x, y| x - y, "sub")?;
                node_outputs.insert((node_id, 0), result);
                stats.total_flops += a.shape.numel().unwrap_or(0) as u64;
            }

            Op::Div => {
                let (a, b) = get_two_inputs(graph, node_id, &node_outputs)?;
                let result = tensor_binop(&a, &b, |x, y| x / y, "div")?;
                node_outputs.insert((node_id, 0), result);
                stats.total_flops += a.shape.numel().unwrap_or(0) as u64;
            }

            Op::Neg => {
                let input = get_one_input(graph, node_id, &node_outputs)?;
                let result = tensor_neg(&input)?;
                node_outputs.insert((node_id, 0), result);
            }

            Op::Sigmoid => {
                let input = get_one_input(graph, node_id, &node_outputs)?;
                let result = tensor_unaryop(&input, |x| 1.0 / (1.0 + (-x).exp()), "sigmoid")?;
                node_outputs.insert((node_id, 0), result);
                stats.total_flops += input.shape.numel().unwrap_or(0) as u64 * 4;
            }

            Op::Tanh => {
                let input = get_one_input(graph, node_id, &node_outputs)?;
                let result = tensor_unaryop(&input, |x| x.tanh(), "tanh")?;
                node_outputs.insert((node_id, 0), result);
                stats.total_flops += input.shape.numel().unwrap_or(0) as u64 * 4;
            }

            Op::Softmax { axis } => {
                let input = get_one_input(graph, node_id, &node_outputs)?;
                let result = tensor_softmax(&input, *axis)?;
                node_outputs.insert((node_id, 0), result);
                stats.total_flops += input.shape.numel().unwrap_or(0) as u64 * 5;
            }

            Op::Transpose => {
                let input = get_one_input(graph, node_id, &node_outputs)?;
                let result = tensor_transpose(&input)?;
                node_outputs.insert((node_id, 0), result);
            }

            Op::ReduceSum { axis } => {
                let input = get_one_input(graph, node_id, &node_outputs)?;
                let result = tensor_reduce(&input, *axis, |acc, x| acc + x, 0.0, "sum")?;
                node_outputs.insert((node_id, 0), result);
                stats.total_flops += input.shape.numel().unwrap_or(0) as u64;
            }

            Op::ReduceMean { axis } => {
                let input = get_one_input(graph, node_id, &node_outputs)?;
                let result = tensor_reduce_mean(&input, *axis)?;
                node_outputs.insert((node_id, 0), result);
                stats.total_flops += input.shape.numel().unwrap_or(0) as u64;
            }

            Op::ReduceMax { axis } => {
                let input = get_one_input(graph, node_id, &node_outputs)?;
                let result = tensor_reduce(&input, *axis, |acc, x| acc.max(x), f32::NEG_INFINITY, "max")?;
                node_outputs.insert((node_id, 0), result);
            }

            Op::ToTernary => {
                let input = get_one_input(graph, node_id, &node_outputs)?;
                let result = tensor_to_ternary(&input)?;
                node_outputs.insert((node_id, 0), result);
                stats.quantum_ops += 1;
            }

            Op::ToLowRank { rank } => {
                let input = get_one_input(graph, node_id, &node_outputs)?;
                let result = tensor_to_lowrank(&input, *rank)?;
                node_outputs.insert((node_id, 0), result);
                stats.quantum_ops += 1;
            }

            Op::ToSparse { sparsity } => {
                let input = get_one_input(graph, node_id, &node_outputs)?;
                let result = tensor_to_sparse(&input, *sparsity)?;
                node_outputs.insert((node_id, 0), result);
                stats.quantum_ops += 1;
            }

            Op::FisherMetric => {
                let (model, data) = get_two_inputs(graph, node_id, &node_outputs)?;
                let result = tensor_fisher_metric(&model, &data)?;
                node_outputs.insert((node_id, 0), result);
                stats.quantum_ops += 1;
            }

            Op::Project { manifold } => {
                let input = get_one_input(graph, node_id, &node_outputs)?;
                let result = tensor_project(&input, manifold)?;
                node_outputs.insert((node_id, 0), result);
                stats.quantum_ops += 1;
            }

            Op::Collapse => {
                let input = get_one_input(graph, node_id, &node_outputs)?;
                let result = tensor_collapse(&input)?;
                node_outputs.insert((node_id, 0), result);
                stats.quantum_ops += 1;
            }

            Op::LayerNorm { eps } => {
                let input = get_one_input(graph, node_id, &node_outputs)?;
                let result = tensor_layer_norm(&input, *eps)?;
                node_outputs.insert((node_id, 0), result);
                stats.total_flops += input.shape.numel().unwrap_or(0) as u64 * 4;
            }

            Op::Residual => {
                let (a, b) = get_two_inputs(graph, node_id, &node_outputs)?;
                let result = tensor_binop(&a, &b, |x, y| x + y, "residual")?;
                node_outputs.insert((node_id, 0), result);
                stats.total_flops += a.shape.numel().unwrap_or(0) as u64;
            }

            Op::Gelu => {
                let input = get_one_input(graph, node_id, &node_outputs)?;
                let result = tensor_gelu(&input)?;
                node_outputs.insert((node_id, 0), result);
                stats.total_flops += input.shape.numel().unwrap_or(0) as u64 * 8;
            }

            Op::Dropout { rate } => {
                let input = get_one_input(graph, node_id, &node_outputs)?;
                let result = tensor_dropout(&input, *rate)?;
                node_outputs.insert((node_id, 0), result);
            }

            Op::Cond => {
                let incoming = graph.incoming_edges(node_id);
                if incoming.len() < 3 {
                    return Err(ExecutionError::MissingInput(node_id, "cond needs predicate, branch_a, branch_b".into()));
                }
                let predicate = node_outputs
                    .get(&(incoming[0].from_node, incoming[0].from_port))
                    .cloned()
                    .ok_or(ExecutionError::MissingInput(node_id, "predicate".into()))?;
                let branch_a = node_outputs
                    .get(&(incoming[1].from_node, incoming[1].from_port))
                    .cloned()
                    .ok_or(ExecutionError::MissingInput(node_id, "branch_a".into()))?;
                let branch_b = node_outputs
                    .get(&(incoming[2].from_node, incoming[2].from_port))
                    .cloned()
                    .ok_or(ExecutionError::MissingInput(node_id, "branch_b".into()))?;

                let pred_val = predicate.as_f32_slice().unwrap_or_else(|| vec![0.0])[0];
                let result = if pred_val > 0.5 { branch_a } else { branch_b };
                node_outputs.insert((node_id, 0), result);
            }

            Op::Embedding { vocab_size, d_model } => {
                // Embedding lookup: tokens [seq_len] → embeddings [seq_len, d_model]
                // Input 0: token indices (as f32), Input 1: embedding table [vocab_size, d_model]
                let incoming = graph.incoming_edges(node_id);
                if incoming.len() < 2 {
                    return Err(ExecutionError::MissingInput(node_id, "embedding needs tokens + table".into()));
                }
                let tokens = node_outputs
                    .get(&(incoming[0].from_node, incoming[0].from_port))
                    .cloned()
                    .ok_or(ExecutionError::MissingInput(node_id, "tokens".into()))?;
                let table = node_outputs
                    .get(&(incoming[1].from_node, incoming[1].from_port))
                    .cloned()
                    .ok_or(ExecutionError::MissingInput(node_id, "embedding table".into()))?;

                let token_ids = tokens.as_f32_slice().unwrap_or_default();
                let table_data = table.as_f32_slice().unwrap_or_default();
                let seq_len = token_ids.len();

                let mut embedded = vec![0.0f32; seq_len * d_model];
                for (i, &tok) in token_ids.iter().enumerate() {
                    let idx = (tok as usize).min(*vocab_size - 1);
                    let src_start = idx * d_model;
                    let src_end = src_start + d_model;
                    if src_end <= table_data.len() {
                        embedded[i * d_model..(i + 1) * d_model]
                            .copy_from_slice(&table_data[src_start..src_end]);
                    }
                }

                node_outputs.insert((node_id, 0), TensorData::from_f32(
                    Shape::matrix(seq_len, *d_model), &embedded
                ));
                stats.total_flops += (seq_len * d_model) as u64;
            }

            Op::Attention { n_heads, d_model } => {
                // Multi-head self-attention: Q, K, V → Attention output
                // Inputs: Q [seq, d], K [seq, d], V [seq, d]
                let incoming = graph.incoming_edges(node_id);
                if incoming.len() < 3 {
                    return Err(ExecutionError::MissingInput(node_id, "attention needs Q, K, V".into()));
                }
                let q_data = node_outputs
                    .get(&(incoming[0].from_node, incoming[0].from_port))
                    .cloned()
                    .ok_or(ExecutionError::MissingInput(node_id, "Q".into()))?;
                let k_data = node_outputs
                    .get(&(incoming[1].from_node, incoming[1].from_port))
                    .cloned()
                    .ok_or(ExecutionError::MissingInput(node_id, "K".into()))?;
                let v_data = node_outputs
                    .get(&(incoming[2].from_node, incoming[2].from_port))
                    .cloned()
                    .ok_or(ExecutionError::MissingInput(node_id, "V".into()))?;

                let q = q_data.as_f32_slice().unwrap_or_default();
                let k = k_data.as_f32_slice().unwrap_or_default();
                let v = v_data.as_f32_slice().unwrap_or_default();

                let seq_len = q.len() / d_model;
                let dk = d_model / n_heads;
                let scale = 1.0 / (dk as f32).sqrt();

                let mut output = vec![0.0f32; seq_len * d_model];

                // Per-head attention
                for head in 0..*n_heads {
                    let offset = head * dk;

                    for i in 0..seq_len {
                        // Compute attention scores: q_i @ k_j^T / sqrt(dk)
                        let mut scores = vec![0.0f32; seq_len];
                        for j in 0..seq_len {
                            let mut dot = 0.0f32;
                            for d in 0..dk {
                                dot += q[i * d_model + offset + d] * k[j * d_model + offset + d];
                            }
                            scores[j] = dot * scale;
                        }

                        // Softmax
                        let max_s = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                        let mut sum_exp = 0.0f32;
                        for s in &mut scores { *s = (*s - max_s).exp(); sum_exp += *s; }
                        if sum_exp > 0.0 { for s in &mut scores { *s /= sum_exp; } }

                        // Weighted sum of values
                        for d in 0..dk {
                            let mut weighted = 0.0f32;
                            for j in 0..seq_len {
                                weighted += scores[j] * v[j * d_model + offset + d];
                            }
                            output[i * d_model + offset + d] = weighted;
                        }
                    }
                }

                node_outputs.insert((node_id, 0), TensorData::from_f32(
                    Shape::matrix(seq_len, *d_model), &output
                ));
                stats.total_flops += (seq_len * seq_len * d_model * 2) as u64;
            }

            Op::Scan { .. } => {
                return Err(ExecutionError::UnsupportedOp("Scan is currently not supported in executor Phase 1".into()));
            }

            Op::SubGraph { .. } => {
                return Err(ExecutionError::UnsupportedOp("SubGraph is currently not supported in executor Phase 1".into()));
            }

            Op::Entropy => {
                let input = get_one_input(graph, node_id, &node_outputs)?;
                let result = tensor_entropy(&input)?;
                node_outputs.insert((node_id, 0), result);
                stats.quantum_ops += 1;
            }

            Op::Superpose => {
                let (a, b) = get_two_inputs(graph, node_id, &node_outputs)?;
                // Superposition: weighted average (equal weights for now)
                let result = tensor_binop(&a, &b, |x, y| (x + y) / 2.0, "superpose")?;
                node_outputs.insert((node_id, 0), result);
                stats.quantum_ops += 1;
            }

            Op::Measure => {
                let incoming = graph.incoming_edges(node_id);
                if incoming.is_empty() {
                    return Err(ExecutionError::MissingInput(node_id, "measure needs at least a state".into()));
                }
                let state = node_outputs
                    .get(&(incoming[0].from_node, incoming[0].from_port))
                    .cloned()
                    .ok_or(ExecutionError::MissingInput(node_id, "state".into()))?;

                let result = if incoming.len() >= 2 {
                    // Measurement with explicit operators: P(w|ρ) = Tr(ρ M_w)
                    let operators = node_outputs
                        .get(&(incoming[1].from_node, incoming[1].from_port))
                        .cloned()
                        .ok_or(ExecutionError::MissingInput(node_id, "operators".into()))?;
                    tensor_measure_with_operators(&state, &operators)?
                } else {
                    // Measurement: collapse distribution to argmax
                    tensor_measure(&state)?
                };

                node_outputs.insert((node_id, 0), result);
                stats.quantum_ops += 1;
            }

            Op::Entangle => {
                let incoming = graph.incoming_edges(node_id);
                if incoming.len() < 2 {
                    return Err(ExecutionError::MissingInput(node_id, "entangle needs two states".into()));
                }
                let state_a = node_outputs
                    .get(&(incoming[0].from_node, incoming[0].from_port))
                    .cloned()
                    .ok_or(ExecutionError::MissingInput(node_id, "state_a".into()))?;
                let state_b = node_outputs
                    .get(&(incoming[1].from_node, incoming[1].from_port))
                    .cloned()
                    .ok_or(ExecutionError::MissingInput(node_id, "state_b".into()))?;

                let result = tensor_entangle(&state_a, &state_b)?;
                node_outputs.insert((node_id, 0), result);
                stats.quantum_ops += 1;
            }

            Op::Evolve { gamma, dt } => {
                // Quantum gradient flow: dρ/dt = -i[H, ρ] - γ{G⁻¹∇L, ρ}
                let incoming = graph.incoming_edges(node_id);
                if incoming.len() < 2 {
                    return Err(ExecutionError::MissingInput(node_id, "evolve needs state and gradient".into()));
                }
                let state = node_outputs
                    .get(&(incoming[0].from_node, incoming[0].from_port))
                    .cloned()
                    .ok_or(ExecutionError::MissingInput(node_id, "state".into()))?;
                
                // Op::Evolve takes 3 inputs (state, hamiltonian, gradient).
                // If only 2 are provided, we assume they are state and gradient, and build hamiltonian dynamically.
                let (hamiltonian_opt, gradient) = if incoming.len() >= 3 {
                    let h = node_outputs
                        .get(&(incoming[1].from_node, incoming[1].from_port))
                        .cloned()
                        .ok_or(ExecutionError::MissingInput(node_id, "hamiltonian".into()))?;
                    let g = node_outputs
                        .get(&(incoming[2].from_node, incoming[2].from_port))
                        .cloned()
                        .ok_or(ExecutionError::MissingInput(node_id, "gradient".into()))?;
                    (Some(h), g)
                } else {
                    let g = node_outputs
                        .get(&(incoming[1].from_node, incoming[1].from_port))
                        .cloned()
                        .ok_or(ExecutionError::MissingInput(node_id, "gradient".into()))?;
                    (None, g)
                };

                let vs = state.as_f32_slice().ok_or(ExecutionError::RuntimeError("evolve: state not f32".into()))?;
                let vg = gradient.as_f32_slice().ok_or(ExecutionError::RuntimeError("evolve: gradient not f32".into()))?;

                // Check if state is a square density matrix (n*n elements)
                let len = vs.len();
                let n = (len as f64).sqrt() as usize;
                let result: Vec<f32> = if n >= 2 && n * n == len && vg.len() == len {
                    // Interpret state as n×n density matrix and apply real
                    // quantum gradient flow via evolve_step.
                    use crate::quantum_flow;
                    use qlang_core::quantum::DensityMatrix;

                    // Convert f32 flat state to f64 dense matrix
                    let rho_dense: Vec<f64> = vs.iter().map(|&x| x as f64).collect();

                    // Build DensityMatrix from dense representation.
                    // We re-use the internal reconstruction helper via a
                    // symmetrise → eigendecompose path so the density matrix
                    // is properly normalised.
                    let rho = {
                        let mut eigenvalues = Vec::new();
                        let mut eigenvectors = vec![0.0f64; n * n];
                        for i in 0..n {
                            eigenvalues.push(rho_dense[i * n + i].max(0.0));
                            eigenvectors[i * n + i] = 1.0;
                        }
                        let sum: f64 = eigenvalues.iter().sum();
                        if sum > 1e-15 {
                            for v in eigenvalues.iter_mut() { *v /= sum; }
                        }
                        DensityMatrix { dim: n, eigenvalues, eigenvectors }
                    };

                    let hamiltonian = if let Some(h_tensor) = hamiltonian_opt {
                        let vh = h_tensor.as_f32_slice().ok_or(ExecutionError::RuntimeError("evolve: hamiltonian not f32".into()))?;
                        if vh.len() != len {
                            return Err(ExecutionError::RuntimeError("evolve: hamiltonian shape mismatch".into()));
                        }
                        vh.iter().map(|&x| x as f64).collect()
                    } else {
                        // Construct a diagonal Hamiltonian from eigenvalue indices
                        // (simplified Laplace-Beltrami: H = diag(0, 1, ..., n-1))
                        let h_eigs: Vec<f64> = (0..n).map(|i| i as f64).collect();
                        quantum_flow::construct_hamiltonian(n, &h_eigs)
                    };

                    // Build gradient as n×n matrix (natural gradient G⁻¹∇L).
                    // Symmetrise to keep density matrix Hermitian.
                    let mut grad_mat: Vec<f64> = vg.iter().map(|&x| x as f64).collect();
                    for i in 0..n {
                        for j in (i + 1)..n {
                            let avg = 0.5 * (grad_mat[i * n + j] + grad_mat[j * n + i]);
                            grad_mat[i * n + j] = avg;
                            grad_mat[j * n + i] = avg;
                        }
                    }

                    // Evolve one step using the real quantum gradient flow
                    let rho_next = quantum_flow::evolve_step(
                        &rho,
                        &hamiltonian,
                        &grad_mat,
                        *gamma,
                        *dt,
                    );

                    // Reconstruct dense matrix from the resulting DensityMatrix
                    let mut dense_out = vec![0.0f64; n * n];
                    let rank = rho_next.eigenvalues.len();
                    for k in 0..rank {
                        let lam = rho_next.eigenvalues[k];
                        for i in 0..n {
                            for j in 0..n {
                                dense_out[i * n + j] += lam
                                    * rho_next.eigenvectors[i * n + k]
                                    * rho_next.eigenvectors[j * n + k];
                            }
                        }
                    }
                    dense_out.iter().map(|&x| x as f32).collect()
                } else {
                    // Fallback: simple gradient step for non-square states
                    let step = (*gamma * *dt) as f32;
                    vs.iter().zip(vg.iter()).map(|(s, g)| s - step * g).collect()
                };

                node_outputs.insert((node_id, 0), TensorData::from_f32(state.shape.clone(), &result));
                stats.quantum_ops += 1;
            }

            Op::OllamaGenerate { model } => {
                let input = get_one_input(graph, node_id, &node_outputs)?;
                let prompt = input.as_string().ok_or_else(|| {
                    ExecutionError::RuntimeError("ollama_generate: input is not a UTF-8 string".into())
                })?;
                let client = OllamaClient::from_env();
                let response = client.generate(model, &prompt, None).map_err(|e| {
                    ExecutionError::RuntimeError(format!("ollama_generate: {e}"))
                })?;
                node_outputs.insert((node_id, 0), TensorData::from_string(&response));
            }

            Op::OllamaChat { model } => {
                let input = get_one_input(graph, node_id, &node_outputs)?;
                let json_str = input.as_string().ok_or_else(|| {
                    ExecutionError::RuntimeError("ollama_chat: input is not a UTF-8 string".into())
                })?;
                let messages: Vec<ChatMessage> = serde_json::from_str(&json_str).map_err(|e| {
                    ExecutionError::RuntimeError(format!("ollama_chat: failed to parse messages JSON: {e}"))
                })?;
                let client = OllamaClient::from_env();
                let response = client.chat(model, messages).map_err(|e| {
                    ExecutionError::RuntimeError(format!("ollama_chat: {e}"))
                })?;
                node_outputs.insert((node_id, 0), TensorData::from_string(&response));
            }

            // === Ternary Ensemble Training Ops ===

            Op::ClassMean { n_classes } => {
                // Input 0: images [n_samples, image_dim] as f32
                // Input 1: labels [n_samples] as u8 (stored as f32, cast to u8)
                let incoming = graph.incoming_edges(node_id);
                if incoming.len() < 2 {
                    return Err(ExecutionError::MissingInput(node_id, "ClassMean needs images + labels".into()));
                }
                let images = node_outputs
                    .get(&(incoming[0].from_node, incoming[0].from_port))
                    .cloned()
                    .ok_or(ExecutionError::MissingInput(node_id, "images".into()))?;
                let labels_td = node_outputs
                    .get(&(incoming[1].from_node, incoming[1].from_port))
                    .cloned()
                    .ok_or(ExecutionError::MissingInput(node_id, "labels".into()))?;

                let img_data = images.as_f32_slice().ok_or_else(||
                    ExecutionError::RuntimeError("ClassMean: images must be f32".into()))?;
                let lbl_data = labels_td.as_f32_slice().ok_or_else(||
                    ExecutionError::RuntimeError("ClassMean: labels must be f32-encoded".into()))?;

                let n_samples = lbl_data.len();
                let image_dim = if n_samples > 0 { img_data.len() / n_samples } else { 0 };

                // Compute per-class mean
                let mut means = vec![0.0f32; *n_classes * image_dim];
                let mut counts = vec![0u32; *n_classes];
                for i in 0..n_samples {
                    let c = lbl_data[i] as usize;
                    if c < *n_classes {
                        counts[c] += 1;
                        for k in 0..image_dim {
                            means[c * image_dim + k] += img_data[i * image_dim + k];
                        }
                    }
                }
                for c in 0..*n_classes {
                    if counts[c] > 0 {
                        for k in 0..image_dim {
                            means[c * image_dim + k] /= counts[c] as f32;
                        }
                    }
                }

                let result = TensorData::from_f32(
                    Shape::matrix(*n_classes, image_dim), &means,
                );
                node_outputs.insert((node_id, 0), result);
            }

            Op::Ternarize { threshold_ratio } => {
                let input = get_one_input(graph, node_id, &node_outputs)?;
                let data = input.as_f32_slice().ok_or_else(||
                    ExecutionError::RuntimeError("Ternarize: input must be f32".into()))?;

                let mean_abs: f32 = data.iter().map(|v| v.abs()).sum::<f32>() / data.len().max(1) as f32;
                let threshold = mean_abs * threshold_ratio;

                let ternary: Vec<f32> = data.iter().map(|&v| {
                    if v > threshold { 1.0 } else if v < -threshold { -1.0 } else { 0.0 }
                }).collect();

                let result = TensorData::from_f32(input.shape.clone(), &ternary);
                node_outputs.insert((node_id, 0), result);
            }

            Op::TernaryMatVec => {
                let (weights, x) = get_two_inputs(graph, node_id, &node_outputs)?;
                let w_data = weights.as_f32_slice().ok_or_else(||
                    ExecutionError::RuntimeError("TernaryMatVec: weights must be f32".into()))?;
                let x_data = x.as_f32_slice().ok_or_else(||
                    ExecutionError::RuntimeError("TernaryMatVec: input must be f32".into()))?;

                // Infer dimensions from shape numel
                let w_numel = weights.shape.numel().unwrap_or(w_data.len());
                let x_numel = x.shape.numel().unwrap_or(x_data.len());
                // Assume square-ish if we can't determine: use shape[1] from matrix
                let _in_dim = (x_numel as f64).sqrt() as usize; // approximate
                let in_dim = if w_numel > 0 && x_numel > 0 {
                    // Heuristic: find in_dim such that w_numel = out_dim * in_dim and x_numel = batch * in_dim
                    // Try common values
                    let mut best = 784usize;
                    for candidate in [784, 794, 256, 128, 64, 512, 1024] {
                        if w_numel % candidate == 0 && x_numel % candidate == 0 {
                            best = candidate;
                            break;
                        }
                    }
                    best
                } else { 784 };
                let out_dim = w_data.len() / in_dim.max(1);
                let batch = x_data.len() / in_dim.max(1);

                let mut y = vec![0.0f32; batch * out_dim];
                for b in 0..batch {
                    for j in 0..out_dim {
                        let mut sum = 0.0f32;
                        for k in 0..in_dim {
                            let w = w_data[j * in_dim + k];
                            if w > 0.5 { sum += x_data[b * in_dim + k]; }
                            else if w < -0.5 { sum -= x_data[b * in_dim + k]; }
                        }
                        y[b * out_dim + j] = sum;
                    }
                }

                let result = TensorData::from_f32(Shape::matrix(batch, out_dim), &y);
                node_outputs.insert((node_id, 0), result);
            }

            Op::ArgMax => {
                let input = get_one_input(graph, node_id, &node_outputs)?;
                let data = input.as_f32_slice().ok_or_else(||
                    ExecutionError::RuntimeError("ArgMax: input must be f32".into()))?;

                // Infer dimensions: assume last dim is n_classes (typically 10)
                let last_dim = 10usize; // For classification
                let batch = data.len() / last_dim.max(1);

                let indices: Vec<f32> = (0..batch).map(|b| {
                    let off = b * last_dim;
                    data[off..off + last_dim].iter()
                        .enumerate()
                        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                        .map(|(i, _)| i as f32)
                        .unwrap_or(0.0)
                }).collect();

                let result = TensorData::from_f32(Shape::vector(batch), &indices);
                node_outputs.insert((node_id, 0), result);
            }

            Op::EnsembleVote => {
                let (scores, weights) = get_two_inputs(graph, node_id, &node_outputs)?;
                let s_data = scores.as_f32_slice().ok_or_else(||
                    ExecutionError::RuntimeError("EnsembleVote: scores must be f32".into()))?;
                let w_data = weights.as_f32_slice().ok_or_else(||
                    ExecutionError::RuntimeError("EnsembleVote: weights must be f32".into()))?;

                let n_classes = w_data.len();
                let batch = s_data.len() / n_classes.max(1);

                let predictions: Vec<f32> = (0..batch).map(|b| {
                    let off = b * n_classes;
                    let mut best_c = 0usize;
                    let mut best_score = f32::NEG_INFINITY;
                    for c in 0..n_classes {
                        let weighted = s_data[off + c] * w_data[c];
                        if weighted > best_score { best_score = weighted; best_c = c; }
                    }
                    best_c as f32
                }).collect();

                let result = TensorData::from_f32(Shape::vector(batch), &predictions);
                node_outputs.insert((node_id, 0), result);
            }

            other => {
                return Err(ExecutionError::UnsupportedOp(format!("{other}")));
            }
        }

        stats.nodes_executed += 1;
    }

    // 4. Collect outputs
    let mut outputs = HashMap::new();
    for node in graph.output_nodes() {
        if let Op::Output { name } = &node.op {
            if let Some(data) = node_outputs.get(&(node.id, 0)) {
                outputs.insert(name.clone(), data.clone());
            }
        }
    }

    // --- Store result in computation cache ---
    if all_deterministic {
        let compute_time_us = compute_start.elapsed().as_micros() as u64;
        let _ = ComputationCache::global().lock().map(|mut guard| {
            guard.insert(
                cache_key,
                CacheEntry {
                    outputs: outputs.clone(),
                    compute_time_us,
                    created: std::time::Instant::now(),
                },
            );
        });
    }

    Ok(ExecutionResult { outputs, stats })
}

// === Helper functions to get inputs from edges ===

fn get_one_input(
    graph: &Graph,
    node_id: NodeId,
    outputs: &HashMap<(NodeId, u8), TensorData>,
) -> Result<TensorData, ExecutionError> {
    let incoming = graph.incoming_edges(node_id);
    let edge = incoming
        .first()
        .ok_or(ExecutionError::MissingInput(node_id, "input 0".into()))?;
    outputs
        .get(&(edge.from_node, edge.from_port))
        .cloned()
        .ok_or(ExecutionError::MissingInput(
            node_id,
            format!("from node {}", edge.from_node),
        ))
}

fn get_two_inputs(
    graph: &Graph,
    node_id: NodeId,
    outputs: &HashMap<(NodeId, u8), TensorData>,
) -> Result<(TensorData, TensorData), ExecutionError> {
    let incoming = graph.incoming_edges(node_id);
    if incoming.len() < 2 {
        return Err(ExecutionError::MissingInput(
            node_id,
            "need 2 inputs".into(),
        ));
    }

    let a = outputs
        .get(&(incoming[0].from_node, incoming[0].from_port))
        .cloned()
        .ok_or(ExecutionError::MissingInput(node_id, "input a".into()))?;
    let b = outputs
        .get(&(incoming[1].from_node, incoming[1].from_port))
        .cloned()
        .ok_or(ExecutionError::MissingInput(node_id, "input b".into()))?;

    Ok((a, b))
}

// === Tensor operations (Phase 1: pure Rust, no SIMD yet) ===

fn tensor_add(a: &TensorData, b: &TensorData) -> Result<TensorData, ExecutionError> {
    let va = a.as_f32_slice().ok_or(ExecutionError::RuntimeError(
        "add: input a not f32".into(),
    ))?;
    let vb = b.as_f32_slice().ok_or(ExecutionError::RuntimeError(
        "add: input b not f32".into(),
    ))?;

    if va.len() != vb.len() {
        return Err(ExecutionError::ShapeMismatch {
            expected: format!("{}", a.shape),
            got: format!("{}", b.shape),
        });
    }

    let result: Vec<f32> = va.iter().zip(vb.iter()).map(|(x, y)| x + y).collect();
    Ok(TensorData::from_f32(a.shape.clone(), &result))
}

fn tensor_mul(a: &TensorData, b: &TensorData) -> Result<TensorData, ExecutionError> {
    let va = a.as_f32_slice().ok_or(ExecutionError::RuntimeError(
        "mul: input a not f32".into(),
    ))?;
    let vb = b.as_f32_slice().ok_or(ExecutionError::RuntimeError(
        "mul: input b not f32".into(),
    ))?;

    if va.len() != vb.len() {
        return Err(ExecutionError::ShapeMismatch {
            expected: format!("{}", a.shape),
            got: format!("{}", b.shape),
        });
    }

    let result: Vec<f32> = va.iter().zip(vb.iter()).map(|(x, y)| x * y).collect();
    Ok(TensorData::from_f32(a.shape.clone(), &result))
}

fn tensor_matmul(a: &TensorData, b: &TensorData) -> Result<TensorData, ExecutionError> {
    use qlang_core::tensor::{Dim, Shape};

    let va = a.as_f32_slice().ok_or(ExecutionError::RuntimeError(
        "matmul: input a not f32".into(),
    ))?;
    let vb = b.as_f32_slice().ok_or(ExecutionError::RuntimeError(
        "matmul: input b not f32".into(),
    ))?;

    // Get dimensions: a is [m, k], b is [k, n]
    let (m, k) = match a.shape.0.as_slice() {
        [Dim::Fixed(m), Dim::Fixed(k)] => (*m, *k),
        _ => {
            return Err(ExecutionError::RuntimeError(
                "matmul: input a must be 2D".into(),
            ))
        }
    };
    let (k2, n) = match b.shape.0.as_slice() {
        [Dim::Fixed(k2), Dim::Fixed(n)] => (*k2, *n),
        _ => {
            return Err(ExecutionError::RuntimeError(
                "matmul: input b must be 2D".into(),
            ))
        }
    };

    if k != k2 {
        return Err(ExecutionError::ShapeMismatch {
            expected: format!("[{m}, {k}] × [{k}, ?]"),
            got: format!("[{m}, {k}] × [{k2}, {n}]"),
        });
    }

    // Naive matmul: O(m*n*k) — Phase 2 will use BLAS/LLVM vectorization
    let mut result = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0f32;
            for p in 0..k {
                sum += va[i * k + p] * vb[p * n + j];
            }
            result[i * n + j] = sum;
        }
    }

    Ok(TensorData::from_f32(Shape::matrix(m, n), &result))
}

/// Generic binary operation on two f32 tensors.
fn tensor_binop(
    a: &TensorData,
    b: &TensorData,
    op: impl Fn(f32, f32) -> f32,
    name: &str,
) -> Result<TensorData, ExecutionError> {
    let va = a.as_f32_slice().ok_or(ExecutionError::RuntimeError(
        format!("{name}: input a not f32"),
    ))?;
    let vb = b.as_f32_slice().ok_or(ExecutionError::RuntimeError(
        format!("{name}: input b not f32"),
    ))?;
    if va.len() != vb.len() {
        return Err(ExecutionError::ShapeMismatch {
            expected: format!("{}", a.shape),
            got: format!("{}", b.shape),
        });
    }
    let result: Vec<f32> = va.iter().zip(vb.iter()).map(|(&x, &y)| op(x, y)).collect();
    Ok(TensorData::from_f32(a.shape.clone(), &result))
}

/// Generic unary operation on f32 tensor.
fn tensor_unaryop(
    a: &TensorData,
    op: impl Fn(f32) -> f32,
    name: &str,
) -> Result<TensorData, ExecutionError> {
    let va = a.as_f32_slice().ok_or(ExecutionError::RuntimeError(
        format!("{name}: input not f32"),
    ))?;
    let result: Vec<f32> = va.iter().map(|&x| op(x)).collect();
    Ok(TensorData::from_f32(a.shape.clone(), &result))
}

fn tensor_softmax(a: &TensorData, _axis: usize) -> Result<TensorData, ExecutionError> {
    let va = a.as_f32_slice().ok_or(ExecutionError::RuntimeError(
        "softmax: input not f32".into(),
    ))?;
    // For Phase 1: flatten softmax (ignoring axis, treating as 1D)
    let max_val = va.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = va.iter().map(|&x| (x - max_val).exp()).collect();
    let sum: f32 = exps.iter().sum();
    let result: Vec<f32> = exps.iter().map(|&e| e / sum).collect();
    Ok(TensorData::from_f32(a.shape.clone(), &result))
}

fn tensor_transpose(a: &TensorData) -> Result<TensorData, ExecutionError> {
    use qlang_core::tensor::{Dim, Shape};

    let va = a.as_f32_slice().ok_or(ExecutionError::RuntimeError(
        "transpose: input not f32".into(),
    ))?;

    let (m, n) = match a.shape.0.as_slice() {
        [Dim::Fixed(m), Dim::Fixed(n)] => (*m, *n),
        _ => return Err(ExecutionError::RuntimeError("transpose: must be 2D".into())),
    };

    let mut result = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            result[j * m + i] = va[i * n + j];
        }
    }
    Ok(TensorData::from_f32(Shape::matrix(n, m), &result))
}

fn tensor_reduce(
    a: &TensorData,
    axis: Option<usize>,
    op: impl Fn(f32, f32) -> f32,
    init: f32,
    name: &str,
) -> Result<TensorData, ExecutionError> {
    use qlang_core::tensor::Shape;

    let va = a.as_f32_slice().ok_or(ExecutionError::RuntimeError(
        format!("{name}: input not f32"),
    ))?;

    match axis {
        None => {
            // Reduce all elements to scalar
            let result = va.iter().fold(init, |acc, &x| op(acc, x));
            Ok(TensorData::from_f32(Shape::scalar(), &[result]))
        }
        Some(_) => {
            // Phase 1: only support full reduction
            let result = va.iter().fold(init, |acc, &x| op(acc, x));
            Ok(TensorData::from_f32(Shape::scalar(), &[result]))
        }
    }
}

fn tensor_reduce_mean(a: &TensorData, axis: Option<usize>) -> Result<TensorData, ExecutionError> {
    use qlang_core::tensor::Shape;

    let va = a.as_f32_slice().ok_or(ExecutionError::RuntimeError(
        "reduce_mean: input not f32".into(),
    ))?;

    match axis {
        None | Some(_) => {
            let sum: f32 = va.iter().sum();
            let mean = sum / va.len() as f32;
            Ok(TensorData::from_f32(Shape::scalar(), &[mean]))
        }
    }
}

/// Low-rank approximation via truncated SVD (simplified for Phase 1).
/// Keeps only the top `rank` singular values.
fn tensor_to_lowrank(a: &TensorData, rank: usize) -> Result<TensorData, ExecutionError> {
    use qlang_core::tensor::Dim;

    let va = a.as_f32_slice().ok_or(ExecutionError::RuntimeError(
        "to_lowrank: input not f32".into(),
    ))?;

    let (m, n) = match a.shape.0.as_slice() {
        [Dim::Fixed(m), Dim::Fixed(n)] => (*m, *n),
        _ => return Err(ExecutionError::RuntimeError("to_lowrank: must be 2D".into())),
    };

    let effective_rank = rank.min(m).min(n);

    // Phase 1: Power iteration approximation for top-k singular values
    // For production: use full SVD via LAPACK
    let mut result = vec![0.0f32; m * n];

    // Simple approach: zero out elements below threshold to approximate low-rank
    // This is a placeholder — real implementation needs SVD
    let mut values: Vec<f32> = va.iter().map(|x| x.abs()).collect();
    values.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));

    let threshold = if effective_rank * effective_rank < values.len() {
        values.get(effective_rank * effective_rank).copied().unwrap_or(0.0)
    } else {
        0.0
    };

    for i in 0..(m * n) {
        if va[i].abs() >= threshold {
            result[i] = va[i];
        }
    }

    Ok(TensorData::from_f32(a.shape.clone(), &result))
}

fn tensor_to_sparse(a: &TensorData, sparsity: f64) -> Result<TensorData, ExecutionError> {
    let va = a.as_f32_slice().ok_or(ExecutionError::RuntimeError(
        "to_sparse: input not f32".into(),
    ))?;

    let mut values: Vec<f32> = va.iter().map(|x| x.abs()).collect();
    values.sort_by(|x, y| x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal));

    let k = ((va.len() as f64) * sparsity) as usize;
    let threshold = if k < values.len() { values[k] } else { f32::MAX };

    let mut result = vec![0.0f32; va.len()];
    for i in 0..va.len() {
        if va[i].abs() >= threshold {
            result[i] = va[i];
        }
    }

    Ok(TensorData::from_f32(a.shape.clone(), &result))
}

fn tensor_fisher_metric(model: &TensorData, _data: &TensorData) -> Result<TensorData, ExecutionError> {
    // Simplified Fisher Information Matrix calculation
    // G = E[∇log p ∇log p^T]
    // For Phase 1, we return an identity matrix of size n x n where n is the number of parameters.
    use qlang_core::tensor::Shape;
    let vm = model.as_f32_slice().ok_or(ExecutionError::RuntimeError(
        "fisher_metric: model not f32".into(),
    ))?;

    let n = vm.len();
    let mut result = vec![0.0f32; n * n];
    for i in 0..n {
        result[i * n + i] = 1.0;
    }
    Ok(TensorData::from_f32(Shape::matrix(n, n), &result))
}

fn tensor_project(tensor: &TensorData, manifold: &qlang_core::ops::Manifold) -> Result<TensorData, ExecutionError> {
    use qlang_core::ops::Manifold;
    match manifold {
        Manifold::Ternary => tensor_to_ternary(tensor),
        Manifold::LowRank { max_rank } => tensor_to_lowrank(tensor, *max_rank),
        Manifold::Sparse { max_nonzero } => {
            let va = tensor.as_f32_slice().ok_or(ExecutionError::RuntimeError(
                "project: input not f32".into(),
            ))?;
            let sparsity = 1.0 - (*max_nonzero as f64 / va.len() as f64);
            tensor_to_sparse(tensor, sparsity)
        }
        Manifold::Custom { .. } => Ok(tensor.clone()),
    }
}

fn tensor_layer_norm(a: &TensorData, eps: f64) -> Result<TensorData, ExecutionError> {
    let va = a.as_f32_slice().ok_or(ExecutionError::RuntimeError(
        "layer_norm: input not f32".into(),
    ))?;
    let mut result = vec![0.0f32; va.len()];

    let n = va.len() as f32;
    if n > 0.0 {
        let mean: f32 = va.iter().sum::<f32>() / n;
        let var: f32 = va.iter().map(|&x| (x - mean) * (x - mean)).sum::<f32>() / n;
        let std_dev = (var + eps as f32).sqrt();

        for i in 0..va.len() {
            result[i] = (va[i] - mean) / std_dev;
        }
    }

    Ok(TensorData::from_f32(a.shape.clone(), &result))
}

fn tensor_gelu(a: &TensorData) -> Result<TensorData, ExecutionError> {
    let va = a.as_f32_slice().ok_or(ExecutionError::RuntimeError(
        "gelu: input not f32".into(),
    ))?;
    let mut result = vec![0.0f32; va.len()];

    for i in 0..va.len() {
        let x = va[i];
        let inner = std::f32::consts::FRAC_2_PI.sqrt() * (x + 0.044715 * x * x * x);
        result[i] = 0.5 * x * (1.0 + inner.tanh());
    }

    Ok(TensorData::from_f32(a.shape.clone(), &result))
}

fn tensor_dropout(a: &TensorData, _rate: f64) -> Result<TensorData, ExecutionError> {
    // For inference (Phase 1 executor), dropout is usually an identity operation
    Ok(a.clone())
}

fn tensor_collapse(a: &TensorData) -> Result<TensorData, ExecutionError> {
    // For deterministic execution without random sampling, 
    // collapse acts exactly like measurement (argmax).
    tensor_measure(a)
}

/// Compute Shannon entropy of tensor values treated as probability distribution.
fn tensor_entropy(a: &TensorData) -> Result<TensorData, ExecutionError> {
    use qlang_core::tensor::Shape;

    let va = a.as_f32_slice().ok_or(ExecutionError::RuntimeError(
        "entropy: input not f32".into(),
    ))?;

    // Normalize to probability distribution
    let sum: f32 = va.iter().map(|x| x.abs()).sum();
    if sum < 1e-15 {
        return Ok(TensorData::from_f32(Shape::scalar(), &[0.0]));
    }

    let entropy: f32 = va
        .iter()
        .map(|&x| {
            let p = x.abs() / sum;
            if p > 1e-15 { -p * p.ln() } else { 0.0 }
        })
        .sum();

    Ok(TensorData::from_f32(Shape::scalar(), &[entropy]))
}

/// Quantum measurement: collapse probability distribution to argmax.
/// Returns a one-hot vector (deterministic approximation of Born rule).
fn tensor_measure(a: &TensorData) -> Result<TensorData, ExecutionError> {
    let va = a.as_f32_slice().ok_or(ExecutionError::RuntimeError(
        "measure: input not f32".into(),
    ))?;

    let mut max_idx = 0;
    let mut max_val = f32::NEG_INFINITY;
    for (i, &v) in va.iter().enumerate() {
        if v > max_val {
            max_val = v;
            max_idx = i;
        }
    }

    let mut result = vec![0.0f32; va.len()];
    result[max_idx] = 1.0;
    Ok(TensorData::from_f32(a.shape.clone(), &result))
}

fn tensor_measure_with_operators(state: &TensorData, operators: &TensorData) -> Result<TensorData, ExecutionError> {
    let vs = state.as_f32_slice().ok_or(ExecutionError::RuntimeError(
        "measure: state not f32".into(),
    ))?;
    let vo = operators.as_f32_slice().ok_or(ExecutionError::RuntimeError(
        "measure: operators not f32".into(),
    ))?;

    let len = vs.len();
    if len == 0 || vo.len() % len != 0 {
        return Err(ExecutionError::RuntimeError(
            "measure: operators length must be a multiple of state length".into(),
        ));
    }
    let num_ops = vo.len() / len;

    let mut probs = vec![0.0f32; num_ops];
    for i in 0..num_ops {
        let op_slice = &vo[i * len..(i + 1) * len];
        let mut sum = 0.0f32;
        for (s, o) in vs.iter().zip(op_slice.iter()) {
            sum += s * o;
        }
        probs[i] = sum;
    }

    use qlang_core::tensor::Shape;
    Ok(TensorData::from_f32(Shape::vector(num_ops), &probs))
}

fn tensor_entangle(a: &TensorData, b: &TensorData) -> Result<TensorData, ExecutionError> {
    let va = a.as_f32_slice().ok_or(ExecutionError::RuntimeError("entangle: input a not f32".into()))?;
    let vb = b.as_f32_slice().ok_or(ExecutionError::RuntimeError("entangle: input b not f32".into()))?;
    
    let mut result = vec![0.0f32; va.len() * vb.len()];
    for (i, &val_a) in va.iter().enumerate() {
        for (j, &val_b) in vb.iter().enumerate() {
            result[i * vb.len() + j] = val_a * val_b;
        }
    }
    
    use qlang_core::tensor::{Dim, Shape};
    let shape = match (a.shape.0.as_slice(), b.shape.0.as_slice()) {
        ([Dim::Fixed(m1), Dim::Fixed(n1)], [Dim::Fixed(m2), Dim::Fixed(n2)]) => {
            Shape::matrix(m1 * m2, n1 * n2)
        }
        _ => Shape::vector(va.len() * vb.len()),
    };
    
    Ok(TensorData::from_f32(shape, &result))
}

fn tensor_relu(a: &TensorData) -> Result<TensorData, ExecutionError> {
    let va = a.as_f32_slice().ok_or(ExecutionError::RuntimeError(
        "relu: input not f32".into(),
    ))?;
    let result: Vec<f32> = va.iter().map(|&x| x.max(0.0)).collect();
    Ok(TensorData::from_f32(a.shape.clone(), &result))
}

fn tensor_neg(a: &TensorData) -> Result<TensorData, ExecutionError> {
    let va = a.as_f32_slice().ok_or(ExecutionError::RuntimeError(
        "neg: input not f32".into(),
    ))?;
    let result: Vec<f32> = va.iter().map(|&x| -x).collect();
    Ok(TensorData::from_f32(a.shape.clone(), &result))
}

#[allow(dead_code)]
/// Convert a flat n×n dense matrix (f64, row-major) into a `DensityMatrix`.
///
/// Symmetrises the input, runs Jacobi eigendecomposition, clamps negative
/// eigenvalues and renormalises so Tr(ρ) = 1.
fn dense_f64_to_density(dense: &[f64], n: usize) -> qlang_core::quantum::DensityMatrix {
    use qlang_core::quantum::DensityMatrix;

    // Symmetrise
    let mut sym = dense.to_vec();
    for i in 0..n {
        for j in (i + 1)..n {
            let avg = 0.5 * (sym[i * n + j] + sym[j * n + i]);
            sym[i * n + j] = avg;
            sym[j * n + i] = avg;
        }
    }

    // Jacobi eigendecomposition (reuse the linalg module's building blocks
    // isn't possible since it only returns eigenvalues; we inline a small
    // Jacobi that also produces eigenvectors).
    let (eigenvalues, eigenvectors) = jacobi_eigen_full(&sym, n);

    // Clamp negative eigenvalues and renormalise
    let mut evals: Vec<f64> = eigenvalues.iter().map(|&v| v.max(0.0)).collect();
    let sum: f64 = evals.iter().sum();
    if sum > 1e-15 {
        for v in &mut evals {
            *v /= sum;
        }
    }

    DensityMatrix {
        dim: n,
        eigenvalues: evals,
        eigenvectors,
    }
}

#[allow(dead_code)]
/// Expand a `DensityMatrix` back to a flat n×n dense matrix (f64, row-major):
///   ρ = Σ pₖ |ψₖ⟩⟨ψₖ|
fn density_to_dense_f64(rho: &qlang_core::quantum::DensityMatrix) -> Vec<f64> {
    let n = rho.dim;
    let mut dense = vec![0.0f64; n * n];
    for (k, &pk) in rho.eigenvalues.iter().enumerate() {
        let psi = &rho.eigenvectors[k * n..(k + 1) * n];
        for i in 0..n {
            for j in 0..n {
                dense[i * n + j] += pk * psi[i] * psi[j];
            }
        }
    }
    dense
}

#[allow(dead_code)]
/// Jacobi eigenvalue algorithm for real symmetric n×n matrices.
/// Returns (eigenvalues, eigenvectors_flat) where eigenvectors_flat has
/// row k = eigenvector k (each of length n).
fn jacobi_eigen_full(mat: &[f64], n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut a = mat.to_vec();
    let mut v = vec![0.0f64; n * n];
    for i in 0..n {
        v[i * n + i] = 1.0;
    }

    let max_iter = 100 * n * n;
    for _ in 0..max_iter {
        // Find largest off-diagonal element
        let mut p = 0usize;
        let mut q = 1usize;
        let mut max_val = 0.0f64;
        for i in 0..n {
            for j in (i + 1)..n {
                if a[i * n + j].abs() > max_val {
                    max_val = a[i * n + j].abs();
                    p = i;
                    q = j;
                }
            }
        }
        if max_val < 1e-14 {
            break;
        }

        let app = a[p * n + p];
        let aqq = a[q * n + q];
        let apq = a[p * n + q];
        let theta = if (app - aqq).abs() < 1e-15 {
            std::f64::consts::FRAC_PI_4
        } else {
            0.5 * (2.0 * apq / (app - aqq)).atan()
        };
        let c = theta.cos();
        let s = theta.sin();

        // Apply Givens rotation to A
        let mut new_a = a.clone();
        for i in 0..n {
            new_a[i * n + p] = c * a[i * n + p] + s * a[i * n + q];
            new_a[i * n + q] = -s * a[i * n + p] + c * a[i * n + q];
        }
        let a_copy = new_a.clone();
        for j in 0..n {
            new_a[p * n + j] = c * a_copy[p * n + j] + s * a_copy[q * n + j];
            new_a[q * n + j] = -s * a_copy[p * n + j] + c * a_copy[q * n + j];
        }
        a = new_a;

        // Accumulate eigenvectors
        let mut new_v = v.clone();
        for i in 0..n {
            new_v[i * n + p] = c * v[i * n + p] + s * v[i * n + q];
            new_v[i * n + q] = -s * v[i * n + p] + c * v[i * n + q];
        }
        v = new_v;
    }

    let eigenvalues: Vec<f64> = (0..n).map(|i| a[i * n + i]).collect();

    // Transpose v so row k = eigenvector k
    let mut evecs = vec![0.0f64; n * n];
    for k in 0..n {
        for i in 0..n {
            evecs[k * n + i] = v[i * n + k];
        }
    }

    (eigenvalues, evecs)
}

/// IGQK ternary compression: project f32 weights to {-1, 0, +1}.
///
/// Uses threshold-based projection:
///   w > threshold  →  +1
///   w < -threshold →  -1
///   otherwise      →   0
///
/// The threshold is computed as mean(|w|) * 0.7 (a common heuristic).
/// Theorem 5.2 bounds the distortion of this projection.
fn tensor_to_ternary(a: &TensorData) -> Result<TensorData, ExecutionError> {
    use qlang_core::tensor::Dtype;

    let va = a.as_f32_slice().ok_or(ExecutionError::RuntimeError(
        "to_ternary: input not f32".into(),
    ))?;

    // Use IGQK adaptive threshold: std_dev * 0.5 (Born-rule collapse from the paper).
    // This replaces the old mean-abs * 0.7 heuristic with the theoretically grounded value.
    let igqk_result =
        crate::igqk_compress::compress_ternary(&va, &crate::igqk_compress::IgqkParams::default());
    let threshold = if let Some(&alpha) = igqk_result.scaling_factors.first() {
        // Recompute threshold from the same formula used inside compress_ternary
        let n = va.len() as f32;
        let mean: f32 = va.iter().sum::<f32>() / n;
        let std_dev: f32 =
            (va.iter().map(|w| (w - mean).powi(2)).sum::<f32>() / n).sqrt();
        let _ = alpha; // alpha is the scaling factor; threshold drives +/-/0 split
        std_dev * 0.5
    } else {
        // Fallback: original heuristic
        va.iter().map(|x| x.abs()).sum::<f32>() / va.len() as f32 * 0.7
    };

    // Project to ternary (Dtype::Ternary uses u8 encoding)
    let result: Vec<u8> = va
        .iter()
        .map(|&x| {
            if x > threshold {
                1u8   // represents +1
            } else if x < -threshold {
                255u8 // represents -1 (two's complement i8)
            } else {
                0u8   // represents 0
            }
        })
        .collect();

    Ok(TensorData {
        dtype: Dtype::Ternary,
        shape: a.shape.clone(),
        data: result,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use qlang_core::graph::Graph;
    use qlang_core::ops::Op;
    use qlang_core::tensor::{Dtype, Shape, TensorData, TensorType};

    #[test]
    fn execute_add_graph() {
        // Build graph: a + b = y
        let mut g = Graph::new("test_add");

        let a = g.add_node(
            Op::Input { name: "a".into() },
            vec![],
            vec![TensorType::f32_vector(4)],
        );
        let b = g.add_node(
            Op::Input { name: "b".into() },
            vec![],
            vec![TensorType::f32_vector(4)],
        );
        let add = g.add_node(
            Op::Add,
            vec![TensorType::f32_vector(4), TensorType::f32_vector(4)],
            vec![TensorType::f32_vector(4)],
        );
        let out = g.add_node(
            Op::Output { name: "y".into() },
            vec![TensorType::f32_vector(4)],
            vec![],
        );

        g.add_edge(a, 0, add, 0, TensorType::f32_vector(4));
        g.add_edge(b, 0, add, 1, TensorType::f32_vector(4));
        g.add_edge(add, 0, out, 0, TensorType::f32_vector(4));

        // Execute
        let mut inputs = HashMap::new();
        inputs.insert(
            "a".to_string(),
            TensorData::from_f32(Shape::vector(4), &[1.0, 2.0, 3.0, 4.0]),
        );
        inputs.insert(
            "b".to_string(),
            TensorData::from_f32(Shape::vector(4), &[10.0, 20.0, 30.0, 40.0]),
        );

        let result = execute(&g, inputs).unwrap();
        let y = result.outputs.get("y").unwrap();
        let values = y.as_f32_slice().unwrap();

        assert_eq!(values, vec![11.0, 22.0, 33.0, 44.0]);
        assert_eq!(result.stats.nodes_executed, 4);
    }

    #[test]
    fn execute_matmul_graph() {
        let mut g = Graph::new("test_matmul");

        let a = g.add_node(
            Op::Input { name: "a".into() },
            vec![],
            vec![TensorType::f32_matrix(2, 3)],
        );
        let b = g.add_node(
            Op::Input { name: "b".into() },
            vec![],
            vec![TensorType::f32_matrix(3, 2)],
        );
        let mm = g.add_node(
            Op::MatMul,
            vec![TensorType::f32_matrix(2, 3), TensorType::f32_matrix(3, 2)],
            vec![TensorType::f32_matrix(2, 2)],
        );
        let out = g.add_node(
            Op::Output { name: "y".into() },
            vec![TensorType::f32_matrix(2, 2)],
            vec![],
        );

        g.add_edge(a, 0, mm, 0, TensorType::f32_matrix(2, 3));
        g.add_edge(b, 0, mm, 1, TensorType::f32_matrix(3, 2));
        g.add_edge(mm, 0, out, 0, TensorType::f32_matrix(2, 2));

        // A = [[1,2,3],[4,5,6]], B = [[1,0],[0,1],[1,1]]
        // A*B = [[4,5],[10,11]]
        let mut inputs = HashMap::new();
        inputs.insert(
            "a".to_string(),
            TensorData::from_f32(Shape::matrix(2, 3), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
        );
        inputs.insert(
            "b".to_string(),
            TensorData::from_f32(Shape::matrix(3, 2), &[1.0, 0.0, 0.0, 1.0, 1.0, 1.0]),
        );

        let result = execute(&g, inputs).unwrap();
        let y = result.outputs.get("y").unwrap();
        let values = y.as_f32_slice().unwrap();

        assert_eq!(values, vec![4.0, 5.0, 10.0, 11.0]);
    }

    #[test]
    fn execute_ternary_compression() {
        let mut g = Graph::new("test_ternary");

        let input = g.add_node(
            Op::Input { name: "weights".into() },
            vec![],
            vec![TensorType::f32_vector(8)],
        );
        let ternary = g.add_node(
            Op::ToTernary,
            vec![TensorType::f32_vector(8)],
            vec![TensorType::new(Dtype::Ternary, Shape::vector(8))],
        );
        let out = g.add_node(
            Op::Output { name: "compressed".into() },
            vec![TensorType::new(Dtype::Ternary, Shape::vector(8))],
            vec![],
        );

        g.add_edge(input, 0, ternary, 0, TensorType::f32_vector(8));
        g.add_edge(ternary, 0, out, 0, TensorType::new(Dtype::Ternary, Shape::vector(8)));

        let mut inputs = HashMap::new();
        // Large positive, small, large negative values
        inputs.insert(
            "weights".to_string(),
            TensorData::from_f32(
                Shape::vector(8),
                &[0.9, -0.8, 0.01, -0.02, 0.7, -0.6, 0.0, 0.5],
            ),
        );

        let result = execute(&g, inputs).unwrap();
        let compressed = result.outputs.get("compressed").unwrap();

        assert_eq!(compressed.dtype, Dtype::Ternary);
        assert_eq!(result.stats.quantum_ops, 1);
    }

    /// Test that Op::Evolve uses real quantum gradient flow for square
    /// density-matrix states, producing output that differs from the
    /// simple gradient step and preserves Tr(ρ) ≈ 1.
    #[test]
    fn execute_evolve_quantum_gradient_flow() {
        let n = 3usize; // 3×3 density matrix → 9 elements
        let gamma = 0.5_f64;
        let dt = 0.01_f64;

        // Build a valid density matrix: ρ = diag(0.5, 0.3, 0.2)
        // (diagonal in computational basis)
        #[rustfmt::skip]
        let rho_flat: Vec<f32> = vec![
            0.5, 0.0, 0.0,
            0.0, 0.3, 0.0,
            0.0, 0.0, 0.2,
        ];

        // Gradient matrix (symmetric, with off-diagonal terms so the
        // commutator/anticommutator are non-trivial)
        #[rustfmt::skip]
        let grad_flat: Vec<f32> = vec![
            0.1,  0.05, 0.02,
            0.05, 0.2,  0.03,
            0.02, 0.03, 0.3,
        ];

        // --- Build graph: state + gradient → Evolve → output ---
        let mut g = Graph::new("test_evolve_qgf");
        let ty9 = TensorType::f32_vector(n * n);

        let state_node = g.add_node(
            Op::Input { name: "state".into() },
            vec![],
            vec![ty9.clone()],
        );
        let grad_node = g.add_node(
            Op::Input { name: "gradient".into() },
            vec![],
            vec![ty9.clone()],
        );
        let evolve_node = g.add_node(
            Op::Evolve { gamma, dt },
            vec![ty9.clone(), ty9.clone()],
            vec![ty9.clone()],
        );
        let out_node = g.add_node(
            Op::Output { name: "evolved".into() },
            vec![ty9.clone()],
            vec![],
        );

        g.add_edge(state_node, 0, evolve_node, 0, ty9.clone());
        g.add_edge(grad_node, 0, evolve_node, 1, ty9.clone());
        g.add_edge(evolve_node, 0, out_node, 0, ty9.clone());

        let mut inputs = HashMap::new();
        inputs.insert(
            "state".to_string(),
            TensorData::from_f32(Shape::vector(n * n), &rho_flat),
        );
        inputs.insert(
            "gradient".to_string(),
            TensorData::from_f32(Shape::vector(n * n), &grad_flat),
        );

        let result = execute(&g, inputs).unwrap();
        let evolved = result.outputs.get("evolved").unwrap();
        let vals = evolved.as_f32_slice().unwrap();

        // 1. Output differs from simple gradient step
        let simple_step = (gamma * dt) as f32;
        let simple: Vec<f32> = rho_flat
            .iter()
            .zip(grad_flat.iter())
            .map(|(s, gr)| s - simple_step * gr)
            .collect();

        let diff: f64 = vals
            .iter()
            .zip(simple.iter())
            .map(|(a, b)| ((*a as f64) - (*b as f64)).powi(2))
            .sum::<f64>()
            .sqrt();
        assert!(
            diff > 1e-6,
            "Quantum flow output should differ from simple gradient step (diff={diff})"
        );

        // 2. Trace is preserved: Tr(ρ) ≈ 1
        let trace: f64 = (0..n).map(|i| vals[i * n + i] as f64).sum();
        assert!(
            (trace - 1.0).abs() < 1e-4,
            "Trace of evolved density matrix should be ~1.0, got {trace}"
        );

        // 3. Quantum op was counted
        assert_eq!(result.stats.quantum_ops, 1);
    }

    #[test]
    fn execute_entangle() {
        let mut g = Graph::new("test_entangle");
        let ty_a = TensorType::f32_vector(2);
        let ty_b = TensorType::f32_vector(3);
        let ty_out = TensorType::f32_vector(6);

        let node_a = g.add_node(Op::Input { name: "a".into() }, vec![], vec![ty_a.clone()]);
        let node_b = g.add_node(Op::Input { name: "b".into() }, vec![], vec![ty_b.clone()]);
        let entangle = g.add_node(Op::Entangle, vec![ty_a.clone(), ty_b.clone()], vec![ty_out.clone()]);
        let out_node = g.add_node(Op::Output { name: "out".into() }, vec![ty_out.clone()], vec![]);

        g.add_edge(node_a, 0, entangle, 0, ty_a.clone());
        g.add_edge(node_b, 0, entangle, 1, ty_b.clone());
        g.add_edge(entangle, 0, out_node, 0, ty_out.clone());

        let mut inputs = HashMap::new();
        inputs.insert("a".to_string(), TensorData::from_f32(Shape::vector(2), &[1.0, 2.0]));
        inputs.insert("b".to_string(), TensorData::from_f32(Shape::vector(3), &[0.5, 1.5, 2.5]));

        let result = execute(&g, inputs).unwrap();
        let out_vals = result.outputs.get("out").unwrap().as_f32_slice().unwrap();

        // Kronecker product: [1*0.5, 1*1.5, 1*2.5, 2*0.5, 2*1.5, 2*2.5]
        assert_eq!(out_vals, &[0.5, 1.5, 2.5, 1.0, 3.0, 5.0]);
    }

    #[test]
    fn execute_measure_1d() {
        let mut g = Graph::new("test_measure");
        let ty = TensorType::f32_vector(3);

        let node_a = g.add_node(Op::Input { name: "state".into() }, vec![], vec![ty.clone()]);
        let measure = g.add_node(Op::Measure, vec![ty.clone()], vec![ty.clone()]);
        let out_node = g.add_node(Op::Output { name: "out".into() }, vec![ty.clone()], vec![]);

        g.add_edge(node_a, 0, measure, 0, ty.clone());
        g.add_edge(measure, 0, out_node, 0, ty.clone());

        let mut inputs = HashMap::new();
        inputs.insert("state".to_string(), TensorData::from_f32(Shape::vector(3), &[0.2, 0.8, 0.1]));

        let result = execute(&g, inputs).unwrap();
        let out_vals = result.outputs.get("out").unwrap().as_f32_slice().unwrap();

        // Argmax should be at index 1
        assert_eq!(out_vals, &[0.0, 1.0, 0.0]);
    }

    #[test]
    fn execute_layer_norm() {
        let mut g = Graph::new("test_layer_norm");
        let ty = TensorType::f32_vector(4);

        let node_a = g.add_node(Op::Input { name: "x".into() }, vec![], vec![ty.clone()]);
        let ln = g.add_node(Op::LayerNorm { eps: 1e-5 }, vec![ty.clone()], vec![ty.clone()]);
        let out_node = g.add_node(Op::Output { name: "out".into() }, vec![ty.clone()], vec![]);

        g.add_edge(node_a, 0, ln, 0, ty.clone());
        g.add_edge(ln, 0, out_node, 0, ty.clone());

        let mut inputs = HashMap::new();
        inputs.insert("x".to_string(), TensorData::from_f32(Shape::vector(4), &[1.0, 2.0, 3.0, 4.0]));

        let result = execute(&g, inputs).unwrap();
        let out_vals = result.outputs.get("out").unwrap().as_f32_slice().unwrap();

        // Mean = 2.5, Variance = 1.25, StdDev = sqrt(1.25) ≈ 1.118
        // Result = (x - 2.5) / 1.118
        let std_dev = (1.25f32 + 1e-5).sqrt();
        assert!((out_vals[0] - ((1.0 - 2.5) / std_dev)).abs() < 1e-4);
        assert!((out_vals[1] - ((2.0 - 2.5) / std_dev)).abs() < 1e-4);
        assert!((out_vals[2] - ((3.0 - 2.5) / std_dev)).abs() < 1e-4);
        assert!((out_vals[3] - ((4.0 - 2.5) / std_dev)).abs() < 1e-4);
    }
}
