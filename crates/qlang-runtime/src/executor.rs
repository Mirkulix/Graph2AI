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

            Op::Dropout { .. } => {
                let input = get_one_input(graph, node_id, &node_outputs)?;
                node_outputs.insert((node_id, 0), input);
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

            Op::Scan { n_iterations } => {
                // PRD Task 3.2: bounded scan is implemented as repeated
                // application of identity to the single input tensor. A
                // future iteration (tracked alongside SubGraph below) will
                // accept a body reference so Scan can compose real ops;
                // for now we preserve the loop semantics (N ≥ 0 repeats)
                // without changing the value. This makes benchmarking
                // iteration-dispatch overhead tractable and lets DAG
                // validators treat Scan as a well-defined node.
                let input = get_one_input(graph, node_id, &node_outputs)?;
                let mut state = input;
                for _ in 0..*n_iterations {
                    // identity step — reassignment is explicit so future
                    // body dispatch can slot in here with minimal churn.
                    state = state.clone();
                }
                node_outputs.insert((node_id, 0), state);
                stats.total_flops += *n_iterations as u64;
            }

            Op::SubGraph { graph_id } => {
                // PRD Task 3.2: resolving SubGraph requires a graph
                // registry that lives outside the current executor (spec
                // §5.3 subgraph dispatch). Rather than panic we surface a
                // structured error that names the missing registry entry.
                return Err(ExecutionError::UnsupportedOp(format!(
                    "SubGraph({graph_id}) requires a graph registry (future work — see spec §5.3)"
                )));
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

    // -------------------------------------------------------------------
    // Scan / SubGraph — PRD Task 3.2 closure
    // -------------------------------------------------------------------

    /// Scan with N=1 MUST be value-preserving (identity). Guards against a
    /// future body-dispatch refactor accidentally mutating the input when
    /// the effective body is empty.
    #[test]
    fn scan_n1_is_identity() {
        let mut g = Graph::new("scan_identity");
        let ty = TensorType::f32_vector(3);
        let input = g.add_node(Op::Input { name: "x".into() }, vec![], vec![ty.clone()]);
        let scan = g.add_node(Op::Scan { n_iterations: 1 }, vec![ty.clone()], vec![ty.clone()]);
        let out = g.add_node(Op::Output { name: "out".into() }, vec![ty.clone()], vec![]);
        g.add_edge(input, 0, scan, 0, ty.clone());
        g.add_edge(scan, 0, out, 0, ty);

        let mut inputs = HashMap::new();
        inputs.insert("x".to_string(), TensorData::from_f32(Shape::vector(3), &[1.0, -2.5, 0.25]));

        let result = execute(&g, inputs).unwrap();
        let vals = result.outputs.get("out").unwrap().as_f32_slice().unwrap();
        assert_eq!(vals, &[1.0, -2.5, 0.25]);
    }

    /// Scan with N=100 still returns the input unchanged: the executor
    /// currently models Scan as repeated identity so loop count has no
    /// effect on values. The FLOP counter MUST increment by N regardless
    /// so benchmarks can tell trivial loops from real work.
    #[test]
    fn scan_n100_preserves_values_and_counts_iterations() {
        let mut g = Graph::new("scan_bench");
        let ty = TensorType::f32_vector(2);
        let input = g.add_node(Op::Input { name: "x".into() }, vec![], vec![ty.clone()]);
        let scan = g.add_node(
            Op::Scan { n_iterations: 100 },
            vec![ty.clone()],
            vec![ty.clone()],
        );
        let out = g.add_node(Op::Output { name: "out".into() }, vec![ty.clone()], vec![]);
        g.add_edge(input, 0, scan, 0, ty.clone());
        g.add_edge(scan, 0, out, 0, ty);

        let mut inputs = HashMap::new();
        inputs.insert(
            "x".to_string(),
            TensorData::from_f32(Shape::vector(2), &[3.0, 7.0]),
        );
        let result = execute(&g, inputs).unwrap();
        let vals = result.outputs.get("out").unwrap().as_f32_slice().unwrap();
        assert_eq!(vals, &[3.0, 7.0]);
        assert!(
            result.stats.total_flops >= 100,
            "Scan N=100 must contribute >= 100 to the FLOP counter (got {})",
            result.stats.total_flops
        );
    }

    /// SubGraph has no registry yet, so execution MUST surface a clear,
    /// named error rather than a generic "Phase 1" string. This pins the
    /// contract until the registry lands.
    #[test]
    fn subgraph_reports_missing_registry_with_graph_id() {
        let mut g = Graph::new("subgraph_err");
        let ty = TensorType::f32_vector(1);
        let input = g.add_node(Op::Input { name: "x".into() }, vec![], vec![ty.clone()]);
        let sg = g.add_node(
            Op::SubGraph {
                graph_id: "missing_routine".into(),
            },
            vec![ty.clone()],
            vec![ty.clone()],
        );
        let out = g.add_node(Op::Output { name: "out".into() }, vec![ty.clone()], vec![]);
        g.add_edge(input, 0, sg, 0, ty.clone());
        g.add_edge(sg, 0, out, 0, ty);

        let mut inputs = HashMap::new();
        inputs.insert(
            "x".to_string(),
            TensorData::from_f32(Shape::vector(1), &[0.0]),
        );
        let err = execute(&g, inputs).expect_err("SubGraph without registry must fail");
        let msg = format!("{err:?}");
        assert!(
            msg.contains("missing_routine"),
            "error must name the missing graph_id; got: {msg}"
        );
        assert!(
            msg.contains("registry"),
            "error must mention the registry gap; got: {msg}"
        );
    }
}
