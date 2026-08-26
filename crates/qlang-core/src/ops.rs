use serde::{Deserialize, Serialize};
use std::fmt;

/// Operation catalog — every computation a QLANG node can perform.
///
/// These map directly to machine instructions or LLVM IR intrinsics
/// at compile time. No interpretation overhead at runtime.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Op {
    // === Graph I/O ===
    /// External input to the graph
    Input { name: String },
    /// Output from the graph
    Output { name: String },
    /// Constant tensor embedded in the graph
    Constant,

    // === Tensor Operations (→ direct register/SIMD instructions) ===
    Add,
    Sub,
    Mul,
    Div,
    Neg,
    Exp,
    Log,
    MatMul,
    Transpose,
    Reshape { target_shape: Vec<usize> },
    Slice { start: Vec<usize>, end: Vec<usize> },
    Concat { axis: usize },
    ReduceSum { axis: Option<usize> },
    ReduceMean { axis: Option<usize> },
    ReduceMax { axis: Option<usize> },

    // === Activation Functions ===
    Relu,
    Sigmoid,
    Tanh,
    Softmax { axis: usize },

    // === Transformer Operations ===
    /// Layer normalization: (x - mean) / sqrt(var + eps) * gamma + beta
    LayerNorm { eps: f64 },
    /// Scaled dot-product attention: softmax(Q·K^T / sqrt(d_k)) · V
    Attention { n_heads: usize, d_model: usize },
    /// Token/positional embedding lookup
    Embedding { vocab_size: usize, d_model: usize },
    /// Residual connection: x + f(x)
    Residual,
    /// GELU activation: x * Φ(x) ≈ 0.5x(1 + tanh(√(2/π)(x + 0.044715x³)))
    Gelu,
    /// Dropout (inference identity)
    Dropout { rate: f64 },

    // === LLM / Ollama Operations ===
    /// Generate text via local Ollama LLM
    OllamaGenerate { model: String },
    /// Chat completion via local Ollama LLM
    OllamaChat { model: String },

    // === Control Flow ===
    /// Conditional: evaluates BOTH branches (quantum-style), selects based on predicate
    Cond,
    /// Bounded iteration
    Scan { n_iterations: usize },
    /// Execute a sub-graph
    SubGraph { graph_id: String },
    /// ArgMax along last axis.
    ArgMax,
}

impl Op {
    /// Number of input ports this operation expects.
    pub fn n_inputs(&self) -> usize {
        match self {
            Op::Input { .. } | Op::Constant => 0,
            Op::Output { .. } | Op::Neg | Op::Exp | Op::Log | Op::Relu | Op::Sigmoid | Op::Tanh
            | Op::Softmax { .. } | Op::Transpose | Op::Reshape { .. }
            | Op::Slice { .. } | Op::ArgMax => 1,
            Op::ReduceSum { .. } | Op::ReduceMean { .. } | Op::ReduceMax { .. }
            | Op::LayerNorm { .. } | Op::Gelu
            | Op::Dropout { .. } | Op::Embedding { .. }
            | Op::OllamaGenerate { .. } | Op::OllamaChat { .. } => 1,
            Op::Add | Op::Sub | Op::Mul | Op::Div | Op::MatMul
            | Op::Concat { .. } | Op::Residual => 2,
            Op::Attention { .. } => 3, // Q, K, V
            Op::Cond => 3, // predicate, branch_a, branch_b
            Op::Scan { .. } | Op::SubGraph { .. } => 2, // init + body/graph
        }
    }

    /// Number of output ports this operation produces.
    pub fn n_outputs(&self) -> usize {
        match self {
            Op::Output { .. } => 0,
            _ => 1,
        }
    }

    /// Whether this operation is deterministic.
    pub fn is_deterministic(&self) -> bool {
        !matches!(
            self,
            Op::Dropout { .. } | Op::OllamaGenerate { .. } | Op::OllamaChat { .. }
        )
    }

    /// Whether this is a quantum operation. Always false after radical purge.
    pub fn is_quantum(&self) -> bool {
        false
    }

    /// Whether this is an LLM inference operation.
    pub fn is_llm(&self) -> bool {
        matches!(
            self,
            Op::OllamaGenerate { .. } | Op::OllamaChat { .. }
        )
    }
}

impl fmt::Display for Op {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Op::Input { name } => write!(f, "input({name})"),
            Op::Output { name } => write!(f, "output({name})"),
            Op::Constant => write!(f, "const"),
            Op::Add => write!(f, "add"),
            Op::Sub => write!(f, "sub"),
            Op::Mul => write!(f, "mul"),
            Op::Div => write!(f, "div"),
            Op::Neg => write!(f, "neg"),
            Op::Exp => write!(f, "exp"),
            Op::Log => write!(f, "log"),
            Op::MatMul => write!(f, "matmul"),
            Op::Transpose => write!(f, "transpose"),
            Op::Reshape { target_shape } => write!(f, "reshape({target_shape:?})"),
            Op::Slice { start, end } => write!(f, "slice({start:?}..{end:?})"),
            Op::Concat { axis } => write!(f, "concat(axis={axis})"),
            Op::ReduceSum { axis } => write!(f, "reduce_sum({axis:?})"),
            Op::ReduceMean { axis } => write!(f, "reduce_mean({axis:?})"),
            Op::ReduceMax { axis } => write!(f, "reduce_max({axis:?})"),
            Op::Relu => write!(f, "relu"),
            Op::Sigmoid => write!(f, "sigmoid"),
            Op::Tanh => write!(f, "tanh"),
            Op::Softmax { axis } => write!(f, "softmax(axis={axis})"),
            Op::LayerNorm { eps } => write!(f, "layer_norm(eps={eps})"),
            Op::Attention { n_heads, d_model } => write!(f, "attention(heads={n_heads}, d={d_model})"),
            Op::Embedding { vocab_size, d_model } => write!(f, "embedding(vocab={vocab_size}, d={d_model})"),
            Op::Residual => write!(f, "residual"),
            Op::Gelu => write!(f, "gelu"),
            Op::Dropout { rate } => write!(f, "dropout(rate={rate})"),
            Op::OllamaGenerate { model } => write!(f, "ollama_generate({model})"),
            Op::OllamaChat { model } => write!(f, "ollama_chat({model})"),
            Op::ArgMax => write!(f, "argmax"),
            Op::Cond => write!(f, "cond"),
            Op::Scan { n_iterations } => write!(f, "scan(n={n_iterations})"),
            Op::SubGraph { graph_id } => write!(f, "subgraph({graph_id})"),
        }
    }
}
