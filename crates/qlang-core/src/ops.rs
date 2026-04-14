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

    // === Quantum / IGQK Operations ===
    /// Create superposition of multiple states
    Superpose,
    /// Quantum gradient flow: dρ/dt = -i[H, ρ] - γ{G⁻¹∇L, ρ}
    Evolve {
        gamma: f64, // damping parameter
        dt: f64,    // time step
    },
    /// Quantum measurement: P(w|ρ) = Tr(ρ M_w)
    Measure,
    /// Create entangled state across tensors
    Entangle,
    /// Collapse quantum state to concrete value
    Collapse,
    /// Compute von Neumann entropy: S(ρ) = -Tr(ρ log ρ)
    Entropy,

    // === IGQK Compression ===
    /// Project to ternary weights {-1, 0, +1}
    ToTernary,
    /// Low-rank approximation
    ToLowRank { rank: usize },
    /// Sparsification
    ToSparse { sparsity: f64 },
    /// Compute Fisher information metric
    FisherMetric,
    /// Project onto submanifold
    Project { manifold: Manifold },

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
    /// Dropout (identity at inference, masks at training)
    Dropout { rate: f64 },

    // === LLM / Ollama Operations ===
    /// Generate text via local Ollama LLM
    OllamaGenerate { model: String },
    /// Chat completion via local Ollama LLM
    OllamaChat { model: String },

    // === Ternary Ensemble Training (native QLANG training ops) ===
    /// Compute per-class mean vectors from labeled data.
    /// Input: (images[n,d], labels[n]) → Output: class_means[k,d]
    ClassMean { n_classes: usize },
    /// Ternarize a continuous vector: values above threshold → +1, below → -1, else → 0.
    /// Input: continuous[d] → Output: ternary[d] ∈ {-1, 0, +1}
    Ternarize { threshold_ratio: f32 },
    /// Ternary matrix-vector product using only add/sub/skip.
    /// Input: (ternary_weights[out,in], x[batch,in]) → Output: y[batch,out]
    TernaryMatVec,
    /// ArgMax along last axis.
    /// Input: scores[batch, k] → Output: indices[batch]
    ArgMax,
    /// Ensemble voting: combine K classifier scores via weighted majority.
    /// Input: (scores[batch,k], weights[k]) → Output: predictions[batch]
    EnsembleVote,

    // === Control Flow ===
    /// Conditional: evaluates BOTH branches (quantum-style), selects based on predicate
    Cond,
    /// Bounded iteration
    Scan { n_iterations: usize },
    /// Execute a sub-graph
    SubGraph { graph_id: String },
}

/// Target manifold for projection operations.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Manifold {
    /// Ternary weights: {-1, 0, +1}
    Ternary,
    /// Low-rank: rank(W) ≤ r
    LowRank { max_rank: usize },
    /// Sparse: ||W||_0 ≤ s
    Sparse { max_nonzero: usize },
    /// Custom submanifold
    Custom { name: String },
}

impl Op {
    /// Number of input ports this operation expects.
    pub fn n_inputs(&self) -> usize {
        match self {
            Op::Input { .. } | Op::Constant => 0,
            Op::Output { .. } | Op::Neg | Op::Exp | Op::Log | Op::Relu | Op::Sigmoid | Op::Tanh
            | Op::Softmax { .. } | Op::Transpose | Op::Reshape { .. }
            | Op::Slice { .. } | Op::ToTernary | Op::ToLowRank { .. }
            | Op::ToSparse { .. } | Op::Entropy | Op::Collapse
            | Op::ReduceSum { .. } | Op::ReduceMean { .. } | Op::ReduceMax { .. }
            | Op::Project { .. } | Op::LayerNorm { .. } | Op::Gelu
            | Op::Dropout { .. } | Op::Embedding { .. }
            | Op::OllamaGenerate { .. } | Op::OllamaChat { .. }
            | Op::Ternarize { .. } | Op::ArgMax => 1,
            Op::Add | Op::Sub | Op::Mul | Op::Div | Op::MatMul
            | Op::Concat { .. } | Op::Entangle | Op::FisherMetric
            | Op::Residual | Op::TernaryMatVec | Op::EnsembleVote
            | Op::ClassMean { .. } => 2,
            Op::Attention { .. } => 3, // Q, K, V
            Op::Cond => 3, // predicate, branch_a, branch_b
            Op::Evolve { .. } => 3, // ρ, hamiltonian, gradient
            Op::Measure | Op::Superpose => 2, // state + operators/states
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
        match self {
            Op::Measure | Op::Collapse | Op::Superpose | Op::Dropout { .. }
            | Op::OllamaGenerate { .. } | Op::OllamaChat { .. } => false,
            _ => true,
        }
    }

    /// Whether this is a quantum/IGQK operation.
    pub fn is_quantum(&self) -> bool {
        matches!(
            self,
            Op::Superpose | Op::Evolve { .. } | Op::Measure | Op::Entangle
            | Op::Collapse | Op::Entropy | Op::ToTernary | Op::ToLowRank { .. }
            | Op::ToSparse { .. } | Op::FisherMetric | Op::Project { .. }
        )
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
            Op::Superpose => write!(f, "superpose"),
            Op::Evolve { gamma, dt } => write!(f, "evolve(γ={gamma}, dt={dt})"),
            Op::Measure => write!(f, "measure"),
            Op::Entangle => write!(f, "entangle"),
            Op::Collapse => write!(f, "collapse"),
            Op::Entropy => write!(f, "entropy"),
            Op::ToTernary => write!(f, "to_ternary"),
            Op::ToLowRank { rank } => write!(f, "to_lowrank(r={rank})"),
            Op::ToSparse { sparsity } => write!(f, "to_sparse(s={sparsity})"),
            Op::FisherMetric => write!(f, "fisher_metric"),
            Op::Project { manifold } => write!(f, "project({manifold:?})"),
            Op::LayerNorm { eps } => write!(f, "layer_norm(eps={eps})"),
            Op::Attention { n_heads, d_model } => write!(f, "attention(heads={n_heads}, d={d_model})"),
            Op::Embedding { vocab_size, d_model } => write!(f, "embedding(vocab={vocab_size}, d={d_model})"),
            Op::Residual => write!(f, "residual"),
            Op::Gelu => write!(f, "gelu"),
            Op::Dropout { rate } => write!(f, "dropout(rate={rate})"),
            Op::OllamaGenerate { model } => write!(f, "ollama_generate({model})"),
            Op::OllamaChat { model } => write!(f, "ollama_chat({model})"),
            Op::ClassMean { n_classes } => write!(f, "class_mean(k={n_classes})"),
            Op::Ternarize { threshold_ratio } => write!(f, "ternarize(t={threshold_ratio})"),
            Op::TernaryMatVec => write!(f, "ternary_matvec"),
            Op::ArgMax => write!(f, "argmax"),
            Op::EnsembleVote => write!(f, "ensemble_vote"),
            Op::Cond => write!(f, "cond"),
            Op::Scan { n_iterations } => write!(f, "scan(n={n_iterations})"),
            Op::SubGraph { graph_id } => write!(f, "subgraph({graph_id})"),
        }
    }
}
