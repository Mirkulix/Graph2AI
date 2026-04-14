//! Binary Graph Format — compact binary serialization for QLANG graphs.
//!
//! This replaces JSON for machine-to-machine communication. Graphs are
//! serialized to a compact binary representation that is much smaller
//! and faster to parse than JSON.
//!
//! Wire format:
//! ```text
//! [4 bytes]  Magic: "QLBG" (0x51, 0x4C, 0x42, 0x47)
//! [2 bytes]  Version: u16 LE
//! [var]      Graph ID (length-prefixed string)
//! [var]      Graph version (length-prefixed string)
//! [4 bytes]  Node count: u32 LE
//! [var]      Nodes (each: id + op + input_types + output_types)
//! [4 bytes]  Edge count: u32 LE
//! [var]      Edges (each: from_node + from_port + to_node + to_port + tensor_type)
//! [32 bytes] SHA-256 content hash of everything preceding
//! ```

use std::collections::HashMap;

use crate::crypto::sha256;
use crate::graph::{Edge, Graph, Node};
use crate::ops::{Manifold, Op};
use crate::tensor::{Dim, Dtype, Shape, TensorType};

/// Magic bytes identifying a QLANG binary graph: "QLBG"
pub const BINARY_MAGIC: [u8; 4] = [0x51, 0x4C, 0x42, 0x47];

/// Current binary format version.
pub const BINARY_VERSION: u16 = 1;

/// Errors that can occur during binary deserialization.
#[derive(Debug, thiserror::Error)]
pub enum BinaryError {
    #[error("data too short")]
    TooShort,
    #[error("invalid magic bytes (expected QLBG)")]
    InvalidMagic,
    #[error("unsupported version {0}")]
    UnsupportedVersion(u16),
    #[error("unexpected end of data at offset {0}")]
    UnexpectedEof(usize),
    #[error("invalid op tag {0}")]
    InvalidOpTag(u8),
    #[error("invalid dtype tag {0}")]
    InvalidDtypeTag(u8),
    #[error("invalid UTF-8 string")]
    InvalidUtf8,
    #[error("hash mismatch: data may be corrupted")]
    HashMismatch,
}

// -----------------------------------------------------------------------
// Op tag assignments
// -----------------------------------------------------------------------

const OP_INPUT: u8 = 0;
const OP_OUTPUT: u8 = 1;
const OP_CONSTANT: u8 = 2;
const OP_ADD: u8 = 3;
const OP_SUB: u8 = 4;
const OP_MUL: u8 = 5;
const OP_DIV: u8 = 6;
const OP_NEG: u8 = 7;
const OP_MATMUL: u8 = 8;
const OP_TRANSPOSE: u8 = 9;
const OP_RESHAPE: u8 = 10;
const OP_SLICE: u8 = 11;
const OP_CONCAT: u8 = 12;
const OP_REDUCE_SUM: u8 = 13;
const OP_REDUCE_MEAN: u8 = 14;
const OP_REDUCE_MAX: u8 = 15;
const OP_RELU: u8 = 16;
const OP_SIGMOID: u8 = 17;
const OP_TANH: u8 = 18;
const OP_SOFTMAX: u8 = 19;
const OP_SUPERPOSE: u8 = 20;
const OP_EVOLVE: u8 = 21;
const OP_MEASURE: u8 = 22;
const OP_ENTANGLE: u8 = 23;
const OP_COLLAPSE: u8 = 24;
const OP_ENTROPY: u8 = 25;
const OP_TO_TERNARY: u8 = 26;
const OP_TO_LOW_RANK: u8 = 27;
const OP_TO_SPARSE: u8 = 28;
const OP_FISHER_METRIC: u8 = 29;
const OP_PROJECT: u8 = 30;
const OP_LAYER_NORM: u8 = 31;
const OP_ATTENTION: u8 = 32;
const OP_EMBEDDING: u8 = 33;
const OP_RESIDUAL: u8 = 34;
const OP_GELU: u8 = 35;
const OP_DROPOUT: u8 = 36;
const OP_OLLAMA_GENERATE: u8 = 37;
const OP_OLLAMA_CHAT: u8 = 38;
const OP_COND: u8 = 39;
const OP_SCAN: u8 = 40;
const OP_SUB_GRAPH: u8 = 41;
const OP_EXP: u8 = 42;
const OP_LOG: u8 = 43;
const OP_CLASS_MEAN: u8 = 44;
const OP_TERNARIZE: u8 = 45;
const OP_TERNARY_MATVEC: u8 = 46;
const OP_ARGMAX: u8 = 47;
const OP_ENSEMBLE_VOTE: u8 = 48;

// Dtype tags (reuse the same encoding from tensor.rs wire format)
const DTYPE_F16: u8 = 0;
const DTYPE_F32: u8 = 1;
const DTYPE_F64: u8 = 2;
const DTYPE_I8: u8 = 3;
const DTYPE_I16: u8 = 4;
const DTYPE_I32: u8 = 5;
const DTYPE_I64: u8 = 6;
const DTYPE_BOOL: u8 = 7;
const DTYPE_TERNARY: u8 = 8;
const DTYPE_UTF8: u8 = 9;

// Manifold tags
const MANIFOLD_TERNARY: u8 = 0;
const MANIFOLD_LOW_RANK: u8 = 1;
const MANIFOLD_SPARSE: u8 = 2;
const MANIFOLD_CUSTOM: u8 = 3;

// -----------------------------------------------------------------------
// Serialization
// -----------------------------------------------------------------------

/// Serialize a graph to compact binary format.
pub fn to_binary(graph: &Graph) -> Vec<u8> {
    let mut buf = Vec::with_capacity(1024);

    // Header
    buf.extend_from_slice(&BINARY_MAGIC);
    buf.extend_from_slice(&BINARY_VERSION.to_le_bytes());

    // Graph ID and version
    write_string(&mut buf, &graph.id);
    write_string(&mut buf, &graph.version);

    // Node count
    buf.extend_from_slice(&(graph.nodes.len() as u32).to_le_bytes());

    // Nodes
    for node in &graph.nodes {
        buf.extend_from_slice(&node.id.to_le_bytes());
        write_op(&mut buf, &node.op);
        // Input types
        buf.push(node.input_types.len() as u8);
        for tt in &node.input_types {
            write_tensor_type(&mut buf, tt);
        }
        // Output types
        buf.push(node.output_types.len() as u8);
        for tt in &node.output_types {
            write_tensor_type(&mut buf, tt);
        }
    }

    // Edge count
    buf.extend_from_slice(&(graph.edges.len() as u32).to_le_bytes());

    // Edges
    for edge in &graph.edges {
        buf.extend_from_slice(&edge.id.to_le_bytes());
        buf.extend_from_slice(&edge.from_node.to_le_bytes());
        buf.push(edge.from_port);
        buf.extend_from_slice(&edge.to_node.to_le_bytes());
        buf.push(edge.to_port);
        write_tensor_type(&mut buf, &edge.tensor_type);
    }

    // Append content hash at the end (integrity check)
    let hash = sha256(&buf);
    buf.extend_from_slice(&hash);

    buf
}

/// Deserialize a graph from binary format.
pub fn from_binary(data: &[u8]) -> Result<Graph, BinaryError> {
    if data.len() < 6 {
        return Err(BinaryError::TooShort);
    }
    if data[0..4] != BINARY_MAGIC {
        return Err(BinaryError::InvalidMagic);
    }

    let version = u16::from_le_bytes([data[4], data[5]]);
    if version != BINARY_VERSION {
        return Err(BinaryError::UnsupportedVersion(version));
    }

    // Verify trailing hash
    if data.len() < 32 {
        return Err(BinaryError::TooShort);
    }
    let payload = &data[..data.len() - 32];
    let stored_hash = &data[data.len() - 32..];
    let computed_hash = sha256(payload);
    if stored_hash != computed_hash {
        return Err(BinaryError::HashMismatch);
    }

    let mut pos = 6; // skip magic + version

    let id = read_string(data, &mut pos)?;
    let version_str = read_string(data, &mut pos)?;
    let node_count = read_u32(data, &mut pos)? as usize;

    let mut nodes = Vec::with_capacity(node_count);
    for _ in 0..node_count {
        let node_id = read_u32(data, &mut pos)?;
        let op = read_op(data, &mut pos)?;
        let n_inputs = read_u8(data, &mut pos)? as usize;
        let mut input_types = Vec::with_capacity(n_inputs);
        for _ in 0..n_inputs {
            input_types.push(read_tensor_type(data, &mut pos)?);
        }
        let n_outputs = read_u8(data, &mut pos)? as usize;
        let mut output_types = Vec::with_capacity(n_outputs);
        for _ in 0..n_outputs {
            output_types.push(read_tensor_type(data, &mut pos)?);
        }
        nodes.push(Node {
            id: node_id,
            op,
            input_types,
            output_types,
            constraints: vec![],
            metadata: HashMap::new(),
        });
    }

    let edge_count = read_u32(data, &mut pos)? as usize;
    let mut edges = Vec::with_capacity(edge_count);
    for _ in 0..edge_count {
        let edge_id = read_u32(data, &mut pos)?;
        let from_node = read_u32(data, &mut pos)?;
        let from_port = read_u8(data, &mut pos)?;
        let to_node = read_u32(data, &mut pos)?;
        let to_port = read_u8(data, &mut pos)?;
        let tensor_type = read_tensor_type(data, &mut pos)?;
        edges.push(Edge {
            id: edge_id,
            from_node,
            from_port,
            to_node,
            to_port,
            tensor_type,
        });
    }

    Ok(Graph {
        id,
        version: version_str,
        nodes,
        edges,
        constraints: vec![],
        metadata: HashMap::new(),
    })
}

/// Compute the content hash of a graph's binary representation.
///
/// This hashes the binary payload (everything except the trailing hash).
pub fn content_hash(graph: &Graph) -> [u8; 32] {
    let bin = to_binary(graph);
    // Hash everything except the last 32 bytes (which IS the hash)
    sha256(&bin[..bin.len() - 32])
}

// -----------------------------------------------------------------------
// Write helpers
// -----------------------------------------------------------------------

fn write_string(buf: &mut Vec<u8>, s: &str) {
    let bytes = s.as_bytes();
    buf.extend_from_slice(&(bytes.len() as u32).to_le_bytes());
    buf.extend_from_slice(bytes);
}

fn write_op(buf: &mut Vec<u8>, op: &Op) {
    match op {
        Op::Input { name } => {
            buf.push(OP_INPUT);
            write_string(buf, name);
        }
        Op::Output { name } => {
            buf.push(OP_OUTPUT);
            write_string(buf, name);
        }
        Op::Constant => buf.push(OP_CONSTANT),
        Op::Add => buf.push(OP_ADD),
        Op::Sub => buf.push(OP_SUB),
        Op::Mul => buf.push(OP_MUL),
        Op::Div => buf.push(OP_DIV),
        Op::Neg => buf.push(OP_NEG),
        Op::MatMul => buf.push(OP_MATMUL),
        Op::Transpose => buf.push(OP_TRANSPOSE),
        Op::Reshape { target_shape } => {
            buf.push(OP_RESHAPE);
            write_usize_vec(buf, target_shape);
        }
        Op::Slice { start, end } => {
            buf.push(OP_SLICE);
            write_usize_vec(buf, start);
            write_usize_vec(buf, end);
        }
        Op::Concat { axis } => {
            buf.push(OP_CONCAT);
            buf.extend_from_slice(&(*axis as u32).to_le_bytes());
        }
        Op::ReduceSum { axis } => {
            buf.push(OP_REDUCE_SUM);
            write_optional_usize(buf, axis);
        }
        Op::ReduceMean { axis } => {
            buf.push(OP_REDUCE_MEAN);
            write_optional_usize(buf, axis);
        }
        Op::ReduceMax { axis } => {
            buf.push(OP_REDUCE_MAX);
            write_optional_usize(buf, axis);
        }
        Op::Relu => buf.push(OP_RELU),
        Op::Sigmoid => buf.push(OP_SIGMOID),
        Op::Tanh => buf.push(OP_TANH),
        Op::Softmax { axis } => {
            buf.push(OP_SOFTMAX);
            buf.extend_from_slice(&(*axis as u32).to_le_bytes());
        }
        Op::Superpose => buf.push(OP_SUPERPOSE),
        Op::Evolve { gamma, dt } => {
            buf.push(OP_EVOLVE);
            buf.extend_from_slice(&gamma.to_le_bytes());
            buf.extend_from_slice(&dt.to_le_bytes());
        }
        Op::Measure => buf.push(OP_MEASURE),
        Op::Entangle => buf.push(OP_ENTANGLE),
        Op::Collapse => buf.push(OP_COLLAPSE),
        Op::Entropy => buf.push(OP_ENTROPY),
        Op::ToTernary => buf.push(OP_TO_TERNARY),
        Op::ToLowRank { rank } => {
            buf.push(OP_TO_LOW_RANK);
            buf.extend_from_slice(&(*rank as u32).to_le_bytes());
        }
        Op::ToSparse { sparsity } => {
            buf.push(OP_TO_SPARSE);
            buf.extend_from_slice(&sparsity.to_le_bytes());
        }
        Op::FisherMetric => buf.push(OP_FISHER_METRIC),
        Op::Project { manifold } => {
            buf.push(OP_PROJECT);
            write_manifold(buf, manifold);
        }
        Op::LayerNorm { eps } => {
            buf.push(OP_LAYER_NORM);
            buf.extend_from_slice(&eps.to_le_bytes());
        }
        Op::Attention { n_heads, d_model } => {
            buf.push(OP_ATTENTION);
            buf.extend_from_slice(&(*n_heads as u32).to_le_bytes());
            buf.extend_from_slice(&(*d_model as u32).to_le_bytes());
        }
        Op::Embedding { vocab_size, d_model } => {
            buf.push(OP_EMBEDDING);
            buf.extend_from_slice(&(*vocab_size as u32).to_le_bytes());
            buf.extend_from_slice(&(*d_model as u32).to_le_bytes());
        }
        Op::Residual => buf.push(OP_RESIDUAL),
        Op::Gelu => buf.push(OP_GELU),
        Op::Dropout { rate } => {
            buf.push(OP_DROPOUT);
            buf.extend_from_slice(&rate.to_le_bytes());
        }
        Op::OllamaGenerate { model } => {
            buf.push(OP_OLLAMA_GENERATE);
            write_string(buf, model);
        }
        Op::OllamaChat { model } => {
            buf.push(OP_OLLAMA_CHAT);
            write_string(buf, model);
        }
        Op::Cond => buf.push(OP_COND),
        Op::Scan { n_iterations } => {
            buf.push(OP_SCAN);
            buf.extend_from_slice(&(*n_iterations as u32).to_le_bytes());
        }
        Op::SubGraph { graph_id } => {
            buf.push(OP_SUB_GRAPH);
            write_string(buf, graph_id);
        }
        Op::Exp => buf.push(OP_EXP),
        Op::Log => buf.push(OP_LOG),
        Op::ClassMean { n_classes } => {
            buf.push(OP_CLASS_MEAN);
            buf.extend_from_slice(&(*n_classes as u32).to_le_bytes());
        }
        Op::Ternarize { threshold_ratio } => {
            buf.push(OP_TERNARIZE);
            buf.extend_from_slice(&threshold_ratio.to_le_bytes());
        }
        Op::TernaryMatVec => buf.push(OP_TERNARY_MATVEC),
        Op::ArgMax => buf.push(OP_ARGMAX),
        Op::EnsembleVote => buf.push(OP_ENSEMBLE_VOTE),
    }
}

fn write_manifold(buf: &mut Vec<u8>, m: &Manifold) {
    match m {
        Manifold::Ternary => buf.push(MANIFOLD_TERNARY),
        Manifold::LowRank { max_rank } => {
            buf.push(MANIFOLD_LOW_RANK);
            buf.extend_from_slice(&(*max_rank as u32).to_le_bytes());
        }
        Manifold::Sparse { max_nonzero } => {
            buf.push(MANIFOLD_SPARSE);
            buf.extend_from_slice(&(*max_nonzero as u32).to_le_bytes());
        }
        Manifold::Custom { name } => {
            buf.push(MANIFOLD_CUSTOM);
            write_string(buf, name);
        }
    }
}

fn write_tensor_type(buf: &mut Vec<u8>, tt: &TensorType) {
    write_dtype(buf, &tt.dtype);
    write_shape(buf, &tt.shape);
}

fn write_dtype(buf: &mut Vec<u8>, dtype: &Dtype) {
    let tag = match dtype {
        Dtype::F16 => DTYPE_F16,
        Dtype::F32 => DTYPE_F32,
        Dtype::F64 => DTYPE_F64,
        Dtype::I8 => DTYPE_I8,
        Dtype::I16 => DTYPE_I16,
        Dtype::I32 => DTYPE_I32,
        Dtype::I64 => DTYPE_I64,
        Dtype::Bool => DTYPE_BOOL,
        Dtype::Ternary => DTYPE_TERNARY,
        Dtype::Utf8 => DTYPE_UTF8,
    };
    buf.push(tag);
}

fn write_shape(buf: &mut Vec<u8>, shape: &Shape) {
    buf.push(shape.0.len() as u8);
    for dim in &shape.0 {
        match dim {
            Dim::Fixed(n) => {
                buf.push(0); // tag: fixed
                buf.extend_from_slice(&(*n as u64).to_le_bytes());
            }
            Dim::Dynamic => {
                buf.push(1); // tag: dynamic
            }
        }
    }
}

fn write_usize_vec(buf: &mut Vec<u8>, v: &[usize]) {
    buf.extend_from_slice(&(v.len() as u32).to_le_bytes());
    for &x in v {
        buf.extend_from_slice(&(x as u64).to_le_bytes());
    }
}

fn write_optional_usize(buf: &mut Vec<u8>, val: &Option<usize>) {
    match val {
        Some(v) => {
            buf.push(1);
            buf.extend_from_slice(&(*v as u64).to_le_bytes());
        }
        None => buf.push(0),
    }
}

// -----------------------------------------------------------------------
// Read helpers
// -----------------------------------------------------------------------

fn read_u8(data: &[u8], pos: &mut usize) -> Result<u8, BinaryError> {
    if *pos >= data.len() {
        return Err(BinaryError::UnexpectedEof(*pos));
    }
    let v = data[*pos];
    *pos += 1;
    Ok(v)
}

fn read_u32(data: &[u8], pos: &mut usize) -> Result<u32, BinaryError> {
    if *pos + 4 > data.len() {
        return Err(BinaryError::UnexpectedEof(*pos));
    }
    let v = u32::from_le_bytes([data[*pos], data[*pos + 1], data[*pos + 2], data[*pos + 3]]);
    *pos += 4;
    Ok(v)
}

fn read_u64(data: &[u8], pos: &mut usize) -> Result<u64, BinaryError> {
    if *pos + 8 > data.len() {
        return Err(BinaryError::UnexpectedEof(*pos));
    }
    let v = u64::from_le_bytes([
        data[*pos],
        data[*pos + 1],
        data[*pos + 2],
        data[*pos + 3],
        data[*pos + 4],
        data[*pos + 5],
        data[*pos + 6],
        data[*pos + 7],
    ]);
    *pos += 8;
    Ok(v)
}

fn read_f32(data: &[u8], pos: &mut usize) -> Result<f32, BinaryError> {
    let bits = read_u32(data, pos)?;
    Ok(f32::from_le_bytes(bits.to_le_bytes()))
}

fn read_f64(data: &[u8], pos: &mut usize) -> Result<f64, BinaryError> {
    let bits = read_u64(data, pos)?;
    Ok(f64::from_le_bytes(bits.to_le_bytes()))
}

fn read_string(data: &[u8], pos: &mut usize) -> Result<String, BinaryError> {
    let len = read_u32(data, pos)? as usize;
    if *pos + len > data.len() {
        return Err(BinaryError::UnexpectedEof(*pos));
    }
    let s = std::str::from_utf8(&data[*pos..*pos + len]).map_err(|_| BinaryError::InvalidUtf8)?;
    *pos += len;
    Ok(s.to_string())
}

fn read_op(data: &[u8], pos: &mut usize) -> Result<Op, BinaryError> {
    let tag = read_u8(data, pos)?;
    match tag {
        OP_INPUT => {
            let name = read_string(data, pos)?;
            Ok(Op::Input { name })
        }
        OP_OUTPUT => {
            let name = read_string(data, pos)?;
            Ok(Op::Output { name })
        }
        OP_CONSTANT => Ok(Op::Constant),
        OP_ADD => Ok(Op::Add),
        OP_SUB => Ok(Op::Sub),
        OP_MUL => Ok(Op::Mul),
        OP_DIV => Ok(Op::Div),
        OP_NEG => Ok(Op::Neg),
        OP_MATMUL => Ok(Op::MatMul),
        OP_TRANSPOSE => Ok(Op::Transpose),
        OP_RESHAPE => {
            let target_shape = read_usize_vec(data, pos)?;
            Ok(Op::Reshape { target_shape })
        }
        OP_SLICE => {
            let start = read_usize_vec(data, pos)?;
            let end = read_usize_vec(data, pos)?;
            Ok(Op::Slice { start, end })
        }
        OP_CONCAT => {
            let axis = read_u32(data, pos)? as usize;
            Ok(Op::Concat { axis })
        }
        OP_REDUCE_SUM => {
            let axis = read_optional_usize(data, pos)?;
            Ok(Op::ReduceSum { axis })
        }
        OP_REDUCE_MEAN => {
            let axis = read_optional_usize(data, pos)?;
            Ok(Op::ReduceMean { axis })
        }
        OP_REDUCE_MAX => {
            let axis = read_optional_usize(data, pos)?;
            Ok(Op::ReduceMax { axis })
        }
        OP_RELU => Ok(Op::Relu),
        OP_SIGMOID => Ok(Op::Sigmoid),
        OP_TANH => Ok(Op::Tanh),
        OP_SOFTMAX => {
            let axis = read_u32(data, pos)? as usize;
            Ok(Op::Softmax { axis })
        }
        OP_SUPERPOSE => Ok(Op::Superpose),
        OP_EVOLVE => {
            let gamma = read_f64(data, pos)?;
            let dt = read_f64(data, pos)?;
            Ok(Op::Evolve { gamma, dt })
        }
        OP_MEASURE => Ok(Op::Measure),
        OP_ENTANGLE => Ok(Op::Entangle),
        OP_COLLAPSE => Ok(Op::Collapse),
        OP_ENTROPY => Ok(Op::Entropy),
        OP_TO_TERNARY => Ok(Op::ToTernary),
        OP_TO_LOW_RANK => {
            let rank = read_u32(data, pos)? as usize;
            Ok(Op::ToLowRank { rank })
        }
        OP_TO_SPARSE => {
            let sparsity = read_f64(data, pos)?;
            Ok(Op::ToSparse { sparsity })
        }
        OP_FISHER_METRIC => Ok(Op::FisherMetric),
        OP_PROJECT => {
            let manifold = read_manifold(data, pos)?;
            Ok(Op::Project { manifold })
        }
        OP_LAYER_NORM => {
            let eps = read_f64(data, pos)?;
            Ok(Op::LayerNorm { eps })
        }
        OP_ATTENTION => {
            let n_heads = read_u32(data, pos)? as usize;
            let d_model = read_u32(data, pos)? as usize;
            Ok(Op::Attention { n_heads, d_model })
        }
        OP_EMBEDDING => {
            let vocab_size = read_u32(data, pos)? as usize;
            let d_model = read_u32(data, pos)? as usize;
            Ok(Op::Embedding { vocab_size, d_model })
        }
        OP_RESIDUAL => Ok(Op::Residual),
        OP_GELU => Ok(Op::Gelu),
        OP_DROPOUT => {
            let rate = read_f64(data, pos)?;
            Ok(Op::Dropout { rate })
        }
        OP_OLLAMA_GENERATE => {
            let model = read_string(data, pos)?;
            Ok(Op::OllamaGenerate { model })
        }
        OP_OLLAMA_CHAT => {
            let model = read_string(data, pos)?;
            Ok(Op::OllamaChat { model })
        }
        OP_COND => Ok(Op::Cond),
        OP_SCAN => {
            let n_iterations = read_u32(data, pos)? as usize;
            Ok(Op::Scan { n_iterations })
        }
        OP_SUB_GRAPH => {
            let graph_id = read_string(data, pos)?;
            Ok(Op::SubGraph { graph_id })
        }
        OP_EXP => Ok(Op::Exp),
        OP_LOG => Ok(Op::Log),
        OP_CLASS_MEAN => {
            let n_classes = read_u32(data, pos)? as usize;
            Ok(Op::ClassMean { n_classes })
        }
        OP_TERNARIZE => {
            let threshold_ratio = read_f32(data, pos)?;
            Ok(Op::Ternarize { threshold_ratio })
        }
        OP_TERNARY_MATVEC => Ok(Op::TernaryMatVec),
        OP_ARGMAX => Ok(Op::ArgMax),
        OP_ENSEMBLE_VOTE => Ok(Op::EnsembleVote),
        _ => Err(BinaryError::InvalidOpTag(tag)),
    }
}

fn read_manifold(data: &[u8], pos: &mut usize) -> Result<Manifold, BinaryError> {
    let tag = read_u8(data, pos)?;
    match tag {
        MANIFOLD_TERNARY => Ok(Manifold::Ternary),
        MANIFOLD_LOW_RANK => {
            let max_rank = read_u32(data, pos)? as usize;
            Ok(Manifold::LowRank { max_rank })
        }
        MANIFOLD_SPARSE => {
            let max_nonzero = read_u32(data, pos)? as usize;
            Ok(Manifold::Sparse { max_nonzero })
        }
        MANIFOLD_CUSTOM => {
            let name = read_string(data, pos)?;
            Ok(Manifold::Custom { name })
        }
        _ => Err(BinaryError::InvalidOpTag(tag)),
    }
}

fn read_tensor_type(data: &[u8], pos: &mut usize) -> Result<TensorType, BinaryError> {
    let dtype = read_dtype(data, pos)?;
    let shape = read_shape(data, pos)?;
    Ok(TensorType { dtype, shape })
}

fn read_dtype(data: &[u8], pos: &mut usize) -> Result<Dtype, BinaryError> {
    let tag = read_u8(data, pos)?;
    match tag {
        DTYPE_F16 => Ok(Dtype::F16),
        DTYPE_F32 => Ok(Dtype::F32),
        DTYPE_F64 => Ok(Dtype::F64),
        DTYPE_I8 => Ok(Dtype::I8),
        DTYPE_I16 => Ok(Dtype::I16),
        DTYPE_I32 => Ok(Dtype::I32),
        DTYPE_I64 => Ok(Dtype::I64),
        DTYPE_BOOL => Ok(Dtype::Bool),
        DTYPE_TERNARY => Ok(Dtype::Ternary),
        DTYPE_UTF8 => Ok(Dtype::Utf8),
        _ => Err(BinaryError::InvalidDtypeTag(tag)),
    }
}

fn read_shape(data: &[u8], pos: &mut usize) -> Result<Shape, BinaryError> {
    let ndims = read_u8(data, pos)? as usize;
    let mut dims = Vec::with_capacity(ndims);
    for _ in 0..ndims {
        let tag = read_u8(data, pos)?;
        match tag {
            0 => {
                let n = read_u64(data, pos)? as usize;
                dims.push(Dim::Fixed(n));
            }
            1 => {
                dims.push(Dim::Dynamic);
            }
            _ => return Err(BinaryError::UnexpectedEof(*pos)),
        }
    }
    Ok(Shape(dims))
}

fn read_usize_vec(data: &[u8], pos: &mut usize) -> Result<Vec<usize>, BinaryError> {
    let len = read_u32(data, pos)? as usize;
    let mut v = Vec::with_capacity(len);
    for _ in 0..len {
        v.push(read_u64(data, pos)? as usize);
    }
    Ok(v)
}

fn read_optional_usize(data: &[u8], pos: &mut usize) -> Result<Option<usize>, BinaryError> {
    let tag = read_u8(data, pos)?;
    match tag {
        0 => Ok(None),
        1 => {
            let v = read_u64(data, pos)? as usize;
            Ok(Some(v))
        }
        _ => Err(BinaryError::UnexpectedEof(*pos)),
    }
}

// -----------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::Graph;
    use crate::ops::{Manifold, Op};
    use crate::tensor::{Dtype, Shape, TensorType};

    fn f32_vec(n: usize) -> TensorType {
        TensorType::new(Dtype::F32, Shape::vector(n))
    }

    fn f32_mat(m: usize, n: usize) -> TensorType {
        TensorType::new(Dtype::F32, Shape::matrix(m, n))
    }

    #[test]
    fn binary_roundtrip_simple() {
        let mut g = Graph::new("roundtrip-test");
        let inp = g.add_node(Op::Input { name: "x".into() }, vec![], vec![f32_vec(4)]);
        let relu = g.add_node(Op::Relu, vec![f32_vec(4)], vec![f32_vec(4)]);
        let out = g.add_node(
            Op::Output { name: "y".into() },
            vec![f32_vec(4)],
            vec![],
        );
        g.add_edge(inp, 0, relu, 0, f32_vec(4));
        g.add_edge(relu, 0, out, 0, f32_vec(4));

        let bin = to_binary(&g);
        let g2 = from_binary(&bin).unwrap();

        assert_eq!(g.id, g2.id);
        assert_eq!(g.nodes.len(), g2.nodes.len());
        assert_eq!(g.edges.len(), g2.edges.len());

        // Check nodes
        for (n1, n2) in g.nodes.iter().zip(g2.nodes.iter()) {
            assert_eq!(n1.id, n2.id);
            assert_eq!(n1.op, n2.op);
            assert_eq!(n1.input_types, n2.input_types);
            assert_eq!(n1.output_types, n2.output_types);
        }

        // Check edges
        for (e1, e2) in g.edges.iter().zip(g2.edges.iter()) {
            assert_eq!(e1.from_node, e2.from_node);
            assert_eq!(e1.from_port, e2.from_port);
            assert_eq!(e1.to_node, e2.to_node);
            assert_eq!(e1.to_port, e2.to_port);
            assert_eq!(e1.tensor_type, e2.tensor_type);
        }
    }

    #[test]
    fn binary_roundtrip_all_ops() {
        // Test every op variant to ensure encode/decode works
        let ops: Vec<Op> = vec![
            Op::Input { name: "in".into() },
            Op::Output { name: "out".into() },
            Op::Constant,
            Op::Add,
            Op::Sub,
            Op::Mul,
            Op::Div,
            Op::Neg,
            Op::MatMul,
            Op::Transpose,
            Op::Reshape { target_shape: vec![2, 3, 4] },
            Op::Slice { start: vec![0, 1], end: vec![2, 3] },
            Op::Concat { axis: 1 },
            Op::ReduceSum { axis: Some(0) },
            Op::ReduceSum { axis: None },
            Op::ReduceMean { axis: Some(1) },
            Op::ReduceMax { axis: None },
            Op::Relu,
            Op::Sigmoid,
            Op::Tanh,
            Op::Softmax { axis: 1 },
            Op::Superpose,
            Op::Evolve { gamma: 0.01, dt: 0.001 },
            Op::Measure,
            Op::Entangle,
            Op::Collapse,
            Op::Entropy,
            Op::ToTernary,
            Op::ToLowRank { rank: 16 },
            Op::ToSparse { sparsity: 0.9 },
            Op::FisherMetric,
            Op::Project { manifold: Manifold::Ternary },
            Op::Project { manifold: Manifold::LowRank { max_rank: 8 } },
            Op::Project { manifold: Manifold::Sparse { max_nonzero: 100 } },
            Op::Project { manifold: Manifold::Custom { name: "my_manifold".into() } },
            Op::LayerNorm { eps: 1e-5 },
            Op::Attention { n_heads: 8, d_model: 512 },
            Op::Embedding { vocab_size: 50000, d_model: 768 },
            Op::Residual,
            Op::Gelu,
            Op::Dropout { rate: 0.1 },
            Op::OllamaGenerate { model: "llama3".into() },
            Op::OllamaChat { model: "mistral".into() },
            Op::Cond,
            Op::Scan { n_iterations: 10 },
            Op::SubGraph { graph_id: "sub1".into() },
            Op::Exp,
            Op::Log,
        ];

        for (i, op) in ops.iter().enumerate() {
            let mut g = Graph::new(format!("op_test_{i}"));
            g.add_node(op.clone(), vec![], vec![f32_vec(1)]);

            let bin = to_binary(&g);
            let g2 = from_binary(&bin).unwrap();

            assert_eq!(g2.nodes[0].op, *op, "Op roundtrip failed for: {op}");
        }
    }

    #[test]
    fn binary_content_hash_determinism() {
        let mut g = Graph::new("hash-test");
        g.add_node(Op::Add, vec![f32_vec(4), f32_vec(4)], vec![f32_vec(4)]);

        let h1 = content_hash(&g);
        let h2 = content_hash(&g);
        assert_eq!(h1, h2);
    }

    #[test]
    fn binary_content_hash_changes() {
        let mut g1 = Graph::new("graph-a");
        g1.add_node(Op::Add, vec![f32_vec(4)], vec![f32_vec(4)]);

        let mut g2 = Graph::new("graph-b");
        g2.add_node(Op::Add, vec![f32_vec(4)], vec![f32_vec(4)]);

        assert_ne!(content_hash(&g1), content_hash(&g2));
    }

    #[test]
    fn binary_invalid_magic() {
        let mut data = to_binary(&Graph::new("test"));
        data[0] = 0xFF; // corrupt magic
        assert!(matches!(from_binary(&data), Err(BinaryError::InvalidMagic)));
    }

    #[test]
    fn binary_hash_mismatch() {
        let mut data = to_binary(&Graph::new("test"));
        // Corrupt a byte in the payload (not the trailing hash)
        if data.len() > 10 {
            data[8] ^= 0xFF;
        }
        assert!(matches!(
            from_binary(&data),
            Err(BinaryError::HashMismatch)
        ));
    }

    #[test]
    fn binary_too_short() {
        assert!(matches!(from_binary(&[0x51, 0x4C]), Err(BinaryError::TooShort)));
    }

    #[test]
    fn binary_vs_json_size() {
        // Build a moderately complex graph
        let mut g = Graph::new("size-comparison");
        let a = g.add_node(
            Op::Input { name: "a".into() },
            vec![],
            vec![f32_mat(128, 64)],
        );
        let b = g.add_node(
            Op::Input { name: "b".into() },
            vec![],
            vec![f32_mat(64, 32)],
        );
        let mm = g.add_node(
            Op::MatMul,
            vec![f32_mat(128, 64), f32_mat(64, 32)],
            vec![f32_mat(128, 32)],
        );
        let relu = g.add_node(
            Op::Relu,
            vec![f32_mat(128, 32)],
            vec![f32_mat(128, 32)],
        );
        let ln = g.add_node(
            Op::LayerNorm { eps: 1e-5 },
            vec![f32_mat(128, 32)],
            vec![f32_mat(128, 32)],
        );
        let out = g.add_node(
            Op::Output { name: "y".into() },
            vec![f32_mat(128, 32)],
            vec![],
        );
        g.add_edge(a, 0, mm, 0, f32_mat(128, 64));
        g.add_edge(b, 0, mm, 1, f32_mat(64, 32));
        g.add_edge(mm, 0, relu, 0, f32_mat(128, 32));
        g.add_edge(relu, 0, ln, 0, f32_mat(128, 32));
        g.add_edge(ln, 0, out, 0, f32_mat(128, 32));

        let json_bytes = serde_json::to_vec(&g).unwrap();
        let binary_bytes = to_binary(&g);

        assert!(
            binary_bytes.len() < json_bytes.len(),
            "Binary ({} bytes) should be smaller than JSON ({} bytes)",
            binary_bytes.len(),
            json_bytes.len()
        );
    }

    #[test]
    fn binary_roundtrip_complex_graph() {
        // Test with a more realistic graph: attention + residual + layer_norm
        let mut g = Graph::new("transformer-block");
        let x = g.add_node(
            Op::Input { name: "x".into() },
            vec![],
            vec![f32_mat(32, 512)],
        );
        let attn = g.add_node(
            Op::Attention { n_heads: 8, d_model: 512 },
            vec![f32_mat(32, 512), f32_mat(32, 512), f32_mat(32, 512)],
            vec![f32_mat(32, 512)],
        );
        let res = g.add_node(
            Op::Residual,
            vec![f32_mat(32, 512), f32_mat(32, 512)],
            vec![f32_mat(32, 512)],
        );
        let ln = g.add_node(
            Op::LayerNorm { eps: 1e-6 },
            vec![f32_mat(32, 512)],
            vec![f32_mat(32, 512)],
        );
        let out = g.add_node(
            Op::Output { name: "y".into() },
            vec![f32_mat(32, 512)],
            vec![],
        );

        g.add_edge(x, 0, attn, 0, f32_mat(32, 512));
        g.add_edge(x, 0, attn, 1, f32_mat(32, 512));
        g.add_edge(x, 0, attn, 2, f32_mat(32, 512));
        g.add_edge(x, 0, res, 0, f32_mat(32, 512));
        g.add_edge(attn, 0, res, 1, f32_mat(32, 512));
        g.add_edge(res, 0, ln, 0, f32_mat(32, 512));
        g.add_edge(ln, 0, out, 0, f32_mat(32, 512));

        let bin = to_binary(&g);
        let g2 = from_binary(&bin).unwrap();

        assert_eq!(g.id, g2.id);
        assert_eq!(g.nodes.len(), g2.nodes.len());
        assert_eq!(g.edges.len(), g2.edges.len());

        for (n1, n2) in g.nodes.iter().zip(g2.nodes.iter()) {
            assert_eq!(n1.op, n2.op);
        }
    }
}
