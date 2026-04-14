//! ONNX-compatible graph import/export for QLANG.
//!
//! Provides conversion between QLANG's internal graph representation and an
//! ONNX-like intermediate format, enabling interoperability with PyTorch and
//! TensorFlow models.
//!
//! This module does NOT depend on ONNX protobuf libraries. Instead it defines
//! lightweight Rust structs that mirror the ONNX spec and offers a JSON
//! serialization path for easy exchange with Python tooling.

use std::collections::HashMap;
use std::io;
use std::path::Path;

use serde::{Deserialize, Serialize};

use qlang_core::graph::{Graph, NodeId};
use qlang_core::ops::Op;
use qlang_core::tensor::{Dim, Dtype, Shape, TensorType};

// ---------------------------------------------------------------------------
// ONNX-like data structures
// ---------------------------------------------------------------------------

/// A single attribute value on an ONNX node.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", content = "value")]
pub enum OnnxAttribute {
    Int(i64),
    Float(f64),
    String(String),
    Ints(Vec<i64>),
    Floats(Vec<f64>),
}

/// A node in an ONNX graph.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OnnxNode {
    pub name: String,
    pub op_type: String,
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
    pub attributes: HashMap<String, OnnxAttribute>,
}

/// A named tensor value-info (type descriptor) used for graph inputs/outputs.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OnnxValueInfo {
    pub name: String,
    pub elem_type: String,
    pub shape: Vec<OnnxDim>,
}

/// A dimension in an ONNX tensor shape.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum OnnxDim {
    Fixed(usize),
    Dynamic(String),
}

/// A named initializer (constant tensor) in an ONNX graph.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OnnxInitializer {
    pub name: String,
    pub elem_type: String,
    pub shape: Vec<usize>,
    /// Raw data encoded as base64 in JSON; left empty for structural conversion.
    pub data_base64: Option<String>,
    /// Extracted weight data as f32 values (from float_data or raw_data in TensorProto).
    pub weights: Option<Vec<f32>>,
}

/// Top-level ONNX-like graph.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OnnxGraph {
    pub name: String,
    pub nodes: Vec<OnnxNode>,
    pub inputs: Vec<OnnxValueInfo>,
    pub outputs: Vec<OnnxValueInfo>,
    pub initializers: Vec<OnnxInitializer>,
}

// ---------------------------------------------------------------------------
// Conversion errors
// ---------------------------------------------------------------------------

/// Errors produced when converting between ONNX and QLANG representations.
#[derive(Debug, thiserror::Error)]
pub enum ConversionError {
    #[error("unsupported ONNX op type: {0}")]
    UnsupportedOp(String),

    #[error("missing required attribute '{attr}' on node '{node}'")]
    MissingAttribute { node: String, attr: String },

    #[error("attribute type mismatch for '{attr}' on node '{node}'")]
    AttributeTypeMismatch { node: String, attr: String },

    #[error("unknown element type: {0}")]
    UnknownElemType(String),

    #[error("edge wiring error: {0}")]
    WiringError(String),

    #[error("I/O error: {0}")]
    Io(String),

    #[error("weight store error: {0}")]
    WeightStore(String),

    #[error("compression error: {0}")]
    Compression(String),

    #[error("model too large for in-memory processing: {0}")]
    ModelTooLarge(String),
}

impl From<io::Error> for ConversionError {
    fn from(e: io::Error) -> Self {
        ConversionError::Io(e.to_string())
    }
}

// ---------------------------------------------------------------------------
// Minimal Protobuf wire-format parser (no external dependencies)
// ---------------------------------------------------------------------------

/// Protobuf wire types.
#[derive(Debug, Clone, Copy, PartialEq)]
enum WireType {
    Varint,           // 0
    Fixed64,          // 1
    LengthDelimited,  // 2
    Fixed32,          // 5
}

/// A raw protobuf field.
#[derive(Debug, Clone)]
struct ProtoField {
    field_number: u32,
    _wire_type: WireType,
    data: ProtoData,
}

#[derive(Debug, Clone)]
enum ProtoData {
    Varint(u64),
    Fixed64(u64),
    Fixed32(u32),
    Bytes(Vec<u8>),
}

impl ProtoData {
    fn as_u64(&self) -> u64 {
        match self {
            ProtoData::Varint(v) => *v,
            ProtoData::Fixed64(v) => *v,
            ProtoData::Fixed32(v) => *v as u64,
            ProtoData::Bytes(_) => 0,
        }
    }

    fn as_i64(&self) -> i64 {
        self.as_u64() as i64
    }

    fn as_bytes(&self) -> &[u8] {
        match self {
            ProtoData::Bytes(b) => b,
            _ => &[],
        }
    }

    fn as_str(&self) -> &str {
        std::str::from_utf8(self.as_bytes()).unwrap_or("")
    }
}

/// Parse protobuf wire format from bytes.
fn parse_protobuf(data: &[u8]) -> Vec<ProtoField> {
    let mut fields = Vec::new();
    let mut pos = 0;

    while pos < data.len() {
        let (tag, consumed) = match decode_varint(&data[pos..]) {
            Some(v) => v,
            None => break,
        };
        pos += consumed;

        let field_number = (tag >> 3) as u32;
        let wire_type = match tag & 0x07 {
            0 => WireType::Varint,
            1 => WireType::Fixed64,
            2 => WireType::LengthDelimited,
            5 => WireType::Fixed32,
            _ => break, // Unknown wire type
        };

        let (proto_data, consumed) = match wire_type {
            WireType::Varint => {
                match decode_varint(&data[pos..]) {
                    Some((val, c)) => (ProtoData::Varint(val), c),
                    None => break,
                }
            }
            WireType::Fixed64 => {
                if pos + 8 > data.len() { break; }
                let val = u64::from_le_bytes(data[pos..pos+8].try_into().unwrap());
                (ProtoData::Fixed64(val), 8)
            }
            WireType::Fixed32 => {
                if pos + 4 > data.len() { break; }
                let val = u32::from_le_bytes(data[pos..pos+4].try_into().unwrap());
                (ProtoData::Fixed32(val), 4)
            }
            WireType::LengthDelimited => {
                match decode_varint(&data[pos..]) {
                    Some((len, c)) => {
                        let start = pos + c;
                        let end = start + len as usize;
                        if end > data.len() { break; }
                        (ProtoData::Bytes(data[start..end].to_vec()), c + len as usize)
                    }
                    None => break,
                }
            }
        };

        pos += consumed;
        fields.push(ProtoField { field_number, _wire_type: wire_type, data: proto_data });
    }

    fields
}

/// Decode a varint, return (value, bytes_consumed).
fn decode_varint(data: &[u8]) -> Option<(u64, usize)> {
    let mut result: u64 = 0;
    let mut shift = 0;
    for (i, &byte) in data.iter().enumerate() {
        result |= ((byte & 0x7F) as u64) << shift;
        if byte & 0x80 == 0 {
            return Some((result, i + 1));
        }
        shift += 7;
        if shift >= 64 { return None; }
    }
    None
}

/// Map ONNX protobuf TensorProto.DataType enum to element type string.
fn onnx_data_type_to_str(dt: u64) -> &'static str {
    match dt {
        1 => "f32",    // FLOAT
        2 => "u8",     // UINT8
        3 => "i8",     // INT8
        5 => "i16",    // INT16
        6 => "i32",    // INT32
        7 => "i64",    // INT64
        9 => "bool",   // BOOL
        10 => "f16",   // FLOAT16
        11 => "f64",   // DOUBLE
        16 => "bf16",  // BFLOAT16
        _ => "f32",    // Default
    }
}

/// Parse a real ONNX protobuf file (.onnx) into our OnnxGraph.
///
/// ONNX protobuf schema (simplified):
///   ModelProto (field 7 = GraphProto)
///   GraphProto:
///     field 1 = node[] (NodeProto)
///     field 2 = name (string)
///     field 5 = initializer[] (TensorProto)
///     field 11 = input[] (ValueInfoProto)
///     field 12 = output[] (ValueInfoProto)
///   NodeProto:
///     field 1 = input[] (string)
///     field 2 = output[] (string)
///     field 3 = name (string)
///     field 4 = op_type (string)
///     field 5 = attribute[] (AttributeProto)
///   TensorProto:
///     field 1 = dims[] (int64)
///     field 2 = data_type (int32)
///     field 4 = float_data[] (float, packed)
///     field 8 = name (string)
///     field 13 = raw_data (bytes)
///   ValueInfoProto:
///     field 1 = name (string)
///     field 2 = type (TypeProto)
///   TypeProto:
///     field 1 = tensor_type (Tensor)
///   TypeProto.Tensor:
///     field 1 = elem_type (int32)
///     field 2 = shape (TensorShapeProto)
///   TensorShapeProto:
///     field 1 = dim[] (Dimension)
///   TensorShapeProto.Dimension:
///     field 1 = dim_value (int64)
///     field 2 = dim_param (string)
pub fn from_onnx_protobuf(data: &[u8]) -> Result<OnnxGraph, ConversionError> {
    let model_fields = parse_protobuf(data);

    // Find graph (field 7 in ModelProto)
    let graph_data = model_fields.iter()
        .find(|f| f.field_number == 7)
        .ok_or_else(|| ConversionError::Io("No graph found in ONNX model".into()))?
        .data.as_bytes();

    let graph_fields = parse_protobuf(graph_data);

    let graph_name = graph_fields.iter()
        .find(|f| f.field_number == 2)
        .map(|f| f.data.as_str().to_string())
        .unwrap_or_else(|| "imported".into());

    // Parse nodes (field 1)
    let mut nodes = Vec::new();
    for field in graph_fields.iter().filter(|f| f.field_number == 1) {
        let node_fields = parse_protobuf(field.data.as_bytes());
        let mut inputs = Vec::new();
        let mut outputs = Vec::new();
        let mut name = String::new();
        let mut op_type = String::new();
        let mut attributes = HashMap::new();

        for nf in &node_fields {
            match nf.field_number {
                1 => inputs.push(nf.data.as_str().to_string()),
                2 => outputs.push(nf.data.as_str().to_string()),
                3 => name = nf.data.as_str().to_string(),
                4 => op_type = nf.data.as_str().to_string(),
                5 => {
                    // Parse AttributeProto
                    let attr_fields = parse_protobuf(nf.data.as_bytes());
                    let attr_name = attr_fields.iter()
                        .find(|f| f.field_number == 1)
                        .map(|f| f.data.as_str().to_string())
                        .unwrap_or_default();
                    // field 2 = type, field 3 = f, field 4 = i, field 6 = ints
                    let attr_type = attr_fields.iter()
                        .find(|f| f.field_number == 2)
                        .map(|f| f.data.as_u64())
                        .unwrap_or(0);

                    let attr_val = match attr_type {
                        1 => { // FLOAT
                            let f_val = attr_fields.iter()
                                .find(|f| f.field_number == 4)
                                .map(|f| f32::from_bits(f.data.as_u64() as u32))
                                .unwrap_or(0.0);
                            OnnxAttribute::Float(f_val as f64)
                        }
                        2 => { // INT
                            let i_val = attr_fields.iter()
                                .find(|f| f.field_number == 3)
                                .map(|f| f.data.as_i64())
                                .unwrap_or(0);
                            OnnxAttribute::Int(i_val)
                        }
                        7 => { // INTS
                            let ints: Vec<i64> = attr_fields.iter()
                                .filter(|f| f.field_number == 8)
                                .map(|f| f.data.as_i64())
                                .collect();
                            OnnxAttribute::Ints(ints)
                        }
                        _ => OnnxAttribute::Int(0),
                    };

                    if !attr_name.is_empty() {
                        attributes.insert(attr_name, attr_val);
                    }
                }
                _ => {}
            }
        }

        nodes.push(OnnxNode { name, op_type, inputs, outputs, attributes });
    }

    // Parse inputs (field 11)
    let mut onnx_inputs = Vec::new();
    for field in graph_fields.iter().filter(|f| f.field_number == 11) {
        if let Some(vi) = parse_value_info(field.data.as_bytes()) {
            onnx_inputs.push(vi);
        }
    }

    // Parse outputs (field 12)
    let mut onnx_outputs = Vec::new();
    for field in graph_fields.iter().filter(|f| f.field_number == 12) {
        if let Some(vi) = parse_value_info(field.data.as_bytes()) {
            onnx_outputs.push(vi);
        }
    }

    // Parse initializers (field 5)
    let mut initializers = Vec::new();
    for field in graph_fields.iter().filter(|f| f.field_number == 5) {
        let tensor_fields = parse_protobuf(field.data.as_bytes());
        let name = tensor_fields.iter()
            .find(|f| f.field_number == 8)
            .map(|f| f.data.as_str().to_string())
            .unwrap_or_default();
        let data_type = tensor_fields.iter()
            .find(|f| f.field_number == 2)
            .map(|f| f.data.as_u64())
            .unwrap_or(1);
        let dims: Vec<usize> = tensor_fields.iter()
            .filter(|f| f.field_number == 1)
            .map(|f| f.data.as_u64() as usize)
            .collect();

        // Extract weight data from float_data (field 4, packed) or raw_data (field 13)
        let weights = extract_weight_data(&tensor_fields);

        initializers.push(OnnxInitializer {
            name,
            elem_type: onnx_data_type_to_str(data_type).to_string(),
            shape: dims,
            data_base64: None,
            weights,
        });
    }

    Ok(OnnxGraph {
        name: graph_name,
        nodes,
        inputs: onnx_inputs,
        outputs: onnx_outputs,
        initializers,
    })
}

/// Parse a ValueInfoProto from protobuf bytes.
fn parse_value_info(data: &[u8]) -> Option<OnnxValueInfo> {
    let fields = parse_protobuf(data);
    let name = fields.iter()
        .find(|f| f.field_number == 1)?
        .data.as_str().to_string();

    // Parse type (field 2 -> TypeProto -> field 1 -> Tensor -> elem_type + shape)
    let type_data = fields.iter()
        .find(|f| f.field_number == 2)?
        .data.as_bytes();
    let type_fields = parse_protobuf(type_data);

    let tensor_data = type_fields.iter()
        .find(|f| f.field_number == 1)?
        .data.as_bytes();
    let tensor_fields = parse_protobuf(tensor_data);

    let elem_type = tensor_fields.iter()
        .find(|f| f.field_number == 1)
        .map(|f| f.data.as_u64())
        .unwrap_or(1);

    let mut shape = Vec::new();
    if let Some(shape_field) = tensor_fields.iter().find(|f| f.field_number == 2) {
        let shape_fields = parse_protobuf(shape_field.data.as_bytes());
        for dim_field in shape_fields.iter().filter(|f| f.field_number == 1) {
            let dim_inner = parse_protobuf(dim_field.data.as_bytes());
            if let Some(val) = dim_inner.iter().find(|f| f.field_number == 1) {
                shape.push(OnnxDim::Fixed(val.data.as_u64() as usize));
            } else if let Some(param) = dim_inner.iter().find(|f| f.field_number == 2) {
                shape.push(OnnxDim::Dynamic(param.data.as_str().to_string()));
            } else {
                shape.push(OnnxDim::Dynamic("?".into()));
            }
        }
    }

    Some(OnnxValueInfo {
        name,
        elem_type: onnx_data_type_to_str(elem_type).to_string(),
        shape,
    })
}

/// Extract f32 weight data from TensorProto fields.
///
/// Looks for float_data (field 4, packed repeated float as length-delimited bytes)
/// and raw_data (field 13, raw little-endian bytes). float_data takes priority if
/// present; otherwise raw_data is used.
fn extract_weight_data(tensor_fields: &[ProtoField]) -> Option<Vec<f32>> {
    // Try float_data first (field 4) -- packed repeated float sent as length-delimited
    let float_data_fields: Vec<&ProtoField> = tensor_fields
        .iter()
        .filter(|f| f.field_number == 4)
        .collect();

    if !float_data_fields.is_empty() {
        let mut floats = Vec::new();
        for field in &float_data_fields {
            match &field.data {
                // Packed encoding: length-delimited blob of little-endian f32s
                ProtoData::Bytes(bytes) => {
                    floats.extend(parse_packed_f32s(bytes));
                }
                // Individual float encoded as Fixed32 (non-packed, rare but valid)
                ProtoData::Fixed32(v) => {
                    floats.push(f32::from_bits(*v));
                }
                _ => {}
            }
        }
        if !floats.is_empty() {
            return Some(floats);
        }
    }

    // Fall back to raw_data (field 13) -- raw little-endian bytes
    if let Some(raw_field) = tensor_fields.iter().find(|f| f.field_number == 13) {
        let bytes = raw_field.data.as_bytes();
        if !bytes.is_empty() {
            let floats = parse_packed_f32s(bytes);
            if !floats.is_empty() {
                return Some(floats);
            }
        }
    }

    None
}

/// Parse a byte slice as packed little-endian f32 values.
fn parse_packed_f32s(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect()
}

impl OnnxGraph {
    /// Total size in bytes of all extracted weight data across initializers.
    ///
    /// Counts only initializers that have actual weight data loaded (i.e. `weights`
    /// is `Some`). Each f32 value contributes 4 bytes.
    pub fn total_weight_bytes(&self) -> usize {
        self.initializers
            .iter()
            .filter_map(|init| init.weights.as_ref())
            .map(|w| w.len() * std::mem::size_of::<f32>())
            .sum()
    }
}

/// Load an ONNX protobuf file and convert to our OnnxGraph.
pub fn from_onnx_file(path: &Path) -> Result<OnnxGraph, ConversionError> {
    let data = std::fs::read(path)?;
    from_onnx_protobuf(&data)
}

// ---------------------------------------------------------------------------
// Op mapping: QLANG Op <-> ONNX op_type string
// ---------------------------------------------------------------------------

/// Map a QLANG Op to an ONNX op_type string and optional attributes.
fn op_to_onnx(op: &Op) -> Option<(String, HashMap<String, OnnxAttribute>)> {
    let mut attrs = HashMap::new();
    let op_type = match op {
        // Tensor ops
        Op::Add => "Add",
        Op::Sub => "Sub",
        Op::Mul => "Mul",
        Op::Div => "Div",
        Op::Neg => "Neg",
        Op::Exp => "Exp",
        Op::Log => "Log",
        Op::MatMul => "MatMul",
        Op::Transpose => "Transpose",
        Op::Reshape { target_shape } => {
            attrs.insert(
                "shape".to_string(),
                OnnxAttribute::Ints(target_shape.iter().map(|&d| d as i64).collect()),
            );
            "Reshape"
        }
        Op::Slice { start, end } => {
            attrs.insert(
                "starts".to_string(),
                OnnxAttribute::Ints(start.iter().map(|&s| s as i64).collect()),
            );
            attrs.insert(
                "ends".to_string(),
                OnnxAttribute::Ints(end.iter().map(|&e| e as i64).collect()),
            );
            "Slice"
        }
        Op::Concat { axis } => {
            attrs.insert("axis".to_string(), OnnxAttribute::Int(*axis as i64));
            "Concat"
        }
        Op::ReduceSum { axis } => {
            if let Some(a) = axis {
                attrs.insert("axes".to_string(), OnnxAttribute::Ints(vec![*a as i64]));
            }
            "ReduceSum"
        }
        Op::ReduceMean { axis } => {
            if let Some(a) = axis {
                attrs.insert("axes".to_string(), OnnxAttribute::Ints(vec![*a as i64]));
            }
            "ReduceMean"
        }
        Op::ReduceMax { axis } => {
            if let Some(a) = axis {
                attrs.insert("axes".to_string(), OnnxAttribute::Ints(vec![*a as i64]));
            }
            "ReduceMax"
        }
        // Activations
        Op::Relu => "Relu",
        Op::Sigmoid => "Sigmoid",
        Op::Tanh => "Tanh",
        Op::Softmax { axis } => {
            attrs.insert("axis".to_string(), OnnxAttribute::Int(*axis as i64));
            "Softmax"
        }
        // Transformer ops
        Op::LayerNorm { eps } => {
            attrs.insert("epsilon".to_string(), OnnxAttribute::Float(*eps));
            "LayerNormalization"
        }
        Op::Gelu => "Gelu",
        Op::Dropout { rate } => {
            attrs.insert("ratio".to_string(), OnnxAttribute::Float(*rate));
            "Dropout"
        }
        // Constant
        Op::Constant => "Constant",
        // Residual mapped as Add (it's x + f(x))
        Op::Residual => "Add",
        // Ops that have no standard ONNX equivalent get a prefixed domain.
        Op::Superpose
        | Op::Evolve { .. }
        | Op::Measure
        | Op::Entangle
        | Op::Collapse
        | Op::Entropy
        | Op::ToTernary
        | Op::ToLowRank { .. }
        | Op::ToSparse { .. }
        | Op::FisherMetric
        | Op::Project { .. }
        | Op::Attention { .. }
        | Op::Embedding { .. }
        | Op::Cond
        | Op::Scan { .. }
        | Op::SubGraph { .. }
        | Op::OllamaGenerate { .. }
        | Op::OllamaChat { .. }
        | Op::ClassMean { .. }
        | Op::Ternarize { .. }
        | Op::TernaryMatVec
        | Op::ArgMax
        | Op::EnsembleVote => {
            return Some((format!("com.qlang.{op}"), op_to_qlang_attrs(op)));
        }
        // Input/Output are not emitted as ONNX nodes.
        Op::Input { .. } | Op::Output { .. } => return None,
    };
    Some((op_type.to_string(), attrs))
}

/// Produce QLANG-specific attributes for quantum/custom ops.
fn op_to_qlang_attrs(op: &Op) -> HashMap<String, OnnxAttribute> {
    let mut attrs = HashMap::new();
    match op {
        Op::Evolve { gamma, dt } => {
            attrs.insert("gamma".to_string(), OnnxAttribute::Float(*gamma));
            attrs.insert("dt".to_string(), OnnxAttribute::Float(*dt));
        }
        Op::ToLowRank { rank } => {
            attrs.insert("rank".to_string(), OnnxAttribute::Int(*rank as i64));
        }
        Op::ToSparse { sparsity } => {
            attrs.insert("sparsity".to_string(), OnnxAttribute::Float(*sparsity));
        }
        Op::Softmax { axis } => {
            attrs.insert("axis".to_string(), OnnxAttribute::Int(*axis as i64));
        }
        Op::Attention { n_heads, d_model } => {
            attrs.insert("n_heads".to_string(), OnnxAttribute::Int(*n_heads as i64));
            attrs.insert("d_model".to_string(), OnnxAttribute::Int(*d_model as i64));
        }
        Op::Embedding { vocab_size, d_model } => {
            attrs.insert(
                "vocab_size".to_string(),
                OnnxAttribute::Int(*vocab_size as i64),
            );
            attrs.insert("d_model".to_string(), OnnxAttribute::Int(*d_model as i64));
        }
        Op::Scan { n_iterations } => {
            attrs.insert(
                "n_iterations".to_string(),
                OnnxAttribute::Int(*n_iterations as i64),
            );
        }
        Op::SubGraph { graph_id } => {
            attrs.insert(
                "graph_id".to_string(),
                OnnxAttribute::String(graph_id.clone()),
            );
        }
        _ => {}
    }
    attrs
}

/// Map an ONNX op_type back to a QLANG Op, using provided attributes.
fn onnx_op_to_qlang(
    op_type: &str,
    attrs: &HashMap<String, OnnxAttribute>,
    node_name: &str,
) -> Result<Op, ConversionError> {
    match op_type {
        "Add" => Ok(Op::Add),
        "Sub" => Ok(Op::Sub),
        "Mul" => Ok(Op::Mul),
        "Div" => Ok(Op::Div),
        "Neg" => Ok(Op::Neg),
        "MatMul" => Ok(Op::MatMul),
        "Transpose" => Ok(Op::Transpose),
        "Reshape" => {
            let shape = get_ints_attr(attrs, "shape", node_name)?;
            Ok(Op::Reshape {
                target_shape: shape.iter().map(|&v| v as usize).collect(),
            })
        }
        "Slice" => {
            let starts = get_ints_attr(attrs, "starts", node_name)?;
            let ends = get_ints_attr(attrs, "ends", node_name)?;
            Ok(Op::Slice {
                start: starts.iter().map(|&v| v as usize).collect(),
                end: ends.iter().map(|&v| v as usize).collect(),
            })
        }
        "Concat" => {
            let axis = get_int_attr(attrs, "axis", node_name)?;
            Ok(Op::Concat {
                axis: axis as usize,
            })
        }
        "ReduceSum" => {
            let axis = optional_axes_attr(attrs);
            Ok(Op::ReduceSum { axis })
        }
        "ReduceMean" => {
            let axis = optional_axes_attr(attrs);
            Ok(Op::ReduceMean { axis })
        }
        "ReduceMax" => {
            let axis = optional_axes_attr(attrs);
            Ok(Op::ReduceMax { axis })
        }
        "Relu" => Ok(Op::Relu),
        "Sigmoid" => Ok(Op::Sigmoid),
        "Tanh" => Ok(Op::Tanh),
        "Softmax" => {
            let axis = get_int_attr(attrs, "axis", node_name).unwrap_or(1);
            Ok(Op::Softmax {
                axis: axis as usize,
            })
        }
        "LayerNormalization" => {
            let eps = get_float_attr(attrs, "epsilon", node_name).unwrap_or(1e-5);
            Ok(Op::LayerNorm { eps })
        }
        "Gelu" => Ok(Op::Gelu),
        "Dropout" => {
            let rate = get_float_attr(attrs, "ratio", node_name).unwrap_or(0.5);
            Ok(Op::Dropout { rate })
        }
        "Constant" => Ok(Op::Constant),
        other if other.starts_with("com.qlang.") => {
            qlang_custom_op(other.strip_prefix("com.qlang.").unwrap(), attrs, node_name)
        }
        _ => Err(ConversionError::UnsupportedOp(op_type.to_string())),
    }
}

/// Convert QLANG custom-domain ops back to Op.
fn qlang_custom_op(
    tag: &str,
    attrs: &HashMap<String, OnnxAttribute>,
    node_name: &str,
) -> Result<Op, ConversionError> {
    // The tag is the Display representation of the Op.
    // We match on known prefixes.
    if tag.starts_with("superpose") {
        return Ok(Op::Superpose);
    }
    if tag.starts_with("evolve") {
        let gamma = get_float_attr(attrs, "gamma", node_name)?;
        let dt = get_float_attr(attrs, "dt", node_name)?;
        return Ok(Op::Evolve { gamma, dt });
    }
    if tag.starts_with("measure") {
        return Ok(Op::Measure);
    }
    if tag.starts_with("entangle") {
        return Ok(Op::Entangle);
    }
    if tag.starts_with("collapse") {
        return Ok(Op::Collapse);
    }
    if tag.starts_with("entropy") {
        return Ok(Op::Entropy);
    }
    if tag.starts_with("to_ternary") {
        return Ok(Op::ToTernary);
    }
    if tag.starts_with("to_lowrank") {
        let rank = get_int_attr(attrs, "rank", node_name)? as usize;
        return Ok(Op::ToLowRank { rank });
    }
    if tag.starts_with("to_sparse") {
        let sparsity = get_float_attr(attrs, "sparsity", node_name)?;
        return Ok(Op::ToSparse { sparsity });
    }
    if tag.starts_with("fisher_metric") {
        return Ok(Op::FisherMetric);
    }
    if tag.starts_with("attention") {
        let n_heads = get_int_attr(attrs, "n_heads", node_name)? as usize;
        let d_model = get_int_attr(attrs, "d_model", node_name)? as usize;
        return Ok(Op::Attention { n_heads, d_model });
    }
    if tag.starts_with("embedding") {
        let vocab_size = get_int_attr(attrs, "vocab_size", node_name)? as usize;
        let d_model = get_int_attr(attrs, "d_model", node_name)? as usize;
        return Ok(Op::Embedding {
            vocab_size,
            d_model,
        });
    }
    if tag.starts_with("scan") {
        let n = get_int_attr(attrs, "n_iterations", node_name)? as usize;
        return Ok(Op::Scan { n_iterations: n });
    }
    if tag.starts_with("subgraph") {
        let gid = get_string_attr(attrs, "graph_id", node_name)?;
        return Ok(Op::SubGraph { graph_id: gid });
    }
    Err(ConversionError::UnsupportedOp(format!(
        "com.qlang.{tag}"
    )))
}

// ---------------------------------------------------------------------------
// Attribute helpers
// ---------------------------------------------------------------------------

fn get_int_attr(
    attrs: &HashMap<String, OnnxAttribute>,
    key: &str,
    node: &str,
) -> Result<i64, ConversionError> {
    match attrs.get(key) {
        Some(OnnxAttribute::Int(v)) => Ok(*v),
        Some(_) => Err(ConversionError::AttributeTypeMismatch {
            node: node.to_string(),
            attr: key.to_string(),
        }),
        None => Err(ConversionError::MissingAttribute {
            node: node.to_string(),
            attr: key.to_string(),
        }),
    }
}

fn get_float_attr(
    attrs: &HashMap<String, OnnxAttribute>,
    key: &str,
    node: &str,
) -> Result<f64, ConversionError> {
    match attrs.get(key) {
        Some(OnnxAttribute::Float(v)) => Ok(*v),
        Some(_) => Err(ConversionError::AttributeTypeMismatch {
            node: node.to_string(),
            attr: key.to_string(),
        }),
        None => Err(ConversionError::MissingAttribute {
            node: node.to_string(),
            attr: key.to_string(),
        }),
    }
}

fn get_ints_attr(
    attrs: &HashMap<String, OnnxAttribute>,
    key: &str,
    node: &str,
) -> Result<Vec<i64>, ConversionError> {
    match attrs.get(key) {
        Some(OnnxAttribute::Ints(v)) => Ok(v.clone()),
        Some(_) => Err(ConversionError::AttributeTypeMismatch {
            node: node.to_string(),
            attr: key.to_string(),
        }),
        None => Err(ConversionError::MissingAttribute {
            node: node.to_string(),
            attr: key.to_string(),
        }),
    }
}

fn get_string_attr(
    attrs: &HashMap<String, OnnxAttribute>,
    key: &str,
    node: &str,
) -> Result<String, ConversionError> {
    match attrs.get(key) {
        Some(OnnxAttribute::String(v)) => Ok(v.clone()),
        Some(_) => Err(ConversionError::AttributeTypeMismatch {
            node: node.to_string(),
            attr: key.to_string(),
        }),
        None => Err(ConversionError::MissingAttribute {
            node: node.to_string(),
            attr: key.to_string(),
        }),
    }
}

fn optional_axes_attr(attrs: &HashMap<String, OnnxAttribute>) -> Option<usize> {
    if let Some(OnnxAttribute::Ints(axes)) = attrs.get("axes") {
        axes.first().map(|&a| a as usize)
    } else {
        None
    }
}

// ---------------------------------------------------------------------------
// Dtype <-> ONNX elem_type mapping
// ---------------------------------------------------------------------------

fn dtype_to_onnx(dtype: Dtype) -> &'static str {
    match dtype {
        Dtype::F16 => "FLOAT16",
        Dtype::F32 => "FLOAT",
        Dtype::F64 => "DOUBLE",
        Dtype::I8 => "INT8",
        Dtype::I16 => "INT16",
        Dtype::I32 => "INT32",
        Dtype::I64 => "INT64",
        Dtype::Bool => "BOOL",
        Dtype::Ternary => "INT8", // closest standard type
        Dtype::Utf8 => "STRING",
    }
}

fn onnx_to_dtype(elem_type: &str) -> Result<Dtype, ConversionError> {
    match elem_type {
        "FLOAT16" => Ok(Dtype::F16),
        "FLOAT" => Ok(Dtype::F32),
        "DOUBLE" => Ok(Dtype::F64),
        "INT8" => Ok(Dtype::I8),
        "INT16" => Ok(Dtype::I16),
        "INT32" => Ok(Dtype::I32),
        "INT64" => Ok(Dtype::I64),
        "BOOL" => Ok(Dtype::Bool),
        _ => Err(ConversionError::UnknownElemType(elem_type.to_string())),
    }
}

fn shape_to_onnx(shape: &Shape) -> Vec<OnnxDim> {
    shape
        .0
        .iter()
        .map(|d| match d {
            Dim::Fixed(n) => OnnxDim::Fixed(*n),
            Dim::Dynamic => OnnxDim::Dynamic("?".to_string()),
        })
        .collect()
}

fn onnx_to_shape(dims: &[OnnxDim]) -> Shape {
    Shape(
        dims.iter()
            .map(|d| match d {
                OnnxDim::Fixed(n) => Dim::Fixed(*n),
                OnnxDim::Dynamic(_) => Dim::Dynamic,
            })
            .collect(),
    )
}

fn tensor_type_to_value_info(name: &str, tt: &TensorType) -> OnnxValueInfo {
    OnnxValueInfo {
        name: name.to_string(),
        elem_type: dtype_to_onnx(tt.dtype).to_string(),
        shape: shape_to_onnx(&tt.shape),
    }
}

// ---------------------------------------------------------------------------
// Wire-name helpers
// ---------------------------------------------------------------------------

/// Produce a deterministic edge wire name: `node{id}_out{port}`.
fn wire_name(node_id: NodeId, port: u8) -> String {
    format!("node{node_id}_out{port}")
}

// ---------------------------------------------------------------------------
// QLANG Graph -> OnnxGraph
// ---------------------------------------------------------------------------

/// Convert a QLANG [`Graph`] into an [`OnnxGraph`] for export.
///
/// Input and Output nodes are mapped to graph-level inputs/outputs rather than
/// ONNX operator nodes.
pub fn to_onnx(graph: &Graph) -> OnnxGraph {
    let mut onnx_nodes = Vec::new();
    let mut onnx_inputs = Vec::new();
    let mut onnx_outputs = Vec::new();

    // Pre-build a map: for each (to_node, to_port) find the producing wire.
    let mut input_wire: HashMap<(NodeId, u8), String> = HashMap::new();
    for edge in &graph.edges {
        input_wire.insert(
            (edge.to_node, edge.to_port),
            wire_name(edge.from_node, edge.from_port),
        );
    }

    for node in &graph.nodes {
        match &node.op {
            Op::Input { name } => {
                // Register as graph input.
                let out_name = wire_name(node.id, 0);
                if let Some(tt) = node.output_types.first() {
                    onnx_inputs.push(tensor_type_to_value_info(&out_name, tt));
                } else {
                    onnx_inputs.push(OnnxValueInfo {
                        name: out_name.clone(),
                        elem_type: "FLOAT".to_string(),
                        shape: vec![],
                    });
                }
                // Store the friendly name in metadata so roundtrip preserves it.
                // We add a minimal "Identity" passthrough only if we need to
                // keep the name mapping; skip for cleanliness.
                let _ = name; // used implicitly via node metadata
            }
            Op::Output { name } => {
                // Register as graph output.
                let in_wire = input_wire
                    .get(&(node.id, 0))
                    .cloned()
                    .unwrap_or_else(|| format!("output_{name}"));
                if let Some(tt) = node.input_types.first() {
                    onnx_outputs.push(tensor_type_to_value_info(&in_wire, tt));
                } else {
                    onnx_outputs.push(OnnxValueInfo {
                        name: in_wire,
                        elem_type: "FLOAT".to_string(),
                        shape: vec![],
                    });
                }
            }
            op => {
                if let Some((op_type, mut attributes)) = op_to_onnx(op) {
                    // Build input wire list (in port order).
                    let n_in = op.n_inputs();
                    let inputs: Vec<String> = (0..n_in as u8)
                        .map(|port| {
                            input_wire
                                .get(&(node.id, port))
                                .cloned()
                                .unwrap_or_default()
                        })
                        .collect();

                    // Build output wire list.
                    let n_out = op.n_outputs();
                    let outputs: Vec<String> =
                        (0..n_out as u8).map(|port| wire_name(node.id, port)).collect();

                    // Carry over QLANG metadata as string attributes.
                    for (k, v) in &node.metadata {
                        attributes
                            .entry(format!("qlang.{k}"))
                            .or_insert_with(|| OnnxAttribute::String(v.clone()));
                    }

                    onnx_nodes.push(OnnxNode {
                        name: format!("node_{}", node.id),
                        op_type,
                        inputs,
                        outputs,
                        attributes,
                    });
                }
            }
        }
    }

    OnnxGraph {
        name: graph.id.clone(),
        nodes: onnx_nodes,
        inputs: onnx_inputs,
        outputs: onnx_outputs,
        initializers: Vec::new(),
    }
}

// ---------------------------------------------------------------------------
// OnnxGraph -> QLANG Graph
// ---------------------------------------------------------------------------

/// Convert an [`OnnxGraph`] into a QLANG [`Graph`].
pub fn from_onnx(onnx: &OnnxGraph) -> Result<Graph, ConversionError> {
    let mut graph = Graph::new(&onnx.name);

    // Maps wire-name -> (producing NodeId, port).
    let mut wire_producers: HashMap<String, (NodeId, u8)> = HashMap::new();

    // 1. Create Input nodes for each graph input.
    for vi in &onnx.inputs {
        let dtype = onnx_to_dtype(&vi.elem_type)?;
        let shape = onnx_to_shape(&vi.shape);
        let tt = TensorType::new(dtype, shape);
        let nid = graph.add_node(
            Op::Input {
                name: vi.name.clone(),
            },
            vec![],
            vec![tt],
        );
        wire_producers.insert(vi.name.clone(), (nid, 0));
    }

    // 2. Process ONNX nodes in order (assumed topological).
    for onode in &onnx.nodes {
        let op = onnx_op_to_qlang(&onode.op_type, &onode.attributes, &onode.name)?;

        // Build input types from producers where possible; default to f32 scalar.
        let input_types: Vec<TensorType> = onode
            .inputs
            .iter()
            .map(|wire| {
                if let Some(&(pid, _port)) = wire_producers.get(wire) {
                    if let Some(pnode) = graph.node(pid) {
                        pnode.output_types.first().cloned().unwrap_or_else(TensorType::f32_scalar)
                    } else {
                        TensorType::f32_scalar()
                    }
                } else {
                    TensorType::f32_scalar()
                }
            })
            .collect();

        // Output types: for simplicity, propagate the first input type (correct
        // for element-wise ops; callers can refine via type inference).
        let output_types: Vec<TensorType> = onode
            .outputs
            .iter()
            .map(|_| input_types.first().cloned().unwrap_or_else(TensorType::f32_scalar))
            .collect();

        let nid = graph.add_node(op, input_types, output_types);

        // Register output wires.
        for (port, wire) in onode.outputs.iter().enumerate() {
            wire_producers.insert(wire.clone(), (nid, port as u8));
        }

        // Create edges for inputs.
        for (to_port, wire) in onode.inputs.iter().enumerate() {
            if let Some(&(from_node, from_port)) = wire_producers.get(wire) {
                let tt = if let Some(pnode) = graph.node(from_node) {
                    pnode
                        .output_types
                        .get(from_port as usize)
                        .cloned()
                        .unwrap_or_else(TensorType::f32_scalar)
                } else {
                    TensorType::f32_scalar()
                };
                graph.add_edge(from_node, from_port, nid, to_port as u8, tt);
            }
        }
    }

    // 3. Create Output nodes for each graph output.
    for vi in &onnx.outputs {
        let dtype = onnx_to_dtype(&vi.elem_type)?;
        let shape = onnx_to_shape(&vi.shape);
        let tt = TensorType::new(dtype, shape);
        let nid = graph.add_node(
            Op::Output {
                name: vi.name.clone(),
            },
            vec![tt.clone()],
            vec![],
        );
        if let Some(&(from_node, from_port)) = wire_producers.get(&vi.name) {
            graph.add_edge(from_node, from_port, nid, 0, tt);
        }
    }

    Ok(graph)
}

// ---------------------------------------------------------------------------
// JSON export
// ---------------------------------------------------------------------------

/// Serialize a QLANG [`Graph`] as ONNX-like JSON.
///
/// This produces a JSON string representing the [`OnnxGraph`] that can be
/// consumed by Python tooling (e.g., `json.loads(...)` and then fed into
/// `onnx.helper.make_graph`).
pub fn to_onnx_json(graph: &Graph) -> String {
    let onnx = to_onnx(graph);
    serde_json::to_string_pretty(&onnx).expect("OnnxGraph serialization cannot fail")
}

// ---------------------------------------------------------------------------
// Large Model Support
// ---------------------------------------------------------------------------

/// Metadata about a model (extracted without loading weights).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelMetadata {
    pub name: String,
    pub num_nodes: usize,
    pub num_parameters: u64,
    pub num_layers: usize,
    pub op_histogram: HashMap<String, usize>,
    pub input_shapes: Vec<(String, Vec<OnnxDim>)>,
    pub output_shapes: Vec<(String, Vec<OnnxDim>)>,
    pub estimated_size_mb: f64,
}

/// Extract metadata from an ONNX graph without loading weight data.
pub fn model_metadata(onnx: &OnnxGraph) -> ModelMetadata {
    let mut op_histogram: HashMap<String, usize> = HashMap::new();
    for node in &onnx.nodes {
        *op_histogram.entry(node.op_type.clone()).or_insert(0) += 1;
    }

    let mut num_parameters: u64 = 0;
    for init in &onnx.initializers {
        let elems: u64 = init.shape.iter().map(|&d| d as u64).product();
        num_parameters += elems;
    }

    let estimated_size_mb = (num_parameters * 4) as f64 / (1024.0 * 1024.0);

    let num_layers = onnx
        .nodes
        .iter()
        .filter(|n| matches!(n.op_type.as_str(), "MatMul" | "Conv" | "Gemm" | "Linear"))
        .count();

    ModelMetadata {
        name: onnx.name.clone(),
        num_nodes: onnx.nodes.len(),
        num_parameters,
        num_layers,
        op_histogram,
        input_shapes: onnx
            .inputs
            .iter()
            .map(|i| (i.name.clone(), i.shape.clone()))
            .collect(),
        output_shapes: onnx
            .outputs
            .iter()
            .map(|o| (o.name.clone(), o.shape.clone()))
            .collect(),
        estimated_size_mb,
    }
}

/// Process an ONNX graph layer-by-layer for large models.
/// Calls `processor` for each layer (MatMul/Conv/Gemm node) in order.
/// This enables streaming processing without loading the entire model.
pub fn process_layers<F>(onnx: &OnnxGraph, mut processor: F) -> Result<(), ConversionError>
where
    F: FnMut(usize, &OnnxNode) -> Result<(), ConversionError>,
{
    let mut layer_idx = 0;
    for node in &onnx.nodes {
        if matches!(
            node.op_type.as_str(),
            "MatMul" | "Conv" | "Gemm" | "Linear" | "LayerNorm" | "Attention"
        ) {
            processor(layer_idx, node)?;
            layer_idx += 1;
        }
    }
    Ok(())
}

/// Load an ONNX JSON model from a file.
pub fn from_onnx_json_file(path: &Path) -> Result<OnnxGraph, ConversionError> {
    let data = std::fs::read_to_string(path)?;
    serde_json::from_str(&data).map_err(|e| ConversionError::Io(format!("JSON parse error: {e}")))
}

/// Save an ONNX JSON model to a file.
pub fn to_onnx_json_file(graph: &Graph, path: &Path) -> Result<(), ConversionError> {
    let json = to_onnx_json(graph);
    std::fs::write(path, json)?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use qlang_core::graph::Graph;
    use qlang_core::ops::Op;
    use qlang_core::tensor::{Dtype, Shape, TensorType};

    fn f32_vec(n: usize) -> TensorType {
        TensorType::new(Dtype::F32, Shape::vector(n))
    }

    fn f32_mat(m: usize, n: usize) -> TensorType {
        TensorType::new(Dtype::F32, Shape::matrix(m, n))
    }

    /// Build a simple graph: input -> relu -> output
    fn simple_relu_graph() -> Graph {
        let mut g = Graph::new("relu_test");
        let inp = g.add_node(Op::Input { name: "x".into() }, vec![], vec![f32_vec(4)]);
        let relu = g.add_node(Op::Relu, vec![f32_vec(4)], vec![f32_vec(4)]);
        let out = g.add_node(
            Op::Output { name: "y".into() },
            vec![f32_vec(4)],
            vec![],
        );
        g.add_edge(inp, 0, relu, 0, f32_vec(4));
        g.add_edge(relu, 0, out, 0, f32_vec(4));
        g
    }

    #[test]
    fn test_to_onnx_simple() {
        let g = simple_relu_graph();
        let onnx = to_onnx(&g);

        assert_eq!(onnx.name, "relu_test");
        assert_eq!(onnx.inputs.len(), 1);
        assert_eq!(onnx.outputs.len(), 1);
        assert_eq!(onnx.nodes.len(), 1);
        assert_eq!(onnx.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_to_onnx_json_parses() {
        let g = simple_relu_graph();
        let json = to_onnx_json(&g);
        let parsed: serde_json::Value = serde_json::from_str(&json).expect("valid JSON");
        assert_eq!(parsed["name"], "relu_test");
        assert!(parsed["nodes"].as_array().unwrap().len() == 1);
    }

    #[test]
    fn test_roundtrip_relu() {
        let g = simple_relu_graph();
        let onnx = to_onnx(&g);
        let g2 = from_onnx(&onnx).expect("conversion succeeds");

        // Should have input + relu + output = 3 nodes.
        assert_eq!(g2.nodes.len(), 3);
        // Should have 2 edges.
        assert_eq!(g2.edges.len(), 2);
        // The relu node should exist.
        assert!(g2.nodes.iter().any(|n| n.op == Op::Relu));
    }

    #[test]
    fn test_roundtrip_matmul_add() {
        let mut g = Graph::new("linear");
        let x = g.add_node(Op::Input { name: "x".into() }, vec![], vec![f32_mat(1, 4)]);
        let w = g.add_node(Op::Input { name: "w".into() }, vec![], vec![f32_mat(4, 2)]);
        let b = g.add_node(
            Op::Input { name: "b".into() },
            vec![],
            vec![f32_vec(2)],
        );
        let mm = g.add_node(
            Op::MatMul,
            vec![f32_mat(1, 4), f32_mat(4, 2)],
            vec![f32_mat(1, 2)],
        );
        let add = g.add_node(
            Op::Add,
            vec![f32_mat(1, 2), f32_vec(2)],
            vec![f32_mat(1, 2)],
        );
        let out = g.add_node(
            Op::Output { name: "y".into() },
            vec![f32_mat(1, 2)],
            vec![],
        );
        g.add_edge(x, 0, mm, 0, f32_mat(1, 4));
        g.add_edge(w, 0, mm, 1, f32_mat(4, 2));
        g.add_edge(mm, 0, add, 0, f32_mat(1, 2));
        g.add_edge(b, 0, add, 1, f32_vec(2));
        g.add_edge(add, 0, out, 0, f32_mat(1, 2));

        let onnx = to_onnx(&g);
        assert_eq!(onnx.nodes.len(), 2); // MatMul + Add
        assert_eq!(onnx.inputs.len(), 3); // x, w, b

        let g2 = from_onnx(&onnx).expect("roundtrip succeeds");
        assert_eq!(g2.nodes.len(), 6); // 3 inputs + 2 ops + 1 output
        assert!(g2.nodes.iter().any(|n| n.op == Op::MatMul));
        assert!(g2.nodes.iter().any(|n| n.op == Op::Add));
    }

    #[test]
    fn test_op_mapping_coverage() {
        // Verify that standard ops map to known ONNX names.
        let cases: Vec<(Op, &str)> = vec![
            (Op::Add, "Add"),
            (Op::Sub, "Sub"),
            (Op::Mul, "Mul"),
            (Op::Div, "Div"),
            (Op::Neg, "Neg"),
            (Op::MatMul, "MatMul"),
            (Op::Relu, "Relu"),
            (Op::Sigmoid, "Sigmoid"),
            (Op::Tanh, "Tanh"),
            (Op::Transpose, "Transpose"),
            (Op::Constant, "Constant"),
            (Op::Gelu, "Gelu"),
        ];
        for (op, expected) in cases {
            let (onnx_type, _) = op_to_onnx(&op).expect("op should map");
            assert_eq!(onnx_type, expected, "mismatch for {op:?}");
        }
    }

    #[test]
    fn test_quantum_op_roundtrip() {
        let mut g = Graph::new("quantum");
        let inp = g.add_node(Op::Input { name: "rho".into() }, vec![], vec![f32_mat(4, 4)]);
        let ent = g.add_node(
            Op::Entropy,
            vec![f32_mat(4, 4)],
            vec![TensorType::f32_scalar()],
        );
        let out = g.add_node(
            Op::Output { name: "s".into() },
            vec![TensorType::f32_scalar()],
            vec![],
        );
        g.add_edge(inp, 0, ent, 0, f32_mat(4, 4));
        g.add_edge(ent, 0, out, 0, TensorType::f32_scalar());

        let onnx = to_onnx(&g);
        assert_eq!(onnx.nodes.len(), 1);
        assert!(onnx.nodes[0].op_type.starts_with("com.qlang."));

        let g2 = from_onnx(&onnx).expect("roundtrip");
        assert!(g2.nodes.iter().any(|n| n.op == Op::Entropy));
    }

    #[test]
    fn test_unsupported_op_error() {
        let onnx = OnnxGraph {
            name: "bad".into(),
            nodes: vec![OnnxNode {
                name: "n0".into(),
                op_type: "FooBarBaz".into(),
                inputs: vec![],
                outputs: vec!["out".into()],
                attributes: HashMap::new(),
            }],
            inputs: vec![],
            outputs: vec![],
            initializers: vec![],
        };
        let result = from_onnx(&onnx);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            matches!(err, ConversionError::UnsupportedOp(ref s) if s == "FooBarBaz"),
            "expected UnsupportedOp, got {err:?}"
        );
    }

    #[test]
    fn test_json_roundtrip() {
        let g = simple_relu_graph();
        let json = to_onnx_json(&g);
        let onnx: OnnxGraph = serde_json::from_str(&json).expect("deserialize");
        let g2 = from_onnx(&onnx).expect("from_onnx");
        assert_eq!(g2.nodes.len(), 3);
    }

    #[test]
    fn test_softmax_with_axis() {
        let mut g = Graph::new("sm");
        let inp = g.add_node(Op::Input { name: "x".into() }, vec![], vec![f32_mat(2, 3)]);
        let sm = g.add_node(
            Op::Softmax { axis: 1 },
            vec![f32_mat(2, 3)],
            vec![f32_mat(2, 3)],
        );
        let out = g.add_node(
            Op::Output { name: "y".into() },
            vec![f32_mat(2, 3)],
            vec![],
        );
        g.add_edge(inp, 0, sm, 0, f32_mat(2, 3));
        g.add_edge(sm, 0, out, 0, f32_mat(2, 3));

        let onnx = to_onnx(&g);
        assert_eq!(onnx.nodes[0].op_type, "Softmax");
        assert_eq!(
            onnx.nodes[0].attributes.get("axis"),
            Some(&OnnxAttribute::Int(1))
        );

        let g2 = from_onnx(&onnx).expect("roundtrip");
        let sm_node = g2.nodes.iter().find(|n| matches!(n.op, Op::Softmax { .. }));
        assert!(sm_node.is_some());
        assert_eq!(sm_node.unwrap().op, Op::Softmax { axis: 1 });
    }

    #[test]
    fn test_model_metadata() {
        let g = simple_relu_graph();
        let onnx = to_onnx(&g);
        let meta = model_metadata(&onnx);
        assert_eq!(meta.name, "relu_test");
        assert_eq!(meta.num_nodes, 1); // only relu (input/output excluded by ONNX export)
        assert!(!meta.op_histogram.is_empty());
    }

    #[test]
    fn test_process_layers() {
        let mut g = Graph::new("mlp");
        let x = g.add_node(Op::Input { name: "x".into() }, vec![], vec![f32_mat(1, 4)]);
        let w = g.add_node(Op::Input { name: "w".into() }, vec![], vec![f32_mat(4, 2)]);
        let mm = g.add_node(Op::MatMul, vec![f32_mat(1, 4), f32_mat(4, 2)], vec![f32_mat(1, 2)]);
        let relu = g.add_node(Op::Relu, vec![f32_mat(1, 2)], vec![f32_mat(1, 2)]);
        let out = g.add_node(Op::Output { name: "y".into() }, vec![f32_mat(1, 2)], vec![]);
        g.add_edge(x, 0, mm, 0, f32_mat(1, 4));
        g.add_edge(w, 0, mm, 1, f32_mat(4, 2));
        g.add_edge(mm, 0, relu, 0, f32_mat(1, 2));
        g.add_edge(relu, 0, out, 0, f32_mat(1, 2));

        let onnx = to_onnx(&g);
        let mut layer_count = 0;
        process_layers(&onnx, |_idx, _node| {
            layer_count += 1;
            Ok(())
        }).unwrap();
        assert_eq!(layer_count, 1); // one MatMul layer
    }

    #[test]
    fn test_decode_varint() {
        // 1 byte: 0x08 = 8
        assert_eq!(decode_varint(&[0x08]), Some((8, 1)));
        // 2 bytes: 300 = 0xAC 0x02
        assert_eq!(decode_varint(&[0xAC, 0x02]), Some((300, 2)));
        // Empty
        assert_eq!(decode_varint(&[]), None);
    }

    #[test]
    fn test_parse_protobuf_simple() {
        // Encode: field 1, varint, value 42
        // tag = (1 << 3) | 0 = 8, value = 42
        let data = vec![0x08, 42];
        let fields = parse_protobuf(&data);
        assert_eq!(fields.len(), 1);
        assert_eq!(fields[0].field_number, 1);
        assert_eq!(fields[0].data.as_u64(), 42);
    }

    #[test]
    fn test_parse_protobuf_string() {
        // field 2, length-delimited (wire type 2), string "test"
        // tag = (2 << 3) | 2 = 18, length = 4, data = "test"
        let data = vec![0x12, 4, b't', b'e', b's', b't'];
        let fields = parse_protobuf(&data);
        assert_eq!(fields.len(), 1);
        assert_eq!(fields[0].field_number, 2);
        assert_eq!(fields[0].data.as_str(), "test");
    }

    #[test]
    fn test_parse_protobuf_multiple_fields() {
        // field 1, varint 10; field 2, string "hi"
        let data = vec![0x08, 10, 0x12, 2, b'h', b'i'];
        let fields = parse_protobuf(&data);
        assert_eq!(fields.len(), 2);
        assert_eq!(fields[0].data.as_u64(), 10);
        assert_eq!(fields[1].data.as_str(), "hi");
    }

    #[test]
    fn test_onnx_data_type_mapping() {
        assert_eq!(onnx_data_type_to_str(1), "f32");
        assert_eq!(onnx_data_type_to_str(7), "i64");
        assert_eq!(onnx_data_type_to_str(10), "f16");
        assert_eq!(onnx_data_type_to_str(11), "f64");
    }

    #[test]
    fn test_from_onnx_protobuf_minimal() {
        // Build a minimal ONNX model protobuf by hand:
        // ModelProto { graph: GraphProto { name: "test", nodes: [] } }

        // GraphProto: field 2 = "test"
        let graph_name = vec![0x12, 4, b't', b'e', b's', b't'];

        // ModelProto: field 7 = graph_name (length-delimited)
        let mut model = vec![0x3A]; // (7 << 3) | 2 = 58 = 0x3A
        model.push(graph_name.len() as u8);
        model.extend_from_slice(&graph_name);

        let result = from_onnx_protobuf(&model);
        assert!(result.is_ok(), "Failed to parse minimal protobuf: {:?}", result.err());
        let graph = result.unwrap();
        assert_eq!(graph.name, "test");
        assert!(graph.nodes.is_empty());
    }

    #[test]
    fn test_parse_packed_f32s() {
        let val1: f32 = 1.0;
        let val2: f32 = -2.5;
        let val3: f32 = 3.14;
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&val1.to_le_bytes());
        bytes.extend_from_slice(&val2.to_le_bytes());
        bytes.extend_from_slice(&val3.to_le_bytes());

        let floats = parse_packed_f32s(&bytes);
        assert_eq!(floats.len(), 3);
        assert_eq!(floats[0], 1.0);
        assert_eq!(floats[1], -2.5);
        assert!((floats[2] - 3.14).abs() < 1e-6);
    }

    #[test]
    fn test_parse_packed_f32s_partial_trailing() {
        // 5 bytes: one full f32 + 1 trailing byte (ignored by chunks_exact)
        let val: f32 = 42.0;
        let mut bytes = val.to_le_bytes().to_vec();
        bytes.push(0xFF); // trailing garbage
        let floats = parse_packed_f32s(&bytes);
        assert_eq!(floats.len(), 1);
        assert_eq!(floats[0], 42.0);
    }

    #[test]
    fn test_parse_packed_f32s_empty() {
        let floats = parse_packed_f32s(&[]);
        assert!(floats.is_empty());
    }

    /// Helper: encode a protobuf varint into bytes.
    fn encode_varint(mut val: u64) -> Vec<u8> {
        let mut buf = Vec::new();
        loop {
            let mut byte = (val & 0x7F) as u8;
            val >>= 7;
            if val != 0 {
                byte |= 0x80;
            }
            buf.push(byte);
            if val == 0 {
                break;
            }
        }
        buf
    }

    /// Helper: encode a protobuf field with length-delimited data.
    fn encode_ld_field(field_number: u32, data: &[u8]) -> Vec<u8> {
        let tag = (field_number << 3) | 2;
        let mut buf = encode_varint(tag as u64);
        buf.extend(encode_varint(data.len() as u64));
        buf.extend_from_slice(data);
        buf
    }

    /// Helper: encode a protobuf varint field.
    fn encode_varint_field(field_number: u32, value: u64) -> Vec<u8> {
        let tag = (field_number << 3) | 0;
        let mut buf = encode_varint(tag as u64);
        buf.extend(encode_varint(value));
        buf
    }

    /// Build a minimal ONNX model protobuf containing an initializer with weight data.
    fn build_model_with_initializer(tensor_proto: &[u8]) -> Vec<u8> {
        // GraphProto: field 2 = name "g", field 5 = initializer (TensorProto)
        let mut graph = Vec::new();
        graph.extend(encode_ld_field(2, b"g"));          // name
        graph.extend(encode_ld_field(5, tensor_proto));   // initializer

        // ModelProto: field 7 = graph
        encode_ld_field(7, &graph)
    }

    #[test]
    fn test_weight_extraction_float_data_packed() {
        // Build TensorProto with:
        //   field 1 = dim 3 (varint)
        //   field 2 = data_type 1 (FLOAT)
        //   field 4 = float_data packed (3 floats as length-delimited bytes)
        //   field 8 = name "w1"
        let floats: [f32; 3] = [1.0, 2.0, 3.0];
        let mut packed = Vec::new();
        for f in &floats {
            packed.extend_from_slice(&f.to_le_bytes());
        }

        let mut tensor = Vec::new();
        tensor.extend(encode_varint_field(1, 3));         // dims = [3]
        tensor.extend(encode_varint_field(2, 1));         // data_type = FLOAT
        tensor.extend(encode_ld_field(4, &packed));       // float_data (packed)
        tensor.extend(encode_ld_field(8, b"w1"));         // name

        let model = build_model_with_initializer(&tensor);
        let graph = from_onnx_protobuf(&model).expect("parse model");

        assert_eq!(graph.initializers.len(), 1);
        let init = &graph.initializers[0];
        assert_eq!(init.name, "w1");
        assert_eq!(init.shape, vec![3]);
        assert_eq!(init.elem_type, "f32");

        let weights = init.weights.as_ref().expect("weights should be present");
        assert_eq!(weights.len(), 3);
        assert_eq!(weights[0], 1.0);
        assert_eq!(weights[1], 2.0);
        assert_eq!(weights[2], 3.0);
    }

    #[test]
    fn test_weight_extraction_raw_data() {
        // Build TensorProto with raw_data (field 13) instead of float_data
        let floats: [f32; 2] = [-1.5, 0.25];
        let mut raw = Vec::new();
        for f in &floats {
            raw.extend_from_slice(&f.to_le_bytes());
        }

        let mut tensor = Vec::new();
        tensor.extend(encode_varint_field(1, 2));         // dims = [2]
        tensor.extend(encode_varint_field(2, 1));         // data_type = FLOAT
        tensor.extend(encode_ld_field(8, b"w2"));         // name
        tensor.extend(encode_ld_field(13, &raw));         // raw_data

        let model = build_model_with_initializer(&tensor);
        let graph = from_onnx_protobuf(&model).expect("parse model");

        let init = &graph.initializers[0];
        assert_eq!(init.name, "w2");
        let weights = init.weights.as_ref().expect("weights from raw_data");
        assert_eq!(weights.len(), 2);
        assert_eq!(weights[0], -1.5);
        assert_eq!(weights[1], 0.25);
    }

    #[test]
    fn test_weight_extraction_float_data_priority_over_raw_data() {
        // When both float_data and raw_data are present, float_data wins.
        let float_vals: [f32; 1] = [7.0];
        let raw_vals: [f32; 1] = [9.0];
        let mut packed = Vec::new();
        for f in &float_vals {
            packed.extend_from_slice(&f.to_le_bytes());
        }
        let mut raw = Vec::new();
        for f in &raw_vals {
            raw.extend_from_slice(&f.to_le_bytes());
        }

        let mut tensor = Vec::new();
        tensor.extend(encode_varint_field(1, 1));
        tensor.extend(encode_varint_field(2, 1));
        tensor.extend(encode_ld_field(4, &packed));       // float_data
        tensor.extend(encode_ld_field(8, b"w3"));
        tensor.extend(encode_ld_field(13, &raw));         // raw_data

        let model = build_model_with_initializer(&tensor);
        let graph = from_onnx_protobuf(&model).expect("parse model");

        let weights = graph.initializers[0].weights.as_ref().unwrap();
        assert_eq!(weights, &[7.0]); // float_data wins
    }

    #[test]
    fn test_no_weight_data_yields_none() {
        // TensorProto with only name/shape/type, no float_data or raw_data
        let mut tensor = Vec::new();
        tensor.extend(encode_varint_field(1, 4));
        tensor.extend(encode_varint_field(2, 1));
        tensor.extend(encode_ld_field(8, b"empty"));

        let model = build_model_with_initializer(&tensor);
        let graph = from_onnx_protobuf(&model).expect("parse model");

        assert!(graph.initializers[0].weights.is_none());
    }

    #[test]
    fn test_total_weight_bytes() {
        let graph = OnnxGraph {
            name: "test".into(),
            nodes: vec![],
            inputs: vec![],
            outputs: vec![],
            initializers: vec![
                OnnxInitializer {
                    name: "a".into(),
                    elem_type: "f32".into(),
                    shape: vec![3],
                    data_base64: None,
                    weights: Some(vec![1.0, 2.0, 3.0]),
                },
                OnnxInitializer {
                    name: "b".into(),
                    elem_type: "f32".into(),
                    shape: vec![2],
                    data_base64: None,
                    weights: Some(vec![4.0, 5.0]),
                },
                OnnxInitializer {
                    name: "c".into(),
                    elem_type: "f32".into(),
                    shape: vec![10],
                    data_base64: None,
                    weights: None, // no data loaded
                },
            ],
        };
        // 3 floats + 2 floats = 5 floats * 4 bytes = 20 bytes
        assert_eq!(graph.total_weight_bytes(), 20);
    }

    #[test]
    fn test_total_weight_bytes_empty() {
        let graph = OnnxGraph {
            name: "empty".into(),
            nodes: vec![],
            inputs: vec![],
            outputs: vec![],
            initializers: vec![],
        };
        assert_eq!(graph.total_weight_bytes(), 0);
    }
}
