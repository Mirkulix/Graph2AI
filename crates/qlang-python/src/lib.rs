//! QLANG Python Bindings — PyO3 bridge for Python developers.
//!
//! Exposes the QLANG graph-based AI programming language to Python:
//!   - Build computation graphs with `Graph`
//!   - Execute graphs with the QLANG runtime
//!   - Compress weights with IGQK ternary compression
//!   - Train simple MLPs

use std::collections::HashMap;

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

use qlang_core::graph::Graph as CoreGraph;
use qlang_core::ops::Op;
use qlang_core::serial;
use qlang_core::tensor::{Dim, Dtype, Shape, TensorData, TensorType};
use qlang_runtime::executor;
use qlang_runtime::training::MlpWeights;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Parse a dtype string like "f32", "f64", "i32", etc.
fn parse_dtype(s: &str) -> PyResult<Dtype> {
    match s {
        "f16" => Ok(Dtype::F16),
        "f32" => Ok(Dtype::F32),
        "f64" => Ok(Dtype::F64),
        "i8" => Ok(Dtype::I8),
        "i16" => Ok(Dtype::I16),
        "i32" => Ok(Dtype::I32),
        "i64" => Ok(Dtype::I64),
        "bool" => Ok(Dtype::Bool),
        "ternary" => Ok(Dtype::Ternary),
        other => Err(PyRuntimeError::new_err(format!("unknown dtype: {other}"))),
    }
}

/// Build a `TensorType` from dtype string + shape list.
fn make_tensor_type(dtype: &str, shape: &[usize]) -> PyResult<TensorType> {
    Ok(TensorType::new(
        parse_dtype(dtype)?,
        Shape(shape.iter().map(|&d| Dim::Fixed(d)).collect()),
    ))
}

/// Look up the output TensorType for a given node id in the graph.
fn node_output_type(graph: &CoreGraph, node_id: u32) -> PyResult<TensorType> {
    let node = graph
        .node(node_id)
        .ok_or_else(|| PyRuntimeError::new_err(format!("node {node_id} not found")))?;
    node.output_types
        .first()
        .cloned()
        .ok_or_else(|| PyRuntimeError::new_err(format!("node {node_id} has no output type")))
}

// ---------------------------------------------------------------------------
// PyGraph — the main Python-facing class
// ---------------------------------------------------------------------------

/// A QLANG computation graph.
///
/// Build graphs by adding input, operation, and output nodes, then execute
/// them or serialize to JSON / human-readable text.
#[pyclass(name = "Graph")]
struct PyGraph {
    inner: CoreGraph,
}

#[pymethods]
impl PyGraph {
    /// Create a new empty graph with the given name.
    #[new]
    fn new(name: &str) -> Self {
        Self {
            inner: CoreGraph::new(name),
        }
    }

    /// Add an input node.
    ///
    /// Args:
    ///     name: Input name (used as key in `execute` inputs dict).
    ///     dtype: Data type string, e.g. "f32".
    ///     shape: List of dimension sizes, e.g. [4] or [28, 28].
    ///
    /// Returns:
    ///     Node ID (int).
    fn add_input(&mut self, name: &str, dtype: &str, shape: Vec<usize>) -> PyResult<u32> {
        let tt = make_tensor_type(dtype, &shape)?;
        let id = self
            .inner
            .add_node(Op::Input { name: name.into() }, vec![], vec![tt]);
        Ok(id)
    }

    /// Add a matrix multiplication node.
    ///
    /// Both `a` and `b` must refer to nodes producing 2-D tensors.
    ///
    /// Returns:
    ///     Node ID of the matmul result.
    fn add_matmul(&mut self, a: u32, b: u32) -> PyResult<u32> {
        let a_type = node_output_type(&self.inner, a)?;
        let b_type = node_output_type(&self.inner, b)?;

        // Compute output shape: [m, k] x [k, n] -> [m, n]
        let (m, _k, n) = match (&a_type.shape.0[..], &b_type.shape.0[..]) {
            ([Dim::Fixed(m), Dim::Fixed(_k)], [Dim::Fixed(_k2), Dim::Fixed(n)]) => (*m, *_k, *n),
            _ => {
                return Err(PyRuntimeError::new_err(
                    "matmul requires two 2-D tensors",
                ))
            }
        };
        let out_type = TensorType::new(a_type.dtype, Shape::matrix(m, n));

        let id = self.inner.add_node(
            Op::MatMul,
            vec![a_type.clone(), b_type.clone()],
            vec![out_type.clone()],
        );
        self.inner.add_edge(a, 0, id, 0, a_type);
        self.inner.add_edge(b, 0, id, 1, b_type);
        Ok(id)
    }

    /// Add a ReLU activation node.
    ///
    /// Returns:
    ///     Node ID of the relu result.
    fn add_relu(&mut self, input: u32) -> PyResult<u32> {
        let tt = node_output_type(&self.inner, input)?;
        let id = self
            .inner
            .add_node(Op::Relu, vec![tt.clone()], vec![tt.clone()]);
        self.inner.add_edge(input, 0, id, 0, tt);
        Ok(id)
    }

    /// Add an element-wise addition node.
    ///
    /// Returns:
    ///     Node ID of the add result.
    fn add_add(&mut self, a: u32, b: u32) -> PyResult<u32> {
        let tt = node_output_type(&self.inner, a)?;
        let id = self
            .inner
            .add_node(Op::Add, vec![tt.clone(), tt.clone()], vec![tt.clone()]);
        self.inner.add_edge(a, 0, id, 0, tt.clone());
        self.inner.add_edge(b, 0, id, 1, tt);
        Ok(id)
    }

    /// Add a softmax node (axis 0 by default, treating tensor as 1-D).
    ///
    /// Returns:
    ///     Node ID of the softmax result.
    fn add_softmax(&mut self, input: u32) -> PyResult<u32> {
        let tt = node_output_type(&self.inner, input)?;
        let id = self.inner.add_node(
            Op::Softmax { axis: 0 },
            vec![tt.clone()],
            vec![tt.clone()],
        );
        self.inner.add_edge(input, 0, id, 0, tt);
        Ok(id)
    }

    /// Add an IGQK ternary compression node.
    ///
    /// Projects continuous weights to {-1, 0, +1}.
    ///
    /// Returns:
    ///     Node ID of the compressed result.
    fn add_to_ternary(&mut self, input: u32) -> PyResult<u32> {
        let input_type = node_output_type(&self.inner, input)?;
        let output_type = TensorType::new(Dtype::Ternary, input_type.shape.clone());
        let id = self.inner.add_node(
            Op::ToTernary,
            vec![input_type.clone()],
            vec![output_type],
        );
        self.inner.add_edge(input, 0, id, 0, input_type);
        Ok(id)
    }

    /// Add an output node.
    ///
    /// Args:
    ///     name: Output name (used as key in `execute` result dict).
    ///     source: Node ID whose output feeds this output.
    ///
    /// Returns:
    ///     Node ID of the output node.
    fn add_output(&mut self, name: &str, source: u32) -> PyResult<u32> {
        let tt = node_output_type(&self.inner, source)?;
        let id = self.inner.add_node(
            Op::Output { name: name.into() },
            vec![tt.clone()],
            vec![],
        );
        self.inner.add_edge(source, 0, id, 0, tt);
        Ok(id)
    }

    /// Execute the graph.
    ///
    /// Args:
    ///     inputs: Dict mapping input names to lists of floats.
    ///
    /// Returns:
    ///     Dict mapping output names to lists of floats.
    fn execute(&self, inputs: HashMap<String, Vec<f32>>) -> PyResult<HashMap<String, Vec<f32>>> {
        // Convert Python inputs to TensorData
        let mut tensor_inputs: HashMap<String, TensorData> = HashMap::new();
        for (name, values) in &inputs {
            // Find the input node to get its shape
            let input_node = self
                .inner
                .input_nodes()
                .into_iter()
                .find(|n| matches!(&n.op, Op::Input { name: n_name } if n_name == name))
                .ok_or_else(|| {
                    PyRuntimeError::new_err(format!("no input node named '{name}'"))
                })?;
            let shape = input_node
                .output_types
                .first()
                .map(|t| t.shape.clone())
                .unwrap_or_else(|| Shape(vec![Dim::Fixed(values.len())]));
            tensor_inputs.insert(name.clone(), TensorData::from_f32(shape, values));
        }

        // Execute
        let result = executor::execute(&self.inner, tensor_inputs)
            .map_err(|e| PyRuntimeError::new_err(format!("execution error: {e}")))?;

        // Convert outputs back to Python
        let mut py_outputs: HashMap<String, Vec<f32>> = HashMap::new();
        for (name, data) in &result.outputs {
            let values = data
                .as_f32_slice()
                .unwrap_or_default();
            py_outputs.insert(name.clone(), values);
        }
        Ok(py_outputs)
    }

    /// Serialize the graph to JSON.
    fn to_json(&self) -> PyResult<String> {
        serial::to_json(&self.inner)
            .map_err(|e| PyRuntimeError::new_err(format!("serialization error: {e}")))
    }

    /// Human-readable text representation of the graph.
    fn to_qlang_text(&self) -> String {
        format!("{}", self.inner)
    }

    /// Number of nodes in the graph.
    fn num_nodes(&self) -> usize {
        self.inner.nodes.len()
    }

    /// Verify graph validity (types, connections, acyclicity).
    ///
    /// Returns:
    ///     True if the graph is valid.
    fn verify(&self) -> bool {
        self.inner.validate().is_ok()
    }

    fn __repr__(&self) -> String {
        format!(
            "Graph('{}', nodes={}, edges={})",
            self.inner.id,
            self.inner.nodes.len(),
            self.inner.edges.len()
        )
    }

    fn __str__(&self) -> String {
        format!("{}", self.inner)
    }
}

// ---------------------------------------------------------------------------
// Standalone functions
// ---------------------------------------------------------------------------

/// Compress a list of f32 weights to ternary {-1.0, 0.0, +1.0}.
///
/// Uses threshold-based projection: threshold = mean(|w|) * 0.7
/// (IGQK Theorem 5.2 bounds the distortion of this projection).
///
/// Args:
///     weights: List of float weights.
///
/// Returns:
///     List of floats, each being -1.0, 0.0, or +1.0.
#[pyfunction]
fn compress_ternary(weights: Vec<f32>) -> Vec<f32> {
    let mean_abs: f32 = weights.iter().map(|x| x.abs()).sum::<f32>() / weights.len().max(1) as f32;
    let threshold = mean_abs * 0.7;
    weights
        .iter()
        .map(|&x| {
            if x > threshold {
                1.0
            } else if x < -threshold {
                -1.0
            } else {
                0.0
            }
        })
        .collect()
}

/// Train a simple MLP (multi-layer perceptron).
///
/// Uses numerical gradient descent with cross-entropy loss.
///
/// Args:
///     x: Flat list of input features (batch_size * input_dim).
///     y: Flat list of target labels (batch_size * output_dim, used to infer output_dim).
///     targets: List of integer target class indices (one per sample).
///     epochs: Number of training epochs.
///     lr: Learning rate.
///
/// Returns:
///     Dict with keys:
///       - "final_loss": float
///       - "final_accuracy": float
///       - "weights_w1": list[float]
///       - "weights_w2": list[float]
///       - "param_count": int (as float)
#[pyfunction]
fn train_mlp(
    x: Vec<f32>,
    y: Vec<f32>,
    targets: Vec<u8>,
    epochs: usize,
    lr: f32,
) -> PyResult<HashMap<String, Vec<f32>>> {
    let batch_size = targets.len();
    if batch_size == 0 {
        return Err(PyRuntimeError::new_err("targets must not be empty"));
    }
    let input_dim = x.len() / batch_size;
    let output_dim = y.len() / batch_size;
    if input_dim == 0 || output_dim == 0 {
        return Err(PyRuntimeError::new_err(
            "could not infer input_dim or output_dim from data sizes",
        ));
    }

    let hidden_dim = ((input_dim + output_dim) / 2).max(4);
    let mut model = MlpWeights::new(input_dim, hidden_dim, output_dim);

    let mut final_loss = 0.0f32;
    for _epoch in 0..epochs {
        final_loss = model.train_step(&x, &targets, lr);
    }

    let probs = model.forward(&x);
    let final_accuracy = model.accuracy(&probs, &targets);

    let mut result: HashMap<String, Vec<f32>> = HashMap::new();
    result.insert("final_loss".into(), vec![final_loss]);
    result.insert("final_accuracy".into(), vec![final_accuracy]);
    result.insert("weights_w1".into(), model.w1.clone());
    result.insert("weights_w2".into(), model.w2.clone());
    result.insert("param_count".into(), vec![model.param_count() as f32]);
    Ok(result)
}

// ---------------------------------------------------------------------------
// Module definition
// ---------------------------------------------------------------------------

/// QLANG — Graph-based AI-to-AI programming language (Python bindings).
#[pymodule]
fn qlang(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyGraph>()?;
    m.add_function(wrap_pyfunction!(compress_ternary, m)?)?;
    m.add_function(wrap_pyfunction!(train_mlp, m)?)?;
    Ok(())
}
