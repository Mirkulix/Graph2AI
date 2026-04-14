//! C FFI bindings for QLANG — embed QLANG graphs in C/C++ programs.
//!
//! All functions use `extern "C"` with `#[unsafe(no_mangle)]` and operate on
//! raw pointers and C-compatible types only.

use std::ffi::{CStr, CString};
use std::os::raw::c_char;
use std::ptr;
use std::slice;

use crate::graph::Graph;
use crate::ops::Op;
use crate::serial;
use crate::tensor::{Dim, Dtype, Shape, TensorType};
use crate::verify;

/// Create a new QLANG graph with the given name.
///
/// The caller must free the returned graph with `qlang_graph_free`.
/// Returns null on invalid input.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qlang_graph_new(name: *const c_char) -> *mut Graph {
    if name.is_null() {
        return ptr::null_mut();
    }
    let name_str = match unsafe { CStr::from_ptr(name) }.to_str() {
        Ok(s) => s,
        Err(_) => return ptr::null_mut(),
    };
    let graph = Graph::new(name_str);
    Box::into_raw(Box::new(graph))
}

/// Free a graph previously created by `qlang_graph_new`.
///
/// Passing null is a no-op.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qlang_graph_free(graph: *mut Graph) {
    if !graph.is_null() {
        drop(unsafe { Box::from_raw(graph) });
    }
}

/// Add an input node to the graph.
///
/// - `name`: C string for the input name.
/// - `dtype`: integer encoding the data type (0=F16, 1=F32, 2=F64, 3=I8,
///   4=I16, 5=I32, 6=I64, 7=Bool, 8=Ternary).
/// - `shape_ptr`: pointer to an array of dimension sizes.
/// - `shape_len`: number of dimensions.
///
/// Returns the node ID of the new input node, or `u32::MAX` on error.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qlang_graph_add_input(
    graph: *mut Graph,
    name: *const c_char,
    dtype: u32,
    shape_ptr: *const usize,
    shape_len: usize,
) -> u32 {
    if graph.is_null() || name.is_null() {
        return u32::MAX;
    }
    let graph = unsafe { &mut *graph };
    let name_str = match unsafe { CStr::from_ptr(name) }.to_str() {
        Ok(s) => s,
        Err(_) => return u32::MAX,
    };
    let dt = match dtype_from_u32(dtype) {
        Some(d) => d,
        None => return u32::MAX,
    };
    let dims: Vec<Dim> = if shape_ptr.is_null() || shape_len == 0 {
        vec![]
    } else {
        unsafe { slice::from_raw_parts(shape_ptr, shape_len) }
            .iter()
            .map(|&n| Dim::Fixed(n))
            .collect()
    };
    let tt = TensorType::new(dt, Shape(dims));
    graph.add_node(Op::Input { name: name_str.to_string() }, vec![], vec![tt])
}

/// Add an operation node to the graph.
///
/// - `op_name`: C string naming the operation (e.g. "add", "relu", "matmul").
/// - `input_a`, `input_b`: node IDs of inputs. Pass `u32::MAX` for unused
///   second input on unary ops.
///
/// This creates the node and connecting edges with a default f32-scalar type.
/// Returns the new node ID, or `u32::MAX` on error.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qlang_graph_add_op(
    graph: *mut Graph,
    op_name: *const c_char,
    input_a: u32,
    input_b: u32,
) -> u32 {
    if graph.is_null() || op_name.is_null() {
        return u32::MAX;
    }
    let graph = unsafe { &mut *graph };
    let op_str = match unsafe { CStr::from_ptr(op_name) }.to_str() {
        Ok(s) => s,
        Err(_) => return u32::MAX,
    };
    let op = match op_from_str(op_str) {
        Some(o) => o,
        None => return u32::MAX,
    };
    let n_inputs = op.n_inputs();
    let placeholder = TensorType::f32_scalar();

    let input_types: Vec<TensorType> = (0..n_inputs).map(|_| placeholder.clone()).collect();
    let output_types = if op.n_outputs() > 0 {
        vec![placeholder.clone()]
    } else {
        vec![]
    };

    let node_id = graph.add_node(op, input_types, output_types);

    // Wire edges from input nodes
    if n_inputs >= 1 && input_a != u32::MAX {
        graph.add_edge(input_a, 0, node_id, 0, placeholder.clone());
    }
    if n_inputs >= 2 && input_b != u32::MAX {
        graph.add_edge(input_b, 0, node_id, 1, placeholder.clone());
    }

    node_id
}

/// Add an output node to the graph.
///
/// - `name`: C string for the output name.
/// - `source`: node ID whose output feeds this output node.
///
/// Returns the new node ID, or `u32::MAX` on error.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qlang_graph_add_output(
    graph: *mut Graph,
    name: *const c_char,
    source: u32,
) -> u32 {
    if graph.is_null() || name.is_null() {
        return u32::MAX;
    }
    let graph = unsafe { &mut *graph };
    let name_str = match unsafe { CStr::from_ptr(name) }.to_str() {
        Ok(s) => s,
        Err(_) => return u32::MAX,
    };
    let placeholder = TensorType::f32_scalar();
    let node_id = graph.add_node(
        Op::Output { name: name_str.to_string() },
        vec![placeholder.clone()],
        vec![],
    );
    if source != u32::MAX {
        graph.add_edge(source, 0, node_id, 0, placeholder);
    }
    node_id
}

/// Execute a graph with the provided inputs.
///
/// This is a **stub** — full execution requires the runtime crate. Currently
/// returns `-1` (not implemented). The signature is provided so that C code
/// can link against the API today and execution can be filled in later.
///
/// - `input_names`: array of C strings, one per input tensor.
/// - `input_data`: array of `*const f32` pointers to input data.
/// - `n_inputs`: number of inputs.
/// - `output_data`: pointer to caller-allocated buffer for output f32 values.
/// - `output_len`: length of the output buffer.
///
/// Returns 0 on success, negative on error.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qlang_graph_execute(
    _graph: *mut Graph,
    _input_names: *const *const c_char,
    _input_data: *const *const f32,
    _n_inputs: usize,
    _output_data: *mut f32,
    _output_len: usize,
) -> i32 {
    // Stub: execution requires qlang-runtime integration
    -1
}

/// Serialize the graph to JSON. The returned string must be freed with
/// `qlang_free_string`. Returns null on error.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qlang_graph_to_json(graph: *mut Graph) -> *mut c_char {
    if graph.is_null() {
        return ptr::null_mut();
    }
    let graph = unsafe { &*graph };
    match serial::to_json(graph) {
        Ok(json) => match CString::new(json) {
            Ok(cs) => cs.into_raw(),
            Err(_) => ptr::null_mut(),
        },
        Err(_) => ptr::null_mut(),
    }
}

/// Free a string previously returned by `qlang_graph_to_json` or
/// `qlang_version`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qlang_free_string(s: *mut c_char) {
    if !s.is_null() {
        drop(unsafe { CString::from_raw(s) });
    }
}

/// Return the number of nodes in the graph, or 0 if the pointer is null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qlang_graph_num_nodes(graph: *mut Graph) -> u32 {
    if graph.is_null() {
        return 0;
    }
    let graph = unsafe { &*graph };
    graph.nodes.len() as u32
}

/// Verify the graph. Returns 0 if verification passes, -1 on null pointer,
/// or a positive integer equal to the number of verification failures.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qlang_graph_verify(graph: *mut Graph) -> i32 {
    if graph.is_null() {
        return -1;
    }
    let graph = unsafe { &*graph };
    let result = verify::verify_graph(graph);
    if result.is_ok() {
        0
    } else {
        result.failed.len() as i32
    }
}

/// Return the QLANG version string. The returned pointer is to a static
/// string and must NOT be freed.
#[unsafe(no_mangle)]
pub extern "C" fn qlang_version() -> *const c_char {
    // Static null-terminated string — lives for the entire program.
    static VERSION: &[u8] = b"0.1.0\0";
    VERSION.as_ptr() as *const c_char
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

fn dtype_from_u32(v: u32) -> Option<Dtype> {
    match v {
        0 => Some(Dtype::F16),
        1 => Some(Dtype::F32),
        2 => Some(Dtype::F64),
        3 => Some(Dtype::I8),
        4 => Some(Dtype::I16),
        5 => Some(Dtype::I32),
        6 => Some(Dtype::I64),
        7 => Some(Dtype::Bool),
        8 => Some(Dtype::Ternary),
        9 => Some(Dtype::Utf8),
        _ => None,
    }
}

fn op_from_str(s: &str) -> Option<Op> {
    match s {
        "add" => Some(Op::Add),
        "sub" => Some(Op::Sub),
        "mul" => Some(Op::Mul),
        "div" => Some(Op::Div),
        "neg" => Some(Op::Neg),
        "matmul" => Some(Op::MatMul),
        "transpose" => Some(Op::Transpose),
        "relu" => Some(Op::Relu),
        "sigmoid" => Some(Op::Sigmoid),
        "tanh" => Some(Op::Tanh),
        "to_ternary" => Some(Op::ToTernary),
        "entropy" => Some(Op::Entropy),
        "collapse" => Some(Op::Collapse),
        "superpose" => Some(Op::Superpose),
        "measure" => Some(Op::Measure),
        "entangle" => Some(Op::Entangle),
        "fisher_metric" => Some(Op::FisherMetric),
        "residual" => Some(Op::Residual),
        "gelu" => Some(Op::Gelu),
        "constant" => Some(Op::Constant),
        "ollama_generate" => Some(Op::OllamaGenerate { model: "llama3".into() }),
        "ollama_chat" => Some(Op::OllamaChat { model: "llama3".into() }),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::CString;

    #[test]
    fn test_create_and_free_graph() {
        let name = CString::new("test_graph").unwrap();
        unsafe {
            let g = qlang_graph_new(name.as_ptr());
            assert!(!g.is_null());
            assert_eq!(qlang_graph_num_nodes(g), 0);
            qlang_graph_free(g);
        }
    }

    #[test]
    fn test_null_name_returns_null() {
        unsafe {
            let g = qlang_graph_new(ptr::null());
            assert!(g.is_null());
        }
    }

    #[test]
    fn test_add_nodes() {
        let name = CString::new("node_test").unwrap();
        let input_name = CString::new("x").unwrap();
        let op_name = CString::new("relu").unwrap();
        let out_name = CString::new("y").unwrap();
        let shape: [usize; 1] = [4];

        unsafe {
            let g = qlang_graph_new(name.as_ptr());

            let inp = qlang_graph_add_input(g, input_name.as_ptr(), 1, shape.as_ptr(), 1);
            assert_eq!(inp, 0);

            let relu = qlang_graph_add_op(g, op_name.as_ptr(), inp, u32::MAX);
            assert_eq!(relu, 1);

            let out = qlang_graph_add_output(g, out_name.as_ptr(), relu);
            assert_eq!(out, 2);

            assert_eq!(qlang_graph_num_nodes(g), 3);
            qlang_graph_free(g);
        }
    }

    #[test]
    fn test_num_nodes() {
        let name = CString::new("count_test").unwrap();
        let inp_a = CString::new("a").unwrap();
        let inp_b = CString::new("b").unwrap();
        let shape: [usize; 1] = [8];

        unsafe {
            let g = qlang_graph_new(name.as_ptr());
            assert_eq!(qlang_graph_num_nodes(g), 0);

            qlang_graph_add_input(g, inp_a.as_ptr(), 1, shape.as_ptr(), 1);
            assert_eq!(qlang_graph_num_nodes(g), 1);

            qlang_graph_add_input(g, inp_b.as_ptr(), 1, shape.as_ptr(), 1);
            assert_eq!(qlang_graph_num_nodes(g), 2);

            // null graph returns 0
            assert_eq!(qlang_graph_num_nodes(ptr::null_mut()), 0);

            qlang_graph_free(g);
        }
    }

    #[test]
    fn test_version_string() {
        let v = qlang_version();
        assert!(!v.is_null());
        let s = unsafe { CStr::from_ptr(v) }.to_str().unwrap();
        assert_eq!(s, "0.1.0");
    }

    #[test]
    fn test_to_json() {
        let name = CString::new("json_test").unwrap();
        let input_name = CString::new("x").unwrap();
        let shape: [usize; 1] = [4];

        unsafe {
            let g = qlang_graph_new(name.as_ptr());
            qlang_graph_add_input(g, input_name.as_ptr(), 1, shape.as_ptr(), 1);

            let json_ptr = qlang_graph_to_json(g);
            assert!(!json_ptr.is_null());

            let json_str = CStr::from_ptr(json_ptr).to_str().unwrap();
            assert!(json_str.contains("json_test"));
            assert!(json_str.contains("input"));

            qlang_free_string(json_ptr);
            qlang_graph_free(g);
        }
    }

    #[test]
    fn test_verify_graph() {
        let name = CString::new("verify_test").unwrap();
        let inp_name = CString::new("x").unwrap();
        let op_name = CString::new("relu").unwrap();
        let out_name = CString::new("y").unwrap();

        unsafe {
            let g = qlang_graph_new(name.as_ptr());

            // Use scalar (empty shape) so edge types match the f32_scalar
            // placeholders created by add_op / add_output.
            let inp = qlang_graph_add_input(g, inp_name.as_ptr(), 1, ptr::null(), 0);
            let relu = qlang_graph_add_op(g, op_name.as_ptr(), inp, u32::MAX);
            qlang_graph_add_output(g, out_name.as_ptr(), relu);

            let result = qlang_graph_verify(g);
            assert_eq!(result, 0);

            // null graph returns -1
            assert_eq!(qlang_graph_verify(ptr::null_mut()), -1);

            qlang_graph_free(g);
        }
    }

    #[test]
    fn test_execute_stub_returns_error() {
        let name = CString::new("exec_test").unwrap();
        unsafe {
            let g = qlang_graph_new(name.as_ptr());
            let result = qlang_graph_execute(
                g,
                ptr::null(),
                ptr::null(),
                0,
                ptr::null_mut(),
                0,
            );
            assert_eq!(result, -1);
            qlang_graph_free(g);
        }
    }
}
