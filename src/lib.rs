//! QLANG — Graph-based AI-to-AI Programming Language
//!
//! A programming language where programs are directed acyclic graphs (DAGs),
//! not text. Designed for AI systems to communicate computations directly,
//! without the lossy detour through human-readable syntax.
//!
//! Built on IGQK (Information-Geometric Quantum Compression) theory.

pub use qlang_core as core;
pub use qlang_runtime as runtime;
pub use qlang_agent as agent;

/// C FFI — execute a QLANG graph via the runtime.
///
/// This bridges `qlang-core` (data structures) with `qlang-runtime` (execution).
/// The matching stub in `qlang_core::ffi` returns `-1`; link against the full
/// `qlang` crate to get working execution.
pub mod ffi_runtime {
    use std::collections::HashMap;
    use std::ffi::CStr;
    use std::os::raw::c_char;

    use qlang_core::graph::Graph;
    use qlang_core::tensor::{Dim, Shape, TensorData};

    /// Execute a graph with the provided f32 inputs and write results to the
    /// output buffer.
    ///
    /// - `graph`: pointer to a `Graph` created with `qlang_graph_new`.
    /// - `input_names`: array of C strings, one per input tensor.
    /// - `input_data`: array of `*const f32` pointers to input data.
    /// - `input_lens`: array of lengths for each input data pointer.
    /// - `n_inputs`: number of inputs.
    /// - `output_data`: caller-allocated buffer for output f32 values.
    /// - `output_len`: length of the output buffer.
    ///
    /// Returns 0 on success, -1 on null/invalid args, -2 on execution error.
    #[unsafe(no_mangle)]
    pub unsafe extern "C" fn qlang_runtime_execute(
        graph: *mut Graph,
        input_names: *const *const c_char,
        input_data: *const *const f32,
        input_lens: *const usize,
        n_inputs: usize,
        output_data: *mut f32,
        output_len: usize,
    ) -> i32 {
        if graph.is_null() || input_names.is_null() || input_data.is_null()
            || input_lens.is_null() || output_data.is_null()
        {
            return -1;
        }

        let graph = unsafe { &*graph };

        // Build inputs HashMap
        let mut inputs: HashMap<String, TensorData> = HashMap::new();
        for i in 0..n_inputs {
            let name_ptr = unsafe { *input_names.add(i) };
            if name_ptr.is_null() {
                return -1;
            }
            let name = match unsafe { CStr::from_ptr(name_ptr) }.to_str() {
                Ok(s) => s.to_string(),
                Err(_) => return -1,
            };
            let data_ptr = unsafe { *input_data.add(i) };
            let len = unsafe { *input_lens.add(i) };
            if data_ptr.is_null() {
                return -1;
            }
            let values = unsafe { std::slice::from_raw_parts(data_ptr, len) };
            let td = TensorData::from_f32(Shape(vec![Dim::Fixed(len)]), values);
            inputs.insert(name, td);
        }

        // Execute
        let result = match qlang_runtime::executor::execute(graph, inputs) {
            Ok(r) => r,
            Err(_) => return -2,
        };

        // Write first output to buffer
        let mut written = 0usize;
        for (_name, tensor) in &result.outputs {
            if let Some(values) = tensor.as_f32_slice() {
                for &v in &values {
                    if written >= output_len {
                        break;
                    }
                    unsafe { *output_data.add(written) = v };
                    written += 1;
                }
            }
        }

        0
    }
}
