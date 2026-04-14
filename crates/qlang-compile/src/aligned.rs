//! Aligned memory allocation for SIMD operations.
//!
//! AVX2 requires 32-byte aligned memory for <8 x float> loads/stores.
//! Standard Vec<f32> only guarantees 4-byte alignment.
//! This module provides AlignedVec which guarantees 32-byte alignment.

use std::alloc::{Layout, alloc, dealloc};
use std::ops::{Deref, DerefMut};

/// A Vec-like container with guaranteed memory alignment.
///
/// For AVX2: 32-byte alignment (8 floats per vector).
/// For AVX-512: 64-byte alignment (16 floats per vector).
pub struct AlignedVec {
    ptr: *mut f32,
    len: usize,
    capacity: usize,
    alignment: usize,
}

impl AlignedVec {
    /// Alignment for AVX2 (256 bits = 32 bytes).
    pub const AVX2_ALIGN: usize = 32;

    /// Alignment for AVX-512 (512 bits = 64 bytes).
    pub const AVX512_ALIGN: usize = 64;

    /// Create a new aligned vector with the given capacity and alignment.
    pub fn with_capacity(capacity: usize, alignment: usize) -> Self {
        assert!(alignment.is_power_of_two());
        assert!(alignment >= std::mem::align_of::<f32>());

        let size = capacity * std::mem::size_of::<f32>();
        let layout = Layout::from_size_align(size.max(alignment), alignment).unwrap();

        let ptr = unsafe { alloc(layout) as *mut f32 };
        if ptr.is_null() {
            panic!("Failed to allocate aligned memory");
        }

        Self {
            ptr,
            len: 0,
            capacity,
            alignment,
        }
    }

    /// Create an aligned vector of zeros.
    pub fn zeros(len: usize, alignment: usize) -> Self {
        let mut v = Self::with_capacity(len, alignment);
        unsafe {
            std::ptr::write_bytes(v.ptr, 0, len);
        }
        v.len = len;
        v
    }

    /// Create an aligned vector from a slice (copies data).
    pub fn from_slice(data: &[f32], alignment: usize) -> Self {
        let mut v = Self::with_capacity(data.len(), alignment);
        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr(), v.ptr, data.len());
        }
        v.len = data.len();
        v
    }

    /// Get a raw pointer to the data (for passing to JIT functions).
    pub fn as_ptr(&self) -> *const f32 {
        self.ptr
    }

    /// Get a mutable raw pointer.
    pub fn as_mut_ptr(&mut self) -> *mut f32 {
        self.ptr
    }

    /// Number of elements.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Verify alignment.
    pub fn is_aligned(&self) -> bool {
        (self.ptr as usize) % self.alignment == 0
    }

    /// Convert to a regular Vec (copies data).
    pub fn to_vec(&self) -> Vec<f32> {
        self.as_slice().to_vec()
    }

    /// Get as slice.
    pub fn as_slice(&self) -> &[f32] {
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }

    /// Get as mutable slice.
    pub fn as_mut_slice(&mut self) -> &mut [f32] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }
}

impl Drop for AlignedVec {
    fn drop(&mut self) {
        if !self.ptr.is_null() && self.capacity > 0 {
            let size = self.capacity * std::mem::size_of::<f32>();
            let layout = Layout::from_size_align(size.max(self.alignment), self.alignment).unwrap();
            unsafe {
                dealloc(self.ptr as *mut u8, layout);
            }
        }
    }
}

impl Deref for AlignedVec {
    type Target = [f32];
    fn deref(&self) -> &[f32] {
        self.as_slice()
    }
}

impl DerefMut for AlignedVec {
    fn deref_mut(&mut self) -> &mut [f32] {
        self.as_mut_slice()
    }
}

// Safety: AlignedVec owns its memory exclusively
unsafe impl Send for AlignedVec {}
unsafe impl Sync for AlignedVec {}

/// Execute a compiled graph with aligned memory.
///
/// This is the SIMD-safe version of `execute_compiled`.
pub fn execute_aligned(
    compiled: &crate::codegen::CompiledGraph,
    input_a: &[f32],
    input_b: &[f32],
) -> Result<Vec<f32>, crate::codegen::CodegenError> {
    use inkwell::execution_engine::JitFunction;

    let n = input_a.len();

    // Allocate aligned memory
    let a_aligned = AlignedVec::from_slice(input_a, AlignedVec::AVX2_ALIGN);
    let b_aligned = AlignedVec::from_slice(input_b, AlignedVec::AVX2_ALIGN);
    let mut out_aligned = AlignedVec::zeros(n, AlignedVec::AVX2_ALIGN);

    debug_assert!(a_aligned.is_aligned(), "input_a not aligned");
    debug_assert!(b_aligned.is_aligned(), "input_b not aligned");
    debug_assert!(out_aligned.is_aligned(), "output not aligned");

    type GraphFn = unsafe extern "C" fn(*const f32, *const f32, *mut f32, u64);

    unsafe {
        let func: JitFunction<GraphFn> = compiled
            .execution_engine
            .get_function(&compiled.function_name)
            .map_err(|e| crate::codegen::CodegenError::LlvmError(e.to_string()))?;

        func.call(
            a_aligned.as_ptr(),
            b_aligned.as_ptr(),
            out_aligned.as_mut_ptr(),
            n as u64,
        );
    }

    Ok(out_aligned.to_vec())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn aligned_vec_basics() {
        let v = AlignedVec::zeros(16, AlignedVec::AVX2_ALIGN);
        assert_eq!(v.len(), 16);
        assert!(v.is_aligned());
        assert_eq!(v[0], 0.0);
    }

    #[test]
    fn aligned_vec_from_slice() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let v = AlignedVec::from_slice(&data, AlignedVec::AVX2_ALIGN);
        assert!(v.is_aligned());
        assert_eq!(v.to_vec(), data);
    }

    #[test]
    fn aligned_vec_large() {
        let n = 100_000;
        let data: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let v = AlignedVec::from_slice(&data, AlignedVec::AVX2_ALIGN);
        assert!(v.is_aligned());
        assert_eq!(v.len(), n);
        assert_eq!(v[0], 0.0);
        assert_eq!(v[n - 1], (n - 1) as f32);
    }

    #[test]
    fn simd_execution_with_aligned_memory() {
        use qlang_core::graph::Graph;
        use qlang_core::ops::Op;
        use qlang_core::tensor::TensorType;
        use inkwell::context::Context;

        let mut g = Graph::new("aligned_test");
        let a = g.add_node(Op::Input { name: "a".into() }, vec![], vec![TensorType::f32_vector(16)]);
        let b = g.add_node(Op::Input { name: "b".into() }, vec![], vec![TensorType::f32_vector(16)]);
        let add = g.add_node(Op::Add, vec![TensorType::f32_vector(16); 2], vec![TensorType::f32_vector(16)]);
        let out = g.add_node(Op::Output { name: "y".into() }, vec![TensorType::f32_vector(16)], vec![]);
        g.add_edge(a, 0, add, 0, TensorType::f32_vector(16));
        g.add_edge(b, 0, add, 1, TensorType::f32_vector(16));
        g.add_edge(add, 0, out, 0, TensorType::f32_vector(16));

        let context = Context::create();
        let compiled = crate::simd::compile_graph_simd(&context, &g).unwrap();

        let a_data: Vec<f32> = (0..16).map(|i| i as f32).collect();
        let b_data: Vec<f32> = (0..16).map(|i| (i * 10) as f32).collect();
        let result = execute_aligned(&compiled, &a_data, &b_data).unwrap();

        let expected: Vec<f32> = (0..16).map(|i| i as f32 + (i * 10) as f32).collect();
        assert_eq!(result, expected);
    }

    #[test]
    fn simd_relu_aligned() {
        use qlang_core::graph::Graph;
        use qlang_core::ops::Op;
        use qlang_core::tensor::TensorType;
        use inkwell::context::Context;

        let mut g = Graph::new("relu_aligned");
        let a = g.add_node(Op::Input { name: "a".into() }, vec![], vec![TensorType::f32_vector(24)]);
        let _b = g.add_node(Op::Input { name: "b".into() }, vec![], vec![TensorType::f32_vector(24)]);
        let relu = g.add_node(Op::Relu, vec![TensorType::f32_vector(24)], vec![TensorType::f32_vector(24)]);
        let out = g.add_node(Op::Output { name: "y".into() }, vec![TensorType::f32_vector(24)], vec![]);
        g.add_edge(a, 0, relu, 0, TensorType::f32_vector(24));
        g.add_edge(relu, 0, out, 0, TensorType::f32_vector(24));

        let context = Context::create();
        let compiled = crate::simd::compile_graph_simd(&context, &g).unwrap();

        let input: Vec<f32> = (-12..12).map(|i| i as f32).collect();
        let dummy = vec![0.0f32; 24];
        let result = execute_aligned(&compiled, &input, &dummy).unwrap();

        let expected: Vec<f32> = (-12..12).map(|i| (i as f32).max(0.0)).collect();
        assert_eq!(result, expected);
    }

    #[test]
    fn simd_ternary_aligned() {
        use qlang_core::graph::Graph;
        use qlang_core::ops::Op;
        use qlang_core::tensor::TensorType;
        use inkwell::context::Context;

        let mut g = Graph::new("ternary_aligned");
        let a = g.add_node(Op::Input { name: "a".into() }, vec![], vec![TensorType::f32_vector(16)]);
        let _b = g.add_node(Op::Input { name: "b".into() }, vec![], vec![TensorType::f32_vector(16)]);
        let ternary = g.add_node(Op::ToTernary, vec![TensorType::f32_vector(16)], vec![TensorType::f32_vector(16)]);
        let out = g.add_node(Op::Output { name: "y".into() }, vec![TensorType::f32_vector(16)], vec![]);
        g.add_edge(a, 0, ternary, 0, TensorType::f32_vector(16));
        g.add_edge(ternary, 0, out, 0, TensorType::f32_vector(16));

        let context = Context::create();
        let compiled = crate::simd::compile_graph_simd(&context, &g).unwrap();

        let input: Vec<f32> = vec![
            0.5, -0.5, 0.1, -0.1, 0.8, -0.8, 0.0, 0.25,
            -0.25, 0.4, -0.4, 0.05, -0.05, 1.0, -1.0, 0.29,
        ];
        let dummy = vec![0.0f32; 16];
        let result = execute_aligned(&compiled, &input, &dummy).unwrap();

        // Values > 0.3 → +1, < -0.3 → -1, else → 0
        let expected: Vec<f32> = input.iter().map(|&x| {
            if x > 0.3 { 1.0 } else if x < -0.3 { -1.0 } else { 0.0 }
        }).collect();
        assert_eq!(result, expected);
    }
}
