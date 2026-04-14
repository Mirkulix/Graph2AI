//! Apple Accelerate framework (BLAS) support for fast matrix multiplication.
//!
//! On macOS, uses `cblas_sgemm` from the Accelerate framework for hardware-optimized
//! matrix multiply on M1/M2/M3 (NEON + AMX). Falls back to pure-Rust loops on other
//! platforms.

// ---- macOS: Apple Accelerate BLAS bindings ----

#[cfg(target_os = "macos")]
mod platform {
    // CBLAS enums (from Accelerate/vecLib/cblas.h)
    const CBLAS_ROW_MAJOR: i32 = 101;
    const CBLAS_NO_TRANS: i32 = 111;
    const CBLAS_TRANS: i32 = 112;

    unsafe extern "C" {
        /// Single-precision general matrix multiply from Apple Accelerate.
        ///
        /// C = alpha * op(A) * op(B) + beta * C
        ///
        /// Parameters (in CBLAS order):
        ///   order  – CblasRowMajor (101)
        ///   transA – CblasNoTrans (111) or CblasTrans (112)
        ///   transB – CblasNoTrans (111) or CblasTrans (112)
        ///   m      – rows of op(A) and C
        ///   n      – cols of op(B) and C
        ///   k      – cols of op(A) / rows of op(B)
        ///   alpha  – scalar multiplier
        ///   a      – pointer to A
        ///   lda    – leading dimension of A
        ///   b      – pointer to B
        ///   ldb    – leading dimension of B
        ///   beta   – scalar multiplier for C
        ///   c      – pointer to C (output)
        ///   ldc    – leading dimension of C
        fn cblas_sgemm(
            order: i32,
            trans_a: i32,
            trans_b: i32,
            m: i32,
            n: i32,
            k: i32,
            alpha: f32,
            a: *const f32,
            lda: i32,
            b: *const f32,
            ldb: i32,
            beta: f32,
            c: *mut f32,
            ldc: i32,
        );
    }

    /// C = A * B where A is [m, k] and B is [k, n], both row-major.
    pub fn matmul(a: &[f32], b: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
        debug_assert_eq!(a.len(), m * k, "matmul: a must be [m,k]");
        debug_assert_eq!(b.len(), k * n, "matmul: b must be [k,n]");
        let mut c = vec![0.0f32; m * n];
        unsafe {
            cblas_sgemm(
                CBLAS_ROW_MAJOR,
                CBLAS_NO_TRANS,
                CBLAS_NO_TRANS,
                m as i32,
                n as i32,
                k as i32,
                1.0,
                a.as_ptr(),
                k as i32, // lda = k (row-major, no-trans)
                b.as_ptr(),
                n as i32, // ldb = n
                0.0,
                c.as_mut_ptr(),
                n as i32, // ldc = n
            );
        }
        c
    }

    /// C = A^T * B where A is [k, m] (transposed to [m, k]) and B is [k, n].
    /// Used for gradient computations: dW = X^T @ dY.
    pub fn matmul_at_b(a: &[f32], b: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
        // A is stored as [k, m] in memory, we want A^T [m, k] * B [k, n] = C [m, n]
        debug_assert_eq!(a.len(), k * m, "matmul_at_b: a must be [k,m]");
        debug_assert_eq!(b.len(), k * n, "matmul_at_b: b must be [k,n]");
        let mut c = vec![0.0f32; m * n];
        unsafe {
            cblas_sgemm(
                CBLAS_ROW_MAJOR,
                CBLAS_TRANS,   // transpose A
                CBLAS_NO_TRANS,
                m as i32,
                n as i32,
                k as i32,
                1.0,
                a.as_ptr(),
                m as i32, // lda = m (A is [k,m] in memory, transposed)
                b.as_ptr(),
                n as i32,
                0.0,
                c.as_mut_ptr(),
                n as i32,
            );
        }
        c
    }

    /// C = A * B^T where A is [m, k] and B is [n, k] (transposed to [k, n]).
    /// Used for backprop: d_hidden = d_logits @ W^T.
    pub fn matmul_a_bt(a: &[f32], b: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
        // A [m, k] * B^T [k, n] = C [m, n], but B is stored as [n, k]
        debug_assert_eq!(a.len(), m * k, "matmul_a_bt: a must be [m,k]");
        debug_assert_eq!(b.len(), n * k, "matmul_a_bt: b must be [n,k]");
        let mut c = vec![0.0f32; m * n];
        unsafe {
            cblas_sgemm(
                CBLAS_ROW_MAJOR,
                CBLAS_NO_TRANS,
                CBLAS_TRANS, // transpose B
                m as i32,
                n as i32,
                k as i32,
                1.0,
                a.as_ptr(),
                k as i32,
                b.as_ptr(),
                k as i32, // ldb = k (B is [n,k] in memory, transposed)
                0.0,
                c.as_mut_ptr(),
                n as i32,
            );
        }
        c
    }
}

// ---- Fallback: pure Rust for non-macOS ----

#[cfg(not(target_os = "macos"))]
mod platform {
    use rayon::prelude::*;

    /// C = A * B where A is [m, k] and B is [k, n], both row-major.
    pub fn matmul(a: &[f32], b: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
        debug_assert_eq!(a.len(), m * k);
        debug_assert_eq!(b.len(), k * n);
        let mut c = vec![0.0f32; m * n];
        
        // Parallelize over rows of C
        c.par_chunks_exact_mut(n).enumerate().for_each(|(i, c_row)| {
            for p in 0..k {
                let a_val = a[i * k + p];
                for j in 0..n {
                    c_row[j] += a_val * b[p * n + j];
                }
            }
        });
        c
    }

    /// C = A^T * B where A is [k, m] (transposed to [m, k]) and B is [k, n].
    pub fn matmul_at_b(a: &[f32], b: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
        debug_assert_eq!(a.len(), k * m);
        debug_assert_eq!(b.len(), k * n);
        let mut c = vec![0.0f32; m * n];
        
        // Parallelize over rows of C (m)
        c.par_chunks_exact_mut(n).enumerate().for_each(|(i, c_row)| {
            for p in 0..k {
                let a_val = a[p * m + i]; // A^T[i,p] = A[p,i]
                for j in 0..n {
                    c_row[j] += a_val * b[p * n + j];
                }
            }
        });
        c
    }

    /// C = A * B^T where A is [m, k] and B is [n, k] (transposed to [k, n]).
    pub fn matmul_a_bt(a: &[f32], b: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
        debug_assert_eq!(a.len(), m * k);
        debug_assert_eq!(b.len(), n * k);
        let mut c = vec![0.0f32; m * n];
        
        // Parallelize over rows of C (m)
        c.par_chunks_exact_mut(n).enumerate().for_each(|(i, c_row)| {
            for j in 0..n {
                let mut sum = 0.0f32;
                for p in 0..k {
                    sum += a[i * k + p] * b[j * k + p]; // B^T[p,j] = B[j,p]
                }
                c_row[j] = sum;
            }
        });
        c
    }
}

// ---- Apple MLX GPU backend (Metal-accelerated, macOS only) ----

#[cfg(all(target_os = "macos", feature = "mlx"))]
pub mod mlx_gpu {
    use mlx_rs::Array;

    /// GPU matmul: C = A * B where A is [m,k], B is [k,n], result is [m,n].
    pub fn matmul(a: &[f32], b: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
        debug_assert_eq!(a.len(), m * k, "mlx matmul: a must be [m,k]");
        debug_assert_eq!(b.len(), k * n, "mlx matmul: b must be [k,n]");
        let a_arr = Array::from_slice(a, &[m as i32, k as i32]);
        let b_arr = Array::from_slice(b, &[k as i32, n as i32]);
        let c = a_arr.matmul(&b_arr).unwrap();
        c.eval().unwrap();
        c.as_slice::<f32>().to_vec()
    }

    /// GPU matmul: C = A^T * B where A is stored as [k,m], B is [k,n].
    pub fn matmul_at_b(a: &[f32], b: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
        debug_assert_eq!(a.len(), k * m, "mlx matmul_at_b: a must be [k,m]");
        debug_assert_eq!(b.len(), k * n, "mlx matmul_at_b: b must be [k,n]");
        let a_arr = Array::from_slice(a, &[k as i32, m as i32]);
        let b_arr = Array::from_slice(b, &[k as i32, n as i32]);
        let at = a_arr.t();
        let c = at.matmul(&b_arr).unwrap();
        c.eval().unwrap();
        c.as_slice::<f32>().to_vec()
    }

    /// GPU matmul: C = A * B^T where A is [m,k], B is stored as [n,k].
    pub fn matmul_a_bt(a: &[f32], b: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
        debug_assert_eq!(a.len(), m * k, "mlx matmul_a_bt: a must be [m,k]");
        debug_assert_eq!(b.len(), n * k, "mlx matmul_a_bt: b must be [n,k]");
        let a_arr = Array::from_slice(a, &[m as i32, k as i32]);
        let b_arr = Array::from_slice(b, &[n as i32, k as i32]);
        let bt = b_arr.t();
        let c = a_arr.matmul(&bt).unwrap();
        c.eval().unwrap();
        c.as_slice::<f32>().to_vec()
    }
}

// ---- wgpu GPU backend (cross-platform: NVIDIA, AMD, Intel, Apple) ----

#[cfg(feature = "gpu")]
pub mod wgpu_gpu {
    use crate::gpu_compute::gpu_compute::get_gpu;

    /// Try GPU matmul.  Returns `None` when no GPU adapter is available.
    pub fn matmul(a: &[f32], b: &[f32], m: usize, n: usize, k: usize) -> Option<Vec<f32>> {
        get_gpu().map(|gpu| gpu.matmul(a, b, m, n, k))
    }

    pub fn matmul_at_b(a: &[f32], b: &[f32], m: usize, n: usize, k: usize) -> Option<Vec<f32>> {
        get_gpu().map(|gpu| gpu.matmul_at_b(a, b, m, n, k))
    }

    pub fn matmul_a_bt(a: &[f32], b: &[f32], m: usize, n: usize, k: usize) -> Option<Vec<f32>> {
        get_gpu().map(|gpu| gpu.matmul_a_bt(a, b, m, n, k))
    }

    /// Returns the adapter name if a GPU is available.
    pub fn adapter_name() -> Option<&'static str> {
        get_gpu().map(|gpu| gpu.adapter_name())
    }
}

// ---- Public API ----
//
// Backend priority:
//   1. wgpu GPU   (feature = "gpu", any platform, any vendor)
//   2. Apple MLX  (feature = "mlx", macOS only)
//   3. Apple Accelerate BLAS (macOS, always linked)
//   4. Pure Rust fallback

/// C = A * B where A is [m, k] and B is [k, n], both row-major.
/// Returns C as [m, n].
pub fn matmul(a: &[f32], b: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
    #[cfg(feature = "gpu")]
    {
        if let Some(result) = wgpu_gpu::matmul(a, b, m, n, k) {
            return result;
        }
    }
    #[cfg(all(target_os = "macos", feature = "mlx"))]
    {
        return mlx_gpu::matmul(a, b, m, n, k);
    }
    #[cfg(not(all(target_os = "macos", feature = "mlx")))]
    {
        platform::matmul(a, b, m, n, k)
    }
}

/// C = A^T * B where A is stored as [k, m] and B is [k, n].
/// Returns C as [m, n]. Used for gradient computations (dW = X^T @ dY).
pub fn matmul_at_b(a: &[f32], b: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
    #[cfg(feature = "gpu")]
    {
        if let Some(result) = wgpu_gpu::matmul_at_b(a, b, m, n, k) {
            return result;
        }
    }
    #[cfg(all(target_os = "macos", feature = "mlx"))]
    {
        return mlx_gpu::matmul_at_b(a, b, m, n, k);
    }
    #[cfg(not(all(target_os = "macos", feature = "mlx")))]
    {
        platform::matmul_at_b(a, b, m, n, k)
    }
}

/// C = A * B^T where A is [m, k] and B is stored as [n, k].
/// Returns C as [m, n]. Used for backprop (d_hidden = d_logits @ W^T).
pub fn matmul_a_bt(a: &[f32], b: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
    #[cfg(feature = "gpu")]
    {
        if let Some(result) = wgpu_gpu::matmul_a_bt(a, b, m, n, k) {
            return result;
        }
    }
    #[cfg(all(target_os = "macos", feature = "mlx"))]
    {
        return mlx_gpu::matmul_a_bt(a, b, m, n, k);
    }
    #[cfg(not(all(target_os = "macos", feature = "mlx")))]
    {
        platform::matmul_a_bt(a, b, m, n, k)
    }
}

/// Returns the name of the active matmul backend.
pub fn backend_name() -> &'static str {
    #[cfg(feature = "gpu")]
    {
        if let Some(name) = wgpu_gpu::adapter_name() {
            // Leak is fine -- this is a 'static str from a singleton that lives forever.
            return Box::leak(format!("wgpu ({})", name).into_boxed_str());
        }
    }
    #[cfg(all(target_os = "macos", feature = "mlx"))]
    {
        return "Apple MLX (Metal GPU)";
    }
    #[cfg(all(target_os = "macos", not(feature = "mlx")))]
    {
        return "Apple Accelerate (CPU BLAS)";
    }
    #[cfg(not(target_os = "macos"))]
    {
        return "Pure Rust (fallback)";
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matmul_identity() {
        // 2x2 identity * [1,2; 3,4]
        let eye = vec![1.0, 0.0, 0.0, 1.0f32];
        let b = vec![1.0, 2.0, 3.0, 4.0f32];
        let c = matmul(&eye, &b, 2, 2, 2);
        assert_eq!(c, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_matmul_basic() {
        // [1, 2, 3; 4, 5, 6] (2x3) * [7, 8; 9, 10; 11, 12] (3x2) = [58, 64; 139, 154]
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0f32];
        let b = vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0f32];
        let c = matmul(&a, &b, 2, 2, 3);
        assert!((c[0] - 58.0).abs() < 1e-4);
        assert!((c[1] - 64.0).abs() < 1e-4);
        assert!((c[2] - 139.0).abs() < 1e-4);
        assert!((c[3] - 154.0).abs() < 1e-4);
    }

    #[test]
    fn test_matmul_at_b() {
        // A stored as [3, 2] = [[1,4],[2,5],[3,6]]
        // A^T = [1,2,3; 4,5,6] (2x3)
        // B = [7,8; 9,10; 11,12] (3x2)
        // A^T * B = [58, 64; 139, 154]
        let a = vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0f32]; // [k=3, m=2]
        let b = vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0f32]; // [k=3, n=2]
        let c = matmul_at_b(&a, &b, 2, 2, 3);
        assert!((c[0] - 58.0).abs() < 1e-4);
        assert!((c[1] - 64.0).abs() < 1e-4);
        assert!((c[2] - 139.0).abs() < 1e-4);
        assert!((c[3] - 154.0).abs() < 1e-4);
    }

    #[test]
    fn test_matmul_a_bt() {
        // A = [1,2,3; 4,5,6] (2x3)
        // B stored as [2, 3] = [[7,9,11],[8,10,12]]
        // B^T = [7,8; 9,10; 11,12] (3x2)
        // A * B^T = [58, 64; 139, 154]
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0f32]; // [m=2, k=3]
        let b = vec![7.0, 9.0, 11.0, 8.0, 10.0, 12.0f32]; // [n=2, k=3]
        let c = matmul_a_bt(&a, &b, 2, 2, 3);
        assert!((c[0] - 58.0).abs() < 1e-4);
        assert!((c[1] - 64.0).abs() < 1e-4);
        assert!((c[2] - 139.0).abs() < 1e-4);
        assert!((c[3] - 154.0).abs() < 1e-4);
    }
}
