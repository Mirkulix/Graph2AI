//! Linear algebra foundation for IGQK math engine.
//!
//! Dense matrix operations on `Vec<f64>` in row-major order.

/// Matrix multiply: [m,k] × [k,n] -> [m,n]
pub fn mat_mul(a: &[f64], b: &[f64], m: usize, k: usize, n: usize) -> Vec<f64> {
    assert_eq!(a.len(), m * k, "a must have m*k elements");
    assert_eq!(b.len(), k * n, "b must have k*n elements");
    let mut c = vec![0.0; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0;
            for p in 0..k {
                sum += a[i * k + p] * b[p * n + j];
            }
            c[i * n + j] = sum;
        }
    }
    c
}

/// Element-wise add of two n×n matrices.
pub fn mat_add(a: &[f64], b: &[f64], n: usize) -> Vec<f64> {
    let len = n * n;
    assert_eq!(a.len(), len);
    assert_eq!(b.len(), len);
    a.iter().zip(b.iter()).map(|(x, y)| x + y).collect()
}

/// Element-wise subtract of two n×n matrices: A - B.
pub fn mat_sub(a: &[f64], b: &[f64], n: usize) -> Vec<f64> {
    let len = n * n;
    assert_eq!(a.len(), len);
    assert_eq!(b.len(), len);
    a.iter().zip(b.iter()).map(|(x, y)| x - y).collect()
}

/// Scale all elements of an n×n matrix by a scalar.
pub fn mat_scale(a: &[f64], scalar: f64, n: usize) -> Vec<f64> {
    assert_eq!(a.len(), n * n);
    a.iter().map(|x| x * scalar).collect()
}

/// Transpose [m,n] -> [n,m].
pub fn mat_transpose(a: &[f64], m: usize, n: usize) -> Vec<f64> {
    assert_eq!(a.len(), m * n);
    let mut t = vec![0.0; m * n];
    for i in 0..m {
        for j in 0..n {
            t[j * m + i] = a[i * n + j];
        }
    }
    t
}

/// Trace (sum of diagonal) of an n×n matrix.
pub fn mat_trace(a: &[f64], n: usize) -> f64 {
    assert!(a.len() >= n * n);
    (0..n).map(|i| a[i * n + i]).sum()
}

/// n×n identity matrix.
pub fn mat_identity(n: usize) -> Vec<f64> {
    let mut m = vec![0.0; n * n];
    for i in 0..n {
        m[i * n + i] = 1.0;
    }
    m
}

/// Commutator [A,B] = AB - BA for n×n matrices.
pub fn mat_commutator(a: &[f64], b: &[f64], n: usize) -> Vec<f64> {
    let ab = mat_mul(a, b, n, n, n);
    let ba = mat_mul(b, a, n, n, n);
    mat_sub(&ab, &ba, n)
}

/// Anticommutator {A,B} = AB + BA for n×n matrices.
pub fn mat_anticommutator(a: &[f64], b: &[f64], n: usize) -> Vec<f64> {
    let ab = mat_mul(a, b, n, n, n);
    let ba = mat_mul(b, a, n, n, n);
    mat_add(&ab, &ba, n)
}

/// Frobenius norm: ||A||_F = sqrt(sum(a_ij²)).
pub fn mat_frobenius_norm(a: &[f64]) -> f64 {
    a.iter().map(|x| x * x).sum::<f64>().sqrt()
}

/// Inverse for small matrices (n <= 4) via Gauss-Jordan elimination.
/// Returns `None` if the matrix is singular.
pub fn mat_inverse_small(a: &[f64], n: usize) -> Option<Vec<f64>> {
    assert!(n <= 4, "mat_inverse_small only supports n <= 4");
    assert_eq!(a.len(), n * n);

    // Augmented matrix [A | I], stored as n rows of 2n columns
    let mut aug = vec![0.0; n * 2 * n];
    for i in 0..n {
        for j in 0..n {
            aug[i * 2 * n + j] = a[i * n + j];
        }
        aug[i * 2 * n + n + i] = 1.0;
    }

    let cols = 2 * n;

    for col in 0..n {
        // Find pivot
        let mut pivot_row = None;
        let mut max_val = 1e-12;
        for row in col..n {
            let v = aug[row * cols + col].abs();
            if v > max_val {
                max_val = v;
                pivot_row = Some(row);
            }
        }
        let pivot_row = pivot_row?; // singular

        // Swap rows
        if pivot_row != col {
            for j in 0..cols {
                let tmp = aug[col * cols + j];
                aug[col * cols + j] = aug[pivot_row * cols + j];
                aug[pivot_row * cols + j] = tmp;
            }
        }

        // Scale pivot row
        let diag = aug[col * cols + col];
        for j in 0..cols {
            aug[col * cols + j] /= diag;
        }

        // Eliminate column
        for row in 0..n {
            if row == col {
                continue;
            }
            let factor = aug[row * cols + col];
            for j in 0..cols {
                aug[row * cols + j] -= factor * aug[col * cols + j];
            }
        }
    }

    // Extract inverse from right half
    let mut inv = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            inv[i * n + j] = aug[i * cols + n + j];
        }
    }
    Some(inv)
}

/// Vector outer product: a⊗b where a has length m and b has length n,
/// producing an m×n matrix.
pub fn mat_outer_product(a: &[f64], b: &[f64]) -> Vec<f64> {
    let m = a.len();
    let n = b.len();
    let mut result = vec![0.0; m * n];
    for i in 0..m {
        for j in 0..n {
            result[i * n + j] = a[i] * b[j];
        }
    }
    result
}

/// Check if an n×n symmetric matrix is positive semidefinite using Gershgorin circle theorem.
/// This is a sufficient condition check: if all Gershgorin discs have non-negative left edges,
/// the matrix is PSD. Returns `false` if any disc extends below zero.
pub fn mat_is_positive_semidefinite(a: &[f64], n: usize) -> bool {
    assert_eq!(a.len(), n * n);
    for i in 0..n {
        let diag = a[i * n + i];
        let mut off_diag_sum = 0.0;
        for j in 0..n {
            if j != i {
                off_diag_sum += a[i * n + j].abs();
            }
        }
        // Gershgorin: eigenvalue lies in [diag - R, diag + R]
        // If diag - R < 0 for any row, might not be PSD
        if diag - off_diag_sum < -1e-10 {
            return false;
        }
    }
    true
}

/// Eigenvalues of a symmetric matrix via the classical Jacobi iteration method.
/// Returns eigenvalues in descending order.
pub fn eigenvalues_symmetric(a: &[f64], n: usize) -> Vec<f64> {
    assert_eq!(a.len(), n * n);

    if n == 0 {
        return vec![];
    }
    if n == 1 {
        return vec![a[0]];
    }

    let mut work = a.to_vec();
    let max_iter = 100 * n * n;
    let tol = 1e-12;

    for _ in 0..max_iter {
        // Find largest off-diagonal element
        let mut max_off = 0.0f64;
        let mut p = 0;
        let mut q = 1;
        for i in 0..n {
            for j in (i + 1)..n {
                let v = work[i * n + j].abs();
                if v > max_off {
                    max_off = v;
                    p = i;
                    q = j;
                }
            }
        }

        if max_off < tol {
            break;
        }

        // Compute rotation
        let app = work[p * n + p];
        let aqq = work[q * n + q];
        let apq = work[p * n + q];

        let theta = if (app - aqq).abs() < tol {
            std::f64::consts::FRAC_PI_4
        } else {
            0.5 * ((2.0 * apq) / (app - aqq)).atan()
        };

        let c = theta.cos();
        let s = theta.sin();

        // Apply Jacobi rotation
        let mut new_work = work.clone();

        // Update rows/cols p and q
        for i in 0..n {
            if i != p && i != q {
                let aip = work[i * n + p];
                let aiq = work[i * n + q];
                new_work[i * n + p] = c * aip + s * aiq;
                new_work[p * n + i] = new_work[i * n + p];
                new_work[i * n + q] = -s * aip + c * aiq;
                new_work[q * n + i] = new_work[i * n + q];
            }
        }

        new_work[p * n + p] = c * c * app + 2.0 * s * c * apq + s * s * aqq;
        new_work[q * n + q] = s * s * app - 2.0 * s * c * apq + c * c * aqq;
        new_work[p * n + q] = 0.0;
        new_work[q * n + p] = 0.0;

        work = new_work;
    }

    let mut eigenvalues: Vec<f64> = (0..n).map(|i| work[i * n + i]).collect();
    eigenvalues.sort_by(|a, b| b.partial_cmp(a).unwrap());
    eigenvalues
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f64 = 1e-9;

    fn approx_eq(a: f64, b: f64) -> bool {
        (a - b).abs() < EPS
    }

    fn vecs_approx_eq(a: &[f64], b: &[f64]) -> bool {
        a.len() == b.len() && a.iter().zip(b).all(|(x, y)| approx_eq(*x, *y))
    }

    #[test]
    fn test_commutator_antisymmetry() {
        // [A,B] = -[B,A]
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![5.0, 6.0, 7.0, 8.0];
        let ab_comm = mat_commutator(&a, &b, 2);
        let ba_comm = mat_commutator(&b, &a, 2);
        let neg_ba = mat_scale(&ba_comm, -1.0, 2);
        assert!(vecs_approx_eq(&ab_comm, &neg_ba), "[A,B] should equal -[B,A]");
    }

    #[test]
    fn test_anticommutator_symmetry() {
        // {A,B} = {B,A}
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![5.0, 6.0, 7.0, 8.0];
        let ab_anti = mat_anticommutator(&a, &b, 2);
        let ba_anti = mat_anticommutator(&b, &a, 2);
        assert!(vecs_approx_eq(&ab_anti, &ba_anti), "{{A,B}} should equal {{B,A}}");
    }

    #[test]
    fn test_identity_mul() {
        // I × A = A
        let n = 3;
        let id = mat_identity(n);
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let result = mat_mul(&id, &a, n, n, n);
        assert!(vecs_approx_eq(&result, &a), "I*A should equal A");
    }

    #[test]
    fn test_trace_identity() {
        for n in 1..=5 {
            let id = mat_identity(n);
            assert!(approx_eq(mat_trace(&id, n), n as f64), "Tr(I) should equal n");
        }
    }

    #[test]
    fn test_inverse_roundtrip() {
        // A × A⁻¹ = I
        let a = vec![4.0, 7.0, 2.0, 6.0];
        let inv = mat_inverse_small(&a, 2).expect("matrix should be invertible");
        let product = mat_mul(&a, &inv, 2, 2, 2);
        let id = mat_identity(2);
        assert!(vecs_approx_eq(&product, &id), "A*A^-1 should equal I");
    }

    #[test]
    fn test_inverse_3x3() {
        let a = vec![
            1.0, 2.0, 3.0,
            0.0, 1.0, 4.0,
            5.0, 6.0, 0.0,
        ];
        let inv = mat_inverse_small(&a, 3).expect("should be invertible");
        let product = mat_mul(&a, &inv, 3, 3, 3);
        let id = mat_identity(3);
        assert!(vecs_approx_eq(&product, &id), "3x3 A*A^-1 should equal I");
    }

    #[test]
    fn test_singular_matrix() {
        let a = vec![1.0, 2.0, 2.0, 4.0]; // rank 1
        assert!(mat_inverse_small(&a, 2).is_none(), "singular matrix should return None");
    }

    #[test]
    fn test_frobenius_norm_identity() {
        for n in 1..=4 {
            let id = mat_identity(n);
            assert!(approx_eq(mat_frobenius_norm(&id), (n as f64).sqrt()),
                "||I||_F should equal sqrt(n)");
        }
    }

    #[test]
    fn test_outer_product_dimensions() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0];
        let result = mat_outer_product(&a, &b);
        assert_eq!(result.len(), 6, "outer product of 3-vec and 2-vec should have 6 elements");
        // Check values: result[i,j] = a[i]*b[j]
        assert!(approx_eq(result[0], 4.0));  // 1*4
        assert!(approx_eq(result[1], 5.0));  // 1*5
        assert!(approx_eq(result[4], 12.0)); // 3*4
        assert!(approx_eq(result[5], 15.0)); // 3*5
    }

    #[test]
    fn test_transpose_of_transpose() {
        let m = 2;
        let n = 3;
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let t = mat_transpose(&a, m, n);
        let tt = mat_transpose(&t, n, m);
        assert!(vecs_approx_eq(&tt, &a), "transpose(transpose(A)) should equal A");
    }

    #[test]
    fn test_psd_identity() {
        let id = mat_identity(3);
        assert!(mat_is_positive_semidefinite(&id, 3), "identity is PSD");
    }

    #[test]
    fn test_psd_negative() {
        // Diagonal matrix with a negative entry
        let a = vec![
            1.0, 0.0, 0.0,
            0.0, -2.0, 0.0,
            0.0, 0.0, 1.0,
        ];
        assert!(!mat_is_positive_semidefinite(&a, 3), "matrix with negative eigenvalue is not PSD");
    }

    #[test]
    fn test_eigenvalues_diagonal() {
        let a = vec![
            3.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 2.0,
        ];
        let eigs = eigenvalues_symmetric(&a, 3);
        assert_eq!(eigs.len(), 3);
        // Should be sorted descending: 3, 2, 1
        assert!(approx_eq(eigs[0], 3.0), "largest eigenvalue should be 3.0, got {}", eigs[0]);
        assert!(approx_eq(eigs[1], 2.0), "middle eigenvalue should be 2.0, got {}", eigs[1]);
        assert!(approx_eq(eigs[2], 1.0), "smallest eigenvalue should be 1.0, got {}", eigs[2]);
    }

    #[test]
    fn test_eigenvalues_symmetric_2x2() {
        // [[2, 1], [1, 2]] has eigenvalues 3 and 1
        let a = vec![2.0, 1.0, 1.0, 2.0];
        let eigs = eigenvalues_symmetric(&a, 2);
        assert!(approx_eq(eigs[0], 3.0), "expected 3.0, got {}", eigs[0]);
        assert!(approx_eq(eigs[1], 1.0), "expected 1.0, got {}", eigs[1]);
    }

    #[test]
    fn test_mat_add_sub_roundtrip() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![5.0, 6.0, 7.0, 8.0];
        let sum = mat_add(&a, &b, 2);
        let diff = mat_sub(&sum, &b, 2);
        assert!(vecs_approx_eq(&diff, &a), "A + B - B should equal A");
    }

    #[test]
    fn test_mat_scale() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let scaled = mat_scale(&a, 2.0, 2);
        assert!(vecs_approx_eq(&scaled, &[2.0, 4.0, 6.0, 8.0]));
    }
}
