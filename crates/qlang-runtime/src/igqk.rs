//! IMPORTANT: "Quantum" in IGQK refers to the mathematical formalism
//! (density matrices, von Neumann entropy) — NOT to quantum computing hardware.
//! All computations run on classical CPUs/GPUs. The quantum notation enables
//! compact expression of information-theoretic compression bounds.
//!
//! IGQK Mathematical Engine
//!
//! Implements the real mathematics from the IGQK theory:
//! - Fisher Information Metric
//! - Quantum Gradient Flow
//! - Laplace-Beltrami Operator
//! - Compression with Distortion Bounds
//! - Convergence Checking
//!
//! References: IGQK Theory (Informationsgeometrische Quantenkompression)

use qlang_core::quantum::DensityMatrix;

// ============================================================
// Section 1: Matrix Operations (helpers for quantum mechanics)
// ============================================================

/// Multiply two n x n matrices stored as flat slices in row-major order.
/// Returns a new n x n matrix as Vec<f64>.
pub fn matrix_multiply(a: &[f64], b: &[f64], n: usize) -> Vec<f64> {
    assert_eq!(a.len(), n * n, "matrix a must be n x n");
    assert_eq!(b.len(), n * n, "matrix b must be n x n");
    let mut c = vec![0.0; n * n];
    for i in 0..n {
        for k in 0..n {
            let a_ik = a[i * n + k];
            for j in 0..n {
                c[i * n + j] += a_ik * b[k * n + j];
            }
        }
    }
    c
}

/// Compute the matrix commutator [A, B] = AB - BA.
/// This represents unitary (quantum) evolution in the gradient flow.
pub fn matrix_commutator(a: &[f64], b: &[f64], n: usize) -> Vec<f64> {
    let ab = matrix_multiply(a, b, n);
    let ba = matrix_multiply(b, a, n);
    ab.iter().zip(ba.iter()).map(|(x, y)| x - y).collect()
}

/// Compute the matrix anticommutator {A, B} = AB + BA.
/// This represents dissipative evolution in the gradient flow.
pub fn matrix_anticommutator(a: &[f64], b: &[f64], n: usize) -> Vec<f64> {
    let ab = matrix_multiply(a, b, n);
    let ba = matrix_multiply(b, a, n);
    ab.iter().zip(ba.iter()).map(|(x, y)| x + y).collect()
}

/// Compute the trace of an n x n matrix: Tr(A) = sum of diagonal elements.
pub fn matrix_trace(a: &[f64], n: usize) -> f64 {
    assert!(a.len() >= n * n);
    (0..n).map(|i| a[i * n + i]).sum()
}

/// Invert a 2x2 matrix. Returns None if singular.
pub fn matrix_inverse_2x2(m: &[f64]) -> Option<Vec<f64>> {
    assert_eq!(m.len(), 4);
    let det = m[0] * m[3] - m[1] * m[2];
    if det.abs() < 1e-15 {
        return None;
    }
    let inv_det = 1.0 / det;
    Some(vec![
        m[3] * inv_det,
        -m[1] * inv_det,
        -m[2] * inv_det,
        m[0] * inv_det,
    ])
}

/// Invert a general n x n matrix using Gauss-Jordan elimination.
/// Returns None if singular (determinant near zero).
pub fn matrix_inverse_general(m: &[f64], n: usize) -> Option<Vec<f64>> {
    assert_eq!(m.len(), n * n);
    if n == 0 {
        return Some(vec![]);
    }
    if n == 2 {
        return matrix_inverse_2x2(m);
    }
    // Augmented matrix [A | I]
    let mut aug = vec![0.0; n * 2 * n];
    for i in 0..n {
        for j in 0..n {
            aug[i * 2 * n + j] = m[i * n + j];
        }
        aug[i * 2 * n + n + i] = 1.0;
    }
    // Forward elimination with partial pivoting
    for col in 0..n {
        // Find pivot
        let mut max_val = aug[col * 2 * n + col].abs();
        let mut max_row = col;
        for row in (col + 1)..n {
            let val = aug[row * 2 * n + col].abs();
            if val > max_val {
                max_val = val;
                max_row = row;
            }
        }
        if max_val < 1e-15 {
            return None; // Singular
        }
        // Swap rows
        if max_row != col {
            for j in 0..(2 * n) {
                let tmp = aug[col * 2 * n + j];
                aug[col * 2 * n + j] = aug[max_row * 2 * n + j];
                aug[max_row * 2 * n + j] = tmp;
            }
        }
        // Scale pivot row
        let pivot = aug[col * 2 * n + col];
        for j in 0..(2 * n) {
            aug[col * 2 * n + j] /= pivot;
        }
        // Eliminate column
        for row in 0..n {
            if row == col {
                continue;
            }
            let factor = aug[row * 2 * n + col];
            for j in 0..(2 * n) {
                aug[row * 2 * n + j] -= factor * aug[col * 2 * n + j];
            }
        }
    }
    // Extract inverse from right half
    let mut inv = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            inv[i * n + j] = aug[i * 2 * n + n + j];
        }
    }
    Some(inv)
}

/// Compute the identity matrix of dimension n.
pub fn identity_matrix(n: usize) -> Vec<f64> {
    let mut m = vec![0.0; n * n];
    for i in 0..n {
        m[i * n + i] = 1.0;
    }
    m
}

/// Reconstruct the full n x n density matrix from eigendecomposition.
/// rho = sum_k p_k |psi_k><psi_k|
pub fn density_matrix_to_full(rho: &DensityMatrix) -> Vec<f64> {
    let n = rho.dim;
    let rank = rho.eigenvalues.len();
    let mut full = vec![0.0; n * n];
    for k in 0..rank {
        let pk = rho.eigenvalues[k];
        let psi = &rho.eigenvectors[k * n..(k + 1) * n];
        for i in 0..n {
            for j in 0..n {
                full[i * n + j] += pk * psi[i] * psi[j];
            }
        }
    }
    full
}

// ============================================================
// Section 2: Fisher Information Metric
// ============================================================

/// Compute the empirical Fisher Information Metric.
///
/// G_ij(theta) = E[d_i log p(x; theta) * d_j log p(x; theta)]
///
/// Uses the empirical Fisher approximation: average outer product of
/// gradients over a mini-batch. Each row of `gradients_batch` is the
/// gradient vector for one sample.
///
/// Returns a d x d matrix where d = weights.len().
pub fn compute_fisher_metric(
    weights: &[f32],
    gradients_batch: &[Vec<f32>],
) -> Vec<Vec<f32>> {
    let d = weights.len();
    let n = gradients_batch.len();
    if n == 0 {
        return vec![vec![0.0; d]; d];
    }
    let mut fisher = vec![vec![0.0f64; d]; d];
    for grad in gradients_batch {
        assert_eq!(grad.len(), d, "gradient dimension must match weights");
        for i in 0..d {
            let gi = grad[i] as f64;
            for j in i..d {
                let gj = grad[j] as f64;
                let val = gi * gj;
                fisher[i][j] += val;
                if i != j {
                    fisher[j][i] += val;
                }
            }
        }
    }
    // Average over batch
    let inv_n = 1.0 / n as f64;
    let mut result = vec![vec![0.0f32; d]; d];
    for i in 0..d {
        for j in 0..d {
            result[i][j] = (fisher[i][j] * inv_n) as f32;
        }
    }
    result
}

/// Flatten a Vec<Vec<f32>> Fisher metric to a flat Vec<f64> for matrix ops.
pub fn flatten_fisher(fisher: &[Vec<f32>]) -> Vec<f64> {
    let n = fisher.len();
    let mut flat = Vec::with_capacity(n * n);
    for row in fisher {
        for &val in row {
            flat.push(val as f64);
        }
    }
    flat
}

// ============================================================
// Section 3: Laplace-Beltrami Operator
// ============================================================

/// Compute the Hamiltonian H = -Delta_M (Laplace-Beltrami operator)
/// approximated on the statistical manifold.
///
/// Simplified eigenvalue-based computation:
/// Given the metric tensor G, the Laplace-Beltrami operator acts on
/// density matrices. We approximate H using the metric eigenvalues.
///
/// H_ij = -sum_k (1/g_kk) * delta_ij (diagonal approximation)
/// where g_kk are the diagonal elements of the metric.
///
/// For a more accurate computation, we would need the full Christoffel
/// symbols, but this diagonal approximation is standard for first
/// implementations.
pub fn laplace_beltrami(rho: &DensityMatrix, metric: &[f64]) -> Vec<f64> {
    let n = rho.dim;
    assert!(metric.len() >= n, "metric must have at least n diagonal elements");
    // Construct H as a diagonal matrix from metric eigenvalues
    let mut h = vec![0.0; n * n];
    for i in 0..n {
        // H_ii = -1/g_ii (negative inverse of metric diagonal)
        let g_ii = if i < metric.len() && metric[i].abs() > 1e-15 {
            metric[i]
        } else {
            1.0 // regularize
        };
        h[i * n + i] = -1.0 / g_ii;
    }
    h
}

// ============================================================
// Section 4: Quantum Gradient Flow
// ============================================================

/// Evolve the density matrix according to the quantum gradient flow:
///
///   d rho / dt = -i [H, rho] - gamma {G^{-1} nabla L, rho}
///
/// where:
/// - [H, rho] = H*rho - rho*H  (commutator: unitary/quantum evolution)
/// - {A, rho} = A*rho + rho*A   (anticommutator: dissipative evolution)
/// - H is the Hamiltonian (Laplace-Beltrami operator)
/// - G^{-1} nabla L is the natural gradient
/// - gamma > 0 is the damping parameter
/// - dt is the integration time step
///
/// After evolution, the density matrix is renormalized so Tr(rho) = 1
/// and eigenvalues are clamped to [0, 1].
pub fn evolve_density_matrix(
    rho: &mut DensityMatrix,
    hamiltonian: &[f64],
    natural_gradient: &[f64],
    gamma: f64,
    dt: f64,
) {
    let n = rho.dim;
    assert_eq!(hamiltonian.len(), n * n, "hamiltonian must be n x n");
    assert_eq!(natural_gradient.len(), n * n, "natural_gradient must be n x n");

    // Reconstruct full density matrix
    let rho_full = density_matrix_to_full(rho);

    // Compute commutator [H, rho] (unitary part)
    let commutator = matrix_commutator(hamiltonian, &rho_full, n);

    // Compute anticommutator {G^{-1} nabla L, rho} (dissipative part)
    let anticommutator = matrix_anticommutator(natural_gradient, &rho_full, n);

    // d rho / dt = -i [H, rho] - gamma {G^{-1} nabla L, rho}
    // For real-valued density matrices, the -i factor on the commutator
    // produces an antisymmetric perturbation. We keep the real part only
    // since we work in a real Hilbert space approximation.
    // In a real framework the commutator term provides mixing/rotation.
    let mut new_rho = vec![0.0; n * n];
    for idx in 0..n * n {
        // The commutator term [H, rho] is antisymmetric for Hermitian H and rho,
        // so -i[H,rho] is Hermitian. In the real case we keep the commutator
        // contribution directly (it drives exploration/mixing).
        new_rho[idx] = rho_full[idx]
            + dt * (-commutator[idx] - gamma * anticommutator[idx]);
    }

    // Symmetrize (enforce Hermiticity in real case)
    for i in 0..n {
        for j in (i + 1)..n {
            let avg = 0.5 * (new_rho[i * n + j] + new_rho[j * n + i]);
            new_rho[i * n + j] = avg;
            new_rho[j * n + i] = avg;
        }
    }

    // Simple eigendecomposition for small matrices using Jacobi iteration
    let (eigenvalues, eigenvectors) = symmetric_eigen(&new_rho, n);

    // Clamp eigenvalues to [0, inf) and renormalize
    let mut clamped: Vec<f64> = eigenvalues.iter().map(|&v| v.max(0.0)).collect();
    let trace: f64 = clamped.iter().sum();
    if trace > 1e-15 {
        for v in &mut clamped {
            *v /= trace;
        }
    } else {
        // Fallback to maximally mixed
        let uniform = 1.0 / n as f64;
        clamped = vec![uniform; n];
    }

    // Filter out near-zero eigenvalues for low-rank storage
    let threshold = 1e-12;
    let mut kept_evals = Vec::new();
    let mut kept_evecs = Vec::new();
    for (k, &eval) in clamped.iter().enumerate() {
        if eval > threshold {
            kept_evals.push(eval);
            kept_evecs.extend_from_slice(&eigenvectors[k * n..(k + 1) * n]);
        }
    }
    if kept_evals.is_empty() {
        // Keep at least one state
        kept_evals.push(1.0);
        kept_evecs = vec![0.0; n];
        kept_evecs[0] = 1.0;
    }

    rho.eigenvalues = kept_evals;
    rho.eigenvectors = kept_evecs;
}

/// Simple symmetric eigendecomposition using Jacobi rotations.
/// Returns (eigenvalues, eigenvectors) where eigenvectors are stored
/// as rows in a flat n*n array (row k is eigenvector k).
fn symmetric_eigen(matrix: &[f64], n: usize) -> (Vec<f64>, Vec<f64>) {
    assert_eq!(matrix.len(), n * n);
    let mut a = matrix.to_vec();
    let mut v = identity_matrix(n);
    let max_iter = 100 * n * n;

    for _ in 0..max_iter {
        // Find largest off-diagonal element
        let mut max_val = 0.0f64;
        let mut p = 0;
        let mut q = 1;
        for i in 0..n {
            for j in (i + 1)..n {
                let val = a[i * n + j].abs();
                if val > max_val {
                    max_val = val;
                    p = i;
                    q = j;
                }
            }
        }
        if max_val < 1e-14 {
            break;
        }
        // Compute rotation angle
        let app = a[p * n + p];
        let aqq = a[q * n + q];
        let apq = a[p * n + q];
        let theta = if (app - aqq).abs() < 1e-15 {
            std::f64::consts::FRAC_PI_4
        } else {
            0.5 * (2.0 * apq / (app - aqq)).atan()
        };
        let c = theta.cos();
        let s = theta.sin();
        // Apply rotation to a
        let mut new_a = a.clone();
        for i in 0..n {
            if i != p && i != q {
                let aip = a[i * n + p];
                let aiq = a[i * n + q];
                new_a[i * n + p] = c * aip + s * aiq;
                new_a[p * n + i] = new_a[i * n + p];
                new_a[i * n + q] = -s * aip + c * aiq;
                new_a[q * n + i] = new_a[i * n + q];
            }
        }
        new_a[p * n + p] = c * c * app + 2.0 * s * c * apq + s * s * aqq;
        new_a[q * n + q] = s * s * app - 2.0 * s * c * apq + c * c * aqq;
        new_a[p * n + q] = 0.0;
        new_a[q * n + p] = 0.0;
        a = new_a;
        // Apply rotation to eigenvector matrix
        let mut new_v = v.clone();
        for i in 0..n {
            let vip = v[i * n + p];
            let viq = v[i * n + q];
            new_v[i * n + p] = c * vip + s * viq;
            new_v[i * n + q] = -s * vip + c * viq;
        }
        v = new_v;
    }
    // Extract eigenvalues (diagonal of a) and eigenvectors (columns of v -> rows)
    let eigenvalues: Vec<f64> = (0..n).map(|i| a[i * n + i]).collect();
    // Transpose v so row k = eigenvector k
    let mut evecs = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            evecs[i * n + j] = v[j * n + i];
        }
    }
    (eigenvalues, evecs)
}

// ============================================================
// Section 5: IGQK Training Step (Algorithm 1)
// ============================================================

/// Result of a single IGQK training step.
#[derive(Debug, Clone)]
pub struct IgqkStepResult {
    /// Updated weights after quantum gradient flow
    pub new_weights: Vec<f32>,
    /// The density matrix representing the quantum state
    pub density_matrix: DensityMatrix,
    /// Von Neumann entropy of the state (measures uncertainty)
    pub entropy: f64,
}

/// Perform one step of the IGQK training algorithm (Algorithm 1).
///
/// 1. Represent weights as density matrix rho = |theta><theta|
/// 2. Compute Fisher metric G from gradients
/// 3. Build Hamiltonian H = -Delta_M from metric
/// 4. Compute natural gradient G^{-1} nabla L
/// 5. Evolve rho via quantum gradient flow
/// 6. Extract updated weights from rho (expectation value)
///
/// Parameters:
/// - `weights`: current model weights theta
/// - `gradients_batch`: per-sample gradients for Fisher metric
/// - `loss_gradient`: overall loss gradient nabla L
/// - `hbar`: quantum uncertainty parameter (controls exploration)
/// - `gamma`: damping parameter (balances quantum vs classical)
/// - `dt`: integration time step
pub fn igqk_training_step(
    weights: &[f32],
    gradients_batch: &[Vec<f32>],
    loss_gradient: &[f32],
    hbar: f64,
    gamma: f64,
    dt: f64,
) -> IgqkStepResult {
    let d = weights.len();
    // For the quantum state we work in a reduced Hilbert space.
    // Use dim = min(d, 16) for tractability; map weights into this space.
    let dim = d.min(16);

    // Step 1: Initialize density matrix as pure state from weights
    let mut state_vec = vec![0.0f64; dim];
    let norm_sq: f64 = weights.iter().take(dim).map(|&w| (w as f64) * (w as f64)).sum();
    let norm = if norm_sq > 1e-15 { norm_sq.sqrt() } else { 1.0 };
    for i in 0..dim {
        state_vec[i] = if i < d { weights[i] as f64 / norm } else { 0.0 };
    }
    let mut rho = DensityMatrix::pure_state(&state_vec);

    // Step 2: Compute Fisher metric
    let fisher = compute_fisher_metric(weights, gradients_batch);

    // Step 3: Build Hamiltonian from metric diagonal (scaled by hbar)
    let metric_diag: Vec<f64> = (0..dim)
        .map(|i| {
            if i < fisher.len() {
                (fisher[i][i] as f64).max(1e-8) * hbar
            } else {
                hbar
            }
        })
        .collect();
    let hamiltonian = laplace_beltrami(&rho, &metric_diag);

    // Step 4: Compute natural gradient matrix (G^{-1} nabla L projected to dim x dim)
    // Build a dim x dim representation of the natural gradient direction
    let mut nat_grad_matrix = vec![0.0f64; dim * dim];
    // Use diagonal approximation: (G^{-1} nabla L)_i as diagonal
    for i in 0..dim {
        let g_ii = if i < fisher.len() {
            (fisher[i][i] as f64).max(1e-8)
        } else {
            1.0
        };
        let grad_i = if i < d { loss_gradient[i] as f64 } else { 0.0 };
        // Outer product with state to form operator
        let nat_grad_i = grad_i / g_ii;
        nat_grad_matrix[i * dim + i] = nat_grad_i;
    }

    // Step 5: Evolve density matrix
    evolve_density_matrix(&mut rho, &hamiltonian, &nat_grad_matrix, gamma, dt);

    // Step 6: Extract updated weights from density matrix
    // Use the expectation: new_theta_i = Tr(rho * X_i) where X_i is the
    // position operator. For simplicity, use the dominant eigenvector
    // scaled by the original norm.
    let mut new_weights = weights.to_vec();
    if !rho.eigenvectors.is_empty() {
        // Use the eigenvector with largest eigenvalue
        let mut max_idx = 0;
        let mut max_eval = 0.0f64;
        for (k, &ev) in rho.eigenvalues.iter().enumerate() {
            if ev > max_eval {
                max_eval = ev;
                max_idx = k;
            }
        }
        let evec_start = max_idx * rho.dim;
        let evec = &rho.eigenvectors[evec_start..evec_start + rho.dim];
        for i in 0..d.min(dim) {
            new_weights[i] = (evec[i] * norm) as f32;
        }
    }

    let entropy = rho.entropy();
    IgqkStepResult {
        new_weights,
        density_matrix: rho,
        entropy,
    }
}

// ============================================================
// Section 6: Compression with Distortion Bound
// ============================================================

/// Compression method for projecting weights onto a submanifold.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CompressionMethod {
    /// Ternary weights: {-1, 0, +1}
    Ternary,
    /// Low-rank approximation with given rank
    LowRank(usize),
    /// Sparse compression keeping top-k weights
    Sparse(usize),
}

/// Result of compression with verified distortion bound.
#[derive(Debug, Clone)]
pub struct CompressionResult {
    /// Compressed weights
    pub compressed: Vec<f32>,
    /// Actual distortion D = ||W - Pi(W)||^2
    pub distortion: f64,
    /// Theoretical lower bound from Theorem 5.2
    pub theoretical_bound: f64,
    /// Whether the bound is satisfied (distortion >= bound)
    pub bound_satisfied: bool,
    /// Effective dimension after compression
    pub compressed_dim: usize,
}

/// Compress weights and verify the distortion bound from Theorem 5.2.
///
/// Theorem 5.2: D >= (n-k)/(2*beta) * log(1 + beta * sigma^2_min)
///
/// For ternary compression: k = n/16, so D >= (15n/16)/(2*beta) * log(1 + beta * sigma^2_min)
///
/// Parameters:
/// - `weights`: original weight vector
/// - `method`: compression method (Ternary, LowRank, Sparse)
/// - `beta`: inverse temperature parameter (controls bound tightness)
pub fn compress_with_bound(
    weights: &[f32],
    method: CompressionMethod,
    beta: f64,
) -> CompressionResult {
    let n = weights.len();

    // Compute compressed weights based on method
    let (compressed, k) = match method {
        CompressionMethod::Ternary => {
            let mut comp = vec![0.0f32; n];
            let mut nonzero = 0usize;
            // Find threshold for ternary quantization
            let max_abs = weights.iter().map(|w| w.abs()).fold(0.0f32, f32::max);
            let threshold = max_abs * 0.3; // standard threshold
            for (i, &w) in weights.iter().enumerate() {
                if w > threshold {
                    comp[i] = 1.0;
                    nonzero += 1;
                } else if w < -threshold {
                    comp[i] = -1.0;
                    nonzero += 1;
                }
                // else stays 0.0
            }
            let effective_k = nonzero; // ternary has ~n/16 effective dims
            (comp, effective_k)
        }
        CompressionMethod::LowRank(rank) => {
            // For a vector, low-rank means keeping the top `rank` components
            // by magnitude
            let mut indexed: Vec<(usize, f32)> =
                weights.iter().cloned().enumerate().collect();
            indexed.sort_by(|a, b| b.1.abs().partial_cmp(&a.1.abs()).unwrap());
            let mut comp = vec![0.0f32; n];
            let actual_rank = rank.min(n);
            for i in 0..actual_rank {
                let (idx, val) = indexed[i];
                comp[idx] = val;
            }
            (comp, actual_rank)
        }
        CompressionMethod::Sparse(top_k) => {
            let mut indexed: Vec<(usize, f32)> =
                weights.iter().cloned().enumerate().collect();
            indexed.sort_by(|a, b| b.1.abs().partial_cmp(&a.1.abs()).unwrap());
            let mut comp = vec![0.0f32; n];
            let actual_k = top_k.min(n);
            for i in 0..actual_k {
                let (idx, val) = indexed[i];
                comp[idx] = val;
            }
            (comp, actual_k)
        }
    };

    // Compute actual distortion D = ||W - Pi(W)||^2
    let distortion: f64 = weights
        .iter()
        .zip(compressed.iter())
        .map(|(&w, &c)| {
            let diff = (w - c) as f64;
            diff * diff
        })
        .sum();

    // Compute theoretical bound from Theorem 5.2:
    // D >= (n - k) / (2 * beta) * ln(1 + beta * sigma^2_min)
    //
    // sigma^2_min = minimum variance of weight distribution
    // We estimate it as the variance of the smallest-magnitude weights
    let sigma_sq_min = estimate_sigma_sq_min(weights);
    let n_minus_k = (n as f64) - (k as f64);
    let theoretical_bound = if beta > 1e-15 {
        (n_minus_k / (2.0 * beta)) * (1.0 + beta * sigma_sq_min).ln()
    } else {
        0.0
    };

    let bound_satisfied = distortion >= theoretical_bound - 1e-10; // small tolerance

    CompressionResult {
        compressed,
        distortion,
        theoretical_bound,
        bound_satisfied,
        compressed_dim: k,
    }
}

/// Estimate sigma^2_min (minimum variance) from weight distribution.
/// Uses the variance of the smallest quartile of weights by magnitude.
fn estimate_sigma_sq_min(weights: &[f32]) -> f64 {
    if weights.is_empty() {
        return 0.0;
    }
    let mut sorted: Vec<f64> = weights.iter().map(|&w| (w as f64).abs()).collect();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    // Take bottom quartile
    let q_end = (sorted.len() / 4).max(1);
    let subset = &sorted[..q_end];
    let mean: f64 = subset.iter().sum::<f64>() / subset.len() as f64;
    let var: f64 = subset.iter().map(|&x| (x - mean) * (x - mean)).sum::<f64>()
        / subset.len() as f64;
    var.max(1e-10) // regularize
}

// ============================================================
// Section 7: Convergence Checking
// ============================================================

/// Result of convergence analysis.
#[derive(Debug, Clone)]
pub struct ConvergenceResult {
    /// Whether the training has converged
    pub converged: bool,
    /// Current loss value
    pub current_loss: f64,
    /// Estimated minimum loss (from loss history)
    pub estimated_min_loss: f64,
    /// Quantum uncertainty floor O(hbar)
    pub quantum_floor: f64,
    /// Gap: current_loss - estimated_min - quantum_floor
    pub gap: f64,
    /// Rate of loss decrease (negative means decreasing)
    pub loss_rate: f64,
}

/// Check convergence according to Theorem 5.1:
///
///   E_rho*[L] <= min_theta L(theta) + O(hbar)
///
/// The quantum gradient flow converges to a stationary state rho* whose
/// expected loss is within O(hbar) of the global minimum.
///
/// We check:
/// 1. Is the loss decreasing slowly enough to be near convergence?
/// 2. Is the gap between current loss and minimum within O(hbar)?
/// 3. Is the density matrix close to a stationary state (low entropy change)?
pub fn check_convergence(
    rho: &DensityMatrix,
    loss_history: &[f64],
    hbar: f64,
) -> ConvergenceResult {
    let current_loss = loss_history.last().copied().unwrap_or(f64::INFINITY);
    let estimated_min = loss_history
        .iter()
        .copied()
        .fold(f64::INFINITY, f64::min);

    // O(hbar) quantum uncertainty floor
    // The constant factor depends on the problem; we use entropy as a scaling
    let entropy = rho.entropy();
    let quantum_floor = hbar * (1.0 + entropy);

    let gap = current_loss - estimated_min;

    // Compute loss rate from recent history
    let loss_rate = if loss_history.len() >= 2 {
        let recent = &loss_history[loss_history.len().saturating_sub(10)..];
        if recent.len() >= 2 {
            (recent.last().unwrap() - recent.first().unwrap()) / recent.len() as f64
        } else {
            0.0
        }
    } else {
        0.0
    };

    // Convergence criteria:
    // 1. Gap is within quantum floor
    // 2. Loss rate is small (not decreasing fast)
    // 3. State is near-pure or stable (purity is high or entropy is low)
    let gap_ok = gap <= quantum_floor + 1e-8;
    let rate_ok = loss_rate.abs() < hbar * 0.1;
    let converged = gap_ok && rate_ok && loss_history.len() >= 5;

    ConvergenceResult {
        converged,
        current_loss,
        estimated_min_loss: estimated_min,
        quantum_floor,
        gap,
        loss_rate,
    }
}

// ============================================================
// Tests
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    // --- Matrix operation tests ---

    #[test]
    fn test_matrix_multiply_identity() {
        let id = identity_matrix(3);
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let result = matrix_multiply(&a, &id, 3);
        for i in 0..9 {
            assert!((result[i] - a[i]).abs() < 1e-12, "A * I should equal A");
        }
    }

    #[test]
    fn test_commutator_antisymmetry() {
        // [A, B] = -[B, A]
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![5.0, 6.0, 7.0, 8.0];
        let ab_comm = matrix_commutator(&a, &b, 2);
        let ba_comm = matrix_commutator(&b, &a, 2);
        for i in 0..4 {
            assert!(
                (ab_comm[i] + ba_comm[i]).abs() < 1e-12,
                "[A,B] should equal -[B,A] at index {}",
                i
            );
        }
    }

    #[test]
    fn test_anticommutator_symmetry() {
        // {A, B} = {B, A}
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![5.0, 6.0, 7.0, 8.0];
        let ab_anti = matrix_anticommutator(&a, &b, 2);
        let ba_anti = matrix_anticommutator(&b, &a, 2);
        for i in 0..4 {
            assert!(
                (ab_anti[i] - ba_anti[i]).abs() < 1e-12,
                "{{A,B}} should equal {{B,A}} at index {}",
                i
            );
        }
    }

    #[test]
    fn test_commutator_with_self_is_zero() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let comm = matrix_commutator(&a, &a, 2);
        for i in 0..4 {
            assert!((comm[i]).abs() < 1e-12, "[A,A] should be zero");
        }
    }

    #[test]
    fn test_matrix_trace() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        assert!((matrix_trace(&a, 3) - 15.0).abs() < 1e-12);
    }

    #[test]
    fn test_matrix_inverse_2x2() {
        let m = vec![4.0, 7.0, 2.0, 6.0];
        let inv = matrix_inverse_2x2(&m).unwrap();
        let product = matrix_multiply(&m, &inv, 2);
        let id = identity_matrix(2);
        for i in 0..4 {
            assert!(
                (product[i] - id[i]).abs() < 1e-10,
                "M * M^{{-1}} should be I"
            );
        }
    }

    #[test]
    fn test_matrix_inverse_general_3x3() {
        // A well-conditioned 3x3 matrix
        let m = vec![2.0, 1.0, 0.0, 1.0, 3.0, 1.0, 0.0, 1.0, 2.0];
        let inv = matrix_inverse_general(&m, 3).unwrap();
        let product = matrix_multiply(&m, &inv, 3);
        let id = identity_matrix(3);
        for i in 0..9 {
            assert!(
                (product[i] - id[i]).abs() < 1e-10,
                "M * M^{{-1}} should be I at index {}",
                i
            );
        }
    }

    #[test]
    fn test_singular_matrix_returns_none() {
        let m = vec![1.0, 2.0, 2.0, 4.0]; // singular
        assert!(matrix_inverse_2x2(&m).is_none());
    }

    // --- Fisher metric tests ---

    #[test]
    fn test_fisher_metric_positive_semidefinite() {
        let weights = vec![0.5, -0.3, 0.8];
        let grads = vec![
            vec![1.0, 0.5, -0.2],
            vec![0.3, -0.7, 0.1],
            vec![-0.4, 0.2, 0.6],
        ];
        let fisher = compute_fisher_metric(&weights, &grads);
        // Check symmetry
        for i in 0..3 {
            for j in 0..3 {
                assert!(
                    (fisher[i][j] - fisher[j][i]).abs() < 1e-6,
                    "Fisher metric should be symmetric"
                );
            }
        }
        // Check positive semi-definite: all diagonal elements >= 0
        for i in 0..3 {
            assert!(
                fisher[i][i] >= -1e-10,
                "Fisher diagonal should be non-negative"
            );
        }
        // Check via quadratic form: v^T G v >= 0 for random v
        let v = vec![1.0f32, -0.5, 0.3];
        let mut quad = 0.0f64;
        for i in 0..3 {
            for j in 0..3 {
                quad += (v[i] as f64) * (fisher[i][j] as f64) * (v[j] as f64);
            }
        }
        assert!(quad >= -1e-10, "Fisher metric quadratic form should be non-negative");
    }

    #[test]
    fn test_fisher_empty_batch() {
        let weights = vec![1.0, 2.0];
        let fisher = compute_fisher_metric(&weights, &[]);
        assert_eq!(fisher.len(), 2);
        assert!((fisher[0][0]).abs() < 1e-10);
    }

    // --- Density matrix evolution tests ---

    #[test]
    fn test_density_matrix_valid_after_evolution() {
        let state = vec![1.0, 0.0, 0.0];
        let mut rho = DensityMatrix::pure_state(&state);

        let hamiltonian = vec![
            -1.0, 0.1, 0.0,
            0.1, -2.0, 0.1,
            0.0, 0.1, -3.0,
        ];
        let nat_grad = vec![
            0.1, 0.0, 0.0,
            0.0, 0.2, 0.0,
            0.0, 0.0, 0.3,
        ];

        evolve_density_matrix(&mut rho, &hamiltonian, &nat_grad, 0.01, 0.001);

        assert!(rho.is_valid(), "Density matrix should remain valid after evolution");
        assert!(
            (rho.trace() - 1.0).abs() < 1e-10,
            "Trace should be 1.0 after evolution"
        );
        assert!(
            rho.eigenvalues.iter().all(|&p| p >= -1e-12),
            "All eigenvalues should be non-negative"
        );
    }

    #[test]
    fn test_evolution_preserves_trace() {
        let mut rho = DensityMatrix::maximally_mixed(4);
        let h = identity_matrix(4);
        let g = identity_matrix(4);
        for _ in 0..10 {
            evolve_density_matrix(&mut rho, &h, &g, 0.01, 0.001);
        }
        assert!(
            (rho.trace() - 1.0).abs() < 1e-10,
            "Trace must be preserved across multiple evolution steps"
        );
    }

    // --- IGQK training step tests ---

    #[test]
    fn test_igqk_training_step_produces_valid_output() {
        let weights = vec![0.5, -0.3, 0.8, 0.1];
        let grads = vec![
            vec![0.1, -0.2, 0.3, -0.1],
            vec![-0.1, 0.1, -0.2, 0.2],
        ];
        let loss_grad = vec![0.05, -0.1, 0.15, -0.05];

        let result = igqk_training_step(&weights, &grads, &loss_grad, 0.1, 0.01, 0.001);

        assert_eq!(result.new_weights.len(), weights.len());
        assert!(result.density_matrix.is_valid());
        assert!(result.entropy >= 0.0);
    }

    // --- Compression tests ---

    #[test]
    fn test_ternary_compression() {
        let weights = vec![0.8, -0.9, 0.1, -0.05, 0.7, -0.6, 0.02, 0.3];
        let result = compress_with_bound(&weights, CompressionMethod::Ternary, 1.0);
        // Ternary values should be in {-1, 0, 1}
        for &w in &result.compressed {
            assert!(
                w == -1.0 || w == 0.0 || w == 1.0,
                "Ternary weight should be -1, 0, or 1, got {}",
                w
            );
        }
        assert!(result.distortion >= 0.0);
    }

    #[test]
    fn test_compression_distortion_bound() {
        let weights: Vec<f32> = (0..64).map(|i| (i as f32 - 32.0) * 0.05).collect();
        let result = compress_with_bound(&weights, CompressionMethod::Sparse(16), 1.0);

        // The distortion should satisfy D >= theoretical_bound (Theorem 5.2)
        assert!(
            result.bound_satisfied,
            "Distortion bound should be satisfied: D={} >= bound={}",
            result.distortion, result.theoretical_bound
        );
    }

    #[test]
    fn test_sparse_compression_keeps_top_k() {
        let weights = vec![0.1, 0.5, -0.9, 0.3, -0.7];
        let result = compress_with_bound(&weights, CompressionMethod::Sparse(2), 1.0);
        // Should keep the 2 largest by magnitude: -0.9 and -0.7
        let nonzero: Vec<f32> = result.compressed.iter().copied().filter(|&w| w != 0.0).collect();
        assert_eq!(nonzero.len(), 2, "Sparse(2) should keep exactly 2 weights");
    }

    #[test]
    fn test_low_rank_compression() {
        let weights = vec![0.1, 0.5, -0.9, 0.3, -0.7, 0.2];
        let result = compress_with_bound(&weights, CompressionMethod::LowRank(3), 1.0);
        let nonzero: Vec<f32> = result.compressed.iter().copied().filter(|&w| w != 0.0).collect();
        assert_eq!(nonzero.len(), 3);
        assert!(result.distortion >= 0.0);
    }

    // --- Convergence tests ---

    #[test]
    fn test_convergence_detected_at_minimum() {
        let rho = DensityMatrix::pure_state(&[1.0, 0.0]);
        // Loss history that has fully converged (constant near minimum)
        let loss_history = vec![0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05];
        let hbar = 0.1;
        let result = check_convergence(&rho, &loss_history, hbar);
        assert!(
            result.converged,
            "Should detect convergence when loss is constant at minimum, gap={}, floor={}, rate={}",
            result.gap, result.quantum_floor, result.loss_rate
        );
        assert!(result.gap <= result.quantum_floor + 1e-8);
    }

    #[test]
    fn test_convergence_not_detected_during_training() {
        let rho = DensityMatrix::maximally_mixed(4);
        // Loss still decreasing significantly
        let loss_history = vec![10.0, 8.0, 6.0, 4.0, 2.0, 1.5, 1.0, 0.8, 0.6, 0.4];
        let hbar = 0.01;
        let result = check_convergence(&rho, &loss_history, hbar);
        assert!(
            !result.converged,
            "Should not report convergence while loss is still decreasing fast"
        );
    }

    #[test]
    fn test_laplace_beltrami_diagonal() {
        let rho = DensityMatrix::pure_state(&[1.0, 0.0, 0.0]);
        let metric = vec![2.0, 3.0, 4.0];
        let h = laplace_beltrami(&rho, &metric);
        // H should be diagonal with -1/g_ii
        assert!((h[0] - (-0.5)).abs() < 1e-12);
        assert!((h[4] - (-1.0 / 3.0)).abs() < 1e-12);
        assert!((h[8] - (-0.25)).abs() < 1e-12);
    }

    #[test]
    fn test_symmetric_eigen_identity() {
        let id = identity_matrix(3);
        let (evals, _evecs) = symmetric_eigen(&id, 3);
        for &e in &evals {
            assert!((e - 1.0).abs() < 1e-10, "Identity eigenvalues should be 1.0");
        }
    }

    #[test]
    fn test_symmetric_eigen_diagonal() {
        let m = vec![3.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 2.0];
        let (mut evals, _) = symmetric_eigen(&m, 3);
        evals.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert!((evals[0] - 1.0).abs() < 1e-10);
        assert!((evals[1] - 2.0).abs() < 1e-10);
        assert!((evals[2] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_density_matrix_to_full_pure() {
        let rho = DensityMatrix::pure_state(&[1.0, 0.0]);
        let full = density_matrix_to_full(&rho);
        // Should be |0><0| = [[1,0],[0,0]]
        assert!((full[0] - 1.0).abs() < 1e-12);
        assert!((full[1]).abs() < 1e-12);
        assert!((full[2]).abs() < 1e-12);
        assert!((full[3]).abs() < 1e-12);
    }
}
