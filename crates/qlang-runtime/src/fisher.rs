//! Fisher Information Metric
//!
//! Implements the Fisher Information Metric from IGQK theory:
//!   G_ij(θ) = E_θ[∂_i log p(x;θ) · ∂_j log p(x;θ)]
//!
//! This module provides:
//! - Numerical gradient computation via finite differences
//! - Empirical Fisher metric estimation
//! - Fisher matrix inversion with damping for numerical stability
//! - Natural gradient: G⁻¹ · ∇L (follows geodesics on the statistical manifold)

/// Configuration for Fisher metric computation.
#[derive(Debug, Clone)]
pub struct FisherMetricConfig {
    /// Finite difference step size for numerical gradients.
    pub eps: f32,
    /// Regularization damping λ added to diagonal: G + λI.
    pub damping: f64,
    /// Number of data samples used for empirical Fisher estimation.
    pub n_samples: usize,
}

impl Default for FisherMetricConfig {
    fn default() -> Self {
        Self {
            eps: 1e-4,
            damping: 1e-6,
            n_samples: 100,
        }
    }
}

/// Compute numerical gradients of forward_fn output w.r.t. each weight parameter
/// using central finite differences.
///
/// Returns a Vec of length `weights.len()`, where each element is a Vec<f32>
/// containing the partial derivative of each output w.r.t. that weight.
pub fn compute_gradients(
    weights: &[f32],
    forward_fn: fn(&[f32], &[f32]) -> Vec<f32>,
    input: &[f32],
    eps: f32,
) -> Vec<Vec<f32>> {
    let n = weights.len();
    let mut grads = Vec::with_capacity(n);

    for i in 0..n {
        // Perturb weight i in both directions
        let mut w_plus = weights.to_vec();
        let mut w_minus = weights.to_vec();
        w_plus[i] += eps;
        w_minus[i] -= eps;

        let out_plus = forward_fn(&w_plus, input);
        let out_minus = forward_fn(&w_minus, input);

        // Central difference: (f(θ+ε) - f(θ-ε)) / (2ε)
        let grad: Vec<f32> = out_plus
            .iter()
            .zip(out_minus.iter())
            .map(|(&p, &m)| (p - m) / (2.0 * eps))
            .collect();

        grads.push(grad);
    }

    grads
}

/// Compute the empirical Fisher Information Metric.
///
/// G = (1/N) Σ_n (∇ log p_n)(∇ log p_n)^T
///
/// Returns a flat n×n matrix (row-major) where n = weights.len().
///
/// For each data sample, we compute numerical gradients of the log-probability
/// (log of the forward function output), then accumulate the outer product.
pub fn empirical_fisher(
    weights: &[f32],
    forward_fn: fn(&[f32], &[f32]) -> Vec<f32>,
    data: &[Vec<f32>],
    n_samples: usize,
) -> Vec<f64> {
    let n = weights.len();
    let mut fisher = vec![0.0f64; n * n];
    let samples = n_samples.min(data.len());

    if samples == 0 {
        return fisher;
    }

    for s in 0..samples {
        let input = &data[s];

        // Compute gradients of output w.r.t. weights
        let grads = compute_gradients(weights, forward_fn, input, 1e-4);

        // For each output dimension, the gradient of log p w.r.t. weight i
        // is (∂ output / ∂ θ_i) / output. We compute the log-probability
        // gradient as ∂ log p / ∂ θ_i = (1/p) * ∂p/∂θ_i.
        let output = forward_fn(weights, input);

        // For each output dimension k, compute ∂ log p_k / ∂ θ_i and
        // accumulate the outer product. This keeps contributions from
        // independent output dimensions separate.
        for k in 0..output.len() {
            let p = output[k].max(1e-10) as f64;
            // grad_log_p_k[i] = (1/p_k) * ∂p_k/∂θ_i
            let grad_log_p_k: Vec<f64> = (0..n)
                .map(|i| (grads[i][k] as f64) / p)
                .collect();

            // Accumulate outer product: G += grad_log_p_k * grad_log_p_k^T
            for i in 0..n {
                for j in 0..n {
                    fisher[i * n + j] += grad_log_p_k[i] * grad_log_p_k[j];
                }
            }
        }
    }

    // Average over samples
    let inv_n = 1.0 / (samples as f64);
    for v in fisher.iter_mut() {
        *v *= inv_n;
    }

    fisher
}

/// Invert the Fisher metric matrix G⁻¹ with diagonal damping for numerical stability.
///
/// Computes (G + λI)⁻¹ using Gauss-Jordan elimination.
///
/// `fisher` is a flat n×n row-major matrix, `n` is the dimension.
/// Returns the inverse as a flat n×n row-major matrix.
pub fn fisher_inverse(fisher: &[f64], n: usize) -> Vec<f64> {
    fisher_inverse_with_damping(fisher, n, 1e-6)
}

/// Invert the Fisher metric with a specified damping factor.
fn fisher_inverse_with_damping(fisher: &[f64], n: usize, damping: f64) -> Vec<f64> {
    assert_eq!(fisher.len(), n * n, "Fisher matrix must be n×n");

    // Build augmented matrix [G + λI | I]
    let mut aug = vec![0.0f64; n * 2 * n];
    for i in 0..n {
        for j in 0..n {
            aug[i * 2 * n + j] = fisher[i * n + j];
        }
        // Add damping to diagonal
        aug[i * 2 * n + i] += damping;
        // Identity on the right
        aug[i * 2 * n + n + i] = 1.0;
    }

    // Gauss-Jordan elimination with partial pivoting
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

        // Swap rows if needed
        if max_row != col {
            for k in 0..(2 * n) {
                let tmp = aug[col * 2 * n + k];
                aug[col * 2 * n + k] = aug[max_row * 2 * n + k];
                aug[max_row * 2 * n + k] = tmp;
            }
        }

        let pivot = aug[col * 2 * n + col];
        assert!(pivot.abs() > 1e-15, "Fisher matrix is singular even with damping");

        // Scale pivot row
        let inv_pivot = 1.0 / pivot;
        for k in 0..(2 * n) {
            aug[col * 2 * n + k] *= inv_pivot;
        }

        // Eliminate column in all other rows
        for row in 0..n {
            if row == col {
                continue;
            }
            let factor = aug[row * 2 * n + col];
            for k in 0..(2 * n) {
                aug[row * 2 * n + k] -= factor * aug[col * 2 * n + k];
            }
        }
    }

    // Extract inverse from augmented matrix
    let mut inv = vec![0.0f64; n * n];
    for i in 0..n {
        for j in 0..n {
            inv[i * n + j] = aug[i * 2 * n + n + j];
        }
    }

    inv
}

/// Compute the natural gradient: G⁻¹ · ∇L
///
/// The natural gradient follows geodesics on the statistical manifold,
/// providing an information-geometrically optimal descent direction.
///
/// `gradient` is the Euclidean gradient ∇L of length n.
/// `fisher_inv` is G⁻¹ as a flat n×n row-major matrix.
/// Returns G⁻¹ · ∇L as a Vec of length n.
pub fn natural_gradient(gradient: &[f64], fisher_inv: &[f64], n: usize) -> Vec<f64> {
    assert_eq!(gradient.len(), n, "Gradient length must match n");
    assert_eq!(fisher_inv.len(), n * n, "Fisher inverse must be n×n");

    let mut result = vec![0.0f64; n];
    for i in 0..n {
        let mut sum = 0.0f64;
        for j in 0..n {
            sum += fisher_inv[i * n + j] * gradient[j];
        }
        result[i] = sum;
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Simple linear forward function: output = weights · input (dot product)
    fn linear_forward(weights: &[f32], input: &[f32]) -> Vec<f32> {
        let dot: f32 = weights.iter().zip(input.iter()).map(|(w, x)| w * x).sum();
        // Return softmax-like positive output to serve as "probability"
        vec![(dot.exp()) / (1.0 + dot.exp())]
    }

    /// Identity-like forward: each weight independently maps to its own output,
    /// scaled by corresponding input element to create data-dependent variation.
    fn identity_forward(weights: &[f32], input: &[f32]) -> Vec<f32> {
        weights
            .iter()
            .enumerate()
            .map(|(i, &w)| {
                let x = if i < input.len() { input[i] } else { 1.0 };
                let z = w * x;
                z.exp() / (1.0 + z.exp())
            })
            .collect()
    }

    fn make_data(n_samples: usize, dim: usize) -> Vec<Vec<f32>> {
        let mut data = Vec::new();
        for i in 0..n_samples {
            let val = (i as f32 + 1.0) / (n_samples as f32 + 1.0);
            data.push(vec![val; dim]);
        }
        data
    }

    #[test]
    fn test_fisher_symmetric() {
        let weights = vec![0.5, -0.3, 0.8];
        let data = make_data(20, 3);
        let fisher = empirical_fisher(&weights, linear_forward, &data, 20);
        let n = weights.len();

        for i in 0..n {
            for j in 0..n {
                let diff = (fisher[i * n + j] - fisher[j * n + i]).abs();
                assert!(
                    diff < 1e-10,
                    "Fisher not symmetric at ({},{}): {} vs {}",
                    i, j, fisher[i * n + j], fisher[j * n + i]
                );
            }
        }
    }

    #[test]
    fn test_fisher_positive_semidefinite_diagonal() {
        let weights = vec![0.5, -0.3, 0.8];
        let data = make_data(20, 3);
        let fisher = empirical_fisher(&weights, linear_forward, &data, 20);
        let n = weights.len();

        for i in 0..n {
            assert!(
                fisher[i * n + i] >= 0.0,
                "Diagonal element {} is negative: {}",
                i, fisher[i * n + i]
            );
        }
    }

    #[test]
    fn test_natural_gradient_dimension() {
        let n = 4;
        let gradient = vec![1.0, 2.0, 3.0, 4.0];
        // Use identity as Fisher inverse
        let mut fisher_inv = vec![0.0f64; n * n];
        for i in 0..n {
            fisher_inv[i * n + i] = 1.0;
        }

        let ng = natural_gradient(&gradient, &fisher_inv, n);
        assert_eq!(ng.len(), n, "Natural gradient must have same dimension as gradient");
    }

    #[test]
    fn test_identity_model_fisher_near_diagonal() {
        // For an identity-like model where each weight independently controls
        // one output, the Fisher metric should be approximately diagonal.
        let weights = vec![0.0, 0.0, 0.0];
        let data = make_data(50, 3);
        let fisher = empirical_fisher(&weights, identity_forward, &data, 50);
        let n = weights.len();

        // Check off-diagonal elements are much smaller than diagonal
        for i in 0..n {
            for j in 0..n {
                if i != j {
                    let ratio = fisher[i * n + j].abs()
                        / (fisher[i * n + i].abs().max(1e-15));
                    assert!(
                        ratio < 0.1,
                        "Off-diagonal ({},{}) ratio {} too large",
                        i, j, ratio
                    );
                }
            }
        }
    }

    #[test]
    fn test_inverse_times_original_is_identity() {
        let n = 3;
        // Build a known positive-definite matrix
        let mat = vec![
            4.0, 1.0, 0.5,
            1.0, 3.0, 0.2,
            0.5, 0.2, 2.0,
        ];

        let inv = fisher_inverse_with_damping(&mat, n, 0.0);

        // Compute mat * inv and check it's close to identity
        for i in 0..n {
            for j in 0..n {
                let mut val = 0.0;
                for k in 0..n {
                    val += mat[i * n + k] * inv[k * n + j];
                }
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (val - expected).abs() < 1e-10,
                    "Product[{},{}] = {}, expected {}",
                    i, j, val, expected
                );
            }
        }
    }

    #[test]
    fn test_damping_prevents_singular() {
        let n = 2;
        // Singular matrix (rank 1)
        let mat = vec![
            1.0, 2.0,
            2.0, 4.0,
        ];

        // Without damping this would fail; with damping it should succeed
        let inv = fisher_inverse(&mat, n); // uses default damping 1e-6
        // Just verify we got a result of the right size
        assert_eq!(inv.len(), n * n);

        // Verify (G + λI) * inv ≈ I
        let damping = 1e-6;
        let mut damped = mat.clone();
        damped[0] += damping;
        damped[3] += damping;

        for i in 0..n {
            for j in 0..n {
                let mut val = 0.0;
                for k in 0..n {
                    val += damped[i * n + k] * inv[k * n + j];
                }
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (val - expected).abs() < 1e-6,
                    "Damped product[{},{}] = {}, expected {}",
                    i, j, val, expected
                );
            }
        }
    }

    #[test]
    fn test_natural_gradient_improves_direction() {
        // When the Fisher metric is non-isotropic, the natural gradient
        // should differ from the raw gradient. For a diagonal Fisher with
        // unequal entries, natural gradient rescales components.
        let n = 2;
        let gradient = vec![1.0, 1.0];

        // Fisher metric is diagonal with different scales
        let fisher = vec![
            4.0, 0.0,
            0.0, 1.0,
        ];
        let fisher_inv = fisher_inverse_with_damping(&fisher, n, 0.0);
        let ng = natural_gradient(&gradient, &fisher_inv, n);

        // Natural gradient should scale inversely with Fisher diagonal:
        // ng[0] ≈ 1/4 = 0.25, ng[1] ≈ 1/1 = 1.0
        assert!(
            (ng[0] - 0.25).abs() < 1e-10,
            "ng[0] = {}, expected 0.25", ng[0]
        );
        assert!(
            (ng[1] - 1.0).abs() < 1e-10,
            "ng[1] = {}, expected 1.0", ng[1]
        );

        // The natural gradient should differ from the raw gradient
        assert!(
            (ng[0] - gradient[0]).abs() > 0.1,
            "Natural gradient should differ from raw gradient"
        );
    }

    #[test]
    fn test_config_defaults_sensible() {
        let config = FisherMetricConfig::default();

        assert!(config.eps > 0.0, "eps must be positive");
        assert!(config.eps < 1.0, "eps must be small");
        assert!(config.damping > 0.0, "damping must be positive");
        assert!(config.damping < 1.0, "damping must be small");
        assert!(config.n_samples > 0, "n_samples must be positive");
        assert!(config.n_samples <= 10000, "n_samples should be reasonable");
    }

    #[test]
    fn test_compute_gradients_basic() {
        // For f(w, x) = [w0*x0 + w1*x1] (linear), gradients should be [x0, x1]
        fn linear(weights: &[f32], input: &[f32]) -> Vec<f32> {
            let dot: f32 = weights.iter().zip(input.iter()).map(|(w, x)| w * x).sum();
            vec![dot]
        }

        let weights = vec![1.0, 2.0];
        let input = vec![3.0, 4.0];
        let grads = compute_gradients(&weights, linear, &input, 1e-4);

        assert_eq!(grads.len(), 2);
        assert!((grads[0][0] - 3.0).abs() < 0.05, "dF/dw0 should be x0=3.0, got {}", grads[0][0]);
        assert!((grads[1][0] - 4.0).abs() < 0.05, "dF/dw1 should be x1=4.0, got {}", grads[1][0]);
    }
}
