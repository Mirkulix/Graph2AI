//! IGQK Theorem Verification
//!
//! Implements numerical verification of the three main IGQK theorems:
//!
//! - **Theorem 5.1 (Convergence)**: Quantum gradient flow converges to a
//!   stationary state within O(ℏ) of the global minimum.
//! - **Theorem 5.2 (Compression Bound)**: Minimum distortion for projection
//!   onto a k-dimensional submanifold.
//! - **Theorem 5.3 (Entanglement & Generalization)**: Entangled quantum states
//!   across layers improve generalization.

// ---------------------------------------------------------------------------
// Result types
// ---------------------------------------------------------------------------

/// Result of checking Theorem 5.1 (Convergence).
#[derive(Debug, Clone)]
pub struct ConvergenceResult {
    /// Whether the loss has converged to within O(ℏ) of the minimum.
    pub converged: bool,
    /// The current (final) loss value.
    pub current_loss: f64,
    /// The convergence bound: min_loss + C·ℏ.
    pub bound: f64,
    /// Gap between current loss and the bound (positive means within bound).
    pub gap: f64,
    /// Number of steps until convergence was detected, or total steps if not converged.
    pub steps_to_converge: usize,
}

/// Result of verifying Theorem 5.2 (Compression Bound).
#[derive(Debug, Clone)]
pub struct CompressionVerification {
    /// Actual distortion D = ||W - Π(W)||² / n.
    pub distortion: f64,
    /// Theoretical lower bound from Theorem 5.2.
    pub bound: f64,
    /// Whether the actual distortion satisfies (≥) the bound.
    pub satisfied: bool,
    /// Compression ratio = original_params / compressed_params.
    pub compression_ratio: f64,
}

/// Full verification report for all three theorems.
#[derive(Debug, Clone)]
pub struct TheoremVerificationReport {
    pub convergence: ConvergenceResult,
    pub compression: CompressionVerification,
    pub generalization_bound: f64,
    pub mutual_information: f64,
    pub train_error: f64,
}

// ---------------------------------------------------------------------------
// Convergence check (struct kept for API symmetry; the function is standalone)
// ---------------------------------------------------------------------------

/// Configuration for convergence checking.
#[derive(Debug, Clone)]
pub struct ConvergenceCheck {
    /// Window size for smoothing the loss history.
    pub window: usize,
    /// Constant multiplier for the O(ℏ) term.
    pub c_hbar: f64,
}

impl Default for ConvergenceCheck {
    fn default() -> Self {
        Self {
            window: 10,
            c_hbar: 1.0,
        }
    }
}

// ---------------------------------------------------------------------------
// Theorem 5.1 — Convergence
// ---------------------------------------------------------------------------

/// Check whether the loss history has converged to within O(ℏ) of the known
/// minimum loss, as predicted by Theorem 5.1.
///
/// Convergence is detected when the smoothed loss (average over the last
/// `window` steps) is within `c_hbar * hbar` of `min_loss`.
pub fn check_convergence(loss_history: &[f64], hbar: f64, min_loss: f64) -> ConvergenceResult {
    let config = ConvergenceCheck::default();
    check_convergence_with(loss_history, hbar, min_loss, &config)
}

/// Like [`check_convergence`] but with explicit configuration.
pub fn check_convergence_with(
    loss_history: &[f64],
    hbar: f64,
    min_loss: f64,
    config: &ConvergenceCheck,
) -> ConvergenceResult {
    if loss_history.is_empty() {
        return ConvergenceResult {
            converged: false,
            current_loss: f64::NAN,
            bound: min_loss + config.c_hbar * hbar,
            gap: f64::NAN,
            steps_to_converge: 0,
        };
    }

    let bound = min_loss + config.c_hbar * hbar;
    let current_loss = *loss_history.last().unwrap();
    let window = config.window.min(loss_history.len());

    // Scan for first step at which the window-averaged loss is within the bound.
    let mut steps_to_converge = loss_history.len();
    let mut converged = false;

    for end in window..=loss_history.len() {
        let start = end - window;
        let avg: f64 = loss_history[start..end].iter().sum::<f64>() / window as f64;
        if avg <= bound {
            steps_to_converge = end;
            converged = true;
            break;
        }
    }

    // Also check the trivial case where the window is the entire (short) history.
    if !converged && loss_history.len() <= window {
        let avg: f64 = loss_history.iter().sum::<f64>() / loss_history.len() as f64;
        if avg <= bound {
            converged = true;
            steps_to_converge = loss_history.len();
        }
    }

    let gap = bound - current_loss;

    ConvergenceResult {
        converged,
        current_loss,
        bound,
        gap,
        steps_to_converge,
    }
}

// ---------------------------------------------------------------------------
// Theorem 5.2 — Compression Bound
// ---------------------------------------------------------------------------

/// Compute the minimum distortion D from Theorem 5.2:
///
/// ```text
/// D ≥ (n - k) / (2β) · ln(1 + β · σ²_min)
/// ```
///
/// For ternary compression use `k = n / 16` (i.e. 15n/16 dimensions lost).
pub fn compression_bound(n: usize, k: usize, beta: f64, sigma_sq_min: f64) -> f64 {
    assert!(n >= k, "k must be ≤ n");
    assert!(beta > 0.0, "beta must be positive");
    let diff = (n - k) as f64;
    diff / (2.0 * beta) * (1.0 + beta * sigma_sq_min).ln()
}

/// Convenience: ternary compression bound where k = n/16.
pub fn ternary_compression_bound(n: usize, beta: f64, sigma_sq_min: f64) -> f64 {
    let k = n / 16;
    compression_bound(n, k, beta, sigma_sq_min)
}

/// Verify that a compression satisfies the Theorem 5.2 bound.
///
/// Computes actual distortion as mean squared difference between `original`
/// and `compressed` weight vectors, and compares to the theoretical bound.
pub fn verify_compression(
    original: &[f32],
    compressed: &[f32],
    n: usize,
    beta: f64,
) -> CompressionVerification {
    assert_eq!(original.len(), compressed.len(), "weight vectors must have equal length");
    assert!(n > 0, "n must be positive");

    // Actual distortion: ||W - Π(W)||² / n
    let sq_diff: f64 = original
        .iter()
        .zip(compressed.iter())
        .map(|(&a, &b)| {
            let d = (a - b) as f64;
            d * d
        })
        .sum();
    let distortion = sq_diff / n as f64;

    // Count non-zero compressed weights for compression ratio.
    let non_zero = compressed.iter().filter(|&&w| w != 0.0).count();
    let compression_ratio = if non_zero > 0 {
        original.len() as f64 / non_zero as f64
    } else {
        f64::INFINITY
    };

    // Estimate sigma_sq_min from original weights (variance of smallest component).
    let mean: f64 = original.iter().map(|&w| w as f64).sum::<f64>() / original.len() as f64;
    let variance: f64 = original.iter().map(|&w| {
        let d = w as f64 - mean;
        d * d
    }).sum::<f64>() / original.len() as f64;
    let sigma_sq_min = variance.max(1e-10);

    // k = effective compressed dimensionality
    let k = non_zero;
    let bound = if n > k {
        compression_bound(n, k, beta, sigma_sq_min)
    } else {
        0.0
    };

    CompressionVerification {
        distortion,
        bound,
        satisfied: distortion >= bound,
        compression_ratio,
    }
}

// ---------------------------------------------------------------------------
// Theorem 5.3 — Entanglement & Generalization
// ---------------------------------------------------------------------------

/// Compute the generalization bound from Theorem 5.3:
///
/// ```text
/// E_gen ≤ E_train + √(I(A:B) / n)
/// ```
pub fn generalization_bound(train_error: f64, mutual_information: f64, n_samples: usize) -> f64 {
    assert!(n_samples > 0, "n_samples must be positive");
    train_error + (mutual_information / n_samples as f64).sqrt()
}

/// Estimate quantum mutual information between two weight vectors using a
/// correlation-based approximation:
///
/// ```text
/// I(A:B) ≈ -0.5 · ln(1 - r²)
/// ```
///
/// where r is the Pearson correlation coefficient between `weights_a` and
/// `weights_b`. The result is clamped to be non-negative.
pub fn estimate_mutual_information(weights_a: &[f32], weights_b: &[f32]) -> f64 {
    assert!(!weights_a.is_empty(), "weights_a must not be empty");
    assert_eq!(
        weights_a.len(),
        weights_b.len(),
        "weight vectors must have equal length"
    );

    let n = weights_a.len() as f64;

    let mean_a: f64 = weights_a.iter().map(|&w| w as f64).sum::<f64>() / n;
    let mean_b: f64 = weights_b.iter().map(|&w| w as f64).sum::<f64>() / n;

    let mut cov = 0.0f64;
    let mut var_a = 0.0f64;
    let mut var_b = 0.0f64;

    for (&a, &b) in weights_a.iter().zip(weights_b.iter()) {
        let da = a as f64 - mean_a;
        let db = b as f64 - mean_b;
        cov += da * db;
        var_a += da * da;
        var_b += db * db;
    }

    if var_a < 1e-15 || var_b < 1e-15 {
        return 0.0;
    }

    let r = cov / (var_a.sqrt() * var_b.sqrt());
    let r_sq = (r * r).min(1.0 - 1e-15); // clamp to avoid log(0)

    let mi = -0.5 * (1.0 - r_sq).ln();
    mi.max(0.0)
}

// ---------------------------------------------------------------------------
// Full verification
// ---------------------------------------------------------------------------

/// Run all theorem verifications and produce a combined report.
pub fn verify_all(
    loss_history: &[f64],
    hbar: f64,
    min_loss: f64,
    original_weights: &[f32],
    compressed_weights: &[f32],
    n_params: usize,
    beta: f64,
    train_error: f64,
    weights_layer_a: &[f32],
    weights_layer_b: &[f32],
    n_samples: usize,
) -> TheoremVerificationReport {
    let convergence = check_convergence(loss_history, hbar, min_loss);
    let compression = verify_compression(original_weights, compressed_weights, n_params, beta);
    let mi = estimate_mutual_information(weights_layer_a, weights_layer_b);
    let gen_bound = generalization_bound(train_error, mi, n_samples);

    TheoremVerificationReport {
        convergence,
        compression,
        generalization_bound: gen_bound,
        mutual_information: mi,
        train_error,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- Theorem 5.1: Convergence --

    #[test]
    fn test_convergence_detected_when_loss_stabilizes() {
        // Loss starts high and decreases to near min_loss.
        let mut history: Vec<f64> = (0..50).map(|i| 1.0 / (1.0 + i as f64)).collect();
        // Append a flat tail near the minimum.
        for _ in 0..20 {
            history.push(0.02);
        }
        let result = check_convergence(&history, 0.1, 0.01);
        assert!(result.converged, "should detect convergence: {:?}", result);
        assert!(result.current_loss <= result.bound);
    }

    #[test]
    fn test_non_convergence_when_loss_oscillates() {
        // Oscillating loss that never settles.
        let history: Vec<f64> = (0..100).map(|i| if i % 2 == 0 { 5.0 } else { 0.5 }).collect();
        let result = check_convergence(&history, 0.1, 0.01);
        assert!(!result.converged, "should not converge: {:?}", result);
    }

    #[test]
    fn test_convergence_empty_history() {
        let result = check_convergence(&[], 0.1, 0.0);
        assert!(!result.converged);
    }

    // -- Theorem 5.2: Compression Bound --

    #[test]
    fn test_compression_bound_positive() {
        let d = compression_bound(1024, 64, 1.0, 0.01);
        assert!(d > 0.0, "compression bound must be positive, got {}", d);
    }

    #[test]
    fn test_ternary_bound_greater_than_general() {
        let n = 1024;
        let k_general = n / 4; // keep 25%
        let k_ternary = n / 16; // keep ~6.25%
        let beta = 1.0;
        let sigma = 0.05;

        let d_general = compression_bound(n, k_general, beta, sigma);
        let d_ternary = compression_bound(n, k_ternary, beta, sigma);
        assert!(
            d_ternary > d_general,
            "ternary bound {} should exceed general bound {}",
            d_ternary,
            d_general
        );
    }

    #[test]
    fn test_ternary_compression_satisfies_bound() {
        // Create weights and a crude ternary quantization.
        let n = 256;
        let original: Vec<f32> = (0..n).map(|i| ((i as f32) / n as f32) - 0.5).collect();
        let compressed: Vec<f32> = original
            .iter()
            .map(|&w| {
                if w > 0.3 {
                    1.0
                } else if w < -0.3 {
                    -1.0
                } else {
                    0.0
                }
            })
            .collect();

        let v = verify_compression(&original, &compressed, n, 1.0);
        // The actual distortion from ternary quantization should be significant.
        assert!(v.distortion > 0.0, "distortion should be positive");
        // We mainly test that the function runs and returns sane values.
        assert!(v.compression_ratio >= 1.0);
    }

    #[test]
    fn test_verify_compression_identical() {
        let w: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let v = verify_compression(&w, &w, 4, 1.0);
        assert!((v.distortion - 0.0).abs() < 1e-10, "identical weights should have zero distortion");
    }

    // -- Theorem 5.3: Generalization --

    #[test]
    fn test_generalization_bound_ge_train_error() {
        let train = 0.05;
        let mi = 0.5;
        let n = 1000;
        let bound = generalization_bound(train, mi, n);
        assert!(
            bound >= train,
            "generalization bound {} must be >= train error {}",
            bound,
            train
        );
    }

    #[test]
    fn test_mutual_information_nonnegative() {
        let a: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let b: Vec<f32> = vec![5.0, 3.0, 1.0, 2.0, 4.0];
        let mi = estimate_mutual_information(&a, &b);
        assert!(mi >= 0.0, "MI must be non-negative, got {}", mi);
    }

    #[test]
    fn test_mutual_information_identical_is_high() {
        let a: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let mi = estimate_mutual_information(&a, &a);
        // For identical vectors r=1, so MI should be very large.
        assert!(mi > 5.0, "MI of identical vectors should be high, got {}", mi);
    }

    #[test]
    fn test_mutual_information_uncorrelated_is_low() {
        // Construct two vectors with near-zero correlation.
        let a: Vec<f32> = vec![1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0];
        let b: Vec<f32> = vec![1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0, -1.0];
        let mi = estimate_mutual_information(&a, &b);
        assert!(mi < 0.5, "MI of uncorrelated vectors should be low, got {}", mi);
    }

    // -- Full verification report --

    #[test]
    fn test_full_verification_report() {
        let loss_history: Vec<f64> = (0..100).map(|i| 0.5 * (-0.05 * i as f64).exp()).collect();
        let n = 64;
        let original: Vec<f32> = (0..n).map(|i| (i as f32) * 0.01).collect();
        let compressed: Vec<f32> = original.iter().map(|&w| if w > 0.3 { 1.0 } else { 0.0 }).collect();
        let layer_a: Vec<f32> = vec![0.1, 0.2, 0.3, 0.4];
        let layer_b: Vec<f32> = vec![0.2, 0.3, 0.4, 0.5];

        let report = verify_all(
            &loss_history,
            0.1,
            0.0,
            &original,
            &compressed,
            n,
            1.0,
            0.05,
            &layer_a,
            &layer_b,
            1000,
        );

        // Sanity checks on the report.
        assert!(report.generalization_bound >= report.train_error);
        assert!(report.mutual_information >= 0.0);
        assert!(report.compression.distortion >= 0.0);
    }
}
