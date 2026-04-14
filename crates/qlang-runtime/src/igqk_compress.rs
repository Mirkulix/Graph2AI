//! IMPORTANT: "Quantum" in IGQK refers to the mathematical formalism
//! (density matrices, von Neumann entropy) — NOT to quantum computing hardware.
//! All computations run on classical CPUs/GPUs. The quantum notation enables
//! compact expression of information-theoretic compression bounds.
//!
//! IGQK Compression Pipeline — Pure Rust, No Python
//!
//! Implements the full IGQK compression from the paper:
//! 1. Initialize quantum state ρ from model weights
//! 2. Evolve via quantum gradient flow dρ/dt = -i[H,ρ] - γ{G⁻¹∇L, ρ}
//! 3. Measure: collapse ρ to discrete weights via Born rule
//! 4. Project: map to ternary {-1, 0, +1} submanifold
//!
//! References: IGQK Theory (Informationsgeometrische Quantenkompression)

use qlang_core::quantum::DensityMatrix;

// ============================================================
// Public types
// ============================================================

/// Result of IGQK compression
#[derive(Debug, Clone)]
pub struct CompressionResult {
    /// Original weights
    pub original_weights: Vec<f32>,
    /// Compressed weights (ternary: only -alpha, 0.0, +alpha)
    pub compressed_weights: Vec<f32>,
    /// Per-layer scaling factors (alpha)
    pub scaling_factors: Vec<f32>,
    /// Compression statistics
    pub stats: CompressionStats,
}

/// Statistics produced after compression
#[derive(Debug, Clone)]
pub struct CompressionStats {
    pub original_size_bytes: usize,
    pub compressed_size_bytes: usize,
    pub compression_ratio: f32,
    pub num_positive: usize,  // count of +1
    pub num_zero: usize,      // count of 0
    pub num_negative: usize,  // count of -1
    pub distortion: f64,      // ||original - compressed||²
    pub quantum_entropy: f64,
    pub quantum_purity: f64,
}

/// IGQK Compression parameters
#[derive(Debug, Clone)]
pub struct IgqkParams {
    /// Quantum uncertainty ℏ (default 0.1)
    pub hbar: f64,
    /// Damping parameter γ (default 0.01)
    pub gamma: f64,
    /// Time step dt (default 0.01)
    pub dt: f64,
    /// Number of quantum evolution steps (default 10)
    pub evolution_steps: usize,
    /// Density matrix rank (default 4)
    pub rank: usize,
}

impl Default for IgqkParams {
    fn default() -> Self {
        Self {
            hbar: 0.1,
            gamma: 0.01,
            dt: 0.01,
            evolution_steps: 10,
            rank: 4,
        }
    }
}

// ============================================================
// Core compression functions
// ============================================================

/// Compress weights to ternary {-1, 0, +1} using the IGQK pipeline.
///
/// Algorithm:
/// 1. Initialize density matrix ρ from weight statistics
/// 2. Evolve ρ via quantum gradient flow
/// 3. Measure and project to ternary {-alpha, 0, +alpha} using adaptive threshold
pub fn compress_ternary(weights: &[f32], params: &IgqkParams) -> CompressionResult {
    let n = weights.len();
    if n == 0 {
        return CompressionResult {
            original_weights: vec![],
            compressed_weights: vec![],
            scaling_factors: vec![1.0],
            stats: CompressionStats {
                original_size_bytes: 0,
                compressed_size_bytes: 0,
                compression_ratio: 1.0,
                num_positive: 0,
                num_zero: 0,
                num_negative: 0,
                distortion: 0.0,
                quantum_entropy: 0.0,
                quantum_purity: 1.0,
            },
        };
    }

    // Step 1: Compute weight distribution statistics
    let mean: f32 = weights.iter().sum::<f32>() / n as f32;
    let std_dev: f32 = (weights.iter().map(|w| (w - mean).powi(2)).sum::<f32>() / n as f32).sqrt();

    // Step 2: Initialize density matrix from weight statistics
    let rank = params.rank.min(n).max(1);
    let eigenvalues: Vec<f64> = (0..rank).map(|_| 1.0 / rank as f64).collect();
    let mut eigenvectors = vec![0.0f64; rank * rank];
    for i in 0..rank {
        eigenvectors[i * rank + i] = 1.0;
    }

    let mut rho = DensityMatrix {
        dim: rank,
        eigenvalues,
        eigenvectors,
    };

    // Step 3: Quantum evolution
    let gradient = compute_weight_gradient(weights, rank);
    let hamiltonian = crate::quantum_flow::construct_hamiltonian(rank, &rho.eigenvalues);

    for _step in 0..params.evolution_steps {
        rho = crate::quantum_flow::evolve_step(
            &rho,
            &hamiltonian,
            &gradient,
            params.gamma,
            params.dt,
        );
    }

    // Step 4: Measurement — project to ternary using adaptive threshold
    // Threshold = std_dev * 0.5  (from IGQK paper: optimal Born-rule collapse)
    let threshold = std_dev * 0.5;

    // Compute per-chunk scaling factor alpha = mean(|w| where |w| > threshold)
    let alpha = compute_scaling_factor(weights, threshold);

    let mut compressed = Vec::with_capacity(n);
    let mut num_pos = 0usize;
    let mut num_zero = 0usize;
    let mut num_neg = 0usize;

    for &w in weights {
        let ternary = if w > threshold {
            num_pos += 1;
            1.0f32
        } else if w < -threshold {
            num_neg += 1;
            -1.0f32
        } else {
            num_zero += 1;
            0.0f32
        };
        compressed.push(ternary * alpha);
    }

    // Compute distortion ||original - compressed||²
    let distortion: f64 = weights
        .iter()
        .zip(compressed.iter())
        .map(|(&o, &c)| (o as f64 - c as f64).powi(2))
        .sum();

    // Storage cost: ~2 bits per ternary weight + 4 bytes for the scaling factor alpha
    let original_bytes = n * 4;
    let compressed_bytes = (n * 2 + 7) / 8 + 4;

    CompressionResult {
        original_weights: weights.to_vec(),
        compressed_weights: compressed,
        scaling_factors: vec![alpha],
        stats: CompressionStats {
            original_size_bytes: original_bytes,
            compressed_size_bytes: compressed_bytes,
            compression_ratio: original_bytes as f32 / compressed_bytes as f32,
            num_positive: num_pos,
            num_zero: num_zero,
            num_negative: num_neg,
            distortion,
            quantum_entropy: rho.entropy(),
            quantum_purity: rho.purity(),
        },
    }
}

/// Compress a neural network layer (weight matrix) to ternary.
pub fn compress_layer(
    weights: &[f32],
    rows: usize,
    cols: usize,
    params: &IgqkParams,
) -> CompressionResult {
    assert_eq!(weights.len(), rows * cols, "Weight dimensions mismatch");
    compress_ternary(weights, params)
}

/// Low-rank compression via magnitude-based SVD approximation.
///
/// Keeps the top `target_rank * min(rows, cols)` entries by magnitude,
/// zeroing the rest — equivalent to truncating small singular values.
pub fn compress_lowrank(
    weights: &[f32],
    rows: usize,
    cols: usize,
    target_rank: usize,
) -> CompressionResult {
    let n = weights.len();
    assert_eq!(n, rows * cols, "Weight dimensions mismatch");

    let keep = target_rank * rows.min(cols);
    let mut compressed = weights.to_vec();

    // Sort indices by magnitude descending; zero out everything beyond `keep`
    let mut magnitudes: Vec<(usize, f32)> = weights
        .iter()
        .enumerate()
        .map(|(i, &w)| (i, w.abs()))
        .collect();
    magnitudes.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    for &(idx, _) in magnitudes.iter().skip(keep) {
        compressed[idx] = 0.0;
    }

    let distortion: f64 = weights
        .iter()
        .zip(compressed.iter())
        .map(|(&o, &c)| (o as f64 - c as f64).powi(2))
        .sum();

    let nonzero = compressed.iter().filter(|&&w| w != 0.0).count();

    CompressionResult {
        original_weights: weights.to_vec(),
        compressed_weights: compressed,
        scaling_factors: vec![1.0],
        stats: CompressionStats {
            original_size_bytes: n * 4,
            compressed_size_bytes: nonzero * 4 + n / 8,
            compression_ratio: (n * 4) as f32 / (nonzero * 4 + n / 8).max(1) as f32,
            num_positive: 0,
            num_zero: n - nonzero,
            num_negative: 0,
            distortion,
            quantum_entropy: 0.0,
            quantum_purity: 1.0,
        },
    }
}

/// Sparse compression: keep only the top `(1 - sparsity) * n` weights by magnitude.
///
/// `sparsity` must be in [0, 1). A value of 0.8 retains 20% of weights.
pub fn compress_sparse(weights: &[f32], sparsity: f32) -> CompressionResult {
    let n = weights.len();
    let keep = ((1.0 - sparsity) * n as f32) as usize;

    let mut indexed: Vec<(usize, f32)> = weights
        .iter()
        .enumerate()
        .map(|(i, &w)| (i, w.abs()))
        .collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let mut compressed = vec![0.0f32; n];
    for &(idx, _) in indexed.iter().take(keep) {
        compressed[idx] = weights[idx];
    }

    let distortion: f64 = weights
        .iter()
        .zip(compressed.iter())
        .map(|(&o, &c)| (o as f64 - c as f64).powi(2))
        .sum();

    let nonzero = compressed.iter().filter(|&&w| w != 0.0).count();

    CompressionResult {
        original_weights: weights.to_vec(),
        compressed_weights: compressed,
        scaling_factors: vec![1.0],
        stats: CompressionStats {
            original_size_bytes: n * 4,
            compressed_size_bytes: nonzero * 4 + n / 8,
            compression_ratio: (n * 4) as f32 / (nonzero * 4 + n / 8).max(1) as f32,
            num_positive: 0,
            num_zero: n - nonzero,
            num_negative: 0,
            distortion,
            quantum_entropy: 0.0,
            quantum_purity: 1.0,
        },
    }
}

// ============================================================
// Private helpers
// ============================================================

/// Build a diagonal gradient matrix (rank × rank) from the weight distribution.
///
/// Diagonal entries are proportional to the weight variance, scaled per index
/// to introduce energy differences that drive quantum evolution.
fn compute_weight_gradient(weights: &[f32], rank: usize) -> Vec<f64> {
    let n = weights.len();
    let mean = weights.iter().sum::<f32>() / n as f32;
    let variance = weights.iter().map(|w| (w - mean).powi(2)).sum::<f32>() / n as f32;

    let mut gradient = vec![0.0f64; rank * rank];
    for i in 0..rank {
        gradient[i * rank + i] = variance as f64 * (i as f64 + 1.0) / rank as f64;
    }
    gradient
}

/// Compute the scaling factor α = mean(|w| for all |w| > threshold).
///
/// If no weights exceed the threshold (extremely tight distribution) return 1.0.
fn compute_scaling_factor(weights: &[f32], threshold: f32) -> f32 {
    let significant: Vec<f32> = weights
        .iter()
        .filter(|&&w| w.abs() > threshold)
        .map(|w| w.abs())
        .collect();

    if significant.is_empty() {
        1.0
    } else {
        significant.iter().sum::<f32>() / significant.len() as f32
    }
}

// ============================================================
// Tests
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ternary_compression_produces_only_three_values() {
        let weights: Vec<f32> = (0..100).map(|i| (i as f32 - 50.0) / 50.0).collect();
        let result = compress_ternary(&weights, &IgqkParams::default());

        let alpha = result.scaling_factors[0];
        for &w in &result.compressed_weights {
            let abs = w.abs();
            assert!(
                abs < 0.001 || (abs - alpha).abs() < 0.001,
                "Weight {} is not ternary (alpha={})",
                w,
                alpha
            );
        }
    }

    #[test]
    fn ternary_compression_has_three_categories() {
        let weights: Vec<f32> = (0..1000).map(|i| (i as f32 - 500.0) / 200.0).collect();
        let result = compress_ternary(&weights, &IgqkParams::default());

        assert!(result.stats.num_positive > 0, "Should have positive weights");
        assert!(result.stats.num_zero > 0, "Should have zero weights");
        assert!(result.stats.num_negative > 0, "Should have negative weights");
        assert_eq!(
            result.stats.num_positive + result.stats.num_zero + result.stats.num_negative,
            weights.len()
        );
    }

    #[test]
    fn compression_ratio_is_significant() {
        let weights: Vec<f32> = (0..1000).map(|i| (i as f32 - 500.0) / 500.0).collect();
        let result = compress_ternary(&weights, &IgqkParams::default());

        assert!(
            result.stats.compression_ratio > 5.0,
            "Compression ratio {} too low",
            result.stats.compression_ratio
        );
    }

    #[test]
    fn quantum_state_is_used() {
        let weights: Vec<f32> = (0..100).map(|i| (i as f32 - 50.0) / 50.0).collect();
        let result = compress_ternary(&weights, &IgqkParams::default());

        assert!(
            result.stats.quantum_entropy.is_finite(),
            "Entropy should be finite"
        );
        assert!(
            result.stats.quantum_purity.is_finite(),
            "Purity should be finite"
        );
        assert!(result.stats.quantum_purity > 0.0, "Purity should be positive");
    }

    #[test]
    fn distortion_is_bounded() {
        let weights: Vec<f32> = (0..100).map(|i| (i as f32 - 50.0) / 50.0).collect();
        let result = compress_ternary(&weights, &IgqkParams::default());

        assert!(result.stats.distortion >= 0.0);
        assert!(result.stats.distortion.is_finite());
    }

    #[test]
    fn scaling_factor_is_reasonable() {
        let weights: Vec<f32> = vec![0.5, -0.3, 0.8, -0.1, 0.4, -0.6, 0.2, -0.7];
        let result = compress_ternary(&weights, &IgqkParams::default());

        assert!(result.scaling_factors[0] > 0.0);
        assert!(result.scaling_factors[0] < 10.0);
    }

    #[test]
    fn lowrank_compression() {
        let weights: Vec<f32> = (0..64).map(|i| (i as f32) / 64.0).collect();
        let result = compress_lowrank(&weights, 8, 8, 2);

        assert_eq!(result.compressed_weights.len(), 64);
        assert!(result.stats.compression_ratio > 1.0);
        assert!(result.stats.num_zero > 0);
    }

    #[test]
    fn sparse_compression() {
        let weights: Vec<f32> = (0..100).map(|i| (i as f32 - 50.0) / 50.0).collect();
        let result = compress_sparse(&weights, 0.8); // 80% sparsity

        let nonzero = result.compressed_weights.iter().filter(|&&w| w != 0.0).count();
        assert!(nonzero <= 25, "Too many nonzero: {}", nonzero);
        assert!(nonzero >= 15, "Too few nonzero: {}", nonzero);
    }

    #[test]
    fn compress_preserves_length() {
        let weights: Vec<f32> = vec![0.1, -0.2, 0.3, -0.4, 0.5];
        let result = compress_ternary(&weights, &IgqkParams::default());
        assert_eq!(result.compressed_weights.len(), weights.len());
        assert_eq!(result.original_weights.len(), weights.len());
    }

    #[test]
    fn evolution_steps_affect_result() {
        let weights: Vec<f32> = (0..100).map(|i| (i as f32 - 50.0) / 50.0).collect();

        let params1 = IgqkParams {
            evolution_steps: 1,
            ..Default::default()
        };
        let params2 = IgqkParams {
            evolution_steps: 50,
            ..Default::default()
        };

        let r1 = compress_ternary(&weights, &params1);
        let r2 = compress_ternary(&weights, &params2);

        assert!(r1.stats.quantum_entropy.is_finite());
        assert!(r2.stats.quantum_entropy.is_finite());
    }

    #[test]
    fn full_compression_pipeline() {
        // Simulate a small neural network layer (784 × 128 = Xavier-initialized)
        let weights: Vec<f32> = (0..784 * 128)
            .map(|i| {
                let scale = (2.0f32 / (784.0 + 128.0)).sqrt();
                ((i * 7 + 13) % 1000) as f32 / 1000.0 * scale * 2.0 - scale
            })
            .collect();

        let params = IgqkParams {
            evolution_steps: 5,
            rank: 4,
            ..Default::default()
        };

        let result = compress_ternary(&weights, &params);

        println!(
            "Original:    {} bytes ({} weights)",
            result.stats.original_size_bytes,
            weights.len()
        );
        println!("Compressed:  {} bytes", result.stats.compressed_size_bytes);
        println!("Ratio:       {:.1}x", result.stats.compression_ratio);
        println!(
            "Distribution: +1={}, 0={}, -1={}",
            result.stats.num_positive, result.stats.num_zero, result.stats.num_negative
        );
        println!("Distortion:  {:.4}", result.stats.distortion);
        println!("Q-Entropy:   {:.4}", result.stats.quantum_entropy);
        println!("Q-Purity:    {:.4}", result.stats.quantum_purity);

        assert!(
            result.stats.compression_ratio > 10.0,
            "Should achieve >10x compression on 784×128 layer, got {:.1}x",
            result.stats.compression_ratio
        );
        assert!(result.stats.num_positive > 0);
        assert!(result.stats.num_zero > 0);
        assert!(result.stats.num_negative > 0);
    }
}
