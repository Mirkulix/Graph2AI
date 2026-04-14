//! Ternary Matrix Weights — multi-resolution ternary representation.
//!
//! Core idea: Instead of one f32 per weight, use a d×d ternary matrix.
//! The effective scalar weight is derived via a reduction function φ(M).
//!
//! A 2×2 ternary matrix has 81 states = 9 effective weight levels.
//! A 3×3 ternary matrix has 19,683 states = 19 effective weight levels.
//! A 4×4 ternary matrix has ~43M states ≈ 25 bits of resolution.
//!
//! This is analogous to Sigma-Delta modulation: many low-bit values
//! encoding a high-resolution signal through spatial structure.
//!
//! Mathematical framework:
//! - Weight space: {-1, 0, +1}^(d×d) — a discrete lattice per weight
//! - Reduction map: φ: {-1,0,+1}^(d×d) → ℝ (mean, FFT, eigenvalue, etc.)
//! - Training: flip one matrix element, keep if loss improves
//! - Forward pass: compute φ(M) for each weight, then standard matmul
//!
//! Connections:
//! - Sigma-Delta modulation (1-bit DAC → 24-bit audio)
//! - Finite Element Method (continuous field → discrete mesh)
//! - Ising model (spin lattice → macroscopic magnetization)
//! - Holographic memory (interference patterns → stored information)

use rayon::prelude::*;

/// Dimension of the ternary sub-matrix per weight.
/// 2 = 81 states (9 levels), 3 = 19683 states (19 levels)
pub const MATRIX_DIM: usize = 2;
pub const MATRIX_SIZE: usize = MATRIX_DIM * MATRIX_DIM; // elements per weight

/// A ternary sub-matrix stored as flat i8 array.
type SubMatrix = [i8; MATRIX_SIZE];

// ============================================================
// Reduction functions: φ(M) → scalar weight
// ============================================================

/// Mean reduction: φ(M) = Σ M_ij / d²
/// For 2×2: gives values in {-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1}
#[inline]
fn phi_mean(m: &SubMatrix) -> f32 {
    let sum: i32 = m.iter().map(|&v| v as i32).sum();
    sum as f32 / MATRIX_SIZE as f32
}

/// Weighted reduction using position-dependent coefficients.
/// Inspired by wavelet decomposition: different positions capture different frequencies.
#[inline]
#[allow(dead_code)] // Alternative scoring method for ternary matrix decomposition
fn phi_wavelet(m: &SubMatrix) -> f32 {
    // For 2×2: coefficients are [1, 1/2, 1/2, 1/4] (low → high frequency)
    // This gives finer granularity than simple mean
    const COEFFS_2X2: [f32; 4] = [0.4, 0.3, 0.2, 0.1];
    const COEFFS_3X3: [f32; 9] = [0.2, 0.15, 0.1, 0.15, 0.1, 0.08, 0.1, 0.08, 0.04];

    let coeffs = if MATRIX_SIZE == 4 { &COEFFS_2X2[..] } else { &COEFFS_3X3[..MATRIX_SIZE.min(9)] };

    let mut sum = 0.0f32;
    for (i, &v) in m.iter().enumerate() {
        if i < coeffs.len() {
            sum += v as f32 * coeffs[i];
        }
    }
    sum
}

// ============================================================
// Ternary Matrix Layer
// ============================================================

pub struct MatrixLayer {
    /// Sub-matrices: [out_dim × in_dim × MATRIX_SIZE] as i8
    pub matrices: Vec<i8>,
    /// Cached effective weights [out_dim × in_dim] as f32
    effective: Vec<f32>,
    /// Biases [out_dim]
    pub biases: Vec<f32>,
    /// Scale factor per layer
    pub scale: f32,
    pub in_dim: usize,
    pub out_dim: usize,
    pub n_weights: usize,
}

impl MatrixLayer {
    pub fn new(in_dim: usize, out_dim: usize, seed: u64) -> Self {
        let n_weights = out_dim * in_dim;
        let total_elements = n_weights * MATRIX_SIZE;

        // Initialize random ternary matrices
        let matrices: Vec<i8> = (0..total_elements)
            .map(|i| {
                let v = ((i as u64).wrapping_mul(seed).wrapping_mul(2654435761) >> 30) % 3;
                match v { 0 => -1i8, 1 => 0, _ => 1 }
            })
            .collect();

        // Compute initial effective weights
        let mut effective = vec![0.0f32; n_weights];
        for w in 0..n_weights {
            let sub: SubMatrix = {
                let mut s = [0i8; MATRIX_SIZE];
                s.copy_from_slice(&matrices[w * MATRIX_SIZE..(w + 1) * MATRIX_SIZE]);
                s
            };
            effective[w] = phi_mean(&sub);
        }

        Self {
            matrices,
            effective,
            biases: vec![0.0; out_dim],
            scale: 1.0,
            in_dim,
            out_dim,
            n_weights,
        }
    }

    /// Recompute all effective weights from matrices.
    pub fn sync_effective(&mut self) {
        for w in 0..self.n_weights {
            let sub: SubMatrix = {
                let mut s = [0i8; MATRIX_SIZE];
                s.copy_from_slice(&self.matrices[w * MATRIX_SIZE..(w + 1) * MATRIX_SIZE]);
                s
            };
            self.effective[w] = phi_mean(&sub);
        }

        // Update scale: normalize effective weights
        let mean_abs: f32 = self.effective.iter().map(|w| w.abs()).sum::<f32>()
            / self.effective.len().max(1) as f32;
        if mean_abs > 0.0 {
            self.scale = (2.0 / (self.in_dim + self.out_dim) as f32).sqrt() / mean_abs;
        }
    }

    /// Forward pass using cached effective weights.
    pub fn forward(&self, input: &[f32], batch: usize) -> Vec<f32> {
        let out_dim = self.out_dim;
        let in_dim = self.in_dim;
        let scale = self.scale;
        let eff = &self.effective;
        let biases = &self.biases;

        let chunks: Vec<Vec<f32>> = (0..batch)
            .into_par_iter()
            .map(|b| {
                let x = &input[b * in_dim..(b + 1) * in_dim];
                let mut out = vec![0.0f32; out_dim];
                for j in 0..out_dim {
                    let mut sum = biases[j];
                    for k in 0..in_dim {
                        sum += scale * eff[j * in_dim + k] * x[k];
                    }
                    out[j] = sum.max(0.0); // ReLU
                }
                out
            })
            .collect();

        let mut output = vec![0.0f32; batch * out_dim];
        for (b, chunk) in chunks.into_iter().enumerate() {
            output[b * out_dim..(b + 1) * out_dim].copy_from_slice(&chunk);
        }
        output
    }

    /// Goodness (sum of squared activations).
    pub fn goodness(act: &[f32], batch: usize, dim: usize) -> f32 {
        let mut total = 0.0f32;
        for b in 0..batch {
            let off = b * dim;
            total += act[off..off + dim].iter().map(|x| x * x).sum::<f32>();
        }
        total / batch as f32
    }

    pub fn normalize(act: &mut [f32], batch: usize, dim: usize) {
        for b in 0..batch {
            let off = b * dim;
            let norm: f32 = act[off..off + dim].iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-8);
            for j in 0..dim { act[off + j] /= norm; }
        }
    }

    /// Train one step: flip random matrix elements, keep if goodness improves.
    /// This is the ternary equivalent of gradient descent — but fully discrete.
    ///
    /// `pos_input` / `neg_input`: positive and negative samples (Forward-Forward style)
    /// `n_flips`: how many matrix elements to try flipping
    pub fn train_step(
        &mut self,
        pos_input: &[f32],
        neg_input: &[f32],
        batch: usize,
        n_flips: usize,
        rng_seed: u64,
    ) -> (f32, f32, usize) {
        // Measure current goodness
        let pos_act = self.forward(pos_input, batch);
        let neg_act = self.forward(neg_input, batch);
        let pos_g = Self::goodness(&pos_act, batch, self.out_dim);
        let neg_g = Self::goodness(&neg_act, batch, self.out_dim);
        let current_gap = pos_g - neg_g; // want this to be large

        let mut improved = 0;
        let total_elements = self.matrices.len();

        for flip in 0..n_flips {
            // Pick random matrix element
            let idx = ((rng_seed.wrapping_mul(2654435761 + flip as u64 * 7919)) % total_elements as u64) as usize;
            let old_val = self.matrices[idx];

            // Flip to different value
            let new_val: i8 = match old_val {
                1 => if flip % 2 == 0 { 0 } else { -1 },
                -1 => if flip % 2 == 0 { 0 } else { 1 },
                _ => if flip % 2 == 0 { 1 } else { -1 },
            };

            // Apply flip
            self.matrices[idx] = new_val;

            // Recompute the affected effective weight
            let weight_idx = idx / MATRIX_SIZE;
            let old_eff = self.effective[weight_idx];
            let sub: SubMatrix = {
                let mut s = [0i8; MATRIX_SIZE];
                let off = weight_idx * MATRIX_SIZE;
                s.copy_from_slice(&self.matrices[off..off + MATRIX_SIZE]);
                s
            };
            self.effective[weight_idx] = phi_mean(&sub);

            // Measure new goodness
            let new_pos_act = self.forward(pos_input, batch);
            let new_neg_act = self.forward(neg_input, batch);
            let new_pos_g = Self::goodness(&new_pos_act, batch, self.out_dim);
            let new_neg_g = Self::goodness(&new_neg_act, batch, self.out_dim);
            let new_gap = new_pos_g - new_neg_g;

            if new_gap > current_gap {
                // Improvement — keep the flip
                improved += 1;
            } else {
                // Revert
                self.matrices[idx] = old_val;
                self.effective[weight_idx] = old_eff;
            }
        }

        // Update biases via simple goodness-driven rule
        let pos_act = self.forward(pos_input, batch);
        let neg_act = self.forward(neg_input, batch);
        for j in 0..self.out_dim {
            let mut pos_sum = 0.0f32;
            let mut neg_sum = 0.0f32;
            for b in 0..batch {
                pos_sum += pos_act[b * self.out_dim + j];
                neg_sum += neg_act[b * self.out_dim + j];
            }
            // Push biases to increase pos goodness, decrease neg goodness
            self.biases[j] += 0.001 * (pos_sum - neg_sum) / batch as f32;
        }

        (pos_g, neg_g, improved)
    }

    /// Statistics about the ternary matrix elements.
    pub fn stats(&self) -> (usize, usize, usize) {
        let pos = self.matrices.iter().filter(|&&v| v == 1).count();
        let zero = self.matrices.iter().filter(|&&v| v == 0).count();
        let neg = self.matrices.iter().filter(|&&v| v == -1).count();
        (pos, zero, neg)
    }

    /// Effective weight distribution statistics.
    pub fn effective_stats(&self) -> (f32, f32, usize) {
        let min = self.effective.iter().cloned().fold(f32::MAX, f32::min);
        let max = self.effective.iter().cloned().fold(f32::MIN, f32::max);
        let unique: std::collections::HashSet<i32> = self.effective.iter()
            .map(|&w| (w * 10000.0) as i32)
            .collect();
        (min, max, unique.len())
    }

    /// Storage size in bytes (i8 per matrix element + f32 per bias)
    pub fn size_bytes(&self) -> usize {
        self.matrices.len() + self.biases.len() * 4
    }
}

// ============================================================
// Full Matrix-Ternary Network
// ============================================================

pub struct MatrixNet {
    pub layers: Vec<MatrixLayer>,
    pub n_classes: usize,
}

impl MatrixNet {
    pub fn new(layer_sizes: &[usize], n_classes: usize) -> Self {
        let mut layers = Vec::new();
        for i in 0..layer_sizes.len() - 1 {
            layers.push(MatrixLayer::new(
                layer_sizes[i], layer_sizes[i + 1],
                37 + i as u64 * 17,
            ));
        }
        Self { layers, n_classes }
    }

    pub fn embed_label(images: &[f32], labels: &[u8], image_dim: usize, n_classes: usize, batch: usize) -> Vec<f32> {
        let total_dim = image_dim + n_classes;
        let mut out = vec![0.0f32; batch * total_dim];
        for b in 0..batch {
            out[b * total_dim..b * total_dim + image_dim]
                .copy_from_slice(&images[b * image_dim..(b + 1) * image_dim]);
            let label = labels[b] as usize;
            if label < n_classes {
                out[b * total_dim + image_dim + label] = 1.0;
            }
        }
        out
    }

    fn make_negative(images: &[f32], labels: &[u8], image_dim: usize, n_classes: usize, batch: usize) -> Vec<f32> {
        let mut wrong = labels.to_vec();
        for b in 0..batch {
            wrong[b] = ((labels[b] as usize + 1 + (b * 7 + 3) % (n_classes - 1)) % n_classes) as u8;
        }
        Self::embed_label(images, &wrong, image_dim, n_classes, batch)
    }

    /// Train one epoch.
    pub fn train_epoch(
        &mut self,
        images: &[f32],
        labels: &[u8],
        image_dim: usize,
        n_samples: usize,
        batch_size: usize,
        flips_per_batch: usize,
    ) -> (f32, f32) {
        let n_batches = n_samples / batch_size;
        let mut total_pos = 0.0f32;
        let mut total_neg = 0.0f32;

        for batch_idx in 0..n_batches {
            let offset = batch_idx * batch_size;
            let batch_images = &images[offset * image_dim..(offset + batch_size) * image_dim];
            let batch_labels = &labels[offset..offset + batch_size];

            let pos = Self::embed_label(batch_images, batch_labels, image_dim, self.n_classes, batch_size);
            let neg = Self::make_negative(batch_images, batch_labels, image_dim, self.n_classes, batch_size);

            let mut pos_in = pos;
            let mut neg_in = neg;

            for (i, layer) in self.layers.iter_mut().enumerate() {
                let seed = (batch_idx * 1000 + i) as u64;
                let (pg, ng, _improved) = layer.train_step(&pos_in, &neg_in, batch_size, flips_per_batch, seed);
                total_pos += pg;
                total_neg += ng;

                let mut pos_out = layer.forward(&pos_in, batch_size);
                MatrixLayer::normalize(&mut pos_out, batch_size, layer.out_dim);
                let mut neg_out = layer.forward(&neg_in, batch_size);
                MatrixLayer::normalize(&mut neg_out, batch_size, layer.out_dim);

                pos_in = pos_out;
                neg_in = neg_out;
            }
        }

        // Sync effective weights
        for layer in &mut self.layers {
            layer.sync_effective();
        }

        let count = (n_batches * self.layers.len()).max(1) as f32;
        (total_pos / count, total_neg / count)
    }

    pub fn predict(&self, images: &[f32], image_dim: usize, n_samples: usize) -> Vec<u8> {
        (0..n_samples)
            .into_par_iter()
            .map(|i| {
                let image = &images[i * image_dim..(i + 1) * image_dim];
                let mut best = 0u8;
                let mut best_g = f32::NEG_INFINITY;

                for c in 0..self.n_classes {
                    let label = [c as u8];
                    let input = Self::embed_label(image, &label, image_dim, self.n_classes, 1);
                    let mut current = input;
                    let mut total_g = 0.0f32;

                    for layer in &self.layers {
                        let mut act = layer.forward(&current, 1);
                        total_g += MatrixLayer::goodness(&act, 1, layer.out_dim);
                        MatrixLayer::normalize(&mut act, 1, layer.out_dim);
                        current = act;
                    }

                    if total_g > best_g { best_g = total_g; best = c as u8; }
                }
                best
            })
            .collect()
    }

    pub fn accuracy(&self, images: &[f32], labels: &[u8], image_dim: usize, n_samples: usize) -> f32 {
        let preds = self.predict(images, image_dim, n_samples);
        preds.iter().zip(labels).filter(|(p, l)| p == l).count() as f32 / n_samples as f32
    }
}

// ============================================================
// Tests
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mnist::MnistData;

    #[test]
    fn sub_matrix_gives_multiple_levels() {
        // 2×2 ternary matrix should give 9 distinct effective weight levels
        let mut levels = std::collections::HashSet::new();
        for a in [-1i8, 0, 1] {
            for b in [-1i8, 0, 1] {
                for c in [-1i8, 0, 1] {
                    for d in [-1i8, 0, 1] {
                        let m: SubMatrix = [a, b, c, d];
                        let w = phi_mean(&m);
                        levels.insert((w * 1000.0) as i32);
                    }
                }
            }
        }
        println!("2x2 ternary matrix: {} distinct weight levels", levels.len());
        assert!(levels.len() >= 9, "Should have >=9 levels, got {}", levels.len());
    }

    #[test]
    fn effective_weights_have_fine_granularity() {
        let layer = MatrixLayer::new(100, 50, 37);
        let (min, max, unique) = layer.effective_stats();
        println!("Effective weights: min={:.3}, max={:.3}, {} unique values", min, max, unique);
        assert!(unique > 3, "Should have >3 unique effective weights (got {})", unique);
    }

    #[test]
    fn all_matrix_elements_are_ternary() {
        let layer = MatrixLayer::new(100, 50, 37);
        for &v in &layer.matrices {
            assert!(v == -1 || v == 0 || v == 1, "Not ternary: {}", v);
        }
    }

    #[test]
    fn storage_is_compact() {
        let layer = MatrixLayer::new(784, 256, 37);
        let matrix_bytes = layer.size_bytes();
        let f32_bytes = layer.n_weights * 4;
        let ratio = f32_bytes as f32 / matrix_bytes as f32;
        println!("Matrix ternary: {} bytes ({} elements × {} per weight)",
            matrix_bytes, layer.n_weights, MATRIX_SIZE);
        println!("f32 equivalent: {} bytes", f32_bytes);
        println!("Ratio: {:.1}x", ratio);
        // 2×2: each weight = 4 bytes (i8) vs 4 bytes (f32) → same size but 9 levels
        // The advantage is not size but RESOLUTION while staying ternary
    }

    #[test]
    fn matrix_ternary_training() {
        let data = MnistData::synthetic(500, 200);
        let mut net = MatrixNet::new(&[794, 64, 32], 10);

        println!("\n=== Ternary Matrix Training (d={}) ===\n", MATRIX_DIM);

        for epoch in 0..10 {
            let (pg, ng) = net.train_epoch(
                &data.train_images, &data.train_labels,
                784, data.n_train, 50, 100,
            );

            if epoch % 3 == 0 || epoch == 9 {
                let acc = net.accuracy(&data.test_images, &data.test_labels, 784, data.n_test);
                let (pos, zero, neg) = net.layers[0].stats();
                let (_, _, unique) = net.layers[0].effective_stats();
                println!("  Epoch {:>2}: pg={:.3} ng={:.3} acc={:.1}% | +1:{} 0:{} -1:{} | {} eff. levels",
                    epoch + 1, pg, ng, acc * 100.0, pos, zero, neg, unique);
            }
        }

        let final_acc = net.accuracy(&data.test_images, &data.test_labels, 784, data.n_test);
        println!("\n  Final: {:.1}%", final_acc * 100.0);
        println!("  All weights: 100% ternary matrices ({}x{} per weight)", MATRIX_DIM, MATRIX_DIM);

        // Verify ternary
        for layer in &net.layers {
            for &v in &layer.matrices { assert!(v == -1 || v == 0 || v == 1); }
        }
    }
}
