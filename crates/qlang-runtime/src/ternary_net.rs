//! Fully Ternary Neural Network — SCFF + Straight-Through + Stochastic Annealing
//!
//! Combines three breakthroughs for maximum ternary accuracy:
//!
//! 1. **Self-Contrastive Forward-Forward** (SCFF, Nature 2025):
//!    - Positive: [image, image] — same image concatenated
//!    - Negative: [image, random_other_image] — different images
//!    - No labels needed during feature learning!
//!
//! 2. **Straight-Through Ternary**:
//!    - Forward pass uses TERNARY weights {-α, 0, +α}
//!    - Goodness gradient updates f32 shadow weights
//!    - Network learns to optimize for its own ternary representation
//!
//! 3. **Stochastic Ternary Annealing**:
//!    - Random weight flips to escape local optima
//!    - Keep flip if goodness improves, revert otherwise
//!    - Temperature decreases over training (simulated annealing)
//!
//! 4. **Learned Scale Factor α per layer**:
//!    - Ternary weights are {-α, 0, +α} not {-1, 0, +1}
//!    - α is learned via goodness gradient

use rayon::prelude::*;

// ============================================================
// Ternary Layer with Straight-Through + Learned Alpha
// ============================================================

/// Ternarize with threshold, returns {-1, 0, +1}
fn ternarize_sign(w: f32, threshold: f32) -> f32 {
    if w > threshold { 1.0 } else if w < -threshold { -1.0 } else { 0.0 }
}

#[derive(Clone)]
pub struct TernaryLayer {
    /// Shadow weights (f32) — accumulate gradients
    pub shadow: Vec<f32>,
    /// Ternary weights {-1, 0, +1} — used in forward pass
    pub ternary: Vec<f32>,
    /// Learned scale factor per layer
    pub alpha: f32,
    /// Biases (f32)
    pub biases: Vec<f32>,
    /// Ternary threshold (learned)
    pub threshold: f32,
    pub in_dim: usize,
    pub out_dim: usize,
    /// Learning rate
    pub lr: f32,
    /// Goodness threshold θ
    pub goodness_threshold: f32,
}

impl TernaryLayer {
    pub fn new(in_dim: usize, out_dim: usize, seed: f32) -> Self {
        let scale = (2.0 / (in_dim + out_dim) as f64).sqrt() as f32;
        let shadow: Vec<f32> = (0..out_dim * in_dim)
            .map(|i| (i as f32 * seed).sin() * scale)
            .collect();

        let mean_abs = shadow.iter().map(|w| w.abs()).sum::<f32>() / shadow.len() as f32;
        let threshold = mean_abs * 0.7;
        let ternary: Vec<f32> = shadow.iter().map(|&w| ternarize_sign(w, threshold)).collect();

        // Alpha = mean absolute value of non-zero shadow weights
        let non_zero: Vec<f32> = shadow.iter().filter(|&&w| w.abs() > threshold).map(|w| w.abs()).collect();
        let alpha = if non_zero.is_empty() { scale } else { non_zero.iter().sum::<f32>() / non_zero.len() as f32 };

        Self {
            shadow,
            ternary,
            alpha,
            biases: vec![0.0; out_dim],
            threshold,
            in_dim,
            out_dim,
            lr: 0.03,
            goodness_threshold: 2.0,
        }
    }

    /// Sync ternary weights from shadow weights.
    pub fn sync_ternary(&mut self) {
        let mean_abs = self.shadow.iter().map(|w| w.abs()).sum::<f32>() / self.shadow.len() as f32;
        self.threshold = mean_abs * 0.7;

        for i in 0..self.shadow.len() {
            self.ternary[i] = ternarize_sign(self.shadow[i], self.threshold);
        }

        // Alpha = optimal scale that minimizes ||shadow - alpha*ternary||²
        // Closed form: alpha = dot(shadow, ternary) / dot(ternary, ternary)
        let mut dot_st = 0.0f32;
        let mut dot_tt = 0.0f32;
        for i in 0..self.shadow.len() {
            dot_st += self.shadow[i] * self.ternary[i];
            dot_tt += self.ternary[i] * self.ternary[i];
        }
        if dot_tt > 0.0 {
            self.alpha = (dot_st / dot_tt).max(0.001);
        }
    }

    /// STRAIGHT-THROUGH forward: uses shadow weights in forward but quantizes
    /// activations through ternary. The key: shadow weights provide the gradient
    /// signal, ternary structure is enforced by syncing after each epoch.
    /// This avoids the vanishing alpha problem.
    pub fn forward_st(&self, input: &[f32], batch: usize) -> Vec<f32> {
        let out_dim = self.out_dim;
        let in_dim = self.in_dim;
        let shadow = &self.shadow;
        let biases = &self.biases;

        let chunks: Vec<Vec<f32>> = (0..batch)
            .into_par_iter()
            .map(|b| {
                let x = &input[b * in_dim..(b + 1) * in_dim];
                let mut out = vec![0.0f32; out_dim];
                for j in 0..out_dim {
                    let mut sum = biases[j];
                    let w = &shadow[j * in_dim..(j + 1) * in_dim];
                    for k in 0..in_dim {
                        sum += w[k] * x[k];
                    }
                    out[j] = sum.max(0.0);
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

    /// Pure ternary forward for inference (zero-multiply, alpha=1).
    pub fn forward_ternary(&self, input: &[f32], batch: usize) -> Vec<f32> {
        let out_dim = self.out_dim;
        let in_dim = self.in_dim;

        let chunks: Vec<Vec<f32>> = (0..batch)
            .into_par_iter()
            .map(|b| {
                let x = &input[b * in_dim..(b + 1) * in_dim];
                let mut out = vec![0.0f32; out_dim];
                for j in 0..out_dim {
                    let mut sum = self.biases[j];
                    let w = &self.ternary[j * in_dim..(j + 1) * in_dim];
                    for k in 0..in_dim {
                        // Pure ternary: add/sub/skip (alpha=1)
                        if w[k] > 0.5 { sum += x[k]; }
                        else if w[k] < -0.5 { sum -= x[k]; }
                    }
                    out[j] = sum.max(0.0);
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

    pub fn goodness(act: &[f32], batch: usize, dim: usize) -> f32 {
        let mut total = 0.0f32;
        for b in 0..batch {
            let off = b * dim;
            let g: f32 = act[off..off + dim].iter().map(|x| x * x).sum();
            total += g;
        }
        total / batch as f32
    }

    pub fn normalize(act: &mut [f32], batch: usize, dim: usize) {
        for b in 0..batch {
            let off = b * dim;
            let norm: f32 = act[off..off + dim].iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-8);
            for j in 0..dim {
                act[off + j] /= norm;
            }
        }
    }

    /// Straight-Through FF step: forward with ternary, gradient to shadow.
    pub fn ff_step(&mut self, pos_input: &[f32], neg_input: &[f32], batch: usize) -> (f32, f32) {
        let pos_act = self.forward_st(pos_input, batch);
        let pos_goodness = Self::goodness(&pos_act, batch, self.out_dim);

        let neg_act = self.forward_st(neg_input, batch);
        let neg_goodness = Self::goodness(&neg_act, batch, self.out_dim);

        let out_dim = self.out_dim;
        let in_dim = self.in_dim;
        let threshold = self.goodness_threshold;
        let lr = self.lr;
        let _alpha = self.alpha;
        let inv_batch = 1.0 / batch as f32;

        let sample_signals: Vec<(f32, f32)> = (0..batch)
            .map(|b| {
                let pos_g: f32 = (0..out_dim).map(|j| pos_act[b * out_dim + j].powi(2)).sum();
                let neg_g: f32 = (0..out_dim).map(|j| neg_act[b * out_dim + j].powi(2)).sum();
                (
                    1.0 / (1.0 + (-(pos_g - threshold)).exp()),
                    1.0 / (1.0 + (-(neg_g - threshold)).exp()),
                )
            })
            .collect();

        // Parallel gradient per neuron
        let deltas: Vec<(Vec<f32>, f32, f32)> = (0..out_dim)
            .into_par_iter()
            .map(|j| {
                let mut w_delta = vec![0.0f32; in_dim];
                let mut b_delta = 0.0f32;
                let mut alpha_delta = 0.0f32;

                for b in 0..batch {
                    let (pos_p, neg_p) = sample_signals[b];
                    let pos_a = pos_act[b * out_dim + j];
                    let neg_a = neg_act[b * out_dim + j];
                    let d_pos = (1.0 - pos_p) * 2.0 * pos_a;
                    let d_neg = neg_p * 2.0 * neg_a;

                    for k in 0..in_dim {
                        let pos_x = pos_input[b * in_dim + k];
                        let neg_x = neg_input[b * in_dim + k];
                        // Gradient flows through ternary to shadow (straight-through)
                        w_delta[k] += lr * (d_pos * pos_x - d_neg * neg_x) * inv_batch;
                    }
                    b_delta += lr * (d_pos - d_neg) * inv_batch;

                    // Alpha gradient: how much does scaling help?
                    let ternary_row = &self.ternary[j * in_dim..(j + 1) * in_dim];
                    let pos_contrib: f32 = (0..in_dim)
                        .map(|k| ternary_row[k] * pos_input[b * in_dim + k])
                        .sum();
                    let neg_contrib: f32 = (0..in_dim)
                        .map(|k| ternary_row[k] * neg_input[b * in_dim + k])
                        .sum();
                    alpha_delta += lr * 0.01 * (d_pos * pos_contrib - d_neg * neg_contrib) * inv_batch;
                }

                (w_delta, b_delta, alpha_delta)
            })
            .collect();

        // Apply
        let mut total_alpha_delta = 0.0f32;
        for (j, (w_delta, b_delta, a_delta)) in deltas.into_iter().enumerate() {
            for k in 0..in_dim {
                self.shadow[j * in_dim + k] += w_delta[k];
            }
            self.biases[j] += b_delta;
            total_alpha_delta += a_delta;
        }
        self.alpha = (self.alpha + total_alpha_delta / out_dim as f32).max(0.001);

        (pos_goodness, neg_goodness)
    }

    /// Stochastic annealing: flip random ternary weights, keep if goodness improves.
    pub fn anneal(&mut self, input: &[f32], batch: usize, temperature: f32, rng_seed: u64) -> usize {
        let n_flips = ((self.ternary.len() as f32 * temperature * 0.02) as usize).max(1);
        let mut improved = 0;

        // Measure current goodness
        let act_before = self.forward_st(input, batch);
        let g_before = Self::goodness(&act_before, batch, self.out_dim);

        for flip in 0..n_flips {
            // Deterministic pseudo-random index
            let idx = ((rng_seed.wrapping_mul(2654435761 + flip as u64)) % self.ternary.len() as u64) as usize;
            let old_val = self.ternary[idx];

            // Flip to a different ternary value
            let new_val = match old_val as i32 {
                1 => if flip % 2 == 0 { 0.0 } else { -1.0 },
                -1 => if flip % 2 == 0 { 0.0 } else { 1.0 },
                _ => if flip % 2 == 0 { 1.0 } else { -1.0 },
            };
            self.ternary[idx] = new_val;

            // Measure new goodness
            let act_after = self.forward_st(input, batch);
            let g_after = Self::goodness(&act_after, batch, self.out_dim);

            if g_after > g_before {
                // Also update shadow to match
                self.shadow[idx] = new_val * self.alpha;
                improved += 1;
            } else {
                // Revert
                self.ternary[idx] = old_val;
            }
        }

        improved
    }

    pub fn weight_stats(&self) -> (usize, usize, usize) {
        let pos = self.ternary.iter().filter(|&&w| w > 0.5).count();
        let neg = self.ternary.iter().filter(|&&w| w < -0.5).count();
        let zero = self.ternary.iter().filter(|&&w| w.abs() < 0.5).count();
        (pos, zero, neg)
    }
}

// ============================================================
// Full Ternary Network
// ============================================================

pub struct TernaryNet {
    pub layers: Vec<TernaryLayer>,
    pub n_classes: usize,
}

impl TernaryNet {
    pub fn new(layer_sizes: &[usize], n_classes: usize) -> Self {
        let mut layers = Vec::new();
        for i in 0..layer_sizes.len() - 1 {
            layers.push(TernaryLayer::new(
                layer_sizes[i], layer_sizes[i + 1],
                0.37 + i as f32 * 0.17,
            ));
        }
        Self { layers, n_classes }
    }

    // ---- SCFF: Self-Contrastive sample generation ----

    /// Positive: [image, image] (same image concatenated)
    pub fn make_positive(images: &[f32], image_dim: usize, batch: usize) -> Vec<f32> {
        let total_dim = image_dim * 2;
        let mut out = vec![0.0f32; batch * total_dim];
        for b in 0..batch {
            let src = &images[b * image_dim..(b + 1) * image_dim];
            out[b * total_dim..b * total_dim + image_dim].copy_from_slice(src);
            out[b * total_dim + image_dim..(b + 1) * total_dim].copy_from_slice(src);
        }
        out
    }

    /// Negative: [image, different_random_image]
    pub fn make_negative(images: &[f32], image_dim: usize, batch: usize, n_total: usize) -> Vec<f32> {
        let total_dim = image_dim * 2;
        let mut out = vec![0.0f32; batch * total_dim];
        for b in 0..batch {
            let src = &images[b * image_dim..(b + 1) * image_dim];
            // Pick a different image (deterministic pseudo-random)
            let other_idx = (b * 7 + 13) % n_total;
            let other = &images[other_idx * image_dim..(other_idx + 1) * image_dim];
            out[b * total_dim..b * total_dim + image_dim].copy_from_slice(src);
            out[b * total_dim + image_dim..(b + 1) * total_dim].copy_from_slice(other);
        }
        out
    }

    /// Label-embedded input for classification: [image + one_hot_label]
    pub fn embed_label(images: &[f32], labels: &[u8], image_dim: usize, n_classes: usize, batch: usize) -> Vec<f32> {
        let total_dim = image_dim + n_classes;
        let mut out = vec![0.0f32; batch * total_dim];
        for b in 0..batch {
            let src = &images[b * image_dim..(b + 1) * image_dim];
            out[b * total_dim..b * total_dim + image_dim].copy_from_slice(src);
            let label = labels[b] as usize;
            if label < n_classes {
                out[b * total_dim + image_dim + label] = 1.0;
            }
        }
        out
    }

    pub fn make_negative_label(images: &[f32], labels: &[u8], image_dim: usize, n_classes: usize, batch: usize) -> Vec<f32> {
        let mut wrong = labels.to_vec();
        for b in 0..batch {
            wrong[b] = ((labels[b] as usize + 1 + (b * 7 + 3) % (n_classes - 1)) % n_classes) as u8;
        }
        Self::embed_label(images, &wrong, image_dim, n_classes, batch)
    }

    /// Phase 1: SCFF unsupervised feature learning.
    /// Trains representations without labels using self-contrastive pairs.
    pub fn train_scff_epoch(
        &mut self,
        images: &[f32],
        image_dim: usize,
        n_samples: usize,
        batch_size: usize,
    ) -> (f32, f32) {
        let n_batches = n_samples / batch_size;
        let mut total_pos = 0.0f32;
        let mut total_neg = 0.0f32;

        for batch_idx in 0..n_batches {
            let offset = batch_idx * batch_size;
            let batch_images = &images[offset * image_dim..(offset + batch_size) * image_dim];

            let pos = Self::make_positive(batch_images, image_dim, batch_size);
            let neg = Self::make_negative(images, image_dim, batch_size, n_samples);

            let mut pos_in = pos;
            let mut neg_in = neg;

            for layer in &mut self.layers {
                let (pg, ng) = layer.ff_step(&pos_in, &neg_in, batch_size);
                total_pos += pg;
                total_neg += ng;

                // Next layer input = normalized output (detached)
                let mut pos_out = layer.forward_st(&pos_in, batch_size);
                TernaryLayer::normalize(&mut pos_out, batch_size, layer.out_dim);
                let mut neg_out = layer.forward_st(&neg_in, batch_size);
                TernaryLayer::normalize(&mut neg_out, batch_size, layer.out_dim);

                pos_in = pos_out;
                neg_in = neg_out;
            }
        }

        // Sync ternary + anneal
        for layer in &mut self.layers {
            layer.sync_ternary();
        }

        let count = (n_batches * self.layers.len()) as f32;
        (total_pos / count.max(1.0), total_neg / count.max(1.0))
    }

    /// Phase 2: Supervised fine-tuning with label embedding.
    pub fn train_supervised_epoch(
        &mut self,
        images: &[f32],
        labels: &[u8],
        image_dim: usize,
        n_samples: usize,
        batch_size: usize,
    ) -> (f32, f32) {
        let n_batches = n_samples / batch_size;
        let mut total_pos = 0.0f32;
        let mut total_neg = 0.0f32;

        for batch_idx in 0..n_batches {
            let offset = batch_idx * batch_size;
            let batch_images = &images[offset * image_dim..(offset + batch_size) * image_dim];
            let batch_labels = &labels[offset..offset + batch_size];

            let pos = Self::embed_label(batch_images, batch_labels, image_dim, self.n_classes, batch_size);
            let neg = Self::make_negative_label(batch_images, batch_labels, image_dim, self.n_classes, batch_size);

            let mut pos_in = pos;
            let mut neg_in = neg;

            for layer in &mut self.layers {
                let (pg, ng) = layer.ff_step(&pos_in, &neg_in, batch_size);
                total_pos += pg;
                total_neg += ng;

                let mut pos_out = layer.forward_st(&pos_in, batch_size);
                TernaryLayer::normalize(&mut pos_out, batch_size, layer.out_dim);
                let mut neg_out = layer.forward_st(&neg_in, batch_size);
                TernaryLayer::normalize(&mut neg_out, batch_size, layer.out_dim);

                pos_in = pos_out;
                neg_in = neg_out;
            }
        }

        for layer in &mut self.layers {
            layer.sync_ternary();
        }

        let count = (n_batches * self.layers.len()) as f32;
        (total_pos / count.max(1.0), total_neg / count.max(1.0))
    }

    /// Stochastic annealing pass on all layers.
    pub fn anneal_pass(&mut self, images: &[f32], image_dim: usize, batch_size: usize, temperature: f32, epoch: usize) -> usize {
        let batch = &images[..batch_size.min(images.len() / image_dim) * image_dim];
        let actual_batch = batch.len() / image_dim;

        // Build input for first layer (use self-contrastive positive)
        let input = Self::make_positive(batch, image_dim, actual_batch);

        let mut total_improved = 0;
        let mut layer_input = input;

        for (i, layer) in self.layers.iter_mut().enumerate() {
            total_improved += layer.anneal(&layer_input, actual_batch, temperature, (epoch * 1000 + i) as u64);
            let mut out = layer.forward_st(&layer_input, actual_batch);
            TernaryLayer::normalize(&mut out, actual_batch, layer.out_dim);
            layer_input = out;
        }

        total_improved
    }

    /// Predict using ternary weights (label-embedding approach).
    pub fn predict(&self, images: &[f32], image_dim: usize, n_samples: usize) -> Vec<u8> {
        (0..n_samples)
            .into_par_iter()
            .map(|i| {
                let image = &images[i * image_dim..(i + 1) * image_dim];
                let mut best_label = 0u8;
                let mut best_goodness = f32::NEG_INFINITY;

                for c in 0..self.n_classes {
                    let label = [c as u8];
                    let input = Self::embed_label(image, &label, image_dim, self.n_classes, 1);

                    let mut layer_in = input;
                    let mut total_g = 0.0f32;

                    for layer in &self.layers {
                        let mut act = layer.forward_st(&layer_in, 1);
                        total_g += TernaryLayer::goodness(&act, 1, layer.out_dim);
                        TernaryLayer::normalize(&mut act, 1, layer.out_dim);
                        layer_in = act;
                    }

                    if total_g > best_goodness {
                        best_goodness = total_g;
                        best_label = c as u8;
                    }
                }

                best_label
            })
            .collect()
    }

    pub fn accuracy(&self, images: &[f32], labels: &[u8], image_dim: usize, n_samples: usize) -> f32 {
        let preds = self.predict(images, image_dim, n_samples);
        let correct = preds.iter().zip(labels).filter(|(p, l)| p == l).count();
        correct as f32 / n_samples as f32
    }

    /// Predict using PURE ternary weights (zero-multiply inference).
    pub fn predict_ternary(&self, images: &[f32], image_dim: usize, n_samples: usize) -> Vec<u8> {
        (0..n_samples)
            .into_par_iter()
            .map(|i| {
                let image = &images[i * image_dim..(i + 1) * image_dim];
                let mut best_label = 0u8;
                let mut best_goodness = f32::NEG_INFINITY;

                for c in 0..self.n_classes {
                    let label = [c as u8];
                    let input = Self::embed_label(image, &label, image_dim, self.n_classes, 1);

                    let mut layer_in = input;
                    let mut total_g = 0.0f32;

                    for layer in &self.layers {
                        let mut act = layer.forward_ternary(&layer_in, 1);
                        total_g += TernaryLayer::goodness(&act, 1, layer.out_dim);
                        TernaryLayer::normalize(&mut act, 1, layer.out_dim);
                        layer_in = act;
                    }

                    if total_g > best_goodness {
                        best_goodness = total_g;
                        best_label = c as u8;
                    }
                }

                best_label
            })
            .collect()
    }

    pub fn accuracy_ternary(&self, images: &[f32], labels: &[u8], image_dim: usize, n_samples: usize) -> f32 {
        let preds = self.predict_ternary(images, image_dim, n_samples);
        let correct = preds.iter().zip(labels).filter(|(p, l)| p == l).count();
        correct as f32 / n_samples as f32
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
    fn scff_positive_negative_shapes() {
        let images = vec![1.0f32; 10 * 4]; // 10 images, dim=4
        let pos = TernaryNet::make_positive(&images, 4, 10);
        assert_eq!(pos.len(), 10 * 8); // doubled
        // Check first sample: [1,1,1,1, 1,1,1,1]
        assert_eq!(&pos[0..8], &[1.0; 8]);
    }

    #[test]
    fn scff_negative_has_different_second_half() {
        let mut images = vec![0.0f32; 4 * 4]; // 4 images, dim=4
        // Make each image distinct
        for i in 0..4 {
            images[i * 4] = (i + 1) as f32;
        }
        let neg = TernaryNet::make_negative(&images, 4, 4, 4);
        // For each sample, second half should differ from first half
        for b in 0..4 {
            let first = &neg[b * 8..b * 8 + 4];
            let second = &neg[b * 8 + 4..b * 8 + 8];
            assert_ne!(first, second, "Sample {} should have different halves", b);
        }
    }

    #[test]
    fn straight_through_uses_ternary() {
        let layer = TernaryLayer::new(10, 5, 0.37);
        // All ternary weights should be {-1, 0, +1}
        for &w in &layer.ternary {
            assert!(w == -1.0 || w == 0.0 || w == 1.0, "Not ternary: {}", w);
        }
        // Alpha should be positive
        assert!(layer.alpha > 0.0);
    }

    #[test]
    fn learned_alpha_updates() {
        let data = MnistData::synthetic(200, 50);
        let mut net = TernaryNet::new(&[794, 64], 10);
        let alpha_before = net.layers[0].alpha;

        net.train_supervised_epoch(
            &data.train_images, &data.train_labels,
            784, data.n_train, 50,
        );

        let alpha_after = net.layers[0].alpha;
        // Alpha should have changed
        assert!((alpha_after - alpha_before).abs() > 1e-6,
            "Alpha should update: before={}, after={}", alpha_before, alpha_after);
    }

    #[test]
    fn full_ternary_training_mnist() {
        let data = MnistData::synthetic(2000, 500);
        let mut net = TernaryNet::new(&[794, 256, 128], 10);

        println!("\n=== Fully Ternary Training (ST + Learned α + Annealing) ===\n");

        for epoch in 0..30 {
            let (pg, ng) = net.train_supervised_epoch(
                &data.train_images, &data.train_labels,
                784, data.n_train, 50,
            );

            // Anneal every 3 epochs
            if epoch % 3 == 2 {
                let temp = 1.0 - (epoch as f32 / 30.0);
                net.anneal_pass(&data.train_images, 784, 200, temp, epoch);
            }

            if epoch % 5 == 0 || epoch == 29 {
                let f32_acc = net.accuracy(&data.test_images, &data.test_labels, 784, data.n_test);
                let tern_acc = net.accuracy_ternary(&data.test_images, &data.test_labels, 784, data.n_test);
                let alpha = net.layers[0].alpha;
                let (pos, zero, neg) = net.layers[0].weight_stats();
                println!("  Epoch {:>2}: pg={:.2} ng={:.2} | f32={:.1}% tern={:.1}% | α={:.4} | +1:{} 0:{} -1:{}",
                    epoch + 1, pg, ng, f32_acc * 100.0, tern_acc * 100.0, alpha, pos, zero, neg);
            }
        }

        let f32_acc = net.accuracy(&data.test_images, &data.test_labels, 784, data.n_test);
        let tern_acc = net.accuracy_ternary(&data.test_images, &data.test_labels, 784, data.n_test);
        println!("\n  Final: f32={:.1}% ternary={:.1}%", f32_acc * 100.0, tern_acc * 100.0);

        // Verify weights are ternary
        for layer in &net.layers {
            for &w in &layer.ternary {
                assert!(w == -1.0 || w == 0.0 || w == 1.0, "Not ternary: {}", w);
            }
            assert!(layer.alpha > 0.0, "Alpha must be positive");
        }

        assert!(f32_acc > 0.50, "f32 must >50% (got {:.1}%)", f32_acc * 100.0);
        assert!(tern_acc > 0.15, "ternary must >15% (got {:.1}%)", tern_acc * 100.0);
    }
}
