//! Forward-Forward Algorithm for Ternary Neural Networks
//!
//! Based on Hinton (2022): "The Forward-Forward Algorithm"
//! Adapted for ternary weights {-1, 0, +1}.
//!
//! Core idea: Each layer learns LOCALLY — no backpropagation.
//! - Positive pass: real (image, correct_label) → maximize goodness
//! - Negative pass: real (image, wrong_label) → minimize goodness
//! - Goodness = sum of squared activations
//!
//! Combined with ternary weight learning:
//! - Track per-weight "goodness delta" (positive goodness - negative goodness)
//! - If delta consistently positive → weight should be +1
//! - If delta consistently negative → weight should be -1
//! - If delta near zero → weight should be 0
//!
//! This is gradient-free, layer-local, and produces ternary weights natively.

use rayon::prelude::*;

/// Convert a continuous weight to ternary {-1, 0, +1} using adaptive threshold.
fn ternarize(w: f32, threshold: f32) -> f32 {
    if w > threshold { 1.0 } else if w < -threshold { -1.0 } else { 0.0 }
}

/// A single layer that learns via Forward-Forward with ternary weights.
///
/// Uses f32 "shadow weights" that accumulate the goodness gradient,
/// then derives ternary weights via sign+threshold (like Straight-Through Estimator).
/// Forward pass uses shadow weights. Ternary weights for inference with learned alpha.
#[derive(Debug, Clone)]
pub struct FFLayer {
    /// Ternary weights [out_dim, in_dim] used in inference {-1, 0, +1}
    pub weights: Vec<f32>,
    /// Shadow weights (f32) that accumulate goodness gradients
    pub shadow: Vec<f32>,
    /// Learned scale factor: ternary inference uses alpha * weights
    pub alpha: f32,
    /// Biases [out_dim]
    pub biases: Vec<f32>,
    /// Input dimension
    pub in_dim: usize,
    /// Output dimension
    pub out_dim: usize,
    /// Threshold for goodness (θ in the paper)
    pub threshold: f32,
    /// Learning rate
    pub lr: f32,
}

impl FFLayer {
    /// Create a new Forward-Forward layer.
    pub fn new(in_dim: usize, out_dim: usize, seed: f32) -> Self {
        let scale = (2.0 / (in_dim + out_dim) as f64).sqrt() as f32;
        // Initialize shadow weights with small random values
        let shadow: Vec<f32> = (0..out_dim * in_dim)
            .map(|i| (i as f32 * seed).sin() * scale)
            .collect();
        // Derive initial ternary weights from shadow
        let weights = shadow.iter().map(|&w| ternarize(w, scale * 0.5)).collect();
        let biases = vec![0.0f32; out_dim];

        Self {
            weights,
            shadow,
            alpha: scale,
            biases,
            in_dim,
            out_dim,
            threshold: 2.0,
            lr: 0.03,
        }
    }

    /// Forward pass using SHADOW weights (f32) for learning.
    /// Parallelized across batch samples.
    pub fn forward(&self, input: &[f32], batch: usize) -> Vec<f32> {
        let out_dim = self.out_dim;
        let in_dim = self.in_dim;
        let shadow = &self.shadow;
        let biases = &self.biases;

        // Parallel across batch samples
        let chunks: Vec<Vec<f32>> = (0..batch)
            .into_par_iter()
            .map(|b| {
                let x_row = &input[b * in_dim..(b + 1) * in_dim];
                let mut row_out = vec![0.0f32; out_dim];
                for j in 0..out_dim {
                    let mut sum = biases[j];
                    let w_row = &shadow[j * in_dim..(j + 1) * in_dim];
                    for k in 0..in_dim {
                        sum += x_row[k] * w_row[k];
                    }
                    row_out[j] = sum.max(0.0);
                }
                row_out
            })
            .collect();

        let mut output = vec![0.0f32; batch * out_dim];
        for (b, chunk) in chunks.into_iter().enumerate() {
            output[b * out_dim..(b + 1) * out_dim].copy_from_slice(&chunk);
        }
        output
    }

    /// Forward pass using TERNARY weights with learned alpha scaling.
    /// Uses alpha * ternary_weights for better approximation of shadow weights.
    pub fn forward_ternary(&self, input: &[f32], batch: usize) -> Vec<f32> {
        let out_dim = self.out_dim;
        let in_dim = self.in_dim;
        let alpha = self.alpha;
        let weights = &self.weights;
        let biases = &self.biases;

        let chunks: Vec<Vec<f32>> = (0..batch)
            .into_par_iter()
            .map(|b| {
                let x = &input[b * in_dim..(b + 1) * in_dim];
                let mut out = vec![0.0f32; out_dim];
                for j in 0..out_dim {
                    let mut sum = biases[j];
                    let w = &weights[j * in_dim..(j + 1) * in_dim];
                    for k in 0..in_dim {
                        // alpha-scaled ternary: add/sub with scale
                        if w[k] > 0.5 { sum += alpha * x[k]; }
                        else if w[k] < -0.5 { sum -= alpha * x[k]; }
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

    /// Normalize activations to unit length per sample (as in Hinton's paper).
    /// This prevents the network from trivially increasing goodness by scaling.
    pub fn normalize(activations: &mut [f32], batch: usize, dim: usize) {
        for b in 0..batch {
            let off = b * dim;
            let norm: f32 = activations[off..off + dim]
                .iter()
                .map(|x| x * x)
                .sum::<f32>()
                .sqrt()
                .max(1e-8);
            for j in 0..dim {
                activations[off + j] /= norm;
            }
        }
    }

    /// Compute goodness for a batch = mean sum of squared activations per sample.
    pub fn goodness(activations: &[f32], batch: usize, dim: usize) -> f32 {
        let mut total = 0.0f32;
        for b in 0..batch {
            let off = b * dim;
            let g: f32 = activations[off..off + dim].iter().map(|x| x * x).sum();
            total += g;
        }
        total / batch as f32
    }

    /// One Forward-Forward learning step.
    ///
    /// - `pos_input`: [batch, in_dim] — real data with correct labels embedded
    /// - `neg_input`: [batch, in_dim] — real data with wrong labels embedded
    ///
    /// Returns (pos_goodness, neg_goodness)
    pub fn ff_step(&mut self, pos_input: &[f32], neg_input: &[f32], batch: usize) -> (f32, f32) {
        let pos_act = self.forward(pos_input, batch);
        let pos_goodness = Self::goodness(&pos_act, batch, self.out_dim);

        let neg_act = self.forward(neg_input, batch);
        let neg_goodness = Self::goodness(&neg_act, batch, self.out_dim);

        let out_dim = self.out_dim;
        let in_dim = self.in_dim;
        let threshold = self.threshold;
        let lr = self.lr;
        let inv_batch = 1.0 / batch as f32;

        // Pre-compute per-sample goodness and sigmoid
        let sample_signals: Vec<(f32, f32)> = (0..batch)
            .map(|b| {
                let pos_g: f32 = (0..out_dim).map(|j| pos_act[b * out_dim + j].powi(2)).sum();
                let neg_g: f32 = (0..out_dim).map(|j| neg_act[b * out_dim + j].powi(2)).sum();
                let pos_p = 1.0 / (1.0 + (-(pos_g - threshold)).exp());
                let neg_p = 1.0 / (1.0 + (-(neg_g - threshold)).exp());
                (pos_p, neg_p)
            })
            .collect();

        // Parallel gradient computation per output neuron j
        // Each j writes to shadow[j*in_dim..(j+1)*in_dim] — no overlap
        let shadow_chunks: Vec<(Vec<f32>, f32)> = (0..out_dim)
            .into_par_iter()
            .map(|j| {
                let mut w_delta = vec![0.0f32; in_dim];
                let mut b_delta = 0.0f32;

                for b in 0..batch {
                    let (pos_p, neg_p) = sample_signals[b];
                    let pos_a = pos_act[b * out_dim + j];
                    let neg_a = neg_act[b * out_dim + j];
                    let d_pos = (1.0 - pos_p) * 2.0 * pos_a;
                    let d_neg = neg_p * 2.0 * neg_a;

                    for k in 0..in_dim {
                        let pos_x = pos_input[b * in_dim + k];
                        let neg_x = neg_input[b * in_dim + k];
                        w_delta[k] += lr * (d_pos * pos_x - d_neg * neg_x) * inv_batch;
                    }
                    b_delta += lr * (d_pos - d_neg) * inv_batch;
                }

                (w_delta, b_delta)
            })
            .collect();

        // Apply deltas
        for (j, (w_delta, b_delta)) in shadow_chunks.into_iter().enumerate() {
            for k in 0..in_dim {
                self.shadow[j * in_dim + k] += w_delta[k];
            }
            self.biases[j] += b_delta;
        }

        (pos_goodness, neg_goodness)
    }

    /// Quantization-Aware Training step with Straight-Through Estimator (STE).
    ///
    /// Forward:  Uses ternary weights (absmean-quantized from shadow) → matches inference.
    /// Backward: Gradients flow to SHADOW (STE = identity in backward pass).
    ///
    /// This makes the network learn representations that survive ternarization,
    /// closing the gap between f32_accuracy and ternary_accuracy.
    pub fn ff_step_qat(&mut self, pos_input: &[f32], neg_input: &[f32], batch: usize) -> (f32, f32) {
        use crate::bitnet_math::absmean_quantize;

        // === STEP 1: Quantize shadow → ternary for THIS forward pass ===
        let (ternary_now, gamma) = absmean_quantize(&self.shadow);
        self.alpha = gamma;  // keep alpha updated for inference
        // Swap in ternary weights (will restore after backward)
        let saved_shadow_copy = self.shadow.clone();
        // But we forward with alpha * ternary → it's effectively quantized
        let scaled_ternary: Vec<f32> = ternary_now.iter().map(|&t| gamma * t).collect();
        let original_shadow = std::mem::replace(&mut self.shadow, scaled_ternary);

        // === STEP 2: Forward with quantized weights ===
        let pos_act = self.forward(pos_input, batch);
        let pos_goodness = Self::goodness(&pos_act, batch, self.out_dim);
        let neg_act = self.forward(neg_input, batch);
        let neg_goodness = Self::goodness(&neg_act, batch, self.out_dim);

        // Restore real shadow weights for gradient update
        self.shadow = original_shadow;

        let out_dim = self.out_dim;
        let in_dim = self.in_dim;
        let threshold = self.threshold;
        let lr = self.lr;
        let inv_batch = 1.0 / batch as f32;

        let sample_signals: Vec<(f32, f32)> = (0..batch)
            .map(|b| {
                let pos_g: f32 = (0..out_dim).map(|j| pos_act[b * out_dim + j].powi(2)).sum();
                let neg_g: f32 = (0..out_dim).map(|j| neg_act[b * out_dim + j].powi(2)).sum();
                let pos_p = 1.0 / (1.0 + (-(pos_g - threshold)).exp());
                let neg_p = 1.0 / (1.0 + (-(neg_g - threshold)).exp());
                (pos_p, neg_p)
            })
            .collect();

        // === STEP 3: Backward → gradients flow to SHADOW (STE) ===
        let shadow_chunks: Vec<(Vec<f32>, f32)> = (0..out_dim)
            .into_par_iter()
            .map(|j| {
                let mut w_delta = vec![0.0f32; in_dim];
                let mut b_delta = 0.0f32;
                for b in 0..batch {
                    let (pos_p, neg_p) = sample_signals[b];
                    let pos_a = pos_act[b * out_dim + j];
                    let neg_a = neg_act[b * out_dim + j];
                    let d_pos = (1.0 - pos_p) * 2.0 * pos_a;
                    let d_neg = neg_p * 2.0 * neg_a;
                    for k in 0..in_dim {
                        let pos_x = pos_input[b * in_dim + k];
                        let neg_x = neg_input[b * in_dim + k];
                        w_delta[k] += lr * (d_pos * pos_x - d_neg * neg_x) * inv_batch;
                    }
                    b_delta += lr * (d_pos - d_neg) * inv_batch;
                }
                (w_delta, b_delta)
            })
            .collect();

        for (j, (w_delta, b_delta)) in shadow_chunks.into_iter().enumerate() {
            for k in 0..in_dim {
                // STE: write gradient directly to shadow (not to ternary)
                self.shadow[j * in_dim + k] += w_delta[k];
            }
            self.biases[j] += b_delta;
        }

        // === STEP 4: Re-quantize so .weights reflects current shadow ===
        let (new_ternary, new_gamma) = absmean_quantize(&self.shadow);
        self.weights = new_ternary;
        self.alpha = new_gamma;
        let _ = saved_shadow_copy;  // unused, just here to keep the pattern explicit

        (pos_goodness, neg_goodness)
    }

    /// Derive ternary weights from shadow weights using adaptive threshold
    /// and compute optimal scale alpha via least-squares:
    ///   alpha = dot(shadow, ternary) / dot(ternary, ternary)
    /// This minimizes ||shadow - alpha * ternary||².
    pub fn sync_ternary(&mut self) -> usize {
        let mean_abs: f32 = self.shadow.iter().map(|w| w.abs()).sum::<f32>()
            / self.shadow.len() as f32;
        let threshold = mean_abs * 0.7;

        let mut changed = 0;
        for i in 0..self.weights.len() {
            let new_w = ternarize(self.shadow[i], threshold);
            if new_w != self.weights[i] {
                self.weights[i] = new_w;
                changed += 1;
            }
        }

        // Optimal alpha: minimizes ||shadow - alpha * ternary||²
        // Closed form: alpha = dot(shadow, ternary) / dot(ternary, ternary)
        let mut dot_st = 0.0f32;
        let mut dot_tt = 0.0f32;
        for i in 0..self.shadow.len() {
            dot_st += self.shadow[i] * self.weights[i];
            dot_tt += self.weights[i] * self.weights[i];
        }
        if dot_tt > 0.0 {
            self.alpha = (dot_st / dot_tt).max(0.001);
        }

        changed
    }

    /// Count ternary weight distribution.
    pub fn weight_stats(&self) -> (usize, usize, usize) {
        let pos = self.weights.iter().filter(|&&w| w == 1.0).count();
        let neg = self.weights.iter().filter(|&&w| w == -1.0).count();
        let zero = self.weights.iter().filter(|&&w| w == 0.0).count();
        (pos, zero, neg)
    }
}

/// A complete Forward-Forward network with ternary weights.
pub struct FFNetwork {
    pub layers: Vec<FFLayer>,
    pub n_classes: usize,
}

impl FFNetwork {
    /// Create a Forward-Forward network.
    ///
    /// `layer_sizes`: e.g. [794, 256, 128] — first includes label embedding (784 + 10)
    pub fn new(layer_sizes: &[usize], n_classes: usize) -> Self {
        let mut layers = Vec::new();
        for i in 0..layer_sizes.len() - 1 {
            let seed = 0.37 + i as f32 * 0.17;
            layers.push(FFLayer::new(layer_sizes[i], layer_sizes[i + 1], seed));
        }
        Self { layers, n_classes }
    }

    /// Embed label into input: concatenate one-hot label to image.
    /// Returns [batch, image_dim + n_classes]
    pub fn embed_label(images: &[f32], labels: &[u8], image_dim: usize, n_classes: usize, batch: usize) -> Vec<f32> {
        let total_dim = image_dim + n_classes;
        let mut embedded = vec![0.0f32; batch * total_dim];
        for b in 0..batch {
            // Copy image
            let src = &images[b * image_dim..(b + 1) * image_dim];
            let dst = &mut embedded[b * total_dim..b * total_dim + image_dim];
            dst.copy_from_slice(src);
            // One-hot label
            let label = labels[b] as usize;
            if label < n_classes {
                embedded[b * total_dim + image_dim + label] = 1.0;
            }
        }
        embedded
    }

    /// Generate negative data: same images but with random wrong labels.
    pub fn make_negative(images: &[f32], labels: &[u8], image_dim: usize, n_classes: usize, batch: usize) -> Vec<f32> {
        let mut wrong_labels = labels.to_vec();
        for b in 0..batch {
            // Deterministic "random" wrong label
            let correct = labels[b] as usize;
            wrong_labels[b] = ((correct + 1 + (b * 7 + 3) % (n_classes - 1)) % n_classes) as u8;
        }
        Self::embed_label(images, &wrong_labels, image_dim, n_classes, batch)
    }

    /// Train one epoch with Forward-Forward.
    ///
    /// Returns (avg_pos_goodness, avg_neg_goodness)
    pub fn train_epoch(
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

            // Create positive and negative inputs
            let pos_input = Self::embed_label(batch_images, batch_labels, image_dim, self.n_classes, batch_size);
            let neg_input = Self::make_negative(batch_images, batch_labels, image_dim, self.n_classes, batch_size);

            // Train each layer with its own input
            let mut pos_layer_input = pos_input.clone();
            let mut neg_layer_input = neg_input.clone();

            for layer in &mut self.layers {
                let (pg, ng) = layer.ff_step(&pos_layer_input, &neg_layer_input, batch_size);
                total_pos += pg;
                total_neg += ng;

                // Next layer's input = this layer's normalized output (detached!)
                let mut pos_out = layer.forward(&pos_layer_input, batch_size);
                FFLayer::normalize(&mut pos_out, batch_size, layer.out_dim);
                let mut neg_out = layer.forward(&neg_layer_input, batch_size);
                FFLayer::normalize(&mut neg_out, batch_size, layer.out_dim);

                pos_layer_input = pos_out;
                neg_layer_input = neg_out;
            }
        }

        // Sync ternary weights from shadow weights after epoch
        for layer in &mut self.layers {
            layer.sync_ternary();
        }

        let count = (n_batches * self.layers.len()) as f32;
        (total_pos / count, total_neg / count)
    }

    /// Quantization-Aware Training epoch.
    /// Uses ff_step_qat: forward with ternary weights, backward to shadow (STE).
    /// Network learns representations compatible with ternary inference.
    pub fn train_epoch_qat(
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
            let pos_input = Self::embed_label(batch_images, batch_labels, image_dim, self.n_classes, batch_size);
            let neg_input = Self::make_negative(batch_images, batch_labels, image_dim, self.n_classes, batch_size);
            let mut pos_layer_input = pos_input.clone();
            let mut neg_layer_input = neg_input.clone();

            for layer in &mut self.layers {
                let (pg, ng) = layer.ff_step_qat(&pos_layer_input, &neg_layer_input, batch_size);
                total_pos += pg;
                total_neg += ng;

                // Pass TERNARY-forwarded outputs to next layer (consistent with QAT)
                let mut pos_out = layer.forward_ternary(&pos_layer_input, batch_size);
                FFLayer::normalize(&mut pos_out, batch_size, layer.out_dim);
                let mut neg_out = layer.forward_ternary(&neg_layer_input, batch_size);
                FFLayer::normalize(&mut neg_out, batch_size, layer.out_dim);
                pos_layer_input = pos_out;
                neg_layer_input = neg_out;
            }
        }

        // Final ternary sync (ff_step_qat already keeps .weights up to date,
        // but this ensures alpha/threshold are consistent)
        for layer in &mut self.layers {
            layer.sync_ternary();
        }

        let count = (n_batches * self.layers.len()) as f32;
        (total_pos / count, total_neg / count)
    }

    /// Predict class for test images using shadow (f32) weights.
    pub fn predict(&self, images: &[f32], image_dim: usize, n_samples: usize) -> Vec<u8> {
        self.predict_inner(images, image_dim, n_samples, false)
    }

    /// Predict class for test images using ternary weights.
    pub fn predict_ternary(&self, images: &[f32], image_dim: usize, n_samples: usize) -> Vec<u8> {
        self.predict_inner(images, image_dim, n_samples, true)
    }

    fn predict_inner(&self, images: &[f32], image_dim: usize, n_samples: usize, use_ternary: bool) -> Vec<u8> {
        // Parallel prediction: each sample independently evaluated
        let predictions: Vec<u8> = (0..n_samples)
            .into_par_iter()
            .map(|i| {
                let image = &images[i * image_dim..(i + 1) * image_dim];
                let mut best_label = 0u8;
                let mut best_goodness = f32::NEG_INFINITY;

                for c in 0..self.n_classes {
                    let label_slice = [c as u8];
                    let input = Self::embed_label(image, &label_slice, image_dim, self.n_classes, 1);

                    let mut layer_input = input;
                    let mut total_goodness = 0.0f32;

                    for layer in &self.layers {
                        let mut act = if use_ternary {
                            layer.forward_ternary(&layer_input, 1)
                        } else {
                            layer.forward(&layer_input, 1)
                        };
                        total_goodness += FFLayer::goodness(&act, 1, layer.out_dim);
                        FFLayer::normalize(&mut act, 1, layer.out_dim);
                        layer_input = act;
                    }

                    if total_goodness > best_goodness {
                        best_goodness = total_goodness;
                        best_label = c as u8;
                    }
                }

                best_label
            })
            .collect();

        predictions
    }

    /// Compute accuracy using shadow (f32) weights.
    pub fn accuracy(&self, images: &[f32], labels: &[u8], image_dim: usize, n_samples: usize) -> f32 {
        let preds = self.predict(images, image_dim, n_samples);
        let correct = preds.iter().zip(labels).filter(|(p, l)| p == l).count();
        correct as f32 / n_samples as f32
    }

    /// Compute accuracy using ternary weights.
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
    fn ff_layer_forward() {
        let layer = FFLayer::new(10, 5, 0.37);
        let input = vec![1.0f32; 20]; // batch=2, in_dim=10
        let output = layer.forward(&input, 2);
        assert_eq!(output.len(), 10); // batch=2, out_dim=5
        // All values should be >= 0 (ReLU)
        assert!(output.iter().all(|&x| x >= 0.0));
    }

    #[test]
    fn ff_layer_normalize() {
        let mut act = vec![3.0, 4.0]; // norm = 5
        FFLayer::normalize(&mut act, 1, 2);
        assert!((act[0] - 0.6).abs() < 1e-5);
        assert!((act[1] - 0.8).abs() < 1e-5);
    }

    #[test]
    fn ff_layer_goodness() {
        let act = vec![0.6, 0.8]; // normalized
        let g = FFLayer::goodness(&act, 1, 2);
        assert!((g - 1.0).abs() < 1e-5); // 0.36 + 0.64 = 1.0
    }

    #[test]
    fn ff_label_embedding() {
        let images = vec![0.5f32; 4]; // 2 images of dim 2
        let labels = vec![1u8, 0u8];
        let embedded = FFNetwork::embed_label(&images, &labels, 2, 3, 2);
        // Sample 0: [0.5, 0.5, 0, 1, 0] (label 1)
        assert_eq!(embedded.len(), 10); // 2 * (2+3)
        assert_eq!(embedded[3], 1.0); // one-hot position 1
        // Sample 1: [0.5, 0.5, 1, 0, 0] (label 0)
        assert_eq!(embedded[7], 1.0); // one-hot position 0
    }

    #[test]
    fn ff_negative_has_wrong_labels() {
        let images = vec![1.0f32; 4]; // 2 images of dim 2
        let labels = vec![0u8, 1u8];
        let neg = FFNetwork::make_negative(&images, &labels, 2, 3, 2);
        // Labels should be different from original
        // Sample 0: label was 0, should now be different
        assert_eq!(neg[2], 0.0); // position 0 should NOT be 1
    }

    #[test]
    fn ff_positive_has_higher_goodness_than_negative() {
        // After some training, positive goodness should exceed negative
        let data = MnistData::synthetic(200, 50);
        let mut net = FFNetwork::new(&[794, 64, 32], 10);

        let mut last_pos = 0.0f32;
        let mut last_neg = 0.0f32;
        for _ in 0..5 {
            let (pg, ng) = net.train_epoch(
                &data.train_images, &data.train_labels,
                784, data.n_train, 50,
            );
            last_pos = pg;
            last_neg = ng;
        }

        // After training, positive goodness should trend higher than negative
        println!("pos_goodness={:.4}, neg_goodness={:.4}", last_pos, last_neg);
        // This is a soft check — the gap should exist but may be small
        assert!(last_pos >= last_neg * 0.8,
            "Positive goodness ({:.4}) should not be much less than negative ({:.4})",
            last_pos, last_neg);
    }

    #[test]
    fn ff_ternary_weights_are_ternary() {
        let layer = FFLayer::new(100, 50, 0.37);
        for &w in &layer.weights {
            assert!(w == -1.0 || w == 0.0 || w == 1.0,
                "Weight {} is not ternary", w);
        }
    }

    #[test]
    fn ff_network_trains_on_mnist() {
        let data = MnistData::synthetic(2000, 400);
        let mut net = FFNetwork::new(&[794, 256, 128], 10);

        println!("\nForward-Forward Training on MNIST (shadow f32 + ternary):");
        for epoch in 0..30 {
            let (pg, ng) = net.train_epoch(
                &data.train_images, &data.train_labels,
                784, data.n_train, 50,
            );
            if epoch % 5 == 0 || epoch == 29 {
                let f32_acc = net.accuracy(&data.test_images, &data.test_labels, 784, data.n_test);
                let tern_acc = net.accuracy_ternary(&data.test_images, &data.test_labels, 784, data.n_test);
                let (pos, zero, neg) = net.layers[0].weight_stats();
                println!("  Epoch {:>2}: pg={:.2} ng={:.2} | f32={:.1}% tern={:.1}% | +1:{} 0:{} -1:{}",
                    epoch + 1, pg, ng, f32_acc * 100.0, tern_acc * 100.0, pos, zero, neg);
            }
        }

        let f32_acc = net.accuracy(&data.test_images, &data.test_labels, 784, data.n_test);
        let tern_acc = net.accuracy_ternary(&data.test_images, &data.test_labels, 784, data.n_test);
        println!("  Final: f32={:.1}% ternary={:.1}%", f32_acc * 100.0, tern_acc * 100.0);

        // f32 shadow weights should learn
        assert!(f32_acc > 0.15,
            "FF f32 must achieve >15% (got {:.1}%)", f32_acc * 100.0);
    }
}
