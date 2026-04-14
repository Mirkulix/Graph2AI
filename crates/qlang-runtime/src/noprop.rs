//! NoProp - Local loss propagation without backprop chain.
//!
//! NOTE: Reference paper claims 99.47% MNIST / 80.5% CIFAR-10.
//! THIS implementation: tested on synthetic data only, asserts >30% accuracy.
//! Real benchmark TODO.

use rayon::prelude::*;

/// One denoising block in the NoProp pipeline.
#[derive(Clone)]
pub struct NoPropBlock {
    /// Shadow weights [in_dim, out_dim] (f32 for learning)
    pub shadow: Vec<f32>,
    /// Ternary weights (for inference)
    pub ternary: Vec<f32>,
    pub alpha: f32,
    /// Biases
    pub biases: Vec<f32>,
    pub in_dim: usize,
    pub out_dim: usize, // = label_dim (embedding dimension for labels)
    pub lr: f32,
}

impl NoPropBlock {
    pub fn new(in_dim: usize, out_dim: usize, seed: f32) -> Self {
        let scale = (2.0 / (in_dim + out_dim) as f64).sqrt() as f32;
        let shadow: Vec<f32> = (0..in_dim * out_dim)
            .map(|i| (i as f32 * seed).sin() * scale)
            .collect();
        let ternary = shadow.iter().map(|&w| {
            if w > scale * 0.5 { 1.0 } else if w < -scale * 0.5 { -1.0 } else { 0.0 }
        }).collect();
        let biases = vec![0.0f32; out_dim];

        Self { shadow, ternary, alpha: scale, biases, in_dim, out_dim, lr: 0.005 }
    }

    /// Forward: concatenate(noisy_label, input) → predict clean label embedding.
    /// Input: [batch, in_dim] where in_dim = label_dim + feat_dim
    /// Output: [batch, label_dim]
    fn forward(&self, input: &[f32], batch: usize) -> Vec<f32> {
        let mut output = vec![0.0f32; batch * self.out_dim];
        for b in 0..batch {
            for j in 0..self.out_dim {
                let mut sum = self.biases[j];
                for k in 0..self.in_dim {
                    sum += input[b * self.in_dim + k] * self.shadow[k * self.out_dim + j];
                }
                // Tanh activation (bounded output for label prediction)
                output[b * self.out_dim + j] = sum.tanh();
            }
        }
        output
    }

    /// Train one step: given (noisy_label + features) pairs, learn to predict clean label.
    ///
    /// Loss = ||predicted_label - clean_label_embedding||^2
    /// Gradient: dL/dW = 2 * (pred - target) * input (per weight)
    pub fn train_step(
        &mut self,
        noisy_concat: &[f32], // [batch, label_dim + feat_dim]
        clean_labels: &[f32], // [batch, label_dim]
        batch: usize,
    ) -> f32 {
        let predicted = self.forward(noisy_concat, batch);

        // Compute L2 loss
        let mut loss = 0.0f32;
        for i in 0..batch * self.out_dim {
            let diff = predicted[i] - clean_labels[i];
            loss += diff * diff;
        }
        loss /= (batch * self.out_dim) as f32;

        // Gradient update (parallel over output dim)
        let inv_batch = 1.0 / batch as f32;
        let deltas: Vec<(Vec<f32>, f32)> = (0..self.out_dim)
            .into_par_iter()
            .map(|j| {
                let mut w_delta = vec![0.0f32; self.in_dim];
                let mut b_delta = 0.0f32;
                for b in 0..batch {
                    let pred = predicted[b * self.out_dim + j];
                    let target = clean_labels[b * self.out_dim + j];
                    // d(tanh)/dx = 1 - tanh^2(x)
                    let grad = 2.0 * (pred - target) * (1.0 - pred * pred);
                    for k in 0..self.in_dim {
                        w_delta[k] -= self.lr * grad * noisy_concat[b * self.in_dim + k] * inv_batch;
                    }
                    b_delta -= self.lr * grad * inv_batch;
                }
                (w_delta, b_delta)
            })
            .collect();

        for (j, (w_delta, b_delta)) in deltas.into_iter().enumerate() {
            for k in 0..self.in_dim {
                self.shadow[k * self.out_dim + j] += w_delta[k];
            }
            self.biases[j] += b_delta;
        }

        loss
    }

    /// Sync ternary weights from shadow.
    pub fn sync_ternary(&mut self) {
        let mean_abs: f32 = self.shadow.iter().map(|w| w.abs()).sum::<f32>() / self.shadow.len() as f32;
        let threshold = mean_abs * 0.7;
        for i in 0..self.shadow.len() {
            self.ternary[i] = if self.shadow[i] > threshold { 1.0 }
                else if self.shadow[i] < -threshold { -1.0 }
                else { 0.0 };
        }
        let mut dot_st = 0.0f32;
        let mut dot_tt = 0.0f32;
        for i in 0..self.shadow.len() {
            dot_st += self.shadow[i] * self.ternary[i];
            dot_tt += self.ternary[i] * self.ternary[i];
        }
        if dot_tt > 0.0 { self.alpha = (dot_st / dot_tt).max(0.001); }
    }

    /// Denoise one step using ternary weights (for inference).
    fn denoise_ternary(&self, input: &[f32], batch: usize) -> Vec<f32> {
        let mut output = vec![0.0f32; batch * self.out_dim];
        for b in 0..batch {
            for j in 0..self.out_dim {
                let mut sum = self.biases[j];
                for k in 0..self.in_dim {
                    let w = self.ternary[k * self.out_dim + j];
                    if w > 0.5 { sum += self.alpha * input[b * self.in_dim + k]; }
                    else if w < -0.5 { sum -= self.alpha * input[b * self.in_dim + k]; }
                }
                output[b * self.out_dim + j] = sum.tanh();
            }
        }
        output
    }
}

/// NoProp network: T denoising blocks.
pub struct NoPropNet {
    pub blocks: Vec<NoPropBlock>,
    pub n_classes: usize,
    pub label_dim: usize,
    pub feat_dim: usize,
    pub n_steps: usize,
}

impl NoPropNet {
    /// Create a NoProp network with T denoising steps.
    ///
    /// Each block: [label_dim + feat_dim] → [label_dim]
    pub fn new(feat_dim: usize, label_dim: usize, n_classes: usize, n_steps: usize) -> Self {
        let in_dim = label_dim + feat_dim;
        let blocks: Vec<NoPropBlock> = (0..n_steps)
            .map(|t| NoPropBlock::new(in_dim, label_dim, 0.37 + t as f32 * 0.13))
            .collect();
        Self { blocks, n_classes, label_dim, feat_dim, n_steps }
    }

    /// Create one-hot label embedding: label → [label_dim] vector.
    fn label_embedding(&self, label: u8) -> Vec<f32> {
        let mut emb = vec![0.0f32; self.label_dim];
        if (label as usize) < self.label_dim {
            emb[label as usize] = 1.0;
        }
        emb
    }

    /// Add noise to label embedding at step t.
    /// More noise for early steps, less for later steps.
    fn add_noise(&self, clean: &[f32], step: usize, sample_seed: usize) -> Vec<f32> {
        let noise_level = 1.0 - (step as f32 / self.n_steps as f32); // 1.0 → 0.0
        let mut noisy = vec![0.0f32; clean.len()];
        let mut rng = (sample_seed * 31337 + step * 7919) as u64;
        for i in 0..clean.len() {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            let noise = ((rng >> 33) as f64 / u32::MAX as f64 - 0.5) as f32 * 2.0;
            noisy[i] = clean[i] * (1.0 - noise_level) + noise * noise_level;
        }
        noisy
    }

    /// Train one epoch: for each sample and each step, train the corresponding block.
    pub fn train_epoch(
        &mut self,
        features: &[f32],
        labels: &[u8],
        n_samples: usize,
        batch_size: usize,
    ) -> f32 {
        let n_batches = n_samples / batch_size;
        let mut total_loss = 0.0f32;

        for batch_idx in 0..n_batches {
            let off = batch_idx * batch_size;

            // For each denoising step, train that block
            for step in 0..self.n_steps {
                // Build training data for this block:
                // input = concat(noisy_label_at_step, features)
                // target = clean_label_embedding
                let mut noisy_concat = vec![0.0f32; batch_size * (self.label_dim + self.feat_dim)];
                let mut clean_targets = vec![0.0f32; batch_size * self.label_dim];

                for b in 0..batch_size {
                    let sample_idx = off + b;
                    let clean = self.label_embedding(labels[sample_idx]);
                    let noisy = self.add_noise(&clean, step, sample_idx);

                    // Concat: [noisy_label, features]
                    let row_start = b * (self.label_dim + self.feat_dim);
                    noisy_concat[row_start..row_start + self.label_dim].copy_from_slice(&noisy);
                    noisy_concat[row_start + self.label_dim..row_start + self.label_dim + self.feat_dim]
                        .copy_from_slice(&features[sample_idx * self.feat_dim..(sample_idx + 1) * self.feat_dim]);

                    // Target: clean label
                    clean_targets[b * self.label_dim..(b + 1) * self.label_dim].copy_from_slice(&clean);
                }

                let loss = self.blocks[step].train_step(&noisy_concat, &clean_targets, batch_size);
                total_loss += loss;
            }
        }

        // Sync ternary
        for block in &mut self.blocks { block.sync_ternary(); }

        total_loss / (n_batches * self.n_steps) as f32
    }

    /// Predict: start from noise, denoise T steps, pick argmax.
    pub fn predict(&self, features: &[f32], n_samples: usize, use_ternary: bool) -> Vec<u8> {
        (0..n_samples)
            .into_par_iter()
            .map(|i| {
                let feat = &features[i * self.feat_dim..(i + 1) * self.feat_dim];

                // Start from zeros (or small noise)
                let mut z = vec![0.0f32; self.label_dim];

                // Denoise through all blocks
                for (_step, block) in self.blocks.iter().enumerate() {
                    let mut concat = Vec::with_capacity(self.label_dim + self.feat_dim);
                    concat.extend_from_slice(&z);
                    concat.extend_from_slice(feat);

                    z = if use_ternary {
                        block.denoise_ternary(&concat, 1)
                    } else {
                        block.forward(&concat, 1)
                    };
                }

                // Argmax of final denoised label
                z.iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .map(|(idx, _)| idx as u8)
                    .unwrap_or(0)
            })
            .collect()
    }

    pub fn accuracy(&self, features: &[f32], labels: &[u8], n_samples: usize, use_ternary: bool) -> f32 {
        let preds = self.predict(features, n_samples, use_ternary);
        preds.iter().zip(labels).filter(|(p, l)| p == l).count() as f32 / n_samples as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mnist::MnistData;

    #[test]
    fn noprop_trains_on_mnist() {
        let data = MnistData::synthetic(2000, 500);

        let mut net = NoPropNet::new(784, 10, 10, 5); // 5 denoising steps

        println!("\nNoProp Training on MNIST:");
        for epoch in 0..15 {
            let loss = net.train_epoch(&data.train_images, &data.train_labels, data.n_train, 50);
            if epoch % 3 == 0 || epoch == 14 {
                let f32_acc = net.accuracy(&data.test_images, &data.test_labels, data.n_test, false);
                let tern_acc = net.accuracy(&data.test_images, &data.test_labels, data.n_test, true);
                println!("  Epoch {:>2}: loss={:.4} f32={:.1}% tern={:.1}%",
                    epoch + 1, loss, f32_acc * 100.0, tern_acc * 100.0);
            }
        }

        let f32_acc = net.accuracy(&data.test_images, &data.test_labels, data.n_test, false);
        println!("  Final f32: {:.1}%", f32_acc * 100.0);

        // Sanity check only: NoProp + synthetic data should at least beat random (10%).
        // Real performance benchmarks on MNIST TODO.
        assert!(f32_acc > 0.30, "NoProp must learn (got {:.1}%)", f32_acc * 100.0);
    }
}
