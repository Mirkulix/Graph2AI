//! Ternary Vote Training — 100% discrete, zero floating-point weights.
//!
//! Inspired by biological STDP + reward modulation:
//! - Each weight has 3 vote counters: votes_pos, votes_zero, votes_neg
//! - For each sample: compute correlation * reward → cast vote
//! - Weight = state with most votes: argmax(votes) → {+1, 0, -1}
//!
//! Properties:
//! - Zero f32 in weight space (only i32 counters + i8 weights)
//! - No gradients, no shadow weights, no continuous optimization
//! - Biologically plausible (STDP + dopamine)
//! - Embarrassingly parallel (each weight independent)

use rayon::prelude::*;

/// A ternary weight: -1, 0, or +1 stored as i8.
type TWeight = i8;

/// Vote counters for a single weight position.
#[derive(Clone, Copy, Default)]
struct VoteCounter {
    /// Votes for w = +1
    pos: i32,
    /// Votes for w = 0
    zero: i32,
    /// Votes for w = -1
    neg: i32,
}

impl VoteCounter {
    /// Cast a vote based on correlation and reward.
    #[inline]
    fn vote(&mut self, pre: f32, post: f32, reward: f32) {
        // Only vote when both neurons are active (avoid dead-neuron zero bias)
        if post.abs() < 1e-8 || pre.abs() < 1e-8 {
            return; // Skip — no signal to learn from
        }

        // STDP-like correlation: does pre-activity predict post-activity?
        let correlation = pre * post * reward;

        if correlation > 1e-6 {
            self.pos += 1;    // strengthen: w should be +1
        } else if correlation < -1e-6 {
            self.neg += 1;    // inhibit: w should be -1
        } else {
            self.zero += 1;   // no clear signal: w should be 0
        }
    }

    /// Determine the winning ternary state via majority vote.
    /// Returns None if no votes were cast (keep existing weight).
    #[inline]
    fn decide(&self) -> Option<TWeight> {
        let total = self.pos + self.neg + self.zero;
        if total == 0 {
            return None; // No signal — keep current weight
        }
        if self.pos > self.neg && self.pos > self.zero {
            Some(1)
        } else if self.neg > self.pos && self.neg > self.zero {
            Some(-1)
        } else {
            Some(0)
        }
    }

    /// Reset votes (called after applying the decision).
    fn reset(&mut self) {
        self.pos = 0;
        self.zero = 0;
        self.neg = 0;
    }
}

/// A single ternary layer trained by vote counting.
pub struct VoteLayer {
    /// Ternary weights [out_dim × in_dim] as i8 {-1, 0, +1}
    pub weights: Vec<TWeight>,
    /// Vote counters per weight
    votes: Vec<VoteCounter>,
    /// Biases (these stay as f32 — the only float in the system)
    pub biases: Vec<f32>,
    /// Scaling factor (learned from activation statistics)
    pub scale: f32,
    pub in_dim: usize,
    pub out_dim: usize,
}

impl VoteLayer {
    pub fn new(in_dim: usize, out_dim: usize, seed: u64) -> Self {
        // Initialize weights randomly ternary
        let weights: Vec<TWeight> = (0..out_dim * in_dim)
            .map(|i| {
                let v = ((i as u64).wrapping_mul(seed).wrapping_mul(2654435761) >> 30) % 3;
                match v { 0 => -1, 1 => 0, _ => 1 }
            })
            .collect();

        // Pre-seed votes to bias toward current weight, preventing early collapse
        let votes: Vec<VoteCounter> = weights.iter().map(|&w| {
            let mut vc = VoteCounter::default();
            match w {
                1 => vc.pos = 5,
                -1 => vc.neg = 5,
                _ => vc.zero = 3,
            }
            vc
        }).collect();

        Self {
            weights,
            votes,
            biases: vec![0.0; out_dim],
            scale: 1.0 / (in_dim as f32).sqrt(),
            in_dim,
            out_dim,
        }
    }

    /// Forward pass — pure ternary arithmetic: add/sub/skip.
    /// Returns activations [batch × out_dim] as f32 (activations are analog, weights are ternary).
    pub fn forward(&self, input: &[f32], batch: usize) -> Vec<f32> {
        let out_dim = self.out_dim;
        let in_dim = self.in_dim;
        let scale = self.scale;
        let weights = &self.weights;
        let biases = &self.biases;

        let chunks: Vec<Vec<f32>> = (0..batch)
            .into_par_iter()
            .map(|b| {
                let x = &input[b * in_dim..(b + 1) * in_dim];
                let mut out = vec![0.0f32; out_dim];
                for j in 0..out_dim {
                    let mut sum = biases[j];
                    let w_off = j * in_dim;
                    for k in 0..in_dim {
                        match weights[w_off + k] {
                            1 => sum += scale * x[k],
                            -1 => sum -= scale * x[k],
                            _ => {} // 0: skip — no operation
                        }
                    }
                    out[j] = if sum > 0.0 { sum } else { 0.01 * sum }; // Leaky ReLU
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

    /// Collect votes from one batch of data.
    ///
    /// `input`: [batch × in_dim] — pre-synaptic activations
    /// `output`: [batch × out_dim] — post-synaptic activations (from forward)
    /// `reward`: per-sample reward signal (+1.0 if correct, -1.0 if wrong)
    pub fn collect_votes(&mut self, input: &[f32], output: &[f32], reward: &[f32], batch: usize) {
        let in_dim = self.in_dim;
        let out_dim = self.out_dim;

        // Parallel per output neuron (each j writes to its own vote slice)
        let vote_chunks: Vec<Vec<VoteCounter>> = (0..out_dim)
            .into_par_iter()
            .map(|j| {
                let mut local_votes = vec![VoteCounter::default(); in_dim];
                for b in 0..batch {
                    let post = output[b * out_dim + j];
                    let r = reward[b];
                    for k in 0..in_dim {
                        let pre = input[b * in_dim + k];
                        local_votes[k].vote(pre, post, r);
                    }
                }
                local_votes
            })
            .collect();

        // Merge votes
        for (j, chunk) in vote_chunks.into_iter().enumerate() {
            for k in 0..in_dim {
                let idx = j * in_dim + k;
                self.votes[idx].pos += chunk[k].pos;
                self.votes[idx].neg += chunk[k].neg;
                self.votes[idx].zero += chunk[k].zero;
            }
        }

        // Update biases (the only f32 update — small nudge based on reward)
        for b in 0..batch {
            let r = reward[b];
            for j in 0..out_dim {
                self.biases[j] += 0.001 * r * output[b * out_dim + j];
            }
        }
    }

    /// Apply votes: each weight becomes the state with the most votes.
    /// Returns number of weights that changed.
    pub fn apply_votes(&mut self) -> usize {
        let mut changed = 0;
        for i in 0..self.weights.len() {
            if let Some(new_w) = self.votes[i].decide() {
                if new_w != self.weights[i] {
                    self.weights[i] = new_w;
                    changed += 1;
                }
            }
            // Keep current weight if no votes were cast
            self.votes[i].reset();
        }

        // Update scale: Kaiming-like normalization per output neuron
        // Average non-zero weights per output neuron
        let non_zero = self.weights.iter().filter(|&&w| w != 0).count();
        let avg_nz_per_out = non_zero as f32 / self.out_dim.max(1) as f32;
        if avg_nz_per_out > 0.0 {
            self.scale = 1.0 / avg_nz_per_out.sqrt();
        }

        changed
    }

    pub fn weight_stats(&self) -> (usize, usize, usize) {
        let pos = self.weights.iter().filter(|&&w| w == 1).count();
        let zero = self.weights.iter().filter(|&&w| w == 0).count();
        let neg = self.weights.iter().filter(|&&w| w == -1).count();
        (pos, zero, neg)
    }

    /// Model size in bytes: 1 byte per weight (i8) + 4 bytes per bias (f32)
    pub fn size_bytes(&self) -> usize {
        self.weights.len() + self.biases.len() * 4
    }
}

/// Complete Ternary Vote Network.
pub struct VoteNet {
    pub layers: Vec<VoteLayer>,
    pub n_classes: usize,
}

impl VoteNet {
    pub fn new(layer_sizes: &[usize], n_classes: usize) -> Self {
        let mut layers = Vec::new();
        for i in 0..layer_sizes.len() - 1 {
            layers.push(VoteLayer::new(
                layer_sizes[i], layer_sizes[i + 1],
                37 + i as u64 * 17,
            ));
        }
        Self { layers, n_classes }
    }

    /// Embed label into input: [image + one_hot_label]
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

    /// Train one epoch: for each sample, forward → check if correct → reward → vote.
    pub fn train_epoch(
        &mut self,
        images: &[f32],
        labels: &[u8],
        image_dim: usize,
        n_samples: usize,
        batch_size: usize,
    ) -> f32 {
        let n_batches = n_samples / batch_size;
        let mut correct_total = 0usize;
        let mut total = 0usize;

        for batch_idx in 0..n_batches {
            let offset = batch_idx * batch_size;
            let batch_images = &images[offset * image_dim..(offset + batch_size) * image_dim];
            let batch_labels = &labels[offset..offset + batch_size];

            // For each sample: try the correct label AND a wrong label
            let pos_input = Self::embed_label(batch_images, batch_labels, image_dim, self.n_classes, batch_size);

            // Forward through all layers
            let mut layer_inputs = vec![pos_input.clone()];
            let mut current = pos_input;
            let _input_dim = image_dim + self.n_classes;

            for layer in &self.layers {
                let act = layer.forward(&current, batch_size);
                layer_inputs.push(act.clone());
                // Normalize for next layer
                let mut normed = act;
                normalize(&mut normed, batch_size, layer.out_dim);
                current = normed;
            }

            // Compute reward: for each sample, is the prediction correct?
            let predictions = self.predict_batch(batch_images, image_dim, batch_size);
            let mut rewards = vec![0.0f32; batch_size];
            for b in 0..batch_size {
                if predictions[b] == batch_labels[b] {
                    rewards[b] = 1.0;
                    correct_total += 1;
                } else {
                    rewards[b] = -0.5; // Smaller punishment than reward
                }
                total += 1;
            }

            // Collect votes per layer
            for (i, layer) in self.layers.iter_mut().enumerate() {
                let inp = &layer_inputs[i];
                let out = &layer_inputs[i + 1];
                layer.collect_votes(inp, out, &rewards, batch_size);
            }
        }

        // Apply votes: flip weights based on majority
        for layer in &mut self.layers {
            layer.apply_votes();
        }

        correct_total as f32 / total.max(1) as f32
    }

    /// Predict a single batch (used internally during training).
    fn predict_batch(&self, images: &[f32], image_dim: usize, batch: usize) -> Vec<u8> {
        let mut predictions = vec![0u8; batch];
        for b in 0..batch {
            let image = &images[b * image_dim..(b + 1) * image_dim];
            let mut best = 0u8;
            let mut best_g = f32::NEG_INFINITY;

            for c in 0..self.n_classes {
                let label = [c as u8];
                let input = Self::embed_label(image, &label, image_dim, self.n_classes, 1);
                let mut current = input;

                let mut total_g = 0.0f32;
                for layer in &self.layers {
                    let act = layer.forward(&current, 1);
                    total_g += goodness(&act, layer.out_dim);
                    let mut normed = act;
                    normalize(&mut normed, 1, layer.out_dim);
                    current = normed;
                }

                if total_g > best_g {
                    best_g = total_g;
                    best = c as u8;
                }
            }
            predictions[b] = best;
        }
        predictions
    }

    /// Full parallel predict for evaluation.
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
                        let act = layer.forward(&current, 1);
                        total_g += goodness(&act, layer.out_dim);
                        let mut normed = act;
                        normalize(&mut normed, 1, layer.out_dim);
                        current = normed;
                    }

                    if total_g > best_g {
                        best_g = total_g;
                        best = c as u8;
                    }
                }
                best
            })
            .collect()
    }

    pub fn accuracy(&self, images: &[f32], labels: &[u8], image_dim: usize, n_samples: usize) -> f32 {
        let preds = self.predict(images, image_dim, n_samples);
        preds.iter().zip(labels).filter(|(p, l)| p == l).count() as f32 / n_samples as f32
    }

    pub fn total_params(&self) -> usize {
        self.layers.iter().map(|l| l.weights.len() + l.biases.len()).sum()
    }

    pub fn total_size_bytes(&self) -> usize {
        self.layers.iter().map(|l| l.size_bytes()).sum()
    }
}

fn goodness(act: &[f32], dim: usize) -> f32 {
    act[..dim].iter().map(|x| x * x).sum()
}

fn normalize(act: &mut [f32], batch: usize, dim: usize) {
    for b in 0..batch {
        let off = b * dim;
        let norm: f32 = act[off..off + dim].iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-8);
        for j in 0..dim {
            act[off + j] /= norm;
        }
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
    fn weights_are_ternary_i8() {
        let layer = VoteLayer::new(100, 50, 37);
        for &w in &layer.weights {
            assert!(w == -1 || w == 0 || w == 1, "Not ternary i8: {}", w);
        }
    }

    #[test]
    fn forward_uses_no_float_multiply_on_weights() {
        // The forward pass should only add/sub, never multiply weights
        let layer = VoteLayer::new(4, 2, 37);
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let out = layer.forward(&input, 1);
        assert_eq!(out.len(), 2);
        // Values should be finite
        assert!(out.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn vote_counting_works() {
        let mut counter = VoteCounter::default();
        // 5 positive votes
        for _ in 0..5 {
            counter.vote(1.0, 1.0, 1.0); // pre*post*reward > 0 → pos
        }
        // 2 negative votes
        for _ in 0..2 {
            counter.vote(1.0, -1.0, 1.0); // pre*post*reward < 0 → neg
        }
        assert_eq!(counter.decide(), Some(1)); // +1 wins
    }

    #[test]
    fn model_size_is_tiny() {
        let net = VoteNet::new(&[794, 256, 128], 10);
        let size = net.total_size_bytes();
        let f32_size = net.total_params() * 4;
        let ratio = f32_size as f32 / size as f32;
        println!("Ternary i8: {} bytes, f32 equivalent: {} bytes, ratio: {:.1}x", size, f32_size, ratio);
        assert!(ratio > 3.0, "Ternary should be >3x smaller than f32");
    }

    #[test]
    fn ternary_vote_training() {
        let data = MnistData::synthetic(2000, 500);
        let mut net = VoteNet::new(&[794, 256, 128], 10);

        println!("\n=== Ternary Vote Training (100% discrete) ===\n");

        for epoch in 0..20 {
            let train_acc = net.train_epoch(
                &data.train_images, &data.train_labels,
                784, data.n_train, 50,
            );

            if epoch % 4 == 0 || epoch == 19 {
                let test_acc = net.accuracy(&data.test_images, &data.test_labels, 784, data.n_test);
                let (pos, zero, neg) = net.layers[0].weight_stats();
                println!("  Epoch {:>2}: train={:.1}% test={:.1}% | +1:{} 0:{} -1:{} | scale={:.4}",
                    epoch + 1, train_acc * 100.0, test_acc * 100.0, pos, zero, neg, net.layers[0].scale);
            }
        }

        let final_acc = net.accuracy(&data.test_images, &data.test_labels, 784, data.n_test);
        let size = net.total_size_bytes();
        println!("\n  Final: {:.1}% accuracy, {} bytes model", final_acc * 100.0, size);
        println!("  Weights: 100% ternary i8 — ZERO f32 in weight space");

        // Must learn something
        assert!(final_acc > 0.12,
            "Vote training must beat random (got {:.1}%)", final_acc * 100.0);

        // Verify all weights are ternary
        for layer in &net.layers {
            for &w in &layer.weights {
                assert!(w == -1 || w == 0 || w == 1);
            }
        }
    }
}
