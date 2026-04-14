//! Forward-Forward Training ON QLANG Graphs.
//!
//! This is the key innovation: take any QLANG graph with MatMul nodes,
//! and train it using Forward-Forward — each MatMul layer learns locally.
//!
//! Pipeline:
//! 1. Build a QLANG graph (e.g., Input → MatMul → ReLU → MatMul → Softmax → Output)
//! 2. Each MatMul node gets shadow weights (f32) + ternary weights
//! 3. Forward-Forward: positive pass (correct label) vs negative pass (wrong label)
//! 4. Each MatMul layer updates its own shadow weights based on local goodness
//! 5. Sync: shadow → ternary after each epoch
//!
//! No gradient flows between layers. Each layer is independent.
//! This means layers can train on different machines.

use rayon::prelude::*;

/// Trainable weights for one MatMul layer in a QLANG graph.
#[derive(Clone)]
pub struct LayerWeights {
    /// Shadow weights (f32) for learning
    pub shadow: Vec<f32>,
    /// Ternary weights {-1, 0, +1} for inference
    pub ternary: Vec<f32>,
    /// Biases
    pub biases: Vec<f32>,
    /// Learned scale factor
    pub alpha: f32,
    /// Dimensions
    pub in_dim: usize,
    pub out_dim: usize,
    /// Goodness threshold
    pub threshold: f32,
    /// Learning rate
    pub lr: f32,
}

impl LayerWeights {
    pub fn new(in_dim: usize, out_dim: usize, seed: f32) -> Self {
        let scale = (2.0 / (in_dim + out_dim) as f64).sqrt() as f32;
        let shadow: Vec<f32> = (0..in_dim * out_dim)
            .map(|i| (i as f32 * seed).sin() * scale)
            .collect();
        let ternary = shadow.iter().map(|&w| {
            if w > scale * 0.5 { 1.0 } else if w < -scale * 0.5 { -1.0 } else { 0.0 }
        }).collect();
        let biases = vec![0.0f32; out_dim];

        Self { shadow, ternary, biases, alpha: scale, in_dim, out_dim, threshold: out_dim as f32, lr: 0.03 }
    }

    /// Forward pass using shadow weights: output = ReLU(input @ W + bias)
    pub fn forward(&self, input: &[f32], batch: usize) -> Vec<f32> {
        let mut output = vec![0.0f32; batch * self.out_dim];
        for b in 0..batch {
            for j in 0..self.out_dim {
                let mut sum = self.biases[j];
                for k in 0..self.in_dim {
                    sum += input[b * self.in_dim + k] * self.shadow[k * self.out_dim + j];
                }
                output[b * self.out_dim + j] = sum.max(0.0);
            }
        }
        output
    }

    /// Forward with ternary weights (for inference)
    pub fn forward_ternary(&self, input: &[f32], batch: usize) -> Vec<f32> {
        let mut output = vec![0.0f32; batch * self.out_dim];
        for b in 0..batch {
            for j in 0..self.out_dim {
                let mut sum = self.biases[j];
                for k in 0..self.in_dim {
                    let w = self.ternary[k * self.out_dim + j];
                    if w > 0.5 { sum += self.alpha * input[b * self.in_dim + k]; }
                    else if w < -0.5 { sum -= self.alpha * input[b * self.in_dim + k]; }
                }
                output[b * self.out_dim + j] = sum.max(0.0);
            }
        }
        output
    }

    /// Normalize activations per sample.
    fn normalize(act: &mut [f32], batch: usize, dim: usize) {
        for b in 0..batch {
            let off = b * dim;
            let norm: f32 = act[off..off + dim].iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-8);
            for j in 0..dim { act[off + j] /= norm; }
        }
    }

    /// Goodness = mean sum of squared activations.
    fn goodness(act: &[f32], batch: usize, dim: usize) -> f32 {
        let mut total = 0.0f32;
        for b in 0..batch {
            let off = b * dim;
            total += act[off..off + dim].iter().map(|x| x * x).sum::<f32>();
        }
        total / batch as f32
    }

    /// One Forward-Forward step on this layer.
    pub fn ff_step(&mut self, pos_input: &[f32], neg_input: &[f32], batch: usize) -> (f32, f32) {
        let pos_act = self.forward(pos_input, batch);
        let pos_goodness = Self::goodness(&pos_act, batch, self.out_dim);
        let neg_act = self.forward(neg_input, batch);
        let neg_goodness = Self::goodness(&neg_act, batch, self.out_dim);

        let inv_batch = 1.0 / batch as f32;

        // Pre-compute per-sample signals
        let signals: Vec<(f32, f32)> = (0..batch).map(|b| {
            let pg: f32 = (0..self.out_dim).map(|j| pos_act[b * self.out_dim + j].powi(2)).sum();
            let ng: f32 = (0..self.out_dim).map(|j| neg_act[b * self.out_dim + j].powi(2)).sum();
            (1.0 / (1.0 + (-(pg - self.threshold)).exp()), 1.0 / (1.0 + (-(ng - self.threshold)).exp()))
        }).collect();

        // Parallel weight update per output neuron
        let deltas: Vec<(Vec<f32>, f32)> = (0..self.out_dim)
            .into_par_iter()
            .map(|j| {
                let mut w_delta = vec![0.0f32; self.in_dim];
                let mut b_delta = 0.0f32;
                for b in 0..batch {
                    let (pos_p, neg_p) = signals[b];
                    let d_pos = (1.0 - pos_p) * 2.0 * pos_act[b * self.out_dim + j];
                    let d_neg = neg_p * 2.0 * neg_act[b * self.out_dim + j];
                    for k in 0..self.in_dim {
                        w_delta[k] += self.lr * (d_pos * pos_input[b * self.in_dim + k]
                            - d_neg * neg_input[b * self.in_dim + k]) * inv_batch;
                    }
                    b_delta += self.lr * (d_pos - d_neg) * inv_batch;
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

        (pos_goodness, neg_goodness)
    }

    /// Sync ternary from shadow + compute optimal alpha.
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
}

/// A trainable QLANG graph: multiple layers that learn via Forward-Forward.
pub struct TrainableGraph {
    pub layers: Vec<LayerWeights>,
    pub n_classes: usize,
}

impl TrainableGraph {
    /// Create from layer dimensions: e.g. [512, 256, 128, 10]
    /// First dim includes label embedding (+ n_classes).
    pub fn new(dims: &[usize], n_classes: usize) -> Self {
        let mut layers = Vec::new();
        for i in 0..dims.len() - 1 {
            let seed = 0.37 + i as f32 * 0.17;
            layers.push(LayerWeights::new(dims[i], dims[i + 1], seed));
        }
        Self { layers, n_classes }
    }

    /// Embed label into features: [feat_dim] → [feat_dim + n_classes]
    pub fn embed_label(features: &[f32], labels: &[u8], feat_dim: usize, n_classes: usize, batch: usize) -> Vec<f32> {
        let total = feat_dim + n_classes;
        let mut embedded = vec![0.0f32; batch * total];
        for b in 0..batch {
            let feat = &features[b * feat_dim..(b + 1) * feat_dim];
            embedded[b * total..b * total + feat_dim].copy_from_slice(feat);
            let label = labels[b] as usize;
            if label < n_classes {
                // Scale label to match feature magnitude so FF can distinguish pos/neg
                let rms: f32 = feat.iter().map(|x| x * x).sum::<f32>() / feat_dim as f32;
                let scale = rms.sqrt().max(0.1);
                embedded[b * total + feat_dim + label] = scale;
            }
        }
        embedded
    }

    /// Generate negative data: same features but wrong labels.
    pub fn make_negative(features: &[f32], labels: &[u8], feat_dim: usize, n_classes: usize, batch: usize) -> Vec<f32> {
        let mut wrong = labels.to_vec();
        for b in 0..batch {
            wrong[b] = ((labels[b] as usize + 1 + (b * 7 + 3) % (n_classes - 1)) % n_classes) as u8;
        }
        Self::embed_label(features, &wrong, feat_dim, n_classes, batch)
    }

    /// Train one epoch.
    pub fn train_epoch(
        &mut self,
        features: &[f32],
        labels: &[u8],
        feat_dim: usize,
        n_samples: usize,
        batch_size: usize,
    ) -> (f32, f32) {
        let n_batches = n_samples / batch_size;
        let mut total_pos = 0.0f32;
        let mut total_neg = 0.0f32;

        for batch_idx in 0..n_batches {
            let off = batch_idx * batch_size;
            let batch_feat = &features[off * feat_dim..(off + batch_size) * feat_dim];
            let batch_labels = &labels[off..off + batch_size];

            let pos_input = Self::embed_label(batch_feat, batch_labels, feat_dim, self.n_classes, batch_size);
            let neg_input = Self::make_negative(batch_feat, batch_labels, feat_dim, self.n_classes, batch_size);

            let mut pos_layer = pos_input;
            let mut neg_layer = neg_input;

            for layer in &mut self.layers {
                let (pg, ng) = layer.ff_step(&pos_layer, &neg_layer, batch_size);
                total_pos += pg;
                total_neg += ng;

                // Next layer input = normalized output (detached)
                let mut pos_out = layer.forward(&pos_layer, batch_size);
                LayerWeights::normalize(&mut pos_out, batch_size, layer.out_dim);
                let mut neg_out = layer.forward(&neg_layer, batch_size);
                LayerWeights::normalize(&mut neg_out, batch_size, layer.out_dim);

                pos_layer = pos_out;
                neg_layer = neg_out;
            }
        }

        // Sync ternary
        for layer in &mut self.layers { layer.sync_ternary(); }

        let count = (n_batches * self.layers.len()) as f32;
        (total_pos / count.max(1.0), total_neg / count.max(1.0))
    }

    /// Predict: try all labels, pick highest total goodness (f32 weights).
    pub fn predict(&self, features: &[f32], feat_dim: usize, n_samples: usize) -> Vec<u8> {
        (0..n_samples)
            .into_par_iter()
            .map(|i| {
                let feat = &features[i * feat_dim..(i + 1) * feat_dim];
                let mut best_label = 0u8;
                let mut best_goodness = f32::NEG_INFINITY;

                for c in 0..self.n_classes {
                    let label = [c as u8];
                    let input = Self::embed_label(feat, &label, feat_dim, self.n_classes, 1);
                    let mut layer_input = input;
                    let mut total_g = 0.0f32;

                    for layer in &self.layers {
                        let mut act = layer.forward(&layer_input, 1);
                        total_g += LayerWeights::goodness(&act, 1, layer.out_dim);
                        LayerWeights::normalize(&mut act, 1, layer.out_dim);
                        layer_input = act;
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

    /// Predict with ternary weights.
    pub fn predict_ternary(&self, features: &[f32], feat_dim: usize, n_samples: usize) -> Vec<u8> {
        (0..n_samples)
            .into_par_iter()
            .map(|i| {
                let feat = &features[i * feat_dim..(i + 1) * feat_dim];
                let mut best_label = 0u8;
                let mut best_goodness = f32::NEG_INFINITY;

                for c in 0..self.n_classes {
                    let label = [c as u8];
                    let input = Self::embed_label(feat, &label, feat_dim, self.n_classes, 1);
                    let mut layer_input = input;
                    let mut total_g = 0.0f32;

                    for layer in &self.layers {
                        let mut act = layer.forward_ternary(&layer_input, 1);
                        total_g += LayerWeights::goodness(&act, 1, layer.out_dim);
                        LayerWeights::normalize(&mut act, 1, layer.out_dim);
                        layer_input = act;
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

    pub fn accuracy(&self, features: &[f32], labels: &[u8], feat_dim: usize, n_samples: usize) -> f32 {
        let preds = self.predict(features, feat_dim, n_samples);
        preds.iter().zip(labels).filter(|(p, l)| p == l).count() as f32 / n_samples as f32
    }

    pub fn accuracy_ternary(&self, features: &[f32], labels: &[u8], feat_dim: usize, n_samples: usize) -> f32 {
        let preds = self.predict_ternary(features, feat_dim, n_samples);
        preds.iter().zip(labels).filter(|(p, l)| p == l).count() as f32 / n_samples as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mnist::MnistData;

    #[test]
    fn graph_ff_trains_on_mnist() {
        let data = MnistData::synthetic(2000, 400);

        // 3-layer network: 794 → 256 → 128 → 64
        let mut graph = TrainableGraph::new(&[794, 256, 128, 64], 10);

        println!("\nGraph Forward-Forward Training (3 layers):");
        for epoch in 0..15 {
            let (pg, ng) = graph.train_epoch(
                &data.train_images, &data.train_labels,
                784, data.n_train, 50,
            );
            if epoch % 3 == 0 || epoch == 14 {
                let f32_acc = graph.accuracy(&data.test_images, &data.test_labels, 784, data.n_test);
                let tern_acc = graph.accuracy_ternary(&data.test_images, &data.test_labels, 784, data.n_test);
                println!("  Epoch {:>2}: f32={:.1}% tern={:.1}% pg={:.2} ng={:.2}",
                    epoch + 1, f32_acc * 100.0, tern_acc * 100.0, pg, ng);
            }
        }

        let f32_acc = graph.accuracy(&data.test_images, &data.test_labels, 784, data.n_test);
        let tern_acc = graph.accuracy_ternary(&data.test_images, &data.test_labels, 784, data.n_test);
        println!("  Final: f32={:.1}% ternary={:.1}%", f32_acc * 100.0, tern_acc * 100.0);

        assert!(f32_acc > 0.30, "Graph FF must learn (got {:.1}%)", f32_acc * 100.0);
    }
}
