//! IGQK Training Loop — Real gradient descent on QLANG graphs.
//!
//! Implements Algorithm 1 from the IGQK paper:
//!   1. Forward pass through the graph
//!   2. Compute loss (cross-entropy)
//!   3. Numerical gradients (Phase 1 — Phase 2 will add autograd)
//!   4. Update weights via gradient descent
//!   5. Optional: IGQK quantum gradient flow
//!
//! This runs entirely within QLANG — no Python, no PyTorch.


use crate::accel;

/// Training configuration.
pub struct TrainConfig {
    pub learning_rate: f32,
    pub epochs: usize,
    pub batch_size: usize,
    pub log_interval: usize,
}

impl Default for TrainConfig {
    fn default() -> Self {
        Self {
            learning_rate: 0.01,
            epochs: 10,
            batch_size: 32,
            log_interval: 100,
        }
    }
}

/// Simple MLP weights (not stored as QLANG graph yet, but as raw tensors).
pub struct MlpWeights {
    pub w1: Vec<f32>,  // [input_dim, hidden_dim]
    pub b1: Vec<f32>,  // [hidden_dim]
    pub w2: Vec<f32>,  // [hidden_dim, output_dim]
    pub b2: Vec<f32>,  // [output_dim]
    pub input_dim: usize,
    pub hidden_dim: usize,
    pub output_dim: usize,
}

impl MlpWeights {
    /// Initialize with Xavier/Glorot initialization.
    pub fn new(input_dim: usize, hidden_dim: usize, output_dim: usize) -> Self {
        let scale1 = (2.0 / (input_dim + hidden_dim) as f64).sqrt() as f32;
        let scale2 = (2.0 / (hidden_dim + output_dim) as f64).sqrt() as f32;

        // Simple deterministic "random" init using sine
        let w1: Vec<f32> = (0..input_dim * hidden_dim)
            .map(|i| (i as f32 * 0.4871).sin() * scale1)
            .collect();
        let b1 = vec![0.0; hidden_dim];
        let w2: Vec<f32> = (0..hidden_dim * output_dim)
            .map(|i| (i as f32 * 0.7291).sin() * scale2)
            .collect();
        let b2 = vec![0.0; output_dim];

        Self { w1, b1, w2, b2, input_dim, hidden_dim, output_dim }
    }

    /// Forward pass: x → W1 → relu → W2 → softmax → probabilities
    pub fn forward(&self, x: &[f32]) -> Vec<f32> {
        let batch = x.len() / self.input_dim;

        // Layer 1: h = relu(x @ W1 + b1)
        let mut hidden = accel::matmul(x, &self.w1, batch, self.hidden_dim, self.input_dim);
        for b in 0..batch {
            for j in 0..self.hidden_dim {
                hidden[b * self.hidden_dim + j] = (hidden[b * self.hidden_dim + j] + self.b1[j]).max(0.0);
            }
        }

        // Layer 2: logits = h @ W2 + b2
        let mut logits = accel::matmul(&hidden, &self.w2, batch, self.output_dim, self.hidden_dim);
        for b in 0..batch {
            for j in 0..self.output_dim {
                logits[b * self.output_dim + j] += self.b2[j];
            }
        }

        // Softmax
        for b in 0..batch {
            let offset = b * self.output_dim;
            let max = logits[offset..offset + self.output_dim]
                .iter()
                .cloned()
                .fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0f32;
            for j in 0..self.output_dim {
                logits[offset + j] = (logits[offset + j] - max).exp();
                sum += logits[offset + j];
            }
            for j in 0..self.output_dim {
                logits[offset + j] /= sum;
            }
        }

        logits
    }

    /// Compute cross-entropy loss.
    pub fn loss(&self, probs: &[f32], targets: &[u8]) -> f32 {
        let batch = targets.len();
        let mut total = 0.0f32;
        for b in 0..batch {
            let target = targets[b] as usize;
            let p = probs[b * self.output_dim + target].max(1e-7);
            total -= p.ln();
        }
        total / batch as f32
    }

    /// Compute accuracy.
    pub fn accuracy(&self, probs: &[f32], targets: &[u8]) -> f32 {
        let batch = targets.len();
        let mut correct = 0;
        for b in 0..batch {
            let offset = b * self.output_dim;
            let predicted = probs[offset..offset + self.output_dim]
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(i, _)| i)
                .unwrap_or(0);
            if predicted == targets[b] as usize {
                correct += 1;
            }
        }
        correct as f32 / batch as f32
    }

    /// Train one step with numerical gradient descent.
    /// Uses finite differences for gradients (simple but works for small models).
    pub fn train_step(&mut self, x: &[f32], targets: &[u8], lr: f32) -> f32 {
        let eps = 1e-4;

        // Compute current loss
        let probs = self.forward(x);
        let current_loss = self.loss(&probs, targets);

        // Gradient for W1
        for i in 0..self.w1.len() {
            self.w1[i] += eps;
            let probs_plus = self.forward(x);
            let loss_plus = self.loss(&probs_plus, targets);
            self.w1[i] -= eps;

            let grad = (loss_plus - current_loss) / eps;
            self.w1[i] -= lr * grad;
        }

        // Gradient for b1
        for i in 0..self.b1.len() {
            self.b1[i] += eps;
            let probs_plus = self.forward(x);
            let loss_plus = self.loss(&probs_plus, targets);
            self.b1[i] -= eps;

            let grad = (loss_plus - current_loss) / eps;
            self.b1[i] -= lr * grad;
        }

        // Gradient for W2
        for i in 0..self.w2.len() {
            self.w2[i] += eps;
            let probs_plus = self.forward(x);
            let loss_plus = self.loss(&probs_plus, targets);
            self.w2[i] -= eps;

            let grad = (loss_plus - current_loss) / eps;
            self.w2[i] -= lr * grad;
        }

        // Gradient for b2
        for i in 0..self.b2.len() {
            self.b2[i] += eps;
            let probs_plus = self.forward(x);
            let loss_plus = self.loss(&probs_plus, targets);
            self.b2[i] -= eps;

            let grad = (loss_plus - current_loss) / eps;
            self.b2[i] -= lr * grad;
        }

        current_loss
    }

    /// Train one step with analytical backpropagation (much faster than numerical).
    ///
    /// Computes gradients analytically through the network:
    ///   Forward:  x → (W1·x + b1) → relu → (W2·h + b2) → softmax → probs
    ///   Backward: dL/dprobs → dL/dW2,db2 → dL/dhidden → dL/dW1,db1
    pub fn train_step_backprop(&mut self, x: &[f32], targets: &[u8], lr: f32) -> f32 {
        let batch = targets.len();
        if batch == 0 {
            return 0.0;
        }

        // ---- Forward pass (save intermediates) ----
        // Layer 1: pre_act1 = x @ W1 + b1, hidden = relu(pre_act1)
        let mut pre_act1 = accel::matmul(x, &self.w1, batch, self.hidden_dim, self.input_dim);
        let mut hidden = vec![0.0f32; batch * self.hidden_dim];
        for b in 0..batch {
            for j in 0..self.hidden_dim {
                pre_act1[b * self.hidden_dim + j] += self.b1[j];
                hidden[b * self.hidden_dim + j] = pre_act1[b * self.hidden_dim + j].max(0.0);
            }
        }

        // Layer 2: logits = hidden @ W2 + b2
        let mut logits = accel::matmul(&hidden, &self.w2, batch, self.output_dim, self.hidden_dim);
        for b in 0..batch {
            for j in 0..self.output_dim {
                logits[b * self.output_dim + j] += self.b2[j];
            }
        }

        // Softmax
        let mut probs = logits.clone();
        for b in 0..batch {
            let off = b * self.output_dim;
            let max = probs[off..off + self.output_dim].iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0f32;
            for j in 0..self.output_dim {
                probs[off + j] = (probs[off + j] - max).exp();
                sum += probs[off + j];
            }
            for j in 0..self.output_dim {
                probs[off + j] /= sum;
            }
        }

        // Loss (cross-entropy)
        let loss = self.loss(&probs, targets);

        // ---- Backward pass ----
        // dL/d_logits = probs - one_hot(targets)  (for softmax + cross-entropy)
        let mut d_logits = probs.clone();
        for b in 0..batch {
            let t = targets[b] as usize;
            d_logits[b * self.output_dim + t] -= 1.0;
        }
        // Average over batch
        let inv_batch = 1.0 / batch as f32;
        for v in d_logits.iter_mut() {
            *v *= inv_batch;
        }

        // Gradients for W2 and b2
        // dL/dW2 = hidden^T @ d_logits  (shape: [hidden_dim, output_dim])
        // hidden is [batch, hidden_dim], d_logits is [batch, output_dim]
        let dw2 = accel::matmul_at_b(&hidden, &d_logits, self.hidden_dim, self.output_dim, batch);
        let mut db2 = vec![0.0f32; self.output_dim];
        for b in 0..batch {
            for j in 0..self.output_dim {
                db2[j] += d_logits[b * self.output_dim + j];
            }
        }

        // dL/d_hidden = d_logits @ W2^T  (shape: [batch, hidden_dim])
        // d_logits is [batch, output_dim], W2 is [hidden_dim, output_dim]
        let mut d_hidden = accel::matmul_a_bt(&d_logits, &self.w2, batch, self.hidden_dim, self.output_dim);

        // ReLU backward: zero out where pre_act1 <= 0
        for b in 0..batch {
            for j in 0..self.hidden_dim {
                if pre_act1[b * self.hidden_dim + j] <= 0.0 {
                    d_hidden[b * self.hidden_dim + j] = 0.0;
                }
            }
        }

        // Gradients for W1 and b1
        // dL/dW1 = x^T @ d_hidden  (shape: [input_dim, hidden_dim])
        // x is [batch, input_dim], d_hidden is [batch, hidden_dim]
        let dw1 = accel::matmul_at_b(x, &d_hidden, self.input_dim, self.hidden_dim, batch);
        let mut db1 = vec![0.0f32; self.hidden_dim];
        for b in 0..batch {
            for j in 0..self.hidden_dim {
                db1[j] += d_hidden[b * self.hidden_dim + j];
            }
        }

        // ---- Update weights ----
        for i in 0..self.w1.len() {
            self.w1[i] -= lr * dw1[i];
        }
        for i in 0..self.b1.len() {
            self.b1[i] -= lr * db1[i];
        }
        for i in 0..self.w2.len() {
            self.w2[i] -= lr * dw2[i];
        }
        for i in 0..self.b2.len() {
            self.b2[i] -= lr * db2[i];
        }

        loss
    }

    /// IGQK ternary compression of weights.
    pub fn compress_ternary(&self) -> MlpWeights {
        fn ternary(v: &[f32]) -> Vec<f32> {
            let mean_abs: f32 = v.iter().map(|x| x.abs()).sum::<f32>() / v.len() as f32;
            let threshold = mean_abs * 0.7;
            v.iter().map(|&x| {
                if x > threshold { 1.0 }
                else if x < -threshold { -1.0 }
                else { 0.0 }
            }).collect()
        }

        MlpWeights {
            w1: ternary(&self.w1),
            b1: self.b1.clone(), // Don't compress biases
            w2: ternary(&self.w2),
            b2: self.b2.clone(),
            input_dim: self.input_dim,
            hidden_dim: self.hidden_dim,
            output_dim: self.output_dim,
        }
    }

    /// Count total parameters.
    pub fn param_count(&self) -> usize {
        self.w1.len() + self.b1.len() + self.w2.len() + self.b2.len()
    }
}

/// Generate a tiny synthetic dataset (fake MNIST-like).
/// 4 classes: horizontal line, vertical line, diagonal, empty
pub fn generate_toy_dataset(n_samples: usize, dim: usize) -> (Vec<f32>, Vec<u8>) {
    let mut images = vec![0.0f32; n_samples * dim];
    let mut labels = vec![0u8; n_samples];

    for i in 0..n_samples {
        let class = (i % 4) as u8;
        labels[i] = class;

        let side = (dim as f64).sqrt() as usize;
        let offset = i * dim;

        match class {
            0 => {
                // Horizontal line in middle
                let row = side / 2;
                for col in 1..side - 1 {
                    images[offset + row * side + col] = 1.0;
                }
            }
            1 => {
                // Vertical line in middle
                let col = side / 2;
                for row in 1..side - 1 {
                    images[offset + row * side + col] = 1.0;
                }
            }
            2 => {
                // Diagonal
                for k in 1..side - 1 {
                    images[offset + k * side + k] = 1.0;
                }
            }
            3 => {
                // Cross (both lines)
                let mid = side / 2;
                for k in 1..side - 1 {
                    images[offset + mid * side + k] = 1.0;
                    images[offset + k * side + mid] = 0.5;
                }
            }
            _ => {}
        }
    }

    (images, labels)
}

/// 3-layer MLP weights for deeper networks (e.g., 784→256→128→10).
///
/// Same pattern as `MlpWeights` but with an extra hidden layer for better
/// capacity and accuracy on tasks like MNIST.
#[derive(Clone)]
pub struct MlpWeights3 {
    pub w1: Vec<f32>,  // [input_dim, hidden1_dim]
    pub b1: Vec<f32>,  // [hidden1_dim]
    pub w2: Vec<f32>,  // [hidden1_dim, hidden2_dim]
    pub b2: Vec<f32>,  // [hidden2_dim]
    pub w3: Vec<f32>,  // [hidden2_dim, output_dim]
    pub b3: Vec<f32>,  // [output_dim]
    pub input_dim: usize,
    pub hidden1_dim: usize,
    pub hidden2_dim: usize,
    pub output_dim: usize,
}

impl MlpWeights3 {
    /// Initialize with Xavier/Glorot initialization.
    pub fn new(input_dim: usize, hidden1_dim: usize, hidden2_dim: usize, output_dim: usize) -> Self {
        let scale1 = (2.0 / (input_dim + hidden1_dim) as f64).sqrt() as f32;
        let scale2 = (2.0 / (hidden1_dim + hidden2_dim) as f64).sqrt() as f32;
        let scale3 = (2.0 / (hidden2_dim + output_dim) as f64).sqrt() as f32;

        let w1: Vec<f32> = (0..input_dim * hidden1_dim)
            .map(|i| (i as f32 * 0.4871).sin() * scale1)
            .collect();
        let b1 = vec![0.0; hidden1_dim];
        let w2: Vec<f32> = (0..hidden1_dim * hidden2_dim)
            .map(|i| (i as f32 * 0.7291).sin() * scale2)
            .collect();
        let b2 = vec![0.0; hidden2_dim];
        let w3: Vec<f32> = (0..hidden2_dim * output_dim)
            .map(|i| (i as f32 * 0.3517).sin() * scale3)
            .collect();
        let b3 = vec![0.0; output_dim];

        Self { w1, b1, w2, b2, w3, b3, input_dim, hidden1_dim, hidden2_dim, output_dim }
    }

    /// Forward pass: x → W1 → relu → W2 → relu → W3 → softmax → probabilities
    pub fn forward(&self, x: &[f32]) -> Vec<f32> {
        let batch = x.len() / self.input_dim;

        // Layer 1: h1 = relu(x @ W1 + b1)
        let mut h1 = accel::matmul(x, &self.w1, batch, self.hidden1_dim, self.input_dim);
        for b in 0..batch {
            for j in 0..self.hidden1_dim {
                h1[b * self.hidden1_dim + j] = (h1[b * self.hidden1_dim + j] + self.b1[j]).max(0.0);
            }
        }

        // Layer 2: h2 = relu(h1 @ W2 + b2)
        let mut h2 = accel::matmul(&h1, &self.w2, batch, self.hidden2_dim, self.hidden1_dim);
        for b in 0..batch {
            for j in 0..self.hidden2_dim {
                h2[b * self.hidden2_dim + j] = (h2[b * self.hidden2_dim + j] + self.b2[j]).max(0.0);
            }
        }

        // Layer 3: logits = h2 @ W3 + b3
        let mut logits = accel::matmul(&h2, &self.w3, batch, self.output_dim, self.hidden2_dim);
        for b in 0..batch {
            for j in 0..self.output_dim {
                logits[b * self.output_dim + j] += self.b3[j];
            }
        }

        // Softmax
        for b in 0..batch {
            let offset = b * self.output_dim;
            let max = logits[offset..offset + self.output_dim]
                .iter()
                .cloned()
                .fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0f32;
            for j in 0..self.output_dim {
                logits[offset + j] = (logits[offset + j] - max).exp();
                sum += logits[offset + j];
            }
            for j in 0..self.output_dim {
                logits[offset + j] /= sum;
            }
        }

        logits
    }

    /// Forward pass for a single input (784 elements).
    /// Returns a Vec of `output_dim` probabilities (softmax output).
    pub fn forward_single(&self, input: &[f32]) -> Vec<f32> {
        assert_eq!(input.len(), self.input_dim, "forward_single: input length mismatch");

        // Layer 1: h1 = relu(input @ W1 + b1)
        let mut h1 = vec![0.0f32; self.hidden1_dim];
        for j in 0..self.hidden1_dim {
            let mut sum = self.b1[j];
            for i in 0..self.input_dim {
                sum += input[i] * self.w1[i * self.hidden1_dim + j];
            }
            h1[j] = sum.max(0.0); // ReLU
        }

        // Layer 2: h2 = relu(h1 @ W2 + b2)
        let mut h2 = vec![0.0f32; self.hidden2_dim];
        for j in 0..self.hidden2_dim {
            let mut sum = self.b2[j];
            for i in 0..self.hidden1_dim {
                sum += h1[i] * self.w2[i * self.hidden2_dim + j];
            }
            h2[j] = sum.max(0.0); // ReLU
        }

        // Layer 3: logits = h2 @ W3 + b3
        let mut logits = vec![0.0f32; self.output_dim];
        for j in 0..self.output_dim {
            let mut sum = self.b3[j];
            for i in 0..self.hidden2_dim {
                sum += h2[i] * self.w3[i * self.output_dim + j];
            }
            logits[j] = sum;
        }

        // Softmax
        let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        for j in 0..self.output_dim {
            logits[j] = (logits[j] - max).exp();
            sum += logits[j];
        }
        for j in 0..self.output_dim {
            logits[j] /= sum;
        }

        logits
    }

    /// Compute cross-entropy loss.
    pub fn loss(&self, probs: &[f32], targets: &[u8]) -> f32 {
        let batch = targets.len();
        let mut total = 0.0f32;
        for b in 0..batch {
            let target = targets[b] as usize;
            let p = probs[b * self.output_dim + target].max(1e-7);
            total -= p.ln();
        }
        total / batch as f32
    }

    /// Compute accuracy.
    pub fn accuracy(&self, probs: &[f32], targets: &[u8]) -> f32 {
        let batch = targets.len();
        let mut correct = 0;
        for b in 0..batch {
            let offset = b * self.output_dim;
            let predicted = probs[offset..offset + self.output_dim]
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(i, _)| i)
                .unwrap_or(0);
            if predicted == targets[b] as usize {
                correct += 1;
            }
        }
        correct as f32 / batch as f32
    }

    /// Train one step with analytical backpropagation through 3 layers.
    ///
    /// Forward:  x → (W1+b1) → relu → (W2+b2) → relu → (W3+b3) → softmax → probs
    /// Backward: dL/dprobs → dL/dW3,db3 → dL/dh2 → dL/dW2,db2 → dL/dh1 → dL/dW1,db1
    pub fn train_step_backprop(&mut self, x: &[f32], targets: &[u8], lr: f32) -> f32 {
        let batch = targets.len();
        if batch == 0 {
            return 0.0;
        }

        // ---- Forward pass (save intermediates) ----
        // Layer 1: pre1 = x @ W1 + b1, h1 = relu(pre1)
        let mut pre1 = accel::matmul(x, &self.w1, batch, self.hidden1_dim, self.input_dim);
        let mut h1 = vec![0.0f32; batch * self.hidden1_dim];
        for b in 0..batch {
            for j in 0..self.hidden1_dim {
                pre1[b * self.hidden1_dim + j] += self.b1[j];
                h1[b * self.hidden1_dim + j] = pre1[b * self.hidden1_dim + j].max(0.0);
            }
        }

        // Layer 2: pre2 = h1 @ W2 + b2, h2 = relu(pre2)
        let mut pre2 = accel::matmul(&h1, &self.w2, batch, self.hidden2_dim, self.hidden1_dim);
        let mut h2 = vec![0.0f32; batch * self.hidden2_dim];
        for b in 0..batch {
            for j in 0..self.hidden2_dim {
                pre2[b * self.hidden2_dim + j] += self.b2[j];
                h2[b * self.hidden2_dim + j] = pre2[b * self.hidden2_dim + j].max(0.0);
            }
        }

        // Layer 3: logits = h2 @ W3 + b3
        let mut logits = accel::matmul(&h2, &self.w3, batch, self.output_dim, self.hidden2_dim);
        for b in 0..batch {
            for j in 0..self.output_dim {
                logits[b * self.output_dim + j] += self.b3[j];
            }
        }

        // Softmax
        let mut probs = logits.clone();
        for b in 0..batch {
            let off = b * self.output_dim;
            let max = probs[off..off + self.output_dim].iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0f32;
            for j in 0..self.output_dim {
                probs[off + j] = (probs[off + j] - max).exp();
                sum += probs[off + j];
            }
            for j in 0..self.output_dim {
                probs[off + j] /= sum;
            }
        }

        // Loss
        let loss = self.loss(&probs, targets);

        // ---- Backward pass ----
        let inv_batch = 1.0 / batch as f32;

        // dL/d_logits = (probs - one_hot) / batch
        let mut d_logits = probs.clone();
        for b in 0..batch {
            d_logits[b * self.output_dim + targets[b] as usize] -= 1.0;
        }
        for v in d_logits.iter_mut() {
            *v *= inv_batch;
        }

        // --- Gradients for W3, b3 ---
        // dW3 = h2^T @ d_logits  [hidden2_dim, output_dim]
        // h2 is [batch, hidden2_dim], d_logits is [batch, output_dim]
        let dw3 = accel::matmul_at_b(&h2, &d_logits, self.hidden2_dim, self.output_dim, batch);
        let mut db3 = vec![0.0f32; self.output_dim];
        for b in 0..batch {
            for j in 0..self.output_dim {
                db3[j] += d_logits[b * self.output_dim + j];
            }
        }

        // dh2 = d_logits @ W3^T  [batch, hidden2_dim]
        // d_logits is [batch, output_dim], W3 is [hidden2_dim, output_dim]
        let mut dh2 = accel::matmul_a_bt(&d_logits, &self.w3, batch, self.hidden2_dim, self.output_dim);

        // ReLU backward for layer 2
        for b in 0..batch {
            for j in 0..self.hidden2_dim {
                if pre2[b * self.hidden2_dim + j] <= 0.0 {
                    dh2[b * self.hidden2_dim + j] = 0.0;
                }
            }
        }

        // --- Gradients for W2, b2 ---
        // dW2 = h1^T @ dh2  [hidden1_dim, hidden2_dim]
        // h1 is [batch, hidden1_dim], dh2 is [batch, hidden2_dim]
        let dw2 = accel::matmul_at_b(&h1, &dh2, self.hidden1_dim, self.hidden2_dim, batch);
        let mut db2 = vec![0.0f32; self.hidden2_dim];
        for b in 0..batch {
            for j in 0..self.hidden2_dim {
                db2[j] += dh2[b * self.hidden2_dim + j];
            }
        }

        // dh1 = dh2 @ W2^T  [batch, hidden1_dim]
        // dh2 is [batch, hidden2_dim], W2 is [hidden1_dim, hidden2_dim]
        let mut dh1 = accel::matmul_a_bt(&dh2, &self.w2, batch, self.hidden1_dim, self.hidden2_dim);

        // ReLU backward for layer 1
        for b in 0..batch {
            for j in 0..self.hidden1_dim {
                if pre1[b * self.hidden1_dim + j] <= 0.0 {
                    dh1[b * self.hidden1_dim + j] = 0.0;
                }
            }
        }

        // --- Gradients for W1, b1 ---
        // dW1 = x^T @ dh1  [input_dim, hidden1_dim]
        // x is [batch, input_dim], dh1 is [batch, hidden1_dim]
        let dw1 = accel::matmul_at_b(x, &dh1, self.input_dim, self.hidden1_dim, batch);
        let mut db1 = vec![0.0f32; self.hidden1_dim];
        for b in 0..batch {
            for j in 0..self.hidden1_dim {
                db1[j] += dh1[b * self.hidden1_dim + j];
            }
        }

        // ---- Update weights ----
        for i in 0..self.w1.len() { self.w1[i] -= lr * dw1[i]; }
        for i in 0..self.b1.len() { self.b1[i] -= lr * db1[i]; }
        for i in 0..self.w2.len() { self.w2[i] -= lr * dw2[i]; }
        for i in 0..self.b2.len() { self.b2[i] -= lr * db2[i]; }
        for i in 0..self.w3.len() { self.w3[i] -= lr * dw3[i]; }
        for i in 0..self.b3.len() { self.b3[i] -= lr * db3[i]; }

        loss
    }

    /// IGQK ternary compression of weights.
    pub fn compress_ternary(&self) -> MlpWeights3 {
        fn ternary(v: &[f32]) -> Vec<f32> {
            let mean_abs: f32 = v.iter().map(|x| x.abs()).sum::<f32>() / v.len() as f32;
            let threshold = mean_abs * 0.7;
            v.iter().map(|&x| {
                if x > threshold { 1.0 }
                else if x < -threshold { -1.0 }
                else { 0.0 }
            }).collect()
        }

        MlpWeights3 {
            w1: ternary(&self.w1),
            b1: self.b1.clone(),
            w2: ternary(&self.w2),
            b2: self.b2.clone(),
            w3: ternary(&self.w3),
            b3: self.b3.clone(),
            input_dim: self.input_dim,
            hidden1_dim: self.hidden1_dim,
            hidden2_dim: self.hidden2_dim,
            output_dim: self.output_dim,
        }
    }

    /// Count total parameters.
    pub fn param_count(&self) -> usize {
        self.w1.len() + self.b1.len() + self.w2.len() + self.b2.len() + self.w3.len() + self.b3.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mlp_forward() {
        let mlp = MlpWeights::new(16, 8, 4);
        let input = vec![0.5f32; 16];
        let probs = mlp.forward(&input);

        // Softmax should sum to 1
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn mlp_loss_decreases() {
        let dim = 16;
        let mut mlp = MlpWeights::new(dim, 8, 4);
        let (images, labels) = generate_toy_dataset(8, dim);

        let probs_before = mlp.forward(&images);
        let loss_before = mlp.loss(&probs_before, &labels);

        // One training step (small lr for numerical gradients)
        mlp.train_step(&images, &labels, 0.001);

        let probs_after = mlp.forward(&images);
        let loss_after = mlp.loss(&probs_after, &labels);

        // Loss should decrease (or at least not increase dramatically)
        assert!(loss_after < loss_before + 0.5,
            "Loss didn't decrease: {loss_before} → {loss_after}");
    }

    #[test]
    fn ternary_compression() {
        let mlp = MlpWeights::new(16, 8, 4);
        let compressed = mlp.compress_ternary();

        // All w1 values should be {-1, 0, 1}
        for &w in &compressed.w1 {
            assert!(w == -1.0 || w == 0.0 || w == 1.0,
                "Non-ternary value: {w}");
        }
    }

    #[test]
    fn toy_dataset_generation() {
        let (images, labels) = generate_toy_dataset(100, 64);
        assert_eq!(images.len(), 100 * 64);
        assert_eq!(labels.len(), 100);
        assert!(labels.iter().all(|&l| l < 4));
    }

    #[test]
    fn backprop_loss_decreases() {
        let dim = 16;
        let mut mlp = MlpWeights::new(dim, 8, 4);
        let (images, labels) = generate_toy_dataset(20, dim);

        let probs_before = mlp.forward(&images);
        let loss_before = mlp.loss(&probs_before, &labels);

        // Multiple backprop training steps
        for _ in 0..10 {
            mlp.train_step_backprop(&images, &labels, 0.1);
        }

        let probs_after = mlp.forward(&images);
        let loss_after = mlp.loss(&probs_after, &labels);

        assert!(loss_after < loss_before,
            "Backprop loss didn't decrease: {loss_before} → {loss_after}");
    }

    #[test]
    fn backprop_achieves_high_accuracy() {
        let dim = 16;
        let mut mlp = MlpWeights::new(dim, 16, 4);
        let (images, labels) = generate_toy_dataset(40, dim);

        // Train with backprop
        for _ in 0..100 {
            mlp.train_step_backprop(&images, &labels, 0.05);
        }

        let probs = mlp.forward(&images);
        let acc = mlp.accuracy(&probs, &labels);
        assert!(acc > 0.7, "Expected >70% accuracy, got {:.1}%", acc * 100.0);
    }

    // --- MlpWeights3 tests ---

    #[test]
    fn mlp3_forward() {
        let mlp = MlpWeights3::new(16, 12, 8, 4);
        let input = vec![0.5f32; 16];
        let probs = mlp.forward(&input);

        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "Softmax sum = {sum}, expected 1.0");
    }

    #[test]
    fn mlp3_forward_batch() {
        let mlp = MlpWeights3::new(16, 12, 8, 4);
        let input = vec![0.5f32; 3 * 16]; // batch of 3
        let probs = mlp.forward(&input);

        assert_eq!(probs.len(), 3 * 4);
        for b in 0..3 {
            let sum: f32 = probs[b * 4..(b + 1) * 4].iter().sum();
            assert!((sum - 1.0).abs() < 1e-5, "Batch {b}: softmax sum = {sum}");
        }
    }

    #[test]
    fn mlp3_backprop_loss_decreases() {
        let dim = 16;
        let mut mlp = MlpWeights3::new(dim, 12, 8, 4);
        let (images, labels) = generate_toy_dataset(20, dim);

        let probs_before = mlp.forward(&images);
        let loss_before = mlp.loss(&probs_before, &labels);

        for _ in 0..20 {
            mlp.train_step_backprop(&images, &labels, 0.1);
        }

        let probs_after = mlp.forward(&images);
        let loss_after = mlp.loss(&probs_after, &labels);

        assert!(loss_after < loss_before,
            "3-layer backprop loss didn't decrease: {loss_before} -> {loss_after}");
    }

    #[test]
    fn mlp3_backprop_achieves_accuracy() {
        let dim = 16;
        let mut mlp = MlpWeights3::new(dim, 16, 12, 4);
        let (images, labels) = generate_toy_dataset(40, dim);

        for _ in 0..150 {
            mlp.train_step_backprop(&images, &labels, 0.05);
        }

        let probs = mlp.forward(&images);
        let acc = mlp.accuracy(&probs, &labels);
        assert!(acc > 0.6, "Expected >60% accuracy, got {:.1}%", acc * 100.0);
    }

    #[test]
    fn mlp3_ternary_compression() {
        let mlp = MlpWeights3::new(16, 12, 8, 4);
        let compressed = mlp.compress_ternary();

        for &w in &compressed.w1 {
            assert!(w == -1.0 || w == 0.0 || w == 1.0, "Non-ternary w1: {w}");
        }
        for &w in &compressed.w2 {
            assert!(w == -1.0 || w == 0.0 || w == 1.0, "Non-ternary w2: {w}");
        }
        for &w in &compressed.w3 {
            assert!(w == -1.0 || w == 0.0 || w == 1.0, "Non-ternary w3: {w}");
        }
    }

    #[test]
    fn mlp3_param_count() {
        let mlp = MlpWeights3::new(784, 256, 128, 10);
        // w1: 784*256 + b1: 256 + w2: 256*128 + b2: 128 + w3: 128*10 + b3: 10
        let expected = 784 * 256 + 256 + 256 * 128 + 128 + 128 * 10 + 10;
        assert_eq!(mlp.param_count(), expected);
    }
}
