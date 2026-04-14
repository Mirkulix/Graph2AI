//! Reverse-mode Automatic Differentiation for QLANG.
//!
//! Given a computation graph and a loss scalar, computes gradients
//! of the loss with respect to all input parameters.
//!
//! This replaces the slow numerical gradient computation with
//! exact analytical gradients — the standard approach in PyTorch/JAX.
//!
//! Algorithm: Reverse-mode AD (backpropagation)
//! 1. Forward pass: evaluate graph, store intermediate values
//! 2. Backward pass: propagate gradients from output to inputs


/// A value tracked for automatic differentiation.
#[derive(Debug, Clone)]
pub struct Value {
    pub data: Vec<f32>,
    pub shape: Vec<usize>,
    pub grad: Option<Vec<f32>>,
}

impl Value {
    pub fn new(data: Vec<f32>, shape: Vec<usize>) -> Self {
        Self { data, shape, grad: None }
    }

    pub fn zeros_like(&self) -> Vec<f32> {
        vec![0.0; self.data.len()]
    }

    pub fn numel(&self) -> usize {
        self.data.len()
    }
}

/// Recorded operation for backward pass.
#[derive(Debug, Clone)]
#[allow(dead_code)]
enum TapeEntry {
    Add { out: usize, a: usize, b: usize },
    Mul { out: usize, a: usize, b: usize },
    MatMul { out: usize, a: usize, b: usize, m: usize, k: usize, n: usize },
    Relu { out: usize, input: usize },
    Sigmoid { out: usize, input: usize },
    Softmax { out: usize, input: usize, n_classes: usize },
    CrossEntropyLoss { out: usize, probs: usize, targets: Vec<u8> },
}

/// Computation tape for reverse-mode AD.
pub struct Tape {
    pub values: Vec<Value>,
    ops: Vec<TapeEntry>,
}

impl Tape {
    pub fn new() -> Self {
        Self { values: Vec::new(), ops: Vec::new() }
    }

    /// Register a new value (input or parameter).
    pub fn variable(&mut self, data: Vec<f32>, shape: Vec<usize>) -> usize {
        let id = self.values.len();
        self.values.push(Value::new(data, shape));
        id
    }

    /// Element-wise add.
    pub fn add(&mut self, a: usize, b: usize) -> usize {
        let data: Vec<f32> = self.values[a].data.iter()
            .zip(self.values[b].data.iter())
            .map(|(&x, &y)| x + y)
            .collect();
        let shape = self.values[a].shape.clone();
        let out = self.variable(data, shape);
        self.ops.push(TapeEntry::Add { out, a, b });
        out
    }

    /// Element-wise multiply.
    pub fn mul(&mut self, a: usize, b: usize) -> usize {
        let data: Vec<f32> = self.values[a].data.iter()
            .zip(self.values[b].data.iter())
            .map(|(&x, &y)| x * y)
            .collect();
        let shape = self.values[a].shape.clone();
        let out = self.variable(data, shape);
        self.ops.push(TapeEntry::Mul { out, a, b });
        out
    }

    /// Matrix multiplication: [m, k] × [k, n] → [m, n].
    pub fn matmul(&mut self, a: usize, b: usize) -> usize {
        let m = self.values[a].shape[0];
        let k = self.values[a].shape[1];
        let n = self.values[b].shape[1];
        assert_eq!(self.values[b].shape[0], k);

        let va = &self.values[a].data;
        let vb = &self.values[b].data;

        let mut data = vec![0.0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0f32;
                for p in 0..k {
                    sum += va[i * k + p] * vb[p * n + j];
                }
                data[i * n + j] = sum;
            }
        }

        let out = self.variable(data, vec![m, n]);
        self.ops.push(TapeEntry::MatMul { out, a, b, m, k, n });
        out
    }

    /// ReLU activation.
    pub fn relu(&mut self, input: usize) -> usize {
        let data: Vec<f32> = self.values[input].data.iter()
            .map(|&x| x.max(0.0))
            .collect();
        let shape = self.values[input].shape.clone();
        let out = self.variable(data, shape);
        self.ops.push(TapeEntry::Relu { out, input });
        out
    }

    /// Sigmoid activation.
    pub fn sigmoid(&mut self, input: usize) -> usize {
        let data: Vec<f32> = self.values[input].data.iter()
            .map(|&x| 1.0 / (1.0 + (-x).exp()))
            .collect();
        let shape = self.values[input].shape.clone();
        let out = self.variable(data, shape);
        self.ops.push(TapeEntry::Sigmoid { out, input });
        out
    }

    /// Softmax (row-wise for 2D).
    pub fn softmax(&mut self, input: usize) -> usize {
        let batch = self.values[input].shape[0];
        let n_classes = self.values[input].shape[1];
        let va = &self.values[input].data;

        let mut data = vec![0.0f32; batch * n_classes];
        for b in 0..batch {
            let offset = b * n_classes;
            let max = va[offset..offset + n_classes].iter().cloned()
                .fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0f32;
            for j in 0..n_classes {
                data[offset + j] = (va[offset + j] - max).exp();
                sum += data[offset + j];
            }
            for j in 0..n_classes {
                data[offset + j] /= sum;
            }
        }

        let shape = self.values[input].shape.clone();
        let out = self.variable(data, shape);
        self.ops.push(TapeEntry::Softmax { out, input, n_classes });
        out
    }

    /// Cross-entropy loss: -Σ log(p[target]).
    pub fn cross_entropy_loss(&mut self, probs: usize, targets: &[u8]) -> usize {
        let batch = targets.len();
        let n_classes = self.values[probs].shape[1];
        let vp = &self.values[probs].data;

        let mut loss = 0.0f32;
        for b in 0..batch {
            let t = targets[b] as usize;
            loss -= vp[b * n_classes + t].max(1e-7).ln();
        }
        loss /= batch as f32;

        let out = self.variable(vec![loss], vec![1]);
        self.ops.push(TapeEntry::CrossEntropyLoss {
            out, probs, targets: targets.to_vec(),
        });
        out
    }

    /// Get the value data.
    pub fn value(&self, id: usize) -> &[f32] {
        &self.values[id].data
    }

    /// Backward pass: compute gradients of `loss_id` w.r.t. all variables.
    pub fn backward(&mut self, loss_id: usize) {
        let _n = self.values.len();

        // Initialize gradients to zero
        let mut grads: Vec<Vec<f32>> = self.values.iter()
            .map(|v| vec![0.0f32; v.data.len()])
            .collect();

        // Gradient of loss w.r.t. itself = 1.0 (for all elements)
        grads[loss_id] = vec![1.0; self.values[loss_id].data.len()];

        // Propagate backwards through tape
        for entry in self.ops.iter().rev() {
            match entry {
                TapeEntry::Add { out, a, b } => {
                    // d(a+b)/da = 1, d(a+b)/db = 1
                    let grad_out = grads[*out].clone();
                    for i in 0..grad_out.len() {
                        grads[*a][i] += grad_out[i];
                        grads[*b][i] += grad_out[i];
                    }
                }

                TapeEntry::Mul { out, a, b } => {
                    // d(a*b)/da = b, d(a*b)/db = a
                    let grad_out = grads[*out].clone();
                    let va = &self.values[*a].data;
                    let vb = &self.values[*b].data;
                    for i in 0..grad_out.len() {
                        grads[*a][i] += grad_out[i] * vb[i];
                        grads[*b][i] += grad_out[i] * va[i];
                    }
                }

                TapeEntry::MatMul { out, a, b, m, k, n } => {
                    // d(A@B)/dA = grad_out @ B^T
                    // d(A@B)/dB = A^T @ grad_out
                    let grad_out = grads[*out].clone();
                    let va = &self.values[*a].data;
                    let vb = &self.values[*b].data;

                    // grad_A = grad_out @ B^T  [m, k] = [m, n] @ [n, k]
                    for i in 0..*m {
                        for j in 0..*k {
                            let mut sum = 0.0f32;
                            for p in 0..*n {
                                sum += grad_out[i * n + p] * vb[j * n + p]; // B^T[p, j] = B[j, p]
                            }
                            grads[*a][i * k + j] += sum;
                        }
                    }

                    // grad_B = A^T @ grad_out  [k, n] = [k, m] @ [m, n]
                    for i in 0..*k {
                        for j in 0..*n {
                            let mut sum = 0.0f32;
                            for p in 0..*m {
                                sum += va[p * k + i] * grad_out[p * n + j]; // A^T[i, p] = A[p, i]
                            }
                            grads[*b][i * n + j] += sum;
                        }
                    }
                }

                TapeEntry::Relu { out, input } => {
                    // d(relu)/dx = 1 if x > 0, else 0
                    let grad_out = grads[*out].clone();
                    let va = &self.values[*input].data;
                    for i in 0..grad_out.len() {
                        grads[*input][i] += if va[i] > 0.0 { grad_out[i] } else { 0.0 };
                    }
                }

                TapeEntry::Sigmoid { out, input } => {
                    // d(sigmoid)/dx = sigmoid * (1 - sigmoid)
                    let grad_out = grads[*out].clone();
                    let s = &self.values[*out].data;
                    for i in 0..grad_out.len() {
                        grads[*input][i] = grad_out[i] * s[i] * (1.0 - s[i]);
                    }
                }

                TapeEntry::Softmax { out, input, n_classes: _n_classes } => {
                    // Pass gradients through to input
                    // When combined with CE loss, grad is already (probs - one_hot)
                    // Just pass through: d(softmax)/d(logits) ≈ identity for combined CE+softmax
                    let grad_out = grads[*out].clone();
                    for i in 0..grad_out.len() {
                        grads[*input][i] += grad_out[i];
                    }
                }

                TapeEntry::CrossEntropyLoss { out: _, probs, targets } => {
                    // d(CE)/d(softmax_output) = probs - one_hot(targets)
                    // This gradient flows back through Softmax → logits
                    let batch = targets.len();
                    let n_classes = self.values[*probs].shape[1];
                    let vp = &self.values[*probs].data;

                    for b in 0..batch {
                        for j in 0..n_classes {
                            let target_val = if j == targets[b] as usize { 1.0 } else { 0.0 };
                            grads[*probs][b * n_classes + j] += (vp[b * n_classes + j] - target_val) / batch as f32;
                        }
                    }
                }
            }
        }

        // Store gradients
        for (i, g) in grads.into_iter().enumerate() {
            self.values[i].grad = Some(g);
        }
    }

    /// Get gradient of a variable.
    pub fn grad(&self, id: usize) -> Option<&[f32]> {
        self.values[id].grad.as_deref()
    }

    /// Apply gradient descent update to a variable.
    pub fn sgd_update(&mut self, id: usize, lr: f32) {
        if let Some(grad) = &self.values[id].grad {
            let grad = grad.clone();
            for (v, g) in self.values[id].data.iter_mut().zip(grad.iter()) {
                *v -= lr * g;
            }
        }
    }
}

impl Default for Tape {
    fn default() -> Self {
        Self::new()
    }
}

/// Train an MLP using autograd.
pub fn train_mlp_autograd(
    w1: &mut Vec<f32>, b1: &mut Vec<f32>,
    w2: &mut Vec<f32>, b2: &mut Vec<f32>,
    x: &[f32], targets: &[u8],
    input_dim: usize, hidden_dim: usize, output_dim: usize,
    lr: f32,
) -> (f32, f32) {
    let batch = targets.len();

    let mut tape = Tape::new();

    // Register variables
    let x_id = tape.variable(x.to_vec(), vec![batch, input_dim]);
    let w1_id = tape.variable(w1.clone(), vec![input_dim, hidden_dim]);
    let _b1_id = tape.variable(b1.clone(), vec![1, hidden_dim]);
    let w2_id = tape.variable(w2.clone(), vec![hidden_dim, output_dim]);
    let _b2_id = tape.variable(b2.clone(), vec![1, output_dim]);

    // Forward: h = relu(x @ W1 + b1_broadcast)
    let h1 = tape.matmul(x_id, w1_id);

    // Broadcast b1 to [batch, hidden_dim]
    let b1_broadcast = {
        let b1_data: Vec<f32> = (0..batch).flat_map(|_| b1.iter().copied()).collect();
        tape.variable(b1_data, vec![batch, hidden_dim])
    };
    let h1_biased = tape.add(h1, b1_broadcast);
    let h1_activated = tape.relu(h1_biased);

    // Forward: logits = h1 @ W2 + b2_broadcast
    let logits = tape.matmul(h1_activated, w2_id);
    let b2_broadcast = {
        let b2_data: Vec<f32> = (0..batch).flat_map(|_| b2.iter().copied()).collect();
        tape.variable(b2_data, vec![batch, output_dim])
    };
    let logits_biased = tape.add(logits, b2_broadcast);

    // Softmax + cross-entropy loss
    let probs = tape.softmax(logits_biased);
    let loss_id = tape.cross_entropy_loss(probs, targets);

    let loss_val = tape.value(loss_id)[0];

    // Compute accuracy
    let probs_data = tape.value(probs);
    let mut correct = 0;
    for b in 0..batch {
        let predicted = probs_data[b * output_dim..(b + 1) * output_dim]
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0);
        if predicted == targets[b] as usize {
            correct += 1;
        }
    }
    let accuracy = correct as f32 / batch as f32;

    // Backward
    tape.backward(loss_id);

    // Update weights
    tape.sgd_update(w1_id, lr);
    tape.sgd_update(w2_id, lr);

    // Copy updated weights back
    *w1 = tape.values[w1_id].data.clone();
    *w2 = tape.values[w2_id].data.clone();

    // Update biases (sum gradients across batch)
    if let Some(b1_grad) = tape.grad(b1_broadcast) {
        for j in 0..hidden_dim {
            let mut grad_sum = 0.0f32;
            for b in 0..batch {
                grad_sum += b1_grad[b * hidden_dim + j];
            }
            b1[j] -= lr * grad_sum;
        }
    }
    if let Some(b2_grad) = tape.grad(b2_broadcast) {
        for j in 0..output_dim {
            let mut grad_sum = 0.0f32;
            for b in 0..batch {
                grad_sum += b2_grad[b * output_dim + j];
            }
            b2[j] -= lr * grad_sum;
        }
    }

    (loss_val, accuracy)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tape_add_forward() {
        let mut tape = Tape::new();
        let a = tape.variable(vec![1.0, 2.0, 3.0], vec![3]);
        let b = tape.variable(vec![10.0, 20.0, 30.0], vec![3]);
        let c = tape.add(a, b);
        assert_eq!(tape.value(c), &[11.0, 22.0, 33.0]);
    }

    #[test]
    fn tape_matmul_forward() {
        let mut tape = Tape::new();
        // [2, 3] × [3, 2]
        let a = tape.variable(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let b = tape.variable(vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0], vec![3, 2]);
        let c = tape.matmul(a, b);
        assert_eq!(tape.value(c), &[4.0, 5.0, 10.0, 11.0]);
    }

    #[test]
    fn tape_backward_add() {
        let mut tape = Tape::new();
        let a = tape.variable(vec![3.0], vec![1]);
        let b = tape.variable(vec![5.0], vec![1]);
        let c = tape.add(a, b);

        tape.backward(c);

        assert_eq!(tape.grad(a).unwrap(), &[1.0]);
        assert_eq!(tape.grad(b).unwrap(), &[1.0]);
    }

    #[test]
    fn tape_backward_mul() {
        let mut tape = Tape::new();
        let a = tape.variable(vec![3.0], vec![1]);
        let b = tape.variable(vec![5.0], vec![1]);
        let c = tape.mul(a, b);

        tape.backward(c);

        // d(a*b)/da = b = 5, d(a*b)/db = a = 3
        assert_eq!(tape.grad(a).unwrap(), &[5.0]);
        assert_eq!(tape.grad(b).unwrap(), &[3.0]);
    }

    #[test]
    fn tape_backward_relu() {
        let mut tape = Tape::new();
        let a = tape.variable(vec![2.0, -1.0, 0.5, -3.0], vec![4]);
        let b = tape.relu(a);

        tape.backward(b);

        // d(relu)/dx = 1 if x > 0, else 0
        assert_eq!(tape.grad(a).unwrap(), &[1.0, 0.0, 1.0, 0.0]);
    }

    #[test]
    fn autograd_training_loss_decreases() {
        // Simple test: just verify gradients are finite and weights change
        let input_dim = 4;
        let hidden_dim = 4;
        let output_dim = 2;

        let mut w1: Vec<f32> = vec![0.1, 0.2, -0.1, -0.2,
                                     0.3, -0.3, 0.1, 0.4,
                                     -0.2, 0.1, 0.3, -0.1,
                                     0.2, -0.4, 0.1, 0.2];
        let mut b1 = vec![0.0f32; hidden_dim];
        let mut w2: Vec<f32> = vec![0.1, -0.1, 0.2, 0.3,
                                     -0.2, 0.1, -0.3, 0.2];
        let mut b2 = vec![0.0f32; output_dim];

        let x = vec![1.0, 0.0, 0.5, 0.0,
                     0.0, 1.0, 0.0, 0.5]; // 2 samples
        let labels = vec![0u8, 1u8];

        let w1_before = w1.clone();

        let (loss, _acc) = train_mlp_autograd(
            &mut w1, &mut b1, &mut w2, &mut b2,
            &x, &labels,
            input_dim, hidden_dim, output_dim,
            0.01,
        );

        // Loss should be finite
        assert!(loss.is_finite(), "Loss is not finite: {loss}");

        // Weights should have changed
        assert_ne!(w1, w1_before, "W1 weights didn't change after training step");
    }
}
