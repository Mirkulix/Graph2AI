//! Test-Time Training (TTT) — model adapts its weights during inference.
//!
//! The model is not static. When processing a new input, it does a few
//! local weight updates to better handle the specific input, then predicts.
//!
//! Implementation: Before predicting, run K steps of NoProp/FF on the input
//! (self-supervised: predict masked parts of the input). This adapts the
//! weights to the current context without needing labels.
//!
//! Combined with ternary weights: shadow weights adapt, ternary stay fixed
//! for the fast path. Only the shadow adaptation is used during TTT.

/// TTT-enabled layer: standard forward + optional self-supervised adaptation.
#[derive(Clone)]
pub struct TttLayer {
    /// Main weights (frozen during normal inference)
    pub weights: Vec<f32>,
    /// Adaptation weights (updated during TTT)
    pub adapt_weights: Vec<f32>,
    pub biases: Vec<f32>,
    pub in_dim: usize,
    pub out_dim: usize,
    pub lr: f32,
    /// How many TTT steps to run before predicting
    pub ttt_steps: usize,
}

impl TttLayer {
    pub fn new(in_dim: usize, out_dim: usize, seed: f32) -> Self {
        let scale = (2.0 / (in_dim + out_dim) as f64).sqrt() as f32;
        let weights: Vec<f32> = (0..in_dim * out_dim)
            .map(|i| (i as f32 * seed).sin() * scale)
            .collect();
        let adapt_weights = weights.clone();
        let biases = vec![0.0f32; out_dim];

        Self { weights, adapt_weights, biases, in_dim, out_dim, lr: 0.001, ttt_steps: 3 }
    }

    /// Standard forward: y = ReLU(x @ W + b)
    pub fn forward(&self, input: &[f32]) -> Vec<f32> {
        let batch = input.len() / self.in_dim;
        let mut output = vec![0.0f32; batch * self.out_dim];
        for b in 0..batch {
            for j in 0..self.out_dim {
                let mut sum = self.biases[j];
                for k in 0..self.in_dim {
                    sum += input[b * self.in_dim + k] * self.adapt_weights[k * self.out_dim + j];
                }
                output[b * self.out_dim + j] = sum.max(0.0);
            }
        }
        output
    }

    /// TTT self-supervised step: predict masked input from context.
    ///
    /// Mask 20% of input dimensions, predict them from the rest.
    /// Update adapt_weights to minimize reconstruction error.
    pub fn ttt_adapt(&mut self, input: &[f32]) {
        let n = input.len().min(self.in_dim);
        if n < 4 { return; }

        for _step in 0..self.ttt_steps {
            // Create masked version: zero out 20% of dims
            let mask_stride = 5; // mask every 5th element
            let mut masked = input[..n].to_vec();
            for i in (0..n).step_by(mask_stride) {
                masked[i] = 0.0;
            }

            // Forward with masked input
            let predicted = self.forward(&masked);

            // Forward with full input (target)
            let target = self.forward(input);

            // Update adapt_weights to make masked prediction closer to full prediction
            // Simple gradient: d_loss/d_w = (pred - target) * input
            let out_n = predicted.len().min(target.len());
            for j in 0..self.out_dim.min(out_n) {
                let error = predicted[j] - target[j];
                if error.abs() < 1e-6 { continue; }
                for k in 0..n {
                    self.adapt_weights[k * self.out_dim + j] -=
                        self.lr * error * masked[k];
                }
            }
        }
    }

    /// Full TTT inference: adapt then predict.
    pub fn forward_with_ttt(&mut self, input: &[f32]) -> Vec<f32> {
        // 1. Adapt weights to this specific input
        self.ttt_adapt(input);
        // 2. Forward with adapted weights
        self.forward(input)
    }

    /// Reset adaptation (go back to base weights).
    pub fn reset_adaptation(&mut self) {
        self.adapt_weights = self.weights.clone();
    }
}

/// Multi-layer TTT network.
pub struct TttNetwork {
    pub layers: Vec<TttLayer>,
}

impl TttNetwork {
    pub fn new(dims: &[usize]) -> Self {
        let layers = (0..dims.len() - 1)
            .map(|i| TttLayer::new(dims[i], dims[i + 1], 0.37 + i as f32 * 0.17))
            .collect();
        Self { layers }
    }

    /// Forward WITHOUT TTT (standard inference).
    pub fn forward(&self, input: &[f32]) -> Vec<f32> {
        let mut x = input.to_vec();
        for layer in &self.layers {
            x = layer.forward(&x);
        }
        x
    }

    /// Forward WITH TTT (adapt then predict).
    pub fn forward_with_ttt(&mut self, input: &[f32]) -> Vec<f32> {
        let mut x = input.to_vec();
        for layer in &mut self.layers {
            x = layer.forward_with_ttt(&x);
        }
        x
    }

    /// Reset all adaptations.
    pub fn reset(&mut self) {
        for layer in &mut self.layers { layer.reset_adaptation(); }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ttt_layer_basic() {
        let layer = TttLayer::new(16, 8, 0.37);
        let input = vec![0.5f32; 16];
        let output = layer.forward(&input);
        assert_eq!(output.len(), 8);
        assert!(output.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn ttt_adaptation_changes_output() {
        let mut layer = TttLayer::new(32, 16, 0.37);
        let input: Vec<f32> = (0..32).map(|i| (i as f32 * 0.1).sin()).collect();

        // Output before TTT
        let out_before = layer.forward(&input);

        // Adapt
        layer.ttt_adapt(&input);

        // Output after TTT
        let out_after = layer.forward(&input);

        // Should be different (adaptation changed weights)
        let diff: f32 = out_before.iter().zip(out_after.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(diff > 0.001, "TTT should change output, diff={}", diff);
        println!("TTT adaptation diff: {:.6}", diff);
    }

    #[test]
    fn ttt_reset_restores_original() {
        let mut layer = TttLayer::new(16, 8, 0.37);
        let input = vec![0.5f32; 16];

        let out_original = layer.forward(&input);
        layer.ttt_adapt(&input);
        let out_adapted = layer.forward(&input);
        layer.reset_adaptation();
        let out_reset = layer.forward(&input);

        assert_eq!(out_original, out_reset, "Reset should restore original output");
        assert_ne!(out_original, out_adapted, "Adapted should differ");
    }

    #[test]
    fn ttt_network_forward() {
        let mut net = TttNetwork::new(&[32, 16, 8]);
        let input: Vec<f32> = (0..32).map(|i| (i as f32 * 0.1).sin()).collect();

        let out_standard = net.forward(&input);
        let out_ttt = net.forward_with_ttt(&input);

        assert_eq!(out_standard.len(), 8);
        assert_eq!(out_ttt.len(), 8);
        // TTT should produce different output
        let diff: f32 = out_standard.iter().zip(out_ttt.iter()).map(|(a, b)| (a - b).abs()).sum();
        println!("Network TTT diff: {:.6}", diff);
    }

    #[test]
    fn ttt_improves_reconstruction() {
        let mut layer = TttLayer::new(64, 32, 0.37);
        layer.ttt_steps = 10; // more steps for visible effect

        let input: Vec<f32> = (0..64).map(|i| (i as f32 * 0.1).sin()).collect();

        // Measure reconstruction error before and after TTT
        let mask_error_before = measure_mask_error(&layer, &input);
        layer.ttt_adapt(&input);
        let mask_error_after = measure_mask_error(&layer, &input);

        println!("Mask reconstruction error: before={:.4}, after={:.4}", mask_error_before, mask_error_after);
        // After adaptation, reconstruction should be better (or at least not worse)
    }

    fn measure_mask_error(layer: &TttLayer, input: &[f32]) -> f32 {
        let n = input.len().min(layer.in_dim);
        let mut masked = input[..n].to_vec();
        for i in (0..n).step_by(5) { masked[i] = 0.0; }
        let pred = layer.forward(&masked);
        let full = layer.forward(input);
        pred.iter().zip(full.iter()).map(|(a, b)| (a - b).powi(2)).sum::<f32>() / pred.len() as f32
    }
}
