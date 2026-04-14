//! Hebbian Learning for Ternary Weight Layers
//!
//! Ported from ParaDiffuse (Python/PyTorch) to Rust.
//!
//! Bio-inspired learning rule: "neurons that fire together wire together."
//!
//! This module implements gradient-free Hebbian learning that works directly
//! with ternary weights {-1, 0, +1}. Instead of backpropagation, it tracks
//! correlations between pre- and post-synaptic activations (the **salience**
//! signal) and flips ternary weights when correlations are strong enough.
//!
//! Key advantages over gradient-based training for ternary networks:
//! - No gradient computation required (O(1) per weight per sample)
//! - Naturally produces ternary outputs (no quantisation artefacts)
//! - Compatible with IGQK quantum measurement for final weight selection

// ---------------------------------------------------------------------------
// HebbianState
// ---------------------------------------------------------------------------

/// Hebbian learning state for a single ternary weight layer.
///
/// Tracks running means of pre/post activations and accumulates a
/// salience signal (centered correlation) that drives weight updates.
#[derive(Debug, Clone)]
pub struct HebbianState {
    /// Accumulated correlation signal `[out_dim * in_dim]` (row-major).
    salience: Vec<f32>,
    /// Running mean of pre-activations `[in_dim]`.
    pre_mean: Vec<f32>,
    /// Running mean of post-activations `[out_dim]`.
    post_mean: Vec<f32>,
    /// Threshold for flipping a ternary weight.
    threshold: f32,
    /// Momentum for exponential moving average of means.
    momentum: f32,
    /// Decay factor applied to salience after each `apply_to_weights` call.
    salience_decay: f32,
    /// Input dimension.
    in_dim: usize,
    /// Output dimension.
    out_dim: usize,
}

impl HebbianState {
    /// Create a new Hebbian state for a layer with the given dimensions.
    pub fn new(in_dim: usize, out_dim: usize) -> Self {
        HebbianState {
            salience: vec![0.0; out_dim * in_dim],
            pre_mean: vec![0.0; in_dim],
            post_mean: vec![0.0; out_dim],
            threshold: 0.1,
            momentum: 0.9,
            salience_decay: 0.95,
            in_dim,
            out_dim,
        }
    }

    /// Create with custom hyper-parameters.
    pub fn with_params(
        in_dim: usize,
        out_dim: usize,
        threshold: f32,
        momentum: f32,
        salience_decay: f32,
    ) -> Self {
        HebbianState {
            salience: vec![0.0; out_dim * in_dim],
            pre_mean: vec![0.0; in_dim],
            post_mean: vec![0.0; out_dim],
            threshold,
            momentum,
            salience_decay,
            in_dim,
            out_dim,
        }
    }

    /// Input dimension.
    pub fn in_dim(&self) -> usize {
        self.in_dim
    }

    /// Output dimension.
    pub fn out_dim(&self) -> usize {
        self.out_dim
    }

    /// Read the current salience matrix (row-major `[out_dim, in_dim]`).
    pub fn salience(&self) -> &[f32] {
        &self.salience
    }

    /// Read the running pre-activation mean.
    pub fn pre_mean(&self) -> &[f32] {
        &self.pre_mean
    }

    /// Read the running post-activation mean.
    pub fn post_mean(&self) -> &[f32] {
        &self.post_mean
    }

    /// Update Hebbian salience from a single sample's pre/post activations.
    ///
    /// 1. Updates running means via exponential moving average.
    /// 2. Computes centered sign-correlation:
    ///    `delta_salience[i,j] = sign(post[i] - post_mean[i]) * sign(pre[j] - pre_mean[j])`
    /// 3. Accumulates into the salience matrix.
    ///
    /// # Panics
    /// Panics if `pre_act` length differs from `in_dim` or `post_act` from `out_dim`.
    pub fn update(&mut self, pre_act: &[f32], post_act: &[f32]) {
        assert_eq!(
            pre_act.len(),
            self.in_dim,
            "pre_act length {} != in_dim {}",
            pre_act.len(),
            self.in_dim
        );
        assert_eq!(
            post_act.len(),
            self.out_dim,
            "post_act length {} != out_dim {}",
            post_act.len(),
            self.out_dim
        );

        let alpha = 1.0 - self.momentum;

        // Update running means.
        for (m, &x) in self.pre_mean.iter_mut().zip(pre_act.iter()) {
            *m = self.momentum * *m + alpha * x;
        }
        for (m, &y) in self.post_mean.iter_mut().zip(post_act.iter()) {
            *m = self.momentum * *m + alpha * y;
        }

        // Accumulate centered sign-correlation.
        for i in 0..self.out_dim {
            let post_sign = sign(post_act[i] - self.post_mean[i]);
            let row_offset = i * self.in_dim;
            for j in 0..self.in_dim {
                let pre_sign = sign(pre_act[j] - self.pre_mean[j]);
                self.salience[row_offset + j] += post_sign * pre_sign;
            }
        }
    }

    /// Apply Hebbian updates to ternary weights.
    ///
    /// For each weight position:
    /// - If salience > threshold: set weight to +1
    /// - If salience < -threshold: set weight to -1
    /// - Otherwise: leave unchanged
    ///
    /// After applying, decays all salience values by `salience_decay`.
    ///
    /// Returns the number of weights that were flipped.
    ///
    /// # Panics
    /// Panics if `weights.len() != out_dim * in_dim`.
    pub fn apply_to_weights(&mut self, weights: &mut [f32]) -> usize {
        assert_eq!(
            weights.len(),
            self.out_dim * self.in_dim,
            "weights length {} != out_dim*in_dim {}",
            weights.len(),
            self.out_dim * self.in_dim
        );

        let mut flipped = 0;
        for i in 0..weights.len() {
            let s = self.salience[i];
            if s > self.threshold {
                if weights[i] != 1.0 {
                    weights[i] = 1.0;
                    flipped += 1;
                }
            } else if s < -self.threshold {
                if weights[i] != -1.0 {
                    weights[i] = -1.0;
                    flipped += 1;
                }
            }
            // Decay salience.
            self.salience[i] *= self.salience_decay;
        }
        flipped
    }

    /// Reset the salience accumulator to zero.
    pub fn reset(&mut self) {
        self.salience.fill(0.0);
    }

    /// Reset everything (salience + running means).
    pub fn reset_all(&mut self) {
        self.salience.fill(0.0);
        self.pre_mean.fill(0.0);
        self.post_mean.fill(0.0);
    }
}

// ---------------------------------------------------------------------------
// Convenience functions
// ---------------------------------------------------------------------------

/// Perform one Hebbian training step on a ternary layer.
///
/// 1. Forward pass: `output = weights * input` (ternary matmul)
/// 2. Update Hebbian state with the (input, output) correlation
/// 3. Apply weight updates based on accumulated salience
///
/// Returns the output activations.
pub fn hebbian_train_step(
    weights: &mut [f32],
    input: &[f32],
    state: &mut HebbianState,
) -> Vec<f32> {
    let out_dim = state.out_dim;
    let in_dim = state.in_dim;

    assert_eq!(
        weights.len(),
        out_dim * in_dim,
        "weights length mismatch"
    );
    assert_eq!(input.len(), in_dim, "input length mismatch");

    // Forward: output[i] = sum_j weights[i, j] * input[j]
    let mut output = vec![0.0; out_dim];
    for i in 0..out_dim {
        let row_offset = i * in_dim;
        let mut sum = 0.0_f32;
        for j in 0..in_dim {
            sum += weights[row_offset + j] * input[j];
        }
        output[i] = sum;
    }

    // Update Hebbian state.
    state.update(input, &output);

    // Apply weight updates.
    state.apply_to_weights(weights);

    output
}

/// Initialise a ternary weight vector: each weight is randomly {-1, 0, +1}.
///
/// Uses a deterministic xorshift seeded from `seed`.
pub fn random_ternary_weights(n: usize, seed: u64) -> Vec<f32> {
    let mut state = if seed == 0 { 0xBAAD_CAFE_u64 } else { seed };
    let mut weights = Vec::with_capacity(n);
    for _ in 0..n {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        let r = state % 3;
        weights.push(match r {
            0 => -1.0,
            1 => 0.0,
            _ => 1.0,
        });
    }
    weights
}

/// Sign function: returns -1.0, 0.0, or 1.0.
#[inline]
fn sign(x: f32) -> f32 {
    if x > 0.0 {
        1.0
    } else if x < 0.0 {
        -1.0
    } else {
        0.0
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn state_creation() {
        let s = HebbianState::new(4, 3);
        assert_eq!(s.in_dim(), 4);
        assert_eq!(s.out_dim(), 3);
        assert_eq!(s.salience().len(), 12);
        assert!(s.salience().iter().all(|&v| v == 0.0));
    }

    #[test]
    fn update_changes_salience() {
        let mut s = HebbianState::new(3, 2);
        let pre = vec![1.0, -1.0, 0.5];
        let post = vec![1.0, -1.0];
        s.update(&pre, &post);

        // After one update with zero running means, salience should be non-zero.
        let total: f32 = s.salience().iter().map(|v| v.abs()).sum();
        assert!(total > 0.0, "Salience should be non-zero after update");
    }

    #[test]
    fn update_running_means() {
        let mut s = HebbianState::new(2, 2);
        let pre = vec![10.0, -10.0];
        let post = vec![5.0, -5.0];

        // After several updates, running means should approach the input values.
        for _ in 0..100 {
            s.update(&pre, &post);
        }

        assert!(
            (s.pre_mean()[0] - 10.0).abs() < 0.5,
            "pre_mean[0] should converge toward 10.0, got {}",
            s.pre_mean()[0]
        );
        assert!(
            (s.post_mean()[0] - 5.0).abs() < 0.5,
            "post_mean[0] should converge toward 5.0, got {}",
            s.post_mean()[0]
        );
    }

    #[test]
    fn apply_flips_correlated_weights() {
        // Use high momentum (0.9) so that running means lag behind, keeping
        // a non-zero centered signal across updates.
        let mut s = HebbianState::with_params(2, 2, 0.5, 0.9, 0.95);
        let mut weights = vec![0.0; 4]; // all zero initially

        // Alternate between two patterns that share the same sign structure.
        // This keeps the centered correlation positive.
        let patterns: &[(&[f32], &[f32])] = &[
            (&[2.0, 2.0], &[3.0, 3.0]),
            (&[1.0, 1.0], &[1.5, 1.5]),
        ];
        for i in 0..20 {
            let (pre, post) = patterns[i % 2];
            s.update(pre, post);
        }
        let flipped = s.apply_to_weights(&mut weights);
        assert!(flipped > 0, "Should have flipped some weights");
        assert!(
            weights.iter().any(|&w| w == 1.0),
            "At least one weight should be +1"
        );
    }

    #[test]
    fn apply_flips_anticorrelated_weights() {
        let mut s = HebbianState::with_params(2, 2, 0.5, 0.9, 0.95);
        let mut weights = vec![0.0; 4];

        // Pre is above mean, post is below mean -> anticorrelation -> -1.
        let patterns: &[(&[f32], &[f32])] = &[
            (&[2.0, 2.0], &[-3.0, -3.0]),
            (&[1.0, 1.0], &[-1.5, -1.5]),
        ];
        for i in 0..20 {
            let (pre, post) = patterns[i % 2];
            s.update(pre, post);
        }
        let flipped = s.apply_to_weights(&mut weights);
        assert!(flipped > 0, "Should have flipped some weights");
        assert!(
            weights.iter().any(|&w| w == -1.0),
            "At least one weight should be -1"
        );
    }

    #[test]
    fn salience_decays() {
        let mut s = HebbianState::new(2, 2);
        let pre = vec![1.0, 1.0];
        let post = vec![1.0, 1.0];
        s.update(&pre, &post);

        let before: Vec<f32> = s.salience().to_vec();
        let mut dummy_weights = vec![0.0; 4];
        // Set threshold very high so no weights flip, just to trigger decay.
        s.threshold = 1e6;
        s.apply_to_weights(&mut dummy_weights);

        for (i, (&b, &a)) in before.iter().zip(s.salience().iter()).enumerate() {
            if b != 0.0 {
                assert!(
                    a.abs() < b.abs(),
                    "Salience[{i}] should have decayed: {b} -> {a}"
                );
            }
        }
    }

    #[test]
    fn reset_clears_salience() {
        let mut s = HebbianState::new(3, 2);
        s.update(&[1.0, 2.0, 3.0], &[1.0, 1.0]);
        assert!(s.salience().iter().any(|&v| v != 0.0));
        s.reset();
        assert!(s.salience().iter().all(|&v| v == 0.0));
    }

    #[test]
    fn reset_all_clears_everything() {
        let mut s = HebbianState::new(2, 2);
        s.update(&[1.0, 2.0], &[3.0, 4.0]);
        s.reset_all();
        assert!(s.salience().iter().all(|&v| v == 0.0));
        assert!(s.pre_mean().iter().all(|&v| v == 0.0));
        assert!(s.post_mean().iter().all(|&v| v == 0.0));
    }

    #[test]
    fn hebbian_train_step_output_shape() {
        let mut state = HebbianState::new(4, 3);
        let mut weights = vec![0.0; 12];
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let output = hebbian_train_step(&mut weights, &input, &mut state);
        assert_eq!(output.len(), 3);
    }

    #[test]
    fn hebbian_train_step_forward_pass_correct() {
        let mut state = HebbianState::new(3, 2);
        // weights: [[1, 0, -1], [0, 1, 1]]
        let mut weights = vec![1.0, 0.0, -1.0, 0.0, 1.0, 1.0];
        let input = vec![1.0, 2.0, 3.0];

        // Manually: out[0] = 1*1 + 0*2 + (-1)*3 = -2
        //           out[1] = 0*1 + 1*2 + 1*3 = 5
        // (Note: weights may be mutated by apply_to_weights after the forward pass,
        //  but the returned output is computed before that.)
        let output = hebbian_train_step(&mut weights, &input, &mut state);
        assert!(
            (output[0] - (-2.0)).abs() < 1e-6,
            "output[0] = {}, expected -2",
            output[0]
        );
        assert!(
            (output[1] - 5.0).abs() < 1e-6,
            "output[1] = {}, expected 5",
            output[1]
        );
    }

    #[test]
    fn random_ternary_weights_values() {
        let w = random_ternary_weights(100, 42);
        assert_eq!(w.len(), 100);
        for &v in &w {
            assert!(
                v == -1.0 || v == 0.0 || v == 1.0,
                "Weight {v} is not ternary"
            );
        }
    }

    #[test]
    fn random_ternary_weights_distribution() {
        // With enough weights, all three values should appear.
        let w = random_ternary_weights(1000, 99);
        let neg = w.iter().filter(|&&v| v == -1.0).count();
        let zero = w.iter().filter(|&&v| v == 0.0).count();
        let pos = w.iter().filter(|&&v| v == 1.0).count();
        assert!(neg > 0, "Expected some -1 weights");
        assert!(zero > 0, "Expected some 0 weights");
        assert!(pos > 0, "Expected some +1 weights");
    }

    #[test]
    fn random_ternary_deterministic() {
        let a = random_ternary_weights(50, 123);
        let b = random_ternary_weights(50, 123);
        assert_eq!(a, b);
    }

    #[test]
    fn sign_function() {
        assert_eq!(sign(1.5), 1.0);
        assert_eq!(sign(-0.3), -1.0);
        assert_eq!(sign(0.0), 0.0);
    }

    #[test]
    #[should_panic(expected = "pre_act length")]
    fn update_panics_on_wrong_pre_dim() {
        let mut s = HebbianState::new(3, 2);
        s.update(&[1.0, 2.0], &[1.0, 1.0]); // pre has 2 elements, expected 3
    }

    #[test]
    #[should_panic(expected = "post_act length")]
    fn update_panics_on_wrong_post_dim() {
        let mut s = HebbianState::new(3, 2);
        s.update(&[1.0, 2.0, 3.0], &[1.0]); // post has 1 element, expected 2
    }

    #[test]
    fn convergence_over_many_steps() {
        // Start with random ternary weights so the forward pass produces
        // non-zero outputs, giving Hebbian learning something to work with.
        let mut state = HebbianState::with_params(3, 2, 0.05, 0.9, 1.0);
        let mut weights = random_ternary_weights(6, 42); // [out=2, in=3]
        let initial_weights = weights.clone();

        // Alternate inputs with varying magnitudes to create centered
        // correlation signals (running mean lags behind actual values).
        let inputs: &[&[f32]] = &[
            &[3.0, -1.0, 2.0],
            &[1.0, -3.0, 0.5],
            &[4.0, -2.0, 3.0],
            &[0.5, -0.5, 1.0],
        ];

        for i in 0..200 {
            let input = inputs[i % inputs.len()];
            hebbian_train_step(&mut weights, input, &mut state);
        }

        // After many steps, at least some weights should have changed.
        let changed = weights
            .iter()
            .zip(initial_weights.iter())
            .filter(|(&w, &iw)| w != iw)
            .count();
        assert!(
            changed > 0,
            "At least some weights should have changed after Hebbian training"
        );
    }
}
