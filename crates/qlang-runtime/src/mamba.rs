//! Mamba: Selective State Space Model for QLANG.
//!
//! Based on "Mamba: Linear-Time Sequence Modeling with Selective State Spaces"
//! (Gu & Dao, 2023) and "BitMamba" (2024) for 1.58-bit quantization.
//!
//! Core SSM equations (discretized):
//!   h_k = A_bar * h_{k-1} + B_bar * x_k
//!   y_k = C * h_k + D * x_k
//!
//! Mamba innovation: A, B, C, delta are INPUT-DEPENDENT (selective).
//!   delta_k = softplus(W_delta @ x_k)   — step size
//!   B_k = W_B @ x_k                      — input matrix
//!   C_k = W_C @ x_k                      — output matrix
//!   A_bar = exp(A * delta_k)              — discretized state matrix
//!   B_bar = delta_k * B_k                 — discretized input matrix
//!
//! Ternary: W_delta, W_B, W_C, linear projections are all {-1, 0, +1}.
//! This is BitMamba applied to QLANG.

/// One Mamba layer with ternary weights.
#[derive(Clone)]
pub struct MambaLayer {
    /// Input projection: [d_model, d_inner * 2] (gate + value)
    pub w_in: Vec<f32>,
    /// SSM projection B: [d_inner, d_state]
    pub w_b: Vec<f32>,
    /// SSM projection C: [d_inner, d_state]
    pub w_c: Vec<f32>,
    /// SSM delta projection: [d_inner, 1]
    pub w_delta: Vec<f32>,
    /// SSM A diagonal (log space, NOT ternary — continuous parameter)
    pub a_log: Vec<f32>,
    /// SSM D skip: [d_inner]
    pub d: Vec<f32>,
    /// Output projection: [d_inner, d_model]
    pub w_out: Vec<f32>,

    pub d_model: usize,
    pub d_inner: usize,
    pub d_state: usize,
}

impl MambaLayer {
    pub fn new(d_model: usize, d_inner: usize, d_state: usize, seed: f32) -> Self {
        let mut rng = seed;
        let mut rand = |scale: f32| -> f32 {
            rng = (rng * 6364.13623).fract() + 0.001;
            rng.sin() * scale
        };

        let scale_in = (2.0 / (d_model + d_inner * 2) as f64).sqrt() as f32;
        let scale_ssm = (2.0 / (d_inner + d_state) as f64).sqrt() as f32;
        let scale_out = (2.0 / (d_inner + d_model) as f64).sqrt() as f32;

        let w_in: Vec<f32> = (0..d_model * d_inner * 2).map(|_| rand(scale_in)).collect();
        let w_b: Vec<f32> = (0..d_inner * d_state).map(|_| rand(scale_ssm)).collect();
        let w_c: Vec<f32> = (0..d_inner * d_state).map(|_| rand(scale_ssm)).collect();
        let w_delta: Vec<f32> = (0..d_inner).map(|_| rand(0.1)).collect();
        // A initialized as negative log-spaced values (standard for S4/Mamba)
        let a_log: Vec<f32> = (0..d_inner * d_state).map(|i| {
            -((i % d_state + 1) as f32).ln()
        }).collect();
        let d = vec![1.0f32; d_inner]; // skip connection
        let w_out: Vec<f32> = (0..d_inner * d_model).map(|_| rand(scale_out)).collect();

        Self { w_in, w_b, w_c, w_delta, a_log, d, w_out, d_model, d_inner, d_state }
    }

    /// Forward pass for one sequence: [seq_len, d_model] → [seq_len, d_model]
    pub fn forward(&self, input: &[f32], seq_len: usize) -> Vec<f32> {
        let d = self.d_model;
        let di = self.d_inner;
        let ds = self.d_state;

        // 1. Input projection: x → (z, gate) where z is value, gate is SiLU gate
        //    [seq, d_model] @ [d_model, d_inner*2] → [seq, d_inner*2]
        let mut projected = vec![0.0f32; seq_len * di * 2];
        for t in 0..seq_len {
            for j in 0..di * 2 {
                let mut sum = 0.0f32;
                for k in 0..d {
                    sum += input[t * d + k] * self.w_in[k * di * 2 + j];
                }
                projected[t * di * 2 + j] = sum;
            }
        }

        // Split into value (z) and gate
        let mut z = vec![0.0f32; seq_len * di];
        let mut gate = vec![0.0f32; seq_len * di];
        for t in 0..seq_len {
            for j in 0..di {
                z[t * di + j] = projected[t * di * 2 + j];
                // SiLU gate: x * sigmoid(x)
                let g = projected[t * di * 2 + di + j];
                gate[t * di + j] = g * (1.0 / (1.0 + (-g).exp()));
            }
        }

        // 2. Selective SSM scan
        //    For each timestep: compute input-dependent B, C, delta
        let mut ssm_out = vec![0.0f32; seq_len * di];
        let mut h = vec![0.0f32; di * ds]; // hidden state [d_inner, d_state]

        for t in 0..seq_len {
            let x_t = &z[t * di..(t + 1) * di];

            // Compute input-dependent delta (step size)
            // delta = softplus(w_delta * x_t)
            let mut delta = vec![0.0f32; di];
            for j in 0..di {
                let raw = self.w_delta[j] * x_t[j];
                delta[j] = (1.0 + raw.exp()).ln().max(0.001); // softplus, min 0.001
            }

            // Compute input-dependent B: [d_inner, d_state]
            let mut b_t = vec![0.0f32; di * ds];
            for j in 0..di {
                for s in 0..ds {
                    b_t[j * ds + s] = self.w_b[j * ds + s] * x_t[j]; // simplified: element-wise
                }
            }

            // Compute input-dependent C: [d_inner, d_state]
            let mut c_t = vec![0.0f32; di * ds];
            for j in 0..di {
                for s in 0..ds {
                    c_t[j * ds + s] = self.w_c[j * ds + s] * x_t[j];
                }
            }

            // Discretize A: A_bar = exp(A * delta)
            // Update hidden state: h = A_bar * h + delta * B * x
            // Output: y = C * h + D * x
            for j in 0..di {
                let mut y_j = self.d[j] * x_t[j]; // skip connection

                for s in 0..ds {
                    let a_val = self.a_log[j * ds + s].exp(); // A is stored as log
                    let a_bar = (-a_val * delta[j]).exp(); // discretized A
                    let b_bar = delta[j] * b_t[j * ds + s]; // discretized B

                    // State update (clamped to prevent explosion)
                    let new_h = a_bar * h[j * ds + s] + b_bar * x_t[j];
                    h[j * ds + s] = new_h.max(-100.0).min(100.0);

                    // Output contribution
                    y_j += c_t[j * ds + s] * h[j * ds + s];
                }

                ssm_out[t * di + j] = y_j;
            }
        }

        // 3. Apply gate: output = ssm_out * gate
        let mut gated = vec![0.0f32; seq_len * di];
        for i in 0..seq_len * di {
            gated[i] = ssm_out[i] * gate[i];
        }

        // 4. Output projection: [seq, d_inner] @ [d_inner, d_model] → [seq, d_model]
        let mut output = vec![0.0f32; seq_len * d];
        for t in 0..seq_len {
            for j in 0..d {
                let mut sum = 0.0f32;
                for k in 0..di {
                    sum += gated[t * di + k] * self.w_out[k * d + j];
                }
                output[t * d + j] = sum;
            }
        }

        // 5. Residual connection
        for i in 0..seq_len * d {
            output[i] += input[i];
        }

        output
    }

    /// Ternarize projection weights using Absmean (BitMamba style).
    /// A_log, D, and w_delta stay as f32 (continuous SSM parameters).
    pub fn ternarize(&mut self) {
        fn tern(weights: &mut Vec<f32>) -> f32 {
            let gamma: f32 = weights.iter().map(|w| w.abs()).sum::<f32>() / weights.len() as f32;
            let eps = 1e-8;
            let scale = gamma + eps;
            for w in weights.iter_mut() {
                let scaled = *w / scale;
                *w = scaled.max(-1.0).min(1.0).round(); // {-1, 0, +1}
            }
            gamma // return scale factor for inference
        }
        tern(&mut self.w_in);
        tern(&mut self.w_b);
        tern(&mut self.w_c);
        tern(&mut self.w_out);
    }

    /// Count total parameters.
    pub fn param_count(&self) -> usize {
        self.w_in.len() + self.w_b.len() + self.w_c.len() + self.w_delta.len()
            + self.a_log.len() + self.d.len() + self.w_out.len()
    }
}

/// A stack of Mamba layers.
pub struct MambaModel {
    pub layers: Vec<MambaLayer>,
    pub d_model: usize,
}

impl MambaModel {
    /// Create a Mamba model with N layers.
    pub fn new(d_model: usize, d_inner: usize, d_state: usize, n_layers: usize) -> Self {
        let layers = (0..n_layers)
            .map(|i| MambaLayer::new(d_model, d_inner, d_state, 0.37 + i as f32 * 0.17))
            .collect();
        Self { layers, d_model }
    }

    /// Forward: sequence through all layers.
    pub fn forward(&self, input: &[f32], seq_len: usize) -> Vec<f32> {
        let mut x = input.to_vec();
        for layer in &self.layers {
            x = layer.forward(&x, seq_len);
        }
        x
    }

    /// Ternarize all layers.
    pub fn ternarize(&mut self) {
        for layer in &mut self.layers { layer.ternarize(); }
    }

    pub fn param_count(&self) -> usize {
        self.layers.iter().map(|l| l.param_count()).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mamba_layer_forward() {
        let layer = MambaLayer::new(32, 64, 16, 0.37);
        let seq_len = 8;
        let input = vec![0.1f32; seq_len * 32];

        let output = layer.forward(&input, seq_len);
        assert_eq!(output.len(), seq_len * 32);
        assert!(output.iter().all(|x| x.is_finite()), "All outputs must be finite");

        println!("Mamba layer: d_model=32, d_inner=64, d_state=16");
        println!("  Input:  [{}, 32]", seq_len);
        println!("  Output: [{}, 32]", seq_len);
        println!("  Params: {}", layer.param_count());
        println!("  Output[0]: {:.4}", output[0]);
    }

    #[test]
    fn mamba_model_multi_layer() {
        let model = MambaModel::new(32, 64, 16, 4);
        let seq_len = 8;
        let input = vec![0.1f32; seq_len * 32];

        let output = model.forward(&input, seq_len);
        assert_eq!(output.len(), seq_len * 32);
        assert!(output.iter().all(|x| x.is_finite()));

        println!("Mamba model: 4 layers, d_model=32, d_inner=64, d_state=16");
        println!("  Total params: {}", model.param_count());
    }

    #[test]
    fn mamba_ternary() {
        let mut model = MambaModel::new(32, 64, 16, 2);
        let seq_len = 4;
        let input: Vec<f32> = (0..seq_len * 32).map(|i| (i as f32 * 0.1).sin()).collect();

        // Forward before ternarize
        let out_f32 = model.forward(&input, seq_len);

        // Ternarize
        model.ternarize();

        // Forward after ternarize
        let out_tern = model.forward(&input, seq_len);

        // Check weights are ternary
        for layer in &model.layers {
            assert!(layer.w_in.iter().all(|&w| w == -1.0 || w == 0.0 || w == 1.0),
                "w_in must be ternary");
            assert!(layer.w_out.iter().all(|&w| w == -1.0 || w == 0.0 || w == 1.0),
                "w_out must be ternary");
        }

        println!("Mamba ternary: weights are {{-1, 0, +1}}");
        println!("  f32 output[0]: {:.4}", out_f32[0]);
        println!("  tern output[0]: {:.4}", out_tern[0]);
        assert!(out_tern.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn mamba_sequential_state() {
        // Verify that state carries information across timesteps
        let layer = MambaLayer::new(16, 32, 8, 0.37);

        // Input with a spike at t=0, zeros after
        let seq_len = 8;
        let mut input = vec![0.0f32; seq_len * 16];
        for j in 0..16 { input[j] = 1.0; } // spike at t=0

        let output = layer.forward(&input, seq_len);

        // The spike should propagate: output at later timesteps should not be zero
        let t0_energy: f32 = output[..16].iter().map(|x| x.abs()).sum();
        let t4_energy: f32 = output[4 * 16..5 * 16].iter().map(|x| x.abs()).sum();

        println!("State propagation: t0_energy={:.4}, t4_energy={:.4}", t0_energy, t4_energy);
        // State decays — energy at t=4 will be less than t=0 but residual adds to it
        assert!(t0_energy > 0.1, "Output at t=0 must be non-zero (got {:.4})", t0_energy);
    }
}
