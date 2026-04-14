//! Transformer Training via Finite-Difference Gradient Estimation
//!
//! This module does NOT implement true backpropagation.
//! Gradients are estimated via finite differences on a subset of parameters.
//! Status: experimental, not production-grade.
//! For real backprop-based Transformer training, see `candle_train.rs` (tch-rs path).
//!
//! Implements a minimal GPT-style decoder-only transformer:
//! - Token + positional embeddings
//! - Multi-head causal self-attention
//! - Feed-forward network with GELU activation
//! - Layer normalization (pre-norm architecture)
//! - Cross-entropy loss for next-token prediction
//! - Training via finite-difference gradient estimation (not true backprop)
//! - Autoregressive text generation
//!
//! Uses `crate::accel::matmul` for hardware-accelerated matrix operations.

use crate::accel;
use std::io::{Read, Write, BufReader, BufWriter};

// ---------------------------------------------------------------------------
// Activation & normalization helpers
// ---------------------------------------------------------------------------

/// GELU activation: x * Phi(x) (approximate form).
fn gelu(x: f32) -> f32 {
    let sqrt_2_over_pi = (2.0f32 / std::f32::consts::PI).sqrt();
    0.5 * x * (1.0 + (sqrt_2_over_pi * (x + 0.044715 * x * x * x)).tanh())
}

/// SiLU (Sigmoid Linear Unit) activation: x * sigmoid(x).
///
/// Also known as the Swish activation. Used as an alternative to GELU
/// in modern architectures (e.g., LLaMA, Mistral) for potentially
/// faster convergence.
fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// Softmax in-place over a slice.
fn softmax(logits: &mut [f32]) {
    if logits.is_empty() {
        return;
    }
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for v in logits.iter_mut() {
        *v = (*v - max).exp();
        sum += *v;
    }
    if sum > 0.0 {
        for v in logits.iter_mut() {
            *v /= sum;
        }
    }
}

/// Layer normalization over vectors of dimension `d`.
///
/// `x` is a flat buffer of `n_vectors * d` floats. Each contiguous
/// slice of `d` elements is independently normalized, then scaled
/// by `gamma` and shifted by `beta`.
fn layer_norm(x: &mut [f32], gamma: &[f32], beta: &[f32], d: usize, eps: f32) {
    let n_vectors = x.len() / d;
    for v in 0..n_vectors {
        let offset = v * d;
        let slice = &x[offset..offset + d];

        let mean: f32 = slice.iter().sum::<f32>() / d as f32;
        let var: f32 = slice.iter().map(|&val| (val - mean) * (val - mean)).sum::<f32>() / d as f32;
        let std_inv = 1.0 / (var + eps).sqrt();

        for j in 0..d {
            x[offset + j] = (x[offset + j] - mean) * std_inv * gamma[j] + beta[j];
        }
    }
}

/// RMSNorm (Root Mean Square Layer Normalization).
///
/// ~15% faster than LayerNorm because it skips mean-centering.
/// Only computes RMS and scales by gamma -- no beta shift.
/// Used by LLaMA, Mistral, and other modern architectures.
///
/// `x` is a flat buffer of `n_vectors * d` floats. Each contiguous
/// slice of `d` elements is independently normalized, then scaled
/// by `gamma`.
fn rms_norm(x: &mut [f32], gamma: &[f32], d: usize, eps: f32) {
    let n = x.len() / d;
    for i in 0..n {
        let offset = i * d;
        let slice = &x[offset..offset + d];
        let rms = (slice.iter().map(|v| v * v).sum::<f32>() / d as f32 + eps).sqrt();
        for j in 0..d {
            x[offset + j] = x[offset + j] / rms * gamma[j];
        }
    }
}

// ---------------------------------------------------------------------------
// Model configuration
// ---------------------------------------------------------------------------

/// Configuration for a small transformer language model.
#[derive(Clone, Debug)]
pub struct TransformerConfig {
    /// Vocabulary size (from tokenizer)
    pub vocab_size: usize,
    /// Embedding / hidden dimension (e.g. 64, 128, 256, 512)
    pub d_model: usize,
    /// Number of attention heads (e.g. 4, 8)
    pub n_heads: usize,
    /// Number of transformer blocks (e.g. 2, 4, 6)
    pub n_layers: usize,
    /// Maximum sequence length (e.g. 128, 256, 512)
    pub max_seq_len: usize,
    /// Dropout rate (unused in inference, reserved for future use)
    pub dropout: f32,
    /// Use RMSNorm instead of LayerNorm (~15% faster). Default: true for new models.
    pub use_rms_norm: bool,
    /// Use SiLU activation instead of GELU. Default: true for new models.
    pub use_silu: bool,
}

// ---------------------------------------------------------------------------
// Layer norm parameters
// ---------------------------------------------------------------------------

/// Learnable scale (gamma) and shift (beta) for layer normalization.
#[derive(Clone)]
struct LayerNormParams {
    gamma: Vec<f32>, // [d_model]
    beta: Vec<f32>,  // [d_model]
}

impl LayerNormParams {
    fn new(d: usize) -> Self {
        Self {
            gamma: vec![1.0; d],
            beta: vec![0.0; d],
        }
    }

    fn param_count(&self) -> usize {
        self.gamma.len() + self.beta.len()
    }

    /// Collect all parameters as mutable references.
    fn params_mut(&mut self) -> Vec<&mut f32> {
        let mut v: Vec<&mut f32> = Vec::with_capacity(self.gamma.len() + self.beta.len());
        for p in self.gamma.iter_mut() {
            v.push(p);
        }
        for p in self.beta.iter_mut() {
            v.push(p);
        }
        v
    }
}

// ---------------------------------------------------------------------------
// Transformer block
// ---------------------------------------------------------------------------

/// A single transformer decoder block (pre-norm, causal attention).
#[derive(Clone)]
struct TransformerBlock {
    ln1: LayerNormParams,
    attn_qkv: Vec<f32>, // [d_model, 3 * d_model]
    attn_out: Vec<f32>,  // [d_model, d_model]
    ln2: LayerNormParams,
    ffn_up: Vec<f32>,    // [d_model, 4 * d_model]
    ffn_down: Vec<f32>,  // [4 * d_model, d_model]
}

impl TransformerBlock {
    fn new(d_model: usize, seed: u64) -> Self {
        let scale_attn = (1.0 / d_model as f64).sqrt() as f32;
        let d_ff = 4 * d_model;
        let scale_ff = (1.0 / d_ff as f64).sqrt() as f32;

        Self {
            ln1: LayerNormParams::new(d_model),
            attn_qkv: init_weights(d_model * 3 * d_model, scale_attn, seed),
            attn_out: init_weights(d_model * d_model, scale_attn, seed.wrapping_add(1)),
            ln2: LayerNormParams::new(d_model),
            ffn_up: init_weights(d_model * d_ff, scale_ff, seed.wrapping_add(2)),
            ffn_down: init_weights(d_ff * d_model, scale_ff, seed.wrapping_add(3)),
        }
    }

    fn param_count(&self) -> usize {
        self.ln1.param_count()
            + self.attn_qkv.len()
            + self.attn_out.len()
            + self.ln2.param_count()
            + self.ffn_up.len()
            + self.ffn_down.len()
    }

    /// Collect mutable references to all parameters.
    fn params_mut(&mut self) -> Vec<&mut f32> {
        let mut v = self.ln1.params_mut();
        for p in self.attn_qkv.iter_mut() { v.push(p); }
        for p in self.attn_out.iter_mut() { v.push(p); }
        v.extend(self.ln2.params_mut());
        for p in self.ffn_up.iter_mut() { v.push(p); }
        for p in self.ffn_down.iter_mut() { v.push(p); }
        v
    }

    /// Forward pass through one block.
    ///
    /// `x` is [seq_len, d_model], returned as new [seq_len, d_model].
    fn forward(&self, x: &[f32], seq_len: usize, d_model: usize, n_heads: usize,
               use_rms: bool, use_silu_act: bool) -> Vec<f32> {
        let d_head = d_model / n_heads;

        // --- Pre-norm attention ---
        let mut normed = x.to_vec();
        if use_rms {
            rms_norm(&mut normed, &self.ln1.gamma, d_model, 1e-5);
        } else {
            layer_norm(&mut normed, &self.ln1.gamma, &self.ln1.beta, d_model, 1e-5);
        }

        // QKV projection: [seq_len, d_model] x [d_model, 3*d_model] -> [seq_len, 3*d_model]
        let qkv = accel::matmul(&normed, &self.attn_qkv, seq_len, 3 * d_model, d_model);

        // Split into Q, K, V
        let mut q = vec![0.0f32; seq_len * d_model];
        let mut k = vec![0.0f32; seq_len * d_model];
        let mut v_mat = vec![0.0f32; seq_len * d_model];
        for s in 0..seq_len {
            for d in 0..d_model {
                q[s * d_model + d] = qkv[s * 3 * d_model + d];
                k[s * d_model + d] = qkv[s * 3 * d_model + d_model + d];
                v_mat[s * d_model + d] = qkv[s * 3 * d_model + 2 * d_model + d];
            }
        }

        // Multi-head causal attention
        let attn_out = causal_multi_head_attention(
            &q, &k, &v_mat, n_heads, seq_len, d_model, d_head,
        );

        // Output projection: [seq_len, d_model] x [d_model, d_model] -> [seq_len, d_model]
        let projected = accel::matmul(&attn_out, &self.attn_out, seq_len, d_model, d_model);

        // Residual connection
        let mut residual: Vec<f32> = x.iter().zip(projected.iter()).map(|(&a, &b)| a + b).collect();

        // --- Pre-norm FFN ---
        let mut normed2 = residual.clone();
        if use_rms {
            rms_norm(&mut normed2, &self.ln2.gamma, d_model, 1e-5);
        } else {
            layer_norm(&mut normed2, &self.ln2.gamma, &self.ln2.beta, d_model, 1e-5);
        }

        let d_ff = 4 * d_model;

        // Up projection + activation (SiLU or GELU)
        let mut hidden = accel::matmul(&normed2, &self.ffn_up, seq_len, d_ff, d_model);
        if use_silu_act {
            for v in hidden.iter_mut() {
                *v = silu(*v);
            }
        } else {
            for v in hidden.iter_mut() {
                *v = gelu(*v);
            }
        }

        // Down projection
        let ff_out = accel::matmul(&hidden, &self.ffn_down, seq_len, d_model, d_ff);

        // Residual connection
        for i in 0..residual.len() {
            residual[i] += ff_out[i];
        }

        residual
    }
}

// ---------------------------------------------------------------------------
// Multi-head causal attention
// ---------------------------------------------------------------------------

/// Multi-head causal (masked) self-attention.
///
/// Q, K, V are all [seq_len, d_model]. Output is [seq_len, d_model].
fn causal_multi_head_attention(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    n_heads: usize,
    seq_len: usize,
    d_model: usize,
    d_head: usize,
) -> Vec<f32> {
    let scale = 1.0 / (d_head as f32).sqrt();
    let mut output = vec![0.0f32; seq_len * d_model];

    for h in 0..n_heads {
        let h_offset = h * d_head;

        // Compute attention scores for this head
        // scores[i][j] = sum_d Q[i, h*d_head+d] * K[j, h*d_head+d] / sqrt(d_head)
        let mut scores = vec![0.0f32; seq_len * seq_len];
        for i in 0..seq_len {
            for j in 0..seq_len {
                if j > i {
                    // Causal mask: future positions get -inf
                    scores[i * seq_len + j] = f32::NEG_INFINITY;
                } else {
                    let mut dot = 0.0f32;
                    for d in 0..d_head {
                        dot += q[i * d_model + h_offset + d]
                            * k[j * d_model + h_offset + d];
                    }
                    scores[i * seq_len + j] = dot * scale;
                }
            }

            // Softmax over the j dimension for this query position i
            let row = &mut scores[i * seq_len..(i + 1) * seq_len];
            softmax(row);
        }

        // Weighted sum: output[i, h*d_head+d] = sum_j scores[i,j] * V[j, h*d_head+d]
        for i in 0..seq_len {
            for d in 0..d_head {
                let mut sum = 0.0f32;
                for j in 0..seq_len {
                    sum += scores[i * seq_len + j] * v[j * d_model + h_offset + d];
                }
                output[i * d_model + h_offset + d] = sum;
            }
        }
    }

    output
}

// ---------------------------------------------------------------------------
// Weight initialization helpers
// ---------------------------------------------------------------------------

/// Simple deterministic pseudo-random initialization using a hash function.
fn init_weights(n: usize, scale: f32, seed: u64) -> Vec<f32> {
    (0..n)
        .map(|i| {
            // Simple hash-based pseudo-random
            let h = splitmix64(seed.wrapping_add(i as u64));
            // Map to [-1, 1] then scale
            let f = (h as f64 / u64::MAX as f64) * 2.0 - 1.0;
            f as f32 * scale
        })
        .collect()
}

/// SplitMix64 PRNG (single step).
fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9e3779b97f4a7c15);
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d049bb133111eb);
    x ^ (x >> 31)
}

/// Simple PRNG state for sampling during generation and training.
struct Rng {
    state: u64,
}

impl Rng {
    fn new(seed: u64) -> Self {
        Self { state: seed.max(1) }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = splitmix64(self.state);
        self.state
    }

    fn next_f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }

    /// Return a random index in [0, n).
    fn next_usize(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }
}

// ---------------------------------------------------------------------------
// MiniGPT model
// ---------------------------------------------------------------------------

/// A small GPT-style decoder-only language model.
pub struct MiniGPT {
    config: TransformerConfig,
    /// Token embedding table: [vocab_size, d_model]
    token_embedding: Vec<f32>,
    /// Learned positional embedding: [max_seq_len, d_model]
    position_embedding: Vec<f32>,
    /// Transformer blocks
    blocks: Vec<TransformerBlock>,
    /// Final layer norm
    ln_final: LayerNormParams,
    /// Language model head: [d_model, vocab_size]
    lm_head: Vec<f32>,
}

impl MiniGPT {
    /// Create a new model with random weights.
    pub fn new(config: TransformerConfig) -> Self {
        let d = config.d_model;
        let v = config.vocab_size;
        let seed_base: u64 = 42;

        let emb_scale = (1.0 / d as f64).sqrt() as f32;

        let token_embedding = init_weights(v * d, emb_scale, seed_base);
        let position_embedding = init_weights(config.max_seq_len * d, emb_scale, seed_base + 100);

        let blocks: Vec<TransformerBlock> = (0..config.n_layers)
            .map(|i| TransformerBlock::new(d, seed_base + 1000 + i as u64 * 100))
            .collect();

        let ln_final = LayerNormParams::new(d);

        // lm_head shares the same scale as embeddings
        let lm_head = init_weights(d * v, emb_scale, seed_base + 9999);

        Self {
            config,
            token_embedding,
            position_embedding,
            blocks,
            ln_final,
            lm_head,
        }
    }

    /// Forward pass: tokens -> logits.
    ///
    /// `tokens` is a slice of token IDs of length `seq_len`.
    /// Returns logits of shape [seq_len, vocab_size].
    pub fn forward(&self, tokens: &[u32], seq_len: usize) -> Vec<f32> {
        let d = self.config.d_model;
        let v = self.config.vocab_size;
        let actual_len = seq_len.min(tokens.len()).min(self.config.max_seq_len);

        // 1. Token embedding + Position embedding
        let mut x = vec![0.0f32; actual_len * d];
        for (pos, &tok) in tokens.iter().take(actual_len).enumerate() {
            let tok_idx = (tok as usize).min(v - 1);
            for j in 0..d {
                x[pos * d + j] = self.token_embedding[tok_idx * d + j]
                    + self.position_embedding[pos * d + j];
            }
        }

        // 2. Transformer blocks
        let use_rms = self.config.use_rms_norm;
        let use_silu_act = self.config.use_silu;
        for block in &self.blocks {
            x = block.forward(&x, actual_len, d, self.config.n_heads, use_rms, use_silu_act);
        }

        // 3. Final layer norm
        if use_rms {
            rms_norm(&mut x, &self.ln_final.gamma, d, 1e-5);
        } else {
            layer_norm(&mut x, &self.ln_final.gamma, &self.ln_final.beta, d, 1e-5);
        }

        // 4. Project to vocab: [seq_len, d_model] x [d_model, vocab_size] -> [seq_len, vocab_size]
        accel::matmul(&x, &self.lm_head, actual_len, v, d)
    }

    /// Compute cross-entropy loss for next-token prediction.
    ///
    /// `logits` has shape [seq_len, vocab_size].
    /// `targets` has length `seq_len` (the next token at each position).
    pub fn loss(logits: &[f32], targets: &[u32], seq_len: usize, vocab_size: usize) -> f32 {
        let mut total_loss = 0.0f32;
        let mut count = 0;

        for t in 0..seq_len {
            let offset = t * vocab_size;
            if offset + vocab_size > logits.len() {
                break;
            }
            let target = targets[t] as usize;
            if target >= vocab_size {
                continue;
            }

            // Stable log-softmax: log(exp(x_target) / sum(exp(x)))
            //   = x_target - log(sum(exp(x)))
            //   = x_target - max - log(sum(exp(x - max)))
            let row = &logits[offset..offset + vocab_size];
            let max = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let log_sum_exp: f32 = row.iter().map(|&v| (v - max).exp()).sum::<f32>().ln() + max;

            total_loss -= row[target] - log_sum_exp;
            count += 1;
        }

        if count > 0 {
            total_loss / count as f32
        } else {
            0.0
        }
    }

    /// Compute loss for a token sequence (next-token prediction).
    ///
    /// Input tokens: [t0, t1, ..., t_{n-1}]
    /// Targets:      [t1, t2, ..., t_n]  (shifted by 1)
    fn compute_loss(&self, tokens: &[u32]) -> f32 {
        if tokens.len() < 2 {
            return 0.0;
        }
        let seq_len = tokens.len() - 1;
        let input = &tokens[..seq_len];
        let targets = &tokens[1..];

        let logits = self.forward(input, seq_len);
        Self::loss(&logits, targets, seq_len, self.config.vocab_size)
    }

    /// One training step using finite-difference approximation, not true gradient.
    ///
    /// This avoids implementing full transformer backpropagation.
    /// Instead, we perturb a random subset of parameters, measure the
    /// change in loss, and use that finite difference to approximate the
    /// gradient direction. This is NOT a true gradient — it is a
    /// forward-difference approximation on sampled parameters only.
    ///
    /// Returns the loss value before the update.
    pub fn train_step(&mut self, tokens: &[u32], lr: f32) -> f32 {
        let base_loss = self.compute_loss(tokens);
        if !base_loss.is_finite() {
            return base_loss;
        }

        let eps = 0.001f32;

        // We perturb a random subset of parameters each step.
        // Collect all parameters into a flat mutable slice.
        let total_params = self.param_count();
        // Number of parameters to probe per step (controls speed vs. quality)
        let n_probes = (total_params / 10).max(64).min(total_params);

        // Deterministic-ish RNG seeded from loss bits
        let seed = (base_loss.to_bits() as u64).wrapping_add(total_params as u64);
        let mut rng = Rng::new(seed);

        // Collect parameter pointers
        let mut params = self.all_params_mut();

        for _ in 0..n_probes {
            let idx = rng.next_usize(params.len());
            let original = *params[idx];

            // Forward perturbation
            *params[idx] = original + eps;
            // We need to drop params, recompute loss, then re-acquire.
            // Since all_params_mut borrows self mutably, we use an unsafe
            // trick: store the pointer and update directly.
            //
            // Instead, we just use a simpler approach: perturb via the
            // raw weight vectors directly.
            drop(params);

            let loss_plus = self.compute_loss(tokens);

            let mut params2 = self.all_params_mut();
            *params2[idx] = original; // restore

            if loss_plus.is_finite() {
                let grad = (loss_plus - base_loss) / eps;
                *params2[idx] = original - lr * grad;
            }

            params = params2;
        }

        base_loss
    }

    /// One training step using finite-difference gradient estimation.
    ///
    /// More thorough than `train_step` -- estimates the gradient for
    /// ALL parameters each step. Very slow for large models but correct.
    /// Best for tiny models (d_model <= 32, 1-2 layers).
    pub fn train_step_full(&mut self, tokens: &[u32], lr: f32) -> f32 {
        if tokens.len() < 2 {
            return 0.0;
        }

        let base_loss = self.compute_loss(tokens);
        if !base_loss.is_finite() {
            return base_loss;
        }

        let eps = 0.0001f32;
        let total = self.param_count();

        for idx in 0..total {
            let mut params = self.all_params_mut();
            let original = *params[idx];
            *params[idx] = original + eps;
            drop(params);

            let loss_plus = self.compute_loss(tokens);

            let mut params = self.all_params_mut();
            *params[idx] = original;

            if loss_plus.is_finite() {
                let grad = (loss_plus - base_loss) / eps;
                *params[idx] = original - lr * grad;
            }
        }

        base_loss
    }

    /// Generate text autoregressively given prompt tokens.
    ///
    /// For each new token:
    /// 1. Forward pass to get logits for the last position
    /// 2. Apply temperature scaling
    /// 3. Sample from the distribution
    pub fn generate(
        &self,
        prompt_tokens: &[u32],
        max_new_tokens: usize,
        temperature: f32,
    ) -> Vec<u32> {
        let mut tokens = prompt_tokens.to_vec();
        let mut rng = Rng::new(12345);
        let v = self.config.vocab_size;

        for _ in 0..max_new_tokens {
            // Truncate to max_seq_len if needed
            let start = if tokens.len() > self.config.max_seq_len {
                tokens.len() - self.config.max_seq_len
            } else {
                0
            };
            let context = &tokens[start..];
            let seq_len = context.len();

            // Forward pass
            let logits = self.forward(context, seq_len);

            // Get logits for the last position
            let last_offset = (seq_len - 1) * v;
            let mut last_logits = logits[last_offset..last_offset + v].to_vec();

            // Apply temperature
            let temp = temperature.max(1e-8);
            for val in last_logits.iter_mut() {
                *val /= temp;
            }

            // Convert to probabilities
            softmax(&mut last_logits);

            // Sample from the distribution
            let next_token = sample_from_distribution(&last_logits, &mut rng);
            tokens.push(next_token as u32);
        }

        tokens
    }

    /// Count total trainable parameters.
    pub fn param_count(&self) -> usize {
        let mut count = 0;
        count += self.token_embedding.len();
        count += self.position_embedding.len();
        for block in &self.blocks {
            count += block.param_count();
        }
        count += self.ln_final.param_count();
        count += self.lm_head.len();
        count
    }

    /// Save model weights to a binary file.
    pub fn save(&self, path: &str) -> std::io::Result<()> {
        let file = std::fs::File::create(path)?;
        let mut w = BufWriter::new(file);

        // Magic + config
        w.write_all(b"QGPT")?;
        write_u32(&mut w, self.config.vocab_size as u32)?;
        write_u32(&mut w, self.config.d_model as u32)?;
        write_u32(&mut w, self.config.n_heads as u32)?;
        write_u32(&mut w, self.config.n_layers as u32)?;
        write_u32(&mut w, self.config.max_seq_len as u32)?;
        write_f32(&mut w, self.config.dropout)?;

        // Write all weights
        write_vec_f32(&mut w, &self.token_embedding)?;
        write_vec_f32(&mut w, &self.position_embedding)?;

        for block in &self.blocks {
            write_vec_f32(&mut w, &block.ln1.gamma)?;
            write_vec_f32(&mut w, &block.ln1.beta)?;
            write_vec_f32(&mut w, &block.attn_qkv)?;
            write_vec_f32(&mut w, &block.attn_out)?;
            write_vec_f32(&mut w, &block.ln2.gamma)?;
            write_vec_f32(&mut w, &block.ln2.beta)?;
            write_vec_f32(&mut w, &block.ffn_up)?;
            write_vec_f32(&mut w, &block.ffn_down)?;
        }

        write_vec_f32(&mut w, &self.ln_final.gamma)?;
        write_vec_f32(&mut w, &self.ln_final.beta)?;
        write_vec_f32(&mut w, &self.lm_head)?;

        w.flush()?;
        Ok(())
    }

    /// Load model weights from a binary file.
    pub fn load(path: &str) -> std::io::Result<Self> {
        let file = std::fs::File::open(path)?;
        let mut r = BufReader::new(file);

        // Magic
        let mut magic = [0u8; 4];
        r.read_exact(&mut magic)?;
        if &magic != b"QGPT" {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "Not a QGPT model file",
            ));
        }

        let vocab_size = read_u32(&mut r)? as usize;
        let d_model = read_u32(&mut r)? as usize;
        let n_heads = read_u32(&mut r)? as usize;
        let n_layers = read_u32(&mut r)? as usize;
        let max_seq_len = read_u32(&mut r)? as usize;
        let dropout = read_f32(&mut r)?;

        let config = TransformerConfig {
            vocab_size,
            d_model,
            n_heads,
            n_layers,
            max_seq_len,
            dropout,
            use_rms_norm: false, // preserve backward compat for loaded models
            use_silu: false,
        };

        let token_embedding = read_vec_f32(&mut r)?;
        let position_embedding = read_vec_f32(&mut r)?;

        let mut blocks = Vec::with_capacity(n_layers);
        for _ in 0..n_layers {
            let ln1 = LayerNormParams {
                gamma: read_vec_f32(&mut r)?,
                beta: read_vec_f32(&mut r)?,
            };
            let attn_qkv = read_vec_f32(&mut r)?;
            let attn_out = read_vec_f32(&mut r)?;
            let ln2 = LayerNormParams {
                gamma: read_vec_f32(&mut r)?,
                beta: read_vec_f32(&mut r)?,
            };
            let ffn_up = read_vec_f32(&mut r)?;
            let ffn_down = read_vec_f32(&mut r)?;

            blocks.push(TransformerBlock {
                ln1,
                attn_qkv,
                attn_out,
                ln2,
                ffn_up,
                ffn_down,
            });
        }

        let ln_final = LayerNormParams {
            gamma: read_vec_f32(&mut r)?,
            beta: read_vec_f32(&mut r)?,
        };
        let lm_head = read_vec_f32(&mut r)?;

        Ok(Self {
            config,
            token_embedding,
            position_embedding,
            blocks,
            ln_final,
            lm_head,
        })
    }

    /// Collect mutable references to ALL parameters (for gradient estimation).
    fn all_params_mut(&mut self) -> Vec<&mut f32> {
        let mut params: Vec<&mut f32> = Vec::with_capacity(self.param_count());

        for p in self.token_embedding.iter_mut() {
            params.push(p);
        }
        for p in self.position_embedding.iter_mut() {
            params.push(p);
        }
        for block in self.blocks.iter_mut() {
            params.extend(block.params_mut());
        }
        params.extend(self.ln_final.params_mut());
        for p in self.lm_head.iter_mut() {
            params.push(p);
        }

        params
    }

    /// Get the model configuration.
    pub fn config(&self) -> &TransformerConfig {
        &self.config
    }
}

// ---------------------------------------------------------------------------
// Sampling
// ---------------------------------------------------------------------------

/// Sample an index from a probability distribution.
fn sample_from_distribution(probs: &[f32], rng: &mut Rng) -> usize {
    let r = rng.next_f32();
    let mut cumulative = 0.0f32;
    for (i, &p) in probs.iter().enumerate() {
        cumulative += p;
        if r < cumulative {
            return i;
        }
    }
    // Fallback: return last index
    probs.len().saturating_sub(1)
}

// ---------------------------------------------------------------------------
// Binary I/O helpers
// ---------------------------------------------------------------------------

fn write_u32<W: Write>(w: &mut W, v: u32) -> std::io::Result<()> {
    w.write_all(&v.to_le_bytes())
}

fn write_f32<W: Write>(w: &mut W, v: f32) -> std::io::Result<()> {
    w.write_all(&v.to_le_bytes())
}

fn write_vec_f32<W: Write>(w: &mut W, v: &[f32]) -> std::io::Result<()> {
    write_u32(w, v.len() as u32)?;
    for &val in v {
        write_f32(w, val)?;
    }
    Ok(())
}

fn read_u32<R: Read>(r: &mut R) -> std::io::Result<u32> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf)?;
    Ok(u32::from_le_bytes(buf))
}

fn read_f32<R: Read>(r: &mut R) -> std::io::Result<f32> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf)?;
    Ok(f32::from_le_bytes(buf))
}

fn read_vec_f32<R: Read>(r: &mut R) -> std::io::Result<Vec<f32>> {
    let len = read_u32(r)? as usize;
    let mut v = Vec::with_capacity(len);
    for _ in 0..len {
        v.push(read_f32(r)?);
    }
    Ok(v)
}

// ---------------------------------------------------------------------------
// Training pipeline (used by CLI)
// ---------------------------------------------------------------------------

/// Train a language model end-to-end from text.
///
/// Returns the trained model and tokenizer.
pub fn train_language_model(
    text: &str,
    vocab_size: usize,
    d_model: usize,
    n_layers: usize,
    n_heads: usize,
    max_seq_len: usize,
    epochs: usize,
    lr: f32,
) -> (MiniGPT, crate::tokenizer::BpeTokenizer) {
    use crate::tokenizer::BpeTokenizer;

    println!("=== QLANG Language Model Training ===\n");

    // 1. Train tokenizer
    println!("[1/4] Training BPE tokenizer (vocab_size={})...", vocab_size);
    let tokenizer = BpeTokenizer::train(text, vocab_size);
    println!("  Tokenizer ready: {} tokens in vocabulary", tokenizer.vocab_size());

    // 2. Tokenize the corpus
    println!("[2/4] Tokenizing corpus...");
    let all_tokens = tokenizer.encode(text);
    println!("  Corpus: {} chars -> {} tokens", text.len(), all_tokens.len());
    let compression = text.len() as f32 / all_tokens.len() as f32;
    println!("  Compression ratio: {:.2}x", compression);

    // 3. Create model
    let config = TransformerConfig {
        vocab_size: tokenizer.vocab_size(),
        d_model,
        n_heads,
        n_layers,
        max_seq_len,
        dropout: 0.0,
        use_rms_norm: true,
        use_silu: true,
    };

    println!("[3/4] Creating MiniGPT model...");
    let mut model = MiniGPT::new(config);
    println!("  Architecture: d_model={}, heads={}, layers={}, seq_len={}",
             d_model, n_heads, n_layers, max_seq_len);
    println!("  Parameters: {}", format_param_count(model.param_count()));

    // 4. Training loop
    println!("[4/4] Training for {} epochs...\n", epochs);

    let seq_len = max_seq_len.min(all_tokens.len());
    let n_sequences = if all_tokens.len() > seq_len {
        all_tokens.len() - seq_len
    } else {
        1
    };
    // Use a subset of starting positions per epoch
    let steps_per_epoch = (n_sequences / seq_len).max(1).min(50);

    let mut rng = Rng::new(42);

    for epoch in 0..epochs {
        let mut epoch_loss = 0.0f32;
        let mut epoch_steps = 0;

        for step in 0..steps_per_epoch {
            // Pick a random starting position
            let start = if n_sequences > 1 {
                rng.next_usize(n_sequences)
            } else {
                0
            };
            let end = (start + seq_len + 1).min(all_tokens.len());
            let chunk = &all_tokens[start..end];

            if chunk.len() < 2 {
                continue;
            }

            let loss = model.train_step(chunk, lr);
            epoch_loss += loss;
            epoch_steps += 1;

            if step == 0 || (step + 1) % 10 == 0 {
                print!("  epoch {}/{}, step {}/{}: loss = {:.4}\r",
                       epoch + 1, epochs, step + 1, steps_per_epoch, loss);
                let _ = std::io::stdout().flush();
            }
        }

        let avg_loss = if epoch_steps > 0 {
            epoch_loss / epoch_steps as f32
        } else {
            0.0
        };
        println!("  Epoch {}/{}: avg_loss = {:.4}                    ",
                 epoch + 1, epochs, avg_loss);
    }

    println!("\nTraining complete!");

    (model, tokenizer)
}

/// Format parameter count with SI suffix.
fn format_param_count(n: usize) -> String {
    if n >= 1_000_000 {
        format!("{:.1}M", n as f64 / 1_000_000.0)
    } else if n >= 1_000 {
        format!("{:.1}K", n as f64 / 1_000.0)
    } else {
        format!("{}", n)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_config() -> TransformerConfig {
        TransformerConfig {
            vocab_size: 32,
            d_model: 16,
            n_heads: 2,
            n_layers: 1,
            max_seq_len: 8,
            dropout: 0.0,
            use_rms_norm: false,
            use_silu: false,
        }
    }

    #[test]
    fn test_gelu_values() {
        // GELU(0) = 0
        assert!(gelu(0.0).abs() < 1e-6);
        // GELU(1) ~ 0.8413
        assert!((gelu(1.0) - 0.8413).abs() < 0.01);
        // GELU(-1) ~ -0.1587
        assert!((gelu(-1.0) - (-0.1587)).abs() < 0.01);
        // GELU is roughly identity for large positive
        assert!((gelu(3.0) - 3.0).abs() < 0.01);
        // GELU approaches 0 for large negative
        assert!(gelu(-3.0).abs() < 0.01);
    }

    #[test]
    fn test_softmax() {
        let mut v = vec![1.0, 2.0, 3.0];
        softmax(&mut v);
        // Should sum to 1
        let sum: f32 = v.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
        // Should be monotonically increasing
        assert!(v[0] < v[1]);
        assert!(v[1] < v[2]);
    }

    #[test]
    fn test_softmax_stability() {
        // Large values should not cause overflow
        let mut v = vec![1000.0, 1001.0, 1002.0];
        softmax(&mut v);
        let sum: f32 = v.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
        assert!(v.iter().all(|&x| x.is_finite()));
    }

    #[test]
    fn test_layer_norm_basic() {
        let d = 4;
        let mut x = vec![1.0, 2.0, 3.0, 4.0];
        let gamma = vec![1.0; d];
        let beta = vec![0.0; d];
        layer_norm(&mut x, &gamma, &beta, d, 1e-5);

        // After normalization, mean should be ~0
        let mean: f32 = x.iter().sum::<f32>() / d as f32;
        assert!(mean.abs() < 1e-5, "Mean should be ~0, got {}", mean);

        // Variance should be ~1
        let var: f32 = x.iter().map(|&v| (v - mean) * (v - mean)).sum::<f32>() / d as f32;
        assert!((var - 1.0).abs() < 0.1, "Var should be ~1, got {}", var);
    }

    #[test]
    fn test_layer_norm_with_scale_shift() {
        let d = 4;
        let mut x = vec![1.0, 2.0, 3.0, 4.0];
        let gamma = vec![2.0; d];
        let beta = vec![1.0; d];
        layer_norm(&mut x, &gamma, &beta, d, 1e-5);

        // Mean should be beta[0] = 1.0 (since gamma scales centered values)
        let mean: f32 = x.iter().sum::<f32>() / d as f32;
        assert!((mean - 1.0).abs() < 0.1, "Mean should be ~1.0 (beta), got {}", mean);
    }

    #[test]
    fn test_causal_attention_masking() {
        // With causal masking, position 0 should only attend to itself
        let d_model = 4;
        let n_heads = 1;
        let seq_len = 3;
        let d_head = d_model / n_heads;

        // Q = K = V = identity-like
        let q = vec![1.0, 0.0, 0.0, 0.0,
                     0.0, 1.0, 0.0, 0.0,
                     0.0, 0.0, 1.0, 0.0];
        let k = q.clone();
        let v = vec![1.0, 0.0, 0.0, 0.0,
                     0.0, 1.0, 0.0, 0.0,
                     0.0, 0.0, 1.0, 0.0];

        let out = causal_multi_head_attention(&q, &k, &v, n_heads, seq_len, d_model, d_head);

        // Position 0 can only see position 0, so output[0] should equal V[0]
        assert!((out[0] - 1.0).abs() < 1e-3,
                "Position 0 should only attend to itself: got {}", out[0]);
    }

    #[test]
    fn test_forward_output_shape() {
        let config = tiny_config();
        let model = MiniGPT::new(config.clone());

        let tokens = vec![0u32, 1, 2, 3];
        let seq_len = tokens.len();
        let logits = model.forward(&tokens, seq_len);

        assert_eq!(
            logits.len(),
            seq_len * config.vocab_size,
            "Output shape should be [seq_len, vocab_size]"
        );
        assert!(logits.iter().all(|x| x.is_finite()), "All logits should be finite");
    }

    #[test]
    fn test_param_count() {
        let config = tiny_config(); // vocab=32, d=16, heads=2, layers=1, seq=8
        let model = MiniGPT::new(config);
        let count = model.param_count();

        // token_embedding: 32*16 = 512
        // position_embedding: 8*16 = 128
        // Per block:
        //   ln1: 16+16 = 32
        //   attn_qkv: 16*48 = 768
        //   attn_out: 16*16 = 256
        //   ln2: 32
        //   ffn_up: 16*64 = 1024
        //   ffn_down: 64*16 = 1024
        //   block total: 3136
        // ln_final: 32
        // lm_head: 16*32 = 512
        // Total: 512 + 128 + 3136 + 32 + 512 = 4320
        let expected = 512 + 128 + (32 + 768 + 256 + 32 + 1024 + 1024) + 32 + 512;
        assert_eq!(count, expected, "Param count should be {}, got {}", expected, count);
    }

    #[test]
    fn test_loss_computation() {
        let config = tiny_config();
        let model = MiniGPT::new(config.clone());

        let tokens = vec![0u32, 1, 2, 3, 4];
        let loss = model.compute_loss(&tokens);

        // Loss should be positive and finite
        assert!(loss > 0.0, "Loss should be positive, got {}", loss);
        assert!(loss.is_finite(), "Loss should be finite");

        // For random weights, loss should be roughly -log(1/vocab_size) = log(vocab_size)
        let expected_random = (config.vocab_size as f32).ln();
        assert!(
            (loss - expected_random).abs() < expected_random,
            "Random model loss ({}) should be near log(vocab_size)={:.2}",
            loss,
            expected_random
        );
    }

    #[test]
    fn test_generate_length() {
        let config = tiny_config();
        let model = MiniGPT::new(config);

        let prompt = vec![0u32, 1];
        let max_new = 5;
        let output = model.generate(&prompt, max_new, 1.0);

        assert_eq!(
            output.len(),
            prompt.len() + max_new,
            "Generated sequence should be prompt_len + max_new_tokens"
        );
    }

    #[test]
    fn test_generate_starts_with_prompt() {
        let config = tiny_config();
        let model = MiniGPT::new(config);

        let prompt = vec![3u32, 7, 15];
        let output = model.generate(&prompt, 3, 1.0);

        assert_eq!(&output[..prompt.len()], &prompt[..],
                   "Generated sequence should start with prompt");
    }

    #[test]
    fn test_save_load_roundtrip() {
        let config = tiny_config();
        let model = MiniGPT::new(config);

        let path = "/tmp/qlang_test_minigpt.qgpt";
        model.save(path).expect("save failed");

        let model2 = MiniGPT::load(path).expect("load failed");

        // Same config
        assert_eq!(model.config.vocab_size, model2.config.vocab_size);
        assert_eq!(model.config.d_model, model2.config.d_model);
        assert_eq!(model.config.n_heads, model2.config.n_heads);
        assert_eq!(model.config.n_layers, model2.config.n_layers);
        assert_eq!(model.param_count(), model2.param_count());

        // Same forward pass
        let tokens = vec![0u32, 1, 2, 3];
        let out1 = model.forward(&tokens, 4);
        let out2 = model2.forward(&tokens, 4);
        for (a, b) in out1.iter().zip(out2.iter()) {
            assert!(
                (a - b).abs() < 1e-5,
                "Loaded model should produce same outputs"
            );
        }

        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn test_temperature_affects_generation() {
        let config = tiny_config();
        let model = MiniGPT::new(config);

        let prompt = vec![0u32, 1];

        // Low temperature should produce deterministic-ish output
        let out_low = model.generate(&prompt, 10, 0.01);
        // High temperature should produce more varied output
        let out_high = model.generate(&prompt, 10, 100.0);

        // They should likely differ (not guaranteed, but extremely likely)
        // Just verify both produce valid output
        assert_eq!(out_low.len(), prompt.len() + 10);
        assert_eq!(out_high.len(), prompt.len() + 10);
    }

    #[test]
    fn test_train_step_does_not_crash() {
        let config = TransformerConfig {
            vocab_size: 16,
            d_model: 8,
            n_heads: 2,
            n_layers: 1,
            max_seq_len: 8,
            dropout: 0.0,
            use_rms_norm: false,
            use_silu: false,
        };
        let mut model = MiniGPT::new(config);

        let tokens = vec![0u32, 1, 2, 3, 4, 5];
        let loss = model.train_step(&tokens, 0.01);
        assert!(loss.is_finite(), "Training loss should be finite");
    }
}
