//! GPU-accelerated Mamba training via wgpu matmul.
//!
//! Offloads the expensive matrix multiplications in the Mamba forward and
//! backward passes to the GPU while keeping the sequential recurrence and
//! element-wise ops (tanh, sigmoid) on the CPU.
//!
//! Falls back transparently to the CPU path when no GPU is available.

use crate::mamba_train::TrainableMamba;

#[cfg(feature = "gpu")]
use crate::gpu_compute::gpu_compute::{get_gpu, GpuContext};

// ---------------------------------------------------------------------------
// Forward pass — GPU-accelerated
// ---------------------------------------------------------------------------

/// GPU-accelerated forward pass for a single Mamba layer.
///
/// Pre-computes the input projections as batched matmuls on the GPU:
///   X_proj   = X @ W_x        [seq_len, d_hidden]
///   Gate_x   = X @ W_gate_x   [seq_len, d_hidden]
///
/// The hidden-state recurrence is inherently sequential and stays on the CPU,
/// but each timestep reuses the pre-computed projections instead of
/// re-computing the inner products.
pub fn forward_gpu(
    layer: &TrainableMamba,
    input: &[f32],
    seq_len: usize,
) -> (Vec<f32>, Vec<Vec<f32>>) {
    #[cfg(feature = "gpu")]
    if let Some(gpu) = get_gpu() {
        return forward_gpu_impl(gpu, layer, input, seq_len);
    }
    // Fallback: pure CPU
    layer.forward(input, seq_len)
}

#[cfg(feature = "gpu")]
fn forward_gpu_impl(
    gpu: &GpuContext,
    layer: &TrainableMamba,
    input: &[f32],
    seq_len: usize,
) -> (Vec<f32>, Vec<Vec<f32>>) {
    let d = layer.d_model;
    let dh = layer.d_hidden;

    // --- Batched GPU matmuls for input projections ---------------------------
    // X is [seq_len, d], W_x is [d, dh] (stored row-major)
    let x_proj = gpu.matmul(input, &layer.w_x, seq_len, dh, d);

    // Gate input part: X @ W_gate_x where W_gate_x = W_gate[0..d, :] is [d, dh]
    let w_gate_x = &layer.w_gate[..d * dh];
    let gate_x_proj = gpu.matmul(input, w_gate_x, seq_len, dh, d);

    // --- Sequential recurrence (CPU) ----------------------------------------
    let mut h = vec![0.0f32; dh];
    let mut output = vec![0.0f32; seq_len * d];
    let mut hidden_states = Vec::with_capacity(seq_len + 1);
    hidden_states.push(h.clone());

    for t in 0..seq_len {
        // Candidate: tanh(x_proj[t] + W_h @ h + b_h)
        let mut candidate = vec![0.0f32; dh];
        for j in 0..dh {
            let mut sum = layer.b_h[j] + x_proj[t * dh + j];
            for k in 0..dh {
                sum += h[k] * layer.w_h[k * dh + j];
            }
            candidate[j] = sum.tanh();
        }

        // Gate: sigmoid(gate_x_proj[t] + W_gate_h @ h + b_gate)
        let mut gate = vec![0.0f32; dh];
        for j in 0..dh {
            let mut sum = layer.b_gate[j] + gate_x_proj[t * dh + j];
            for k in 0..dh {
                sum += h[k] * layer.w_gate[(d + k) * dh + j];
            }
            gate[j] = 1.0 / (1.0 + (-sum).exp());
        }

        // Update hidden state
        for j in 0..dh {
            h[j] = gate[j] * h[j] + (1.0 - gate[j]) * candidate[j];
        }
        hidden_states.push(h.clone());

        // Output projection: W_out @ h + residual
        for j in 0..d {
            let mut sum = 0.0f32;
            for k in 0..dh {
                sum += h[k] * layer.w_out[k * d + j];
            }
            output[t * d + j] = input[t * d + j] + sum;
        }
    }

    (output, hidden_states)
}

// ---------------------------------------------------------------------------
// Backward pass — GPU-accelerated gradient accumulation
// ---------------------------------------------------------------------------

/// GPU-accelerated backward pass with weight update.
///
/// The per-timestep BPTT loop stays on the CPU (sequential dependency), but
/// the large gradient outer-products are accumulated into matrices and then
/// a single GPU matmul computes the weight gradients:
///   dW_out = H^T @ dOut       [dh, d]
///   dW_x   = X^T @ dCand_pre  [d, dh]
///   etc.
pub fn backward_and_update_gpu(
    layer: &mut TrainableMamba,
    input: &[f32],
    hidden_states: &[Vec<f32>],
    d_output: &[f32],
    seq_len: usize,
    lr: f32,
) {
    #[cfg(feature = "gpu")]
    if let Some(gpu) = get_gpu() {
        backward_gpu_impl(gpu, layer, input, hidden_states, d_output, seq_len, lr);
        return;
    }
    // Fallback: pure CPU
    layer.backward_and_update(input, hidden_states, d_output, seq_len, lr);
}

#[cfg(feature = "gpu")]
fn backward_gpu_impl(
    gpu: &GpuContext,
    layer: &mut TrainableMamba,
    input: &[f32],
    hidden_states: &[Vec<f32>],
    d_output: &[f32],
    seq_len: usize,
    lr: f32,
) {
    let d = layer.d_model;
    let dh = layer.d_hidden;

    // Collect per-timestep quantities for batched GPU matmul at the end.
    // d_cand_pre[t] and d_gate_pre[t] are [dh]-vectors; we stack them into matrices.
    let mut d_cand_pre_mat = vec![0.0f32; seq_len * dh]; // [seq_len, dh]
    let mut d_gate_pre_mat = vec![0.0f32; seq_len * dh]; // [seq_len, dh]
    let mut h_prev_mat = vec![0.0f32; seq_len * dh]; // [seq_len, dh]
    let mut h_curr_mat = vec![0.0f32; seq_len * dh]; // [seq_len, dh]
    let mut d_out_mat = vec![0.0f32; seq_len * d]; // copy for GPU matmul

    let mut db_h = vec![0.0f32; dh];
    let mut db_gate = vec![0.0f32; dh];
    let mut dh_next = vec![0.0f32; dh];

    // --- Sequential BPTT (CPU) — collect gradient signals --------------------
    for t in (0..seq_len).rev() {
        let x_t = &input[t * d..(t + 1) * d];
        let h_prev = &hidden_states[t];
        let h_curr = &hidden_states[t + 1];

        // Store for batched matmul later
        h_prev_mat[t * dh..(t + 1) * dh].copy_from_slice(h_prev);
        h_curr_mat[t * dh..(t + 1) * dh].copy_from_slice(h_curr);
        d_out_mat[t * d..(t + 1) * d].copy_from_slice(&d_output[t * d..(t + 1) * d]);

        // d_h from output projection and recurrence
        let mut d_h = dh_next.clone();
        for j in 0..d {
            let d_out_j = d_output[t * d + j];
            for k in 0..dh {
                d_h[k] += d_out_j * layer.w_out[k * d + j];
            }
        }

        // Recompute gate and candidate for this timestep
        let mut candidate = vec![0.0f32; dh];
        let mut gate = vec![0.0f32; dh];
        for j in 0..dh {
            let mut sum_c = layer.b_h[j];
            for k in 0..d {
                sum_c += x_t[k] * layer.w_x[k * dh + j];
            }
            for k in 0..dh {
                sum_c += h_prev[k] * layer.w_h[k * dh + j];
            }
            candidate[j] = sum_c.tanh();

            let mut sum_g = layer.b_gate[j];
            for k in 0..d {
                sum_g += x_t[k] * layer.w_gate[k * dh + j];
            }
            for k in 0..dh {
                sum_g += h_prev[k] * layer.w_gate[(d + k) * dh + j];
            }
            gate[j] = 1.0 / (1.0 + (-sum_g).exp());
        }

        for j in 0..dh {
            let d_gate_j = d_h[j] * (h_prev[j] - candidate[j]);
            let d_candidate_j = d_h[j] * (1.0 - gate[j]);
            let d_gate_pre = d_gate_j * gate[j] * (1.0 - gate[j]);
            let d_cand_pre = d_candidate_j * (1.0 - candidate[j] * candidate[j]);

            d_cand_pre_mat[t * dh + j] = d_cand_pre;
            d_gate_pre_mat[t * dh + j] = d_gate_pre;
            db_h[j] += d_cand_pre;
            db_gate[j] += d_gate_pre;

            // Gradient to h_prev for next iteration
            dh_next[j] = d_h[j] * gate[j];
            for k in 0..dh {
                dh_next[k] += d_cand_pre * layer.w_h[k * dh + j];
                dh_next[k] += d_gate_pre * layer.w_gate[(d + k) * dh + j];
            }
        }

        for v in &mut dh_next {
            *v = v.max(-1.0).min(1.0);
        }
    }

    // --- Batched GPU matmuls for weight gradients ----------------------------
    // dW_out = H_curr^T @ dOut   : [dh, seq_len] @ [seq_len, d] = [dh, d]
    let dw_out = gpu.matmul_at_b(&h_curr_mat, &d_out_mat, dh, d, seq_len);

    // dW_x = X^T @ dCand_pre    : [d, seq_len] @ [seq_len, dh] = [d, dh]
    let dw_x = gpu.matmul_at_b(input, &d_cand_pre_mat, d, dh, seq_len);

    // dW_h = H_prev^T @ dCand_pre : [dh, seq_len] @ [seq_len, dh] = [dh, dh]
    let dw_h = gpu.matmul_at_b(&h_prev_mat, &d_cand_pre_mat, dh, dh, seq_len);

    // dW_gate_x = X^T @ dGate_pre : [d, seq_len] @ [seq_len, dh] = [d, dh]
    let dw_gate_x = gpu.matmul_at_b(input, &d_gate_pre_mat, d, dh, seq_len);

    // dW_gate_h = H_prev^T @ dGate_pre : [dh, seq_len] @ [seq_len, dh] = [dh, dh]
    let dw_gate_h = gpu.matmul_at_b(&h_prev_mat, &d_gate_pre_mat, dh, dh, seq_len);

    // --- Apply gradients with clipping ---------------------------------------
    let inv = lr / seq_len as f32;

    for i in 0..layer.w_x.len() {
        layer.w_x[i] -= inv * dw_x[i].max(-1.0).min(1.0);
    }
    for i in 0..layer.w_h.len() {
        layer.w_h[i] -= inv * dw_h[i].max(-1.0).min(1.0);
    }
    for i in 0..layer.w_out.len() {
        layer.w_out[i] -= inv * dw_out[i].max(-1.0).min(1.0);
    }

    // W_gate is [(d+dh), dh] — first d rows from gate_x, next dh rows from gate_h
    for i in 0..d * dh {
        layer.w_gate[i] -= inv * dw_gate_x[i].max(-1.0).min(1.0);
    }
    for i in 0..dh * dh {
        layer.w_gate[d * dh + i] -= inv * dw_gate_h[i].max(-1.0).min(1.0);
    }

    for i in 0..dh {
        layer.b_h[i] -= inv * db_h[i];
        layer.b_gate[i] -= inv * db_gate[i];
    }
}

// ---------------------------------------------------------------------------
// GPU adapter name helper
// ---------------------------------------------------------------------------

/// Returns the GPU adapter name if available, or None.
pub fn gpu_adapter_name() -> Option<String> {
    #[cfg(feature = "gpu")]
    if let Some(gpu) = get_gpu() {
        return Some(gpu.adapter_name().to_string());
    }
    None
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mamba_train::TrainableMamba;

    #[test]
    fn forward_gpu_matches_cpu() {
        let layer = TrainableMamba::new(16, 32, 0.37);
        let input = vec![0.1f32; 4 * 16]; // 4 timesteps, d=16

        let (cpu_out, cpu_hs) = layer.forward(&input, 4);
        let (gpu_out, gpu_hs) = forward_gpu(&layer, &input, 4);

        assert_eq!(cpu_out.len(), gpu_out.len());
        for (i, (a, b)) in cpu_out.iter().zip(gpu_out.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-3,
                "forward mismatch at {i}: cpu={a}, gpu={b}"
            );
        }
        assert_eq!(cpu_hs.len(), gpu_hs.len());
    }

    #[test]
    fn backward_gpu_reduces_loss() {
        let mut layer = TrainableMamba::new(16, 32, 0.37);
        let input = vec![0.1f32; 4 * 16];

        // Forward
        let (out1, hs1) = forward_gpu(&layer, &input, 4);
        let loss1: f32 = out1.iter().map(|x| x * x).sum();

        // Backward with gradient = 2*output (MSE gradient)
        let d_out: Vec<f32> = out1.iter().map(|x| 2.0 * x).collect();
        backward_and_update_gpu(&mut layer, &input, &hs1, &d_out, 4, 0.01);

        // Forward again
        let (out2, _) = forward_gpu(&layer, &input, 4);
        let loss2: f32 = out2.iter().map(|x| x * x).sum();

        assert!(
            loss2 < loss1,
            "GPU backward must reduce loss: {loss1} -> {loss2}"
        );
    }
}
