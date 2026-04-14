//! Transformer Encoder in QLANG.
//!
//! Demonstrates a complete Transformer encoder stack:
//! - Positional encoding
//! - Multi-head self-attention
//! - Feed-forward network
//! - Layer normalization
//! - Residual connections
//!
//! Architecture: 4-token sequence, d_model=32, 2 heads, 1 layer

fn main() {
    println!("=== QLANG Transformer Encoder ===\n");

    use qlang_runtime::autograd::Tape;
    use qlang_runtime::transformer::*;

    let seq_len = 8;
    let d_model = 32;
    let d_ff = 64;
    let n_heads = 4;
    let d_k = d_model / n_heads; // 8

    println!("[1] Architecture:");
    println!("  Sequence length: {seq_len}");
    println!("  Model dimension: {d_model}");
    println!("  Feed-forward:    {d_ff}");
    println!("  Attention heads: {n_heads}");
    println!("  Head dimension:  {d_k}");
    println!("  Parameters per layer: ~{}", 4 * d_model * d_model + 2 * d_model * d_ff);

    // ─── Create input (token embeddings + positional encoding) ───
    println!("\n[2] Creating input...");

    let mut tape = Tape::new();

    // Simulated token embeddings (random)
    let token_data: Vec<f32> = (0..seq_len * d_model)
        .map(|i| (i as f32 * 0.37).sin() * 0.5)
        .collect();
    let tokens = tape.variable(token_data, vec![seq_len, d_model]);

    // Add positional encoding
    let pe = positional_encoding(seq_len, d_model);
    let pe_id = tape.variable(pe, vec![seq_len, d_model]);
    let input = tape.add(tokens, pe_id);

    println!("  Input shape: [{seq_len}, {d_model}]");

    // ─── Build Transformer layer ───
    println!("\n[3] Building Transformer layer...");

    let make_weights = |tape: &mut Tape, r: usize, c: usize, seed: f32| -> usize {
        let scale = (2.0 / (r + c) as f64).sqrt() as f32;
        let data: Vec<f32> = (0..r * c).map(|i| (i as f32 * seed).sin() * scale).collect();
        tape.variable(data, vec![r, c])
    };

    // Attention weights
    let w_q = make_weights(&mut tape, d_model, d_model, 0.371);
    let w_k = make_weights(&mut tape, d_model, d_model, 0.529);
    let w_v = make_weights(&mut tape, d_model, d_model, 0.691);
    let w_o = make_weights(&mut tape, d_model, d_model, 0.823);

    // FFN weights
    let ff_w1 = make_weights(&mut tape, d_model, d_ff, 0.457);
    let ff_w2 = make_weights(&mut tape, d_ff, d_model, 0.613);

    let total_params = 4 * d_model * d_model + d_model * d_ff + d_ff * d_model;
    println!("  Total parameters: {} ({:.1} KB)", total_params, total_params as f64 * 4.0 / 1024.0);

    // ─── Forward pass ───
    println!("\n[4] Forward pass (1 Transformer layer)...");
    let start = std::time::Instant::now();

    let output = transformer_encoder_layer(
        &mut tape, input,
        w_q, w_k, w_v, w_o,
        ff_w1, ff_w2,
        n_heads, d_model,
    );

    let forward_time = start.elapsed();
    let result = tape.value(output);

    println!("  Output shape: [{}, {}]", seq_len, d_model);
    println!("  Forward time: {:?}", forward_time);
    println!("  Values finite: {}", result.iter().all(|x| x.is_finite()));

    // Show first token's representation
    println!("\n  Token 0 output (first 8 dims):");
    for i in 0..8 {
        let bar_len = ((result[i].abs() * 20.0).min(30.0)) as usize;
        let bar = if result[i] >= 0.0 {
            format!("{:>7.4} {}", result[i], "█".repeat(bar_len))
        } else {
            format!("{:>7.4} {}", result[i], "▒".repeat(bar_len))
        };
        println!("    [{i}] {bar}");
    }

    // ─── Multi-layer stack ───
    println!("\n[5] Stacking 4 Transformer layers...");
    let n_layers = 4;
    let start = std::time::Instant::now();

    let mut tape2 = Tape::new();
    let token_data2: Vec<f32> = (0..seq_len * d_model)
        .map(|i| (i as f32 * 0.37).sin() * 0.5)
        .collect();
    let pe2 = positional_encoding(seq_len, d_model);
    let tokens2 = tape2.variable(token_data2, vec![seq_len, d_model]);
    let pe2_id = tape2.variable(pe2, vec![seq_len, d_model]);
    let mut current = tape2.add(tokens2, pe2_id);

    let mut total_layer_params = 0;
    for layer in 0..n_layers {
        let seed_offset = layer as f32 * 0.1;
        let wq = make_weights(&mut tape2, d_model, d_model, 0.371 + seed_offset);
        let wk = make_weights(&mut tape2, d_model, d_model, 0.529 + seed_offset);
        let wv = make_weights(&mut tape2, d_model, d_model, 0.691 + seed_offset);
        let wo = make_weights(&mut tape2, d_model, d_model, 0.823 + seed_offset);
        let fw1 = make_weights(&mut tape2, d_model, d_ff, 0.457 + seed_offset);
        let fw2 = make_weights(&mut tape2, d_ff, d_model, 0.613 + seed_offset);
        total_layer_params += 4 * d_model * d_model + d_model * d_ff + d_ff * d_model;

        current = transformer_encoder_layer(
            &mut tape2, current,
            wq, wk, wv, wo,
            fw1, fw2,
            n_heads, d_model,
        );
    }

    let stack_time = start.elapsed();
    let stack_result = tape2.value(current);

    println!("  {} layers × {} params = {} total params ({:.1} KB)",
        n_layers, total_params, total_layer_params, total_layer_params as f64 * 4.0 / 1024.0);
    println!("  Forward time: {:?}", stack_time);
    println!("  Values finite: {}", stack_result.iter().all(|x| x.is_finite()));

    // ─── IGQK Compression ───
    println!("\n[6] IGQK Ternary Compression of attention weights...");

    let w_q_data = tape.value(w_q).to_vec();
    let mean_abs: f32 = w_q_data.iter().map(|x| x.abs()).sum::<f32>() / w_q_data.len() as f32;
    let threshold = mean_abs * 0.7;
    let ternary: Vec<f32> = w_q_data.iter().map(|&x| {
        if x > threshold { 1.0 } else if x < -threshold { -1.0 } else { 0.0 }
    }).collect();

    let pos = ternary.iter().filter(|&&w| w == 1.0).count();
    let neg = ternary.iter().filter(|&&w| w == -1.0).count();
    let zero = ternary.iter().filter(|&&w| w == 0.0).count();
    let total = ternary.len();

    println!("  W_Q: [{d_model}×{d_model}] = {} params", d_model * d_model);
    println!("  +1: {} ({:.0}%), 0: {} ({:.0}%), -1: {} ({:.0}%)",
        pos, pos as f64 / total as f64 * 100.0,
        zero, zero as f64 / total as f64 * 100.0,
        neg, neg as f64 / total as f64 * 100.0);
    println!("  Compression: f32 → ternary = 16x (with 2-bit packing)");
    println!("  All 4 attention matrices: {} KB → {} KB",
        4 * d_model * d_model * 4 / 1024,
        4 * d_model * d_model / 4 / 1024);

    // ─── QLANG text ───
    println!("\n[7] QLANG text representation:");
    let qlang = format!(r#"graph transformer_encoder {{
  input tokens: f32[{seq_len}, {d_model}]
  input W_Q: f32[{d_model}, {d_model}]
  input W_K: f32[{d_model}, {d_model}]
  input W_V: f32[{d_model}, {d_model}]
  input W_O: f32[{d_model}, {d_model}]
  input FF_W1: f32[{d_model}, {d_ff}]
  input FF_W2: f32[{d_ff}, {d_model}]

  // Self-attention
  node Q = matmul(tokens, W_Q)
  node K = matmul(tokens, W_K)
  node V = matmul(tokens, W_V)
  // attention(Q, K, V) would be native op
  node attn_out = matmul(Q, W_O)
  node residual1 = add(tokens, attn_out)

  // Feed-forward
  node ff_h = matmul(residual1, FF_W1)
  node ff_act = relu(ff_h)
  node ff_out = matmul(ff_act, FF_W2)
  node residual2 = add(residual1, ff_out)

  // IGQK compression
  node comp_Q = to_ternary(W_Q) @proof theorem_5_2
  node comp_K = to_ternary(W_K) @proof theorem_5_2

  output encoded = residual2
  output compressed_Q = comp_Q
}}"#);

    println!("{qlang}");

    println!("\n=== Transformer Encoder complete. ===");
}
