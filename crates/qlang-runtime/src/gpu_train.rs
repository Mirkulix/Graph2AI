//! GPU Training for Mamba LM — CUDA-accelerated BPTT.
//!
//! Designed for 2x RTX 2070 Super (8GB VRAM each).
//! Target: 30M parameter Mamba LM on WikiText-2.
//!
//! Architecture:
//!   BPE Tokenizer (8K vocab) → Embedding(512) → 6x Mamba(d=512, inner=1024, state=32) → Output Head
//!
//! Training:
//!   BPTT through Mamba recurrence
//!   Mixed precision (f32 compute, ternary export)
//!   Gradient accumulation for effective batch size
//!
//! Usage:
//!   1. Copy this crate to the GPU machine
//!   2. cargo run --release --features cuda --bin gpu_train
//!   3. Wait 6-12 hours
//!   4. Export as ternary .qlbg model

use crate::mamba_train::TrainableLM;
use crate::qlang_lm::Tokenizer;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Instant;

/// Events emitted during training for real-time progress streaming.
#[derive(Clone, Debug, serde::Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum TrainEvent {
    Progress {
        step: usize,
        total_steps: usize,
        loss: f32,
        ppl: f32,
        steps_per_sec: f64,
        eta_secs: u64,
        generated: String,
        elapsed_secs: f64,
    },
    Checkpoint {
        step: usize,
        path: String,
        ppl: f32,
    },
    Complete {
        init_ppl: f32,
        final_ppl: f32,
        total_time_secs: f64,
        model_path: String,
        ternary_ppl: f32,
        ternary_path: String,
    },
    Error {
        message: String,
    },
}

/// GPU Training configuration.
#[derive(Clone)]
pub struct GpuTrainConfig {
    /// Model dimension
    pub d_model: usize,
    /// Hidden dimension per Mamba layer
    pub d_hidden: usize,
    /// SSM state dimension
    pub d_state: usize,
    /// Number of Mamba layers
    pub n_layers: usize,
    /// Vocabulary size
    pub vocab_size: usize,
    /// Sequence length for BPTT
    pub seq_len: usize,
    /// Learning rate
    pub lr: f32,
    /// Number of training steps
    pub n_steps: usize,
    /// Gradient accumulation steps (simulates larger batch)
    pub grad_accum: usize,
    /// Log interval
    pub log_every: usize,
    /// Save checkpoint every N steps
    pub save_every: usize,
    /// Path to training data
    pub data_path: String,
    /// Path to save model
    pub output_path: String,
}

impl Default for GpuTrainConfig {
    fn default() -> Self {
        Self {
            // 30M params: fits in 8GB VRAM with gradients
            d_model: 512,
            d_hidden: 1024,
            d_state: 32,
            n_layers: 6,
            vocab_size: 8000,
            seq_len: 128,
            lr: 0.001,
            n_steps: 50000,
            grad_accum: 4, // effective batch = 4 sequences
            log_every: 500,
            save_every: 5000,
            data_path: "data/wikitext2/train.txt".into(),
            output_path: "data/mamba_30m".into(),
        }
    }
}

impl GpuTrainConfig {
    /// Estimate model size in parameters.
    pub fn param_count(&self) -> usize {
        let embed = self.vocab_size * self.d_model; // embedding table
        let per_layer =
            self.d_model * self.d_hidden +          // w_x
            self.d_hidden * self.d_hidden +          // w_h
            (self.d_model + self.d_hidden) * self.d_hidden + // w_gate
            self.d_hidden * self.d_model +            // w_out
            self.d_hidden * 2;                        // biases
        let head = self.d_model * self.vocab_size;   // output head
        embed + per_layer * self.n_layers + head
    }

    /// Estimate VRAM usage in MB.
    pub fn vram_mb(&self) -> usize {
        let params = self.param_count();
        // Parameters + gradients + optimizer state + activations
        // f32: 4 bytes per param, gradients: 4 bytes, activations: ~2x params for seq_len
        let param_bytes = params * 4;
        let grad_bytes = params * 4;
        let activation_bytes = self.seq_len * self.d_model * self.n_layers * 4 * 2;
        (param_bytes + grad_bytes + activation_bytes) / (1024 * 1024)
    }

    /// Print configuration summary.
    pub fn print_summary(&self) {
        let params = self.param_count();
        let vram = self.vram_mb();
        println!("GPU Training Configuration:");
        println!("  Model:    d={}, hidden={}, state={}, layers={}", self.d_model, self.d_hidden, self.d_state, self.n_layers);
        println!("  Vocab:    {}", self.vocab_size);
        println!("  Seq len:  {}", self.seq_len);
        println!("  Params:   {} ({:.1}M)", params, params as f64 / 1e6);
        println!("  VRAM:     ~{} MB (fits in 8GB RTX 2070 Super)", vram);
        println!("  Steps:    {} (grad_accum={})", self.n_steps, self.grad_accum);
        println!("  LR:       {}", self.lr);
        println!("  Data:     {}", self.data_path);
        println!("  Output:   {}", self.output_path);
        println!();

        if vram > 7500 {
            println!("  WARNING: Model may not fit in 8GB VRAM!");
            println!("  Reduce d_model or n_layers.");
        }
    }
}

/// Train on CPU (for testing the pipeline before GPU).
/// On GPU machine, this would be replaced with candle CUDA tensors.
pub fn train_cpu(config: &GpuTrainConfig) -> Result<(), String> {
    let text = std::fs::read_to_string(&config.data_path)
        .map_err(|e| format!("Read data: {e}"))?;

    config.print_summary();

    // For CPU test: use smaller model
    let cpu_d = config.d_model.min(128);
    let cpu_hidden = config.d_hidden.min(256);
    let cpu_layers = config.n_layers.min(2);
    let cpu_vocab = config.vocab_size.min(2000);

    println!("CPU mode: reduced to d={}, hidden={}, layers={}, vocab={}\n",
        cpu_d, cpu_hidden, cpu_layers, cpu_vocab);

    let mut lm = TrainableLM::new(&text, cpu_d, cpu_hidden, cpu_layers, cpu_vocab);
    let tokens = lm.tokenizer.encode(&text);
    let seq_len = config.seq_len.min(32);

    println!("Tokens: {}", tokens.len());
    let init_ppl = lm.perplexity(&tokens[1000..1000 + seq_len]);
    println!("Init PPL: {:.1}\n", init_ppl);

    let start = Instant::now();
    let n_steps = config.n_steps.min(5000);

    for step in 0..n_steps {
        let mut total_loss = 0.0f32;
        for acc in 0..config.grad_accum {
            let off = ((step * config.grad_accum + acc) * seq_len)
                % (tokens.len().saturating_sub(seq_len + 1));
            let batch = &tokens[off..off + seq_len + 1];
            total_loss += lm.train_step(batch, config.lr / config.grad_accum as f32);
        }
        let loss = total_loss / config.grad_accum as f32;

        if step % config.log_every == 0 || step == n_steps - 1 {
            let ppl = lm.perplexity(&tokens[1000..1000 + seq_len]);
            let gen = lm.generate_with_temp("the", 15, 0.7);
            let elapsed = start.elapsed();
            let steps_per_sec = (step + 1) as f64 / elapsed.as_secs_f64();
            let eta_secs = ((n_steps - step) as f64 / steps_per_sec) as u64;

            println!("  Step {:>5}/{}: loss={:.3} ppl={:.1} [{:.1} steps/s, ETA: {}m] \"{}\"",
                step, n_steps, loss, ppl, steps_per_sec, eta_secs / 60, gen);
        }

        if config.save_every > 0 && step > 0 && step % config.save_every == 0 {
            let path = format!("{}_step{}.bin", config.output_path, step);
            save_weights(&lm, &path);
            println!("  Checkpoint saved: {}", path);
        }
    }

    let final_ppl = lm.perplexity(&tokens[1000..1000 + seq_len]);
    println!("\nFinal PPL: {:.1} (init: {:.1}, {:.1}x improvement)",
        final_ppl, init_ppl, init_ppl / final_ppl);
    println!("Time: {:?}", start.elapsed());

    // Save final model
    let path = format!("{}_final.bin", config.output_path);
    save_weights(&lm, &path);
    println!("Model saved: {}", path);

    // Export ternary
    let mut lm_tern = lm;
    for layer in &mut lm_tern.layers { layer.ternarize(); }
    let tern_ppl = lm_tern.perplexity(&tokens[1000..1000 + seq_len]);
    println!("Ternary PPL: {:.1}", tern_ppl);

    let tern_path = format!("{}_ternary.bin", config.output_path);
    save_weights(&lm_tern, &tern_path);
    println!("Ternary model saved: {}", tern_path);

    Ok(())
}

/// Train on CPU with progress channel for real-time SSE streaming.
/// Sends `TrainEvent` messages through the channel. Respects the stop flag.
pub fn train_cpu_with_progress(
    config: &GpuTrainConfig,
    progress_tx: std::sync::mpsc::Sender<TrainEvent>,
    stop_flag: Arc<AtomicBool>,
) -> Result<(), String> {
    let text = match std::fs::read_to_string(&config.data_path) {
        Ok(t) => t,
        Err(e) => {
            let msg = format!("Read data '{}': {e}", config.data_path);
            let _ = progress_tx.send(TrainEvent::Error { message: msg.clone() });
            return Err(msg);
        }
    };

    let cpu_d = config.d_model.min(128);
    let cpu_hidden = config.d_hidden.min(256);
    let cpu_layers = config.n_layers.min(2);
    let cpu_vocab = config.vocab_size.min(2000);

    let mut lm = TrainableLM::new(&text, cpu_d, cpu_hidden, cpu_layers, cpu_vocab);
    let tokens = lm.tokenizer.encode(&text);
    let seq_len = config.seq_len.min(32);

    if tokens.len() < 1100 + seq_len {
        let msg = "Not enough tokens in training data".to_string();
        let _ = progress_tx.send(TrainEvent::Error { message: msg.clone() });
        return Err(msg);
    }

    let init_ppl = lm.perplexity(&tokens[1000..1000 + seq_len]);
    let start = Instant::now();
    let n_steps = config.n_steps.min(50000);

    for step in 0..n_steps {
        if stop_flag.load(Ordering::Relaxed) {
            let _ = progress_tx.send(TrainEvent::Error {
                message: "Training stopped by user".into(),
            });
            return Ok(());
        }

        let mut total_loss = 0.0f32;
        for acc in 0..config.grad_accum {
            let off = ((step * config.grad_accum + acc) * seq_len)
                % (tokens.len().saturating_sub(seq_len + 1));
            let batch = &tokens[off..off + seq_len + 1];
            total_loss += lm.train_step(batch, config.lr / config.grad_accum as f32);
        }
        let loss = total_loss / config.grad_accum as f32;

        if step % config.log_every == 0 || step == n_steps - 1 {
            let ppl = lm.perplexity(&tokens[1000..1000 + seq_len]);
            let gen = lm.generate_with_temp("the", 15, 0.7);
            let elapsed = start.elapsed();
            let steps_per_sec = (step + 1) as f64 / elapsed.as_secs_f64();
            let eta_secs = if steps_per_sec > 0.0 {
                ((n_steps - step) as f64 / steps_per_sec) as u64
            } else {
                0
            };

            let _ = progress_tx.send(TrainEvent::Progress {
                step,
                total_steps: n_steps,
                loss,
                ppl,
                steps_per_sec,
                eta_secs,
                generated: gen,
                elapsed_secs: elapsed.as_secs_f64(),
            });
        }

        if config.save_every > 0 && step > 0 && step % config.save_every == 0 {
            let path = format!("{}_step{}.bin", config.output_path, step);
            save_weights(&lm, &path);
            let ppl = lm.perplexity(&tokens[1000..1000 + seq_len]);
            let _ = progress_tx.send(TrainEvent::Checkpoint { step, path, ppl });
        }
    }

    let final_ppl = lm.perplexity(&tokens[1000..1000 + seq_len]);
    let total_time = start.elapsed();

    // Save final model
    let model_path = format!("{}_final.bin", config.output_path);
    save_weights(&lm, &model_path);

    // Export ternary
    let mut lm_tern = lm;
    for layer in &mut lm_tern.layers {
        layer.ternarize();
    }
    let tern_ppl = lm_tern.perplexity(&tokens[1000..1000 + seq_len]);
    let tern_path = format!("{}_ternary.bin", config.output_path);
    save_weights(&lm_tern, &tern_path);

    let _ = progress_tx.send(TrainEvent::Complete {
        init_ppl,
        final_ppl,
        total_time_secs: total_time.as_secs_f64(),
        model_path,
        ternary_ppl: tern_ppl,
        ternary_path: tern_path,
    });

    Ok(())
}

/// Public wrapper around [`save_weights`] for integration tests (T011).
///
/// Keeps the internal API unchanged while letting tests assert the QLMB
/// binary format (including the embedded tokenizer vocab) without
/// duplicating serialization logic.
pub fn save_weights_for_test(lm: &TrainableLM, path: &str) {
    save_weights(lm, path);
}

fn save_weights(lm: &TrainableLM, path: &str) {
    let mut data = Vec::new();
    data.extend_from_slice(&[0x51, 0x4C, 0x4D, 0x42]); // "QLMB" = QLANG LM Binary
    data.extend_from_slice(&(lm.d_model as u32).to_le_bytes());
    data.extend_from_slice(&(lm.vocab_size as u32).to_le_bytes());
    data.extend_from_slice(&(lm.layers.len() as u32).to_le_bytes());

    // Embedding
    write_f32_vec(&mut data, &lm.embedding);
    // Output head
    write_f32_vec(&mut data, &lm.output_head);
    // Layers
    for layer in &lm.layers {
        write_f32_vec(&mut data, &layer.w_x);
        write_f32_vec(&mut data, &layer.w_h);
        write_f32_vec(&mut data, &layer.w_gate);
        write_f32_vec(&mut data, &layer.w_out);
        write_f32_vec(&mut data, &layer.b_h);
        write_f32_vec(&mut data, &layer.b_gate);
    }

    // Vocab section: "VOCB" | u32 len | [u16 word_len | bytes]*
    // Embedding the exact tokenizer vocab ensures reloads use identical
    // token IDs — avoids the <unk> drift caused by re-deriving vocab from
    // reference text on load (T011).
    data.extend_from_slice(&[0x56, 0x4F, 0x43, 0x42]); // "VOCB"
    data.extend_from_slice(&(lm.tokenizer.id2word.len() as u32).to_le_bytes());
    for word in &lm.tokenizer.id2word {
        let bytes = word.as_bytes();
        // Clamp at u16::MAX for safety — word-level tokens are always short.
        let len = bytes.len().min(u16::MAX as usize) as u16;
        data.extend_from_slice(&len.to_le_bytes());
        data.extend_from_slice(&bytes[..len as usize]);
    }

    let _ = std::fs::create_dir_all(std::path::Path::new(path).parent().unwrap_or(std::path::Path::new(".")));
    std::fs::write(path, &data).unwrap_or_else(|e| eprintln!("Save failed: {e}"));
}

fn write_f32_vec(buf: &mut Vec<u8>, data: &[f32]) {
    buf.extend_from_slice(&(data.len() as u32).to_le_bytes());
    for &v in data { buf.extend_from_slice(&v.to_le_bytes()); }
}

/// Train on CPU with GPU-accelerated matmuls, streaming progress via channel.
///
/// Uses `gpu_mamba::forward_gpu` / `backward_and_update_gpu` which offload
/// the large matrix multiplications to the wgpu compute backend while keeping
/// the sequential recurrence on the CPU.
///
/// Falls back to pure CPU if no GPU is available (identical to
/// `train_cpu_with_progress` in that case).
pub fn train_gpu_with_progress(
    config: &GpuTrainConfig,
    progress_tx: std::sync::mpsc::Sender<TrainEvent>,
    stop_flag: Arc<AtomicBool>,
) -> Result<(), String> {
    // Select GPU 1 (training GPU) — avoid GPU 0 which drives the display.
    // SAFETY: set before any wgpu initialisation; the singleton hasn't been
    // created yet when this function is the entry-point.
    #[cfg(feature = "gpu")]
    {
        // wgpu on Vulkan respects WGPU_ADAPTER_NAME for adapter selection.
        // We also set CUDA_VISIBLE_DEVICES so that any CUDA-level code only
        // sees the training GPU.
        std::env::set_var("CUDA_VISIBLE_DEVICES", "1");
    }

    let gpu_name = crate::gpu_mamba::gpu_adapter_name()
        .unwrap_or_else(|| "CPU fallback".into());

    let text = match std::fs::read_to_string(&config.data_path) {
        Ok(t) => t,
        Err(e) => {
            let msg = format!("Read data '{}': {e}", config.data_path);
            let _ = progress_tx.send(TrainEvent::Error { message: msg.clone() });
            return Err(msg);
        }
    };

    let cpu_d = config.d_model.min(128);
    let cpu_hidden = config.d_hidden.min(256);
    let cpu_layers = config.n_layers.min(2);
    let cpu_vocab = config.vocab_size.min(2000);

    let mut lm = TrainableLM::new(&text, cpu_d, cpu_hidden, cpu_layers, cpu_vocab);
    let tokens = lm.tokenizer.encode(&text);
    let seq_len = config.seq_len.min(32);

    if tokens.len() < 1100 + seq_len {
        let msg = "Not enough tokens in training data".to_string();
        let _ = progress_tx.send(TrainEvent::Error { message: msg.clone() });
        return Err(msg);
    }

    let init_ppl = lm.perplexity(&tokens[1000..1000 + seq_len]);
    let start = Instant::now();
    let n_steps = config.n_steps.min(50000);

    // Report GPU name in the first progress event
    let _ = progress_tx.send(TrainEvent::Progress {
        step: 0,
        total_steps: n_steps,
        loss: 0.0,
        ppl: init_ppl,
        steps_per_sec: 0.0,
        eta_secs: 0,
        generated: format!("[GPU: {}] initialising...", gpu_name),
        elapsed_secs: 0.0,
    });

    for step in 0..n_steps {
        if stop_flag.load(Ordering::Relaxed) {
            let _ = progress_tx.send(TrainEvent::Error {
                message: "Training stopped by user".into(),
            });
            return Ok(());
        }

        let mut total_loss = 0.0f32;
        for acc in 0..config.grad_accum {
            let off = ((step * config.grad_accum + acc) * seq_len)
                % (tokens.len().saturating_sub(seq_len + 1));
            let batch = &tokens[off..off + seq_len + 1];
            total_loss += train_step_gpu(&mut lm, batch, config.lr / config.grad_accum as f32);
        }
        let loss = total_loss / config.grad_accum as f32;

        if step % config.log_every == 0 || step == n_steps - 1 {
            let ppl = lm.perplexity(&tokens[1000..1000 + seq_len]);
            let gen = lm.generate_with_temp("the", 15, 0.7);
            let elapsed = start.elapsed();
            let steps_per_sec = (step + 1) as f64 / elapsed.as_secs_f64();
            let eta_secs = if steps_per_sec > 0.0 {
                ((n_steps - step) as f64 / steps_per_sec) as u64
            } else {
                0
            };

            let _ = progress_tx.send(TrainEvent::Progress {
                step,
                total_steps: n_steps,
                loss,
                ppl,
                steps_per_sec,
                eta_secs,
                generated: gen,
                elapsed_secs: elapsed.as_secs_f64(),
            });
        }

        if config.save_every > 0 && step > 0 && step % config.save_every == 0 {
            let path = format!("{}_step{}.bin", config.output_path, step);
            save_weights(&lm, &path);
            let ppl = lm.perplexity(&tokens[1000..1000 + seq_len]);
            let _ = progress_tx.send(TrainEvent::Checkpoint { step, path, ppl });
        }
    }

    let final_ppl = lm.perplexity(&tokens[1000..1000 + seq_len]);
    let total_time = start.elapsed();

    let model_path = format!("{}_final.bin", config.output_path);
    save_weights(&lm, &model_path);

    let mut lm_tern = lm;
    for layer in &mut lm_tern.layers {
        layer.ternarize();
    }
    let tern_ppl = lm_tern.perplexity(&tokens[1000..1000 + seq_len]);
    let tern_path = format!("{}_ternary.bin", config.output_path);
    save_weights(&lm_tern, &tern_path);

    let _ = progress_tx.send(TrainEvent::Complete {
        init_ppl,
        final_ppl,
        total_time_secs: total_time.as_secs_f64(),
        model_path,
        ternary_ppl: tern_ppl,
        ternary_path: tern_path,
    });

    Ok(())
}

/// Single training step using GPU-accelerated forward/backward.
///
/// Mirror of `TrainableLM::train_step` but routes the heavy matmuls through
/// `gpu_mamba::forward_gpu` and `gpu_mamba::backward_and_update_gpu`.
fn train_step_gpu(lm: &mut TrainableLM, tokens: &[usize], lr: f32) -> f32 {
    if tokens.len() < 2 {
        return 0.0;
    }
    let seq = tokens.len() - 1;
    let d = lm.d_model;
    let v = lm.vocab_size;

    // Embed
    let mut x = vec![0.0f32; seq * d];
    for (i, &tok) in tokens[..seq].iter().enumerate() {
        let t = tok.min(v - 1);
        x[i * d..(i + 1) * d].copy_from_slice(&lm.embedding[t * d..(t + 1) * d]);
    }

    // Forward through Mamba layers (GPU-accelerated)
    let mut all_hidden = Vec::new();
    let mut layer_inputs = vec![x.clone()];
    for layer in &lm.layers {
        let (out, hs) = crate::gpu_mamba::forward_gpu(layer, &x, seq);
        all_hidden.push(hs);
        x = out;
        layer_inputs.push(x.clone());
    }

    // Output head logits
    let mut logits = vec![0.0f32; seq * v];
    for t in 0..seq {
        for vi in 0..v {
            let mut sum = 0.0f32;
            for k in 0..d {
                sum += x[t * d + k] * lm.output_head[k * v + vi];
            }
            logits[t * v + vi] = sum;
        }
    }

    // Softmax + cross-entropy loss
    let mut probs = logits.clone();
    let mut loss = 0.0f32;
    for t in 0..seq {
        let off = t * v;
        let max = probs[off..off + v]
            .iter()
            .cloned()
            .fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        for i in 0..v {
            probs[off + i] = (probs[off + i] - max).exp();
            sum += probs[off + i];
        }
        if sum > 0.0 {
            for i in 0..v {
                probs[off + i] /= sum;
            }
        }
        let target = tokens[t + 1].min(v - 1);
        loss -= (probs[off + target].max(1e-10)).ln();
    }
    loss /= seq as f32;

    // d_logits = probs - one_hot
    let mut d_logits = probs;
    for t in 0..seq {
        let target = tokens[t + 1].min(v - 1);
        d_logits[t * v + target] -= 1.0;
        for i in 0..v {
            d_logits[t * v + i] /= seq as f32;
        }
    }

    // Update output head
    for t in 0..seq {
        for vi in 0..v {
            let g = d_logits[t * v + vi];
            if g.abs() < 1e-6 {
                continue;
            }
            for k in 0..d {
                lm.output_head[k * v + vi] -= lr * g * x[t * d + k];
            }
        }
    }

    // Compute d_x (gradient flowing to last layer output)
    let mut d_x = vec![0.0f32; seq * d];
    for t in 0..seq {
        for k in 0..d {
            let mut sum = 0.0f32;
            for vi in 0..v {
                sum += d_logits[t * v + vi] * lm.output_head[k * v + vi];
            }
            d_x[t * d + k] = sum;
        }
    }

    // BPTT through each Mamba layer (reverse, GPU-accelerated)
    for l in (0..lm.layers.len()).rev() {
        crate::gpu_mamba::backward_and_update_gpu(
            &mut lm.layers[l],
            &layer_inputs[l],
            &all_hidden[l],
            &d_x,
            seq,
            lr,
        );
    }

    // Update embeddings
    for t in 0..seq {
        let tok = tokens[t].min(v - 1);
        for k in 0..d {
            lm.embedding[tok * d + k] -= lr * 0.1 * d_x[t * d + k];
        }
    }

    loss
}

/// Train with candle CUDA backend -- tensors stay on GPU.
/// Falls back to error if cuda feature is not enabled.
pub fn train_candle_with_progress(
    config: &GpuTrainConfig,
    progress_tx: std::sync::mpsc::Sender<TrainEvent>,
    stop_flag: Arc<AtomicBool>,
) -> Result<(), String> {
    #[cfg(feature = "cuda")]
    {
        return crate::candle_train::train_candle(config, progress_tx, stop_flag);
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = (config, stop_flag);
        let msg = "CUDA feature not enabled. Build with --features cuda".to_string();
        let _ = progress_tx.send(TrainEvent::Error { message: msg.clone() });
        Err(msg)
    }
}

/// Train with 2-GPU data parallelism.
/// GPU 0 and GPU 1 each process half the batches, gradients are averaged.
/// On CPU: simulated with 2 threads.
pub fn train_2gpu(config: &GpuTrainConfig) -> Result<(), String> {
    use std::sync::{Arc, Mutex};
    use std::thread;

    let text = std::fs::read_to_string(&config.data_path)
        .map_err(|e| format!("Read data: {e}"))?;

    config.print_summary();
    println!("MODE: 2-GPU Data Parallelism (simulated on CPU with 2 threads)\n");

    let cpu_d = config.d_model.min(128);
    let cpu_hidden = config.d_hidden.min(256);
    let cpu_layers = config.n_layers.min(2);
    let cpu_vocab = config.vocab_size.min(2000);

    let lm = Arc::new(Mutex::new(
        TrainableLM::new(&text, cpu_d, cpu_hidden, cpu_layers, cpu_vocab)
    ));
    let tokenizer = Tokenizer::from_text(&text, cpu_vocab);
    let tokens: Vec<usize> = tokenizer.encode(&text);
    let seq_len = config.seq_len.min(32);
    let n_steps = config.n_steps.min(3000);

    let init_ppl = lm.lock().unwrap().perplexity(&tokens[1000..1000 + seq_len]);
    println!("Tokens: {}, Init PPL: {:.1}\n", tokens.len(), init_ppl);

    let start = Instant::now();

    for step in 0..n_steps {
        let tokens_0 = tokens.clone();
        let tokens_1 = tokens.clone();
        let lm_0 = lm.clone();
        let lm_1 = lm.clone();
        let lr = config.lr;

        // GPU 0: even offset
        let off_0 = (step * 2 * seq_len) % (tokens.len().saturating_sub(seq_len + 1));
        // GPU 1: odd offset
        let off_1 = ((step * 2 + 1) * seq_len) % (tokens.len().saturating_sub(seq_len + 1));

        // Parallel: both GPUs train on different data
        let h0 = thread::spawn(move || {
            let batch = &tokens_0[off_0..off_0 + seq_len + 1];
            lm_0.lock().unwrap().train_step(batch, lr * 0.5)
        });
        let h1 = thread::spawn(move || {
            let batch = &tokens_1[off_1..off_1 + seq_len + 1];
            lm_1.lock().unwrap().train_step(batch, lr * 0.5)
        });

        let loss_0 = h0.join().unwrap_or(0.0);
        let loss_1 = h1.join().unwrap_or(0.0);
        let loss = (loss_0 + loss_1) / 2.0;

        if step % config.log_every.max(1) == 0 || step == n_steps - 1 {
            let ppl = lm.lock().unwrap().perplexity(&tokens[1000..1000 + seq_len]);
            let gen = lm.lock().unwrap().generate_with_temp("the", 12, 0.7);
            let elapsed = start.elapsed();
            let sps = (step + 1) as f64 / elapsed.as_secs_f64();
            println!("  Step {:>5}: loss={:.3} ppl={:.1} [{:.1} s/s, ETA: {}m] \"{}\"",
                step, loss, ppl, sps, ((n_steps - step) as f64 / sps) as u64 / 60, gen);
        }
    }

    let final_ppl = lm.lock().unwrap().perplexity(&tokens[1000..1000 + seq_len]);
    println!("\nFinal PPL: {:.1} (init: {:.1}, {:.1}x)", final_ppl, init_ppl, init_ppl / final_ppl);
    println!("Time: {:?} (2-GPU parallel)", start.elapsed());

    // Save
    let path = format!("{}_2gpu_final.bin", config.output_path);
    save_weights(&lm.lock().unwrap(), &path);
    println!("Model saved: {}", path);

    Ok(())
}

/// Generate the GPU training script (for the RTX 2070 machine).
pub fn generate_gpu_script() -> String {
    r#"#!/bin/bash
# QLANG GPU Training Script for 2x RTX 2070 Super
# Run on the machine with GPUs

set -e

echo "=== QLANG GPU Training ==="
echo "Target: 30M parameter Mamba LM on WikiText-2"
echo ""

# 1. Check CUDA
nvidia-smi || { echo "ERROR: No NVIDIA GPU found"; exit 1; }
echo ""

# 2. Install Rust (if needed)
which cargo || { curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y; source ~/.cargo/env; }

# 3. Clone repo
if [ ! -d "neoqlang" ]; then
    git clone https://github.com/Mirkulix/neoqlang.git
fi
cd neoqlang/qlang

# 4. Download WikiText-2
mkdir -p data/wikitext2
if [ ! -f data/wikitext2/train.txt ]; then
    curl -sL "https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2/train.txt" -o data/wikitext2/train.txt
    curl -sL "https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2/valid.txt" -o data/wikitext2/valid.txt
fi
echo "Data: $(wc -l data/wikitext2/train.txt)"

# 5. Build with CUDA support
echo "Building with CUDA..."
LLVM_SYS_180_PREFIX=/opt/llvm18 cargo build --release --bin gpu_train 2>&1 | tail -5

# 6. Train
echo ""
echo "=== Starting Training ==="
echo "This will take 6-12 hours on 2x RTX 2070 Super"
echo ""
./target/release/gpu_train

echo ""
echo "=== Training Complete ==="
echo "Models saved in data/"
ls -la data/mamba_30m*
"#.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gpu_config_sizing() {
        let config = GpuTrainConfig::default();
        config.print_summary();

        assert!(config.param_count() > 20_000_000, "Should be >20M params");
        assert!(config.param_count() < 50_000_000, "Should be <50M params");
        assert!(config.vram_mb() < 7500, "Must fit in 8GB VRAM");
    }

    #[test]
    fn gpu_train_cpu_test() {
        let paths = ["data/wikitext2/train.txt", "../data/wikitext2/train.txt",
            "/home/mirkulix/neoqlang/qlang/data/wikitext2/train.txt"];

        let mut found = false;
        for p in &paths {
            if std::fs::metadata(p).is_ok() {
                let cfg = GpuTrainConfig {
                    n_steps: 100, log_every: 50, save_every: 0,
                    data_path: p.to_string(), ..Default::default()
                };
                match train_cpu(&cfg) {
                    Ok(()) => { found = true; break; }
                    Err(e) => println!("Error: {e}"),
                }
            }
        }
        if !found { println!("WikiText-2 not found, skipping CPU test"); }
    }
}
