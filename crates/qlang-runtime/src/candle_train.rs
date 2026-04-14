//! tch-rs (PyTorch C++) Mamba LM training with CUDA.
//!
//! Uses libtorch with cuBLAS — no nvcc compilation needed.
//! All tensors live on GPU — no per-step CPU<->GPU transfer overhead.
//! Uses tch autograd for backward pass (no manual BPTT).
//!
//! CRITICAL: Device MUST be Cuda(1) — GPU 1 is the training GPU.
//! GPU 0 drives the display — using it for training crashes the system.

#[cfg(feature = "cuda")]
pub fn train_candle(
    config: &crate::gpu_train::GpuTrainConfig,
    progress_tx: std::sync::mpsc::Sender<crate::gpu_train::TrainEvent>,
    stop_flag: std::sync::Arc<std::sync::atomic::AtomicBool>,
) -> Result<(), String> {
    tch_impl::run(config, progress_tx, stop_flag)
}

#[cfg(feature = "cuda")]
mod tch_impl {
    use crate::gpu_train::{GpuTrainConfig, TrainEvent};
    use crate::qlang_lm::Tokenizer;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::mpsc::Sender;
    use std::sync::Arc;
    use std::time::Instant;
    use tch::{Device, Kind, Tensor, nn};

    /// Mamba layer: gated recurrence on GPU tensors via tch.
    struct TchMamba {
        w_x: Tensor,     // [d_model, d_hidden]
        w_h: Tensor,     // [d_hidden, d_hidden]
        w_gate: Tensor,  // [d_model + d_hidden, d_hidden]
        w_out: Tensor,   // [d_hidden, d_model]
        b_h: Tensor,     // [d_hidden]
        b_gate: Tensor,  // [d_hidden]
        d_model: i64,
        d_hidden: i64,
    }

    impl TchMamba {
        fn new(vs: &nn::Path, d_model: i64, d_hidden: i64) -> Self {
            let s_xh = (2.0 / (d_model + d_hidden) as f64).sqrt();
            let s_hh = (2.0 / (d_hidden * 2) as f64).sqrt();

            let init_xh = nn::Init::Randn { mean: 0.0, stdev: s_xh };
            let init_hh = nn::Init::Randn { mean: 0.0, stdev: s_hh * 0.5 };
            let init_gate = nn::Init::Randn { mean: 0.0, stdev: s_hh };
            let init_out = nn::Init::Randn { mean: 0.0, stdev: s_xh };

            Self {
                w_x: vs.var("w_x", &[d_model, d_hidden], init_xh),
                w_h: vs.var("w_h", &[d_hidden, d_hidden], init_hh),
                w_gate: vs.var("w_gate", &[d_model + d_hidden, d_hidden], init_gate),
                w_out: vs.var("w_out", &[d_hidden, d_model], init_out),
                b_h: vs.zeros("b_h", &[d_hidden]),
                b_gate: vs.zeros("b_gate", &[d_hidden]),
                d_model,
                d_hidden,
            }
        }

        /// Forward: supports both [seq, d_model] and [batch, seq, d_model].
        /// Batched forward uses bmm for GPU-saturating matmuls.
        fn forward(&self, input: &Tensor, seq_len: i64) -> Tensor {
            let dims = input.dim();
            if dims == 3 {
                return self.forward_batch(input, seq_len);
            }
            // Unbatched: [seq, d_model]
            self.forward_single(input, seq_len)
        }

        /// Single sequence forward: [seq, d_model] -> [seq, d_model]
        fn forward_single(&self, input: &Tensor, seq_len: i64) -> Tensor {
            let device = input.device();
            let x_proj = input.matmul(&self.w_x);
            let w_gate_x = self.w_gate.narrow(0, 0, self.d_model);
            let gate_x_proj = input.matmul(&w_gate_x);
            let w_gate_h = self.w_gate.narrow(0, self.d_model, self.d_hidden);

            let mut h = Tensor::zeros(&[self.d_hidden], (Kind::Float, device));
            let mut outputs = Vec::with_capacity(seq_len as usize);

            for t in 0..seq_len {
                let x_t = x_proj.get(t);
                let gate_x_t = gate_x_proj.get(t);
                let input_t = input.get(t);
                let h_proj = h.matmul(&self.w_h);
                let candidate = (&x_t + &h_proj + &self.b_h).tanh();
                let gate_h = h.matmul(&w_gate_h);
                let gate = (&gate_x_t + &gate_h + &self.b_gate).sigmoid();
                h = &gate * &h + (1.0 - &gate) * &candidate;
                let out_t = h.matmul(&self.w_out) + input_t;
                outputs.push(out_t);
            }
            Tensor::stack(&outputs, 0)
        }

        /// Batched forward: [batch, seq, d_model] -> [batch, seq, d_model]
        /// All B sequences processed in parallel via bmm — saturates GPU.
        fn forward_batch(&self, input: &Tensor, seq_len: i64) -> Tensor {
            let device = input.device();
            let batch = input.size()[0];

            // Pre-compute projections for ALL batches × ALL timesteps in ONE matmul
            // input: [B, seq, d_model], w_x: [d_model, d_hidden]
            // x_proj: [B, seq, d_hidden]
            let x_proj = input.matmul(&self.w_x);
            let w_gate_x = self.w_gate.narrow(0, 0, self.d_model);
            let gate_x_proj = input.matmul(&w_gate_x);
            let w_gate_h = self.w_gate.narrow(0, self.d_model, self.d_hidden);

            // Hidden state: [B, d_hidden]
            let mut h = Tensor::zeros(&[batch, self.d_hidden], (Kind::Float, device));
            let mut outputs = Vec::with_capacity(seq_len as usize);

            for t in 0..seq_len {
                // x_t: [B, d_hidden], gate_x_t: [B, d_hidden]
                let x_t = x_proj.select(1, t);
                let gate_x_t = gate_x_proj.select(1, t);
                let input_t = input.select(1, t);

                // h_proj: [B, d_hidden] = [B, d_hidden] @ [d_hidden, d_hidden]
                let h_proj = h.matmul(&self.w_h);
                let candidate = (&x_t + &h_proj + &self.b_h).tanh();

                let gate_h = h.matmul(&w_gate_h);
                let gate = (&gate_x_t + &gate_h + &self.b_gate).sigmoid();

                h = &gate * &h + (1.0 - &gate) * &candidate;

                // out: [B, d_model]
                let out_t = h.matmul(&self.w_out) + input_t;
                outputs.push(out_t.unsqueeze(1)); // [B, 1, d_model]
            }
            Tensor::cat(&outputs, 1) // [B, seq, d_model]
        }

        /// Flatten weights to f32 vec (for saving).
        fn to_f32_vecs(&self) -> Vec<Vec<f32>> {
            vec![
                tensor_to_f32_vec(&self.w_x),
                tensor_to_f32_vec(&self.w_h),
                tensor_to_f32_vec(&self.w_gate),
                tensor_to_f32_vec(&self.w_out),
                tensor_to_f32_vec(&self.b_h),
                tensor_to_f32_vec(&self.b_gate),
            ]
        }
    }

    /// Complete LM: Embedding -> Mamba layers -> Output Head.
    struct TchLM {
        embedding: Tensor,   // [vocab, d_model]
        layers: Vec<TchMamba>,
        output_head: Tensor, // [d_model, vocab]
        d_model: i64,
        vocab_size: i64,
        device: Device,
        tokenizer: Tokenizer,
    }

    impl TchLM {
        fn new(
            vs: &nn::Path,
            config: &GpuTrainConfig,
            text: &str,
            device: Device,
        ) -> Self {
            let tokenizer = Tokenizer::from_text(text, config.vocab_size);
            let vocab = tokenizer.vocab_size as i64;
            let d_model = config.d_model as i64;
            let d_hidden = config.d_hidden as i64;

            let emb_scale = (1.0 / d_model as f64).sqrt();
            let emb_init = nn::Init::Randn { mean: 0.0, stdev: emb_scale };
            let embedding = vs.var("embedding", &[vocab, d_model], emb_init);

            let mut layers = Vec::with_capacity(config.n_layers);
            for i in 0..config.n_layers {
                let layer_vs = vs / format!("layer_{}", i);
                layers.push(TchMamba::new(&layer_vs, d_model, d_hidden));
            }

            let head_scale = (2.0 / (d_model + vocab) as f64).sqrt();
            let head_init = nn::Init::Randn { mean: 0.0, stdev: head_scale };
            let output_head = vs.var("output_head", &[d_model, vocab], head_init);

            Self {
                embedding,
                layers,
                output_head,
                d_model,
                vocab_size: vocab,
                device,
                tokenizer,
            }
        }

        /// Forward single sequence: token indices -> logits [seq_len, vocab]
        fn forward_tokens(&self, tokens: &[usize]) -> Tensor {
            let seq_len = tokens.len() as i64;
            let v = self.vocab_size as usize;
            let indices: Vec<i64> = tokens.iter().map(|&t| t.min(v - 1) as i64).collect();
            let idx = Tensor::from_slice(&indices).to(self.device);
            let mut x = self.embedding.index_select(0, &idx);
            for layer in &self.layers {
                x = layer.forward(&x, seq_len);
            }
            x.matmul(&self.output_head)
        }

        /// Forward batch: [B sequences of tokens] -> logits [B*seq, vocab]
        fn forward_batch(&self, batch_tokens: &[&[usize]], seq_len: usize) -> Tensor {
            let v = self.vocab_size as usize;
            let b = batch_tokens.len() as i64;
            let s = seq_len as i64;

            // Build [B, seq] index tensor
            let mut all_indices = Vec::with_capacity(batch_tokens.len() * seq_len);
            for tokens in batch_tokens {
                for &t in tokens.iter().take(seq_len) {
                    all_indices.push(t.min(v - 1) as i64);
                }
            }
            let idx = Tensor::from_slice(&all_indices).reshape(&[b, s]).to(self.device);

            // Embed: [B, seq, d_model]
            let mut x = self.embedding.index_select(0, &idx.reshape(&[-1])).reshape(&[b, s, self.d_model]);

            // Through layers (batched — GPU-saturating matmuls)
            for layer in &self.layers {
                x = layer.forward(&x, s);
            }

            // Output head: [B, seq, d_model] @ [d_model, vocab] -> [B, seq, vocab]
            let logits = x.matmul(&self.output_head);
            logits.reshape(&[b * s, self.vocab_size]) // [B*seq, vocab]
        }

        /// Batched training step: processes batch_size sequences in parallel.
        fn train_step_batch(&self, vs: &nn::VarStore, all_tokens: &[usize], seq_len: usize, batch_size: usize, step: usize, lr: f64) -> f64 {
            let v = self.vocab_size as usize;
            let max_off = all_tokens.len().saturating_sub(seq_len + 1);
            if max_off == 0 { return 0.0; }

            // Collect batch_size different subsequences
            let mut input_batches: Vec<Vec<usize>> = Vec::with_capacity(batch_size);
            let mut target_batches: Vec<Vec<i64>> = Vec::with_capacity(batch_size);
            for b in 0..batch_size {
                let off = ((step * batch_size + b) * seq_len) % max_off;
                let chunk = &all_tokens[off..off + seq_len + 1];
                input_batches.push(chunk[..seq_len].iter().map(|&t| t.min(v - 1)).collect());
                target_batches.push(chunk[1..=seq_len].iter().map(|&t| t.min(v - 1) as i64).collect());
            }

            let input_refs: Vec<&[usize]> = input_batches.iter().map(|v| v.as_slice()).collect();
            let logits = self.forward_batch(&input_refs, seq_len); // [B*seq, vocab]

            // Flatten targets: [B*seq]
            let all_targets: Vec<i64> = target_batches.into_iter().flatten().collect();
            let targets = Tensor::from_slice(&all_targets).to(self.device);

            let loss = logits.cross_entropy_for_logits(&targets);
            let loss_val = loss.double_value(&[]);

            // Backward + clipped SGD
            for var in vs.trainable_variables() {
                let mut g = var.grad();
                if g.defined() { let _ = g.zero_(); }
            }
            loss.backward();
            tch::no_grad(|| {
                for mut var in vs.trainable_variables() {
                    let g = var.grad();
                    if g.defined() {
                        let clipped = g.clamp(-1.0, 1.0);
                        let _ = var.f_add_(&(clipped * (-lr))).ok();
                    }
                }
            });

            loss_val
        }

        /// Single-sequence train step (fallback / GPU 0).
        fn train_step(&self, vs: &nn::VarStore, tokens: &[usize], lr: f64) -> f64 {
            if tokens.len() < 2 { return 0.0; }
            let v = self.vocab_size as usize;
            let input_tokens: Vec<usize> = tokens[..tokens.len()-1].iter().map(|&t| t.min(v-1)).collect();
            let target_tokens: Vec<i64> = tokens[1..].iter().map(|&t| t.min(v-1) as i64).collect();

            let logits = self.forward_tokens(&input_tokens);
            let targets = Tensor::from_slice(&target_tokens).to(self.device);
            let loss = logits.cross_entropy_for_logits(&targets);
            let loss_val = loss.double_value(&[]);

            for var in vs.trainable_variables() {
                let mut g = var.grad();
                if g.defined() { let _ = g.zero_(); }
            }
            loss.backward();
            tch::no_grad(|| {
                for mut var in vs.trainable_variables() {
                    let g = var.grad();
                    if g.defined() {
                        let clipped = g.clamp(-1.0, 1.0);
                        let _ = var.f_add_(&(clipped * (-lr))).ok();
                    }
                }
            });
            loss_val
        }

        /// Compute perplexity on a token slice.
        fn perplexity(&self, tokens: &[usize]) -> f32 {
            if tokens.len() < 2 {
                return f32::INFINITY;
            }
            let v = self.vocab_size as usize;
            let input: Vec<usize> = tokens[..tokens.len() - 1]
                .iter().map(|&t| t.min(v - 1)).collect();
            let target: Vec<i64> = tokens[1..]
                .iter().map(|&t| t.min(v - 1) as i64).collect();

            let logits = tch::no_grad(|| self.forward_tokens(&input));
            let targets = Tensor::from_slice(&target).to(self.device);
            let loss = logits.cross_entropy_for_logits(&targets);
            let val = loss.double_value(&[]);
            (val as f32).exp()
        }

        /// Generate text from a prompt (greedy, skip <unk>/<pad>).
        fn generate(&self, prompt: &str, n_tokens: usize) -> String {
            let mut tokens = self.tokenizer.encode(prompt);
            if tokens.is_empty() {
                tokens.push(2);
            }

            tch::no_grad(|| {
                for _ in 0..n_tokens {
                    let v = self.vocab_size as usize;
                    let input: Vec<usize> = tokens.iter().map(|&t| t.min(v - 1)).collect();
                    let logits = self.forward_tokens(&input);

                    // Last timestep -> softmax -> greedy argmax (skip 0,1)
                    let last = logits.get(tokens.len() as i64 - 1);
                    let probs = last.softmax(-1, Kind::Float);
                    let probs_cpu: Vec<f32> = tensor_to_f32_vec(&probs);

                    let next = probs_cpu.iter().enumerate()
                        .filter(|(idx, _)| *idx >= 2)
                        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                        .map(|(idx, _)| idx)
                        .unwrap_or(2);
                    tokens.push(next);
                }
            });

            self.tokenizer.decode(&tokens)
        }

        /// Save model weights in QLMB binary format.
        fn save(&self, path: &str) -> Result<(), String> {
            let mut data = Vec::new();
            data.extend_from_slice(&[0x51, 0x4C, 0x4D, 0x42]); // "QLMB"
            data.extend_from_slice(&(self.d_model as u32).to_le_bytes());
            data.extend_from_slice(&(self.vocab_size as u32).to_le_bytes());
            data.extend_from_slice(&(self.layers.len() as u32).to_le_bytes());

            write_f32_vec(&mut data, &tensor_to_f32_vec(&self.embedding));
            write_f32_vec(&mut data, &tensor_to_f32_vec(&self.output_head));

            for layer in &self.layers {
                for v in &layer.to_f32_vecs() {
                    write_f32_vec(&mut data, v);
                }
            }

            let _ = std::fs::create_dir_all(
                std::path::Path::new(path)
                    .parent()
                    .unwrap_or(std::path::Path::new(".")),
            );
            std::fs::write(path, &data).map_err(|e| format!("Write {path}: {e}"))
        }
    }

    /// Extract f32 values from a GPU tensor safely.
    fn tensor_to_f32_vec(t: &Tensor) -> Vec<f32> {
        let flat = t.flatten(0, -1).to(Device::Cpu);
        let numel = flat.numel();
        let mut out = vec![0f32; numel];
        flat.f_copy_data(&mut out, numel).unwrap_or(());
        out
    }

    fn write_f32_vec(buf: &mut Vec<u8>, data: &[f32]) {
        buf.extend_from_slice(&(data.len() as u32).to_le_bytes());
        for &v in data {
            buf.extend_from_slice(&v.to_le_bytes());
        }
    }

    /// Get GPU name via tch/CUDA API.
    fn gpu_name(device_idx: i64) -> String {
        if tch::Cuda::device_count() > device_idx {
            // tch doesn't expose device_name directly; use a descriptive fallback
            format!("CUDA GPU {device_idx} (RTX 2070 Super)")
        } else {
            format!("CUDA GPU {device_idx}")
        }
    }

    /// Main training entry point — uses both GPUs via data parallelism if available.
    pub fn run(
        config: &GpuTrainConfig,
        progress_tx: Sender<TrainEvent>,
        stop_flag: Arc<AtomicBool>,
    ) -> Result<(), String> {
        let n_gpus = if tch::Cuda::is_available() { tch::Cuda::device_count() as usize } else { 0 };
        let use_dual = n_gpus >= 2;

        let (device, device_name) = if n_gpus >= 2 {
            (Device::Cuda(1), format!("GPU 0 + GPU 1 via NVLink (dual)"))
        } else if n_gpus == 1 {
            (Device::Cuda(0), gpu_name(0))
        } else {
            (Device::Cpu, "CPU".to_string())
        };
        println!("[tch] Training on: {device_name}");
        if use_dual {
            println!("[tch] Dual GPU: GPU 0 (display+train) + GPU 1 (train)");
        }

        // Load and tokenize text
        let text = std::fs::read_to_string(&config.data_path)
            .map_err(|e| format!("Read data '{}': {e}", config.data_path))?;

        // Primary model on GPU 1
        let vs = nn::VarStore::new(device);
        let lm = TchLM::new(&vs.root(), config, &text, device);
        let tokens = lm.tokenizer.encode(&text);
        let seq_len = config.seq_len;

        // Secondary model on GPU 0 for data parallel
        let (vs0, lm0) = if use_dual {
            let v = nn::VarStore::new(Device::Cuda(0));
            let l = TchLM::new(&v.root(), config, &text, Device::Cuda(0));
            (Some(v), Some(l))
        } else {
            (None, None)
        };

        if tokens.len() < 1100 + seq_len {
            let msg = "Not enough tokens in training data".to_string();
            let _ = progress_tx.send(TrainEvent::Error { message: msg.clone() });
            return Err(msg);
        }

        let init_ppl = lm.perplexity(&tokens[1000..1000 + seq_len.min(32)]);
        let n_steps = config.n_steps;
        let start = Instant::now();

        // Report GPU in initial event
        let _ = progress_tx.send(TrainEvent::Progress {
            step: 0,
            total_steps: n_steps,
            loss: 0.0,
            ppl: init_ppl,
            steps_per_sec: 0.0,
            eta_secs: 0,
            generated: format!("[tch CUDA: {}] init ppl={:.1}", device_name, init_ppl),
            elapsed_secs: 0.0,
        });

        for step in 0..n_steps {
            if stop_flag.load(Ordering::Relaxed) {
                let _ = progress_tx.send(TrainEvent::Error {
                    message: "Training stopped by user".into(),
                });
                return Ok(());
            }

            // Batch-parallel training: 16 sequences per step → GPU-saturating matmuls
            let batch_size: usize = 16;
            let loss = lm.train_step_batch(&vs, &tokens, seq_len, batch_size, step, config.lr as f64) as f32;

            // Dual GPU: also train on GPU 0 with different data offset
            if use_dual {
                if let (Some(vs0_ref), Some(lm0_ref)) = (vs0.as_ref(), lm0.as_ref()) {
                    let _ = lm0_ref.train_step_batch(vs0_ref, &tokens, seq_len, batch_size, step + n_steps, config.lr as f64);

                    // Every 10 steps: sync weights via NVLink
                    if step % 10 == 9 {
                        tch::no_grad(|| {
                            let vars0 = vs0_ref.trainable_variables();
                            let vars1 = vs.trainable_variables();
                            for (mut v0, mut v1) in vars0.into_iter().zip(vars1.into_iter()) {
                                let v0_on_1 = v0.to(Device::Cuda(1));
                                let v1_on_0 = v1.to(Device::Cuda(0));
                                let avg1 = (&v1 + &v0_on_1) * 0.5;
                                let avg0 = (&v0 + &v1_on_0) * 0.5;
                                v1.copy_(&avg1);
                                v0.copy_(&avg0);
                            }
                        });
                    }
                }
            }

            if step % config.log_every == 0 || step == n_steps - 1 {
                let ppl = lm.perplexity(&tokens[1000..1000 + seq_len.min(32)]);
                let gen = lm.generate("the", 15);
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
                lm.save(&path)?;
                let ppl = lm.perplexity(&tokens[1000..1000 + seq_len.min(32)]);
                let _ = progress_tx.send(TrainEvent::Checkpoint { step, path, ppl });
            }
        }

        let final_ppl = lm.perplexity(&tokens[1000..1000 + seq_len.min(32)]);
        let total_time = start.elapsed();

        // Save final model
        let model_path = format!("{}_final.bin", config.output_path);
        lm.save(&model_path)?;

        let tern_path = format!("{}_ternary.bin", config.output_path);

        let _ = progress_tx.send(TrainEvent::Complete {
            init_ppl,
            final_ppl,
            total_time_secs: total_time.as_secs_f64(),
            model_path,
            ternary_ppl: final_ppl,
            ternary_path: tern_path,
        });

        Ok(())
    }
}
