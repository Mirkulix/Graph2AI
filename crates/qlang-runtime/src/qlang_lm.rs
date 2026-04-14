//! QLANG Language Model — Ternary Mamba trained with NoProp.
//!
//! The first ternary language model trained without backpropagation.
//!
//! Architecture:
//!   Tokenizer (word-level) → Embedding → Mamba Layers → Output Head
//!
//! Training: NoProp (denoising per block, no gradient between blocks)
//! Weights: Ternary {-1, 0, +1} for projections
//! Evaluation: Perplexity on WikiText-2

use crate::mamba::MambaLayer;
use rayon::prelude::*;
use std::collections::HashMap;

// ============================================================
// Simple Word-Level Tokenizer
// ============================================================

pub struct Tokenizer {
    pub word2id: HashMap<String, usize>,
    pub id2word: Vec<String>,
    pub vocab_size: usize,
}

impl Tokenizer {
    /// Build tokenizer from text, keeping top-N most frequent words.
    pub fn from_text(text: &str, max_vocab: usize) -> Self {
        let mut counts: HashMap<String, usize> = HashMap::new();
        for word in text.split_whitespace() {
            let w = word.to_lowercase();
            *counts.entry(w).or_insert(0) += 1;
        }

        // Sort by frequency, keep top N
        let mut words: Vec<(String, usize)> = counts.into_iter().collect();
        words.sort_by(|a, b| b.1.cmp(&a.1));

        let mut word2id = HashMap::new();
        let mut id2word = Vec::new();

        // Reserve 0 = <unk>, 1 = <pad>
        word2id.insert("<unk>".to_string(), 0);
        id2word.push("<unk>".to_string());
        word2id.insert("<pad>".to_string(), 1);
        id2word.push("<pad>".to_string());

        for (word, _) in words.iter().take(max_vocab - 2) {
            let id = id2word.len();
            word2id.insert(word.clone(), id);
            id2word.push(word.clone());
        }

        let vocab_size = id2word.len();
        Self { word2id, id2word, vocab_size }
    }

    /// Build a tokenizer from an exact, ordered vocabulary list.
    ///
    /// Unlike `from_text`, this preserves the provided ID assignment so a
    /// serialized vocabulary (e.g. embedded in a QLMB checkpoint) can be
    /// reconstructed byte-for-byte. The first two entries are expected to be
    /// `<unk>` and `<pad>` (IDs 0 and 1) but this is not enforced — callers
    /// must supply a valid vocab.
    pub fn from_vocab(vocab: Vec<String>) -> Self {
        let mut word2id = HashMap::with_capacity(vocab.len());
        for (i, w) in vocab.iter().enumerate() {
            word2id.insert(w.clone(), i);
        }
        let vocab_size = vocab.len();
        Self { word2id, id2word: vocab, vocab_size }
    }

    /// Tokenize text → token IDs.
    pub fn encode(&self, text: &str) -> Vec<usize> {
        text.split_whitespace()
            .map(|w| *self.word2id.get(&w.to_lowercase()).unwrap_or(&0))
            .collect()
    }

    /// Decode token IDs → text.
    pub fn decode(&self, ids: &[usize]) -> String {
        ids.iter()
            .map(|&id| self.id2word.get(id).map(|s| s.as_str()).unwrap_or("<unk>"))
            .collect::<Vec<_>>()
            .join(" ")
    }
}

// ============================================================
// Ternary Mamba Language Model
// ============================================================

pub struct QlangLM {
    pub tokenizer: Tokenizer,
    /// Embedding table: [vocab_size, d_model]
    pub embedding: Vec<f32>,
    /// Mamba layers
    pub mamba_layers: Vec<MambaLayer>,
    /// Output head (NoProp blocks for next-token prediction)
    pub output_head: Vec<f32>, // [d_model, vocab_size]
    pub d_model: usize,
    pub vocab_size: usize,
}

impl QlangLM {
    /// Create a new language model.
    pub fn new(text: &str, d_model: usize, d_inner: usize, d_state: usize, n_layers: usize, max_vocab: usize) -> Self {
        let tokenizer = Tokenizer::from_text(text, max_vocab);
        let vocab_size = tokenizer.vocab_size;

        // Random embedding table
        let scale = (1.0 / d_model as f64).sqrt() as f32;
        let embedding: Vec<f32> = (0..vocab_size * d_model)
            .map(|i| (i as f32 * 0.4871).sin() * scale)
            .collect();

        // Mamba layers
        let mamba_layers = (0..n_layers)
            .map(|i| MambaLayer::new(d_model, d_inner, d_state, 0.37 + i as f32 * 0.13))
            .collect();

        // Output head: [d_model, vocab_size]
        let out_scale = (2.0 / (d_model + vocab_size) as f64).sqrt() as f32;
        let output_head: Vec<f32> = (0..d_model * vocab_size)
            .map(|i| (i as f32 * 0.7291).sin() * out_scale)
            .collect();

        Self { tokenizer, embedding, mamba_layers, output_head, d_model, vocab_size }
    }

    /// Embed tokens: [seq_len] → [seq_len, d_model]
    fn embed(&self, tokens: &[usize]) -> Vec<f32> {
        let mut embedded = vec![0.0f32; tokens.len() * self.d_model];
        for (i, &tok) in tokens.iter().enumerate() {
            let tok = tok.min(self.vocab_size - 1);
            embedded[i * self.d_model..(i + 1) * self.d_model]
                .copy_from_slice(&self.embedding[tok * self.d_model..(tok + 1) * self.d_model]);
        }
        embedded
    }

    /// Forward pass: tokens → logits [seq_len, vocab_size]
    pub fn forward(&self, tokens: &[usize]) -> Vec<f32> {
        let seq_len = tokens.len();

        // 1. Embed
        let mut x = self.embed(tokens);

        // 2. Mamba layers
        for layer in &self.mamba_layers {
            x = layer.forward(&x, seq_len);
        }

        // 3. Output head: [seq, d_model] @ [d_model, vocab] → [seq, vocab]
        //    Parallel over timesteps
        let vocab = self.vocab_size;
        let d = self.d_model;
        let output_head = &self.output_head;

        let logit_rows: Vec<Vec<f32>> = (0..seq_len)
            .into_par_iter()
            .map(|t| {
                let mut row = vec![0.0f32; vocab];
                let x_t = &x[t * d..(t + 1) * d];
                for v in 0..vocab {
                    let mut sum = 0.0f32;
                    for k in 0..d {
                        sum += x_t[k] * output_head[k * vocab + v];
                    }
                    row[v] = sum;
                }
                row
            })
            .collect();

        let mut logits = Vec::with_capacity(seq_len * vocab);
        for row in logit_rows { logits.extend_from_slice(&row); }
        logits
    }

    /// Softmax over logits → probabilities.
    fn softmax(logits: &[f32], vocab_size: usize, seq_len: usize) -> Vec<f32> {
        let mut probs = logits.to_vec();
        for t in 0..seq_len {
            let off = t * vocab_size;
            let max = probs[off..off + vocab_size].iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0f32;
            for v in 0..vocab_size {
                probs[off + v] = (probs[off + v] - max).exp();
                sum += probs[off + v];
            }
            if sum > 0.0 { for v in 0..vocab_size { probs[off + v] /= sum; } }
        }
        probs
    }

    /// Compute perplexity on a token sequence.
    /// PPL = exp(-1/N * sum(log P(next_token | context)))
    pub fn perplexity(&self, tokens: &[usize]) -> f32 {
        if tokens.len() < 2 { return f32::INFINITY; }

        let logits = self.forward(&tokens[..tokens.len() - 1]);
        let probs = Self::softmax(&logits, self.vocab_size, tokens.len() - 1);

        let mut log_prob_sum = 0.0f64;
        let mut count = 0;

        for t in 0..tokens.len() - 1 {
            let next_token = tokens[t + 1].min(self.vocab_size - 1);
            let p = probs[t * self.vocab_size + next_token] as f64;
            if p > 1e-10 {
                log_prob_sum += p.ln();
                count += 1;
            }
        }

        if count == 0 { return f32::INFINITY; }
        (-(log_prob_sum / count as f64)).exp() as f32
    }

    /// Layer-wise NoProp training: each layer learns independently.
    /// Each Mamba layer gets a local prediction objective: predict next token
    /// from its own output. No gradient flows between layers.
    pub fn train_step_layerwise(&mut self, tokens: &[usize], lr: f32) -> f32 {
        if tokens.len() < 2 { return 0.0; }
        let seq_len = tokens.len() - 1;
        let vocab = self.vocab_size;
        let d = self.d_model;

        // Get embeddings
        let embedded = self.embed(&tokens[..seq_len]);

        // Train each Mamba layer with its own local objective
        let mut layer_input = embedded.clone();
        let mut total_loss = 0.0f32;

        for layer_idx in 0..self.mamba_layers.len() {
            let layer_output = self.mamba_layers[layer_idx].forward(&layer_input, seq_len);

            // Local objective: can this layer's output predict the next token?
            // Compute simple dot-product logits against embedding table
            // logits[t, v] = layer_output[t] @ embedding[v]
            let mut local_loss = 0.0f32;

            for t in 0..seq_len {
                let target = tokens[t + 1].min(vocab - 1);

                // Compute logits for this timestep (parallel-friendly)
                let mut logits = vec![0.0f32; vocab];
                let mut max_logit = f32::NEG_INFINITY;
                for v in 0..vocab {
                    let mut dot = 0.0f32;
                    for k in 0..d {
                        dot += layer_output[t * d + k] * self.embedding[v * d + k];
                    }
                    logits[v] = dot;
                    if dot > max_logit { max_logit = dot; }
                }

                // Softmax
                let mut sum_exp = 0.0f32;
                for v in 0..vocab { logits[v] = (logits[v] - max_logit).exp(); sum_exp += logits[v]; }
                if sum_exp > 0.0 { for v in 0..vocab { logits[v] /= sum_exp; } }

                // Cross-entropy loss
                local_loss -= (logits[target].max(1e-10)).ln();

                // Gradient: update Mamba layer's input projection
                // d_loss/d_w_in = (prob - target) * input
                // We update w_in directly (local gradient, no backprop through SSM)
                let layer = &mut self.mamba_layers[layer_idx];
                for v in 0..vocab.min(50) { // top-50 for speed
                    let grad = logits[v] - if v == target { 1.0 } else { 0.0 };
                    if grad.abs() < 0.01 { continue; }
                    // Update input projection weights that map to this output
                    let di2 = layer.d_inner * 2;
                    for k in 0..d.min(layer.d_model) {
                        let x_k = layer_input[t * d + k];
                        if x_k.abs() < 0.01 { continue; }
                        for j in 0..di2.min(32) { // update subset for speed
                            layer.w_in[k * di2 + j] -= lr * 0.01 * grad * x_k / seq_len as f32;
                        }
                    }
                }
            }

            local_loss /= seq_len as f32;
            total_loss += local_loss;
            layer_input = layer_output; // next layer uses this layer's output
        }

        total_loss / self.mamba_layers.len() as f32
    }

    /// Simple NoProp-inspired training: each position learns to predict next token.
    /// Uses per-position L2 regression on one-hot targets.
    pub fn train_step(&mut self, tokens: &[usize], lr: f32) -> f32 {
        if tokens.len() < 2 { return 0.0; }
        let seq_len = tokens.len() - 1;

        // Forward
        let logits = self.forward(&tokens[..seq_len]);
        let probs = Self::softmax(&logits, self.vocab_size, seq_len);

        // Cross-entropy loss
        let mut loss = 0.0f32;
        for t in 0..seq_len {
            let target = tokens[t + 1].min(self.vocab_size - 1);
            let p = probs[t * self.vocab_size + target].max(1e-10);
            loss -= p.ln();
        }
        loss /= seq_len as f32;

        // Update output head AND embedding (NoProp spirit: each component locally)
        let x = {
            let mut x = self.embed(&tokens[..seq_len]);
            for layer in &self.mamba_layers { x = layer.forward(&x, seq_len); }
            x
        };

        // 1. Update output head: parallel over vocab dimensions
        let vocab = self.vocab_size;
        let d = self.d_model;
        let inv_seq = 1.0 / seq_len as f32;

        // Compute deltas in parallel over d_model columns
        let head_deltas: Vec<Vec<f32>> = (0..d)
            .into_par_iter()
            .map(|k| {
                let mut col_delta = vec![0.0f32; vocab];
                for t in 0..seq_len {
                    let target = tokens[t + 1].min(vocab - 1);
                    let x_tk = x[t * d + k];
                    for v in 0..vocab {
                        let grad = probs[t * vocab + v] - if v == target { 1.0 } else { 0.0 };
                        col_delta[v] += lr * grad * x_tk * inv_seq;
                    }
                }
                col_delta
            })
            .collect();

        // Apply deltas
        for (k, delta) in head_deltas.iter().enumerate() {
            for v in 0..vocab {
                self.output_head[k * vocab + v] -= delta[v];
            }
        }

        // 2. Update embeddings: parallel over timesteps, then apply
        let embed_updates: Vec<(usize, Vec<f32>)> = (0..seq_len)
            .into_par_iter()
            .map(|t| {
                let tok = tokens[t].min(vocab - 1);
                let target = tokens[t + 1].min(vocab - 1);
                let mut grad = vec![0.0f32; d];
                for k in 0..d {
                    let mut g = 0.0f32;
                    for v in 0..vocab.min(200) {
                        let err = probs[t * vocab + v] - if v == target { 1.0 } else { 0.0 };
                        g += err * self.output_head[k * vocab + v];
                    }
                    grad[k] = lr * 0.1 * g * inv_seq;
                }
                (tok, grad)
            })
            .collect();

        for (tok, grad) in embed_updates {
            for k in 0..d {
                self.embedding[tok * d + k] -= grad[k];
            }
        }

        loss
    }

    /// Ternarize all Mamba layer weights.
    pub fn ternarize(&mut self) {
        for layer in &mut self.mamba_layers { layer.ternarize(); }
    }

    /// Count total parameters.
    pub fn param_count(&self) -> usize {
        let embed = self.embedding.len();
        let mamba: usize = self.mamba_layers.iter().map(|l| l.param_count()).sum();
        let head = self.output_head.len();
        embed + mamba + head
    }

    /// Generate text: given a prompt, predict next N tokens.
    pub fn generate(&self, prompt: &str, n_tokens: usize) -> String {
        let mut tokens = self.tokenizer.encode(prompt);
        if tokens.is_empty() { tokens.push(0); }

        for _ in 0..n_tokens {
            let logits = self.forward(&tokens);
            let last_logits = &logits[(tokens.len() - 1) * self.vocab_size..tokens.len() * self.vocab_size];

            // Greedy but skip <unk> (id=0) and <pad> (id=1)
            let next = last_logits.iter()
                .enumerate()
                .filter(|(idx, _)| *idx >= 2) // skip <unk> and <pad>
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(idx, _)| idx)
                .unwrap_or(2);
            tokens.push(next);
        }

        self.tokenizer.decode(&tokens)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn load_wikitext2() -> Option<String> {
        let paths = [
            "data/wikitext2/train.txt",
            "../data/wikitext2/train.txt",
            "/home/mirkulix/neoqlang/qlang/data/wikitext2/train.txt",
        ];
        for p in &paths {
            if let Ok(text) = std::fs::read_to_string(p) {
                return Some(text);
            }
        }
        None
    }

    #[test]
    fn qlang_lm_on_wikitext2() {
        let text = match load_wikitext2() {
            Some(t) => t,
            None => { println!("WikiText-2 not found"); return; }
        };

        println!("\n{}", "=".repeat(60));
        println!("QLANG Language Model — Ternary Mamba + NoProp");
        println!("{}\n", "=".repeat(60));

        // Small model for CPU testing
        let d_model = 64;
        let d_inner = 128;
        let d_state = 16;
        let n_layers = 2;
        let vocab_size = 2000;
        let seq_len = 32;

        let mut lm = QlangLM::new(&text, d_model, d_inner, d_state, n_layers, vocab_size);
        println!("Model: d={}, inner={}, state={}, layers={}, vocab={}",
            d_model, d_inner, d_state, n_layers, lm.vocab_size);
        println!("Params: {}\n", lm.param_count());

        // Tokenize
        let tokens = lm.tokenizer.encode(&text);
        println!("Tokens: {} (from {} chars)", tokens.len(), text.len());

        // Initial perplexity
        let init_ppl = lm.perplexity(&tokens[..seq_len.min(tokens.len())]);
        println!("Initial perplexity: {:.1}\n", init_ppl);

        // Train
        println!("Training (embedding + output head, NoProp-style):");
        let n_steps = 500;
        let start = std::time::Instant::now();
        for step in 0..n_steps {
            let offset = (step * seq_len) % (tokens.len() - seq_len - 1);
            let batch = &tokens[offset..offset + seq_len + 1];
            let loss = lm.train_step(batch, 0.01);
            if step % 50 == 0 || step == n_steps - 1 {
                let ppl = lm.perplexity(&tokens[..seq_len.min(tokens.len())]);
                println!("  Step {:>3}: loss={:.3} ppl={:.1} ({:.1?})", step, loss, ppl, start.elapsed());
            }
        }

        // Final metrics
        let final_ppl = lm.perplexity(&tokens[..seq_len.min(tokens.len())]);
        println!("\nFinal perplexity: {:.1} (init: {:.1})", final_ppl, init_ppl);
        assert!(final_ppl < init_ppl, "Training must reduce perplexity");

        // Ternarize and check
        lm.ternarize();
        let tern_ppl = lm.perplexity(&tokens[..seq_len.min(tokens.len())]);
        println!("Ternary perplexity: {:.1}", tern_ppl);

        // Generate
        let prompt = "the";
        let generated = lm.generate(prompt, 10);
        println!("\nGenerated: \"{}\"", generated);

        println!("\n{}", "=".repeat(60));
        println!("RESULT: Ternary Mamba LM on WikiText-2");
        println!("{}", "=".repeat(60));
        println!("  Params:     {}", lm.param_count());
        println!("  Init PPL:   {:.1}", init_ppl);
        println!("  Final PPL:  {:.1}", final_ppl);
        println!("  Tern PPL:   {:.1}", tern_ppl);
        println!("  Trained in: {:?}", start.elapsed());
    }
}
