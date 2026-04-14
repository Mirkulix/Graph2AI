//! Production LM: vocab=3000, d=128, hidden=256, 2 layers, 50K steps, seq=64
//! Goal: generate real English sentences.

use qlang_runtime::mamba_train::TrainableLM;
use std::time::Instant;

#[test]
fn lm_production() {
    let paths = ["data/wikitext2/train.txt", "../data/wikitext2/train.txt",
        "/home/mirkulix/neoqlang/qlang/data/wikitext2/train.txt"];
    let text = paths.iter().find_map(|p| std::fs::read_to_string(p).ok())
        .unwrap_or_else(|| String::new());
    if text.is_empty() { println!("WikiText-2 not found"); return; }

    println!("\n{}", "=".repeat(60));
    println!("QLANG LM — Production Training");
    println!("{}\n", "=".repeat(60));

    let mut lm = TrainableLM::new(&text, 128, 256, 2, 3000);
    let tokens = lm.tokenizer.encode(&text);
    let seq_len = 64;

    println!("Model: d=128, hidden=256, 2 layers, vocab={}", lm.vocab_size);
    println!("Params: {}", lm.param_count());
    println!("Tokens: {}", tokens.len());

    let init_ppl = lm.perplexity(&tokens[5000..5000 + seq_len]);
    println!("Init PPL: {:.1}\n", init_ppl);

    let start = Instant::now();
    let n_steps = 50000;
    for step in 0..n_steps {
        let off = (step * seq_len) % (tokens.len().saturating_sub(seq_len + 1));
        let batch = &tokens[off..off + seq_len + 1];
        let loss = lm.train_step(batch, 0.005);
        if step % 5000 == 0 || step == n_steps - 1 {
            let ppl = lm.perplexity(&tokens[5000..5000 + seq_len]);
            let gen = lm.generate_with_temp("the", 20, 0.7);
            println!("  Step {:>5}: loss={:.3} ppl={:.1} ({:.1?})", step, loss, ppl, start.elapsed());
            println!("    \"{}\"", gen);
        }
    }

    let final_ppl = lm.perplexity(&tokens[5000..5000 + seq_len]);
    println!("\n{}", "=".repeat(60));
    println!("FINAL PPL: {:.1} (init: {:.1}, {:.1}x)", final_ppl, init_ppl, init_ppl / final_ppl);
    println!("{}", "=".repeat(60));

    let prompts = ["the", "in the", "he was a", "it is the", "they were", 
                   "the first", "after the", "one of the", "during the"];
    for p in &prompts {
        println!("  \"{}\" → \"{}\"", p, lm.generate_with_temp(p, 20, 0.7));
    }

    // Save model weights for QLANG graph export
    let weights_path = "data/lm_production_weights.bin";
    let mut weight_data = Vec::new();
    // Embedding
    let embed_bytes: Vec<u8> = lm.embedding.iter().flat_map(|f| f.to_le_bytes()).collect();
    weight_data.extend_from_slice(&(embed_bytes.len() as u32).to_le_bytes());
    weight_data.extend_from_slice(&embed_bytes);
    // Output head
    let head_bytes: Vec<u8> = lm.output_head.iter().flat_map(|f| f.to_le_bytes()).collect();
    weight_data.extend_from_slice(&(head_bytes.len() as u32).to_le_bytes());
    weight_data.extend_from_slice(&head_bytes);
    // Mamba layers
    for layer in &lm.layers {
        for w in [&layer.w_x, &layer.w_h, &layer.w_gate, &layer.w_out] {
            let bytes: Vec<u8> = w.iter().flat_map(|f| f.to_le_bytes()).collect();
            weight_data.extend_from_slice(&(bytes.len() as u32).to_le_bytes());
            weight_data.extend_from_slice(&bytes);
        }
    }
    let _ = std::fs::write(weights_path, &weight_data);
    println!("\nWeights saved: {} ({} bytes)", weights_path, weight_data.len());
}
