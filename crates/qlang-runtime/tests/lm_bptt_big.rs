//! Big BPTT LM: d=128, hidden=256, 2 layers, vocab=1000, 20K steps, seq=32

use qlang_runtime::mamba_train::TrainableLM;
use std::time::Instant;

#[test]
fn lm_bptt_big() {
    let paths = ["data/wikitext2/train.txt", "../data/wikitext2/train.txt",
        "/home/mirkulix/neoqlang/qlang/data/wikitext2/train.txt"];
    let text = paths.iter().find_map(|p| std::fs::read_to_string(p).ok())
        .unwrap_or_else(|| String::new());
    if text.is_empty() { println!("WikiText-2 not found"); return; }

    println!("\n{}", "=".repeat(60));
    println!("QLANG LM — Big BPTT Training");
    println!("{}\n", "=".repeat(60));

    let mut lm = TrainableLM::new(&text, 128, 256, 2, 1000);
    let tokens = lm.tokenizer.encode(&text);
    let seq_len = 32;

    println!("Model: d=128, hidden=256, 2 layers, vocab={}", lm.vocab_size);
    println!("Params: {}", lm.param_count());
    println!("Tokens: {}\n", tokens.len());

    let init_ppl = lm.perplexity(&tokens[500..500 + seq_len]);
    println!("Init PPL: {:.1}\n", init_ppl);

    let start = Instant::now();
    for step in 0..10000 {
        let off = (step * seq_len) % (tokens.len().saturating_sub(seq_len + 1));
        let batch = &tokens[off..off + seq_len + 1];
        let loss = lm.train_step(batch, 0.008);
        if step % 1000 == 0 || step == 9999 {
            let ppl = lm.perplexity(&tokens[2000..2000 + seq_len]);
            let gen = lm.generate("the", 12);
            println!("  Step {:>5}: loss={:.3} ppl={:.1} ({:.1?})", step, loss, ppl, start.elapsed());
            println!("    gen: \"{}\"", gen);
        }
    }

    let final_ppl = lm.perplexity(&tokens[2000..2000 + seq_len]);
    println!("\n{}", "=".repeat(60));
    println!("FINAL PPL: {:.1} (init: {:.1}, {:.1}x)", final_ppl, init_ppl, init_ppl / final_ppl);
    println!("{}", "=".repeat(60));

    println!("\n--- Temperature Sampling ---");
    let prompts = ["the", "in the", "he was", "it is", "they were", "she had"];
    for p in &prompts {
        println!("  \"{}\" → \"{}\"", p, lm.generate_with_temp(p, 15, 0.8));
    }
    println!("\n--- Low Temperature (0.3) ---");
    for p in &prompts[..3] {
        println!("  \"{}\" → \"{}\"", p, lm.generate_with_temp(p, 15, 0.3));
    }
    println!("\n--- High Temperature (1.2) ---");
    for p in &prompts[..3] {
        println!("  \"{}\" → \"{}\"", p, lm.generate_with_temp(p, 15, 1.2));
    }
}
