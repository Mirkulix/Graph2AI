//! Real LM Training: 5000 steps on WikiText-2, measure perplexity + generation quality.

use qlang_runtime::qlang_lm::QlangLM;
use std::time::Instant;

fn load_wikitext2() -> Option<String> {
    let paths = [
        "data/wikitext2/train.txt",
        "../data/wikitext2/train.txt",
        "/home/mirkulix/neoqlang/qlang/data/wikitext2/train.txt",
    ];
    for p in &paths {
        if let Ok(text) = std::fs::read_to_string(p) { return Some(text); }
    }
    None
}

#[test]
fn lm_5000_steps() {
    let text = match load_wikitext2() {
        Some(t) => t,
        None => { println!("WikiText-2 not found"); return; }
    };

    println!("\n{}", "=".repeat(60));
    println!("QLANG LM — Real Training (5000 steps)");
    println!("{}\n", "=".repeat(60));

    // Smaller vocab, bigger model proportion
    let mut lm = QlangLM::new(&text, 64, 128, 16, 1, 500);
    println!("Model: d=64, inner=128, state=16, 1 layer, vocab={}", lm.tokenizer.vocab_size);
    println!("Params: {}\n", lm.param_count());

    let tokens = lm.tokenizer.encode(&text);
    let seq_len = 16; // shorter sequences for faster iteration

    let init_ppl = lm.perplexity(&tokens[..seq_len]);
    println!("Initial PPL: {:.1}\n", init_ppl);

    // Generate before training
    let gen_before = lm.generate("the", 15);
    println!("Before: \"{}\"\n", gen_before);

    // Aggressive training: higher LR, more steps, output head only (proven to work)
    let start = Instant::now();
    let n_steps = 10000;
    let mut last_loss = 0.0f32;
    for step in 0..n_steps {
        let offset = (step * seq_len) % (tokens.len().saturating_sub(seq_len + 1));
        let batch = &tokens[offset..offset + seq_len + 1];
        last_loss = lm.train_step(batch, 0.02);
        if step % 2000 == 0 || step == n_steps - 1 {
            let ppl = lm.perplexity(&tokens[1000..1000 + seq_len]);
            let gen = lm.generate("the", 8);
            println!("  Step {:>5}: loss={:.3} ppl={:.1} gen=\"{}\" ({:.1?})",
                step, last_loss, ppl, gen, start.elapsed());
        }
    }

    let final_ppl = lm.perplexity(&tokens[1000..1000 + seq_len]);
    let gen_after = lm.generate("the", 15);
    let gen2 = lm.generate("in the", 15);
    let gen3 = lm.generate("he was", 15);

    // Ternarize and test
    lm.ternarize();
    let tern_ppl = lm.perplexity(&tokens[1000..1000 + seq_len]);
    let gen_tern = lm.generate("the", 15);

    println!("\n{}", "=".repeat(60));
    println!("RESULT");
    println!("{}", "=".repeat(60));
    println!("  Init PPL:    {:.1}", init_ppl);
    println!("  Final PPL:   {:.1}", final_ppl);
    println!("  Ternary PPL: {:.1}", tern_ppl);
    println!("  Improvement: {:.1}x", init_ppl / final_ppl);
    println!("  Time:        {:?}", start.elapsed());
    println!();
    println!("  Gen 1: \"{}\"", gen_after);
    println!("  Gen 2: \"{}\"", gen2);
    println!("  Gen 3: \"{}\"", gen3);
    println!("  Ternary: \"{}\"", gen_tern);

    assert!(final_ppl < init_ppl, "Training must reduce perplexity");
}
