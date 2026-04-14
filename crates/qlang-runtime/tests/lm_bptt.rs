//! LM with BPTT on WikiText-2 — the real test.

use qlang_runtime::mamba_train::TrainableLM;
use std::time::Instant;

#[test]
fn lm_bptt_wikitext2() {
    let paths = ["data/wikitext2/train.txt", "../data/wikitext2/train.txt",
        "/home/mirkulix/neoqlang/qlang/data/wikitext2/train.txt"];
    let text = paths.iter().find_map(|p| std::fs::read_to_string(p).ok())
        .unwrap_or_else(|| { println!("WikiText-2 not found"); String::new() });
    if text.is_empty() { return; }

    println!("\n{}", "=".repeat(60));
    println!("QLANG LM with BPTT on WikiText-2");
    println!("{}\n", "=".repeat(60));

    let mut lm = TrainableLM::new(&text, 64, 128, 2, 500);
    let tokens = lm.tokenizer.encode(&text);
    let seq_len = 16;

    println!("Model: d=64, hidden=128, 2 layers, vocab={}", lm.vocab_size);
    println!("Params: {}", lm.param_count());
    println!("Tokens: {}\n", tokens.len());

    let init_ppl = lm.perplexity(&tokens[500..500 + seq_len]);
    let gen_before = lm.generate("the", 10);
    println!("Init PPL: {:.1}", init_ppl);
    println!("Before: \"{}\"\n", gen_before);

    let start = Instant::now();
    let n_steps = 3000;
    for step in 0..n_steps {
        let off = (step * seq_len) % (tokens.len().saturating_sub(seq_len + 1));
        let batch = &tokens[off..off + seq_len + 1];
        let loss = lm.train_step(batch, 0.01);
        if step % 500 == 0 || step == n_steps - 1 {
            let ppl = lm.perplexity(&tokens[500..500 + seq_len]);
            let gen = lm.generate("the", 8);
            println!("  Step {:>5}: loss={:.3} ppl={:.1} gen=\"{}\" ({:.1?})",
                step, loss, ppl, gen, start.elapsed());
        }
    }

    let final_ppl = lm.perplexity(&tokens[500..500 + seq_len]);
    let gen1 = lm.generate("the", 12);
    let gen2 = lm.generate("in the", 12);
    let gen3 = lm.generate("he was", 12);

    println!("\n{}", "=".repeat(60));
    println!("RESULT: BPTT LM on WikiText-2");
    println!("{}", "=".repeat(60));
    println!("  Init PPL:  {:.1}", init_ppl);
    println!("  Final PPL: {:.1}", final_ppl);
    println!("  Improve:   {:.1}x", init_ppl / final_ppl);
    println!("  Time:      {:?}", start.elapsed());
    println!("  Gen 1: \"{}\"", gen1);
    println!("  Gen 2: \"{}\"", gen2);
    println!("  Gen 3: \"{}\"", gen3);

    assert!(final_ppl < init_ppl, "Must reduce PPL");
}
