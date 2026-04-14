//! T011 — verify tokenizer vocab is embedded in QLMB checkpoints and
//! reconstructed byte-identically on load, so generations contain no
//! spurious `<unk>` tokens caused by vocab-order drift.

use qlang_runtime::mamba_train::{load_mamba_model, TrainableLM};

/// A deterministic mini-corpus with enough variety that the frequency-based
/// tokenizer has ambiguous tie-breaks — reliably exposes any re-derivation
/// that yields a different vocab ordering than the original.
fn corpus() -> &'static str {
    "the quick brown fox jumps over the lazy dog \
     a stitch in time saves nine \
     all that glitters is not gold \
     the pen is mightier than the sword \
     to be or not to be that is the question \
     rome was not built in a day \
     birds of a feather flock together \
     a journey of a thousand miles begins with a single step"
}

fn save_path() -> String {
    let dir = std::env::temp_dir().join("qlang_t011_tokenizer_roundtrip");
    std::fs::create_dir_all(&dir).ok();
    dir.join("model.qlmb").to_string_lossy().into_owned()
}

#[test]
fn tokenizer_vocab_survives_qlmb_roundtrip() {
    let text = corpus();
    let path = save_path();

    // 1. Build tokenizer + tiny model, train a single step so weights are
    //    non-trivial (exercises the full save path).
    let mut lm = TrainableLM::new(text, 16, 32, 1, 64);
    let original_vocab = lm.tokenizer.id2word.clone();
    let original_word2id = lm.tokenizer.word2id.clone();
    assert!(original_vocab.len() >= 3, "corpus should yield a non-trivial vocab");

    let tokens = lm.tokenizer.encode(text);
    assert!(tokens.len() > 4, "need enough tokens for a step");
    lm.train_step(&tokens[..tokens.len().min(8)], 0.01);

    // 2. Save — vocab is embedded in the QLMB via the "VOCB" section.
    qlang_runtime::gpu_train::save_weights_for_test(&lm, &path);

    // 3. Reload with an INTENTIONALLY DIFFERENT reference text — this would
    //    have produced a mismatched vocab under the old behaviour.
    let misleading = "totally different words that do not appear in corpus xyz";
    let reloaded = load_mamba_model(&path, misleading)
        .expect("load_mamba_model must succeed on freshly saved QLMB");

    // 4. Vocab must be byte-identical (same words, same IDs).
    assert_eq!(
        reloaded.tokenizer.id2word, original_vocab,
        "id2word order must be preserved across save/load"
    );
    assert_eq!(
        reloaded.tokenizer.vocab_size, original_vocab.len(),
        "vocab_size must match"
    );
    for (w, id) in &original_word2id {
        assert_eq!(
            reloaded.tokenizer.word2id.get(w),
            Some(id),
            "word2id mapping drifted for {:?}", w
        );
    }

    // 5. Generate and confirm no <unk> tokens in first 20 generations from a
    //    known-in-vocab prompt. The LM is tiny and barely trained, so we
    //    check the TOKENIZER behaviour (decoded output should only use
    //    known vocab words — never the <unk> sentinel).
    let gen = reloaded.generate_with_temp("the", 20, 0.7);
    assert!(
        !gen.split_whitespace().any(|w| w == "<unk>"),
        "generation must not contain <unk> tokens, got: {:?}", gen
    );

    // 6. Backwards compatibility: even if we feed empty reference text, the
    //    embedded vocab still wins.
    let reloaded_empty = load_mamba_model(&path, "")
        .expect("load with empty reference text must still succeed");
    assert_eq!(reloaded_empty.tokenizer.id2word, original_vocab);
}
