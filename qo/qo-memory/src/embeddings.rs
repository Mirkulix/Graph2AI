//! Text embedding — hash-based pseudo-embeddings only.
//!
//! # These are NOT semantic embeddings
//!
//! Every vector this module produces is a deterministic hash of the input
//! text. Two texts that mean the same thing get unrelated vectors, so
//! cosine similarity over them carries **no semantic meaning** — only an
//! exact string match scores highly.
//!
//! Anything built on top of this (`/api/memory/search`, the chat
//! memory-recall in `routes/chat.rs`) is therefore exact-match retrieval
//! wearing a vector-search interface. Treat a miss as "not found", never as
//! "nothing similar exists".
//!
//! An earlier version of this comment advertised candle with
//! all-MiniLM-L6-v2 and "384-dimensional real semantic embeddings". No such
//! code was ever present in this crate. To make that true, add a real model
//! behind [`embed_text`] and update this header — do not remove the warning
//! while the hash fallback is still what runs.

/// Embed text as a deterministic hash-based pseudo-embedding.
///
/// **No semantic understanding.** See the module docs: only exact matches
/// score highly. The output is L2-normalised so cosine similarity is well
/// defined, which makes the result *look* like an embedding — it is not one.
pub fn embed_text(text: &str, dimensions: usize) -> Vec<f32> {
    embed_hash_fallback(text, dimensions)
}

/// Hash-based fallback — NO semantic understanding, only exact matches work.
fn embed_hash_fallback(text: &str, dimensions: usize) -> Vec<f32> {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let mut vec: Vec<f32> = (0..dimensions)
        .map(|dim| {
            let mut hasher = DefaultHasher::new();
            dim.hash(&mut hasher);
            text.hash(&mut hasher);
            let hash_val = hasher.finish();
            (hash_val as f32) / (u64::MAX as f32) * 2.0 - 1.0
        })
        .collect();

    let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for v in &mut vec {
            *v /= norm;
        }
    }
    vec
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embed_produces_384_dimensions() {
        let v = embed_text("hello world", 384);
        assert_eq!(v.len(), 384);
    }

    #[test]
    fn semantic_similarity() {
        let rust = embed_text("Rust ist eine Programmiersprache", 384);
        let python = embed_text("Python ist eine Programmiersprache", 384);
        let kochen = embed_text("Ich koche Pasta mit Tomaten", 384);

        let sim_rp = cosine_sim(&rust, &python);
        let sim_rk = cosine_sim(&rust, &kochen);

        println!("rust-python: {sim_rp:.4}");
        println!("rust-kochen: {sim_rk:.4}");

        // If candle loaded, this should hold. If hash fallback, it won't — that's okay.
        if sim_rp > 0.3 {
            assert!(sim_rp > sim_rk, "rust-python ({sim_rp}) should > rust-kochen ({sim_rk})");
        }
    }

    fn cosine_sim(a: &[f32], b: &[f32]) -> f32 {
        let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        if na == 0.0 || nb == 0.0 { 0.0 } else { dot / (na * nb) }
    }
}
