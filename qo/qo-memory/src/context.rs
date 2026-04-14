use crate::embeddings::embed_text;
use crate::hnsw::VectorStore;
use crate::store::Store;

pub struct MemoryContext {
    vector_store: VectorStore,
    dimension: usize,
}

impl MemoryContext {
    pub fn new(dimension: usize) -> Self {
        Self {
            vector_store: VectorStore::new(),
            dimension,
        }
    }

    /// Add a memory (chat message, goal result, etc.)
    pub fn remember(&mut self, key: String, text: &str, store: &Store) {
        let vector = embed_text(text, self.dimension);
        self.vector_store.insert(key.clone(), vector.clone());
        // Persist embedding as little-endian bytes
        let bytes: Vec<u8> = vector.iter().flat_map(|f| f.to_le_bytes()).collect();
        let _ = store.store_embedding(&key, &bytes);
    }

    /// Search for relevant memories given a query text.
    /// Returns `(key, distance)` pairs sorted by ascending distance (closest first).
    pub fn recall(&self, query: &str, k: usize) -> Vec<(String, f32)> {
        let query_vec = embed_text(query, self.dimension);
        self.vector_store.search(&query_vec, k)
    }

    /// Load persisted embeddings from the store into the in-memory vector index.
    pub fn load_from_store(&mut self, store: &Store) {
        if let Ok(embeddings) = store.load_all_embeddings() {
            for (key, vector) in embeddings {
                self.vector_store.insert(key, vector);
            }
        }
    }

    pub fn count(&self) -> usize {
        self.vector_store.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn remember_and_recall() {
        let dir = TempDir::new().unwrap();
        let store = Store::open(&dir.path().join("test.redb")).unwrap();
        let mut ctx = MemoryContext::new(64);

        ctx.remember("chat_1".into(), "Ich mag Rust Programmierung", &store);
        ctx.remember("chat_2".into(), "Python ist gut für ML", &store);
        ctx.remember("chat_3".into(), "Heute ist schönes Wetter", &store);

        assert_eq!(ctx.count(), 3);

        // Top-2 results are returned; exact order depends on the hash-based embedder
        let results = ctx.recall("Rust Code schreiben", 2);
        assert_eq!(results.len(), 2);
        // Results must be sorted ascending by distance
        assert!(
            results[0].1 <= results[1].1,
            "results should be ordered by ascending distance"
        );
        // Both results must be one of the known keys
        for (key, _) in &results {
            assert!(
                key == "chat_1" || key == "chat_2" || key == "chat_3",
                "unexpected key: {}",
                key
            );
        }
    }

    #[test]
    fn persistence_roundtrip() {
        let dir = TempDir::new().unwrap();
        let store = Store::open(&dir.path().join("test.redb")).unwrap();
        let mut ctx = MemoryContext::new(64);

        ctx.remember("test_1".into(), "Hello World", &store);
        assert_eq!(ctx.count(), 1);

        // Simulate restart
        let mut ctx2 = MemoryContext::new(64);
        ctx2.load_from_store(&store);
        assert_eq!(ctx2.count(), 1);

        let results = ctx2.recall("Hello", 1);
        assert_eq!(results.len(), 1);
    }
}
