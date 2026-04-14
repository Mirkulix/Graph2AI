use std::collections::HashMap;

pub struct VectorStore {
    vectors: HashMap<String, Vec<f32>>,
}

impl VectorStore {
    pub fn new() -> Self {
        Self {
            vectors: HashMap::new(),
        }
    }

    pub fn insert(&mut self, key: impl Into<String>, vector: Vec<f32>) {
        self.vectors.insert(key.into(), vector);
    }

    /// Returns the top-k most similar entries (by cosine similarity, i.e. lowest distance).
    pub fn search(&self, query: &[f32], k: usize) -> Vec<(String, f32)> {
        let mut scored: Vec<(String, f32)> = self
            .vectors
            .iter()
            .map(|(key, vec)| (key.clone(), cosine_distance(query, vec)))
            .collect();

        // Sort ascending by distance (closest first)
        scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(k);
        scored
    }

    pub fn len(&self) -> usize {
        self.vectors.len()
    }

    pub fn is_empty(&self) -> bool {
        self.vectors.is_empty()
    }
}

impl Default for VectorStore {
    fn default() -> Self {
        Self::new()
    }
}

/// Cosine distance in [0, 2]. Returns 0.0 for identical vectors.
pub fn cosine_distance(a: &[f32], b: &[f32]) -> f32 {
    if a.is_empty() || b.is_empty() || a.len() != b.len() {
        return 2.0;
    }
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        return 2.0;
    }
    let similarity = dot / (norm_a * norm_b);
    1.0 - similarity.clamp(-1.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn insert_and_search() {
        let mut store = VectorStore::new();
        store.insert("a", vec![1.0, 0.0, 0.0]);
        store.insert("b", vec![0.0, 1.0, 0.0]);
        store.insert("c", vec![0.0, 0.0, 1.0]);

        assert_eq!(store.len(), 3);
        assert!(!store.is_empty());

        // Search with a query close to "a"
        let results = store.search(&[1.0, 0.0, 0.0], 1);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, "a");
        assert!(results[0].1 < 0.01, "distance should be near 0 for identical vector");
    }

    #[test]
    fn empty_store_search() {
        let store = VectorStore::new();
        assert!(store.is_empty());
        let results = store.search(&[1.0, 0.0, 0.0], 5);
        assert!(results.is_empty());
    }

    #[test]
    fn test_cosine_identical_vectors() {
        let v = vec![1.0, 2.0, 3.0];
        let dist = cosine_distance(&v, &v);
        assert!(dist.abs() < 1e-6, "identical vectors should have distance ~0, got {}", dist);
    }

    #[test]
    fn test_cosine_orthogonal_vectors() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];
        let dist = cosine_distance(&a, &b);
        assert!((dist - 1.0).abs() < 1e-6, "orthogonal vectors should have distance ~1, got {}", dist);
    }
}
