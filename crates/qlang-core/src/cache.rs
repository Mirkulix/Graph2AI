//! Computation Cache — content-addressable caching for graph executions.
//!
//! Every graph computation gets a unique SHA-256 hash derived from the
//! graph structure plus its inputs. If the same computation was done
//! before, the cached result is returned instead of recomputing.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

use crate::crypto::sha256;
use crate::tensor::TensorData;

/// Global computation cache — content-addressable.
static CACHE: OnceLock<Arc<Mutex<ComputationCache>>> = OnceLock::new();

/// Content-addressable computation cache.
///
/// Maps `hash(graph + inputs)` to cached output tensors.
/// Uses LRU-style eviction: when the cache is full, the oldest entry
/// (by creation time) is evicted.
pub struct ComputationCache {
    /// hash(graph + inputs) -> outputs
    entries: HashMap<[u8; 32], CacheEntry>,
    /// Total cache hits since creation or last clear.
    hits: u64,
    /// Total cache misses since creation or last clear.
    misses: u64,
    /// Maximum number of entries before eviction kicks in.
    max_entries: usize,
}

/// A single cached computation result.
pub struct CacheEntry {
    /// The output tensors produced by this computation.
    pub outputs: HashMap<String, TensorData>,
    /// Wall-clock time the original computation took, in microseconds.
    pub compute_time_us: u64,
    /// When this entry was created.
    pub created: std::time::Instant,
}

impl ComputationCache {
    /// Get (or initialize) the global computation cache.
    pub fn global() -> Arc<Mutex<ComputationCache>> {
        CACHE
            .get_or_init(|| {
                Arc::new(Mutex::new(ComputationCache {
                    entries: HashMap::new(),
                    hits: 0,
                    misses: 0,
                    max_entries: 10000,
                }))
            })
            .clone()
    }

    /// Create a new cache with a custom capacity.
    pub fn with_capacity(max_entries: usize) -> Self {
        ComputationCache {
            entries: HashMap::new(),
            hits: 0,
            misses: 0,
            max_entries,
        }
    }

    /// Compute a cache key from a graph hash plus input data hashes.
    ///
    /// The key is `SHA-256(graph_hash || input_hash_0 || input_hash_1 || ...)`.
    pub fn cache_key(graph_hash: &[u8; 32], input_hashes: &[&[u8]]) -> [u8; 32] {
        let mut data = Vec::with_capacity(32 + input_hashes.len() * 32);
        data.extend_from_slice(graph_hash);
        for ih in input_hashes {
            data.extend_from_slice(ih);
        }
        sha256(&data)
    }

    /// Look up a cached entry by key.
    ///
    /// Increments the hit or miss counter accordingly.
    pub fn get(&mut self, key: &[u8; 32]) -> Option<&CacheEntry> {
        if let Some(entry) = self.entries.get(key) {
            self.hits += 1;
            Some(entry)
        } else {
            self.misses += 1;
            None
        }
    }

    /// Insert a computation result into the cache.
    ///
    /// If the cache is at capacity, evicts the oldest entry first.
    pub fn insert(&mut self, key: [u8; 32], entry: CacheEntry) {
        if self.entries.len() >= self.max_entries {
            // Evict oldest entry
            if let Some(oldest_key) = self
                .entries
                .iter()
                .min_by_key(|(_, e)| e.created)
                .map(|(k, _)| *k)
            {
                self.entries.remove(&oldest_key);
            }
        }
        self.entries.insert(key, entry);
    }

    /// Returns `(hits, misses, current_size)`.
    pub fn stats(&self) -> (u64, u64, usize) {
        (self.hits, self.misses, self.entries.len())
    }

    /// Clear all cached entries and reset counters.
    pub fn clear(&mut self) {
        self.entries.clear();
        self.hits = 0;
        self.misses = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::{Shape, TensorData};

    #[test]
    fn cache_hit_miss() {
        let mut cache = ComputationCache::with_capacity(100);

        let graph_hash = sha256(b"test-graph");
        let input_hash = sha256(b"test-input");
        let key = ComputationCache::cache_key(&graph_hash, &[&input_hash]);

        // Miss
        assert!(cache.get(&key).is_none());
        assert_eq!(cache.stats(), (0, 1, 0));

        // Insert
        let mut outputs = HashMap::new();
        outputs.insert(
            "y".to_string(),
            TensorData::from_f32(Shape::vector(2), &[1.0, 2.0]),
        );
        cache.insert(
            key,
            CacheEntry {
                outputs,
                compute_time_us: 42,
                created: std::time::Instant::now(),
            },
        );

        // Hit
        let entry = cache.get(&key).unwrap();
        assert_eq!(entry.compute_time_us, 42);
        assert_eq!(cache.stats(), (1, 1, 1));
    }

    #[test]
    fn cache_eviction() {
        let mut cache = ComputationCache::with_capacity(2);

        for i in 0u8..3 {
            let key = sha256(&[i]);
            let mut outputs = HashMap::new();
            outputs.insert(
                format!("out_{i}"),
                TensorData::from_f32(Shape::scalar(), &[i as f32]),
            );
            cache.insert(
                key,
                CacheEntry {
                    outputs,
                    compute_time_us: i as u64,
                    created: std::time::Instant::now(),
                },
            );
        }

        // Should have evicted one, so only 2 entries remain
        assert_eq!(cache.entries.len(), 2);
    }

    #[test]
    fn cache_clear() {
        let mut cache = ComputationCache::with_capacity(100);
        let key = sha256(b"something");
        cache.insert(
            key,
            CacheEntry {
                outputs: HashMap::new(),
                compute_time_us: 0,
                created: std::time::Instant::now(),
            },
        );
        assert_eq!(cache.stats().2, 1);

        cache.clear();
        assert_eq!(cache.stats(), (0, 0, 0));
    }

    #[test]
    fn cache_key_determinism() {
        let gh = sha256(b"graph-1");
        let ih = sha256(b"input-data");

        let k1 = ComputationCache::cache_key(&gh, &[&ih]);
        let k2 = ComputationCache::cache_key(&gh, &[&ih]);
        assert_eq!(k1, k2);

        // Different input -> different key
        let ih2 = sha256(b"input-data-2");
        let k3 = ComputationCache::cache_key(&gh, &[&ih2]);
        assert_ne!(k1, k3);
    }

    #[test]
    fn content_hash_determinism() {
        let data = b"same data";
        let h1 = sha256(data);
        let h2 = sha256(data);
        assert_eq!(h1, h2);
    }
}
