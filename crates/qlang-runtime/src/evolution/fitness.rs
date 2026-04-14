//! Fitness Tracker — measures specialist performance for evolutionary selection.
//!
//! Records per-specialist accuracy, latency, success-rate and usage across generations.
//! Persists snapshots to a `redb` key-value store (no rusqlite dependency required).
//!
//! Coordination namespace: `qlang-evolution/fitness-tracker-status`

use rayon::prelude::*;
use redb::{Database, ReadableTable, TableDefinition};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};

const RECORDS_TABLE: TableDefinition<&str, Vec<u8>> = TableDefinition::new("fitness_records");
const HISTORY_TABLE: TableDefinition<u32, Vec<u8>> = TableDefinition::new("generation_history");

/// Parallelization threshold: above this many records, use rayon for ranking.
const PARALLEL_THRESHOLD: usize = 100;

fn now_unix() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// Per-specialist fitness data collected during runtime.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FitnessRecord {
    pub specialist_id: String,
    pub generation: u32,
    pub birth_time: u64,
    pub invocations: u64,
    pub successes: u64,
    pub accuracy_samples: Vec<f32>,
    pub latency_samples_us: Vec<u64>,
    pub energy_estimate: f32,
    pub parent_id: Option<String>,
    pub mutation_count: u32,
}

impl FitnessRecord {
    pub fn new(id: String, generation: u32, parent: Option<String>) -> Self {
        Self {
            specialist_id: id,
            generation,
            birth_time: now_unix(),
            invocations: 0,
            successes: 0,
            accuracy_samples: Vec::new(),
            latency_samples_us: Vec::new(),
            energy_estimate: 0.0,
            parent_id: parent,
            mutation_count: 0,
        }
    }

    /// Record an invocation with its result.
    pub fn record_invocation(&mut self, success: bool, accuracy: Option<f32>, latency_us: u64) {
        self.invocations += 1;
        if success {
            self.successes += 1;
        }
        if let Some(a) = accuracy {
            self.accuracy_samples.push(a.clamp(0.0, 1.0));
        }
        self.latency_samples_us.push(latency_us);
        // Rough energy proxy: microseconds of compute per call.
        self.energy_estimate += (latency_us as f32) * 1e-6;
    }

    /// Mean accuracy from samples (0.0 if no samples).
    pub fn mean_accuracy(&self) -> f32 {
        if self.accuracy_samples.is_empty() {
            return 0.0;
        }
        let sum: f32 = self.accuracy_samples.iter().sum();
        sum / self.accuracy_samples.len() as f32
    }

    /// Success rate (0..=1).
    pub fn success_rate(&self) -> f32 {
        if self.invocations == 0 {
            return 0.0;
        }
        (self.successes as f32) / (self.invocations as f32)
    }

    /// Mean latency in microseconds (0 if no samples).
    pub fn mean_latency_us(&self) -> f64 {
        if self.latency_samples_us.is_empty() {
            return 0.0;
        }
        let sum: u64 = self.latency_samples_us.iter().sum();
        (sum as f64) / (self.latency_samples_us.len() as f64)
    }

    /// Composite fitness score (0.0 - 1.0).
    /// Weighted: accuracy 40%, success_rate 30%, usage 20%, speed 10%.
    pub fn fitness_score(&self) -> f32 {
        let acc = self.mean_accuracy().clamp(0.0, 1.0);
        let sr = self.success_rate().clamp(0.0, 1.0);
        // Usage: saturates at 100 invocations.
        let usage = ((self.invocations as f32) / 100.0).min(1.0);
        // Speed: inverse latency, normalised against a 10ms reference.
        let mean_us = self.mean_latency_us() as f32;
        let speed = if mean_us <= 0.0 {
            // No data yet → neutral 0.5 so new specialists aren't punished.
            0.5
        } else {
            (10_000.0 / (10_000.0 + mean_us)).clamp(0.0, 1.0)
        };
        let score = 0.40 * acc + 0.30 * sr + 0.20 * usage + 0.10 * speed;
        score.clamp(0.0, 1.0)
    }

    /// Age in seconds.
    pub fn age_seconds(&self) -> u64 {
        now_unix().saturating_sub(self.birth_time)
    }
}

/// Aggregate statistics for a population snapshot.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationStats {
    pub generation: u32,
    pub timestamp: u64,
    pub population_size: usize,
    pub avg_fitness: f32,
    pub max_fitness: f32,
    pub min_fitness: f32,
    pub best_specialist_id: String,
}

/// Tracks fitness of all specialists across generations.
pub struct FitnessTracker {
    records: HashMap<String, FitnessRecord>,
    generation_history: Vec<GenerationStats>,
    current_generation: u32,
}

impl Default for FitnessTracker {
    fn default() -> Self {
        Self::new()
    }
}

impl FitnessTracker {
    pub fn new() -> Self {
        Self {
            records: HashMap::new(),
            generation_history: Vec::new(),
            current_generation: 0,
        }
    }

    /// Register a new specialist.
    pub fn register(&mut self, id: String, generation: u32, parent: Option<String>) {
        if generation > self.current_generation {
            self.current_generation = generation;
        }
        self.records
            .entry(id.clone())
            .or_insert_with(|| FitnessRecord::new(id, generation, parent));
    }

    /// Record an invocation result for a specialist (no-op if unknown id).
    pub fn record(&mut self, id: &str, success: bool, accuracy: Option<f32>, latency_us: u64) {
        if let Some(r) = self.records.get_mut(id) {
            r.record_invocation(success, accuracy, latency_us);
        }
    }

    pub fn get(&self, id: &str) -> Option<&FitnessRecord> {
        self.records.get(id)
    }

    pub fn len(&self) -> usize {
        self.records.len()
    }

    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }

    fn score_all(&self) -> Vec<(String, f32)> {
        if self.records.len() > PARALLEL_THRESHOLD {
            self.records
                .par_iter()
                .map(|(k, v)| (k.clone(), v.fitness_score()))
                .collect()
        } else {
            self.records
                .iter()
                .map(|(k, v)| (k.clone(), v.fitness_score()))
                .collect()
        }
    }

    /// Top-N specialists by fitness (descending).
    pub fn top_n(&self, n: usize) -> Vec<(String, f32)> {
        let mut scored = self.score_all();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(n);
        scored
    }

    /// Bottom-N specialists by fitness (ascending — worst first).
    pub fn bottom_n(&self, n: usize) -> Vec<(String, f32)> {
        let mut scored = self.score_all();
        scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(n);
        scored
    }

    /// Compute and store a GenerationStats snapshot of current population.
    pub fn snapshot_generation(&mut self) -> GenerationStats {
        let scored = self.score_all();
        let population_size = scored.len();
        let (sum, min, max, best) = scored.iter().fold(
            (0.0f32, f32::INFINITY, f32::NEG_INFINITY, String::new()),
            |(s, mn, mx, best), (id, sc)| {
                let new_best = if *sc > mx { id.clone() } else { best };
                (s + sc, mn.min(*sc), mx.max(*sc), new_best)
            },
        );
        let avg = if population_size == 0 { 0.0 } else { sum / population_size as f32 };
        let (min_fitness, max_fitness) = if population_size == 0 {
            (0.0, 0.0)
        } else {
            (min, max)
        };
        let stats = GenerationStats {
            generation: self.current_generation,
            timestamp: now_unix(),
            population_size,
            avg_fitness: avg,
            max_fitness,
            min_fitness,
            best_specialist_id: best,
        };
        self.generation_history.push(stats.clone());
        self.current_generation = self.current_generation.saturating_add(1);
        stats
    }

    pub fn history(&self) -> &[GenerationStats] {
        &self.generation_history
    }

    /// Persist tracker state to a redb file.
    pub fn to_sqlite(&self, path: &str) -> Result<(), String> {
        // Name kept for API parity with spec; redb is used under the hood.
        let db = Database::create(path).map_err(|e| format!("redb create: {e}"))?;
        let wtx = db.begin_write().map_err(|e| format!("begin_write: {e}"))?;
        {
            let mut tbl = wtx
                .open_table(RECORDS_TABLE)
                .map_err(|e| format!("open records: {e}"))?;
            for (id, rec) in &self.records {
                let bytes = bincode::serialize(rec).map_err(|e| format!("ser rec: {e}"))?;
                tbl.insert(id.as_str(), bytes)
                    .map_err(|e| format!("insert rec: {e}"))?;
            }
            let mut htbl = wtx
                .open_table(HISTORY_TABLE)
                .map_err(|e| format!("open history: {e}"))?;
            for (i, g) in self.generation_history.iter().enumerate() {
                let bytes = bincode::serialize(g).map_err(|e| format!("ser gen: {e}"))?;
                htbl.insert(i as u32, bytes)
                    .map_err(|e| format!("insert gen: {e}"))?;
            }
        }
        wtx.commit().map_err(|e| format!("commit: {e}"))?;
        Ok(())
    }

    /// Load tracker state from a redb file.
    pub fn from_sqlite(path: &str) -> Result<Self, String> {
        let db = Database::open(path).map_err(|e| format!("redb open: {e}"))?;
        let rtx = db.begin_read().map_err(|e| format!("begin_read: {e}"))?;
        let mut records = HashMap::new();
        let mut current_generation = 0u32;
        if let Ok(tbl) = rtx.open_table(RECORDS_TABLE) {
            for entry in tbl.iter().map_err(|e| format!("iter rec: {e}"))? {
                let (k, v) = entry.map_err(|e| format!("entry rec: {e}"))?;
                let rec: FitnessRecord =
                    bincode::deserialize(&v.value()).map_err(|e| format!("de rec: {e}"))?;
                current_generation = current_generation.max(rec.generation);
                records.insert(k.value().to_string(), rec);
            }
        }
        let mut generation_history = Vec::new();
        if let Ok(htbl) = rtx.open_table(HISTORY_TABLE) {
            for entry in htbl.iter().map_err(|e| format!("iter gen: {e}"))? {
                let (_k, v) = entry.map_err(|e| format!("entry gen: {e}"))?;
                let g: GenerationStats =
                    bincode::deserialize(&v.value()).map_err(|e| format!("de gen: {e}"))?;
                current_generation = current_generation.max(g.generation);
                generation_history.push(g);
            }
        }
        Ok(Self {
            records,
            generation_history,
            current_generation,
        })
    }

    /// One-line status string (for shared memory namespace
    /// `qlang-evolution/fitness-tracker-status`).
    pub fn status_line(&self) -> String {
        let top = self.top_n(1);
        let (best_id, best_score) = top
            .first()
            .map(|(i, s)| (i.clone(), *s))
            .unwrap_or_else(|| ("<none>".to_string(), 0.0));
        format!(
            "gen={} pop={} best={} score={:.3} generations_recorded={}",
            self.current_generation,
            self.records.len(),
            best_id,
            best_score,
            self.generation_history.len()
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn populate(tracker: &mut FitnessTracker) {
        tracker.register("alpha".into(), 0, None);
        tracker.register("beta".into(), 0, None);
        tracker.register("gamma".into(), 0, Some("alpha".into()));

        // alpha: strong — high accuracy, high success, fast
        for _ in 0..50 {
            tracker.record("alpha", true, Some(0.95), 500);
        }
        // beta: middling
        for _ in 0..20 {
            tracker.record("beta", true, Some(0.60), 3000);
        }
        for _ in 0..10 {
            tracker.record("beta", false, Some(0.40), 3000);
        }
        // gamma: weak — slow and low accuracy
        for _ in 0..5 {
            tracker.record("gamma", false, Some(0.10), 50_000);
        }
    }

    #[test]
    fn test_fitness_score_weights() {
        let mut r = FitnessRecord::new("t".into(), 0, None);
        for _ in 0..10 {
            r.record_invocation(true, Some(1.0), 100);
        }
        let s = r.fitness_score();
        assert!(s >= 0.0 && s <= 1.0, "score out of range: {s}");
        // Perfect acc(1)*0.4 + sr(1)*0.3 + usage(0.1)*0.2 + speed(~0.99)*0.1 ≈ 0.82
        assert!(s > 0.75, "expected > 0.75 got {s}");

        // Zero-data record: acc=0, sr=0, usage=0, speed=0.5 → 0.05
        let empty = FitnessRecord::new("e".into(), 0, None);
        let es = empty.fitness_score();
        assert!(es >= 0.0 && es <= 1.0);
        assert!(es < 0.2, "empty score unexpectedly high: {es}");
    }

    #[test]
    fn test_top_n_bottom_n_ordering() {
        let mut t = FitnessTracker::new();
        populate(&mut t);
        let top = t.top_n(3);
        let bot = t.bottom_n(3);
        assert_eq!(top.len(), 3);
        assert_eq!(bot.len(), 3);
        assert_eq!(top[0].0, "alpha");
        assert_eq!(bot[0].0, "gamma");
        // descending
        assert!(top[0].1 >= top[1].1 && top[1].1 >= top[2].1);
        // ascending
        assert!(bot[0].1 <= bot[1].1 && bot[1].1 <= bot[2].1);
    }

    #[test]
    fn test_generation_snapshot() {
        let mut t = FitnessTracker::new();
        populate(&mut t);
        let stats = t.snapshot_generation();
        assert_eq!(stats.population_size, 3);
        assert!(stats.avg_fitness >= stats.min_fitness);
        assert!(stats.max_fitness >= stats.avg_fitness);
        assert_eq!(stats.best_specialist_id, "alpha");
        assert_eq!(t.history().len(), 1);
        // Second snapshot bumps generation counter.
        let _ = t.snapshot_generation();
        assert_eq!(t.history().len(), 2);
        assert!(t.history()[1].generation >= t.history()[0].generation);
    }

    #[test]
    fn test_sqlite_roundtrip() {
        let mut t = FitnessTracker::new();
        populate(&mut t);
        let _ = t.snapshot_generation();

        let dir = std::env::temp_dir();
        let path = dir.join(format!(
            "qlang_fitness_{}.redb",
            now_unix() ^ std::process::id() as u64
        ));
        let p = path.to_string_lossy().to_string();
        // Ensure clean slate.
        let _ = std::fs::remove_file(&p);

        t.to_sqlite(&p).expect("persist");
        let loaded = FitnessTracker::from_sqlite(&p).expect("load");

        assert_eq!(loaded.len(), t.len());
        assert_eq!(loaded.history().len(), t.history().len());
        let before = t.get("alpha").unwrap();
        let after = loaded.get("alpha").unwrap();
        assert_eq!(before.invocations, after.invocations);
        assert_eq!(before.successes, after.successes);
        assert_eq!(before.accuracy_samples.len(), after.accuracy_samples.len());

        let _ = std::fs::remove_file(&p);
        println!("roundtrip OK: {}", loaded.status_line());
    }
}
