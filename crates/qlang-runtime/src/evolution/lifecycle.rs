//! Lifecycle Manager — birth, retirement, and death of specialists.
//!
//! Manages the population lifecycle of specialists in the evolving Organism:
//!   * `birth`   — a new specialist is created (from mutation / crossover).
//!   * `retire`  — weak specialists are archived (weights kept, not active).
//!   * `kill`    — fully delete specialists that have been retired for too long.
//!
//! Persists the full registry to a `redb` key-value store. Never loses data silently:
//! every state transition is recorded on the `SpecialistMetadata` and summarised in a
//! `SelectionReport` returned from `selection_step`.
//!
//! Coordination namespace: `qlang-evolution/lifecycle-manager-status`

use redb::{Database, ReadableTable, TableDefinition};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, VecDeque};
use std::time::{SystemTime, UNIX_EPOCH};

const REGISTRY_TABLE: TableDefinition<&str, Vec<u8>> = TableDefinition::new("lifecycle_registry");
const META_TABLE: TableDefinition<&str, Vec<u8>> = TableDefinition::new("lifecycle_meta");

fn now_unix() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// A small deterministic PRNG (xorshift64) used for ID generation.
fn xorshift64(state: &mut u64) -> u64 {
    let mut x = *state;
    if x == 0 {
        x = 0x9E3779B97F4A7C15;
    }
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    x
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum LifecycleStatus {
    /// Currently in use, participating in inference & training.
    Active,
    /// Archived: kept on disk but not routed to.
    Retired,
    /// Fully deleted — only the metadata tombstone remains for lineage lookups.
    Dead,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpecialistMetadata {
    pub id: String,
    pub status: LifecycleStatus,
    pub generation: u32,
    pub birth_time: u64,
    pub retire_time: Option<u64>,
    pub death_time: Option<u64>,
    pub parent_ids: Vec<String>,
    pub children_ids: Vec<String>,
    pub fitness_at_retire: Option<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LifecyclePolicy {
    /// Desired number of *active* specialists.
    pub target_population: usize,
    /// Fitness below this value → retire.
    pub retire_threshold_fitness: f32,
    /// Retired specialists older than this (seconds since retire_time) → killed.
    pub kill_threshold_age_seconds: u64,
    /// Protect newborns: don't retire until they have survived at least this many generations.
    pub min_generations_before_retire: u32,
    /// Cap on children per parent (advisory — bookkeeping is always exact).
    pub max_children_per_parent: u32,
}

impl Default for LifecyclePolicy {
    fn default() -> Self {
        Self {
            target_population: 50,
            retire_threshold_fitness: 0.3,
            kill_threshold_age_seconds: 7 * 24 * 3600,
            min_generations_before_retire: 2,
            max_children_per_parent: 5,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SelectionReport {
    pub generation: u32,
    pub retired: Vec<String>,
    pub killed: Vec<String>,
    pub active_after: usize,
    pub avg_fitness: f32,
}

pub struct LifecycleManager {
    registry: HashMap<String, SpecialistMetadata>,
    policy: LifecyclePolicy,
    current_generation: u32,
    rng_state: u64,
}

impl LifecycleManager {
    pub fn new(policy: LifecyclePolicy) -> Self {
        Self {
            registry: HashMap::new(),
            policy,
            current_generation: 0,
            rng_state: 0xDEADBEEFCAFEBABE,
        }
    }

    pub fn policy(&self) -> &LifecyclePolicy {
        &self.policy
    }

    /// Create a new specialist (birth). Returns its generated ID of the form
    /// `spec-gen{generation}-{rand:08x}`.
    pub fn birth(&mut self, parent_ids: Vec<String>) -> String {
        let rand = xorshift64(&mut self.rng_state) as u32;
        let id = format!("spec-gen{}-{:08x}", self.current_generation, rand);

        // Link to parents (mutate their children lists).
        for pid in &parent_ids {
            if let Some(parent) = self.registry.get_mut(pid) {
                parent.children_ids.push(id.clone());
            }
        }

        let meta = SpecialistMetadata {
            id: id.clone(),
            status: LifecycleStatus::Active,
            generation: self.current_generation,
            birth_time: now_unix(),
            retire_time: None,
            death_time: None,
            parent_ids,
            children_ids: Vec::new(),
            fitness_at_retire: None,
        };
        self.registry.insert(id.clone(), meta);
        id
    }

    /// Mark specialist as retired (weights archived, not active).
    /// Idempotent for already-retired specialists. Dead specialists cannot be retired.
    pub fn retire(&mut self, id: &str, fitness: f32) -> Result<(), String> {
        let meta = self
            .registry
            .get_mut(id)
            .ok_or_else(|| format!("unknown specialist id: {id}"))?;
        match meta.status {
            LifecycleStatus::Retired => Ok(()),
            LifecycleStatus::Dead => Err(format!("specialist {id} is already dead")),
            LifecycleStatus::Active => {
                meta.status = LifecycleStatus::Retired;
                meta.retire_time = Some(now_unix());
                meta.fitness_at_retire = Some(fitness);
                Ok(())
            }
        }
    }

    /// Fully delete a specialist — transitions to `Dead` and stamps `death_time`.
    /// The metadata row stays as a tombstone so lineage queries still work.
    pub fn kill(&mut self, id: &str) -> Result<(), String> {
        let meta = self
            .registry
            .get_mut(id)
            .ok_or_else(|| format!("unknown specialist id: {id}"))?;
        if meta.status == LifecycleStatus::Dead {
            return Ok(());
        }
        meta.status = LifecycleStatus::Dead;
        meta.death_time = Some(now_unix());
        Ok(())
    }

    /// Run the selection step: retire low-fitness actives and kill old retireds.
    /// `fitness_fn` returns the current fitness of an active specialist (or `None`).
    pub fn selection_step(
        &mut self,
        fitness_fn: impl Fn(&str) -> Option<f32>,
    ) -> SelectionReport {
        let now = now_unix();
        let mut retired = Vec::new();
        let mut killed = Vec::new();
        let mut fitness_sum = 0.0f32;
        let mut fitness_count = 0usize;

        // Snapshot IDs for deterministic iteration.
        let mut ids: Vec<String> = self.registry.keys().cloned().collect();
        ids.sort();

        for id in &ids {
            let (status, generation, retire_time) = {
                let meta = self.registry.get(id).expect("id present");
                (meta.status.clone(), meta.generation, meta.retire_time)
            };

            match status {
                LifecycleStatus::Active => {
                    let age_generations = self.current_generation.saturating_sub(generation);
                    if age_generations < self.policy.min_generations_before_retire {
                        continue;
                    }
                    if let Some(f) = fitness_fn(id) {
                        fitness_sum += f;
                        fitness_count += 1;
                        if f < self.policy.retire_threshold_fitness
                            && self.retire(id, f).is_ok()
                        {
                            retired.push(id.clone());
                        }
                    }
                }
                LifecycleStatus::Retired => {
                    if let Some(t) = retire_time {
                        let age = now.saturating_sub(t);
                        if age >= self.policy.kill_threshold_age_seconds
                            && self.kill(id).is_ok()
                        {
                            killed.push(id.clone());
                        }
                    }
                }
                LifecycleStatus::Dead => {}
            }
        }

        let active_after = self.active_count();
        let avg_fitness = if fitness_count > 0 {
            fitness_sum / fitness_count as f32
        } else {
            0.0
        };

        SelectionReport {
            generation: self.current_generation,
            retired,
            killed,
            active_after,
            avg_fitness,
        }
    }

    /// Advance to the next generation. Returns the new generation index.
    pub fn next_generation(&mut self) -> u32 {
        self.current_generation = self.current_generation.saturating_add(1);
        self.current_generation
    }

    pub fn current_generation(&self) -> u32 {
        self.current_generation
    }

    pub fn active_count(&self) -> usize {
        self.registry
            .values()
            .filter(|m| m.status == LifecycleStatus::Active)
            .count()
    }

    pub fn get(&self, id: &str) -> Option<&SpecialistMetadata> {
        self.registry.get(id)
    }

    pub fn active_ids(&self) -> Vec<String> {
        let mut ids: Vec<String> = self
            .registry
            .iter()
            .filter(|(_, m)| m.status == LifecycleStatus::Active)
            .map(|(k, _)| k.clone())
            .collect();
        ids.sort();
        ids
    }

    /// BFS over the parent/child graph to collect the full lineage of `id`
    /// (ancestors + descendants + self). Returns a sorted deduplicated list.
    pub fn lineage(&self, id: &str) -> Vec<String> {
        if !self.registry.contains_key(id) {
            return Vec::new();
        }
        let mut visited: HashSet<String> = HashSet::new();
        let mut frontier: VecDeque<String> = VecDeque::new();
        frontier.push_back(id.to_string());
        while let Some(cur) = frontier.pop_front() {
            if !visited.insert(cur.clone()) {
                continue;
            }
            if let Some(meta) = self.registry.get(&cur) {
                for p in &meta.parent_ids {
                    if !visited.contains(p) {
                        frontier.push_back(p.clone());
                    }
                }
                for c in &meta.children_ids {
                    if !visited.contains(c) {
                        frontier.push_back(c.clone());
                    }
                }
            }
        }
        let mut out: Vec<String> = visited.into_iter().collect();
        out.sort();
        out
    }

    // ---------- persistence ----------

    pub fn save(&self, path: &str) -> Result<(), String> {
        let db = Database::create(path).map_err(|e| format!("redb create: {e}"))?;
        let write_txn = db.begin_write().map_err(|e| format!("begin_write: {e}"))?;
        {
            let mut reg_tab = write_txn
                .open_table(REGISTRY_TABLE)
                .map_err(|e| format!("open registry: {e}"))?;
            for (id, meta) in &self.registry {
                let bytes = bincode::serialize(meta)
                    .map_err(|e| format!("serialize {id}: {e}"))?;
                reg_tab
                    .insert(id.as_str(), bytes)
                    .map_err(|e| format!("insert {id}: {e}"))?;
            }
            let mut meta_tab = write_txn
                .open_table(META_TABLE)
                .map_err(|e| format!("open meta: {e}"))?;
            let policy_bytes = bincode::serialize(&self.policy)
                .map_err(|e| format!("serialize policy: {e}"))?;
            meta_tab
                .insert("policy", policy_bytes)
                .map_err(|e| format!("insert policy: {e}"))?;
            let gen_bytes = bincode::serialize(&self.current_generation)
                .map_err(|e| format!("serialize generation: {e}"))?;
            meta_tab
                .insert("generation", gen_bytes)
                .map_err(|e| format!("insert generation: {e}"))?;
            let rng_bytes = bincode::serialize(&self.rng_state)
                .map_err(|e| format!("serialize rng: {e}"))?;
            meta_tab
                .insert("rng", rng_bytes)
                .map_err(|e| format!("insert rng: {e}"))?;
        }
        write_txn.commit().map_err(|e| format!("commit: {e}"))?;
        Ok(())
    }

    pub fn load(path: &str) -> Result<Self, String> {
        let db = Database::open(path).map_err(|e| format!("redb open: {e}"))?;
        let read_txn = db.begin_read().map_err(|e| format!("begin_read: {e}"))?;
        let mut registry = HashMap::new();
        if let Ok(reg_tab) = read_txn.open_table(REGISTRY_TABLE) {
            for entry in reg_tab.iter().map_err(|e| format!("iter: {e}"))? {
                let (k, v) = entry.map_err(|e| format!("entry: {e}"))?;
                let id = k.value().to_string();
                let meta: SpecialistMetadata = bincode::deserialize(&v.value())
                    .map_err(|e| format!("deserialize {id}: {e}"))?;
                registry.insert(id, meta);
            }
        }
        let meta_tab = read_txn
            .open_table(META_TABLE)
            .map_err(|e| format!("open meta: {e}"))?;
        let policy_bytes = meta_tab
            .get("policy")
            .map_err(|e| format!("get policy: {e}"))?
            .ok_or_else(|| "policy missing".to_string())?;
        let policy: LifecyclePolicy = bincode::deserialize(&policy_bytes.value())
            .map_err(|e| format!("decode policy: {e}"))?;
        let gen_bytes = meta_tab
            .get("generation")
            .map_err(|e| format!("get generation: {e}"))?
            .ok_or_else(|| "generation missing".to_string())?;
        let current_generation: u32 = bincode::deserialize(&gen_bytes.value())
            .map_err(|e| format!("decode generation: {e}"))?;
        let rng_state: u64 = meta_tab
            .get("rng")
            .map_err(|e| format!("get rng: {e}"))?
            .and_then(|v| bincode::deserialize(&v.value()).ok())
            .unwrap_or(0xDEADBEEFCAFEBABE);
        Ok(Self {
            registry,
            policy,
            current_generation,
            rng_state,
        })
    }
}

// =====================================================================
// Tests
// =====================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn mgr() -> LifecycleManager {
        LifecycleManager::new(LifecyclePolicy::default())
    }

    #[test]
    fn test_birth_creates_active_specialist() {
        let mut m = mgr();
        let id = m.birth(vec![]);
        let meta = m.get(&id).expect("present");
        assert_eq!(meta.status, LifecycleStatus::Active);
        assert_eq!(meta.generation, 0);
        assert_eq!(m.active_count(), 1);
        assert!(id.starts_with("spec-gen0-"), "bad id format: {id}");
    }

    #[test]
    fn test_retire_changes_status() {
        let mut m = mgr();
        let id = m.birth(vec![]);
        m.retire(&id, 0.12).expect("retire ok");
        let meta = m.get(&id).unwrap();
        assert_eq!(meta.status, LifecycleStatus::Retired);
        assert_eq!(meta.fitness_at_retire, Some(0.12));
        assert!(meta.retire_time.is_some());
        assert_eq!(m.active_count(), 0);
        // Retiring again is idempotent.
        m.retire(&id, 0.1).expect("idempotent retire");
    }

    #[test]
    fn test_kill_marks_dead() {
        let mut m = mgr();
        let id = m.birth(vec![]);
        m.kill(&id).expect("kill ok");
        let meta = m.get(&id).unwrap();
        assert_eq!(meta.status, LifecycleStatus::Dead);
        assert!(meta.death_time.is_some());
        // Retiring a dead specialist must fail.
        assert!(m.retire(&id, 0.9).is_err());
        // But killing twice is idempotent.
        assert!(m.kill(&id).is_ok());
    }

    #[test]
    fn test_selection_retires_low_fitness() {
        let mut m = LifecycleManager::new(LifecyclePolicy {
            min_generations_before_retire: 0,
            ..Default::default()
        });
        let a = m.birth(vec![]);
        let b = m.birth(vec![]);
        let c = m.birth(vec![]);
        m.next_generation();
        let scores: HashMap<String, f32> = HashMap::from([
            (a.clone(), 0.9),
            (b.clone(), 0.1),
            (c.clone(), 0.05),
        ]);
        let report = m.selection_step(|id| scores.get(id).copied());
        assert_eq!(report.retired.len(), 2);
        assert!(report.retired.contains(&b));
        assert!(report.retired.contains(&c));
        assert_eq!(m.active_count(), 1);
        assert!((report.avg_fitness - (0.9 + 0.1 + 0.05) / 3.0).abs() < 1e-4);
    }

    #[test]
    fn test_kill_old_retired_specialists() {
        let mut m = LifecycleManager::new(LifecyclePolicy {
            kill_threshold_age_seconds: 0,
            min_generations_before_retire: 0,
            ..Default::default()
        });
        let id = m.birth(vec![]);
        m.retire(&id, 0.01).unwrap();
        let report = m.selection_step(|_| None);
        assert_eq!(report.killed, vec![id.clone()]);
        assert_eq!(m.get(&id).unwrap().status, LifecycleStatus::Dead);
    }

    #[test]
    fn test_lineage_tracking() {
        let mut m = mgr();
        let grand = m.birth(vec![]);
        let parent = m.birth(vec![grand.clone()]);
        let child = m.birth(vec![parent.clone()]);
        let sibling = m.birth(vec![parent.clone()]);

        assert!(m.get(&grand).unwrap().children_ids.contains(&parent));
        assert!(m.get(&parent).unwrap().children_ids.contains(&child));
        assert!(m.get(&parent).unwrap().children_ids.contains(&sibling));

        let lin = m.lineage(&parent);
        for must in [&grand, &parent, &child, &sibling] {
            assert!(lin.contains(must), "lineage missing {must}");
        }
        // Unknown ID → empty lineage, no panic.
        assert!(m.lineage("does-not-exist").is_empty());
    }

    #[test]
    fn test_persist_roundtrip() {
        let tmp = std::env::temp_dir().join(format!(
            "lifecycle-roundtrip-{}-{}.redb",
            now_unix(),
            std::process::id()
        ));
        let path = tmp.to_str().unwrap().to_string();
        let _ = std::fs::remove_file(&path);

        let mut m = mgr();
        let a = m.birth(vec![]);
        let b = m.birth(vec![a.clone()]);
        m.retire(&a, 0.25).unwrap();
        m.next_generation();
        m.save(&path).expect("save");

        let loaded = LifecycleManager::load(&path).expect("load");
        assert_eq!(loaded.current_generation(), 1);
        assert_eq!(loaded.active_count(), 1);
        let loaded_a = loaded.get(&a).unwrap();
        assert_eq!(loaded_a.status, LifecycleStatus::Retired);
        assert_eq!(loaded_a.fitness_at_retire, Some(0.25));
        let loaded_b = loaded.get(&b).unwrap();
        assert_eq!(loaded_b.parent_ids, vec![a.clone()]);

        let _ = std::fs::remove_file(&path);
    }
}
