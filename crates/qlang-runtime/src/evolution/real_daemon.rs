//! Real Evolution Daemon — runs evolution cycles autonomously in the background.
//!
//! Sibling to the legacy `daemon` module (which powers a simulation-only dashboard
//! with a different public API). This module implements the spec of a true
//! evolutionary loop that mutates ternary specialists, evaluates their fitness on
//! real MNIST samples, and manages their population through the `LifecycleManager`.
//!
//! The loop runs on its own `std::thread`, never blocking the HTTP server or UI.
//! `stop()` sets an atomic flag; the thread wakes every second during its
//! generation-interval sleep so shutdown latency stays well under five seconds.

use super::fitness::FitnessTracker;
use super::lifecycle::{LifecycleManager, LifecyclePolicy, LifecycleStatus, SelectionReport};
use super::mutation::{crossover_ternary_i8, mutate_ternary_i8, MutationConfig};
use crate::mnist::MnistData;
use crate::ternary_brain::TernaryBrain;

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, RwLock};
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

fn now_unix() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// Configuration for the autonomous evolution loop.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvolutionConfig {
    /// Seconds between generation ticks in the background loop.
    pub generation_interval_secs: u64,
    /// Number of test samples used per fitness evaluation.
    pub eval_sample_size: usize,
    /// Offspring spawned per generation (split across top parents).
    pub mutations_per_generation: usize,
    /// Dataset directory (for the background-loop auto-load path).
    pub data_path: String,
    /// Where to persist fitness + lifecycle state each generation.
    pub persist_path: String,
    /// Mutation config (seed, rates, mode).
    pub mutation_config: MutationConfig,
    /// Lifecycle rules for retirement / kill.
    pub lifecycle_policy: LifecyclePolicy,
}

impl Default for EvolutionConfig {
    fn default() -> Self {
        Self {
            generation_interval_secs: 300,
            eval_sample_size: 500,
            mutations_per_generation: 10,
            data_path: "data/mnist".into(),
            persist_path: "data/evolution_state".into(),
            mutation_config: MutationConfig::default(),
            lifecycle_policy: LifecyclePolicy::default(),
        }
    }
}

/// Per-generation summary (cheap to clone; appended to the in-memory log).
#[derive(Debug, Clone, Serialize)]
pub struct GenerationReport {
    pub generation: u32,
    pub timestamp: u64,
    pub population_size: usize,
    pub mutations_spawned: usize,
    pub retired: usize,
    pub killed: usize,
    pub best_fitness: f32,
    pub avg_fitness: f32,
    pub elapsed_ms: u64,
}

/// Per-specialist snapshot for HTTP consumers (shape mirrors the stub
/// daemon's `SpecialistInfo` so existing UIs keep working).
#[derive(Debug, Clone, Serialize)]
pub struct SpecialistSnapshot {
    pub id: String,
    pub generation_born: u32,
    pub parent_id: Option<String>,
    pub children: Vec<String>,
    pub fitness: f32,
    pub status: String,
    pub mutations: u32,
}

/// Cheap status snapshot for HTTP / CLI.
#[derive(Debug, Clone, Serialize)]
pub struct DaemonStatus {
    pub running: bool,
    pub current_generation: u32,
    pub population_size: usize,
    pub total_generations: u32,
    pub best_fitness: f32,
    pub uptime_secs: u64,
}

/// The real evolution daemon. Multi-thread safe; all state is behind `RwLock`s
/// so the HTTP layer can poll status concurrently with the evolution loop.
pub struct RealEvolutionDaemon {
    fitness: Arc<RwLock<FitnessTracker>>,
    lifecycle: Arc<RwLock<LifecycleManager>>,
    /// Specialist id → flat ternary i8 weight vector.
    specialists: Arc<RwLock<HashMap<String, Vec<i8>>>>,
    /// Template brain used to rebuild any specialist for fitness evaluation.
    template: Arc<RwLock<Option<TernaryBrain>>>,
    config: EvolutionConfig,
    running: Arc<AtomicBool>,
    generation_log: Arc<RwLock<Vec<GenerationReport>>>,
    started_at: Arc<AtomicU64>,
    total_generations: Arc<AtomicU32>,
    best_fitness: Arc<RwLock<f32>>,
}

impl RealEvolutionDaemon {
    pub fn new(config: EvolutionConfig) -> Self {
        let lifecycle = LifecycleManager::new(config.lifecycle_policy.clone());
        Self {
            fitness: Arc::new(RwLock::new(FitnessTracker::new())),
            lifecycle: Arc::new(RwLock::new(lifecycle)),
            specialists: Arc::new(RwLock::new(HashMap::new())),
            template: Arc::new(RwLock::new(None)),
            config,
            running: Arc::new(AtomicBool::new(false)),
            generation_log: Arc::new(RwLock::new(Vec::new())),
            started_at: Arc::new(AtomicU64::new(0)),
            total_generations: Arc::new(AtomicU32::new(0)),
            best_fitness: Arc::new(RwLock::new(0.0)),
        }
    }

    pub fn config(&self) -> &EvolutionConfig {
        &self.config
    }

    /// Seed the population with `n` specialists derived from real MNIST statistics.
    pub fn seed_population(&self, n: usize, mnist: &MnistData) -> Result<(), String> {
        if n == 0 {
            return Err("seed_population: n must be > 0".into());
        }
        let neurons_per_class = 3;
        let init_samples = mnist.n_train.min(2000);
        let reference = TernaryBrain::init(
            &mnist.train_images,
            &mnist.train_labels,
            mnist.image_size,
            init_samples,
            mnist.n_classes,
            neurons_per_class,
        );
        let reference_weights = reference.dump_weights_i8();
        {
            let mut t = self.template.write().map_err(|e| format!("template lock: {e}"))?;
            *t = Some(reference);
        }

        let mut rng_seed: u64 = 0xA11C_ECAF_EBAB_E5ED;
        for i in 0..n {
            let id = {
                let mut lc = self.lifecycle.write().map_err(|e| format!("lc lock: {e}"))?;
                lc.birth(Vec::new())
            };

            let mut weights = reference_weights.clone();
            if i > 0 {
                // Diversify — first member keeps reference weights so the template
                // baseline is always in the gene pool.
                rng_seed = rng_seed
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let cfg = MutationConfig {
                    seed: rng_seed,
                    ..self.config.mutation_config.clone()
                };
                mutate_ternary_i8(&mut weights, &cfg);
            }
            self.specialists
                .write()
                .map_err(|e| format!("specs lock: {e}"))?
                .insert(id.clone(), weights);
            self.fitness
                .write()
                .map_err(|e| format!("fit lock: {e}"))?
                .register(id, 0, None);
        }
        Ok(())
    }

    /// Execute one generation synchronously. Safe to call on demand (UI button,
    /// test harness) without the background loop running.
    pub fn step_once(
        &self,
        test_imgs: &[f32],
        test_lbls: &[u8],
        n_test: usize,
    ) -> Result<GenerationReport, String> {
        let t0 = Instant::now();

        // 1. FITNESS EVALUATION -----------------------------------------------
        let template_guard = self.template.read().map_err(|e| format!("tpl read: {e}"))?;
        let template = template_guard
            .as_ref()
            .ok_or_else(|| "evolution daemon: no template brain — call seed_population first".to_string())?;

        let active_ids: Vec<String> = self
            .lifecycle
            .read()
            .map_err(|e| format!("lc read: {e}"))?
            .active_ids();

        let specs_snapshot: Vec<(String, Vec<i8>)> = {
            let specs = self.specialists.read().map_err(|e| format!("specs read: {e}"))?;
            active_ids
                .iter()
                .filter_map(|id| specs.get(id).map(|w| (id.clone(), w.clone())))
                .collect()
        };

        let eval_n = n_test.min(self.config.eval_sample_size).max(1);
        let required = eval_n * template.image_dim;
        if test_imgs.len() < required || test_lbls.len() < eval_n {
            return Err(format!(
                "step_once: not enough test data (need {} floats / {} labels, got {} / {})",
                required, eval_n, test_imgs.len(), test_lbls.len()
            ));
        }
        let eval_imgs = &test_imgs[..required];
        let eval_lbls = &test_lbls[..eval_n];

        let mut id_to_fitness: HashMap<String, f32> = HashMap::new();
        for (id, weights) in &specs_snapshot {
            let brain = match TernaryBrain::from_template_and_weights(template, weights) {
                Ok(b) => b,
                Err(_) => continue,
            };
            let inf_t0 = Instant::now();
            let acc = brain.accuracy(eval_imgs, eval_lbls, eval_n);
            let elapsed_us = inf_t0.elapsed().as_micros() as u64;
            {
                let mut ft = self.fitness.write().map_err(|e| format!("fit write: {e}"))?;
                ft.record(id, acc >= 0.5, Some(acc), elapsed_us);
            }
            id_to_fitness.insert(id.clone(), acc);
        }

        // 2. SELECT TOP PARENTS -----------------------------------------------
        // Take top 20% by fitness (at least 2 if possible) to form the crossover
        // pool. `top_n(5)` keeps the classical "top 5" parents for the primary
        // mutation path so existing behavior is preserved.
        let top = {
            let ft = self.fitness.read().map_err(|e| format!("fit read: {e}"))?;
            ft.top_n(5)
        };
        let top_parent_ids: Vec<String> = top.iter().map(|(id, _)| id.clone()).collect();

        let crossover_pool: Vec<String> = {
            let active_n = specs_snapshot.len();
            let pool_size = ((active_n as f32 * 0.2).ceil() as usize).max(2);
            let ft = self.fitness.read().map_err(|e| format!("fit read: {e}"))?;
            ft.top_n(pool_size)
                .into_iter()
                .map(|(id, _)| id)
                .collect()
        };

        // 3. MUTATION / CROSSOVER / OFFSPRING ---------------------------------
        let mutations_spawned = if !top_parent_ids.is_empty() {
            let per_parent =
                (self.config.mutations_per_generation / top_parent_ids.len().max(1)).max(1);
            let mut spawned = 0usize;
            let base_seed = now_unix().wrapping_mul(0x9E37_79B9_7F4A_7C15);

            for (pi, parent_id) in top_parent_ids.iter().enumerate() {
                let parent_weights = {
                    let specs = self
                        .specialists
                        .read()
                        .map_err(|e| format!("specs read: {e}"))?;
                    specs.get(parent_id).cloned()
                };
                let parent_weights = match parent_weights {
                    Some(w) => w,
                    None => continue,
                };
                for k in 0..per_parent {
                    if spawned >= self.config.mutations_per_generation {
                        break;
                    }
                    let seed = base_seed
                        .wrapping_add((pi as u64) * 1_000_003)
                        .wrapping_add(k as u64)
                        ^ self.config.mutation_config.seed;

                    // Decide reproduction strategy: 70% single-parent mutation,
                    // 30% two-parent crossover. The draw is derived from `seed`
                    // so a run is fully reproducible.
                    let draw = ((seed.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 33) as u32)
                        as f32
                        / (u32::MAX as f32);
                    let use_crossover = draw < 0.30 && crossover_pool.len() >= 2;

                    let (child, parent_ids) = if use_crossover {
                        // Pick a second distinct parent from the crossover pool.
                        let idx = ((seed >> 11) as usize) % crossover_pool.len();
                        let mut partner_id = crossover_pool[idx].clone();
                        if partner_id == *parent_id {
                            let alt = (idx + 1) % crossover_pool.len();
                            partner_id = crossover_pool[alt].clone();
                        }
                        let partner_weights = {
                            let specs = self
                                .specialists
                                .read()
                                .map_err(|e| format!("specs read: {e}"))?;
                            specs.get(&partner_id).cloned()
                        };
                        match partner_weights {
                            Some(pw) if partner_id != *parent_id => {
                                let cfg = MutationConfig {
                                    seed,
                                    ..self.config.mutation_config.clone()
                                };
                                let c = crossover_ternary_i8(&parent_weights, &pw, &cfg);
                                (c, vec![parent_id.clone(), partner_id])
                            }
                            _ => {
                                // Fallback to mutation when no distinct partner.
                                let cfg = MutationConfig {
                                    seed,
                                    ..self.config.mutation_config.clone()
                                };
                                let mut c = parent_weights.clone();
                                mutate_ternary_i8(&mut c, &cfg);
                                (c, vec![parent_id.clone()])
                            }
                        }
                    } else {
                        let cfg = MutationConfig {
                            seed,
                            ..self.config.mutation_config.clone()
                        };
                        let mut c = parent_weights.clone();
                        mutate_ternary_i8(&mut c, &cfg);
                        (c, vec![parent_id.clone()])
                    };

                    let child_id = {
                        let mut lc =
                            self.lifecycle.write().map_err(|e| format!("lc write: {e}"))?;
                        lc.birth(parent_ids.clone())
                    };
                    self.specialists
                        .write()
                        .map_err(|e| format!("specs write: {e}"))?
                        .insert(child_id.clone(), child);
                    {
                        let gen_now = self
                            .lifecycle
                            .read()
                            .map_err(|e| format!("lc read: {e}"))?
                            .current_generation();
                        let mut ft =
                            self.fitness.write().map_err(|e| format!("fit write: {e}"))?;
                        ft.register(child_id, gen_now, parent_ids.first().cloned());
                    }
                    spawned += 1;
                }
            }
            spawned
        } else {
            0
        };

        // 4. LIFECYCLE SELECTION ----------------------------------------------
        let report: SelectionReport = {
            let snapshot = id_to_fitness.clone();
            let mut lc = self.lifecycle.write().map_err(|e| format!("lc write: {e}"))?;
            let rep = lc.selection_step(|id| snapshot.get(id).copied());
            lc.next_generation();
            rep
        };

        // Drop weights for killed specialists so the map can't grow unbounded.
        if !report.killed.is_empty() {
            let mut specs = self
                .specialists
                .write()
                .map_err(|e| format!("specs write: {e}"))?;
            for id in &report.killed {
                specs.remove(id);
            }
        }

        // 5. SNAPSHOT + PERSIST -----------------------------------------------
        let stats = {
            let mut ft = self.fitness.write().map_err(|e| format!("fit write: {e}"))?;
            ft.snapshot_generation()
        };

        let pop_size = self
            .specialists
            .read()
            .map_err(|e| format!("specs read: {e}"))?
            .len();

        {
            let mut bf = self.best_fitness.write().map_err(|e| format!("bf lock: {e}"))?;
            if stats.max_fitness > *bf {
                *bf = stats.max_fitness;
            }
        }

        let gen_report = GenerationReport {
            generation: report.generation,
            timestamp: now_unix(),
            population_size: pop_size,
            mutations_spawned,
            retired: report.retired.len(),
            killed: report.killed.len(),
            best_fitness: stats.max_fitness,
            avg_fitness: stats.avg_fitness,
            elapsed_ms: t0.elapsed().as_millis() as u64,
        };

        {
            let mut log = self
                .generation_log
                .write()
                .map_err(|e| format!("log lock: {e}"))?;
            log.push(gen_report.clone());
            const MAX_LOG: usize = 10_000;
            if log.len() > MAX_LOG {
                let drop = log.len() - MAX_LOG;
                log.drain(..drop);
            }
        }

        self.total_generations.fetch_add(1, Ordering::Relaxed);
        let _ = self.persist();
        Ok(gen_report)
    }

    fn persist(&self) -> Result<(), String> {
        let base = &self.config.persist_path;
        if base.is_empty() {
            return Ok(());
        }
        let _ = std::fs::create_dir_all(base);
        if let Ok(ft) = self.fitness.read() {
            let _ = ft.to_sqlite(&format!("{}/fitness.redb", base));
        }
        if let Ok(lc) = self.lifecycle.read() {
            let _ = lc.save(&format!("{}/lifecycle.redb", base));
        }
        Ok(())
    }

    /// Start the background loop. Returns the `JoinHandle` so callers may join on
    /// graceful shutdown. Only one loop may be active per daemon instance.
    pub fn start(self: Arc<Self>) -> Result<thread::JoinHandle<()>, String> {
        if self.running.swap(true, Ordering::SeqCst) {
            return Err("evolution daemon already running".into());
        }
        self.started_at.store(now_unix(), Ordering::Relaxed);
        let me = Arc::clone(&self);

        let handle = thread::Builder::new()
            .name("evolution-daemon".into())
            .spawn(move || {
                while me.running.load(Ordering::Relaxed) {
                    // Re-load test data each cycle so the daemon uses fresh samples
                    // without pinning a large buffer in memory.
                    match MnistData::load_from_dir(&me.config.data_path) {
                        Ok(data) => {
                            let n = data.n_test.min(me.config.eval_sample_size);
                            if let Err(e) =
                                me.step_once(&data.test_images, &data.test_labels, n)
                            {
                                eprintln!("[evolution-daemon] step error: {e}");
                            }
                        }
                        Err(e) => {
                            eprintln!("[evolution-daemon] mnist load failed ({e}); sleeping");
                        }
                    }

                    // Responsive sleep: wake every 1s to check the stop flag so
                    // shutdown latency stays bounded.
                    let total = me.config.generation_interval_secs;
                    let mut slept = 0u64;
                    while slept < total && me.running.load(Ordering::Relaxed) {
                        thread::sleep(Duration::from_secs(1));
                        slept += 1;
                    }
                }
            })
            .map_err(|e| format!("thread spawn: {e}"))?;
        Ok(handle)
    }

    pub fn stop(&self) {
        self.running.store(false, Ordering::Relaxed);
    }

    pub fn is_running(&self) -> bool {
        self.running.load(Ordering::Relaxed)
    }

    pub fn get_reports(&self) -> Vec<GenerationReport> {
        self.generation_log
            .read()
            .map(|v| v.clone())
            .unwrap_or_default()
    }

    pub fn population_size(&self) -> usize {
        self.specialists.read().map(|s| s.len()).unwrap_or(0)
    }

    /// Return a clone of the raw ternary weight vector for `id`, if present.
    /// Used by CLI tools and checkpoint exports (T061).
    pub fn get_weights(&self, id: &str) -> Option<Vec<i8>> {
        self.specialists.read().ok()?.get(id).cloned()
    }

    /// Return the template brain's image dimension (for export headers).
    pub fn template_image_dim(&self) -> Option<usize> {
        self.template.read().ok()?.as_ref().map(|t| t.image_dim)
    }

    /// Return the template brain's class count (for export headers).
    pub fn template_n_classes(&self) -> Option<usize> {
        self.template.read().ok()?.as_ref().map(|t| t.n_classes)
    }

    /// Build a snapshot of every specialist (active/retired/dead) with its
    /// latest fitness, for HTTP consumption.
    pub fn snapshot_specialists(&self) -> Vec<SpecialistSnapshot> {
        let lc = match self.lifecycle.read() {
            Ok(g) => g,
            Err(_) => return Vec::new(),
        };
        let ft = match self.fitness.read() {
            Ok(g) => g,
            Err(_) => return Vec::new(),
        };
        // Grab all ids by union of active + any non-active we can see via registry.
        // LifecycleManager doesn't expose iter(), so we approximate by active ids +
        // any id the fitness tracker knows about.
        let mut out: Vec<SpecialistSnapshot> = Vec::new();
        let active: std::collections::HashSet<String> =
            lc.active_ids().into_iter().collect();

        // Walk fitness tracker records (the daemon registers every id it creates
        // with the fitness tracker, so this covers active/retired/dead uniformly).
        // `FitnessTracker` doesn't expose a direct iter either, so we rely on
        // lifecycle::get() for per-id lookups — for any id we can find via
        // fitness tracker.
        // Build id list from fitness history snapshots would be lossy; simpler:
        // traverse lifecycle via lineage on active roots + fitness records.
        //
        // Since neither struct provides `iter_records()`, we synthesise the list
        // from active_ids + their full lineage (covers parents + children).
        let mut id_set: std::collections::HashSet<String> = active.clone();
        for id in &active {
            for lineage_id in lc.lineage(id) {
                id_set.insert(lineage_id);
            }
        }
        for id in id_set {
            let meta = match lc.get(&id) {
                Some(m) => m,
                None => continue,
            };
            let (fitness, mutations) = match ft.get(&id) {
                Some(rec) => (rec.fitness_score(), rec.mutation_count),
                None => (0.0, 0),
            };
            let status = match meta.status {
                LifecycleStatus::Active => "active",
                LifecycleStatus::Retired => "retired",
                LifecycleStatus::Dead => "dead",
            };
            out.push(SpecialistSnapshot {
                id: meta.id.clone(),
                generation_born: meta.generation,
                parent_id: meta.parent_ids.first().cloned(),
                children: meta.children_ids.clone(),
                fitness,
                status: status.into(),
                mutations,
            });
        }
        out.sort_by(|a, b| {
            b.fitness
                .partial_cmp(&a.fitness)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        out
    }

    /// Return the full lineage (ancestors + descendants + self) for `id`.
    pub fn lineage(&self, id: &str) -> Vec<SpecialistSnapshot> {
        let lc = match self.lifecycle.read() {
            Ok(g) => g,
            Err(_) => return Vec::new(),
        };
        let ft = match self.fitness.read() {
            Ok(g) => g,
            Err(_) => return Vec::new(),
        };
        lc.lineage(id)
            .into_iter()
            .filter_map(|sid| {
                let meta = lc.get(&sid)?;
                let (fitness, mutations) = match ft.get(&sid) {
                    Some(rec) => (rec.fitness_score(), rec.mutation_count),
                    None => (0.0, 0),
                };
                let status = match meta.status {
                    LifecycleStatus::Active => "active",
                    LifecycleStatus::Retired => "retired",
                    LifecycleStatus::Dead => "dead",
                };
                Some(SpecialistSnapshot {
                    id: meta.id.clone(),
                    generation_born: meta.generation,
                    parent_id: meta.parent_ids.first().cloned(),
                    children: meta.children_ids.clone(),
                    fitness,
                    status: status.into(),
                    mutations,
                })
            })
            .collect()
    }

    pub fn status(&self) -> DaemonStatus {
        let uptime = {
            let started = self.started_at.load(Ordering::Relaxed);
            if started == 0 {
                0
            } else {
                now_unix().saturating_sub(started)
            }
        };
        DaemonStatus {
            running: self.is_running(),
            current_generation: self
                .lifecycle
                .read()
                .map(|lc| lc.current_generation())
                .unwrap_or(0),
            population_size: self.population_size(),
            total_generations: self.total_generations.load(Ordering::Relaxed),
            best_fitness: self.best_fitness.read().map(|v| *v).unwrap_or(0.0),
            uptime_secs: uptime,
        }
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::evolution::mutation::MutationMode;

    fn build_daemon(mutations: usize) -> (Arc<RealEvolutionDaemon>, MnistData) {
        let data = MnistData::synthetic(400, 120);
        let config = EvolutionConfig {
            generation_interval_secs: 1,
            eval_sample_size: 60,
            mutations_per_generation: mutations,
            data_path: String::new(),
            persist_path: String::new(),
            mutation_config: MutationConfig {
                flip_rate: 0.05,
                zero_rate: 0.01,
                expand_rate: 0.9,
                crossover_rate: 0.5,
                mode: MutationMode::PointMutation,
                seed: 7,
            },
            lifecycle_policy: LifecyclePolicy {
                target_population: 20,
                retire_threshold_fitness: 0.99, // aggressive — forces retirement
                kill_threshold_age_seconds: 0,  // kill retired immediately
                min_generations_before_retire: 0,
                max_children_per_parent: 10,
            },
        };
        let daemon = Arc::new(RealEvolutionDaemon::new(config));
        daemon.seed_population(5, &data).expect("seed");
        (daemon, data)
    }

    #[test]
    fn test_daemon_step_runs_generation() {
        let (daemon, data) = build_daemon(4);
        let report = daemon
            .step_once(&data.test_images, &data.test_labels, data.n_test)
            .expect("step_once");
        println!("report: {:?}", report);
        assert!(report.population_size > 0);
        assert!(report.mutations_spawned > 0, "no mutations spawned");
        assert!((0.0..=1.0).contains(&report.best_fitness));
        assert!((0.0..=1.0).contains(&report.avg_fitness));
        assert_eq!(daemon.get_reports().len(), 1);
    }

    #[test]
    fn test_stop_signal_works() {
        let (daemon, _data) = build_daemon(2);
        // data_path is empty → the loop's load_from_dir fails and it goes to the
        // responsive sleep, where the stop flag is honored within ~1s.
        let d2 = Arc::clone(&daemon);
        let handle = d2.start().expect("start");
        thread::sleep(Duration::from_millis(200));
        let start = Instant::now();
        daemon.stop();
        handle.join().expect("join");
        let elapsed = start.elapsed();
        assert!(
            elapsed < Duration::from_secs(5),
            "stop took too long: {:?}",
            elapsed
        );
        assert!(!daemon.is_running());
    }

    #[test]
    fn test_crossover_creates_child_with_two_parents() {
        // Spawn plenty of offspring so the 30% crossover path is hit.
        let (daemon, data) = build_daemon(40);
        let mut found_two_parent_child = false;
        for _ in 0..5 {
            daemon
                .step_once(&data.test_images, &data.test_labels, data.n_test)
                .expect("step_once");
            let lc = daemon.lifecycle.read().expect("lc read");
            for id in lc.active_ids() {
                if let Some(meta) = lc.get(&id) {
                    if meta.parent_ids.len() == 2 {
                        assert_ne!(
                            meta.parent_ids[0], meta.parent_ids[1],
                            "crossover parents must be distinct"
                        );
                        found_two_parent_child = true;
                    }
                }
            }
            drop(lc);
            if found_two_parent_child {
                break;
            }
        }
        assert!(
            found_two_parent_child,
            "no crossover child (parent_ids.len() == 2) produced over 5 generations"
        );
    }

    #[test]
    fn test_population_bounded() {
        // With kill_threshold_age_seconds=0 and retire_threshold_fitness=0.99 the
        // selection step retires + kills many specialists each generation, so the
        // weight map must stay bounded across many generations.
        let (daemon, data) = build_daemon(8);
        for _ in 0..10 {
            daemon
                .step_once(&data.test_images, &data.test_labels, data.n_test)
                .expect("step_once");
        }
        let pop = daemon.population_size();
        println!("final population = {pop}");
        assert!(pop < 200, "population grew unbounded: {pop}");
    }
}
