//! T061 — Standalone evolution demo on MNIST.
//!
//! Runs 30 generations of the self-evolving organism: seeds 20 TernaryBrain
//! specialists, then uses the `RealEvolutionDaemon` primitives (fitness +
//! mutation + crossover + lifecycle) step-by-step. Prints a per-generation
//! table and exports the best specialist's ternary weights to a QLMB-style
//! binary file (`data/best_evolved.bin`).
//!
//! Run with:
//!     cargo run --release --example mnist_evolution --no-default-features

use std::fs;
use std::io::Write;
use std::sync::Arc;

use qlang_runtime::evolution::lifecycle::LifecyclePolicy;
use qlang_runtime::evolution::mutation::{MutationConfig, MutationMode};
use qlang_runtime::evolution::real_daemon::{EvolutionConfig, RealEvolutionDaemon};
use qlang_runtime::mnist::MnistData;

const GENERATIONS: usize = 30;
const SEED_POP: usize = 20;
const EVAL_SAMPLES: usize = 300;
const MUTATIONS_PER_GEN: usize = 10;
const OUT_PATH: &str = "data/best_evolved.bin";

fn main() {
    println!("=== MNIST Self-Evolution Demo (T061) ===");

    // 1. LOAD MNIST (fallback to synthetic) -----------------------------------
    let data = match MnistData::load_from_dir("data/mnist") {
        Ok(d) => {
            println!(
                "Loaded real MNIST: {} train / {} test (image_size={})",
                d.n_train, d.n_test, d.image_size
            );
            d
        }
        Err(e) => {
            eprintln!("Real MNIST unavailable ({e}); falling back to synthetic.");
            let d = MnistData::synthetic(2000, 500);
            println!(
                "Synthetic MNIST: {} train / {} test (image_size={})",
                d.n_train, d.n_test, d.image_size
            );
            d
        }
    };

    // 2. BUILD DAEMON CONFIG --------------------------------------------------
    let config = EvolutionConfig {
        generation_interval_secs: 1,
        eval_sample_size: EVAL_SAMPLES,
        mutations_per_generation: MUTATIONS_PER_GEN,
        data_path: String::new(),
        persist_path: String::new(), // no disk persistence for the CLI demo
        mutation_config: MutationConfig {
            flip_rate: 0.04,
            zero_rate: 0.01,
            expand_rate: 0.9,
            crossover_rate: 0.5,
            mode: MutationMode::PointMutation,
            seed: 0xC0FFEE_u64,
        },
        lifecycle_policy: LifecyclePolicy {
            target_population: 50,
            retire_threshold_fitness: 0.0, // keep everyone unless pruned by cap
            kill_threshold_age_seconds: 60,
            min_generations_before_retire: 3,
            max_children_per_parent: 50,
        },
    };

    let daemon = Arc::new(RealEvolutionDaemon::new(config));

    // 3. SEED POPULATION ------------------------------------------------------
    if let Err(e) = daemon.seed_population(SEED_POP, &data) {
        eprintln!("seed_population failed: {e}");
        std::process::exit(1);
    }
    println!(
        "Seeded {} specialists. Running {} generations...\n",
        SEED_POP, GENERATIONS
    );

    // 4. RUN 30 GENERATIONS ---------------------------------------------------
    println!(
        "{:>4} | {:>5} | {:>5} | {:>5} | {:>4} | {:>6} | {:>6}",
        "Gen", "Best", "Avg", "Min", "Pop", "Births", "Deaths"
    );
    println!("-----+-------+-------+-------+------+--------+-------");

    let n_eval = data.n_test.min(EVAL_SAMPLES);
    let mut best_ever = 0.0f32;

    for g in 1..=GENERATIONS {
        let report = match daemon.step_once(&data.test_images, &data.test_labels, n_eval) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("[gen {g}] step_once failed: {e}");
                break;
            }
        };

        // Snapshot min fitness from active specialists.
        let snaps = daemon.snapshot_specialists();
        let active_fits: Vec<f32> = snaps
            .iter()
            .filter(|s| s.status == "active")
            .map(|s| s.fitness)
            .collect();
        let min_fit = active_fits
            .iter()
            .cloned()
            .fold(f32::INFINITY, f32::min);
        let min_fit_display = if min_fit.is_finite() { min_fit } else { 0.0 };

        if report.best_fitness > best_ever {
            best_ever = report.best_fitness;
        }

        println!(
            "{:>4} | {:>5.3} | {:>5.3} | {:>5.3} | {:>4} | {:>6} | {:>6}",
            g,
            report.best_fitness,
            report.avg_fitness,
            min_fit_display,
            report.population_size,
            report.mutations_spawned,
            report.killed,
        );
    }

    // 5. FINAL REPORT ---------------------------------------------------------
    println!();
    let snaps = daemon.snapshot_specialists();
    let best = snaps
        .iter()
        .filter(|s| s.status == "active")
        .max_by(|a, b| a.fitness.partial_cmp(&b.fitness).unwrap_or(std::cmp::Ordering::Equal));

    let (best_id, best_fit) = match best {
        Some(s) => (s.id.clone(), s.fitness),
        None => {
            eprintln!("No active specialists remain — aborting.");
            std::process::exit(2);
        }
    };

    let lineage_depth = daemon.lineage(&best_id).len();
    println!("=== Final ===");
    println!("Best specialist  : {best_id}");
    println!("Best fitness     : {best_fit:.4}");
    println!("Best-ever fitness: {best_ever:.4}");
    println!("Lineage size     : {lineage_depth}");
    println!(
        "Improvement over initial: {}",
        if best_ever > 0.30 { "YES" } else { "marginal" }
    );

    // 6. EXPORT BEST WEIGHTS TO QLMB-STYLE BINARY -----------------------------
    // File layout:
    //   magic       : 4 bytes  = b"QLMB"
    //   section tag : 4 bytes  = b"TBRN"   (TernaryBrain payload)
    //   version     : u32 LE   = 1
    //   fitness     : f32 LE
    //   image_dim   : u32 LE
    //   n_classes   : u32 LE
    //   n_weights   : u32 LE
    //   id_len      : u16 LE + id bytes
    //   weights     : n_weights × i8
    match export_best(&daemon, &best_id, best_fit, OUT_PATH) {
        Ok(n) => println!("Saved {n} bytes of best weights → {OUT_PATH}"),
        Err(e) => eprintln!("Failed to save best weights: {e}"),
    }
}

fn export_best(
    daemon: &RealEvolutionDaemon,
    best_id: &str,
    best_fit: f32,
    path: &str,
) -> Result<usize, String> {
    let weights = daemon
        .get_weights(best_id)
        .ok_or_else(|| format!("no weights for id {best_id}"))?;
    let image_dim = daemon.template_image_dim().unwrap_or(784) as u32;
    let n_classes = daemon.template_n_classes().unwrap_or(10) as u32;

    let snap = daemon
        .snapshot_specialists()
        .into_iter()
        .find(|s| s.id == best_id)
        .ok_or_else(|| format!("snapshot missing for {best_id}"))?;

    if let Some(parent) = std::path::Path::new(path).parent() {
        fs::create_dir_all(parent).map_err(|e| format!("mkdir: {e}"))?;
    }

    // File layout (QLMB family, TBRN section):
    //   "QLMB" | "TBRN" | version u32 | fitness f32
    //   | image_dim u32 | n_classes u32
    //   | id_len u16 | id bytes
    //   | generation_born u32 | mutations u32 | children u32
    //   | n_weights u32 | weights[i8 × n]
    let mut buf: Vec<u8> = Vec::new();
    buf.extend_from_slice(b"QLMB");
    buf.extend_from_slice(b"TBRN");
    buf.extend_from_slice(&1u32.to_le_bytes());
    buf.extend_from_slice(&best_fit.to_le_bytes());
    buf.extend_from_slice(&image_dim.to_le_bytes());
    buf.extend_from_slice(&n_classes.to_le_bytes());

    let id_bytes = snap.id.as_bytes();
    let id_len = id_bytes.len().min(u16::MAX as usize) as u16;
    buf.extend_from_slice(&id_len.to_le_bytes());
    buf.extend_from_slice(&id_bytes[..id_len as usize]);
    buf.extend_from_slice(&(snap.generation_born as u32).to_le_bytes());
    buf.extend_from_slice(&(snap.mutations as u32).to_le_bytes());
    buf.extend_from_slice(&(snap.children.len() as u32).to_le_bytes());

    buf.extend_from_slice(&(weights.len() as u32).to_le_bytes());
    // Reinterpret i8 slice as u8 bytes — ternary weights are in {-1, 0, +1},
    // stored as signed bytes, bit-identical either way.
    let weight_bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(weights.as_ptr() as *const u8, weights.len())
    };
    buf.extend_from_slice(weight_bytes);

    let mut f = fs::File::create(path).map_err(|e| format!("create: {e}"))?;
    f.write_all(&buf).map_err(|e| format!("write: {e}"))?;
    Ok(buf.len())
}
